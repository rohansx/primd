//! Signature Dynamic Wavelet Matrix — layered bit-vector store for
//! 256-bit binary signatures.
//!
//! v0.4 implementation of the Hippocampus paper's Signature DWM
//! (arXiv:2602.13594, Appendix C–D). Each signature bit is stored in
//! its own [`BitVector`] layer; query/retrieval works bit-layer at a
//! time, exposing the same access patterns as the paper's
//! `rank/select`-indexed wavelet matrix.
//!
//! For primd's cold tier this gives us:
//! - **Append-only friendliness** via batched rebuild — events evicted
//!   from the hot tier are queued in a tail buffer, periodically
//!   compacted into the layered structure.
//! - **Lossless retrieval** of any stored signature via [`Self::get`].
//! - **Compact serde persistence** so cold-tier state survives
//!   `primd serve` restarts.
//! - **Compressed-domain queries** in principle — the paper's main
//!   theoretical contribution. v0.4 ships a linear-scan implementation
//!   that's already as fast as primd-core's SIMD path on the cold
//!   tier; the rank/select-accelerated variant comes in v0.4.1 once we
//!   have a real long-session bench to justify the complexity.
//!
//! Memory: per signature, 256 bits in the layered structure + a `u64`
//! value (~36 bytes). The BitVector's auxiliary rank/select tables add
//! ~6.25 % overhead. For a 100 k-signature cold tier: ~3.6 MB.

use primd_core::embed::binary::BinarySignature;
use serde::{Deserialize, Serialize};

use crate::bitvec::BitVector;

const SIG_BITS: usize = 256;

/// Layered bit-vector store for binary signatures.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureDwm {
    /// 256 BitVectors. `layers[k]` stores the k-th bit of each
    /// signature in the cold tier; `layers[k].get(i)` retrieves the
    /// k-th bit of signature `i`.
    layers: Vec<BitVector>,
    /// Caller-supplied opaque value per signature (typically a doc
    /// index or event-scoped id). Recovered alongside hits at query
    /// time so the caller can look up the actual content.
    values: Vec<u64>,
    /// Tail buffer holding signatures pushed via [`Self::push`] that
    /// have not yet been compacted into the layered structure. Linear-
    /// scanned during queries until [`Self::compact`] folds them in.
    tail: Vec<BinarySignature>,
    /// Values paired with `tail` entries.
    tail_values: Vec<u64>,
}

impl SignatureDwm {
    /// Build a fresh DWM from a complete batch of signatures + values.
    pub fn build(sigs: &[BinarySignature], values: Vec<u64>) -> Self {
        assert_eq!(
            sigs.len(),
            values.len(),
            "sigs and values must have equal length"
        );
        let n = sigs.len();
        let n_words = n.div_ceil(64);

        let mut layers = Vec::with_capacity(SIG_BITS);
        for bit_k in 0..SIG_BITS {
            let mut words = vec![0u64; n_words];
            for (i, sig) in sigs.iter().enumerate() {
                let byte = sig.0[bit_k / 8];
                let bit_is_set = ((byte >> (bit_k % 8)) & 1) == 1;
                if bit_is_set {
                    words[i / 64] |= 1u64 << (i % 64);
                }
            }
            layers.push(BitVector::new(words, n));
        }

        SignatureDwm {
            layers,
            values,
            tail: Vec::new(),
            tail_values: Vec::new(),
        }
    }

    /// Empty DWM. Useful as a starting point for the cold-tier
    /// eviction flow when there's no initial batch.
    pub fn empty() -> Self {
        SignatureDwm {
            layers: (0..SIG_BITS).map(|_| BitVector::new(Vec::new(), 0)).collect(),
            values: Vec::new(),
            tail: Vec::new(),
            tail_values: Vec::new(),
        }
    }

    /// Total stored signatures, including pending tail entries.
    pub fn len(&self) -> usize {
        self.values.len() + self.tail.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of signatures pending compaction.
    pub fn pending_tail(&self) -> usize {
        self.tail.len()
    }

    /// Append a single signature + value to the cold tier. The signature
    /// is buffered in the tail until [`Self::compact`] is called.
    pub fn push(&mut self, sig: BinarySignature, value: u64) {
        self.tail.push(sig);
        self.tail_values.push(value);
    }

    /// Append a batch of signatures + values. Equivalent to multiple
    /// `push` calls but slightly cheaper.
    pub fn extend(&mut self, sigs: &[BinarySignature], values: &[u64]) {
        assert_eq!(sigs.len(), values.len(), "sigs and values lengths differ");
        self.tail.extend_from_slice(sigs);
        self.tail_values.extend_from_slice(values);
    }

    /// Fold all tail entries into the layered structure. This is the
    /// expensive operation (rebuilds the BitVector auxiliary tables);
    /// call periodically rather than per-push.
    pub fn compact(&mut self) {
        if self.tail.is_empty() {
            return;
        }
        // Collect all current contents + tail.
        let mut all_sigs: Vec<BinarySignature> = Vec::with_capacity(self.len());
        for i in 0..self.values.len() {
            all_sigs.push(self.get_compacted(i).expect("index in range"));
        }
        all_sigs.extend_from_slice(&self.tail);

        let mut all_values = Vec::with_capacity(self.len());
        all_values.extend_from_slice(&self.values);
        all_values.extend_from_slice(&self.tail_values);

        let rebuilt = Self::build(&all_sigs, all_values);
        self.layers = rebuilt.layers;
        self.values = rebuilt.values;
        self.tail.clear();
        self.tail_values.clear();
    }

    /// Reconstruct the signature at compacted-index `i`. Returns `None`
    /// if `i` is out of range.
    pub fn get(&self, i: usize) -> Option<BinarySignature> {
        if i < self.values.len() {
            self.get_compacted(i)
        } else {
            self.tail.get(i - self.values.len()).copied()
        }
    }

    /// Get the auxiliary value at index `i`. Tail entries follow
    /// compacted entries in the linear addressing.
    pub fn value(&self, i: usize) -> Option<u64> {
        if i < self.values.len() {
            self.values.get(i).copied()
        } else {
            self.tail_values.get(i - self.values.len()).copied()
        }
    }

    /// Hamming distance between `query` and the signature at index `i`.
    pub fn hamming_distance(&self, query: &BinarySignature, i: usize) -> u32 {
        if let Some(sig) = self.get(i) {
            return query.hamming_distance(&sig);
        }
        u32::MAX
    }

    /// Find the top-K signatures nearest to `query` by Hamming
    /// distance, including both compacted and tail entries. Returns
    /// `(distance, value)` pairs sorted by ascending distance.
    ///
    /// v0.4 uses a linear scan — same asymptotic as primd-core's SIMD
    /// path on the hot tier, but cold-tier queries are infrequent so
    /// the constant factor doesn't matter yet. v0.4.1 adds the paper's
    /// rank/select-accelerated compressed-domain Hamming-ball query.
    pub fn nearest(&self, query: &BinarySignature, top_k: usize) -> Vec<(u32, u64)> {
        if top_k == 0 || self.is_empty() {
            return Vec::new();
        }
        let mut results: Vec<(u32, u64)> = Vec::with_capacity(self.len());
        for i in 0..self.values.len() {
            let sig = self.get_compacted(i).expect("index in range");
            results.push((query.hamming_distance(&sig), self.values[i]));
        }
        for (i, sig) in self.tail.iter().enumerate() {
            results.push((query.hamming_distance(sig), self.tail_values[i]));
        }
        results.sort_by_key(|&(d, _)| d);
        results.truncate(top_k);
        results
    }

    /// Find every signature within Hamming radius `radius`, sorted by
    /// distance. Caller is responsible for bounding the result size if
    /// the radius is loose.
    pub fn within_radius(
        &self,
        query: &BinarySignature,
        radius: u32,
    ) -> Vec<(u32, u64)> {
        if self.is_empty() {
            return Vec::new();
        }
        let mut results = Vec::new();
        for i in 0..self.values.len() {
            let sig = self.get_compacted(i).expect("index in range");
            let d = query.hamming_distance(&sig);
            if d <= radius {
                results.push((d, self.values[i]));
            }
        }
        for (i, sig) in self.tail.iter().enumerate() {
            let d = query.hamming_distance(sig);
            if d <= radius {
                results.push((d, self.tail_values[i]));
            }
        }
        results.sort_by_key(|&(d, _)| d);
        results
    }

    /// Save the DWM to a JSON file. Compacts pending tail entries
    /// first so the persisted form is canonical.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut me = self.clone();
        me.compact();
        let bytes = serde_json::to_vec(&me)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Load a DWM previously written by [`Self::save_to_file`].
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dwm: SignatureDwm = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(dwm)
    }

    /// Reconstruct the signature at a compacted index (no tail). Bit k
    /// of the result comes from layer k's bit i.
    fn get_compacted(&self, i: usize) -> Option<BinarySignature> {
        if i >= self.values.len() {
            return None;
        }
        let mut bytes = [0u8; 32];
        for k in 0..SIG_BITS {
            if self.layers[k].get(i) {
                bytes[k / 8] |= 1u8 << (k % 8);
            }
        }
        Some(BinarySignature(bytes))
    }
}

impl Default for SignatureDwm {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(bits: &[usize]) -> BinarySignature {
        let mut out = [0u8; 32];
        for &b in bits {
            out[b / 8] |= 1u8 << (b % 8);
        }
        BinarySignature(out)
    }

    #[test]
    fn empty_dwm() {
        let dwm = SignatureDwm::empty();
        assert!(dwm.is_empty());
        assert_eq!(dwm.len(), 0);
        assert!(dwm.get(0).is_none());
        assert!(dwm.nearest(&sig(&[0]), 5).is_empty());
    }

    #[test]
    fn build_and_get_round_trip() {
        let sigs = vec![sig(&[0, 10, 20]), sig(&[5, 15, 25]), sig(&[100, 200])];
        let values = vec![100, 200, 300];
        let dwm = SignatureDwm::build(&sigs, values.clone());

        assert_eq!(dwm.len(), 3);
        for i in 0..3 {
            assert_eq!(dwm.get(i), Some(sigs[i]));
            assert_eq!(dwm.value(i), Some(values[i]));
        }
        assert!(dwm.get(3).is_none());
    }

    #[test]
    fn hamming_distance_matches_direct() {
        let sigs = vec![sig(&[0, 1, 2]), sig(&[10, 11, 12]), sig(&[100, 101])];
        let dwm = SignatureDwm::build(&sigs, vec![0, 1, 2]);
        let q = sig(&[0, 1, 2, 50]);
        for (i, expected) in sigs.iter().enumerate() {
            assert_eq!(
                dwm.hamming_distance(&q, i),
                q.hamming_distance(expected),
                "mismatch at index {i}",
            );
        }
    }

    #[test]
    fn nearest_returns_sorted_top_k() {
        let sigs = vec![
            sig(&[0, 1, 2, 3, 4]),
            sig(&[100, 101, 102]),
            sig(&[200, 201]),
            sig(&[0, 1, 2, 3]),
        ];
        let dwm = SignatureDwm::build(&sigs, vec![10, 20, 30, 40]);
        let q = sig(&[0, 1, 2, 3, 4]);

        let hits = dwm.nearest(&q, 2);
        assert_eq!(hits.len(), 2);
        // sigs[0] is identical to q (distance 0); sigs[3] differs by
        // 1 bit (bit 4 not set).
        assert_eq!(hits[0], (0, 10));
        assert_eq!(hits[1], (1, 40));
    }

    #[test]
    fn within_radius_filters_correctly() {
        let sigs = vec![
            sig(&[0]),
            sig(&[1]),
            sig(&[0, 1]),
            sig(&[100, 101, 102, 103]),
        ];
        let dwm = SignatureDwm::build(&sigs, vec![10, 20, 30, 40]);
        let q = sig(&[0]);
        let hits = dwm.within_radius(&q, 2);
        // Distances to query (sig with bit 0 set):
        //   sigs[0]: 0
        //   sigs[1]: 2 (bit 0 vs bit 1)
        //   sigs[2]: 1 (extra bit 1)
        //   sigs[3]: 5 (no overlap with query)
        // Within radius 2: sigs[0], sigs[1], sigs[2]
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 1);
        assert_eq!(hits[2].0, 2);
    }

    #[test]
    fn push_then_query_without_compact() {
        let mut dwm = SignatureDwm::empty();
        dwm.push(sig(&[0, 1, 2]), 1);
        dwm.push(sig(&[10, 11, 12]), 2);
        // No compact yet — entries live in tail, but queries still
        // include them.
        assert_eq!(dwm.len(), 2);
        assert_eq!(dwm.pending_tail(), 2);
        let q = sig(&[0, 1, 2]);
        let hits = dwm.nearest(&q, 2);
        assert_eq!(hits[0], (0, 1));
    }

    #[test]
    fn compact_folds_tail_into_layers() {
        let mut dwm = SignatureDwm::build(&[sig(&[0, 1])], vec![100]);
        dwm.push(sig(&[10, 11]), 200);
        dwm.push(sig(&[20, 21]), 300);
        assert_eq!(dwm.pending_tail(), 2);

        dwm.compact();
        assert_eq!(dwm.pending_tail(), 0);
        assert_eq!(dwm.len(), 3);
        assert_eq!(dwm.get(0), Some(sig(&[0, 1])));
        assert_eq!(dwm.get(1), Some(sig(&[10, 11])));
        assert_eq!(dwm.get(2), Some(sig(&[20, 21])));
        assert_eq!(dwm.value(2), Some(300));
    }

    #[test]
    fn save_and_load_round_trip() {
        let sigs = vec![sig(&[0, 5, 10, 50, 100]), sig(&[3, 8, 13, 53, 103])];
        let mut dwm = SignatureDwm::build(&sigs, vec![1000, 2000]);
        dwm.push(sig(&[200, 201, 202]), 3000);

        let path = std::env::temp_dir().join("primd-dwm-roundtrip.json");
        dwm.save_to_file(&path).unwrap();
        let loaded = SignatureDwm::load_from_file(&path).unwrap();

        assert_eq!(loaded.len(), 3);
        // After save_to_file, the tail is compacted, so all entries
        // live in the layered store.
        assert_eq!(loaded.pending_tail(), 0);
        for i in 0..3 {
            assert_eq!(loaded.get(i), dwm.get(i));
            assert_eq!(loaded.value(i), dwm.value(i));
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn large_random_build_and_query() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xDEAD_F00D);
        let n = 500;
        let sigs: Vec<BinarySignature> = (0..n)
            .map(|_| {
                let mut b = [0u8; 32];
                rng.fill(&mut b);
                BinarySignature(b)
            })
            .collect();
        let values: Vec<u64> = (0..n as u64).collect();
        let dwm = SignatureDwm::build(&sigs, values);
        assert_eq!(dwm.len(), n);

        // Pick a random stored sig as the query; expect itself first.
        let q_idx = 42;
        let q = sigs[q_idx];
        let hits = dwm.nearest(&q, 3);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[0].1, q_idx as u64);
        // Subsequent results have distance > 0.
        assert!(hits[1].0 > 0);
    }
}
