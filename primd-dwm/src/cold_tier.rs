//! Cold-tier abstraction for primd's session-memory hierarchy.
//!
//! v0.4 ships the trait surface + a DWM-backed default impl. v0.4.1
//! wires it into `primd_core::query_context::QueryContext` so events
//! evicted from the hot tier (HNSW shards) automatically flow into
//! cold storage and remain queryable across multi-day voice sessions.
//!
//! The split:
//! - **Hot tier** (already in `primd-core`): SIMD signature scan,
//!   event-scoped HNSW shards, predictive speculation, delta cache.
//!   Optimized for sub-millisecond user-visible turn latency.
//! - **Cold tier** (here): aged-out events serialized into a
//!   `SignatureDwm`. Optimized for storage compactness and cross-
//!   session persistence. Queries are slower (millisecond range) and
//!   only fired when the hot tier returns too few hits to be
//!   informative.

// Re-export the canonical trait from primd-core so callers of primd-dwm
// don't need a separate primd-core import.
pub use primd_core::cold_tier::ColdTier;
use primd_core::embed::binary::BinarySignature;
use primd_core::predict::EventId;
use serde::{Deserialize, Serialize};

use crate::signature_dwm::SignatureDwm;

/// DWM-backed cold tier. Wraps a [`SignatureDwm`] with a value-encoding
/// scheme that packs `(EventId, doc_idx)` into the DWM's `u64` value
/// slot.
///
/// Value layout: `(event_id as u64) << 32 | (doc_idx as u32 as u64)`.
/// EventId is a `u32`, doc_idx is treated as `u32` (4 B doc-indices is
/// plenty for the cold-tier-evictions use case). For corpora with
/// > 4 B distinct doc indices per session, fork the encoding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DwmColdTier {
    dwm: SignatureDwm,
    /// Pending tail size at which `add_evicted` triggers an automatic
    /// `compact()`. Default 256 evictions — balances per-push cost
    /// with the rebuild amortization.
    auto_compact_threshold: usize,
}

impl DwmColdTier {
    pub fn empty() -> Self {
        DwmColdTier {
            dwm: SignatureDwm::empty(),
            auto_compact_threshold: 256,
        }
    }

    /// Build from a complete batch of evicted entries.
    pub fn from_batch(entries: &[(BinarySignature, EventId, usize)]) -> Self {
        let sigs: Vec<BinarySignature> = entries.iter().map(|(s, _, _)| *s).collect();
        let values: Vec<u64> = entries.iter().map(|(_, e, d)| encode(*e, *d)).collect();
        DwmColdTier {
            dwm: SignatureDwm::build(&sigs, values),
            auto_compact_threshold: 256,
        }
    }

    pub fn with_auto_compact_threshold(mut self, n: usize) -> Self {
        self.auto_compact_threshold = n.max(1);
        self
    }

    pub fn auto_compact_threshold(&self) -> usize {
        self.auto_compact_threshold
    }

    pub fn pending_tail(&self) -> usize {
        self.dwm.pending_tail()
    }

    /// Force a compaction. Normally fires automatically when the tail
    /// exceeds the threshold.
    pub fn compact(&mut self) {
        self.dwm.compact();
    }

    /// Load from a file written by [`Self::save`].
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let me: DwmColdTier = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(me)
    }
}

impl Default for DwmColdTier {
    fn default() -> Self {
        Self::empty()
    }
}

impl DwmColdTier {
    /// Persist the cold tier to disk. `ColdTier` keeps its trait surface
    /// minimal; persistence is an impl-specific helper on `DwmColdTier`
    /// directly.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }
}

impl ColdTier for DwmColdTier {
    fn len(&self) -> usize {
        self.dwm.len()
    }

    fn add_evicted(&mut self, sig: BinarySignature, event: EventId, doc_idx: usize) {
        self.dwm.push(sig, encode(event, doc_idx));
        if self.dwm.pending_tail() >= self.auto_compact_threshold {
            self.dwm.compact();
        }
    }

    fn search(&self, query: &BinarySignature, top_k: usize) -> Vec<(u32, EventId, usize)> {
        self.dwm
            .nearest(query, top_k)
            .into_iter()
            .map(|(d, value)| {
                let (event, doc_idx) = decode(value);
                (d, event, doc_idx)
            })
            .collect()
    }
}

fn encode(event: EventId, doc_idx: usize) -> u64 {
    ((event.0 as u64) << 32) | (doc_idx as u32 as u64)
}

fn decode(value: u64) -> (EventId, usize) {
    let event = EventId((value >> 32) as u32);
    let doc_idx = (value & 0xFFFF_FFFF) as usize;
    (event, doc_idx)
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
    fn encode_decode_round_trip() {
        let cases = [
            (EventId(0), 0usize),
            (EventId(1), 42),
            (EventId(0xDEAD_BEEF), 12345),
            (EventId(u32::MAX), u32::MAX as usize),
        ];
        for (event, doc_idx) in cases {
            let encoded = encode(event, doc_idx);
            let (e2, d2) = decode(encoded);
            assert_eq!(event, e2);
            assert_eq!(doc_idx, d2);
        }
    }

    #[test]
    fn empty_cold_tier() {
        let ct = DwmColdTier::empty();
        assert!(ct.is_empty());
        assert_eq!(ct.len(), 0);
        assert!(ct.search(&sig(&[0]), 5).is_empty());
    }

    #[test]
    fn add_and_search() {
        let mut ct = DwmColdTier::empty();
        ct.add_evicted(sig(&[0, 1, 2]), EventId(7), 100);
        ct.add_evicted(sig(&[100, 101, 102]), EventId(9), 200);
        assert_eq!(ct.len(), 2);

        let q = sig(&[0, 1, 2]);
        let hits = ct.search(&q, 1);
        assert_eq!(hits.len(), 1);
        let (d, event, doc_idx) = hits[0];
        assert_eq!(d, 0);
        assert_eq!(event, EventId(7));
        assert_eq!(doc_idx, 100);
    }

    #[test]
    fn from_batch_constructor() {
        let entries = vec![
            (sig(&[0]), EventId(1), 10),
            (sig(&[1]), EventId(2), 20),
            (sig(&[2]), EventId(3), 30),
        ];
        let ct = DwmColdTier::from_batch(&entries);
        assert_eq!(ct.len(), 3);
        // pending_tail should be 0 — the batch went through `build`.
        assert_eq!(ct.pending_tail(), 0);
        let hits = ct.search(&sig(&[1]), 1);
        assert_eq!(hits[0].1, EventId(2));
        assert_eq!(hits[0].2, 20);
    }

    #[test]
    fn auto_compact_at_threshold() {
        let mut ct = DwmColdTier::empty().with_auto_compact_threshold(3);
        ct.add_evicted(sig(&[0]), EventId(0), 0);
        ct.add_evicted(sig(&[1]), EventId(0), 1);
        // Below threshold — pending.
        assert_eq!(ct.pending_tail(), 2);
        ct.add_evicted(sig(&[2]), EventId(0), 2);
        // Hit threshold — auto-compacted.
        assert_eq!(ct.pending_tail(), 0);
        assert_eq!(ct.len(), 3);
    }

    #[test]
    fn save_and_load_round_trip() {
        let mut ct = DwmColdTier::empty();
        ct.add_evicted(sig(&[0, 10, 20]), EventId(11), 111);
        ct.add_evicted(sig(&[5, 15, 25]), EventId(22), 222);
        ct.add_evicted(sig(&[7, 17, 27]), EventId(33), 333);

        let path = std::env::temp_dir().join("primd-cold-tier-roundtrip.json");
        ct.save(&path).unwrap();
        let loaded = DwmColdTier::load(&path).unwrap();

        assert_eq!(loaded.len(), 3);
        let hits = loaded.search(&sig(&[5, 15, 25]), 1);
        assert_eq!(hits[0].1, EventId(22));
        assert_eq!(hits[0].2, 222);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn trait_object_safe() {
        let mut ct: Box<dyn ColdTier> = Box::new(DwmColdTier::empty());
        ct.add_evicted(sig(&[0]), EventId(1), 10);
        ct.add_evicted(sig(&[1]), EventId(2), 20);
        let hits = ct.search(&sig(&[0]), 2);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].1, EventId(1));
    }
}
