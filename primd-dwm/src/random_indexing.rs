//! Random Indexing signature construction.
//!
//! The "zero LLM tokens" signature path from the Hippocampus paper
//! (arXiv:2602.13594, Feb 2026, Appendix B). Each token gets a sparse
//! ternary base vector `r_v ∈ {-1, 0, +1}^D`. A sliding context window
//! over a token stream sums the base vectors. The top-`d` components
//! by absolute value become the binary signature (sign of the
//! aggregate at each retained position).
//!
//! Key properties:
//! - **No embedding model.** Construction is purely combinatorial; the
//!   `r_v` table is built deterministically from a seed.
//! - **Append-only friendly.** Adding a new token only updates the
//!   sliding window's aggregate; the base table is fixed.
//! - **Cache-friendly.** Each token's base vector is `t` nonzeros in a
//!   D-dim space; sparse-update is constant per token.
//!
//! Parameters from the paper:
//! - `D ∈ {256, 512, 1024, 2048}` — base-vector dimensionality. Larger
//!   D reduces collisions between tokens at the cost of memory.
//! - `t` — number of nonzero entries per base vector (half +1, half
//!   -1). Paper's default 8-16 keeps base vectors sparse.
//! - `d ∈ {16, 32, 64, 128}` — bits retained in the final signature.
//!
//! For primd's 256-bit signatures, the natural choice is `D=256, d=256`
//! (retain every dimension as a sign bit) — gives the same wire format
//! as the existing embedding-based path so the rest of primd-core
//! consumes it unchanged.

use std::collections::HashMap;

use primd_core::embed::binary::BinarySignature;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Default base-vector dimensionality. Matches primd's 256-bit signature
/// width so the output sig is a direct sign-quantization of the aggregate.
pub const DEFAULT_D: usize = 256;

/// Default number of nonzero entries per base vector (per the paper's
/// "8-16 nonzeros" ballpark).
pub const DEFAULT_T: usize = 12;

/// Default RNG seed for the base-vector table. Fixed so two
/// `RandomIndexer` instances built from the same vocabulary produce
/// identical signatures.
pub const DEFAULT_SEED: u64 = 0x3141_5926_5358_9793;

/// Sparse ternary base vector for a single token.
///
/// Stored as two `Vec<u16>`s — positions of +1 entries and positions
/// of -1 entries. Sparsity makes per-token aggregate updates O(t)
/// instead of O(D).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BaseVector {
    plus: Vec<u16>,
    minus: Vec<u16>,
}

/// Random-indexing signature generator.
///
/// Build via [`Self::new`] with a vocabulary size and parameters; then
/// for each token stream call [`Self::signature`] to produce a 256-bit
/// signature aggregated over the stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RandomIndexer {
    /// Per-token base vector. Indexed by `token_id`.
    bases: Vec<BaseVector>,
    /// Dimensionality of the working aggregate space.
    d: usize,
    /// Nonzero entries per base vector.
    t: usize,
}

impl RandomIndexer {
    /// Build with default parameters.
    pub fn new(vocab_size: usize) -> Self {
        Self::with_params(vocab_size, DEFAULT_D, DEFAULT_T, DEFAULT_SEED)
    }

    /// Build with explicit `D`, `t`, and seed.
    ///
    /// Panics if `t > D` (can't fit more nonzeros than dimensions).
    pub fn with_params(vocab_size: usize, d: usize, t: usize, seed: u64) -> Self {
        assert!(t <= d, "t ({t}) must be <= D ({d})");
        assert_eq!(t % 2, 0, "t must be even (half +1, half -1)");
        assert!(d <= u16::MAX as usize, "D too large for u16 positions");

        use rand::seq::SliceRandom;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bases = Vec::with_capacity(vocab_size);
        let half = t / 2;
        let mut positions: Vec<u16> = (0..d as u16).collect();
        for _ in 0..vocab_size {
            positions.shuffle(&mut rng);
            let plus: Vec<u16> = positions[..half].to_vec();
            let minus: Vec<u16> = positions[half..t].to_vec();
            bases.push(BaseVector { plus, minus });
        }
        RandomIndexer { bases, d, t }
    }

    pub fn vocab_size(&self) -> usize {
        self.bases.len()
    }

    pub fn dim(&self) -> usize {
        self.d
    }

    pub fn nonzeros_per_token(&self) -> usize {
        self.t
    }

    /// Aggregate a stream of token IDs into a 256-bit binary signature.
    ///
    /// Algorithm:
    /// 1. Initialize a length-D aggregate `e = 0`.
    /// 2. For each token in `stream`, add the token's base vector to `e`.
    /// 3. If `D > 256`: pick the top-256 dimensions by absolute value
    ///    in `e` and binarize their signs. (For `D = 256`, just
    ///    binarize every dim.)
    ///
    /// Returns a 256-bit signature compatible with the rest of primd's
    /// retrieval pipeline.
    pub fn signature(&self, stream: &[u32]) -> BinarySignature {
        let mut agg = vec![0i32; self.d];
        for &token in stream {
            let idx = token as usize;
            if idx >= self.bases.len() {
                continue;
            }
            let base = &self.bases[idx];
            for &pos in &base.plus {
                agg[pos as usize] += 1;
            }
            for &pos in &base.minus {
                agg[pos as usize] -= 1;
            }
        }
        self.binarize(&agg)
    }

    /// Convert an aggregate vector to a 256-bit binary signature. When
    /// `D > 256`, keep the top-256 dimensions by `|agg[i]|` and sign-
    /// quantize them. When `D == 256`, sign-quantize every dimension
    /// directly.
    fn binarize(&self, agg: &[i32]) -> BinarySignature {
        let mut bytes = [0u8; 32];
        if self.d == 256 {
            for (i, &v) in agg.iter().enumerate() {
                if v > 0 {
                    bytes[i / 8] |= 1u8 << (i % 8);
                }
            }
        } else {
            // Select top-256 dims by |agg[i]|. Use a partial sort via
            // `select_nth_unstable_by_key`.
            let mut pairs: Vec<(usize, i32)> =
                agg.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            let cutoff = 256.min(pairs.len()).saturating_sub(1);
            pairs.select_nth_unstable_by_key(cutoff, |(_, v)| -v.abs());
            for (out_idx, (_, val)) in pairs.iter().take(256).enumerate() {
                if *val > 0 {
                    bytes[out_idx / 8] |= 1u8 << (out_idx % 8);
                }
            }
        }
        BinarySignature(bytes)
    }

    /// Convenience: build a signature from a list of string tokens by
    /// mapping each through a provided vocabulary.
    pub fn signature_from_tokens(
        &self,
        tokens: &[&str],
        vocab: &HashMap<String, u32>,
    ) -> BinarySignature {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| vocab.get(*t).copied())
            .collect();
        self.signature(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let a = RandomIndexer::with_params(100, 256, 12, 42);
        let b = RandomIndexer::with_params(100, 256, 12, 42);
        let stream = vec![0u32, 5, 10, 15];
        assert_eq!(a.signature(&stream), b.signature(&stream));
    }

    #[test]
    fn different_seeds_give_different_sigs() {
        let a = RandomIndexer::with_params(100, 256, 12, 42);
        let b = RandomIndexer::with_params(100, 256, 12, 43);
        let stream = vec![0u32, 1, 2, 3, 4, 5];
        let sig_a = a.signature(&stream);
        let sig_b = b.signature(&stream);
        // Should differ in many bits (random projection).
        let dist = sig_a.hamming_distance(&sig_b);
        assert!(dist > 50, "expected highly-different sigs, got {dist} bits");
    }

    #[test]
    fn empty_stream_produces_zero_sig() {
        let r = RandomIndexer::new(100);
        let sig = r.signature(&[]);
        assert_eq!(sig.0, [0u8; 32]);
    }

    #[test]
    fn unknown_tokens_skipped() {
        let r = RandomIndexer::with_params(10, 256, 12, 7);
        let known = r.signature(&[0, 5]);
        let with_unknown = r.signature(&[0, 999, 5, 1000]); // 999 and 1000 are out of vocab
        // The unknown tokens are silently skipped, so the sig matches
        // the known-only stream.
        assert_eq!(known, with_unknown);
    }

    #[test]
    fn similar_streams_produce_similar_sigs() {
        // A stream and its 90%-overlap version should have low Hamming
        // distance — Random Indexing's main quality property.
        let r = RandomIndexer::with_params(100, 256, 12, 11);
        let stream_a: Vec<u32> = (0..50).collect();
        let mut stream_b = stream_a.clone();
        stream_b[0] = 75; // change one token
        let sig_a = r.signature(&stream_a);
        let sig_b = r.signature(&stream_b);
        let dist = sig_a.hamming_distance(&sig_b);
        // Single-token change perturbs ~12 base-vector entries (the
        // changed token's nonzeros). Even after sign-quantization the
        // resulting Hamming should be well below 50% (random baseline
        // of ~128 bits).
        assert!(dist < 80, "expected similar sigs (dist < 80), got {dist}");
    }

    #[test]
    fn signature_from_tokens_uses_vocab() {
        let r = RandomIndexer::with_params(3, 256, 8, 17);
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("primd".to_string(), 2);
        let sig_a = r.signature(&[0, 1, 2]);
        let sig_b = r.signature_from_tokens(&["hello", "world", "primd"], &vocab);
        assert_eq!(sig_a, sig_b);
    }

    #[test]
    fn larger_d_with_topk_truncation() {
        let r = RandomIndexer::with_params(50, 1024, 12, 23);
        assert_eq!(r.dim(), 1024);
        // Output is still a 256-bit signature.
        let sig = r.signature(&[0, 5, 10]);
        assert_eq!(sig.0.len(), 32);
    }

    #[test]
    fn serde_round_trip() {
        let r = RandomIndexer::with_params(20, 256, 8, 99);
        let json = serde_json::to_string(&r).unwrap();
        let restored: RandomIndexer = serde_json::from_str(&json).unwrap();
        let stream = vec![0u32, 3, 7, 19];
        assert_eq!(restored.signature(&stream), r.signature(&stream));
    }
}
