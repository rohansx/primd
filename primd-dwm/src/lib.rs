//! primd-dwm — Dynamic Wavelet Matrix primitives for primd's
//! Hippocampus-style cold tier.
//!
//! Foundation for porting arXiv:2602.13594 (Li, Cao, Ahmed, Sharma & Li,
//! Feb 2026) into primd as the cold-storage layer for long-session
//! agent memory. The paper has no public reference implementation as of
//! May 2026 — see [the strategy memo](../../docs/business/strategy-2026-05.md)
//! for the first-mover rationale.
//!
//! What v0.3 ships:
//! - [`bitvec::BitVector`] with O(1) `rank_1` / `select_1` primitives.
//! - [`random_indexing::RandomIndexer`] for zero-LLM-token signature
//!   construction (the paper's Random Indexing scheme).
//!
//! What v0.4 plans:
//! - `Signature DWM` — the wavelet-matrix-backed cold-tier signature
//!   store. Append-only, supports compressed-domain Hamming-ball
//!   queries via XOR + popcount over the rank/select-indexed bit
//!   layers.
//! - Cold-tier integration in [`primd_core::query_context::QueryContext`]
//!   so events evicted from the hot tier (HNSW shards) move into the
//!   DWM-backed store and remain queryable across multi-day sessions.

pub mod bitvec;
pub mod cold_tier;
pub mod random_indexing;
pub mod signature_dwm;

pub use bitvec::BitVector;
pub use cold_tier::{ColdTier, DwmColdTier};
pub use random_indexing::{
    DEFAULT_D, DEFAULT_SEED as DEFAULT_RI_SEED, DEFAULT_T, RandomIndexer,
};
pub use signature_dwm::SignatureDwm;
