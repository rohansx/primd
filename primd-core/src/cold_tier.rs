//! Cold-tier trait for primd's session-memory hierarchy.
//!
//! The trait surface lives in `primd-core` so [`crate::QueryContext`]
//! can hold a `Box<dyn ColdTier>` without depending on the concrete
//! `DwmColdTier` impl (which lives in `primd-dwm` and pulls in extra
//! deps for the wavelet-matrix structure). Avoids the circular crate
//! dependency that would otherwise force every primd-core caller to
//! pay the DWM cost even when not using a cold tier.
//!
//! v0.4 ships:
//! - Trait surface here (in primd-core)
//! - `DwmColdTier` impl in primd-dwm
//! - `QueryContext::with_cold_tier` builder
//! - Hot-tier results + cold-tier results merge in `QueryContext::finalize`
//!
//! v0.4.1 will add:
//! - Eviction policy (LRU on events) inside QueryContext
//! - Automatic flow from hot tier into cold tier when an event ages out

use crate::embed::binary::BinarySignature;
use crate::predict::EventId;

/// Trait surface every cold-tier impl must satisfy.
///
/// Cold tiers store signatures evicted from the hot path (HNSW shards,
/// SIMD signature index) so they remain queryable across sessions
/// without consuming the hot-path memory budget. Queries are typically
/// orders of magnitude rarer than hot-path queries, so the trait
/// trades latency for compactness.
pub trait ColdTier: Send + Sync {
    /// Number of signatures currently stored.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a signature + its source event + doc-index to the cold tier.
    /// Caller-driven eviction: the QueryContext doesn't currently
    /// auto-evict (v0.4.1 will).
    fn add_evicted(&mut self, sig: BinarySignature, event: EventId, doc_idx: usize);

    /// Search the cold tier for the top-K nearest signatures to
    /// `query`. Returns `(distance, event_id, doc_idx)` triples sorted
    /// by ascending distance.
    fn search(&self, query: &BinarySignature, top_k: usize) -> Vec<(u32, EventId, usize)>;

    /// Persist the cold tier to `path`. Format is impl-specific —
    /// `DwmColdTier` writes JSON; other impls (e.g. a remote blob
    /// store) might write a manifest pointing at object storage.
    /// Returns errors via `std::io::Error` for filesystem failures.
    fn save(&self, path: &std::path::Path) -> std::io::Result<()>;
}
