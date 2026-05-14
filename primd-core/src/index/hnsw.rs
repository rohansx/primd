//! Per-event HNSW shards.
//!
//! v0.3 deliverable promised in the original roadmap. The v0.1
//! event-scoped path was a SIMD gather + Hamming rescan over the union
//! of candidate-event scopes — fast at 100k corpus size (the bench
//! shows finalize at ~1.5 µs cache hit / ~130 µs naive scan) but
//! scales linearly with the union-scope size. At 1M+ docs with
//! correspondingly large event scopes, the rescan starts to dominate.
//!
//! This module adds per-event HNSW indexes (via `instant-distance`)
//! that get built lazily on first query touching an event. The cache
//! stays in memory after build; persistence to disk is v0.3.1 work.
//!
//! Hamming distance is exposed to `instant-distance` via the `Point`
//! trait on a thin `SignaturePoint` wrapper.
//!
//! Gated behind the `hnsw` feature flag (on by default). When disabled,
//! [`HierarchicalIndex`] falls through to the v0.2 subset-rescan path
//! unconditionally.

use std::collections::HashMap;
use std::sync::RwLock;

use instant_distance::{Builder, HnswMap, Point, Search};

use crate::embed::binary::BinarySignature;
use crate::predict::EventId;

/// Wrapper around `BinarySignature` exposing Hamming distance to
/// `instant-distance`. The `f32` distance contract is fine: 256-bit
/// Hamming distance fits in [0, 256] which is well-representable.
#[derive(Clone, Debug)]
struct SignaturePoint(BinarySignature);

impl Point for SignaturePoint {
    fn distance(&self, other: &Self) -> f32 {
        self.0.hamming_distance(&other.0) as f32
    }
}

/// Build parameters for the per-event HNSW shards.
///
/// Defaults chosen for the 1k–10k docs-per-event regime that primd
/// typically sees. Larger shards may want higher `ef_construction` for
/// recall; smaller shards may want lower for build speed.
#[derive(Clone, Copy, Debug)]
pub struct HnswBuildOptions {
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswBuildOptions {
    fn default() -> Self {
        Self {
            ef_construction: 100,
            ef_search: 64,
        }
    }
}

/// Lazy per-event HNSW index cache.
///
/// One `HnswMap` per `EventId`, built on first access. The `RwLock`
/// allows multiple concurrent queries to hit pre-built shards without
/// contending; only the first build serializes through the write lock.
pub struct EventHnswCache {
    options: HnswBuildOptions,
    /// Source signatures keyed by global doc index. Owned by the cache
    /// so build() can re-derive shard contents without going back to
    /// the underlying `SignatureIndex`.
    docs: Box<[BinarySignature]>,
    /// Per-event doc index lists.
    scopes: HashMap<EventId, Vec<usize>>,
    /// Built shards. Each shard maps from `instant-distance`'s internal
    /// node ids back to the global doc index.
    built: RwLock<HashMap<EventId, HnswMap<SignaturePoint, usize>>>,
    /// Minimum scope size before HNSW build is worth it. Below this
    /// threshold the existing subset-rescan path is cheaper.
    min_shard_size: usize,
}

impl EventHnswCache {
    /// Threshold below which we skip HNSW and let the caller fall
    /// through to the v0.2 subset-rescan path. 1024 is a sweet spot:
    /// shard build at 1k is ~10 ms one-shot; subset rescan at 1k is
    /// ~5 µs per query — break-even at ~2000 queries per session.
    pub const DEFAULT_MIN_SHARD_SIZE: usize = 1024;

    /// Build a new cache wrapping the corpus signatures and the per-
    /// event doc-index lists. Shards are NOT built here — that happens
    /// lazily on first [`Self::search`] touching the event.
    pub fn new(
        docs: Vec<BinarySignature>,
        scopes: HashMap<EventId, Vec<usize>>,
        options: HnswBuildOptions,
    ) -> Self {
        Self {
            options,
            docs: docs.into_boxed_slice(),
            scopes,
            built: RwLock::new(HashMap::new()),
            min_shard_size: Self::DEFAULT_MIN_SHARD_SIZE,
        }
    }

    pub fn with_min_shard_size(mut self, n: usize) -> Self {
        self.min_shard_size = n;
        self
    }

    pub fn min_shard_size(&self) -> usize {
        self.min_shard_size
    }

    /// Should we even try to use HNSW for an event of this size?
    pub fn is_worth_hnsw(&self, scope_size: usize) -> bool {
        scope_size >= self.min_shard_size
    }

    /// Search a single event's HNSW shard for the top-K nearest
    /// signatures to `query`. Builds the shard if it doesn't exist
    /// yet.
    ///
    /// Returns `(distance, global_doc_idx)` pairs sorted ascending by
    /// distance. Returns empty if the event isn't in the cache or its
    /// scope is empty.
    pub fn search(
        &self,
        event: EventId,
        query: &BinarySignature,
        top_k: usize,
    ) -> Vec<(u32, usize)> {
        if top_k == 0 {
            return Vec::new();
        }

        // Fast path: shard already built. Read-lock the cache and
        // query directly.
        {
            let built = self.built.read().expect("hnsw cache poisoned");
            if let Some(shard) = built.get(&event) {
                return query_shard::<256>(shard, query, top_k);
            }
        }

        // Slow path: build the shard. Take the write lock for the
        // build; use the Entry API to handle the racing-thread case
        // (the other thread may have inserted between our read-unlock
        // and write-lock).
        let mut built = self.built.write().expect("hnsw cache poisoned");
        if let std::collections::hash_map::Entry::Vacant(e) = built.entry(event) {
            let Some(scope) = self.scopes.get(&event) else {
                return Vec::new();
            };
            if scope.len() < self.min_shard_size {
                // Caller should have checked `is_worth_hnsw` first;
                // bail without polluting the cache.
                return Vec::new();
            }
            let points: Vec<SignaturePoint> = scope
                .iter()
                .filter_map(|&idx| self.docs.get(idx).map(|s| SignaturePoint(*s)))
                .collect();
            let values: Vec<usize> = scope.clone();
            let shard = Builder::default()
                .ef_construction(self.options.ef_construction)
                .ef_search(self.options.ef_search)
                .build(points, values);
            e.insert(shard);
        }
        let shard = built.get(&event).expect("just inserted");
        query_shard::<256>(shard, query, top_k)
    }
}

/// Query a single built shard. Const-generic K lets us pre-allocate
/// the candidate buffer on the stack.
fn query_shard<const _UNUSED: usize>(
    shard: &HnswMap<SignaturePoint, usize>,
    query: &BinarySignature,
    top_k: usize,
) -> Vec<(u32, usize)> {
    let mut search = Search::default();
    let qp = SignaturePoint(*query);
    shard
        .search(&qp, &mut search)
        .take(top_k)
        .map(|item| (item.distance as u32, *item.value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_sig(rng: &mut StdRng) -> BinarySignature {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        BinarySignature(b)
    }

    fn perturb(s: &BinarySignature, rng: &mut StdRng, n_bits: u32) -> BinarySignature {
        let mut out = *s;
        for _ in 0..n_bits {
            let bit = rng.random_range(0..256);
            out.0[bit / 8] ^= 1 << (bit % 8);
        }
        out
    }

    fn build_cache(rng: &mut StdRng, n_events: u32, docs_per_event: usize) -> EventHnswCache {
        let mut docs = Vec::new();
        let mut scopes: HashMap<EventId, Vec<usize>> = HashMap::new();
        for e in 0..n_events {
            let centroid = random_sig(rng);
            let mut scope = Vec::with_capacity(docs_per_event);
            for _ in 0..docs_per_event {
                let idx = docs.len();
                docs.push(perturb(&centroid, rng, 10));
                scope.push(idx);
            }
            scopes.insert(EventId(e), scope);
        }
        EventHnswCache::new(docs, scopes, HnswBuildOptions::default())
            .with_min_shard_size(50)
    }

    #[test]
    fn search_returns_nearest_for_centroid_query() {
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        let cache = build_cache(&mut rng, 3, 200);

        // Pick a doc from event 1; query with it; expect to find it
        // (distance 0) in the top-K.
        let event = EventId(1);
        let scope = cache.scopes.get(&event).unwrap();
        let query_doc_idx = scope[42];
        let query = cache.docs[query_doc_idx];

        let hits = cache.search(event, &query, 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].1, query_doc_idx);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn search_below_threshold_returns_empty() {
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let cache = build_cache(&mut rng, 1, 200).with_min_shard_size(10_000);
        assert!(!cache.is_worth_hnsw(200));
        let q = random_sig(&mut rng);
        // The implementation refuses to build a tiny shard, returning empty.
        assert!(cache.search(EventId(0), &q, 5).is_empty());
    }

    #[test]
    fn search_unknown_event_returns_empty() {
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let cache = build_cache(&mut rng, 2, 200);
        let q = random_sig(&mut rng);
        assert!(cache.search(EventId(99), &q, 5).is_empty());
    }

    #[test]
    fn shard_caches_after_first_build() {
        let mut rng = StdRng::seed_from_u64(0xFACE);
        let cache = build_cache(&mut rng, 1, 200);
        let q = random_sig(&mut rng);
        // First call triggers build.
        let h1 = cache.search(EventId(0), &q, 3);
        // Second call hits the read-lock fast path.
        let h2 = cache.search(EventId(0), &q, 3);
        assert_eq!(h1, h2);
        assert_eq!(
            cache.built.read().unwrap().len(),
            1,
            "exactly one shard cached after two queries to the same event"
        );
    }

    #[test]
    fn is_worth_hnsw_threshold() {
        let cache = EventHnswCache::new(Vec::new(), HashMap::new(), HnswBuildOptions::default());
        assert!(!cache.is_worth_hnsw(100));
        assert!(!cache.is_worth_hnsw(EventHnswCache::DEFAULT_MIN_SHARD_SIZE - 1));
        assert!(cache.is_worth_hnsw(EventHnswCache::DEFAULT_MIN_SHARD_SIZE));
        assert!(cache.is_worth_hnsw(50_000));
    }
}
