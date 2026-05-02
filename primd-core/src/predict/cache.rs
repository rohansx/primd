//! Predictive coding delta cache.
//!
//! The brain's cortex doesn't reprocess every input from scratch; it caches its
//! predictions and only burns compute when the perceptual signal diverges from
//! prediction (the *prediction error*, in predictive-coding theory).
//!
//! `DeltaCache` plays that role for retrieval. Cached entries are keyed by:
//!   * `scope_hash` — a hash of the predicted event set (different predictions
//!     yield different cached results).
//!   * Hamming-distance proximity to a previously-seen query signature within
//!     `delta_tolerance` bits (similar queries reuse cached results).
//!
//! On a hit, the cached top-K is returned with zero scan work. On a miss, the
//! caller scans, then inserts the (query, results) pair into the cache.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::embed::binary::BinarySignature;

use super::state::EventId;

#[derive(Default, Clone, Copy, Debug)]
pub struct DeltaCacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
}

impl DeltaCacheStats {
    pub fn hit_rate(&self) -> f32 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f32 / self.lookups as f32
    }
}

#[derive(Clone)]
struct Entry {
    query: BinarySignature,
    results: Vec<(u32, usize)>,
    k: usize,
    last_used: u64,
}

pub struct DeltaCache {
    by_scope: HashMap<u64, Vec<Entry>>,
    delta_tolerance: u32,
    max_entries_per_scope: usize,
    use_counter: u64,
    stats: DeltaCacheStats,
}

impl DeltaCache {
    /// `delta_tolerance` is the max Hamming distance between a stored query
    /// and an incoming query for them to be considered the same. For 256-bit
    /// signatures, a value around 12-24 bits (5-10%) is reasonable.
    ///
    /// `max_entries_per_scope` caps memory growth per predicted-scope; the
    /// least-recently-used entry is evicted on insert when full.
    pub fn new(delta_tolerance: u32, max_entries_per_scope: usize) -> Self {
        Self {
            by_scope: HashMap::new(),
            delta_tolerance,
            max_entries_per_scope: max_entries_per_scope.max(1),
            use_counter: 0,
            stats: DeltaCacheStats::default(),
        }
    }

    /// Compute a stable scope hash from a predicted event set. Order-independent:
    /// the same set produces the same hash regardless of insertion order.
    pub fn scope_hash(events: &[EventId]) -> u64 {
        let mut sorted: Vec<u32> = events.iter().map(|e| e.0).collect();
        sorted.sort_unstable();
        sorted.dedup();
        let mut hasher = DefaultHasher::new();
        sorted.hash(&mut hasher);
        hasher.finish()
    }

    /// Look up an entry. Returns the cached top-K if a match exists; otherwise
    /// `None`. A match requires the same scope hash, the same `k`, and a
    /// query within `delta_tolerance` Hamming distance.
    pub fn lookup(
        &mut self,
        scope_hash: u64,
        query: &BinarySignature,
        k: usize,
    ) -> Option<Vec<(u32, usize)>> {
        self.stats.lookups += 1;
        self.use_counter += 1;

        let bucket = match self.by_scope.get_mut(&scope_hash) {
            Some(b) => b,
            None => {
                self.stats.misses += 1;
                return None;
            }
        };

        let mut best: Option<(usize, u32)> = None;
        for (i, entry) in bucket.iter().enumerate() {
            if entry.k != k {
                continue;
            }
            let dist = query.hamming_distance(&entry.query);
            if dist <= self.delta_tolerance {
                match best {
                    Some((_, prev_dist)) if prev_dist <= dist => {}
                    _ => best = Some((i, dist)),
                }
            }
        }

        if let Some((idx, _)) = best {
            bucket[idx].last_used = self.use_counter;
            self.stats.hits += 1;
            Some(bucket[idx].results.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a (query, results) pair. Evicts the LRU entry in the same scope
    /// bucket if the bucket is at capacity.
    pub fn insert(
        &mut self,
        scope_hash: u64,
        query: BinarySignature,
        k: usize,
        results: Vec<(u32, usize)>,
    ) {
        self.stats.inserts += 1;
        self.use_counter += 1;

        let bucket = self.by_scope.entry(scope_hash).or_default();
        if bucket.len() >= self.max_entries_per_scope {
            // Evict LRU
            if let Some(victim_idx) = bucket
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
            {
                bucket.swap_remove(victim_idx);
                self.stats.evictions += 1;
            }
        }
        bucket.push(Entry {
            query,
            results,
            k,
            last_used: self.use_counter,
        });
    }

    pub fn clear(&mut self) {
        self.by_scope.clear();
    }

    pub fn stats(&self) -> DeltaCacheStats {
        self.stats
    }

    /// Total number of cached entries across all scopes.
    pub fn len(&self) -> usize {
        self.by_scope.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_scope.is_empty() || self.by_scope.values().all(|v| v.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: u32) -> EventId {
        EventId(id)
    }

    fn make_sig(seed: u8) -> BinarySignature {
        BinarySignature([seed; 32])
    }

    fn flip_n_bits(sig: &BinarySignature, n: u32) -> BinarySignature {
        let mut out = *sig;
        for i in 0..n {
            let bit = (i * 7) as usize % 256;
            out.0[bit / 8] ^= 1 << (bit % 8);
        }
        out
    }

    #[test]
    fn miss_when_empty() {
        let mut cache = DeltaCache::new(16, 64);
        let sh = DeltaCache::scope_hash(&[ev(1), ev(2)]);
        assert!(cache.lookup(sh, &make_sig(0xAA), 10).is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn exact_query_hits() {
        let mut cache = DeltaCache::new(16, 64);
        let sh = DeltaCache::scope_hash(&[ev(1), ev(2)]);
        let q = make_sig(0xAA);
        let results = vec![(0, 42), (3, 17)];
        cache.insert(sh, q, 2, results.clone());

        let hit = cache.lookup(sh, &q, 2).expect("should hit");
        assert_eq!(hit, results);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn nearby_query_hits_within_tolerance() {
        let mut cache = DeltaCache::new(16, 64);
        let sh = DeltaCache::scope_hash(&[ev(1), ev(2)]);
        let q = make_sig(0xAA);
        let results = vec![(0, 42)];
        cache.insert(sh, q, 1, results.clone());

        let nearby = flip_n_bits(&q, 8);
        let hit = cache.lookup(sh, &nearby, 1).expect("should hit");
        assert_eq!(hit, results);
    }

    #[test]
    fn far_query_misses() {
        let mut cache = DeltaCache::new(16, 64);
        let sh = DeltaCache::scope_hash(&[ev(1), ev(2)]);
        let q = make_sig(0xAA);
        cache.insert(sh, q, 1, vec![(0, 42)]);

        let far = flip_n_bits(&q, 32);
        assert!(cache.lookup(sh, &far, 1).is_none());
    }

    #[test]
    fn different_scope_misses() {
        let mut cache = DeltaCache::new(16, 64);
        let sh1 = DeltaCache::scope_hash(&[ev(1), ev(2)]);
        let sh2 = DeltaCache::scope_hash(&[ev(3), ev(4)]);
        let q = make_sig(0xAA);
        cache.insert(sh1, q, 1, vec![(0, 42)]);

        assert!(cache.lookup(sh2, &q, 1).is_none());
    }

    #[test]
    fn scope_hash_is_order_independent() {
        let h1 = DeltaCache::scope_hash(&[ev(1), ev(2), ev(3)]);
        let h2 = DeltaCache::scope_hash(&[ev(3), ev(1), ev(2)]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_k_misses() {
        let mut cache = DeltaCache::new(16, 64);
        let sh = DeltaCache::scope_hash(&[ev(1)]);
        let q = make_sig(0xAA);
        cache.insert(sh, q, 5, vec![(0, 42)]);
        assert!(cache.lookup(sh, &q, 10).is_none());
    }

    #[test]
    fn lru_evicts_oldest() {
        let mut cache = DeltaCache::new(16, 2); // capacity 2 per scope
        let sh = DeltaCache::scope_hash(&[ev(1)]);

        let a = make_sig(0x11);
        let b = make_sig(0x22);
        let c = make_sig(0x33);

        cache.insert(sh, a, 1, vec![(0, 1)]);
        cache.insert(sh, b, 1, vec![(0, 2)]);
        // Touch a so b becomes LRU
        let _ = cache.lookup(sh, &a, 1);
        cache.insert(sh, c, 1, vec![(0, 3)]); // should evict b

        assert!(cache.lookup(sh, &a, 1).is_some());
        assert!(cache.lookup(sh, &c, 1).is_some());
        // b should be gone (and far from a/c so won't fuzzy-match)
        let q_b_far = flip_n_bits(&b, 64);
        assert!(cache.lookup(sh, &q_b_far, 1).is_none());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn closest_match_wins_when_multiple_in_tolerance() {
        let mut cache = DeltaCache::new(32, 64);
        let sh = DeltaCache::scope_hash(&[ev(1)]);
        let q = make_sig(0xAA);
        let near = flip_n_bits(&q, 4);
        let far = flip_n_bits(&q, 24);

        cache.insert(sh, far, 1, vec![(99, 99)]);
        cache.insert(sh, near, 1, vec![(0, 42)]);

        let hit = cache.lookup(sh, &q, 1).unwrap();
        assert_eq!(hit, vec![(0, 42)]);
    }
}
