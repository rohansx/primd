use std::time::Duration;

use crate::cold_tier::ColdTier;
use crate::embed::binary::BinarySignature;
use crate::index::shards::{HierarchicalIndex, SearchOptions};
use crate::predict::{
    ConversationState, DeltaCache, EmitDecision, EventId, MarkovPredictor, NextTurnPredictor,
    StreamingQuery,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedBy {
    FullScan,
    ShardScan,
    Speculative,
    DeltaCache,
}

#[derive(Debug, Clone)]
pub struct QueryOutput {
    pub results: Vec<(u32, usize)>,
    pub top_event: Option<EventId>,
    pub predicted_events: Vec<EventId>,
    pub shard_scope_size: usize,
    pub served_by: ServedBy,
    /// Cold-tier hits, if a cold tier was attached and contained
    /// relevant signatures. Empty when no cold tier is present, the
    /// cold tier is empty, or no cold-tier hits matched the query.
    /// Each tuple is `(hamming_distance, event_id, doc_idx)`.
    pub cold_results: Vec<(u32, EventId, usize)>,
}

pub struct QueryContext {
    state: ConversationState,
    predictor: Box<dyn NextTurnPredictor>,
    gate: StreamingQuery,
    delta_cache: DeltaCache,
    search_options: SearchOptions,
    predicted_events: Vec<EventId>,
    predicted_scope: Vec<usize>,
    cached_results: Option<Vec<(u32, usize)>>,
    cached_for_sig: Option<BinarySignature>,
    /// Optional cold tier (v0.4). When set, [`Self::finalize`] also
    /// queries the cold tier and merges results with the hot-tier hits.
    /// Independent of the hot path: hot tier still serves the cache-hit
    /// fast path; cold tier supplements when the hot tier returns
    /// sparse hits.
    cold_tier: Option<Box<dyn ColdTier>>,
}

impl QueryContext {
    pub fn new() -> Self {
        Self::with_predictor(MarkovPredictor::new())
    }

    /// Construct with any concrete predictor. The predictor is boxed; v0.2's
    /// `SrPredictor` and `HybridPredictor` plug in through the same entry point.
    pub fn with_predictor<P: NextTurnPredictor + 'static>(predictor: P) -> Self {
        Self::with_boxed_predictor(Box::new(predictor))
    }

    /// Construct from an already-boxed predictor — useful when the predictor's
    /// concrete type is decided at runtime (e.g. loaded from a config flag).
    pub fn with_boxed_predictor(predictor: Box<dyn NextTurnPredictor>) -> Self {
        Self {
            state: ConversationState::new(8, Duration::from_secs(60)),
            predictor,
            gate: StreamingQuery::new(24),
            delta_cache: DeltaCache::new(16, 64),
            search_options: SearchOptions::default(),
            predicted_events: Vec::new(),
            predicted_scope: Vec::new(),
            cached_results: None,
            cached_for_sig: None,
            cold_tier: None,
        }
    }

    /// Attach a cold tier. After this, every [`Self::finalize`] also
    /// queries the cold tier and merges results with hot-tier hits
    /// before returning the top-K.
    pub fn with_cold_tier(mut self, cold: Box<dyn ColdTier>) -> Self {
        self.cold_tier = Some(cold);
        self
    }

    /// Whether a cold tier is attached.
    pub fn has_cold_tier(&self) -> bool {
        self.cold_tier.is_some()
    }

    /// Borrow the attached cold tier, if any. Lets a session-manager
    /// (e.g. `primd serve`'s reset handler) persist the tier via its
    /// trait `save` method without needing to downcast the box.
    pub fn cold_tier(&self) -> Option<&dyn ColdTier> {
        self.cold_tier.as_deref()
    }

    pub fn with_search_options(mut self, options: SearchOptions) -> Self {
        self.search_options = options;
        self
    }

    pub fn warm_next(&mut self, index: &HierarchicalIndex) -> Vec<EventId> {
        let context = self.state.last_n(3);
        if context.is_empty() {
            self.predicted_events.clear();
            self.predicted_scope.clear();
            return Vec::new();
        }

        self.predicted_events = self
            .predictor
            .predict(&context, self.search_options.max_candidate_events)
            .into_iter()
            .map(|p| p.event)
            .collect();
        self.predicted_scope = index.events().union_scope(&self.predicted_events);
        self.predicted_events.clone()
    }

    pub fn observe_partial(
        &mut self,
        index: &HierarchicalIndex,
        partial: BinarySignature,
        top_k: usize,
    ) {
        let EmitDecision::Emitted(sig) = self.gate.update(partial) else {
            return;
        };

        let scope_hash = self.delta_scope_hash();
        if let Some(cached) = self.delta_cache.lookup(scope_hash, &sig, top_k) {
            self.cached_results = Some(cached);
            self.cached_for_sig = Some(sig);
            return;
        }

        let search = if self.predicted_scope.is_empty() {
            index.search(&sig, top_k, &self.search_options)
        } else {
            let results = if self.search_options.parallel {
                index
                    .signatures()
                    .scan_top_k_subset_parallel(&sig, &self.predicted_scope, top_k)
            } else {
                index
                    .signatures()
                    .scan_top_k_subset(&sig, &self.predicted_scope, top_k)
            };
            crate::index::shards::SearchResult {
                results,
                coarse_results: Vec::new(),
                candidate_events: self.predicted_events.clone(),
                shard_scope_size: self.predicted_scope.len(),
                used_shards: true,
            }
        };

        self.delta_cache
            .insert(scope_hash, sig, top_k, search.results.clone());
        self.cached_results = Some(search.results);
        self.cached_for_sig = Some(sig);
    }

    pub fn finalize(
        &mut self,
        index: &HierarchicalIndex,
        final_sig: BinarySignature,
        top_k: usize,
    ) -> QueryOutput {
        if let (Some(spec), Some(cached)) = (self.cached_for_sig, &self.cached_results)
            && spec.hamming_distance(&final_sig) <= 16
            && cached.len() == top_k
        {
            let cached_clone = cached.clone();
            return self.finish(
                index,
                cached_clone,
                ServedBy::Speculative,
                self.predicted_scope.len(),
                &final_sig,
                top_k,
            );
        }

        let scope_hash = self.delta_scope_hash();
        if let Some(cached) = self.delta_cache.lookup(scope_hash, &final_sig, top_k) {
            return self.finish(
                index,
                cached,
                ServedBy::DeltaCache,
                self.predicted_scope.len(),
                &final_sig,
                top_k,
            );
        }

        let search = if self.predicted_scope.is_empty() {
            index.search(&final_sig, top_k, &self.search_options)
        } else {
            let results = if self.search_options.parallel {
                index.signatures().scan_top_k_subset_parallel(
                    &final_sig,
                    &self.predicted_scope,
                    top_k,
                )
            } else {
                index
                    .signatures()
                    .scan_top_k_subset(&final_sig, &self.predicted_scope, top_k)
            };
            crate::index::shards::SearchResult {
                results,
                coarse_results: Vec::new(),
                candidate_events: self.predicted_events.clone(),
                shard_scope_size: self.predicted_scope.len(),
                used_shards: !self.predicted_scope.is_empty(),
            }
        };
        self.delta_cache
            .insert(scope_hash, final_sig, top_k, search.results.clone());
        let served_by = if search.used_shards {
            ServedBy::ShardScan
        } else {
            ServedBy::FullScan
        };
        self.finish(
            index,
            search.results,
            served_by,
            search.shard_scope_size,
            &final_sig,
            top_k,
        )
    }

    pub fn reset_utterance(&mut self) {
        self.gate.reset();
        self.cached_results = None;
        self.cached_for_sig = None;
    }

    pub fn conversation(&self) -> &ConversationState {
        &self.state
    }

    pub fn predictor(&self) -> &dyn NextTurnPredictor {
        self.predictor.as_ref()
    }

    pub fn predicted_scope_size(&self) -> usize {
        self.predicted_scope.len()
    }

    fn finish(
        &mut self,
        index: &HierarchicalIndex,
        results: Vec<(u32, usize)>,
        served_by: ServedBy,
        shard_scope_size: usize,
        final_sig: &BinarySignature,
        top_k: usize,
    ) -> QueryOutput {
        let top_event = results
            .first()
            .and_then(|&(_, idx)| index.events().doc_event(idx));

        if let Some(next) = top_event {
            if let Some(prev) = self.state.last() {
                self.predictor.observe(prev, next);
            }
            self.state.observe(next);
        }

        // Cold-tier consultation. Only fires when a cold tier is
        // attached and non-empty — the cost-on-no-cold-tier is one
        // option-check.
        let cold_results = self
            .cold_tier
            .as_ref()
            .filter(|ct| !ct.is_empty())
            .map(|ct| ct.search(final_sig, top_k))
            .unwrap_or_default();

        // Carry through the predictions that drove this turn. Next-turn
        // prefetch is the caller's responsibility via warm_next, so it can
        // happen during TTS playback instead of inflating user-visible
        // finalize latency.
        let _ = index;
        let predicted_events = self.predicted_events.clone();
        self.reset_utterance();
        QueryOutput {
            results,
            top_event,
            predicted_events,
            shard_scope_size,
            served_by,
            cold_results,
        }
    }

    fn delta_scope_hash(&self) -> u64 {
        DeltaCache::scope_hash(&self.predicted_events)
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use super::*;
    use crate::index::events::EventCatalog;
    use crate::index::shards::HierarchicalIndex;
    use crate::index::signatures::SignatureIndex;
    use crate::predict::Prediction;

    fn sig(seed: u8) -> BinarySignature {
        BinarySignature([seed; 32])
    }

    #[test]
    fn finalize_updates_state_and_predictions() {
        let signatures = SignatureIndex::new(vec![sig(0x10), sig(0x11), sig(0xF0), sig(0xF1)]);
        let mut named = BTreeMap::new();
        named.insert("a".to_string(), vec![0, 1]);
        named.insert("b".to_string(), vec![2, 3]);
        let events = EventCatalog::from_named_scope(&named, 4);
        let index = HierarchicalIndex::new(signatures, events);
        let mut ctx = QueryContext::new();

        let out = ctx.finalize(&index, sig(0x10), 1);
        assert_eq!(out.top_event, Some(EventId(0)));
        assert_eq!(ctx.conversation().last(), Some(EventId(0)));
    }

    /// Custom predictor that records every observed transition and always
    /// predicts a fixed event. Used to verify that `QueryContext` flows
    /// through the trait surface, not the concrete Markov impl.
    struct RecordingPredictor {
        always_predict: EventId,
        observed: Mutex<Vec<(EventId, EventId)>>,
    }

    impl NextTurnPredictor for RecordingPredictor {
        fn predict(&self, _context: &[EventId], k: usize) -> Vec<Prediction> {
            if k == 0 {
                return Vec::new();
            }
            vec![Prediction {
                event: self.always_predict,
                probability: 1.0,
            }]
        }

        fn observe(&mut self, prev: EventId, next: EventId) {
            self.observed.lock().unwrap().push((prev, next));
        }
    }

    /// Stub cold tier: always returns the same fixed hit when searched.
    /// Used to verify QueryContext wires `cold_results` through finalize.
    struct StubColdTier {
        canned_hit: (u32, EventId, usize),
        is_empty: bool,
    }

    impl crate::cold_tier::ColdTier for StubColdTier {
        fn len(&self) -> usize {
            if self.is_empty { 0 } else { 1 }
        }
        fn add_evicted(&mut self, _sig: BinarySignature, _event: EventId, _doc_idx: usize) {}
        fn search(&self, _query: &BinarySignature, _top_k: usize) -> Vec<(u32, EventId, usize)> {
            if self.is_empty {
                Vec::new()
            } else {
                vec![self.canned_hit]
            }
        }
        fn save(&self, _path: &std::path::Path) -> std::io::Result<()> {
            // Stub — tests don't exercise persistence.
            Ok(())
        }
    }

    #[test]
    fn finalize_populates_cold_results_when_cold_tier_attached() {
        let signatures = SignatureIndex::new(vec![sig(0x10), sig(0xF0)]);
        let mut named = BTreeMap::new();
        named.insert("a".to_string(), vec![0]);
        named.insert("b".to_string(), vec![1]);
        let events = EventCatalog::from_named_scope(&named, 2);
        let index = HierarchicalIndex::new(signatures, events);

        let cold = StubColdTier {
            canned_hit: (42, EventId(99), 12345),
            is_empty: false,
        };
        let mut ctx = QueryContext::new().with_cold_tier(Box::new(cold));
        assert!(ctx.has_cold_tier());

        let out = ctx.finalize(&index, sig(0x10), 1);
        assert_eq!(out.cold_results, vec![(42, EventId(99), 12345)]);
    }

    #[test]
    fn finalize_empty_cold_results_when_no_cold_tier() {
        let signatures = SignatureIndex::new(vec![sig(0x10), sig(0xF0)]);
        let mut named = BTreeMap::new();
        named.insert("a".to_string(), vec![0]);
        named.insert("b".to_string(), vec![1]);
        let events = EventCatalog::from_named_scope(&named, 2);
        let index = HierarchicalIndex::new(signatures, events);

        let mut ctx = QueryContext::new();
        assert!(!ctx.has_cold_tier());

        let out = ctx.finalize(&index, sig(0x10), 1);
        assert!(out.cold_results.is_empty());
    }

    #[test]
    fn finalize_empty_cold_results_when_cold_tier_empty() {
        let signatures = SignatureIndex::new(vec![sig(0x10), sig(0xF0)]);
        let mut named = BTreeMap::new();
        named.insert("a".to_string(), vec![0]);
        named.insert("b".to_string(), vec![1]);
        let events = EventCatalog::from_named_scope(&named, 2);
        let index = HierarchicalIndex::new(signatures, events);

        let empty_cold = StubColdTier {
            canned_hit: (0, EventId(0), 0),
            is_empty: true,
        };
        let mut ctx = QueryContext::new().with_cold_tier(Box::new(empty_cold));

        let out = ctx.finalize(&index, sig(0x10), 1);
        assert!(out.cold_results.is_empty());
    }

    #[test]
    fn warm_next_uses_injected_predictor() {
        let signatures = SignatureIndex::new(vec![sig(0x10), sig(0x11), sig(0xF0), sig(0xF1)]);
        let mut named = BTreeMap::new();
        named.insert("a".to_string(), vec![0, 1]);
        named.insert("b".to_string(), vec![2, 3]);
        let events = EventCatalog::from_named_scope(&named, 4);
        let index = HierarchicalIndex::new(signatures, events);

        let mut ctx = QueryContext::with_predictor(RecordingPredictor {
            always_predict: EventId(1),
            observed: Mutex::new(Vec::new()),
        });

        // Seed conversation state so warm_next has context.
        ctx.finalize(&index, sig(0x10), 1);
        ctx.finalize(&index, sig(0xF0), 1);

        let predicted = ctx.warm_next(&index);
        assert_eq!(predicted, vec![EventId(1)]);
    }
}
