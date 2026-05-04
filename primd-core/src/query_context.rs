use std::time::Duration;

use crate::embed::binary::BinarySignature;
use crate::index::shards::{HierarchicalIndex, SearchOptions};
use crate::predict::{
    ConversationState, DeltaCache, EmitDecision, EventId, MarkovPredictor, StreamingQuery,
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
}

pub struct QueryContext {
    state: ConversationState,
    predictor: MarkovPredictor,
    gate: StreamingQuery,
    delta_cache: DeltaCache,
    search_options: SearchOptions,
    predicted_events: Vec<EventId>,
    predicted_scope: Vec<usize>,
    cached_results: Option<Vec<(u32, usize)>>,
    cached_for_sig: Option<BinarySignature>,
}

impl QueryContext {
    pub fn new() -> Self {
        Self::with_predictor(MarkovPredictor::new())
    }

    pub fn with_predictor(predictor: MarkovPredictor) -> Self {
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
        }
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
            .predict_with_context(&context, self.search_options.max_candidate_events)
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
            return self.finish(
                index,
                cached.clone(),
                ServedBy::Speculative,
                self.predicted_scope.len(),
            );
        }

        let scope_hash = self.delta_scope_hash();
        if let Some(cached) = self.delta_cache.lookup(scope_hash, &final_sig, top_k) {
            return self.finish(
                index,
                cached,
                ServedBy::DeltaCache,
                self.predicted_scope.len(),
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
        self.finish(index, search.results, served_by, search.shard_scope_size)
    }

    pub fn reset_utterance(&mut self) {
        self.gate.reset();
        self.cached_results = None;
        self.cached_for_sig = None;
    }

    pub fn conversation(&self) -> &ConversationState {
        &self.state
    }

    pub fn predictor(&self) -> &MarkovPredictor {
        &self.predictor
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

    use super::*;
    use crate::index::events::EventCatalog;
    use crate::index::shards::HierarchicalIndex;
    use crate::index::signatures::SignatureIndex;

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
}
