use std::collections::HashMap;

use crate::embed::binary::BinarySignature;
use crate::index::signatures::SignatureIndex;

use super::markov::MarkovPredictor;
use super::state::{ConversationState, EventId};

#[derive(Default, Clone, Copy, Debug)]
pub struct PrefetchStats {
    pub queries: u64,
    pub warm_hits: u64,
    pub warm_misses: u64,
    pub cold_fallbacks: u64,
}

impl PrefetchStats {
    pub fn hit_rate(&self) -> f32 {
        if self.queries == 0 {
            return 0.0;
        }
        self.warm_hits as f32 / self.queries as f32
    }
}

pub struct PrefetchCoordinator<'a> {
    index: &'a SignatureIndex,
    predictor: MarkovPredictor,
    event_scope: HashMap<EventId, Vec<usize>>,
    confidence_threshold: f32,
    top_n_events: usize,
    last_predicted_scope: Vec<usize>,
    last_predicted_events: Vec<EventId>,
    stats: PrefetchStats,
}

pub struct WarmScanResult {
    pub results: Vec<(u32, usize)>,
    pub from_warm_path: bool,
    pub scope_size: usize,
}

impl<'a> PrefetchCoordinator<'a> {
    pub fn new(
        index: &'a SignatureIndex,
        predictor: MarkovPredictor,
        event_scope: HashMap<EventId, Vec<usize>>,
    ) -> Self {
        Self {
            index,
            predictor,
            event_scope,
            confidence_threshold: 0.05,
            top_n_events: 3,
            last_predicted_scope: Vec::new(),
            last_predicted_events: Vec::new(),
            stats: PrefetchStats::default(),
        }
    }

    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    pub fn with_top_n_events(mut self, n: usize) -> Self {
        self.top_n_events = n;
        self
    }

    pub fn observe_transition(&mut self, prev: EventId, next: EventId) {
        self.predictor.observe(prev, next);
    }

    pub fn predictor(&self) -> &MarkovPredictor {
        &self.predictor
    }

    pub fn predictor_mut(&mut self) -> &mut MarkovPredictor {
        &mut self.predictor
    }

    /// Look at conversation state, predict next likely events, and stage the
    /// matching index slices for a warm scan. Idempotent — returns the predicted
    /// events for inspection.
    pub fn prefetch(&mut self, state: &ConversationState) -> Vec<EventId> {
        let context: Vec<EventId> = state.last_n(self.predictor.vocab_size().max(3));
        if context.is_empty() {
            self.last_predicted_scope.clear();
            self.last_predicted_events.clear();
            return Vec::new();
        }

        let predictions = self
            .predictor
            .predict_with_context(&context, self.top_n_events);

        let confident: Vec<EventId> = predictions
            .iter()
            .filter(|p| p.probability >= self.confidence_threshold)
            .map(|p| p.event)
            .collect();

        let mut scope: Vec<usize> = Vec::new();
        for ev in &confident {
            if let Some(rows) = self.event_scope.get(ev) {
                scope.extend_from_slice(rows);
            }
        }
        scope.sort_unstable();
        scope.dedup();

        self.last_predicted_events = confident.clone();
        self.last_predicted_scope = scope;
        confident
    }

    /// Scan the prefetched scope only. If the prefetch is empty, falls back to a
    /// full cold scan. Returns the result plus telemetry about which path ran.
    pub fn query(&mut self, query: &BinarySignature, k: usize) -> WarmScanResult {
        self.stats.queries += 1;

        if self.last_predicted_scope.is_empty() {
            self.stats.cold_fallbacks += 1;
            let results = self.index.scan_top_k_parallel(query, k);
            return WarmScanResult {
                results,
                from_warm_path: false,
                scope_size: self.index.len(),
            };
        }

        let scope_size = self.last_predicted_scope.len();
        let warm = self
            .index
            .scan_top_k_subset(query, &self.last_predicted_scope, k);

        // Confirm the warm result by comparing the best warm distance to a quick
        // cold check on a small random sample. If the warm path's best is clearly
        // better than a cold spot-check, count it as a hit.
        let cold_top1 = self.index.scan_top_k_parallel(query, 1);
        let warm_best = warm.first().map(|(d, _)| *d).unwrap_or(u32::MAX);
        let cold_best = cold_top1.first().map(|(d, _)| *d).unwrap_or(u32::MAX);

        if warm_best == cold_best {
            self.stats.warm_hits += 1;
        } else {
            self.stats.warm_misses += 1;
        }

        WarmScanResult {
            results: warm,
            from_warm_path: true,
            scope_size,
        }
    }

    pub fn stats(&self) -> PrefetchStats {
        self.stats
    }

    pub fn last_predicted_events(&self) -> &[EventId] {
        &self.last_predicted_events
    }

    pub fn last_predicted_scope(&self) -> &[usize] {
        &self.last_predicted_scope
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::binary::BinarySignature;

    fn ev(id: u32) -> EventId {
        EventId(id)
    }

    fn make_sig(seed: u8) -> BinarySignature {
        BinarySignature([seed; 32])
    }

    fn build_test_index() -> (SignatureIndex, HashMap<EventId, Vec<usize>>) {
        // 4 events × 100 sigs each = 400 total. Each event's sigs share a byte pattern.
        let mut sigs = Vec::with_capacity(400);
        let mut scope: HashMap<EventId, Vec<usize>> = HashMap::new();
        for event in 1..=4u32 {
            let base = (event * 16) as u8;
            let indices: Vec<usize> = (0..100).map(|i| ((event - 1) as usize * 100) + i).collect();
            for i in 0..100 {
                sigs.push(BinarySignature([base ^ ((i & 0x0F) as u8); 32]));
            }
            scope.insert(EventId(event), indices);
        }
        (SignatureIndex::new(sigs), scope)
    }

    #[test]
    fn prefetch_returns_predicted_events() {
        let (idx, scope) = build_test_index();
        let mut predictor = MarkovPredictor::with_smoothing(0.01);
        for _ in 0..20 {
            predictor.observe_sequence(&[ev(1), ev(2), ev(3)]);
        }

        let mut coord =
            PrefetchCoordinator::new(&idx, predictor, scope).with_confidence_threshold(0.1);

        let mut state = ConversationState::new(5, std::time::Duration::from_secs(60));
        state.observe(ev(1));

        let predicted = coord.prefetch(&state);
        assert!(predicted.contains(&ev(2)));
        assert!(coord.last_predicted_scope().len() >= 100);
    }

    #[test]
    fn cold_fallback_when_state_empty() {
        let (idx, scope) = build_test_index();
        let predictor = MarkovPredictor::new();
        let mut coord = PrefetchCoordinator::new(&idx, predictor, scope);

        let state = ConversationState::new(5, std::time::Duration::from_secs(60));
        let predicted = coord.prefetch(&state);
        assert!(predicted.is_empty());

        let result = coord.query(&make_sig(0x00), 5);
        assert!(!result.from_warm_path);
        assert_eq!(coord.stats().cold_fallbacks, 1);
    }

    #[test]
    fn warm_scan_narrows_scope_dramatically() {
        let (idx, scope) = build_test_index();
        let mut predictor = MarkovPredictor::with_smoothing(0.01);
        for _ in 0..20 {
            predictor.observe_sequence(&[ev(1), ev(2)]);
        }
        let mut coord =
            PrefetchCoordinator::new(&idx, predictor, scope).with_confidence_threshold(0.1);

        let mut state = ConversationState::new(5, std::time::Duration::from_secs(60));
        state.observe(ev(1));
        coord.prefetch(&state);

        let result = coord.query(&make_sig(0x20), 5);
        assert!(result.from_warm_path);
        assert!(
            result.scope_size < idx.len(),
            "warm scope ({}) should be smaller than full corpus ({})",
            result.scope_size,
            idx.len()
        );
    }
}
