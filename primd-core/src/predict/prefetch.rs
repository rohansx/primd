use std::collections::HashMap;
use std::time::Instant;

use crate::embed::binary::BinarySignature;
use crate::index::signatures::SignatureIndex;

use super::markov::MarkovPredictor;
use super::state::{ConversationState, EventId};
use super::streaming::{EmitDecision, StreamingQuery};

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

    pub fn index(&self) -> &SignatureIndex {
        self.index
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

/// Telemetry for `StreamingPrefetcher`: how often the speculative scan during
/// user speech beat us to the answer.
#[derive(Default, Clone, Copy, Debug)]
pub struct StreamingPrefetchStats {
    pub utterances: u64,
    pub partial_updates: u64,
    pub speculative_scans: u64,
    pub finals_served_speculatively: u64,
    pub finals_required_rescan: u64,
    pub total_speculative_scan_ns: u128,
}

impl StreamingPrefetchStats {
    pub fn speculative_hit_rate(&self) -> f32 {
        let total = self.finals_served_speculatively + self.finals_required_rescan;
        if total == 0 {
            return 0.0;
        }
        self.finals_served_speculatively as f32 / total as f32
    }
}

/// Wraps `PrefetchCoordinator` and `StreamingQuery` to drive speculative
/// retrieval *during* user speech. The pipeline:
///
/// 1. STT emits partial transcripts every 50-200ms.
/// 2. The application embeds each partial → `BinarySignature`.
/// 3. `on_partial` decides via the streaming gate whether the partial has
///    drifted enough to be worth re-scanning. If so, it runs a warm scan
///    against the predicted scope and caches the draft top-K.
/// 4. When STT finalizes, `on_final` checks whether the final signature is
///    close enough to the last speculative one; if yes, the cached draft is
///    returned with near-zero latency.
pub struct StreamingPrefetcher<'a> {
    coord: PrefetchCoordinator<'a>,
    gate: StreamingQuery,
    cached_results: Option<Vec<(u32, usize)>>,
    cached_for_sig: Option<BinarySignature>,
    last_top_k: usize,
    stats: StreamingPrefetchStats,
}

impl<'a> StreamingPrefetcher<'a> {
    pub fn new(coord: PrefetchCoordinator<'a>, drift_threshold: u32) -> Self {
        Self {
            coord,
            gate: StreamingQuery::new(drift_threshold),
            cached_results: None,
            cached_for_sig: None,
            last_top_k: 10,
            stats: StreamingPrefetchStats::default(),
        }
    }

    /// Refresh the predicted scope. Call this whenever the conversation state
    /// changes (typically once per finalized turn).
    pub fn prefetch(&mut self, state: &ConversationState) {
        self.coord.prefetch(state);
    }

    /// New STT partial. If the gate emits, runs a speculative warm scan and
    /// caches the result. The caller does not block on this — in production
    /// it would run on a worker thread.
    pub fn on_partial(&mut self, partial: BinarySignature, k: usize) {
        self.stats.partial_updates += 1;
        self.last_top_k = k;
        if let EmitDecision::Emitted(sig) = self.gate.update(partial) {
            let scope = self.coord.last_predicted_scope();
            let started = Instant::now();
            let results = self.coord.index().scan_top_k_subset(&sig, scope, k);
            let elapsed = started.elapsed().as_nanos();

            self.cached_results = Some(results);
            self.cached_for_sig = Some(sig);
            self.stats.speculative_scans += 1;
            self.stats.total_speculative_scan_ns += elapsed;
        }
    }

    /// STT finalized. If the final signature is close enough to the last
    /// speculative emit, the cached top-K is returned for free; otherwise a
    /// fresh warm scan runs against the final signature.
    ///
    /// `accept_drift` is the maximum Hamming distance between the final and
    /// the speculative signature for the cache to be considered fresh.
    pub fn on_final(
        &mut self,
        final_sig: BinarySignature,
        k: usize,
        accept_drift: u32,
    ) -> FinalScanResult {
        let cache_fresh = match (self.cached_for_sig, &self.cached_results) {
            (Some(spec), Some(cached)) => {
                spec.hamming_distance(&final_sig) <= accept_drift && cached.len() == k
            }
            _ => false,
        };

        if cache_fresh {
            self.stats.finals_served_speculatively += 1;
            FinalScanResult {
                results: self.cached_results.clone().unwrap_or_default(),
                served_speculatively: true,
            }
        } else {
            self.stats.finals_required_rescan += 1;
            let scope = self.coord.last_predicted_scope();
            let results = self.coord.index().scan_top_k_subset(&final_sig, scope, k);
            FinalScanResult {
                results,
                served_speculatively: false,
            }
        }
    }

    /// Call when STT finalizes (or utterance ends without finalization).
    pub fn end_utterance(&mut self) {
        self.gate.reset();
        self.cached_results = None;
        self.cached_for_sig = None;
        self.stats.utterances += 1;
    }

    pub fn stats(&self) -> StreamingPrefetchStats {
        self.stats
    }

    pub fn coordinator(&self) -> &PrefetchCoordinator<'a> {
        &self.coord
    }

    pub fn coordinator_mut(&mut self) -> &mut PrefetchCoordinator<'a> {
        &mut self.coord
    }
}

#[derive(Debug, Clone)]
pub struct FinalScanResult {
    pub results: Vec<(u32, usize)>,
    pub served_speculatively: bool,
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

    fn flip_n_bits(sig: &BinarySignature, n: u32) -> BinarySignature {
        let mut out = *sig;
        for i in 0..n {
            let bit = (i * 7) as usize % 256;
            out.0[bit / 8] ^= 1 << (bit % 8);
        }
        out
    }

    #[test]
    fn streaming_serves_final_speculatively_when_close() {
        let (idx, scope) = build_test_index();
        let mut predictor = MarkovPredictor::with_smoothing(0.01);
        for _ in 0..20 {
            predictor.observe_sequence(&[ev(1), ev(2)]);
        }
        let coord = PrefetchCoordinator::new(&idx, predictor, scope).with_confidence_threshold(0.1);
        let mut sp = StreamingPrefetcher::new(coord, 16);

        let mut state = ConversationState::new(5, std::time::Duration::from_secs(60));
        state.observe(ev(1));
        sp.prefetch(&state);

        let partial = make_sig(0x20);
        sp.on_partial(partial, 5);
        let nudged = flip_n_bits(&partial, 4); // well under accept_drift
        let result = sp.on_final(nudged, 5, 16);

        assert!(result.served_speculatively);
        assert_eq!(sp.stats().finals_served_speculatively, 1);
    }

    #[test]
    fn streaming_rescans_when_final_diverges() {
        let (idx, scope) = build_test_index();
        let mut predictor = MarkovPredictor::with_smoothing(0.01);
        for _ in 0..20 {
            predictor.observe_sequence(&[ev(1), ev(2)]);
        }
        let coord = PrefetchCoordinator::new(&idx, predictor, scope).with_confidence_threshold(0.1);
        let mut sp = StreamingPrefetcher::new(coord, 16);

        let mut state = ConversationState::new(5, std::time::Duration::from_secs(60));
        state.observe(ev(1));
        sp.prefetch(&state);

        let partial = make_sig(0x20);
        sp.on_partial(partial, 5);
        let very_different = flip_n_bits(&partial, 64);
        let result = sp.on_final(very_different, 5, 16);

        assert!(!result.served_speculatively);
        assert_eq!(sp.stats().finals_required_rescan, 1);
    }

    #[test]
    fn streaming_suppresses_redundant_partials() {
        let (idx, scope) = build_test_index();
        let mut predictor = MarkovPredictor::with_smoothing(0.01);
        for _ in 0..20 {
            predictor.observe_sequence(&[ev(1), ev(2)]);
        }
        let coord = PrefetchCoordinator::new(&idx, predictor, scope).with_confidence_threshold(0.1);
        let mut sp = StreamingPrefetcher::new(coord, 16);

        let mut state = ConversationState::new(5, std::time::Duration::from_secs(60));
        state.observe(ev(1));
        sp.prefetch(&state);

        let base = make_sig(0x20);
        sp.on_partial(base, 5);
        for n in 1..5 {
            sp.on_partial(flip_n_bits(&base, n), 5);
        }
        // Only the first partial should have triggered a speculative scan.
        assert_eq!(sp.stats().speculative_scans, 1);
        assert_eq!(sp.stats().partial_updates, 5);
    }
}
