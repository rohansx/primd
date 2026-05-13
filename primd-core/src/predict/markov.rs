use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::predictor::{NextTurnPredictor, Prediction};
use super::state::EventId;

const MIN_CONTEXT_OBSERVATIONS: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
struct WeightedCount {
    weight: f32,
    last: Instant,
}

impl WeightedCount {
    fn zero(now: Instant) -> Self {
        Self {
            weight: 0.0,
            last: now,
        }
    }

    fn observe(&mut self, now: Instant, half_life: Option<Duration>) {
        if let Some(hl) = half_life {
            self.weight *= decay_factor(self.last, now, hl);
        }
        self.weight += 1.0;
        self.last = now;
    }

    fn current(&self, now: Instant, half_life: Option<Duration>) -> f32 {
        match half_life {
            Some(hl) => self.weight * decay_factor(self.last, now, hl),
            None => self.weight,
        }
    }

    fn from_weight(weight: f32, now: Instant) -> Self {
        Self { weight, last: now }
    }
}

fn decay_factor(from: Instant, to: Instant, half_life: Duration) -> f32 {
    let elapsed = to.saturating_duration_since(from).as_secs_f64();
    let hl = half_life.as_secs_f64().max(1e-9);
    (-elapsed * std::f64::consts::LN_2 / hl).exp() as f32
}

#[derive(Clone)]
pub struct MarkovPredictor {
    tables: Vec<HashMap<Vec<EventId>, HashMap<EventId, WeightedCount>>>,
    totals: Vec<HashMap<Vec<EventId>, WeightedCount>>,
    vocab: HashSet<EventId>,
    smoothing: f32,
    max_order: usize,
    half_life: Option<Duration>,
}

#[derive(Serialize, Deserialize)]
struct PersistedPredictor {
    smoothing: f32,
    max_order: usize,
    half_life_secs: Option<f64>,
    vocab: Vec<u32>,
    rows: Vec<PersistedRow>,
}

#[derive(Serialize, Deserialize)]
struct PersistedRow {
    context: Vec<u32>,
    next: u32,
    weight: f32,
}

impl MarkovPredictor {
    pub fn new() -> Self {
        Self::with_order_and_smoothing(1, 1.0)
    }

    pub fn with_smoothing(alpha: f32) -> Self {
        Self::with_order_and_smoothing(1, alpha)
    }

    pub fn with_max_order(order: usize) -> Self {
        Self::with_order_and_smoothing(order, 1.0)
    }

    pub fn with_order_and_smoothing(max_order: usize, alpha: f32) -> Self {
        assert!(max_order >= 1, "max_order must be at least 1");
        Self {
            tables: vec![HashMap::new(); max_order],
            totals: vec![HashMap::new(); max_order],
            vocab: HashSet::new(),
            smoothing: alpha,
            max_order,
            half_life: None,
        }
    }

    pub fn with_half_life(mut self, half_life: Duration) -> Self {
        self.half_life = Some(half_life);
        self
    }

    pub fn observe(&mut self, prev: EventId, next: EventId) {
        self.observe_at(prev, next, Instant::now());
    }

    pub fn observe_at(&mut self, prev: EventId, next: EventId, at: Instant) {
        self.record(&[prev], next, at);
    }

    pub fn observe_sequence(&mut self, events: &[EventId]) {
        let now = Instant::now();
        for &e in events {
            self.vocab.insert(e);
        }
        for i in 1..events.len() {
            let next = events[i];
            for order in 1..=self.max_order {
                if i >= order {
                    let context = &events[i - order..i];
                    self.record(context, next, now);
                }
            }
        }
    }

    fn record(&mut self, context: &[EventId], next: EventId, at: Instant) {
        for &e in context {
            self.vocab.insert(e);
        }
        self.vocab.insert(next);
        let order = context.len();
        if order == 0 || order > self.max_order {
            return;
        }
        let key: Vec<EventId> = context.to_vec();

        let half_life = self.half_life;
        let row = self.tables[order - 1].entry(key.clone()).or_default();
        row.entry(next)
            .or_insert_with(|| WeightedCount::zero(at))
            .observe(at, half_life);

        self.totals[order - 1]
            .entry(key)
            .or_insert_with(|| WeightedCount::zero(at))
            .observe(at, half_life);
    }

    pub fn predict(&self, prev: EventId, k: usize) -> Vec<Prediction> {
        self.predict_with_context(&[prev], k)
    }

    pub fn predict_with_context(&self, context: &[EventId], k: usize) -> Vec<Prediction> {
        self.predict_with_context_at(context, k, Instant::now())
    }

    pub fn predict_with_context_at(
        &self,
        context: &[EventId],
        k: usize,
        at: Instant,
    ) -> Vec<Prediction> {
        let max_lookback = context.len().min(self.max_order);
        for order in (1..=max_lookback).rev() {
            let key = &context[context.len() - order..];
            let total = self.totals[order - 1]
                .get(key)
                .map(|w| w.current(at, self.half_life))
                .unwrap_or(0.0);
            if total >= MIN_CONTEXT_OBSERVATIONS {
                return self.score(key, total, k, at);
            }
        }
        self.uniform(k)
    }

    fn score(&self, key: &[EventId], total: f32, k: usize, at: Instant) -> Vec<Prediction> {
        let order = key.len();
        let row = self.tables[order - 1].get(key);
        let vocab_size = self.vocab.len().max(1) as f32;
        let denom = total + self.smoothing * vocab_size;

        let mut scored: Vec<Prediction> = self
            .vocab
            .iter()
            .map(|&event| {
                let count = row
                    .and_then(|r| r.get(&event))
                    .map(|w| w.current(at, self.half_life))
                    .unwrap_or(0.0);
                Prediction {
                    event,
                    probability: (count + self.smoothing) / denom,
                }
            })
            .collect();

        scored.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        scored.truncate(k);
        scored
    }

    fn uniform(&self, k: usize) -> Vec<Prediction> {
        let n = self.vocab.len().max(1) as f32;
        let mut scored: Vec<Prediction> = self
            .vocab
            .iter()
            .map(|&event| Prediction {
                event,
                probability: 1.0 / n,
            })
            .collect();
        scored.sort_by_key(|p| p.event);
        scored.truncate(k);
        scored
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn observed_transitions(&self, prev: EventId) -> f32 {
        self.totals[0]
            .get(&vec![prev])
            .map(|w| w.current(Instant::now(), self.half_life))
            .unwrap_or(0.0)
    }

    pub fn save_to_file(&self, path: &Path) -> crate::Result<()> {
        let now = Instant::now();
        let mut rows = Vec::new();
        for table in &self.tables {
            for (context, nexts) in table {
                for (&next, count) in nexts {
                    rows.push(PersistedRow {
                        context: context.iter().map(|e| e.0).collect(),
                        next: next.0,
                        weight: count.current(now, self.half_life),
                    });
                }
            }
        }

        let persisted = PersistedPredictor {
            smoothing: self.smoothing,
            max_order: self.max_order,
            half_life_secs: self.half_life.map(|d| d.as_secs_f64()),
            vocab: self.vocab.iter().map(|e| e.0).collect(),
            rows,
        };
        let bytes = serde_json::to_vec_pretty(&persisted)
            .map_err(|e| crate::PrimdError::Embedder(format!("serialize predictor: {e}")))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> crate::Result<Self> {
        let bytes = std::fs::read(path)?;
        let persisted: PersistedPredictor = serde_json::from_slice(&bytes)
            .map_err(|e| crate::PrimdError::Embedder(format!("parse predictor: {e}")))?;
        let mut out = Self::with_order_and_smoothing(persisted.max_order, persisted.smoothing);
        if let Some(secs) = persisted.half_life_secs {
            out.half_life = Some(Duration::from_secs_f64(secs));
        }

        let now = Instant::now();
        out.vocab = persisted.vocab.into_iter().map(EventId).collect();
        for row in persisted.rows {
            let context: Vec<EventId> = row.context.into_iter().map(EventId).collect();
            let order = context.len();
            if order == 0 || order > out.max_order {
                continue;
            }
            let next = EventId(row.next);
            out.tables[order - 1]
                .entry(context.clone())
                .or_default()
                .insert(next, WeightedCount::from_weight(row.weight, now));
        }

        for (order_idx, table) in out.tables.iter().enumerate() {
            for (context, row) in table {
                let total: f32 = row.values().map(|w| w.weight).sum();
                out.totals[order_idx]
                    .insert(context.clone(), WeightedCount::from_weight(total, now));
            }
        }
        Ok(out)
    }
}

impl Default for MarkovPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl NextTurnPredictor for MarkovPredictor {
    fn predict(&self, context: &[EventId], k: usize) -> Vec<Prediction> {
        self.predict_with_context(context, k)
    }

    fn observe(&mut self, prev: EventId, next: EventId) {
        MarkovPredictor::observe(self, prev, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: u32) -> EventId {
        EventId(id)
    }

    #[test]
    fn predicts_dominant_transition() {
        let mut m = MarkovPredictor::with_smoothing(0.01);
        for _ in 0..100 {
            m.observe(ev(1), ev(2));
        }
        m.observe(ev(1), ev(3));
        let preds = m.predict(ev(1), 2);
        assert_eq!(preds[0].event, ev(2));
        assert!(preds[0].probability > 0.9);
    }

    #[test]
    fn returns_topk_in_descending_order() {
        let mut m = MarkovPredictor::new();
        m.observe_sequence(&[ev(1), ev(2), ev(1), ev(3), ev(1), ev(2), ev(1), ev(4)]);
        let preds = m.predict(ev(1), 3);
        assert_eq!(preds.len(), 3);
        for w in preds.windows(2) {
            assert!(w[0].probability >= w[1].probability);
        }
        assert_eq!(preds[0].event, ev(2));
    }

    #[test]
    fn smoothing_assigns_nonzero_to_unseen() {
        let mut m = MarkovPredictor::with_smoothing(1.0);
        m.observe(ev(1), ev(2));
        m.observe(ev(1), ev(2));
        m.observe(ev(5), ev(6));
        m.observe(ev(5), ev(6));
        let preds = m.predict(ev(1), 10);
        for p in &preds {
            assert!(p.probability > 0.0);
        }
        let prob_seen = preds.iter().find(|p| p.event == ev(2)).unwrap().probability;
        let prob_unseen = preds.iter().find(|p| p.event == ev(6)).unwrap().probability;
        assert!(prob_seen > prob_unseen);
    }

    #[test]
    fn unknown_prev_returns_uniform() {
        let mut m = MarkovPredictor::with_smoothing(1.0);
        m.observe(ev(1), ev(2));
        m.observe(ev(1), ev(3));
        let preds = m.predict(ev(99), 10);
        let probs: Vec<f32> = preds.iter().map(|p| p.probability).collect();
        let max = probs.iter().cloned().fold(f32::MIN, f32::max);
        let min = probs.iter().cloned().fold(f32::MAX, f32::min);
        assert!((max - min).abs() < 1e-5);
    }

    #[test]
    fn higher_order_context_overrides_lower() {
        let mut m = MarkovPredictor::with_order_and_smoothing(2, 0.01);
        for _ in 0..50 {
            m.observe_sequence(&[ev(1), ev(2)]);
        }
        for _ in 0..20 {
            m.observe_sequence(&[ev(4), ev(1), ev(3)]);
        }

        let first_order = m.predict_with_context(&[ev(1)], 2);
        assert_eq!(first_order[0].event, ev(2));

        let second_order = m.predict_with_context(&[ev(4), ev(1)], 2);
        assert_eq!(second_order[0].event, ev(3));
    }

    #[test]
    fn backs_off_when_higher_order_unseen() {
        let mut m = MarkovPredictor::with_order_and_smoothing(3, 0.01);
        for _ in 0..50 {
            m.observe_sequence(&[ev(1), ev(2)]);
        }
        let preds = m.predict_with_context(&[ev(99), ev(100), ev(1)], 1);
        assert_eq!(preds[0].event, ev(2));
    }

    #[test]
    fn time_decay_halves_weight_over_half_life() {
        let half_life = Duration::from_secs(10);
        let mut m = MarkovPredictor::with_smoothing(0.01).with_half_life(half_life);
        let t0 = Instant::now();

        for _ in 0..10 {
            m.observe_at(ev(1), ev(2), t0);
        }

        let immediate = m.totals[0]
            .get(&vec![ev(1)])
            .unwrap()
            .current(t0, m.half_life);
        let after_one_hl = m.totals[0]
            .get(&vec![ev(1)])
            .unwrap()
            .current(t0 + half_life, m.half_life);

        let ratio = after_one_hl / immediate;
        assert!(
            (ratio - 0.5).abs() < 0.01,
            "expected ~0.5 after one half-life, got {ratio}"
        );
    }

    #[test]
    fn recent_pattern_overrides_stale_pattern() {
        let half_life = Duration::from_secs(5);
        let mut m = MarkovPredictor::with_smoothing(0.01).with_half_life(half_life);
        let t0 = Instant::now();

        // Old pattern: 1 → 2 (50 times, long ago)
        for i in 0..50 {
            m.observe_at(ev(1), ev(2), t0 + Duration::from_millis(i));
        }

        // New pattern: 1 → 3 (10 times, recent — many half-lives later)
        let recent = t0 + Duration::from_secs(60);
        for i in 0..10 {
            m.observe_at(ev(1), ev(3), recent + Duration::from_millis(i));
        }

        let preds = m.predict_with_context_at(&[ev(1)], 2, recent + Duration::from_millis(50));
        assert_eq!(preds[0].event, ev(3));
    }

    #[test]
    fn save_and_load_round_trip() {
        let mut m = MarkovPredictor::with_order_and_smoothing(2, 0.01);
        m.observe_sequence(&[ev(1), ev(2), ev(3)]);
        m.observe_sequence(&[ev(1), ev(2), ev(3)]);
        let path = std::env::temp_dir().join("primd-markov-test.json");
        m.save_to_file(&path).unwrap();
        let loaded = MarkovPredictor::load_from_file(&path).unwrap();
        let preds = loaded.predict_with_context(&[ev(1), ev(2)], 1);
        assert_eq!(preds[0].event, ev(3));
        let _ = std::fs::remove_file(path);
    }
}
