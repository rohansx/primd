//! primd-sr — Successor Representation predictor for primd's predictive turn-cache.
//!
//! Replaces the variable-order Markov chain's per-context lookup with a learned
//! predictive map over conversation events. SR generalizes across paraphrased
//! turns the Markov chain treats as distinct, exposes a soft prediction horizon
//! via the discount factor γ, and surfaces a continuous confidence signal
//! that a hybrid wrapper uses to gate between SR and Markov.
//!
//! References:
//! - Dayan 1993, "Improving Generalisation for Temporal Difference Learning: The
//!   Successor Representation," Neural Computation 5(4):613–624.
//! - Stachenfeld, Botvinick, Gershman 2017, "The hippocampus as a predictive
//!   map," Nature Neuroscience 20:1643–1653 (correction at 21:895).
//! - Russek, Momennejad, Botvinick, Gershman, Daw 2017, "The successor
//!   representation in human reinforcement learning," Nature Human Behaviour
//!   1:680–692. TD(0) update rule.
//! - Gershman 2018, "The Successor Representation: Its Computational Logic and
//!   Neural Substrates," J. Neuroscience 38(33):7193–7200.
//!
//! v0.2 ships the tabular variant: SR is stored as a sparse `HashMap` of
//! `EventId` → row. The low-rank (K=32) reduction and SIMD optimization that
//! the strategy memo describes are v0.2.5 work — the tabular form is correct
//! for primd's typical voice corpora (tens to low hundreds of events) and
//! cheap enough that the optimization isn't on any critical path yet.

pub mod low_rank;
pub mod pca;
pub use low_rank::{
    DEFAULT_K as LOW_RANK_DEFAULT_K, LowRankSr, LowRankSrPredictor, MLowInit,
    SIG_BITS as LOW_RANK_SIG_BITS,
};
pub use pca::compute_pca;

use std::collections::{BTreeMap, BTreeSet};

use primd_core::predict::{EventId, MarkovPredictor, NextTurnPredictor, Prediction};
use serde::{Deserialize, Serialize};

/// Default discount factor. ~0.85–0.95 covers typical 5–15-turn voice flows
/// (Russek 2017 §Methods). 0.9 is a balanced default; tune downward for
/// shorter horizons (chitchat) or upward for longer (multi-step support).
pub const DEFAULT_GAMMA: f32 = 0.9;

/// Default base learning rate. Decays as `eta_base / (1 + 0.01 * t)` so early
/// observations move M more aggressively than late ones — matches the
/// `η_t = 0.1 / (1 + 0.01·t)` schedule in the strategy memo (Russek 2017).
pub const DEFAULT_ETA_BASE: f32 = 0.1;

/// Default warmth target: SR is considered confident once it has seen this
/// many transitions. Below it, [`HybridPredictor`] falls back to Markov.
/// Empirically, ~50 transitions cover one moderate-length voice session.
pub const DEFAULT_WARMUP_OBSERVATIONS: u64 = 50;

/// Tabular Successor Representation predictor.
///
/// Each cell `M[s][s']` represents the expected discounted future visits to
/// `s'` starting from `s`, under the conversation policy implicitly defined
/// by the observation stream. The TD(0) update on a transition `s → s'`
/// pushes `M[s, :]` toward `e_s + γ · M[s', :]`, where `e_s` is the one-hot
/// indicator for `s` (this is the t=0 self-visit term).
///
/// First observation of any event `e` initializes `M[e][e] = 1.0` so that
/// bootstrapping correctly propagates the t=0 self-visit through transitions.
#[derive(Clone, Debug)]
pub struct SrPredictor {
    /// Sparse SR matrix: `M[s][s']` ∈ ℝ.
    /// Sparse SR matrix as nested `BTreeMap` for deterministic iteration —
    /// matches the v0.2.6 fix for `LowRankSrPredictor::event_features`.
    /// HashMap-randomized iteration was producing ±5 pp top-1 variance
    /// between bench runs.
    m: BTreeMap<EventId, BTreeMap<EventId, f32>>,
    /// All observed events. Drives the column space of TD updates.
    vocab: BTreeSet<EventId>,
    /// Discount factor γ ∈ (0, 1).
    gamma: f32,
    /// Base learning rate.
    eta_base: f32,
    /// Observation count — feeds `eta` decay and `confidence` warmth.
    t: u64,
    /// Number of transitions before [`Self::confidence`] reaches 1.0.
    warmup: u64,
}

impl SrPredictor {
    /// Predictor with defaults: γ = 0.9, η_base = 0.1, warmup = 50.
    pub fn new() -> Self {
        Self {
            m: BTreeMap::new(),
            vocab: BTreeSet::new(),
            gamma: DEFAULT_GAMMA,
            eta_base: DEFAULT_ETA_BASE,
            t: 0,
            warmup: DEFAULT_WARMUP_OBSERVATIONS,
        }
    }

    /// Override the discount factor. Clamped to (0, 1) — extreme values are
    /// numerically unstable and rarely useful.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma.clamp(1e-3, 1.0 - 1e-3);
        self
    }

    /// Override the base learning rate. Clamped to (0, 1].
    pub fn with_eta_base(mut self, eta_base: f32) -> Self {
        self.eta_base = eta_base.clamp(1e-4, 1.0);
        self
    }

    /// Override the warmup horizon — how many transitions before
    /// `confidence` saturates to 1.0.
    pub fn with_warmup(mut self, warmup: u64) -> Self {
        self.warmup = warmup.max(1);
        self
    }

    /// Current decayed learning rate.
    fn eta(&self) -> f32 {
        self.eta_base / (1.0 + 0.01 * self.t as f32)
    }

    /// Discount factor in current use.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Total transitions observed.
    pub fn observations(&self) -> u64 {
        self.t
    }

    /// Vocabulary size (number of distinct events seen).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Lookup `M[s][s']`, or 0.0 if either coordinate is unseen.
    pub fn get(&self, s: EventId, s_prime: EventId) -> f32 {
        self.m
            .get(&s)
            .and_then(|row| row.get(&s_prime).copied())
            .unwrap_or(0.0)
    }

    /// Ensure `M[s][s] = 1.0` on first sight of `s`. Idempotent — does not
    /// overwrite an existing self-visit cell that's already been TD-updated.
    fn ensure_self_visit(&mut self, s: EventId) {
        self.vocab.insert(s);
        let row = self.m.entry(s).or_default();
        row.entry(s).or_insert(1.0);
    }

    /// Serialize to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let snapshot = Snapshot::from(self);
        let bytes = serde_json::to_vec_pretty(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Deserialize from a JSON file written by [`Self::save_to_file`].
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: Snapshot = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(snapshot.into())
    }
}

impl Default for SrPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl NextTurnPredictor for SrPredictor {
    fn predict(&self, context: &[EventId], k: usize) -> Vec<Prediction> {
        if k == 0 {
            return Vec::new();
        }
        let prev = match context.last() {
            Some(&e) => e,
            None => return Vec::new(),
        };
        let row = match self.m.get(&prev) {
            Some(r) => r,
            None => return Vec::new(),
        };

        // Drop the self-visit (column = prev) — it's the bootstrap anchor,
        // not a useful next-turn prediction.
        let mut entries: Vec<(EventId, f32)> = row
            .iter()
            .filter(|&(&col, _)| col != prev)
            .map(|(&col, &val)| (col, val.max(0.0)))
            .collect();

        // Sort descending by value.
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(k);

        // Normalize to a soft distribution so callers comparing across
        // predictor impls see comparable probability scales.
        let sum: f32 = entries.iter().map(|(_, v)| *v).sum();
        if sum <= 0.0 {
            return Vec::new();
        }
        entries
            .into_iter()
            .map(|(event, value)| Prediction {
                event,
                probability: value / sum,
            })
            .collect()
    }

    fn observe(&mut self, prev: EventId, next: EventId) {
        // Seed self-visits for both endpoints so bootstrap terms are correct
        // on the very first observation.
        self.ensure_self_visit(prev);
        self.ensure_self_visit(next);

        let eta = self.eta();
        let gamma = self.gamma;

        // Snapshot the bootstrap row before mutating the source row to keep
        // the math TD(0) (not TD(λ) or some interleaved update).
        let next_row: BTreeMap<EventId, f32> = self.m.get(&next).cloned().unwrap_or_default();
        let columns: Vec<EventId> = self.vocab.iter().copied().collect();

        let prev_row = self.m.entry(prev).or_default();
        for col in columns {
            let indicator = if col == prev { 1.0 } else { 0.0 };
            let bootstrap = next_row.get(&col).copied().unwrap_or(0.0);
            let target = indicator + gamma * bootstrap;
            let current = prev_row.get(&col).copied().unwrap_or(0.0);
            let updated = current + eta * (target - current);
            prev_row.insert(col, updated);
        }

        self.t += 1;
    }

    /// Linear warmth signal: `t / warmup`, clamped to `[0, 1]`. Sufficient
    /// for the Hybrid wrapper's gating in v0.2. The spectral-gap signal
    /// from the low-rank `M_low` (Stachenfeld 2017 eigenstructure) lands in
    /// v0.2.5 alongside the K=32 reduction.
    fn confidence(&self) -> f32 {
        if self.warmup == 0 {
            return 1.0;
        }
        (self.t as f32 / self.warmup as f32).min(1.0)
    }
}

/// Hybrid SR + Markov predictor. Routes predictions to SR once SR is warm,
/// trains both on every observation, and reports the max confidence so
/// downstream callers (the `QueryContext` warm gate) see a useful value
/// during the cold-start period.
///
/// Default threshold 0.5 — SR is used after roughly half a `warmup`-worth of
/// transitions. Tuning downward biases toward SR sooner (more aggressive,
/// potentially less stable); upward biases toward Markov longer.
pub struct HybridPredictor {
    sr: SrPredictor,
    markov: MarkovPredictor,
    threshold: f32,
}

impl HybridPredictor {
    /// Wrap an existing pair. Threshold defaults to 0.5.
    pub fn new(sr: SrPredictor, markov: MarkovPredictor) -> Self {
        Self {
            sr,
            markov,
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn sr(&self) -> &SrPredictor {
        &self.sr
    }

    pub fn markov(&self) -> &MarkovPredictor {
        &self.markov
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for HybridPredictor {
    fn default() -> Self {
        Self::new(SrPredictor::new(), MarkovPredictor::new())
    }
}

impl NextTurnPredictor for HybridPredictor {
    fn predict(&self, context: &[EventId], k: usize) -> Vec<Prediction> {
        if self.sr.confidence() >= self.threshold {
            let sr_preds = <SrPredictor as NextTurnPredictor>::predict(&self.sr, context, k);
            if !sr_preds.is_empty() {
                return sr_preds;
            }
        }
        // UFCS — Markov has an inherent `predict(prev, k)` that would shadow
        // the trait method's slice signature otherwise.
        <MarkovPredictor as NextTurnPredictor>::predict(&self.markov, context, k)
    }

    fn observe(&mut self, prev: EventId, next: EventId) {
        <SrPredictor as NextTurnPredictor>::observe(&mut self.sr, prev, next);
        <MarkovPredictor as NextTurnPredictor>::observe(&mut self.markov, prev, next);
    }

    fn confidence(&self) -> f32 {
        self.sr.confidence().max(self.markov.confidence())
    }

    fn as_markov(&self) -> Option<&MarkovPredictor> {
        Some(&self.markov)
    }
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Snapshot {
    gamma: f32,
    eta_base: f32,
    t: u64,
    warmup: u64,
    vocab: Vec<u32>,
    rows: Vec<SnapshotRow>,
}

#[derive(Serialize, Deserialize)]
struct SnapshotRow {
    source: u32,
    target: u32,
    value: f32,
}

impl From<&SrPredictor> for Snapshot {
    fn from(p: &SrPredictor) -> Self {
        let mut rows = Vec::new();
        for (src, row) in &p.m {
            for (tgt, val) in row {
                rows.push(SnapshotRow {
                    source: src.0,
                    target: tgt.0,
                    value: *val,
                });
            }
        }
        Snapshot {
            gamma: p.gamma,
            eta_base: p.eta_base,
            t: p.t,
            warmup: p.warmup,
            vocab: p.vocab.iter().map(|e| e.0).collect(),
            rows,
        }
    }
}

impl From<Snapshot> for SrPredictor {
    fn from(s: Snapshot) -> Self {
        let mut p = SrPredictor::new()
            .with_gamma(s.gamma)
            .with_eta_base(s.eta_base)
            .with_warmup(s.warmup);
        p.t = s.t;
        p.vocab = s.vocab.into_iter().map(EventId).collect();
        for row in s.rows {
            p.m.entry(EventId(row.source))
                .or_default()
                .insert(EventId(row.target), row.value);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: u32) -> EventId {
        EventId(id)
    }

    /// Analytical SR for a 3-state cyclic chain A→B→C→A under deterministic
    /// policy with discount γ. Every state visits every other state exactly
    /// once per 3-step cycle, so:
    ///
    ///   M[A,A] = 1 + γ³ + γ⁶ + ... = 1 / (1 − γ³)
    ///   M[A,B] = γ + γ⁴ + γ⁷ + ... = γ / (1 − γ³)
    ///   M[A,C] = γ² + γ⁵ + γ⁸ + ... = γ² / (1 − γ³)
    fn analytical_cyclic(gamma: f32) -> [[f32; 3]; 3] {
        let denom = 1.0 - gamma.powi(3);
        let row_a = [1.0 / denom, gamma / denom, gamma.powi(2) / denom];
        // Rotate one step for B's row, two for C's row.
        let row_b = [row_a[2], row_a[0], row_a[1]];
        let row_c = [row_a[1], row_a[2], row_a[0]];
        [row_a, row_b, row_c]
    }

    #[test]
    fn self_visit_initialized_on_first_observation() {
        let mut sr = SrPredictor::new();
        sr.observe(ev(0), ev(1));
        assert!(
            (sr.get(ev(0), ev(0)) - 1.0).abs() < 0.01,
            "M[A,A] should be ~1.0 after init + first update; got {}",
            sr.get(ev(0), ev(0))
        );
        // M[B, B] is initialized on observe but never updated (B is not a source yet).
        assert!((sr.get(ev(1), ev(1)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn converges_to_analytical_cyclic_chain() {
        let gamma = 0.5;
        let mut sr = SrPredictor::new()
            .with_gamma(gamma)
            .with_eta_base(0.05)
            .with_warmup(1);
        let expected = analytical_cyclic(gamma);

        // Train on many cycles of A → B → C → A.
        let chain = [ev(0), ev(1), ev(2)];
        for _ in 0..3000 {
            for i in 0..chain.len() {
                let prev = chain[i];
                let next = chain[(i + 1) % chain.len()];
                sr.observe(prev, next);
            }
        }

        let tol = 0.05;
        for (i, &src) in chain.iter().enumerate() {
            for (j, &tgt) in chain.iter().enumerate() {
                let got = sr.get(src, tgt);
                let want = expected[i][j];
                assert!(
                    (got - want).abs() < tol,
                    "M[{}][{}] = {got:.4}, expected {want:.4} (tol {tol})",
                    src.0,
                    tgt.0
                );
            }
        }
    }

    #[test]
    fn predict_returns_topk_excluding_self() {
        let mut sr = SrPredictor::new()
            .with_gamma(0.5)
            .with_eta_base(0.1)
            .with_warmup(1);
        // Train A → B and A → C interleaved with a 4:1 ratio. TD(0) has
        // recency bias when observations arrive in blocks (later blocks
        // overwrite earlier ones), so interleaving is required to get the
        // long-run mixing-distribution ranking we want to test.
        for _ in 0..50 {
            for _ in 0..4 {
                sr.observe(ev(0), ev(1));
            }
            sr.observe(ev(0), ev(2));
        }

        let preds = sr.predict(&[ev(0)], 3);
        assert!(!preds.is_empty());
        // self (A=0) must not appear in predictions
        assert!(preds.iter().all(|p| p.event != ev(0)));
        // B (more frequent) should rank above C
        let rank_b = preds.iter().position(|p| p.event == ev(1));
        let rank_c = preds.iter().position(|p| p.event == ev(2));
        assert!(
            rank_b.is_some() && rank_c.is_some() && rank_b.unwrap() < rank_c.unwrap(),
            "expected B before C, got preds {:?}",
            preds
        );
    }

    #[test]
    fn predict_empty_when_no_data() {
        let sr = SrPredictor::new();
        assert!(sr.predict(&[ev(0)], 5).is_empty());
        assert!(sr.predict(&[], 5).is_empty());
    }

    #[test]
    fn predict_zero_k_is_empty() {
        let mut sr = SrPredictor::new().with_warmup(1);
        sr.observe(ev(0), ev(1));
        assert!(sr.predict(&[ev(0)], 0).is_empty());
    }

    #[test]
    fn confidence_grows_with_observations() {
        let mut sr = SrPredictor::new().with_warmup(10);
        assert!(sr.confidence() < 0.01);
        for _ in 0..5 {
            sr.observe(ev(0), ev(1));
        }
        let mid = sr.confidence();
        assert!(mid > 0.4 && mid < 0.6, "expected ~0.5, got {mid}");
        for _ in 0..20 {
            sr.observe(ev(0), ev(1));
        }
        assert!((sr.confidence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trait_object_dispatch() {
        let mut p: Box<dyn NextTurnPredictor> = Box::new(
            SrPredictor::new()
                .with_warmup(1)
                .with_eta_base(0.1)
                .with_gamma(0.5),
        );
        for _ in 0..30 {
            p.observe(ev(0), ev(1));
        }
        let preds = p.predict(&[ev(0)], 1);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].event, ev(1));
        assert!(p.confidence() > 0.9);
    }

    #[test]
    fn hybrid_uses_markov_when_sr_cold() {
        let mut hybrid =
            HybridPredictor::new(SrPredictor::new().with_warmup(1000), MarkovPredictor::new())
                .with_threshold(0.5);
        // SR is cold (confidence very small), but Markov is fed enough to predict.
        for _ in 0..20 {
            hybrid.observe(ev(0), ev(7));
        }
        let preds = hybrid.predict(&[ev(0)], 1);
        assert!(!preds.is_empty(), "hybrid should fall back to Markov");
        assert_eq!(preds[0].event, ev(7));
    }

    #[test]
    fn hybrid_switches_to_sr_when_warm() {
        let mut hybrid = HybridPredictor::new(
            SrPredictor::new()
                .with_warmup(5)
                .with_eta_base(0.2)
                .with_gamma(0.5),
            MarkovPredictor::new(),
        )
        .with_threshold(0.5);
        for _ in 0..50 {
            hybrid.observe(ev(0), ev(7));
        }
        assert!(hybrid.sr().confidence() >= 0.5);
        let preds = hybrid.predict(&[ev(0)], 1);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].event, ev(7));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut sr = SrPredictor::new()
            .with_gamma(0.7)
            .with_eta_base(0.1)
            .with_warmup(20);
        for _ in 0..30 {
            sr.observe(ev(0), ev(1));
            sr.observe(ev(1), ev(2));
        }
        let path = std::env::temp_dir().join("primd-sr-roundtrip.json");
        sr.save_to_file(&path).unwrap();
        let loaded = SrPredictor::load_from_file(&path).unwrap();

        assert!((loaded.gamma() - 0.7).abs() < 1e-6);
        assert_eq!(loaded.observations(), sr.observations());
        for src in [ev(0), ev(1), ev(2)] {
            for tgt in [ev(0), ev(1), ev(2)] {
                let a = sr.get(src, tgt);
                let b = loaded.get(src, tgt);
                assert!(
                    (a - b).abs() < 1e-5,
                    "M[{}][{}] {} vs {}",
                    src.0,
                    tgt.0,
                    a,
                    b
                );
            }
        }
        let _ = std::fs::remove_file(path);
    }
}
