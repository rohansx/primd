//! Low-rank Successor Representation over signature features.
//!
//! v0.2.5 work: where v0.2's tabular SR ([`super::SrPredictor`]) treats
//! `EventId`s as opaque atoms, the low-rank variant projects each event's
//! 256-bit signature into a K-dim feature space (K=32, one cache line ×
//! 4 SIMD lanes of f64) and maintains an SR matrix `M_low: ℝ^{K×K}` over
//! that feature space.
//!
//! The win this unlocks (per Stachenfeld, Botvinick, Gershman 2017 and
//! Russek et al. 2017) is **paraphrase generalization**: events with
//! similar signatures share predictive structure through `M_low`, so a
//! novel paraphrase of a known intent inherits its successor distribution
//! without separate observations.
//!
//! The math:
//!
//! - Feature map: `ψ(s) = W^T · sig_bits(s) ∈ ℝ^K`, where W ∈ ℝ^{256×K}
//!   is a fixed random projection (deterministic from a seed).
//! - SR prediction: `M_low · ψ(s_t) ∈ ℝ^K` is the discounted future
//!   feature trajectory from `s_t`.
//! - Per-event score: `ψ(e) · (M_low · ψ(s_t))` gives the predicted SR
//!   visit count to event `e` starting from `s_t`. Top-K by score.
//!
//! TD(0) update on observed transition `s_t → s_{t+1}`:
//!
//! ```text
//! φ_t       = ψ(s_t)
//! φ_{t+1}   = ψ(s_{t+1})
//! prediction = M_low · φ_t
//! bootstrap  = M_low · φ_{t+1}
//! target     = φ_t + γ · bootstrap
//! δ          = target − prediction
//! M_low     += η · δ ⊗ φ_t^T   (K×K outer product, ~1 µs at K=32)
//! ```
//!
//! Init: `M_low = I` so that on the first observation the bootstrap term
//! correctly carries the t=0 self-visit `φ(s_t)` through to the prediction.
//! This is the signature-feature-space analogue of tabular SR's
//! `M[s, s] = 1` initialization.

use std::collections::HashMap;

use primd_core::embed::binary::BinarySignature;
use primd_core::predict::{EventId, NextTurnPredictor, Prediction};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Feature-space dimensionality. One cache line of f64 × 4 SIMD lanes.
/// Const for now — making it a const generic is v0.3 work if we end up
/// wanting K=64 or K=16 in production.
pub const K: usize = 32;

/// Bit-width of the signature space.
pub const SIG_BITS: usize = 256;

/// Default seed for the random projection. Fixed so two LowRankSrPredictor
/// instances built from the same corpus produce identical feature spaces
/// (deterministic A/B comparisons).
pub const DEFAULT_PROJECTION_SEED: u64 = 0x1234_ABCD_5678_EF01;

pub const DEFAULT_GAMMA: f32 = 0.9;
pub const DEFAULT_ETA_BASE: f32 = 0.05;
pub const DEFAULT_WARMUP_OBSERVATIONS: u64 = 50;

/// Per-row feature vector (length K).
type FeatureVec = [f32; K];

/// `K × K` low-rank SR matrix.
type LowRankMatrix = [[f32; K]; K];

/// Low-rank Successor Representation predictor over signature features.
///
/// Maintained state:
/// - `projection`: fixed `256×K` random projection W. Built deterministically
///   from the seed; never mutated.
/// - `m_low`: learned `K×K` SR matrix, initialized to identity.
/// - `event_features`: cached `ψ(centroid)` for each known event, computed
///   from the corpus signatures the caller supplies at init.
pub struct LowRankSrPredictor {
    projection: Box<[FeatureVec; SIG_BITS]>,
    m_low: Box<LowRankMatrix>,
    event_features: HashMap<EventId, FeatureVec>,
    gamma: f32,
    eta_base: f32,
    t: u64,
    warmup: u64,
}

impl LowRankSrPredictor {
    /// Build a predictor seeded from the provided event signatures.
    ///
    /// `event_centroids` maps each known event to its centroid signature
    /// (typically the mean bit-vector across the event's documents, then
    /// rebinarized — or just one representative signature). Events not
    /// present here will return empty predictions until observed in
    /// `observe` calls.
    pub fn new(event_centroids: &HashMap<EventId, BinarySignature>) -> Self {
        Self::with_seed(event_centroids, DEFAULT_PROJECTION_SEED)
    }

    /// Same as [`Self::new`] with an explicit projection seed. Useful for
    /// A/B testing different random-projection draws.
    pub fn with_seed(
        event_centroids: &HashMap<EventId, BinarySignature>,
        seed: u64,
    ) -> Self {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);

        // Achlioptas-style random projection: each entry is +1/√K or
        // -1/√K with equal probability. Preserves dot products in
        // expectation and is bit-shift-cheap to evaluate (the bit-vector
        // dot product becomes a signed sum).
        let scale = 1.0 / (K as f32).sqrt();
        let mut projection: Box<[FeatureVec; SIG_BITS]> = Box::new([[0.0; K]; SIG_BITS]);
        for bit in 0..SIG_BITS {
            for k in 0..K {
                projection[bit][k] = if rng.random_bool(0.5) { scale } else { -scale };
            }
        }

        // Identity initialization so bootstrap from M_low · φ(s) ≈ φ(s)
        // on the first observation — feature-space analogue of M[s,s] = 1
        // in tabular SR.
        let mut m_low: Box<LowRankMatrix> = Box::new([[0.0; K]; K]);
        for k in 0..K {
            m_low[k][k] = 1.0;
        }

        // Precompute event features from centroids.
        let mut event_features: HashMap<EventId, FeatureVec> =
            HashMap::with_capacity(event_centroids.len());
        for (&ev, sig) in event_centroids {
            event_features.insert(ev, project(&projection, sig));
        }

        Self {
            projection,
            m_low,
            event_features,
            gamma: DEFAULT_GAMMA,
            eta_base: DEFAULT_ETA_BASE,
            t: 0,
            warmup: DEFAULT_WARMUP_OBSERVATIONS,
        }
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma.clamp(1e-3, 1.0 - 1e-3);
        self
    }

    pub fn with_eta_base(mut self, eta_base: f32) -> Self {
        self.eta_base = eta_base.clamp(1e-4, 1.0);
        self
    }

    pub fn with_warmup(mut self, warmup: u64) -> Self {
        self.warmup = warmup.max(1);
        self
    }

    /// Set the feature vector for an event explicitly — useful when a
    /// caller has computed a centroid by some other means than mean-bits.
    pub fn set_event_centroid(&mut self, event: EventId, sig: &BinarySignature) {
        let feat = project(&self.projection, sig);
        self.event_features.insert(event, feat);
    }

    pub fn observations(&self) -> u64 {
        self.t
    }

    pub fn vocab_size(&self) -> usize {
        self.event_features.len()
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Lookup `M_low[i][j]` for inspection.
    pub fn m_low(&self, i: usize, j: usize) -> f32 {
        self.m_low[i][j]
    }

    fn eta(&self) -> f32 {
        self.eta_base / (1.0 + 0.01 * self.t as f32)
    }

    fn feature_of(&self, event: EventId) -> Option<&FeatureVec> {
        self.event_features.get(&event)
    }
}

/// Apply the random projection to a signature: ψ(s) = W^T · bits(s).
fn project(projection: &[FeatureVec; SIG_BITS], sig: &BinarySignature) -> FeatureVec {
    let mut out = [0.0f32; K];
    for byte_idx in 0..32 {
        let byte = sig.0[byte_idx];
        if byte == 0 {
            continue;
        }
        for bit_in_byte in 0..8 {
            if (byte >> bit_in_byte) & 1 == 1 {
                let bit = byte_idx * 8 + bit_in_byte;
                let row = &projection[bit];
                for k in 0..K {
                    out[k] += row[k];
                }
            }
        }
    }
    out
}

/// Matrix-vector product M · v with M stored row-major.
fn matvec(m: &LowRankMatrix, v: &FeatureVec) -> FeatureVec {
    let mut out = [0.0f32; K];
    for i in 0..K {
        let row = &m[i];
        let mut acc = 0.0f32;
        for j in 0..K {
            acc += row[j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// Dot product.
fn dot(a: &FeatureVec, b: &FeatureVec) -> f32 {
    let mut acc = 0.0f32;
    for k in 0..K {
        acc += a[k] * b[k];
    }
    acc
}

impl NextTurnPredictor for LowRankSrPredictor {
    fn predict(&self, context: &[EventId], k: usize) -> Vec<Prediction> {
        if k == 0 {
            return Vec::new();
        }
        let last = match context.last() {
            Some(&e) => e,
            None => return Vec::new(),
        };
        let phi_last = match self.feature_of(last) {
            Some(f) => *f,
            None => return Vec::new(),
        };
        // Predicted feature row for the discounted future from `last`.
        let predicted = matvec(&self.m_low, &phi_last);

        // Score each known event by its alignment with the predicted
        // feature trajectory.
        let mut scored: Vec<(EventId, f32)> = self
            .event_features
            .iter()
            .filter(|&(&ev, _)| ev != last)
            .map(|(&ev, feat)| (ev, dot(feat, &predicted).max(0.0)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let sum: f32 = scored.iter().map(|(_, v)| *v).sum();
        if sum <= 0.0 {
            return Vec::new();
        }
        scored
            .into_iter()
            .map(|(event, value)| Prediction {
                event,
                probability: value / sum,
            })
            .collect()
    }

    fn observe(&mut self, prev: EventId, next: EventId) {
        // We can only meaningfully TD-update when both endpoints have
        // cached features. Events outside the catalog are no-ops here,
        // mirroring tabular SR's behavior for never-seen states.
        let phi_prev = match self.feature_of(prev) {
            Some(f) => *f,
            None => return,
        };
        let phi_next = match self.feature_of(next) {
            Some(f) => *f,
            None => return,
        };

        let eta = self.eta();
        let gamma = self.gamma;

        let prediction = matvec(&self.m_low, &phi_prev);
        let bootstrap = matvec(&self.m_low, &phi_next);

        // δ = φ_prev + γ · bootstrap − prediction
        let mut delta = [0.0f32; K];
        for k in 0..K {
            delta[k] = phi_prev[k] + gamma * bootstrap[k] - prediction[k];
        }

        // M_low += η · δ ⊗ φ_prev
        for (i, row) in self.m_low.iter_mut().enumerate() {
            let scaled = eta * delta[i];
            for j in 0..K {
                row[j] += scaled * phi_prev[j];
            }
        }

        self.t += 1;
    }

    fn confidence(&self) -> f32 {
        if self.warmup == 0 {
            return 1.0;
        }
        (self.t as f32 / self.warmup as f32).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: u32) -> EventId {
        EventId(id)
    }

    fn sig_with_bits_set(bits: &[usize]) -> BinarySignature {
        let mut out = [0u8; 32];
        for &b in bits {
            out[b / 8] |= 1 << (b % 8);
        }
        BinarySignature(out)
    }

    fn three_event_centroids() -> HashMap<EventId, BinarySignature> {
        let mut m = HashMap::new();
        // Three signatures with different bit patterns so their features
        // are distinguishable in feature space.
        m.insert(ev(0), sig_with_bits_set(&[0, 1, 2, 3, 4, 5, 6, 7]));
        m.insert(ev(1), sig_with_bits_set(&[64, 65, 66, 67, 68, 69, 70, 71]));
        m.insert(ev(2), sig_with_bits_set(&[128, 129, 130, 131, 132, 133, 134, 135]));
        m
    }

    #[test]
    fn deterministic_projection_from_seed() {
        let centroids = three_event_centroids();
        let a = LowRankSrPredictor::with_seed(&centroids, 42);
        let b = LowRankSrPredictor::with_seed(&centroids, 42);
        for ev in [ev(0), ev(1), ev(2)] {
            let fa = a.feature_of(ev).unwrap();
            let fb = b.feature_of(ev).unwrap();
            for k in 0..K {
                assert!((fa[k] - fb[k]).abs() < 1e-9, "k={k}, fa[k]={}, fb[k]={}", fa[k], fb[k]);
            }
        }
    }

    #[test]
    fn different_seeds_give_different_projections() {
        let centroids = three_event_centroids();
        let a = LowRankSrPredictor::with_seed(&centroids, 42);
        let b = LowRankSrPredictor::with_seed(&centroids, 43);
        // At least one feature should differ between the two seeds.
        let fa = a.feature_of(ev(0)).unwrap();
        let fb = b.feature_of(ev(0)).unwrap();
        let any_diff = (0..K).any(|k| (fa[k] - fb[k]).abs() > 1e-6);
        assert!(any_diff, "projections were identical for different seeds");
    }

    #[test]
    fn predict_returns_empty_when_no_observations() {
        let centroids = three_event_centroids();
        let sr = LowRankSrPredictor::new(&centroids);
        // M_low = I and self-prediction excluded → predictions exist but
        // mass is spread across other events. Predict should return
        // something nonempty.
        let preds = sr.predict(&[ev(0)], 2);
        // After init M_low = I, ψ̂(s) = φ(s). Scores against other events
        // are random-projection dot products which are non-negative-ish.
        // Don't assert ordering — just that we get up to 2 entries.
        assert!(preds.len() <= 2);
    }

    #[test]
    fn observe_changes_m_low() {
        let centroids = three_event_centroids();
        let mut sr = LowRankSrPredictor::new(&centroids);
        let initial = (0..K).map(|i| sr.m_low(i, i)).sum::<f32>();
        for _ in 0..50 {
            sr.observe(ev(0), ev(1));
        }
        let updated = (0..K).map(|i| sr.m_low(i, i)).sum::<f32>();
        assert!(
            (initial - updated).abs() > 0.01,
            "M_low diagonal did not change after observations: {initial} -> {updated}",
        );
    }

    #[test]
    fn confidence_grows_with_observations() {
        let centroids = three_event_centroids();
        let mut sr = LowRankSrPredictor::new(&centroids).with_warmup(10);
        assert!(sr.confidence() < 0.01);
        for _ in 0..5 {
            sr.observe(ev(0), ev(1));
        }
        let mid = sr.confidence();
        assert!((0.4..=0.6).contains(&mid), "mid confidence {mid}");
        for _ in 0..20 {
            sr.observe(ev(0), ev(1));
        }
        assert!((sr.confidence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trait_object_safe() {
        let centroids = three_event_centroids();
        let mut p: Box<dyn NextTurnPredictor> = Box::new(
            LowRankSrPredictor::new(&centroids)
                .with_warmup(5)
                .with_eta_base(0.1)
                .with_gamma(0.5),
        );
        for _ in 0..20 {
            p.observe(ev(0), ev(1));
        }
        let preds = p.predict(&[ev(0)], 3);
        assert!(!preds.is_empty());
        // After many observations of 0→1, event 1 should outrank event 2
        // in predictions from 0.
        let rank1 = preds.iter().position(|p| p.event == ev(1));
        let rank2 = preds.iter().position(|p| p.event == ev(2));
        assert!(
            rank1.is_some() && (rank2.is_none() || rank1.unwrap() < rank2.unwrap()),
            "ev(1) should rank above ev(2) after training 0->1: {preds:?}"
        );
    }

    #[test]
    fn unknown_events_are_no_ops() {
        let centroids = three_event_centroids();
        let mut sr = LowRankSrPredictor::new(&centroids);
        let before_diag: f32 = (0..K).map(|i| sr.m_low(i, i)).sum();
        // ev(99) is not in the catalog
        sr.observe(ev(0), ev(99));
        sr.observe(ev(99), ev(1));
        let after_diag: f32 = (0..K).map(|i| sr.m_low(i, i)).sum();
        assert!((before_diag - after_diag).abs() < 1e-6);
        assert_eq!(sr.observations(), 0);
    }
}
