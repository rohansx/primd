//! Low-rank Successor Representation over signature features.
//!
//! v0.2.5 work: where v0.2's tabular SR ([`super::SrPredictor`]) treats
//! `EventId`s as opaque atoms, the low-rank variant projects each event's
//! 256-bit signature into a K-dim feature space and maintains an SR matrix
//! `M_low: ℝ^{K×K}` over that feature space.
//!
//! K is a const-generic parameter on [`LowRankSrPredictor`]. The 2026-05-14
//! `paraphrase_ab` bench surfaced that K=32 is information-bottlenecked
//! on a 10-topic-× 10-paraphrase workload (top-1 topic correctness 24.7 %
//! vs Markov's 83.5 %). v0.2.6 explores K=64 and K=128 against the same
//! bench; the public API of the predictor stays stable across K because
//! the trait surface ([`NextTurnPredictor`]) is concrete-type-agnostic.
//!
//! The math (independent of K):
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
//! M_low     += η · δ ⊗ φ_t^T   (K×K outer product; ~1 µs at K=32,
//!                                ~4 µs at K=64, ~16 µs at K=128)
//! ```
//!
//! Init: `M_low = I` so that on the first observation the bootstrap term
//! correctly carries the t=0 self-visit `φ(s_t)` through to the prediction.
//! This is the signature-feature-space analogue of tabular SR's
//! `M[s, s] = 1` initialization. Empirically verified that `M_low = 0`
//! breaks the bootstrap (commit 19677ef).

use std::collections::HashMap;

use primd_core::embed::binary::BinarySignature;
use primd_core::predict::{EventId, NextTurnPredictor, Prediction};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Bit-width of the signature space. Independent of the predictor's K.
pub const SIG_BITS: usize = 256;

/// Default value for the const-generic K parameter — preserves v0.2.5's
/// "K=32 = one cache line × 4 SIMD lanes of f64" choice for the existing
/// API surface. v0.2.6 experimentally sweeps K and may change this.
pub const DEFAULT_K: usize = 32;

/// Type alias for the default-K variant. `LowRankSr` writes shorter than
/// `LowRankSrPredictor<32>` and reads more clearly at call sites.
pub type LowRankSr = LowRankSrPredictor<DEFAULT_K>;

/// Default seed for the random projection. Fixed so two LowRankSrPredictor
/// instances built from the same corpus produce identical feature spaces
/// (deterministic A/B comparisons).
pub const DEFAULT_PROJECTION_SEED: u64 = 0x1234_ABCD_5678_EF01;

pub const DEFAULT_GAMMA: f32 = 0.9;
pub const DEFAULT_ETA_BASE: f32 = 0.05;
pub const DEFAULT_WARMUP_OBSERVATIONS: u64 = 50;

/// Identity is the SR-correct default: the t=0 self-visit term (visiting
/// `s_0` once at the start) is carried by `M_low·φ(s_0) = φ(s_0)` which
/// bootstraps subsequent TD updates correctly. We considered switching to
/// Zero based on the `paraphrase_ab` finding, but verified Zero breaks the
/// SR math: with `M_low = 0` the bootstrap term `M·φ_next` collapses to 0
/// for every update, so the (prev → next) association is never learned —
/// the TD update degenerates to accumulating `φ_prev ⊗ φ_prev` rank-1
/// outer products and predictions stay anchored to the current event's
/// features forever. The right v0.2.6 fixes are K (32 → 64/128) and
/// projection quality (random → PCA), not init.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MLowInit {
    /// `M_low = I`. Bootstrap-correct math; the v0.2 default.
    #[default]
    Identity,
    /// `M_low = 0`. Research knob — see [`MLowInit`] docs for why this is
    /// not a working default.
    Zero,
}

/// Low-rank Successor Representation predictor over signature features,
/// parameterized by the feature-space dimensionality `K`.
///
/// Common choices:
/// - `K = 32` — one cache line × 4 SIMD lanes of f64; the v0.2.5 default
/// - `K = 64` — better information preservation at 4× TD update cost
/// - `K = 128` — high-fidelity feature space; ~64 KB per `M_low`
pub struct LowRankSrPredictor<const K: usize> {
    projection: Box<[[f32; K]; SIG_BITS]>,
    /// Mean signature used for centering each input before projection.
    /// Zero for random-projection predictors (signatures used uncentered);
    /// populated for PCA-projection predictors so centering matches the
    /// projection's coordinate system.
    mean: Box<[f32; SIG_BITS]>,
    m_low: Box<[[f32; K]; K]>,
    event_features: HashMap<EventId, [f32; K]>,
    gamma: f32,
    eta_base: f32,
    t: u64,
    warmup: u64,
}

impl<const K: usize> LowRankSrPredictor<K> {
    /// Build a predictor seeded from the provided event signatures.
    pub fn new(event_centroids: &HashMap<EventId, BinarySignature>) -> Self {
        Self::with_seed_and_init(event_centroids, DEFAULT_PROJECTION_SEED, MLowInit::default())
    }

    /// Same as [`Self::new`] with an explicit projection seed.
    pub fn with_seed(
        event_centroids: &HashMap<EventId, BinarySignature>,
        seed: u64,
    ) -> Self {
        Self::with_seed_and_init(event_centroids, seed, MLowInit::default())
    }

    /// Convenience over [`Self::with_seed_and_init`] for the default seed case.
    pub fn with_init(event_centroids: &HashMap<EventId, BinarySignature>, init: MLowInit) -> Self {
        Self::with_seed_and_init(event_centroids, DEFAULT_PROJECTION_SEED, init)
    }

    /// Full constructor exposing both the projection seed and the `M_low`
    /// initialization. See [`MLowInit`].
    pub fn with_seed_and_init(
        event_centroids: &HashMap<EventId, BinarySignature>,
        seed: u64,
        init: MLowInit,
    ) -> Self {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);

        // Achlioptas-style random projection: each entry is +1/√K or
        // -1/√K with equal probability. Preserves dot products in
        // expectation and is bit-shift-cheap to evaluate (the bit-vector
        // dot product becomes a signed sum).
        let scale = 1.0 / (K as f32).sqrt();
        let mut projection: Box<[[f32; K]; SIG_BITS]> = Box::new([[0.0; K]; SIG_BITS]);
        for bit in 0..SIG_BITS {
            for col in 0..K {
                projection[bit][col] = if rng.random_bool(0.5) { scale } else { -scale };
            }
        }

        // Random projection uses uncentered signatures — the mean stays at 0.
        let mean: Box<[f32; SIG_BITS]> = Box::new([0.0f32; SIG_BITS]);

        Self::finalize_construction(event_centroids, projection, mean, init)
    }

    /// PCA-projection constructor. Computes the top-K principal components
    /// of the corpus signatures at index time and uses them as the
    /// feature-extraction matrix `W`. Better signal-to-noise than the
    /// random Achlioptas projection at the same K, at a one-time
    /// O(K · iter · 256²) construction cost (~50–100 ms for typical
    /// voice corpora).
    pub fn with_pca(event_centroids: &HashMap<EventId, BinarySignature>) -> Self {
        Self::with_pca_and_init(event_centroids, MLowInit::default())
    }

    /// PCA constructor exposing the `M_low` init.
    pub fn with_pca_and_init(
        event_centroids: &HashMap<EventId, BinarySignature>,
        init: MLowInit,
    ) -> Self {
        let centroids_vec: Vec<BinarySignature> =
            event_centroids.values().copied().collect();
        let (projection, mean_arr) = crate::pca::compute_pca::<K>(&centroids_vec);
        let mean: Box<[f32; SIG_BITS]> = Box::new(mean_arr);
        Self::finalize_construction(event_centroids, projection, mean, init)
    }

    /// Internal shared finalization step — builds `m_low`, projects every
    /// event's centroid into feature space, and assembles the struct.
    fn finalize_construction(
        event_centroids: &HashMap<EventId, BinarySignature>,
        projection: Box<[[f32; K]; SIG_BITS]>,
        mean: Box<[f32; SIG_BITS]>,
        init: MLowInit,
    ) -> Self {
        let mut m_low: Box<[[f32; K]; K]> = Box::new([[0.0; K]; K]);
        if matches!(init, MLowInit::Identity) {
            for k in 0..K {
                m_low[k][k] = 1.0;
            }
        }

        let mut event_features: HashMap<EventId, [f32; K]> =
            HashMap::with_capacity(event_centroids.len());
        for (&ev, sig) in event_centroids {
            event_features.insert(ev, project_centered::<K>(&projection, &mean, sig));
        }

        Self {
            projection,
            mean,
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

    pub fn set_event_centroid(&mut self, event: EventId, sig: &BinarySignature) {
        let feat = project_centered::<K>(&self.projection, &self.mean, sig);
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

    pub fn k(&self) -> usize {
        K
    }

    pub fn m_low(&self, i: usize, j: usize) -> f32 {
        self.m_low[i][j]
    }

    fn eta(&self) -> f32 {
        self.eta_base / (1.0 + 0.01 * self.t as f32)
    }

    fn feature_of(&self, event: EventId) -> Option<&[f32; K]> {
        self.event_features.get(&event)
    }
}

/// Apply the random projection to a signature: ψ(s) = W^T · bits(s).
/// Used when the predictor was built with an uncentered random projection.
#[allow(dead_code)]
fn project<const K: usize>(
    projection: &[[f32; K]; SIG_BITS],
    sig: &BinarySignature,
) -> [f32; K] {
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

/// Center a signature by subtracting `mean` and project to feature space.
/// For random-projection predictors `mean` is the zero vector, so this is
/// equivalent to [`project`] but with an extra subtraction per bit. The
/// branch on `mean == 0` is hot-path-free because the compiler can hoist
/// the subtraction into the loop.
fn project_centered<const K: usize>(
    projection: &[[f32; K]; SIG_BITS],
    mean: &[f32; SIG_BITS],
    sig: &BinarySignature,
) -> [f32; K] {
    let mut out = [0.0f32; K];
    for bit in 0..SIG_BITS {
        let byte = sig.0[bit / 8];
        let bit_set = ((byte >> (bit % 8)) & 1) as f32;
        let centered = bit_set - mean[bit];
        if centered == 0.0 {
            continue;
        }
        let row = &projection[bit];
        for k in 0..K {
            out[k] += row[k] * centered;
        }
    }
    out
}

/// Matrix-vector product M · v with M stored row-major.
fn matvec<const K: usize>(m: &[[f32; K]; K], v: &[f32; K]) -> [f32; K] {
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
fn dot<const K: usize>(a: &[f32; K], b: &[f32; K]) -> f32 {
    let mut acc = 0.0f32;
    for k in 0..K {
        acc += a[k] * b[k];
    }
    acc
}

impl<const K: usize> NextTurnPredictor for LowRankSrPredictor<K> {
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
        let predicted = matvec::<K>(&self.m_low, &phi_last);

        let mut scored: Vec<(EventId, f32)> = self
            .event_features
            .iter()
            .filter(|&(&ev, _)| ev != last)
            .map(|(&ev, feat)| (ev, dot::<K>(feat, &predicted).max(0.0)))
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

        let prediction = matvec::<K>(&self.m_low, &phi_prev);
        let bootstrap = matvec::<K>(&self.m_low, &phi_next);

        let mut delta = [0.0f32; K];
        for k in 0..K {
            delta[k] = phi_prev[k] + gamma * bootstrap[k] - prediction[k];
        }

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
        m.insert(ev(0), sig_with_bits_set(&[0, 1, 2, 3, 4, 5, 6, 7]));
        m.insert(ev(1), sig_with_bits_set(&[64, 65, 66, 67, 68, 69, 70, 71]));
        m.insert(ev(2), sig_with_bits_set(&[128, 129, 130, 131, 132, 133, 134, 135]));
        m
    }

    #[test]
    fn deterministic_projection_from_seed() {
        let centroids = three_event_centroids();
        let a: LowRankSrPredictor<32> = LowRankSrPredictor::with_seed(&centroids, 42);
        let b: LowRankSrPredictor<32> = LowRankSrPredictor::with_seed(&centroids, 42);
        for ev in [ev(0), ev(1), ev(2)] {
            let fa = a.feature_of(ev).unwrap();
            let fb = b.feature_of(ev).unwrap();
            for k in 0..32 {
                assert!((fa[k] - fb[k]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn different_seeds_give_different_projections() {
        let centroids = three_event_centroids();
        let a: LowRankSrPredictor<32> = LowRankSrPredictor::with_seed(&centroids, 42);
        let b: LowRankSrPredictor<32> = LowRankSrPredictor::with_seed(&centroids, 43);
        let fa = a.feature_of(ev(0)).unwrap();
        let fb = b.feature_of(ev(0)).unwrap();
        let any_diff = (0..32).any(|k| (fa[k] - fb[k]).abs() > 1e-6);
        assert!(any_diff);
    }

    #[test]
    fn predict_returns_empty_when_no_observations() {
        let centroids = three_event_centroids();
        let sr: LowRankSr = LowRankSr::new(&centroids);
        let preds = sr.predict(&[ev(0)], 2);
        assert!(preds.len() <= 2);
    }

    #[test]
    fn observe_changes_m_low() {
        let centroids = three_event_centroids();
        let mut sr: LowRankSr = LowRankSr::new(&centroids);
        let initial = (0..32).map(|i| sr.m_low(i, i)).sum::<f32>();
        for _ in 0..50 {
            sr.observe(ev(0), ev(1));
        }
        let updated = (0..32).map(|i| sr.m_low(i, i)).sum::<f32>();
        assert!((initial - updated).abs() > 0.01);
    }

    #[test]
    fn confidence_grows_with_observations() {
        let centroids = three_event_centroids();
        let mut sr: LowRankSr = LowRankSr::new(&centroids).with_warmup(10);
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
    fn trait_object_safe_at_k32() {
        let centroids = three_event_centroids();
        let mut p: Box<dyn NextTurnPredictor> = Box::new(
            LowRankSr::new(&centroids)
                .with_warmup(5)
                .with_eta_base(0.1)
                .with_gamma(0.5),
        );
        for _ in 0..20 {
            p.observe(ev(0), ev(1));
        }
        let preds = p.predict(&[ev(0)], 3);
        assert!(!preds.is_empty());
        let rank1 = preds.iter().position(|p| p.event == ev(1));
        let rank2 = preds.iter().position(|p| p.event == ev(2));
        assert!(
            rank1.is_some() && (rank2.is_none() || rank1.unwrap() < rank2.unwrap()),
            "ev(1) should rank above ev(2) after training 0->1: {preds:?}"
        );
    }

    #[test]
    fn trait_object_safe_at_k64() {
        let centroids = three_event_centroids();
        let mut p: Box<dyn NextTurnPredictor> = Box::new(
            LowRankSrPredictor::<64>::new(&centroids)
                .with_warmup(5)
                .with_eta_base(0.1)
                .with_gamma(0.5),
        );
        for _ in 0..20 {
            p.observe(ev(0), ev(1));
        }
        let preds = p.predict(&[ev(0)], 3);
        assert!(!preds.is_empty());
    }

    #[test]
    fn unknown_events_are_no_ops() {
        let centroids = three_event_centroids();
        let mut sr: LowRankSr = LowRankSr::new(&centroids);
        let before_diag: f32 = (0..32).map(|i| sr.m_low(i, i)).sum();
        sr.observe(ev(0), ev(99));
        sr.observe(ev(99), ev(1));
        let after_diag: f32 = (0..32).map(|i| sr.m_low(i, i)).sum();
        assert!((before_diag - after_diag).abs() < 1e-6);
        assert_eq!(sr.observations(), 0);
    }
}
