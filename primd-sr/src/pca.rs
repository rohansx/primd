//! Principal Component Analysis over event centroid signatures.
//!
//! v0.2.6 work: replaces the Achlioptas random projection in
//! [`super::LowRankSrPredictor`] with a PCA learned offline from the
//! corpus signatures. PCA preserves more of the data's variance per
//! dimension than a random projection of the same K, which on the
//! `paraphrase_ab` bench should narrow the projection-quality gap that
//! prevented the K-sweep from fully closing the regression vs Markov.
//!
//! Implementation: power iteration with deflation. For a 256×256
//! covariance matrix and K ≤ 128, this runs in ~50–100 ms at index time;
//! the resulting projection matrix is reused for every signature
//! thereafter at the same per-call cost as the random projection.
//!
//! Mathematical caveat: PCA over N centroids has rank at most N. For the
//! paraphrase bench (N=100 events) we have access to ≤ 100 non-trivial
//! principal components. Asking for K > N gives the first N eigenvectors
//! followed by zero/noise components — still valid but information-bounded
//! by N, not K.

use primd_core::embed::binary::BinarySignature;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::low_rank::SIG_BITS;

const POWER_ITERATIONS: usize = 80;
const NUMERICAL_FLOOR: f32 = 1e-9;
const PCA_SEED: u64 = 0x00C0_FFEE_BABE_F00D;

/// Convert a 256-bit signature to a float vector `[f32; SIG_BITS]`
/// (one entry per bit, 0.0 or 1.0). Caller-owned scratch — no allocation
/// inside the loop.
pub fn signature_to_float(sig: &BinarySignature, out: &mut [f32; SIG_BITS]) {
    for byte_idx in 0..32 {
        let byte = sig.0[byte_idx];
        for bit_in_byte in 0..8 {
            out[byte_idx * 8 + bit_in_byte] = ((byte >> bit_in_byte) & 1) as f32;
        }
    }
}

/// Compute the mean signature (bit-frequency vector) from a list of
/// binary signatures.
pub fn mean_signature(centroids: &[BinarySignature]) -> [f32; SIG_BITS] {
    let n = centroids.len().max(1) as f32;
    let mut mean = [0.0f32; SIG_BITS];
    let mut tmp = [0.0f32; SIG_BITS];
    for sig in centroids {
        signature_to_float(sig, &mut tmp);
        for i in 0..SIG_BITS {
            mean[i] += tmp[i];
        }
    }
    for v in mean.iter_mut() {
        *v /= n;
    }
    mean
}

/// Compute the centered covariance matrix Σ = (1/n) Σ_i (x_i − μ)(x_i − μ)^T
/// over the given centroids. Returns a row-major 256×256 matrix stored as
/// `Box<[[f32; SIG_BITS]; SIG_BITS]>` — ~256 KB.
pub fn covariance_matrix(
    centroids: &[BinarySignature],
    mean: &[f32; SIG_BITS],
) -> Box<[[f32; SIG_BITS]; SIG_BITS]> {
    let n = centroids.len().max(1) as f32;
    let mut cov: Box<[[f32; SIG_BITS]; SIG_BITS]> = Box::new([[0.0f32; SIG_BITS]; SIG_BITS]);
    let mut x = [0.0f32; SIG_BITS];
    let mut centered = [0.0f32; SIG_BITS];
    for sig in centroids {
        signature_to_float(sig, &mut x);
        for i in 0..SIG_BITS {
            centered[i] = x[i] - mean[i];
        }
        for i in 0..SIG_BITS {
            let ci = centered[i];
            if ci.abs() < NUMERICAL_FLOOR {
                continue;
            }
            let row = &mut cov[i];
            for j in 0..SIG_BITS {
                row[j] += ci * centered[j];
            }
        }
    }
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v /= n;
        }
    }
    cov
}

/// Normalize a 256-dim vector in place to unit L2 norm.
fn normalize(v: &mut [f32; SIG_BITS]) {
    let mut norm_sq = 0.0f32;
    for &x in v.iter() {
        norm_sq += x * x;
    }
    let norm = norm_sq.sqrt().max(NUMERICAL_FLOOR);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Matrix-vector product: out = M · v, where M is row-major 256×256.
fn matvec256(m: &[[f32; SIG_BITS]; SIG_BITS], v: &[f32; SIG_BITS]) -> [f32; SIG_BITS] {
    let mut out = [0.0f32; SIG_BITS];
    for i in 0..SIG_BITS {
        let row = &m[i];
        let mut acc = 0.0f32;
        for j in 0..SIG_BITS {
            acc += row[j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// Dot product of two 256-dim vectors.
fn dot256(a: &[f32; SIG_BITS], b: &[f32; SIG_BITS]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..SIG_BITS {
        acc += a[i] * b[i];
    }
    acc
}

/// One power iteration: returns the dominant eigenvector + eigenvalue of `m`.
/// `rng` provides the starting vector; using a deterministic seed gives
/// reproducible projections.
fn power_iterate(
    m: &[[f32; SIG_BITS]; SIG_BITS],
    rng: &mut StdRng,
) -> ([f32; SIG_BITS], f32) {
    let mut v = [0.0f32; SIG_BITS];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    normalize(&mut v);
    for _ in 0..POWER_ITERATIONS {
        v = matvec256(m, &v);
        normalize(&mut v);
    }
    let mv = matvec256(m, &v);
    let lambda = dot256(&v, &mv);
    (v, lambda)
}

/// Deflate a covariance matrix by removing the rank-1 contribution of one
/// eigenpair: `M ← M − λ · v v^T`.
fn deflate(m: &mut [[f32; SIG_BITS]; SIG_BITS], lambda: f32, v: &[f32; SIG_BITS]) {
    for i in 0..SIG_BITS {
        let vi = v[i];
        if vi.abs() < NUMERICAL_FLOOR {
            continue;
        }
        let scaled = lambda * vi;
        let row = &mut m[i];
        for j in 0..SIG_BITS {
            row[j] -= scaled * v[j];
        }
    }
}

/// Compute the top-K principal components of the corpus signatures.
///
/// Returns `(projection, mean)`:
/// - `projection[bit][k]` is the k-th principal component's loading on
///   `bit`, i.e. the projection matrix stored in "feature axis = K" /
///   "input axis = bit" order. Matches the layout used in
///   [`super::LowRankSrPredictor`]'s `projection` field for a drop-in
///   replacement of the Achlioptas matrix.
/// - `mean` is the bit-frequency mean, used to center each input signature
///   before projection.
pub fn compute_pca<const K: usize>(
    centroids: &[BinarySignature],
) -> (Box<[[f32; K]; SIG_BITS]>, [f32; SIG_BITS]) {
    let mean = mean_signature(centroids);
    let mut cov = covariance_matrix(centroids, &mean);

    let mut rng = StdRng::seed_from_u64(PCA_SEED);
    let mut projection: Box<[[f32; K]; SIG_BITS]> = Box::new([[0.0; K]; SIG_BITS]);

    for k in 0..K {
        let (v, lambda) = power_iterate(&cov, &mut rng);
        // Store the eigenvector as the k-th column of the projection.
        for bit in 0..SIG_BITS {
            projection[bit][k] = v[bit];
        }
        // Stop early once eigenvalues collapse to numerical noise.
        if lambda.abs() < NUMERICAL_FLOOR {
            // Leave remaining columns as zeros — projection is
            // information-bounded by the data's rank.
            break;
        }
        deflate(&mut cov, lambda, &v);
    }
    (projection, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig_with_bits_set(bits: &[usize]) -> BinarySignature {
        let mut out = [0u8; 32];
        for &b in bits {
            out[b / 8] |= 1 << (b % 8);
        }
        BinarySignature(out)
    }

    #[test]
    fn signature_to_float_roundtrip() {
        let sig = sig_with_bits_set(&[0, 5, 100, 255]);
        let mut floats = [0.0f32; SIG_BITS];
        signature_to_float(&sig, &mut floats);
        assert_eq!(floats[0], 1.0);
        assert_eq!(floats[1], 0.0);
        assert_eq!(floats[5], 1.0);
        assert_eq!(floats[100], 1.0);
        assert_eq!(floats[255], 1.0);
        assert_eq!(floats[254], 0.0);
    }

    #[test]
    fn mean_matches_bit_frequency() {
        // 3 signatures: bit 0 set in all, bit 1 set in 2/3, bit 100 set in 1/3.
        let sigs = vec![
            sig_with_bits_set(&[0, 1]),
            sig_with_bits_set(&[0, 1]),
            sig_with_bits_set(&[0, 100]),
        ];
        let mean = mean_signature(&sigs);
        assert!((mean[0] - 1.0).abs() < 1e-6);
        assert!((mean[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((mean[100] - 1.0 / 3.0).abs() < 1e-6);
        assert!((mean[200]).abs() < 1e-6);
    }

    #[test]
    fn pca_top_component_aligns_with_dominant_axis() {
        // Build a corpus where one bit dimension has all the variance:
        // bit 7 is "1" in half the signatures and "0" in the other half.
        // The top principal component should be a unit vector along bit 7.
        let mut sigs = Vec::new();
        for i in 0..40 {
            if i % 2 == 0 {
                sigs.push(sig_with_bits_set(&[7]));
            } else {
                sigs.push(sig_with_bits_set(&[]));
            }
        }
        let (proj, _mean) = compute_pca::<8>(&sigs);
        // The top component (k=0) should have a large absolute loading on
        // bit 7, and ~zero on most other bits.
        let load_on_7 = proj[7][0].abs();
        let mut max_other = 0.0f32;
        for bit in 0..SIG_BITS {
            if bit == 7 {
                continue;
            }
            max_other = max_other.max(proj[bit][0].abs());
        }
        assert!(
            load_on_7 > 0.9,
            "expected high loading on bit 7, got {load_on_7}"
        );
        assert!(max_other < 0.1, "unexpected loading elsewhere: {max_other}");
    }

    #[test]
    fn pca_returns_orthogonal_eigenvectors() {
        // Build a corpus with two independent variance axes (bit 5 and bit 50).
        let mut sigs = Vec::new();
        for i in 0..40 {
            let mut bits = vec![];
            if i % 2 == 0 {
                bits.push(5);
            }
            if (i / 2) % 2 == 0 {
                bits.push(50);
            }
            sigs.push(sig_with_bits_set(&bits));
        }
        let (proj, _) = compute_pca::<4>(&sigs);
        // Extract the first two eigenvectors (as columns of `proj`).
        let mut v0 = [0.0f32; SIG_BITS];
        let mut v1 = [0.0f32; SIG_BITS];
        for bit in 0..SIG_BITS {
            v0[bit] = proj[bit][0];
            v1[bit] = proj[bit][1];
        }
        let inner = dot256(&v0, &v1);
        assert!(inner.abs() < 0.05, "eigenvectors not orthogonal: {inner}");
    }

    #[test]
    fn pca_handles_tiny_corpus() {
        // 3 sigs — covariance matrix rank ≤ 2. K=8 should not panic; the
        // surplus components should be zero/noise after rank exhaustion.
        let sigs = vec![
            sig_with_bits_set(&[0]),
            sig_with_bits_set(&[10]),
            sig_with_bits_set(&[20]),
        ];
        let (_proj, _mean) = compute_pca::<8>(&sigs);
        // No assertion; just verify no panic.
    }
}
