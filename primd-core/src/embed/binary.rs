//! Binary quantization: float32 embeddings → 256-bit binary signatures.
//!
//! Pipeline: dense embedding → PCA projection to 256 dims → sign-bit quantization → `[u8; 32]`.

use std::path::Path;

use crate::{PrimdError, Result};

/// A 256-bit binary signature (32 bytes). Core type used throughout primd.
///
/// Serialized as a JSON array of 32 byte integers via the manual
/// [`serde::Serialize`]/[`serde::Deserialize`] impls below. Used by the
/// `primd-sr` LowRankSrPredictor's PCA path (centroids serialized in the
/// `pca` module) and by any caller wanting to round-trip signatures
/// through JSON config / state files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BinarySignature(pub [u8; 32]);

impl serde::Serialize for BinarySignature {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tup = ser.serialize_tuple(32)?;
        for byte in &self.0 {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

impl<'de> serde::Deserialize<'de> for BinarySignature {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let arr: [u8; 32] = serde::Deserialize::deserialize(de)?;
        Ok(BinarySignature(arr))
    }
}

impl BinarySignature {
    pub const BITS: usize = 256;
    pub const BYTES: usize = 32;

    /// Compute Hamming distance to another signature (scalar reference implementation).
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..4 {
            let offset = i * 8;
            let a = u64::from_ne_bytes(self.0[offset..offset + 8].try_into().unwrap());
            let b = u64::from_ne_bytes(other.0[offset..offset + 8].try_into().unwrap());
            dist += (a ^ b).count_ones();
        }
        dist
    }
}

/// Projects dense float32 embeddings to 256 dimensions via PCA, then sign-bit quantizes.
pub struct PcaProjector {
    /// Flat row-major matrix: [256 × source_dim]. Row i contains the i-th principal component.
    matrix: Vec<f32>,
    source_dim: usize,
}

impl PcaProjector {
    /// Create from a precomputed PCA matrix.
    ///
    /// `matrix` is row-major `[256 × source_dim]`: 256 rows of `source_dim` floats each.
    pub fn new(matrix: Vec<f32>, source_dim: usize) -> Result<Self> {
        let expected = 256 * source_dim;
        if matrix.len() != expected {
            return Err(PrimdError::InvalidDimension {
                expected,
                got: matrix.len(),
            });
        }
        Ok(Self { matrix, source_dim })
    }

    /// Load PCA matrix from a binary file.
    ///
    /// Format: first 4 bytes = `source_dim` as little-endian u32,
    /// then `256 × source_dim` little-endian f32 values.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(|_| PrimdError::PcaMatrixMissing)?;
        if data.len() < 4 {
            return Err(PrimdError::PcaMatrixMissing);
        }
        let source_dim = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let expected_bytes = 4 + 256 * source_dim * 4;
        if data.len() != expected_bytes {
            return Err(PrimdError::InvalidDimension {
                expected: expected_bytes,
                got: data.len(),
            });
        }
        let matrix: Vec<f32> = data[4..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::new(matrix, source_dim)
    }

    /// Write PCA matrix to a binary file.
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let mut buf = Vec::with_capacity(4 + self.matrix.len() * 4);
        buf.extend_from_slice(&(self.source_dim as u32).to_le_bytes());
        for &val in &self.matrix {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        std::fs::write(path, &buf)?;
        Ok(())
    }

    /// Project a dense embedding to 256 float32 components.
    pub fn project(&self, embedding: &[f32]) -> Result<[f32; 256]> {
        if embedding.len() != self.source_dim {
            return Err(PrimdError::InvalidDimension {
                expected: self.source_dim,
                got: embedding.len(),
            });
        }
        let mut projected = [0.0f32; 256];
        for (i, out) in projected.iter_mut().enumerate() {
            let row = &self.matrix[i * self.source_dim..(i + 1) * self.source_dim];
            *out = row.iter().zip(embedding.iter()).map(|(a, b)| a * b).sum();
        }
        Ok(projected)
    }

    /// Quantize a dense embedding to a 256-bit binary signature.
    pub fn quantize(&self, embedding: &[f32]) -> Result<BinarySignature> {
        let projected = self.project(embedding)?;
        Ok(sign_bit_quantize(&projected))
    }

    /// Batch quantize multiple embeddings (contiguous, each `source_dim` floats).
    pub fn quantize_batch(&self, embeddings: &[f32]) -> Result<Vec<BinarySignature>> {
        if !embeddings.len().is_multiple_of(self.source_dim) {
            return Err(PrimdError::InvalidDimension {
                expected: self.source_dim,
                got: embeddings.len() % self.source_dim,
            });
        }
        embeddings
            .chunks_exact(self.source_dim)
            .map(|emb| self.quantize(emb))
            .collect()
    }

    pub fn source_dim(&self) -> usize {
        self.source_dim
    }
}

/// Build a deterministic random projection from `source_dim` to 256 dimensions.
///
/// Uses Achlioptas-style sparse signs (±1/√source_dim) seeded by `seed`. The
/// resulting `PcaProjector` approximately preserves L2 distances and inner
/// products by the Johnson-Lindenstrauss lemma. Save just the seed alongside
/// the index — query-time reconstruction is exact.
pub fn random_projection(seed: u64, source_dim: usize) -> PcaProjector {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let scale = 1.0 / (source_dim as f32).sqrt();
    let mut matrix = vec![0.0f32; 256 * source_dim];
    for i in 0..256 {
        for j in 0..source_dim {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            i.hash(&mut h);
            j.hash(&mut h);
            let bit = (h.finish() & 1) == 0;
            matrix[i * source_dim + j] = if bit { scale } else { -scale };
        }
    }
    PcaProjector::new(matrix, source_dim).expect("matrix dim is correct by construction")
}

/// Sign-bit quantization: for each of 256 components, bit = 1 if value >= 0.
pub fn sign_bit_quantize(projected: &[f32; 256]) -> BinarySignature {
    let mut sig = [0u8; 32];
    for (i, &val) in projected.iter().enumerate() {
        if val >= 0.0 {
            sig[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    BinarySignature(sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_projector(dim: usize) -> PcaProjector {
        // Identity-like: first 256 components map 1:1 (if dim >= 256)
        let mut matrix = vec![0.0f32; 256 * dim];
        for i in 0..256.min(dim) {
            matrix[i * dim + i] = 1.0;
        }
        PcaProjector::new(matrix, dim).unwrap()
    }

    #[test]
    fn hamming_identical() {
        let sig = BinarySignature([0xAA; 32]);
        assert_eq!(sig.hamming_distance(&sig), 0);
    }

    #[test]
    fn hamming_complementary() {
        let a = BinarySignature([0x00; 32]);
        let b = BinarySignature([0xFF; 32]);
        assert_eq!(a.hamming_distance(&b), 256);
    }

    #[test]
    fn hamming_known_pattern() {
        let a = BinarySignature([0x00; 32]);
        let mut b_bytes = [0x00u8; 32];
        b_bytes[0] = 0x01; // 1 bit different
        let b = BinarySignature(b_bytes);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn sign_bit_all_positive() {
        let projected = [1.0f32; 256];
        let sig = sign_bit_quantize(&projected);
        assert_eq!(sig, BinarySignature([0xFF; 32]));
    }

    #[test]
    fn sign_bit_all_negative() {
        let projected = [-1.0f32; 256];
        let sig = sign_bit_quantize(&projected);
        assert_eq!(sig, BinarySignature([0x00; 32]));
    }

    #[test]
    fn sign_bit_alternating() {
        let mut projected = [0.0f32; 256];
        for (i, val) in projected.iter_mut().enumerate() {
            *val = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let sig = sign_bit_quantize(&projected);
        // Even bits set, odd bits clear → 0b10101010 = 0xAA per byte
        assert_eq!(sig, BinarySignature([0xAA; 32]));
    }

    #[test]
    fn quantize_with_identity() {
        let proj = identity_projector(384);
        let mut emb = vec![-1.0f32; 384];
        // First 4 components positive
        emb[0] = 1.0;
        emb[1] = 1.0;
        emb[2] = 1.0;
        emb[3] = 1.0;

        let sig = proj.quantize(&emb).unwrap();
        // First 4 bits should be 1, rest 0 in first byte: 0b11110000 = 0xF0
        assert_eq!(sig.0[0], 0xF0);
        // Remaining bytes: components 8..255 are -1.0, components 256..383 are unused → 0
        // Bytes 1..31: components 8..255 are -1.0 → all zero bits
        for &b in &sig.0[1..32] {
            assert_eq!(b, 0x00);
        }
    }

    #[test]
    fn dimension_mismatch() {
        let proj = identity_projector(384);
        let emb = vec![0.0f32; 128];
        assert!(proj.quantize(&emb).is_err());
    }

    #[test]
    fn batch_consistency() {
        let proj = identity_projector(384);
        let emb1 = vec![1.0f32; 384];
        let emb2 = vec![-1.0f32; 384];

        let single1 = proj.quantize(&emb1).unwrap();
        let single2 = proj.quantize(&emb2).unwrap();

        let mut batch_input = emb1.clone();
        batch_input.extend_from_slice(&emb2);
        let batch = proj.quantize_batch(&batch_input).unwrap();

        assert_eq!(batch[0], single1);
        assert_eq!(batch[1], single2);
    }

    #[test]
    fn pca_file_roundtrip() {
        let proj = identity_projector(384);
        let dir = std::env::temp_dir().join("primd_test_pca");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pca_matrix.bin");

        proj.write_to_file(&path).unwrap();
        let loaded = PcaProjector::from_file(&path).unwrap();

        assert_eq!(loaded.source_dim, 384);
        assert_eq!(loaded.matrix.len(), proj.matrix.len());
        assert_eq!(loaded.matrix, proj.matrix);

        std::fs::remove_dir_all(&dir).ok();
    }
}
