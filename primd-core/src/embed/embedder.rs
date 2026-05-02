//! Embedder abstraction.
//!
//! Anything that turns text into a fixed-dimension float vector implements
//! `Embedder`. The `EmbeddingPipeline` glues an `Embedder` to either a
//! `PcaProjector` (for high-dim embedders like MiniLM/BGE-small/OpenAI) or
//! direct sign-bit quantization (when the embedder already produces 256 dims).

use super::binary::{BinarySignature, PcaProjector, sign_bit_quantize};
use crate::{PrimdError, Result};

/// Anything that maps text to a dense float vector. Implementations decide the
/// dimension. The trait is intentionally minimal so any backend — feature
/// hashing, in-process MiniLM via Candle, OpenAI API, custom Rust models —
/// can plug in.
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;

    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Strategy for turning a dense embedding into a 256-bit signature.
pub enum SignaturePath {
    /// Apply a precomputed PCA projection, then sign-bit quantize. Use when
    /// the embedder produces > 256 dimensions (most production models).
    Pca(PcaProjector),

    /// The embedder already produces exactly 256 dims; sign-bit quantize
    /// directly. Use for smaller embedders or when you want to skip PCA.
    Direct,
}

/// End-to-end pipeline: text → embedding → signature.
pub struct EmbeddingPipeline<E: Embedder> {
    embedder: E,
    path: SignaturePath,
}

impl<E: Embedder> EmbeddingPipeline<E> {
    pub fn new_direct(embedder: E) -> Result<Self> {
        if embedder.dim() != 256 {
            return Err(PrimdError::InvalidDimension {
                expected: 256,
                got: embedder.dim(),
            });
        }
        Ok(Self {
            embedder,
            path: SignaturePath::Direct,
        })
    }

    pub fn new_with_pca(embedder: E, pca: PcaProjector) -> Result<Self> {
        if embedder.dim() != pca.source_dim() {
            return Err(PrimdError::InvalidDimension {
                expected: pca.source_dim(),
                got: embedder.dim(),
            });
        }
        Ok(Self {
            embedder,
            path: SignaturePath::Pca(pca),
        })
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    pub fn embed_to_signature(&self, text: &str) -> Result<BinarySignature> {
        let dense = self.embedder.embed(text);
        match &self.path {
            SignaturePath::Pca(pca) => pca.quantize(&dense),
            SignaturePath::Direct => {
                if dense.len() != 256 {
                    return Err(PrimdError::InvalidDimension {
                        expected: 256,
                        got: dense.len(),
                    });
                }
                let mut arr = [0f32; 256];
                arr.copy_from_slice(&dense);
                Ok(sign_bit_quantize(&arr))
            }
        }
    }

    pub fn embed_batch_to_signatures(&self, texts: &[&str]) -> Result<Vec<BinarySignature>> {
        let denses = self.embedder.embed_batch(texts);
        match &self.path {
            SignaturePath::Pca(pca) => denses.iter().map(|d| pca.quantize(d)).collect(),
            SignaturePath::Direct => denses
                .iter()
                .map(|d| {
                    if d.len() != 256 {
                        return Err(PrimdError::InvalidDimension {
                            expected: 256,
                            got: d.len(),
                        });
                    }
                    let mut arr = [0f32; 256];
                    arr.copy_from_slice(d);
                    Ok(sign_bit_quantize(&arr))
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial embedder for tests: produces a constant vector of length `dim`,
    /// all 1.0.
    struct ConstantEmbedder {
        dim: usize,
    }

    impl Embedder for ConstantEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }
        fn embed(&self, _: &str) -> Vec<f32> {
            vec![1.0; self.dim]
        }
    }

    #[test]
    fn direct_path_requires_256_dim() {
        let bad = ConstantEmbedder { dim: 384 };
        assert!(EmbeddingPipeline::new_direct(bad).is_err());
        let good = ConstantEmbedder { dim: 256 };
        assert!(EmbeddingPipeline::new_direct(good).is_ok());
    }

    #[test]
    fn pca_path_validates_dim_match() {
        let embedder = ConstantEmbedder { dim: 384 };
        let identity = identity_pca_for(384);
        assert!(EmbeddingPipeline::new_with_pca(embedder, identity).is_ok());

        let embedder = ConstantEmbedder { dim: 384 };
        let mismatched = identity_pca_for(512);
        assert!(EmbeddingPipeline::new_with_pca(embedder, mismatched).is_err());
    }

    #[test]
    fn direct_path_produces_consistent_signatures() {
        let pipe = EmbeddingPipeline::new_direct(ConstantEmbedder { dim: 256 }).unwrap();
        let s1 = pipe.embed_to_signature("hello").unwrap();
        let s2 = pipe.embed_to_signature("world").unwrap();
        // Constant embedder produces same vector for any text → same signature.
        assert_eq!(s1, s2);
    }

    #[test]
    fn batch_matches_singletons() {
        let pipe = EmbeddingPipeline::new_direct(ConstantEmbedder { dim: 256 }).unwrap();
        let texts = ["a", "b", "c"];
        let batch = pipe.embed_batch_to_signatures(&texts).unwrap();
        let singletons: Vec<_> = texts
            .iter()
            .map(|t| pipe.embed_to_signature(t).unwrap())
            .collect();
        assert_eq!(batch, singletons);
    }

    fn identity_pca_for(dim: usize) -> PcaProjector {
        let mut matrix = vec![0.0f32; 256 * dim];
        for i in 0..256.min(dim) {
            matrix[i * dim + i] = 1.0;
        }
        PcaProjector::new(matrix, dim).unwrap()
    }
}
