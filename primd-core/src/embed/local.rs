//! Local ML embedder via fastembed-rs (ONNX runtime under the hood).
//!
//! Bundles MiniLM, BGE-small, and BGE-base. Models are downloaded from
//! HuggingFace on first use and cached in `~/.cache/fastembed`.
//!
//! Output dimensions:
//!   AllMiniLMV2 → 384
//!   BGESmallEN  → 384
//!   BGEBaseEN   → 768
//!
//! All three exceed primd's 256-bit signature, so the pipeline must combine
//! a `LocalEmbedder` with a `PcaProjector` (via `random_projection` for
//! deterministic sign-bit-friendly compression).

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use super::embedder::Embedder;
use crate::{PrimdError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalModel {
    AllMiniLM,
    BgeSmallEn,
    BgeBaseEn,
}

impl LocalModel {
    pub fn dim(self) -> usize {
        match self {
            LocalModel::AllMiniLM => 384,
            LocalModel::BgeSmallEn => 384,
            LocalModel::BgeBaseEn => 768,
        }
    }

    fn fastembed_model(self) -> EmbeddingModel {
        match self {
            LocalModel::AllMiniLM => EmbeddingModel::AllMiniLML6V2,
            LocalModel::BgeSmallEn => EmbeddingModel::BGESmallENV15,
            LocalModel::BgeBaseEn => EmbeddingModel::BGEBaseENV15,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LocalModel::AllMiniLM => "all-MiniLM-L6-v2",
            LocalModel::BgeSmallEn => "bge-small-en-v1.5",
            LocalModel::BgeBaseEn => "bge-base-en-v1.5",
        }
    }
}

pub struct LocalEmbedder {
    model: TextEmbedding,
    kind: LocalModel,
}

impl LocalEmbedder {
    /// Initialize and download (if not cached) the chosen model. Triggers a
    /// network call on first run; subsequent runs read from
    /// `~/.cache/fastembed`.
    pub fn new(kind: LocalModel) -> Result<Self> {
        let opts = InitOptions::new(kind.fastembed_model()).with_show_download_progress(false);
        let model = TextEmbedding::try_new(opts)
            .map_err(|e| PrimdError::Embedder(format!("fastembed init: {e}")))?;
        Ok(Self { model, kind })
    }

    pub fn kind(&self) -> LocalModel {
        self.kind
    }
}

impl Embedder for LocalEmbedder {
    fn dim(&self) -> usize {
        self.kind.dim()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text])?;
        Ok(out.pop().unwrap_or_default())
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<&str> = texts.to_vec();
        self.model
            .embed(owned, None)
            .map_err(|e| PrimdError::Embedder(format!("fastembed embed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dims_are_correct() {
        assert_eq!(LocalModel::AllMiniLM.dim(), 384);
        assert_eq!(LocalModel::BgeSmallEn.dim(), 384);
        assert_eq!(LocalModel::BgeBaseEn.dim(), 768);
    }

    // Note: the `local_embedder_runs` test would actually download a model
    // (~90MB) from HuggingFace and is therefore not part of the default unit
    // suite. Verify locally with:
    //
    //   cargo test -p primd-core --features local --release \
    //     -- --ignored local_embedder
    #[test]
    #[ignore = "downloads ~90MB model from HuggingFace on first run"]
    fn local_embedder_runs() {
        let e = LocalEmbedder::new(LocalModel::AllMiniLM).expect("init MiniLM");
        let v = e.embed("hello world").expect("embed");
        assert_eq!(v.len(), 384);
        let unit = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(unit > 0.5 && unit < 1.5, "expected ~unit norm, got {unit}");
    }
}
