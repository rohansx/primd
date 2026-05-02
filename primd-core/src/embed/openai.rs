//! OpenAI text embeddings via the public API.
//!
//! Reads `OPENAI_API_KEY` from the environment. By default uses
//! `text-embedding-3-small` with `dimensions=256`, which lets primd skip PCA
//! entirely — OpenAI's server-side projection produces a 256-dim float vector
//! ready for direct sign-bit quantization.
//!
//! Network calls are synchronous (via `ureq`). Embedding latency: ~30-100ms
//! per request depending on batch size and region. For voice-AI workloads,
//! prefer batching as many texts as possible per call to amortize overhead.

use serde::{Deserialize, Serialize};

use super::embedder::Embedder;
use crate::{PrimdError, Result};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIM: usize = 256;
const REQUEST_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone)]
pub struct OpenAIEmbedder {
    api_key: String,
    model: String,
    dim: usize,
    base_url: String,
}

impl OpenAIEmbedder {
    /// Construct from the `OPENAI_API_KEY` environment variable. Defaults:
    /// model = `text-embedding-3-small`, dim = 256.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            PrimdError::Embedder("OPENAI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::with_api_key(api_key))
    }

    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            dim: DEFAULT_DIM,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn request(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbeddingRequest {
            model: &self.model,
            input: texts,
            dimensions: self.dim,
            encoding_format: "float",
        };

        let url = format!("{}/embeddings", self.base_url);
        let auth = format!("Bearer {}", self.api_key);
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build();

        let resp = agent
            .post(&url)
            .set("Authorization", &auth)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| PrimdError::Embedder(format!("OpenAI request failed: {e}")))?;

        let parsed: EmbeddingResponse = resp
            .into_json()
            .map_err(|e| PrimdError::Embedder(format!("OpenAI response parse failed: {e}")))?;

        // OpenAI sometimes returns out-of-order `index` fields in the data array;
        // explicitly sort by index to guarantee correspondence with input order.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);

        if data.len() != texts.len() {
            return Err(PrimdError::Embedder(format!(
                "OpenAI returned {} embeddings for {} inputs",
                data.len(),
                texts.len()
            )));
        }

        for d in &data {
            if d.embedding.len() != self.dim {
                return Err(PrimdError::InvalidDimension {
                    expected: self.dim,
                    got: d.embedding.len(),
                });
            }
        }

        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

impl Embedder for OpenAIEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.request(&[text])?;
        Ok(out.pop().unwrap_or_default())
    }

    /// Override default batch implementation to send all texts in a single
    /// HTTP call — this is the main reason to prefer batch mode for indexing.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.request(texts)
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    dimensions: usize,
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_errors_without_key() {
        // Save and clear (test runs in process; do not pollute other tests)
        let prev = std::env::var("OPENAI_API_KEY").ok();
        // SAFETY: tests that mutate env vars are inherently single-threaded with
        // each other; we restore the variable before returning.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }

        let result = OpenAIEmbedder::from_env();
        assert!(result.is_err());
        match result {
            Err(PrimdError::Embedder(msg)) => assert!(msg.contains("OPENAI_API_KEY")),
            other => panic!("unexpected: {other:?}"),
        }

        if let Some(v) = prev {
            unsafe {
                std::env::set_var("OPENAI_API_KEY", v);
            }
        }
    }

    #[test]
    fn builder_options_apply() {
        let e = OpenAIEmbedder::with_api_key("test-key")
            .with_model("text-embedding-3-large")
            .with_dim(1024)
            .with_base_url("http://localhost:1234");
        assert_eq!(e.dim(), 1024);
        assert_eq!(e.model(), "text-embedding-3-large");
        assert_eq!(e.base_url, "http://localhost:1234");
    }
}
