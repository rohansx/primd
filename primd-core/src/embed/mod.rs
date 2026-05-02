pub mod binary;
pub mod embedder;
pub mod hashed;
#[cfg(feature = "openai")]
pub mod openai;

pub use embedder::{Embedder, EmbeddingPipeline, SignaturePath};
pub use hashed::HashedEmbedder;
#[cfg(feature = "openai")]
pub use openai::OpenAIEmbedder;
