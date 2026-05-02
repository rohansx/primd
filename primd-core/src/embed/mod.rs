pub mod binary;
pub mod embedder;
pub mod hashed;
#[cfg(feature = "local")]
pub mod local;
#[cfg(feature = "openai")]
pub mod openai;

pub use binary::random_projection;
pub use embedder::{Embedder, EmbeddingPipeline, SignaturePath};
pub use hashed::HashedEmbedder;
#[cfg(feature = "local")]
pub use local::{LocalEmbedder, LocalModel};
#[cfg(feature = "openai")]
pub use openai::OpenAIEmbedder;
