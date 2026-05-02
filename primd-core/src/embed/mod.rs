pub mod binary;
pub mod embedder;
pub mod hashed;

pub use embedder::{Embedder, EmbeddingPipeline, SignaturePath};
pub use hashed::HashedEmbedder;
