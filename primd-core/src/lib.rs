//! primd-core: sub-millisecond predictive retrieval runtime for voice AI.
//!
//! This crate provides the core retrieval engine: binary quantization,
//! SIMD-accelerated signature scanning, and hierarchical indexing.

pub mod cold_tier;
pub mod embed;
pub mod index;
pub mod predict;
pub mod query_context;

pub use cold_tier::ColdTier;
pub use embed::binary::{BinarySignature, PcaProjector};
pub use index::events::EventCatalog;
pub use index::heap::TopKHeap;
pub use index::shards::{HierarchicalIndex, SearchOptions, SearchResult};
pub use index::signatures::SignatureIndex;
pub use predict::{ConversationState, EventId, MarkovPredictor, NextTurnPredictor, Prediction};
pub use query_context::{QueryContext, QueryOutput, ServedBy};

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
pub enum PrimdError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid embedding dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("invalid signature file: {0}")]
    InvalidSignatureFile(String),

    #[error("PCA matrix file not found or invalid")]
    PcaMatrixMissing,

    #[error("embedder error: {0}")]
    Embedder(String),
}

pub type Result<T> = std::result::Result<T, PrimdError>;
