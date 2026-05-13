pub mod cache;
pub mod markov;
pub mod predictor;
pub mod prefetch;
pub mod state;
pub mod streaming;
pub mod trainer;

pub use cache::{DeltaCache, DeltaCacheStats};
pub use markov::MarkovPredictor;
pub use predictor::{NextTurnPredictor, Prediction};
pub use prefetch::{
    FinalScanResult, PrefetchCoordinator, PrefetchStats, StreamingPrefetchStats,
    StreamingPrefetcher, WarmScanResult,
};
pub use state::{ConversationState, EventId, Observation};
pub use streaming::{EmitDecision, StreamingQuery, StreamingStats};
