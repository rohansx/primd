//! `NextTurnPredictor` — the trait every conversation-turn predictor implements.
//!
//! v0.1 ships one impl: [`super::markov::MarkovPredictor`]. v0.2 will add a
//! Successor-Representation impl in the `primd-sr` crate and a `HybridPredictor`
//! that gates between SR and Markov via [`NextTurnPredictor::confidence`].
//!
//! Adding the trait now (before SR lands) lets us refactor [`crate::QueryContext`]
//! once and keep its API stable across the v0.2 cycle.

use super::markov::MarkovPredictor;
use super::EventId;

/// A single (event, probability) prediction returned by a predictor.
///
/// Lives in the trait module so it doesn't bind callers to the Markov impl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub event: EventId,
    pub probability: f32,
}

/// Predict and learn the conversation's next-turn event distribution.
///
/// The trait intentionally hides the predictor's internal state (transition
/// counts for Markov; W and M_low matrices for SR). Callers see only the
/// behavioral surface: predict, observe, and an optional confidence signal.
pub trait NextTurnPredictor: Send + Sync {
    /// Predict the top-K most likely next events given the recent context.
    ///
    /// `context` is in chronological order (oldest first); the most recent
    /// event is at `context.last()`. Returned predictions are sorted by
    /// descending probability.
    fn predict(&self, context: &[EventId], k: usize) -> Vec<Prediction>;

    /// Record an observed transition `prev → next`.
    ///
    /// Called from [`crate::QueryContext::finalize`] when a finalize resolves
    /// to a top event; the predictor updates its internal state so future
    /// `warm_next` calls reflect the new observation.
    fn observe(&mut self, prev: EventId, next: EventId);

    /// Confidence in this predictor's current output, in `[0.0, 1.0]`.
    ///
    /// Used by hybrid wrappers to gate between predictors: when this falls
    /// below a threshold, fall back to a more general predictor (e.g. the
    /// uniform prior, or a different impl).
    ///
    /// Default `1.0`. Predictors without a calibrated signal (the v0.1 Markov
    /// impl) keep the default; SR overrides it with the spectral gap of its
    /// low-rank successor matrix.
    fn confidence(&self) -> f32 {
        1.0
    }

    /// Access the underlying [`MarkovPredictor`] if this predictor wraps
    /// one. Used by the `primd serve` per-session persistence layer to
    /// extract Markov state at session reset without needing to downcast
    /// concrete types.
    ///
    /// Default `None`. [`MarkovPredictor`] itself returns `Some(self)`; the
    /// `HybridPredictor` in `primd-sr` returns `Some(&self.markov)`. Pure
    /// SR predictors return `None` — they have no Markov state to persist
    /// in the v0.2.6 surface (full SR persistence is a v0.2.7 deliverable).
    fn as_markov(&self) -> Option<&MarkovPredictor> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal stub used in `query_context` tests — returns a fixed prediction
    /// regardless of context. Confirms the trait is object-safe and callable
    /// behind `Box<dyn NextTurnPredictor>`.
    struct StubPredictor {
        next: EventId,
        last_observed: Option<(EventId, EventId)>,
    }

    impl NextTurnPredictor for StubPredictor {
        fn predict(&self, _context: &[EventId], k: usize) -> Vec<Prediction> {
            if k == 0 {
                return Vec::new();
            }
            vec![Prediction {
                event: self.next,
                probability: 1.0,
            }]
        }

        fn observe(&mut self, prev: EventId, next: EventId) {
            self.last_observed = Some((prev, next));
        }
    }

    #[test]
    fn trait_is_object_safe() {
        let mut p: Box<dyn NextTurnPredictor> = Box::new(StubPredictor {
            next: EventId(42),
            last_observed: None,
        });
        let preds = p.predict(&[EventId(1)], 1);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].event, EventId(42));
        p.observe(EventId(1), EventId(2));
    }

    #[test]
    fn default_confidence_is_one() {
        let p = StubPredictor {
            next: EventId(0),
            last_observed: None,
        };
        assert!((p.confidence() - 1.0).abs() < f32::EPSILON);
    }
}
