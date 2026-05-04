use crate::predict::{EventId, MarkovPredictor};

pub fn train_sequences<I>(sequences: I, max_order: usize, smoothing: f32) -> MarkovPredictor
where
    I: IntoIterator<Item = Vec<EventId>>,
{
    let mut predictor = MarkovPredictor::with_order_and_smoothing(max_order, smoothing);
    for seq in sequences {
        if seq.len() >= 2 {
            predictor.observe_sequence(&seq);
        }
    }
    predictor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trains_from_multiple_sequences() {
        let predictor = train_sequences(
            vec![vec![EventId(1), EventId(2)], vec![EventId(1), EventId(2)]],
            1,
            0.01,
        );
        let preds = predictor.predict(EventId(1), 1);
        assert_eq!(preds[0].event, EventId(2));
    }
}
