use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub event: EventId,
    pub at: Instant,
}

pub struct ConversationState {
    window: VecDeque<Observation>,
    capacity: usize,
    max_age: Duration,
}

impl ConversationState {
    pub fn new(capacity: usize, max_age: Duration) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            max_age,
        }
    }

    pub fn observe(&mut self, event: EventId) {
        self.observe_at(event, Instant::now());
    }

    pub fn observe_at(&mut self, event: EventId, at: Instant) {
        self.evict_stale(at);
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(Observation { event, at });
    }

    pub fn evict_stale(&mut self, now: Instant) {
        while let Some(front) = self.window.front() {
            if now.duration_since(front.at) > self.max_age {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn last(&self) -> Option<EventId> {
        self.window.back().map(|o| o.event)
    }

    pub fn last_n(&self, n: usize) -> Vec<EventId> {
        self.window
            .iter()
            .rev()
            .take(n)
            .rev()
            .map(|o| o.event)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Observation> {
        self.window.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slides_when_full() {
        let mut s = ConversationState::new(3, Duration::from_secs(60));
        for i in 0..5u32 {
            s.observe(EventId(i));
        }
        assert_eq!(s.len(), 3);
        assert_eq!(s.last(), Some(EventId(4)));
        assert_eq!(s.last_n(3), vec![EventId(2), EventId(3), EventId(4)]);
    }

    #[test]
    fn evicts_stale() {
        let mut s = ConversationState::new(10, Duration::from_millis(100));
        let t0 = Instant::now();
        s.observe_at(EventId(1), t0);
        s.observe_at(EventId(2), t0 + Duration::from_millis(50));
        s.observe_at(EventId(3), t0 + Duration::from_millis(300));
        assert_eq!(s.len(), 1);
        assert_eq!(s.last(), Some(EventId(3)));
    }

    #[test]
    fn last_n_returns_chronological_order() {
        let mut s = ConversationState::new(5, Duration::from_secs(60));
        s.observe(EventId(10));
        s.observe(EventId(20));
        s.observe(EventId(30));
        assert_eq!(s.last_n(2), vec![EventId(20), EventId(30)]);
        assert_eq!(s.last_n(5), vec![EventId(10), EventId(20), EventId(30)]);
    }
}
