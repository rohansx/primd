//! Streaming-partials gate.
//!
//! Real STT emits partial transcripts every 50-200ms. Each partial yields a
//! candidate query signature. Naively, every partial would trigger a fresh
//! prefetch — wasting CPU and thrashing the warm cache. The brain doesn't do
//! that: it only re-orients attention when the perceptual signal actually
//! shifts.
//!
//! `StreamingQuery` plays that role. It accepts a stream of partial signatures
//! and emits a "stable" signature only when the latest partial has drifted
//! sufficiently from the last emit. The drift metric is Hamming distance,
//! which matches the rest of the engine's distance model.

use crate::embed::binary::BinarySignature;

#[derive(Default, Clone, Copy, Debug)]
pub struct StreamingStats {
    pub updates_received: u64,
    pub updates_emitted: u64,
    pub updates_suppressed: u64,
}

impl StreamingStats {
    pub fn emit_rate(&self) -> f32 {
        if self.updates_received == 0 {
            return 0.0;
        }
        self.updates_emitted as f32 / self.updates_received as f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitDecision {
    Suppressed,
    Emitted(BinarySignature),
}

/// Tracks the latest partial signature for an in-flight utterance and gates
/// downstream prefetch triggers based on how much the signature has drifted.
pub struct StreamingQuery {
    last_emitted: Option<BinarySignature>,
    current: Option<BinarySignature>,
    drift_threshold: u32,
    stats: StreamingStats,
}

impl StreamingQuery {
    /// `drift_threshold` is the minimum Hamming distance from the previous
    /// emitted signature for a new partial to trigger a fresh emit. For 256-bit
    /// signatures, sensible values are 12-32 bits (~5-12% of the signature).
    pub fn new(drift_threshold: u32) -> Self {
        Self {
            last_emitted: None,
            current: None,
            drift_threshold,
            stats: StreamingStats::default(),
        }
    }

    /// Feed a new partial signature. Returns whether a fresh prefetch should
    /// fire and, if so, the signature to scan with.
    pub fn update(&mut self, partial: BinarySignature) -> EmitDecision {
        self.stats.updates_received += 1;
        self.current = Some(partial);

        let should_emit = match self.last_emitted {
            None => true,
            Some(prev) => prev.hamming_distance(&partial) >= self.drift_threshold,
        };

        if should_emit {
            self.last_emitted = Some(partial);
            self.stats.updates_emitted += 1;
            EmitDecision::Emitted(partial)
        } else {
            self.stats.updates_suppressed += 1;
            EmitDecision::Suppressed
        }
    }

    /// Reset state at end of utterance. Call when the STT finalizes — the
    /// next utterance starts with a fresh emit.
    pub fn reset(&mut self) {
        self.last_emitted = None;
        self.current = None;
    }

    pub fn current(&self) -> Option<BinarySignature> {
        self.current
    }

    pub fn last_emitted(&self) -> Option<BinarySignature> {
        self.last_emitted
    }

    pub fn stats(&self) -> StreamingStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig_with_pattern(pattern: u8) -> BinarySignature {
        BinarySignature([pattern; 32])
    }

    fn flip_n_bits(sig: &BinarySignature, n: u32) -> BinarySignature {
        let mut out = *sig;
        for i in 0..n {
            let bit = (i * 7) as usize % 256;
            out.0[bit / 8] ^= 1 << (bit % 8);
        }
        out
    }

    #[test]
    fn first_partial_always_emits() {
        let mut q = StreamingQuery::new(16);
        let s = sig_with_pattern(0xAA);
        match q.update(s) {
            EmitDecision::Emitted(sig) => assert_eq!(sig, s),
            EmitDecision::Suppressed => panic!("first partial should always emit"),
        }
        assert_eq!(q.stats().updates_emitted, 1);
    }

    #[test]
    fn small_drift_is_suppressed() {
        let mut q = StreamingQuery::new(16);
        let base = sig_with_pattern(0xAA);
        q.update(base);
        // Flip 4 bits — well below threshold of 16
        let nudged = flip_n_bits(&base, 4);
        assert!(matches!(q.update(nudged), EmitDecision::Suppressed));
        assert_eq!(q.stats().updates_emitted, 1);
        assert_eq!(q.stats().updates_suppressed, 1);
    }

    #[test]
    fn large_drift_emits() {
        let mut q = StreamingQuery::new(16);
        let base = sig_with_pattern(0xAA);
        q.update(base);
        let drifted = flip_n_bits(&base, 32);
        assert!(matches!(q.update(drifted), EmitDecision::Emitted(_)));
        assert_eq!(q.stats().updates_emitted, 2);
    }

    #[test]
    fn drift_is_measured_from_last_emit_not_last_seen() {
        // Sequence: base (emit) -> +4 bits (suppress) -> +8 bits (suppress) ->
        // +20 bits cumulative (emit because dist(base, this) >= 16)
        let mut q = StreamingQuery::new(16);
        let base = sig_with_pattern(0xAA);
        q.update(base);

        let s4 = flip_n_bits(&base, 4);
        let s8 = flip_n_bits(&base, 8);
        let s20 = flip_n_bits(&base, 20);

        assert!(matches!(q.update(s4), EmitDecision::Suppressed));
        assert!(matches!(q.update(s8), EmitDecision::Suppressed));
        assert!(matches!(q.update(s20), EmitDecision::Emitted(_)));
    }

    #[test]
    fn reset_makes_next_partial_emit() {
        let mut q = StreamingQuery::new(16);
        let s = sig_with_pattern(0x55);
        q.update(s);
        q.reset();
        assert!(matches!(q.update(s), EmitDecision::Emitted(_)));
    }

    #[test]
    fn current_tracks_latest_regardless_of_emit() {
        let mut q = StreamingQuery::new(64);
        let a = sig_with_pattern(0x11);
        let b = flip_n_bits(&a, 4);
        q.update(a);
        q.update(b);
        assert_eq!(q.current(), Some(b));
        assert_eq!(q.last_emitted(), Some(a));
    }
}
