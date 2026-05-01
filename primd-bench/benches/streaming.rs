//! Benchmark: streaming speculative prefetch vs final-only retrieval.
//!
//! Models a voice utterance as a sequence of partial signatures that
//! progressively converge on the final embedding. The "final-only" path waits
//! for STT to finalize, then runs a warm scan. The "streaming" path runs
//! speculative warm scans during user speech, then either reuses the cached
//! result (if final ≈ last partial) or rescans.
//!
//! In a real voice agent, the speculative work overlaps with user speech, so
//! the *user-visible* latency is just the cache lookup. This bench measures
//! both perspectives:
//!
//! - `final_only`: latency the user observes when retrieval starts at finalization.
//! - `streaming_user_visible`: latency the user observes after speculative serve.
//! - `streaming_total_cpu`: total CPU work across all partials + final.

use std::collections::HashMap;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use primd_core::embed::binary::BinarySignature;
use primd_core::index::signatures::SignatureIndex;
use primd_core::predict::{
    ConversationState, EventId, MarkovPredictor, PrefetchCoordinator, StreamingPrefetcher,
};

const SEED: u64 = 0xFEED_FACE;
const VOCAB: u32 = 50;
const SIGS_PER_EVENT: usize = 2_000;
const TOTAL_SIGS: usize = (VOCAB as usize) * SIGS_PER_EVENT;
const PARTIALS_PER_UTTERANCE: usize = 5;

fn random_sig(rng: &mut StdRng) -> BinarySignature {
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    BinarySignature(bytes)
}

fn perturb(sig: &BinarySignature, rng: &mut StdRng, n_bits: u32) -> BinarySignature {
    let mut out = *sig;
    for _ in 0..n_bits {
        let bit = rng.random_range(0..256);
        out.0[bit / 8] ^= 1 << (bit % 8);
    }
    out
}

fn build_corpus() -> (
    SignatureIndex,
    HashMap<EventId, Vec<usize>>,
    Vec<BinarySignature>,
) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut sigs = Vec::with_capacity(TOTAL_SIGS);
    let mut scope: HashMap<EventId, Vec<usize>> = HashMap::new();
    let mut centroids: Vec<BinarySignature> = Vec::with_capacity(VOCAB as usize);

    for event in 0..VOCAB {
        let centroid = random_sig(&mut rng);
        centroids.push(centroid);
        let start = event as usize * SIGS_PER_EVENT;
        scope.insert(EventId(event), (start..start + SIGS_PER_EVENT).collect());
        for _ in 0..SIGS_PER_EVENT {
            sigs.push(perturb(&centroid, &mut rng, 30));
        }
    }
    (SignatureIndex::new(sigs), scope, centroids)
}

fn build_trace(rng: &mut StdRng, length: usize) -> Vec<EventId> {
    let mut current = 0u32;
    let mut trace = vec![EventId(current)];
    for _ in 1..length {
        let r = rng.random_range(0..10);
        current = match r {
            0..=5 => (current + 1) % VOCAB,
            6..=7 => (current + 7) % VOCAB,
            _ => (current + 13) % VOCAB,
        };
        trace.push(EventId(current));
    }
    trace
}

/// Build an utterance: a sequence of partial signatures that drift from a
/// random starting point toward the final centroid.
fn build_partials(
    final_sig: BinarySignature,
    rng: &mut StdRng,
    count: usize,
) -> (Vec<BinarySignature>, BinarySignature) {
    let mut partials = Vec::with_capacity(count);
    // Start far from the final
    let mut current = perturb(&final_sig, rng, 80);
    for i in 0..count {
        // Each step moves about 60/count fewer bits away
        let drift_remaining = ((count - i) as u32) * 60 / count.max(1) as u32;
        let candidate = perturb(&final_sig, rng, drift_remaining.max(2));
        partials.push(candidate);
        current = candidate;
    }
    (partials, current)
}

fn build_utterances(
    centroids: &[BinarySignature],
    trace: &[EventId],
) -> Vec<(ConversationState, Vec<BinarySignature>, BinarySignature)> {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x9999);
    let mut utterances = Vec::new();
    for i in 3..trace.len() {
        let mut state = ConversationState::new(8, Duration::from_secs(60));
        for &e in &trace[i - 3..i] {
            state.observe(e);
        }
        let next_event = trace[i].0 as usize;
        let final_sig = perturb(&centroids[next_event], &mut rng, 40);
        let (partials, _) = build_partials(final_sig, &mut rng, PARTIALS_PER_UTTERANCE);
        utterances.push((state, partials, final_sig));
    }
    utterances
}

fn fresh_predictor() -> MarkovPredictor {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x1111);
    let trace = build_trace(&mut rng, 5000);
    let mut p = MarkovPredictor::with_smoothing(0.01);
    p.observe_sequence(&trace);
    p
}

fn bench_streaming_vs_final(c: &mut Criterion) {
    let (idx, scope, centroids) = build_corpus();
    let mut test_rng = StdRng::seed_from_u64(SEED ^ 0x2222);
    let trace = build_trace(&mut test_rng, 200);
    let utterances = build_utterances(&centroids, &trace);

    let mut group = c.benchmark_group("voice_turn_latency");
    group.sample_size(50);

    // -----------------------------------------------------------------
    // final_only: model the user-visible path where retrieval starts
    // only after STT finalizes. Each utterance pays a full warm scan.
    // -----------------------------------------------------------------
    group.bench_function("final_only", |b| {
        b.iter_batched(
            || {
                PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
                    .with_confidence_threshold(0.05)
                    .with_top_n_events(3)
            },
            |mut coord| {
                for (state, _partials, final_sig) in &utterances {
                    coord.prefetch(state);
                    let _ = idx.scan_top_k_subset(final_sig, coord.last_predicted_scope(), 10);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // -----------------------------------------------------------------
    // streaming_user_visible: only the work after STT finalizes.
    // The speculative scans during user speech are excluded — they
    // happened in parallel with speech and are off the critical path.
    // -----------------------------------------------------------------
    group.bench_function("streaming_user_visible", |b| {
        b.iter_batched(
            || {
                let coord = PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
                    .with_confidence_threshold(0.05)
                    .with_top_n_events(3);
                let mut sp = StreamingPrefetcher::new(coord, 12);
                // Pre-execute the speculative work outside the timed region
                for (state, partials, _) in &utterances {
                    sp.prefetch(state);
                    for p in partials {
                        sp.on_partial(*p, 10);
                    }
                }
                sp
            },
            |mut sp| {
                // Now time only the on_final calls — the user-visible cost.
                for (_state, _partials, final_sig) in &utterances {
                    let _ = sp.on_final(*final_sig, 10, 16);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // -----------------------------------------------------------------
    // streaming_total_cpu: full work — partials + final. This is the
    // total CPU cost of the streaming approach across the utterance.
    // -----------------------------------------------------------------
    group.bench_function("streaming_total_cpu", |b| {
        b.iter_batched(
            || {
                let coord = PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
                    .with_confidence_threshold(0.05)
                    .with_top_n_events(3);
                StreamingPrefetcher::new(coord, 12)
            },
            |mut sp| {
                for (state, partials, final_sig) in &utterances {
                    sp.prefetch(state);
                    for p in partials {
                        sp.on_partial(*p, 10);
                    }
                    let _ = sp.on_final(*final_sig, 10, 16);
                    sp.end_utterance();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();

    // Print summary stats
    let coord = PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
        .with_confidence_threshold(0.05)
        .with_top_n_events(3);
    let mut sp = StreamingPrefetcher::new(coord, 12);
    for (state, partials, final_sig) in &utterances {
        sp.prefetch(state);
        for p in partials {
            sp.on_partial(*p, 10);
        }
        let _ = sp.on_final(*final_sig, 10, 16);
        sp.end_utterance();
    }
    let s = sp.stats();
    println!(
        "\n=== streaming summary: utterances={}, partials={}, speculative_scans={}, served_speculatively={}, rescans={}, hit_rate={:.1}%",
        s.utterances,
        s.partial_updates,
        s.speculative_scans,
        s.finals_served_speculatively,
        s.finals_required_rescan,
        100.0 * s.speculative_hit_rate(),
    );
}

criterion_group!(streaming_benches, bench_streaming_vs_final);
criterion_main!(streaming_benches);
