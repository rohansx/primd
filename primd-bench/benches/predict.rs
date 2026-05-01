//! Benchmark: cold scan vs prefetch-warmed scan.
//!
//! Simulates a conversation with a learnable transition pattern. Each event
//! owns a slice of signatures. The predictor learns the transition graph from
//! a training trace, then we measure latency for cold (full-corpus) and warm
//! (prefetched scope) retrieval on a held-out test trace.

use std::collections::HashMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use primd_core::embed::binary::BinarySignature;
use primd_core::index::signatures::SignatureIndex;
use primd_core::predict::{ConversationState, EventId, MarkovPredictor, PrefetchCoordinator};

const SEED: u64 = 0xCAFE_BEEF;
const VOCAB: u32 = 50;
const SIGS_PER_EVENT: usize = 2_000;
const TOTAL_SIGS: usize = (VOCAB as usize) * SIGS_PER_EVENT;

fn random_sig(rng: &mut StdRng) -> BinarySignature {
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    BinarySignature(bytes)
}

/// Build the corpus: each event has a "centroid" signature, and the event's
/// signatures are perturbations of that centroid (~30 bits flipped). This
/// mimics real semantic clustering — items belonging to the same event share
/// embedding structure.
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
        let indices: Vec<usize> = (start..start + SIGS_PER_EVENT).collect();
        scope.insert(EventId(event), indices);
        for _ in 0..SIGS_PER_EVENT {
            sigs.push(perturb(&centroid, &mut rng, 30));
        }
    }
    (SignatureIndex::new(sigs), scope, centroids)
}

/// Flip `n_bits` random bits in a signature.
fn perturb(sig: &BinarySignature, rng: &mut StdRng, n_bits: u32) -> BinarySignature {
    let mut out = *sig;
    for _ in 0..n_bits {
        let bit = rng.random_range(0..256);
        let byte_idx = bit / 8;
        let bit_idx = bit % 8;
        out.0[byte_idx] ^= 1 << bit_idx;
    }
    out
}

/// Build a Markov-friendly trace: each event has 2-3 likely successors.
fn build_trace(rng: &mut StdRng, length: usize) -> Vec<EventId> {
    let mut transitions: HashMap<u32, Vec<u32>> = HashMap::new();
    for from in 0..VOCAB {
        // Each event has 3 likely successors, one dominant (60% weight)
        let dominant = (from + 1) % VOCAB;
        let alt1 = (from + 7) % VOCAB;
        let alt2 = (from + 13) % VOCAB;
        let mut weighted = Vec::new();
        for _ in 0..6 {
            weighted.push(dominant);
        }
        for _ in 0..2 {
            weighted.push(alt1);
        }
        for _ in 0..2 {
            weighted.push(alt2);
        }
        transitions.insert(from, weighted);
    }

    let mut trace = Vec::with_capacity(length);
    let mut current = 0u32;
    trace.push(EventId(current));
    for _ in 1..length {
        let options = &transitions[&current];
        current = options[rng.random_range(0..options.len())];
        trace.push(EventId(current));
    }
    trace
}

fn bench_cold_vs_warm(c: &mut Criterion) {
    let (idx, scope, centroids) = build_corpus();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x1111);

    // Train predictor on a large trace
    let train_trace = build_trace(&mut rng, 5000);
    let mut predictor = MarkovPredictor::with_smoothing(0.01);
    predictor.observe_sequence(&train_trace);

    // Test trace — same distribution, fresh randomness
    let mut test_rng = StdRng::seed_from_u64(SEED ^ 0x2222);
    let test_trace = build_trace(&mut test_rng, 200);

    // Build (state, query) pairs. Each query is a perturbation of the centroid
    // of the *next* event in the trace — i.e., the query semantically belongs
    // to the event that the predictor is trying to anticipate.
    let mut queries: Vec<(ConversationState, BinarySignature)> = Vec::new();
    let mut q_rng = StdRng::seed_from_u64(SEED ^ 0x3333);
    for i in 3..test_trace.len() {
        let mut state = ConversationState::new(8, Duration::from_secs(60));
        for &e in &test_trace[i - 3..i] {
            state.observe(e);
        }
        let next_event = test_trace[i].0 as usize;
        let query = perturb(&centroids[next_event], &mut q_rng, 40);
        queries.push((state, query));
    }

    // Pre-stage the predicted scopes so the bench only times retrieval, not
    // prediction overhead (which is amortized over a real conversation).
    let mut staged_queries: Vec<(Vec<usize>, BinarySignature)> = Vec::new();
    {
        let mut coord = PrefetchCoordinator::new(&idx, clone_predictor(&predictor), scope.clone())
            .with_confidence_threshold(0.05)
            .with_top_n_events(3);
        for (state, query) in &queries {
            coord.prefetch(state);
            staged_queries.push((coord.last_predicted_scope().to_vec(), *query));
        }
    }

    let mut group = c.benchmark_group("retrieval_latency");
    group.sample_size(50);

    group.bench_function(BenchmarkId::new("cold_sequential", TOTAL_SIGS), |b| {
        b.iter(|| {
            for (_, query) in &queries {
                let _ = idx.scan_top_k(query, 10);
            }
        });
    });

    group.bench_function(BenchmarkId::new("cold_parallel", TOTAL_SIGS), |b| {
        b.iter(|| {
            for (_, query) in &queries {
                let _ = idx.scan_top_k_parallel(query, 10);
            }
        });
    });

    group.bench_function(BenchmarkId::new("warm_sequential", TOTAL_SIGS), |b| {
        b.iter(|| {
            for (subset, query) in &staged_queries {
                let _ = idx.scan_top_k_subset(query, subset, 10);
            }
        });
    });

    group.bench_function(BenchmarkId::new("warm_parallel", TOTAL_SIGS), |b| {
        b.iter(|| {
            for (subset, query) in &staged_queries {
                let _ = idx.scan_top_k_subset_parallel(query, subset, 10);
            }
        });
    });

    group.finish();

    // Print one-shot recall and scope stats for visibility.
    let mut coord = PrefetchCoordinator::new(&idx, clone_predictor(&predictor), scope.clone())
        .with_confidence_threshold(0.05)
        .with_top_n_events(3);

    let mut total_scope = 0usize;
    let mut hits = 0usize;
    for (state, query) in &queries {
        coord.prefetch(state);
        let warm = idx.scan_top_k_subset(query, coord.last_predicted_scope(), 10);
        let cold = idx.scan_top_k_parallel(query, 10);
        total_scope += coord.last_predicted_scope().len();
        if let (Some((wd, _)), Some((cd, _))) = (warm.first(), cold.first())
            && wd == cd
        {
            hits += 1;
        }
    }
    let avg_scope = total_scope / queries.len().max(1);
    println!(
        "\n=== prefetch summary: avg_scope={} ({:.1}% of corpus), top1_match={}/{} ({:.1}%)",
        avg_scope,
        100.0 * avg_scope as f32 / TOTAL_SIGS as f32,
        hits,
        queries.len(),
        100.0 * hits as f32 / queries.len() as f32,
    );
}

/// Build a fresh predictor copy by replaying the training trace. Cheaper than
/// implementing Clone on the predictor for now.
fn clone_predictor(_p: &MarkovPredictor) -> MarkovPredictor {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x1111);
    let trace = build_trace(&mut rng, 5000);
    let mut fresh = MarkovPredictor::with_smoothing(0.01);
    fresh.observe_sequence(&trace);
    fresh
}

criterion_group!(predict_benches, bench_cold_vs_warm);
criterion_main!(predict_benches);
