//! Full-pipeline benchmark: layers 1+2+3 (streaming + warm scan + predictor)
//! versus layers 1+2+3+4 (adds the predictive coding delta cache).
//!
//! Models a workload where users return to similar questions across multiple
//! sessions — exactly the workload predictive coding is designed for. The
//! cache should pick up repeated query patterns and serve them with zero scan.

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

const SEED: u64 = 0xBEEF_CAFE;
const VOCAB: u32 = 50;
const SIGS_PER_EVENT: usize = 2_000;
const TOTAL_SIGS: usize = (VOCAB as usize) * SIGS_PER_EVENT;

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

fn fresh_predictor() -> MarkovPredictor {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x1111);
    let trace = build_trace(&mut rng, 5000);
    let mut p = MarkovPredictor::with_smoothing(0.01);
    p.observe_sequence(&trace);
    p
}

/// Build a workload where queries cluster around K canonical "intents". Each
/// utterance picks one canonical query and adds noise — modeling users
/// repeatedly asking the same kinds of questions across sessions.
fn build_repeated_workload(
    centroids: &[BinarySignature],
    n_canonical_queries: usize,
    n_utterances: usize,
) -> Vec<(ConversationState, BinarySignature)> {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x4444);
    let trace = build_trace(&mut rng, n_utterances + 4);

    // Build a small set of canonical (state, query) "intents"
    let mut canonicals: Vec<(ConversationState, BinarySignature)> =
        Vec::with_capacity(n_canonical_queries);
    for _ in 0..n_canonical_queries {
        let i = rng.random_range(3..trace.len() - 1);
        let mut state = ConversationState::new(8, Duration::from_secs(60));
        for &e in &trace[i - 3..i] {
            state.observe(e);
        }
        let next_event = trace[i].0 as usize;
        let query = perturb(&centroids[next_event], &mut rng, 40);
        canonicals.push((state, query));
    }

    // Now build the actual workload by sampling canonicals + adding small noise
    let mut workload = Vec::with_capacity(n_utterances);
    for _ in 0..n_utterances {
        let (canonical_state, canonical_query) = &canonicals[rng.random_range(0..canonicals.len())];
        // Clone state. Add small drift to the query (~6 bits) to simulate STT
        // and embedding variability for the same canonical intent.
        let mut state = ConversationState::new(8, Duration::from_secs(60));
        for obs in canonical_state.iter() {
            state.observe(obs.event);
        }
        let drifted = perturb(canonical_query, &mut rng, 6);
        workload.push((state, drifted));
    }
    workload
}

fn bench_full_pipeline(c: &mut Criterion) {
    let (idx, scope, centroids) = build_corpus();
    // 200 utterances across 20 canonical intents → 10 reps per intent on avg
    let workload = build_repeated_workload(&centroids, 20, 200);

    let mut group = c.benchmark_group("full_pipeline_user_visible");
    group.sample_size(50);

    // ---- No cache: every final does a warm scan ----
    group.bench_function("no_cache", |b| {
        b.iter_batched(
            || {
                let coord = PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
                    .with_confidence_threshold(0.05)
                    .with_top_n_events(3);
                StreamingPrefetcher::new(coord, 24)
            },
            |mut sp| {
                for (state, query) in &workload {
                    sp.prefetch(state);
                    let _ = sp.on_final(*query, 10, 16);
                    sp.end_utterance();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // ---- With delta cache (tolerance 16 bits, 64 entries per scope) ----
    group.bench_function("with_delta_cache", |b| {
        b.iter_batched(
            || {
                let coord = PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
                    .with_confidence_threshold(0.05)
                    .with_top_n_events(3);
                StreamingPrefetcher::new(coord, 24).with_delta_cache(16, 64)
            },
            |mut sp| {
                for (state, query) in &workload {
                    sp.prefetch(state);
                    let _ = sp.on_final(*query, 10, 16);
                    sp.end_utterance();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();

    // Summary stats
    let coord = PrefetchCoordinator::new(&idx, fresh_predictor(), scope.clone())
        .with_confidence_threshold(0.05)
        .with_top_n_events(3);
    let mut sp = StreamingPrefetcher::new(coord, 24).with_delta_cache(16, 64);
    for (state, query) in &workload {
        sp.prefetch(state);
        let _ = sp.on_final(*query, 10, 16);
        sp.end_utterance();
    }
    let s = sp.stats();
    let cs = sp.delta_cache_stats().unwrap_or_default();
    println!(
        "\n=== full pipeline summary: utterances={} | served_from_cache={}/{} ({:.1}%) | rescans={} | cache_size={} | cache_hit_rate={:.1}%",
        s.utterances,
        s.finals_served_from_cache,
        s.utterances,
        100.0 * s.finals_served_from_cache as f32 / s.utterances.max(1) as f32,
        s.finals_required_rescan,
        cs.lookups - cs.hits, // rough proxy for cache size
        100.0 * cs.hit_rate(),
    );
}

criterion_group!(full_pipeline_benches, bench_full_pipeline);
criterion_main!(full_pipeline_benches);
