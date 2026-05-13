//! External-baseline benchmark: same voice workload, but the retrieval
//! back-end is an in-memory HNSW (via `instant-distance`) instead of primd.
//!
//! This is the closest fair head-to-head comparison primd can publish
//! without depending on a managed vector DB. `instant-distance` is a
//! mature open-source HNSW implementation written by Djc Crickett (the
//! author of `quinn` and `rustls`), used in production retrieval systems.
//!
//! The fundamental honest framing:
//!
//! * primd's finalize p50 of ~1.6 µs is a **cache hit** — the work was
//!   already done during `observe_partial` in the STT phase. Anyone who
//!   only calls `retrieve()` at end-of-utterance, including any HNSW,
//!   pays the full scan cost at that moment.
//! * What this bench measures is the cost-at-finalize of a *cold* HNSW
//!   call — what a Pipecat / LiveKit / Vapi pipeline would pay per turn
//!   if it integrated an in-memory HNSW instead of primd.
//! * Compare to `voice_session`'s `finalize_naive` (full SIMD scan at
//!   finalize, no speculation) for the no-prediction-anywhere baseline.

use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use instant_distance::{Builder as HnswBuilder, Point, Search};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SEED: u64 = 0xCAFE_F00D;
const DIM: usize = 128;
const N_DOCS: usize = 100_000;
const N_QUERIES: usize = 200;
const TOP_K: usize = 10;

/// A 128-dim f32 point, L2-normalized. Wraps `Vec<f32>` so we can
/// implement `instant_distance::Point`.
#[derive(Clone)]
struct EmbeddingPoint(Vec<f32>);

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Squared L2. instant-distance only requires monotonicity, so the
        // squared form (no sqrt) is correct and ~30% cheaper.
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }
}

fn random_embedding(rng: &mut StdRng) -> EmbeddingPoint {
    let mut v = vec![0.0f32; DIM];
    let mut norm = 0.0f32;
    for x in v.iter_mut() {
        let raw: f32 = rng.random_range(-1.0..1.0);
        *x = raw;
        norm += raw * raw;
    }
    norm = norm.sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= norm;
    }
    EmbeddingPoint(v)
}

fn perturb(p: &EmbeddingPoint, rng: &mut StdRng, sigma: f32) -> EmbeddingPoint {
    let mut v = p.0.clone();
    let mut norm = 0.0f32;
    for x in v.iter_mut() {
        let n: f32 = rng.random_range(-sigma..sigma);
        *x += n;
        norm += *x * *x;
    }
    norm = norm.sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= norm;
    }
    EmbeddingPoint(v)
}

fn build_corpus_and_queries() -> (Vec<EmbeddingPoint>, Vec<EmbeddingPoint>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Build 50 centroid topics with 2000 docs each, mirroring voice_session's
    // event structure so the query distribution is comparable.
    let n_centroids = 50;
    let docs_per = N_DOCS / n_centroids;
    let centroids: Vec<EmbeddingPoint> = (0..n_centroids).map(|_| random_embedding(&mut rng)).collect();
    let mut docs = Vec::with_capacity(N_DOCS);
    for c in &centroids {
        for _ in 0..docs_per {
            docs.push(perturb(c, &mut rng, 0.08));
        }
    }

    // Queries drift around random centroids, simulating user utterances.
    let mut queries = Vec::with_capacity(N_QUERIES);
    let mut qrng = StdRng::seed_from_u64(SEED ^ 0x4444);
    for _ in 0..N_QUERIES {
        let c = &centroids[qrng.random_range(0..centroids.len())];
        queries.push(perturb(c, &mut qrng, 0.04));
    }
    (docs, queries)
}

#[derive(Default)]
struct PhaseStats {
    samples: Vec<u128>,
}

impl PhaseStats {
    fn record(&mut self, d: Duration) {
        self.samples.push(d.as_nanos());
    }
    fn report(&self, name: &str) -> String {
        if self.samples.is_empty() {
            return format!("{name:>20}: no samples");
        }
        let mut s = self.samples.clone();
        s.sort_unstable();
        let n = s.len();
        let to_us = |ns: u128| ns as f64 / 1_000.0;
        format!(
            "{name:>20}: n={n:>4}  mean={:>8.2}us  p50={:>8.2}us  p95={:>8.2}us  p99={:>8.2}us",
            to_us(s.iter().sum::<u128>() / n as u128),
            to_us(s[n / 2]),
            to_us(s[(n * 95 / 100).min(n - 1)]),
            to_us(s[(n * 99 / 100).min(n - 1)]),
        )
    }
}

fn bench_external(c: &mut Criterion) {
    eprintln!("building {}-doc HNSW…", N_DOCS);
    let (docs, queries) = build_corpus_and_queries();
    let build_start = Instant::now();
    let map: instant_distance::HnswMap<EmbeddingPoint, usize> =
        HnswBuilder::default()
            .ef_construction(200)
            .ef_search(64)
            .build(docs.clone(), (0..docs.len()).collect());
    let build_elapsed = build_start.elapsed();
    eprintln!(
        "HNSW build done: {} docs in {:.1}s",
        N_DOCS,
        build_elapsed.as_secs_f64()
    );

    let mut group = c.benchmark_group("external_baseline");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("instant_distance_hnsw_finalize", |b| {
        let mut search = Search::default();
        b.iter(|| {
            for q in &queries {
                let _ = map
                    .search(q, &mut search)
                    .take(TOP_K)
                    .map(|item| (item.distance, item.value))
                    .collect::<Vec<_>>();
            }
        });
    });

    group.finish();

    // Per-query stats for the bench report.
    let mut search = Search::default();
    let mut stats = PhaseStats::default();
    for q in &queries {
        let t = Instant::now();
        let _ = map
            .search(q, &mut search)
            .take(TOP_K)
            .map(|item| (item.distance, item.value))
            .collect::<Vec<_>>();
        stats.record(t.elapsed());
    }
    println!();
    println!(
        "=== external_baseline summary | corpus={} docs | queries={} | dim={} | top_k={} ===",
        N_DOCS, N_QUERIES, DIM, TOP_K
    );
    println!(
        "                  hnsw_build: {:.2}s for {} docs",
        build_elapsed.as_secs_f64(),
        N_DOCS
    );
    println!("{}", stats.report("hnsw_finalize"));
    println!(
        "(compare to voice_session's finalize_primd p50 in the same harness; primd's win at \
         finalize comes from speculation having pre-completed the scan during STT, not from a \
         faster scan kernel)"
    );
}

criterion_group!(external_benches, bench_external);
criterion_main!(external_benches);
