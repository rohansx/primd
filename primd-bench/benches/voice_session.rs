//! Voice-realistic session benchmark.
//!
//! Models a Pipecat-shaped voice agent loop end-to-end at the
//! QueryContext layer (the API users actually call):
//!
//!   for each utterance:
//!     - 4 partial transcript signatures (drift toward the final)
//!     - 1 finalize
//!     - 1 warm_next (prefetch during TTS)
//!
//! For each phase we capture per-call latency and emit p50/p95/mean
//! plus the served_by distribution, so the README can cite real
//! numbers for observe / finalize / warm separately. We also run a
//! naive baseline (full scan on every finalize, no prediction, no
//! cache) so the table can show the speedup primd actually gets on
//! a voice workload — not just SIMD scan latency in isolation.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use primd_core::embed::binary::BinarySignature;
use primd_core::index::events::EventCatalog;
use primd_core::index::shards::{HierarchicalIndex, SearchOptions};
use primd_core::index::signatures::SignatureIndex;
use primd_core::predict::{EventId, MarkovPredictor};
use primd_core::query_context::{QueryContext, ServedBy};

const SEED: u64 = 0xCAFE_F00D;
const VOCAB: u32 = 50;
const SIGS_PER_EVENT: usize = 2_000;
const TOTAL_SIGS: usize = (VOCAB as usize) * SIGS_PER_EVENT;
const N_UTTERANCES: usize = 200;
const N_CANONICAL_INTENTS: usize = 20;
const TOP_K: usize = 10;
const PARTIAL_DRIFTS: [u32; 4] = [30, 12, 6, 2];

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

struct Utterance {
    partials: Vec<BinarySignature>,
    final_sig: BinarySignature,
    target_event: EventId,
}

fn build_corpus_and_workload() -> (HierarchicalIndex, MarkovPredictor, Vec<Utterance>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut sigs = Vec::with_capacity(TOTAL_SIGS);
    let mut centroids = Vec::with_capacity(VOCAB as usize);
    let mut named_scope: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for event in 0..VOCAB {
        let centroid = random_sig(&mut rng);
        centroids.push(centroid);
        let start = event as usize * SIGS_PER_EVENT;
        named_scope.insert(
            format!("event_{event:03}"),
            (start..start + SIGS_PER_EVENT).collect(),
        );
        for _ in 0..SIGS_PER_EVENT {
            sigs.push(perturb(&centroid, &mut rng, 30));
        }
    }
    let signatures = SignatureIndex::new(sigs);
    let events = EventCatalog::from_named_scope(&named_scope, TOTAL_SIGS);
    let index = HierarchicalIndex::new(signatures, events);

    let mut predictor_rng = StdRng::seed_from_u64(SEED ^ 0x1111);
    let mut sequences: Vec<Vec<EventId>> = Vec::new();
    for _ in 0..200 {
        let len = predictor_rng.random_range(8..20);
        let mut seq = Vec::with_capacity(len);
        let mut cur = predictor_rng.random_range(0..VOCAB);
        seq.push(EventId(cur));
        for _ in 1..len {
            let r = predictor_rng.random_range(0..10);
            cur = match r {
                0..=5 => (cur + 1) % VOCAB,
                6..=7 => (cur + 7) % VOCAB,
                _ => (cur + 13) % VOCAB,
            };
            seq.push(EventId(cur));
        }
        sequences.push(seq);
    }
    let predictor = primd_core::predict::trainer::train_sequences(sequences, 3, 0.01);

    let mut work_rng = StdRng::seed_from_u64(SEED ^ 0x4444);

    let canonicals: Vec<(EventId, BinarySignature)> = (0..N_CANONICAL_INTENTS)
        .map(|_| {
            let e = work_rng.random_range(0..VOCAB);
            let q = perturb(&centroids[e as usize], &mut work_rng, 40);
            (EventId(e), q)
        })
        .collect();

    let mut utterances = Vec::with_capacity(N_UTTERANCES);
    for _ in 0..N_UTTERANCES {
        let (event, canonical) = canonicals[work_rng.random_range(0..canonicals.len())];
        let final_sig = perturb(&canonical, &mut work_rng, 6);
        let partials = PARTIAL_DRIFTS
            .iter()
            .map(|&drift| perturb(&final_sig, &mut work_rng, drift))
            .collect();
        utterances.push(Utterance {
            partials,
            final_sig,
            target_event: event,
        });
    }
    (index, predictor, utterances)
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
            return format!("{name:>16}: no samples");
        }
        let mut s = self.samples.clone();
        s.sort_unstable();
        let n = s.len();
        let p50 = s[n / 2];
        let p95 = s[(n * 95 / 100).min(n - 1)];
        let p99 = s[(n * 99 / 100).min(n - 1)];
        let mean = s.iter().sum::<u128>() / n as u128;
        let to_us = |ns: u128| ns as f64 / 1_000.0;
        format!(
            "{name:>16}: n={n:>4}  mean={:>8.2}us  p50={:>8.2}us  p95={:>8.2}us  p99={:>8.2}us",
            to_us(mean),
            to_us(p50),
            to_us(p95),
            to_us(p99),
        )
    }
}

#[derive(Default, Debug)]
struct ServedTally {
    speculative: usize,
    delta_cache: usize,
    shard_scan: usize,
    full_scan: usize,
}

impl ServedTally {
    fn record(&mut self, served: ServedBy) {
        match served {
            ServedBy::Speculative => self.speculative += 1,
            ServedBy::DeltaCache => self.delta_cache += 1,
            ServedBy::ShardScan => self.shard_scan += 1,
            ServedBy::FullScan => self.full_scan += 1,
        }
    }

    fn total(&self) -> usize {
        self.speculative + self.delta_cache + self.shard_scan + self.full_scan
    }

    fn report(&self) -> String {
        let n = self.total().max(1);
        let pct = |k: usize| 100.0 * k as f64 / n as f64;
        format!(
            "served_by: speculative={} ({:.1}%) | delta_cache={} ({:.1}%) | shard_scan={} ({:.1}%) | full_scan={} ({:.1}%)",
            self.speculative,
            pct(self.speculative),
            self.delta_cache,
            pct(self.delta_cache),
            self.shard_scan,
            pct(self.shard_scan),
            self.full_scan,
            pct(self.full_scan),
        )
    }
}

fn run_primd_session(
    index: &HierarchicalIndex,
    predictor: &MarkovPredictor,
    workload: &[Utterance],
) -> (PhaseStats, PhaseStats, PhaseStats, ServedTally) {
    let mut ctx = QueryContext::with_predictor(predictor.clone());
    let mut observe = PhaseStats::default();
    let mut finalize = PhaseStats::default();
    let mut warm = PhaseStats::default();
    let mut served = ServedTally::default();

    for utt in workload {
        for &partial in &utt.partials {
            let t = Instant::now();
            ctx.observe_partial(index, partial, TOP_K);
            observe.record(t.elapsed());
        }
        let t = Instant::now();
        let out = ctx.finalize(index, utt.final_sig, TOP_K);
        finalize.record(t.elapsed());
        served.record(out.served_by);
        let _ = utt.target_event; // workload provides ground truth for future quality stats
        let t = Instant::now();
        ctx.warm_next(index);
        warm.record(t.elapsed());
    }
    (observe, finalize, warm, served)
}

fn run_naive_session(index: &HierarchicalIndex, workload: &[Utterance]) -> PhaseStats {
    let opts = SearchOptions::default();
    let mut finalize = PhaseStats::default();
    for utt in workload {
        let t = Instant::now();
        let _ = index.search(&utt.final_sig, TOP_K, &opts);
        finalize.record(t.elapsed());
    }
    finalize
}

fn bench_voice_session(c: &mut Criterion) {
    let (index, predictor, workload) = build_corpus_and_workload();

    let mut group = c.benchmark_group("voice_session");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("primd_full_pipeline", |b| {
        b.iter(|| {
            let _ = run_primd_session(&index, &predictor, &workload);
        });
    });

    group.bench_function("naive_full_scan", |b| {
        b.iter(|| {
            let _ = run_naive_session(&index, &workload);
        });
    });

    group.finish();

    // Standalone summary run for human-readable numbers in CI logs / README.
    let (obs, fin, warm, served) = run_primd_session(&index, &predictor, &workload);
    let naive_fin = run_naive_session(&index, &workload);

    println!();
    println!(
        "=== voice_session summary | corpus={} docs over {} events | utterances={} | top_k={} ===",
        TOTAL_SIGS, VOCAB, N_UTTERANCES, TOP_K
    );
    println!("{}", obs.report("observe_partial"));
    println!("{}", fin.report("finalize_primd"));
    println!("{}", warm.report("warm_next"));
    println!("{}", naive_fin.report("finalize_naive"));
    println!("{}", served.report());

    if let (Some(p_med), Some(n_med)) = (median(&fin.samples), median(&naive_fin.samples)) {
        let speedup = n_med as f64 / p_med.max(1) as f64;
        println!("primd finalize p50 vs naive p50: {:.1}x faster", speedup);
    }
}

fn median(samples: &[u128]) -> Option<u128> {
    if samples.is_empty() {
        return None;
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    Some(s[s.len() / 2])
}

criterion_group!(voice_session_benches, bench_voice_session);
criterion_main!(voice_session_benches);
