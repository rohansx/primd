//! Predictor A/B harness — measures Markov-only vs SR-only vs Hybrid on a
//! voice-realistic workload, with windowed cumulative metrics so SR's
//! cold-start ramp is visible.
//!
//! v0.2's tabular SR has a known limitation: it operates over EventIds,
//! not signature features, so the "paraphrase generalization" lift from
//! the strategy memo lives in v0.2.5's low-rank reduction (W: 256×K).
//! On most synthetic voice workloads, tabular SR's top-K ranking
//! converges to Markov-1's top-K ranking — same empirical distribution,
//! different bookkeeping.
//!
//! What this harness IS measuring:
//! - **Hybrid robustness**: does the SR + Markov wrapper degrade
//!   gracefully when SR is cold? It should match Markov-only during
//!   the first ~40 transitions (SR confidence < threshold) and gate to
//!   SR afterwards.
//! - **Cumulative hit-rate over time**: cuts the workload into three
//!   windows (0–200, 200–500, 500–1000 utterances) so SR's ramp-up
//!   shows. If Hybrid's hit-rate climbs while Markov's stays flat, that's
//!   evidence the SR side is doing useful work past cold-start.
//! - **Finalize p50/p95/p99 by predictor**: the speculative cache hit
//!   path is identical across predictors; the difference is which
//!   predictor's scope prediction the cache was populated against.

use std::collections::BTreeMap;
use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use primd_core::embed::binary::BinarySignature;
use primd_core::index::events::EventCatalog;
use primd_core::index::shards::HierarchicalIndex;
use primd_core::index::signatures::SignatureIndex;
use primd_core::predict::{EventId, MarkovPredictor, NextTurnPredictor};
use primd_core::query_context::{QueryContext, ServedBy};
use primd_sr::{HybridPredictor, SrPredictor};

const SEED: u64 = 0xCAFE_F00D;
const VOCAB: u32 = 50;
const SIGS_PER_EVENT: usize = 2_000;
const TOTAL_SIGS: usize = (VOCAB as usize) * SIGS_PER_EVENT;
const N_UTTERANCES: usize = 1_000;
const N_CANONICAL_INTENTS: usize = 20;
const TOP_K: usize = 10;
const PARTIAL_DRIFTS: [u32; 4] = [30, 12, 6, 2];

fn random_sig(rng: &mut StdRng) -> BinarySignature {
    let mut b = [0u8; 32];
    rng.fill(&mut b);
    BinarySignature(b)
}

fn perturb(s: &BinarySignature, rng: &mut StdRng, bits: u32) -> BinarySignature {
    let mut out = *s;
    for _ in 0..bits {
        let bit = rng.random_range(0..256);
        out.0[bit / 8] ^= 1 << (bit % 8);
    }
    out
}

struct Utterance {
    partials: Vec<BinarySignature>,
    final_sig: BinarySignature,
}

fn build_corpus_and_workload() -> (HierarchicalIndex, MarkovPredictor, Vec<Utterance>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut sigs = Vec::with_capacity(TOTAL_SIGS);
    let mut centroids = Vec::with_capacity(VOCAB as usize);
    let mut scope: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for e in 0..VOCAB {
        let centroid = random_sig(&mut rng);
        centroids.push(centroid);
        let start = e as usize * SIGS_PER_EVENT;
        scope.insert(
            format!("event_{e:03}"),
            (start..start + SIGS_PER_EVENT).collect(),
        );
        for _ in 0..SIGS_PER_EVENT {
            sigs.push(perturb(&centroid, &mut rng, 30));
        }
    }
    let index = HierarchicalIndex::new(
        SignatureIndex::new(sigs),
        EventCatalog::from_named_scope(&scope, TOTAL_SIGS),
    );

    // Pre-train Markov from synthetic transition sequences so cold-start
    // behavior is identical across predictor configurations — the only
    // variable is which predictor QueryContext consults.
    let mut prng = StdRng::seed_from_u64(SEED ^ 0x1111);
    let mut sequences: Vec<Vec<EventId>> = Vec::new();
    for _ in 0..200 {
        let len = prng.random_range(8..20);
        let mut seq = Vec::with_capacity(len);
        let mut cur = prng.random_range(0..VOCAB);
        seq.push(EventId(cur));
        for _ in 1..len {
            let r = prng.random_range(0..10);
            cur = match r {
                0..=5 => (cur + 1) % VOCAB,
                6..=7 => (cur + 7) % VOCAB,
                _ => (cur + 13) % VOCAB,
            };
            seq.push(EventId(cur));
        }
        sequences.push(seq);
    }
    let markov = primd_core::predict::trainer::train_sequences(sequences, 3, 0.01);

    let mut wrng = StdRng::seed_from_u64(SEED ^ 0x4444);
    let canonicals: Vec<(EventId, BinarySignature)> = (0..N_CANONICAL_INTENTS)
        .map(|_| {
            let e = wrng.random_range(0..VOCAB);
            let q = perturb(&centroids[e as usize], &mut wrng, 40);
            (EventId(e), q)
        })
        .collect();

    let mut utts = Vec::with_capacity(N_UTTERANCES);
    for _ in 0..N_UTTERANCES {
        let (_, canonical) = canonicals[wrng.random_range(0..canonicals.len())];
        let final_sig = perturb(&canonical, &mut wrng, 6);
        let partials = PARTIAL_DRIFTS
            .iter()
            .map(|&d| perturb(&final_sig, &mut wrng, d))
            .collect();
        utts.push(Utterance {
            partials,
            final_sig,
        });
    }

    (index, markov, utts)
}

#[derive(Default, Clone)]
struct WindowedServed {
    speculative: usize,
    delta_cache: usize,
    shard_scan: usize,
    full_scan: usize,
}

impl WindowedServed {
    fn record(&mut self, s: ServedBy) {
        match s {
            ServedBy::Speculative => self.speculative += 1,
            ServedBy::DeltaCache => self.delta_cache += 1,
            ServedBy::ShardScan => self.shard_scan += 1,
            ServedBy::FullScan => self.full_scan += 1,
        }
    }
    fn total(&self) -> usize {
        self.speculative + self.delta_cache + self.shard_scan + self.full_scan
    }
    fn hit_pct(&self) -> f64 {
        let n = self.total().max(1);
        100.0 * (self.speculative + self.delta_cache) as f64 / n as f64
    }
}

#[derive(Default)]
struct PredictorResult {
    name: &'static str,
    finalize_ns: Vec<u128>,
    served_overall: WindowedServed,
    /// Speculative+delta hit rate computed cumulatively at 3 windows of
    /// the workload so the SR warmup ramp shows up if it exists.
    served_windows: Vec<(usize, WindowedServed)>,
}

impl PredictorResult {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    fn finalize_p(&self, p: usize) -> f64 {
        if self.finalize_ns.is_empty() {
            return 0.0;
        }
        let mut s = self.finalize_ns.clone();
        s.sort_unstable();
        let n = s.len();
        let idx = (n * p / 100).min(n - 1);
        s[idx] as f64 / 1_000.0
    }

    fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:>14}: finalize p50={:.2}us p95={:.2}us p99={:.2}us | cache-hit overall={:.1}%\n",
            self.name,
            self.finalize_p(50),
            self.finalize_p(95),
            self.finalize_p(99),
            self.served_overall.hit_pct(),
        ));
        for (boundary, w) in &self.served_windows {
            out.push_str(&format!(
                "{:>14}    window through n={:4}: hit-rate={:.1}% (spec={} delta={} shard={} full={})\n",
                self.name, boundary, w.hit_pct(), w.speculative, w.delta_cache, w.shard_scan, w.full_scan
            ));
        }
        out
    }
}

const WINDOWS: [usize; 3] = [200, 500, 1000];

fn run_session(
    label: &'static str,
    index: &HierarchicalIndex,
    workload: &[Utterance],
    predictor: Box<dyn NextTurnPredictor>,
) -> PredictorResult {
    let mut ctx = QueryContext::with_boxed_predictor(predictor);
    let mut r = PredictorResult::new(label);
    let mut cum = WindowedServed::default();
    let mut window_idx = 0;

    for (i, utt) in workload.iter().enumerate() {
        for &p in &utt.partials {
            ctx.observe_partial(index, p, TOP_K);
        }
        let t = Instant::now();
        let out = ctx.finalize(index, utt.final_sig, TOP_K);
        r.finalize_ns.push(t.elapsed().as_nanos());
        r.served_overall.record(out.served_by);
        cum.record(out.served_by);
        ctx.warm_next(index);

        if window_idx < WINDOWS.len() && i + 1 == WINDOWS[window_idx] {
            r.served_windows.push((WINDOWS[window_idx], cum.clone()));
            window_idx += 1;
        }
    }
    r
}

fn run_all(index: &HierarchicalIndex, markov: &MarkovPredictor, workload: &[Utterance]) -> Vec<PredictorResult> {
    let markov_only = run_session("markov-only", index, workload, Box::new(markov.clone()));
    let sr_only = run_session(
        "sr-only",
        index,
        workload,
        Box::new(SrPredictor::new().with_warmup(40).with_gamma(0.9)),
    );
    let hybrid = run_session(
        "hybrid(0.5)",
        index,
        workload,
        Box::new(
            HybridPredictor::new(
                SrPredictor::new().with_warmup(40).with_gamma(0.9),
                markov.clone(),
            )
            .with_threshold(0.5),
        ),
    );
    vec![markov_only, sr_only, hybrid]
}

fn bench_predictor_ab(c: &mut Criterion) {
    let (index, markov, workload) = build_corpus_and_workload();

    let mut group = c.benchmark_group("predictor_ab");
    group.sample_size(10);

    group.bench_function("markov_session_1000utts", |b| {
        b.iter(|| {
            let _ = run_session(
                "markov",
                &index,
                &workload,
                Box::new(markov.clone()),
            );
        });
    });

    group.bench_function("hybrid_session_1000utts", |b| {
        b.iter(|| {
            let _ = run_session(
                "hybrid",
                &index,
                &workload,
                Box::new(
                    HybridPredictor::new(
                        SrPredictor::new().with_warmup(40).with_gamma(0.9),
                        markov.clone(),
                    )
                    .with_threshold(0.5),
                ),
            );
        });
    });

    group.finish();

    // Human-readable comparison for the bench report.
    let results = run_all(&index, &markov, &workload);
    println!();
    println!(
        "=== predictor_ab summary | corpus={} docs over {} events | utterances={} | top_k={} ===",
        TOTAL_SIGS, VOCAB, N_UTTERANCES, TOP_K
    );
    for r in &results {
        print!("{}", r.report());
    }

    // Hybrid robustness assertion (informational, not a hard failure in bench):
    let markov_hit = results[0].served_overall.hit_pct();
    let hybrid_hit = results[2].served_overall.hit_pct();
    let regression = markov_hit - hybrid_hit;
    println!();
    println!(
        "Hybrid robustness: markov_hit={:.1}% hybrid_hit={:.1}% (regression={:.1}pp)",
        markov_hit, hybrid_hit, regression
    );
    println!(
        "(target: hybrid >= markov - 2pp; tabular SR matches Markov on this workload, \
         lift requires v0.2.5 low-rank signature-aware SR)"
    );
}

criterion_group!(predictor_ab_benches, bench_predictor_ab);
criterion_main!(predictor_ab_benches);
