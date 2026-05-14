//! Paraphrase-aware A/B bench — the workload that should differentiate
//! signature-feature-aware low-rank SR from EventId-based Markov.
//!
//! Workload design:
//!
//! - 10 **topic clusters**, each with 10 paraphrase EventIds = 100 events.
//! - Each topic has a random centroid signature; within-topic events differ
//!   by 10 bits of Hamming drift (LSH-close, but distinct EventIds).
//! - Conversations follow a Markov chain over **topics**, not events:
//!   topic A transitions to topic B according to a learned bigram. Within a
//!   chosen topic, the specific paraphrase is uniformly random.
//! - 1 000 utterances.
//!
//! Why this differentiates Markov-1 from signature-feature SR:
//!
//! - Markov-1 over EventIds sees each `(event_in_A_i → event_in_B_j)`
//!   transition with mass `1/(10·10) = 1 %` of the topic A→B mass. After
//!   200 utterances the matrix is sparse and noisy.
//! - Low-rank SR's `ψ(e)` for events in the same topic are close in
//!   feature space, so `M_low` pools observations across paraphrases.
//!   Effectively needs ~10× fewer transitions to converge.
//!
//! Whether this translates into a *speculative-cache hit-rate lift* depends
//! on how the speculation pipeline composes scope unions. The bench reports
//! the empirical result honestly — either way it's a useful measurement,
//! because the synthetic-workload ceiling (`predictor_ab`) doesn't move
//! and v0.2.6 needs a real differentiation signal.

use std::collections::{BTreeMap, HashMap};
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
use primd_sr::{HybridPredictor, LowRankSrPredictor, SrPredictor};

const SEED: u64 = 0xDEAD_BEEF;
const N_TOPICS: u32 = 10;
const EVENTS_PER_TOPIC: u32 = 10;
const VOCAB: u32 = N_TOPICS * EVENTS_PER_TOPIC; // 100 events
const SIGS_PER_EVENT: usize = 100; // 10k total docs
const TOTAL_SIGS: usize = (VOCAB as usize) * SIGS_PER_EVENT;
const N_UTTERANCES: usize = 1_000;
const WITHIN_TOPIC_DRIFT: u32 = 10;
const WITHIN_EVENT_DRIFT: u32 = 20;
const QUERY_DRIFT: u32 = 6;
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
    target_topic: u32,
}

/// Map a global doc index back to its topic. Since the corpus is laid out
/// as `[topic_0_event_0_docs..., topic_0_event_1_docs..., ...]` with
/// `SIGS_PER_EVENT` docs per event and `EVENTS_PER_TOPIC` events per
/// topic, the topic of a doc is `doc_idx / (SIGS_PER_EVENT * EVENTS_PER_TOPIC)`.
fn topic_of_doc(doc_idx: usize) -> u32 {
    (doc_idx / (SIGS_PER_EVENT * EVENTS_PER_TOPIC as usize)) as u32
}

fn build_corpus_and_workload() -> (
    HierarchicalIndex,
    MarkovPredictor,
    Vec<Utterance>,
    HashMap<EventId, BinarySignature>,
) {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Random topic centroids; events within a topic are 10-bit drifts.
    let mut topic_centroids: Vec<BinarySignature> = (0..N_TOPICS)
        .map(|_| random_sig(&mut rng))
        .collect();
    // Slight noise on topic centroids so the within-topic drift isn't
    // collapsed back to identical centroids via repeated perturbation.
    for c in &mut topic_centroids {
        *c = perturb(c, &mut rng, 0);
    }

    let mut event_centroids: Vec<BinarySignature> = Vec::with_capacity(VOCAB as usize);
    for t in 0..N_TOPICS {
        for _ in 0..EVENTS_PER_TOPIC {
            event_centroids.push(perturb(&topic_centroids[t as usize], &mut rng, WITHIN_TOPIC_DRIFT));
        }
    }

    // Build the doc corpus by perturbing each event's centroid further.
    let mut sigs = Vec::with_capacity(TOTAL_SIGS);
    let mut scope: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (event_idx, centroid) in event_centroids.iter().enumerate() {
        let start = event_idx * SIGS_PER_EVENT;
        scope.insert(
            format!("event_{event_idx:03}"),
            (start..start + SIGS_PER_EVENT).collect(),
        );
        for _ in 0..SIGS_PER_EVENT {
            sigs.push(perturb(centroid, &mut rng, WITHIN_EVENT_DRIFT));
        }
    }
    let index = HierarchicalIndex::new(
        SignatureIndex::new(sigs),
        EventCatalog::from_named_scope(&scope, TOTAL_SIGS),
    );

    // Pre-train Markov from synthetic topic-level transition sequences.
    // The training data is generated at TOPIC level (not event level), so
    // Markov sees the topic structure; what it learns over EventIds is
    // sparse because within a topic the specific event is uniformly random.
    let mut prng = StdRng::seed_from_u64(SEED ^ 0x1111);
    let mut sequences: Vec<Vec<EventId>> = Vec::new();
    for _ in 0..200 {
        let len = prng.random_range(8..20);
        let mut seq = Vec::with_capacity(len);
        let mut cur_topic = prng.random_range(0..N_TOPICS);
        for _ in 0..len {
            // Pick random event within current topic.
            let event_in_topic = prng.random_range(0..EVENTS_PER_TOPIC);
            seq.push(EventId(cur_topic * EVENTS_PER_TOPIC + event_in_topic));
            // Topic transitions: bigram-ish.
            let r = prng.random_range(0..10);
            cur_topic = match r {
                0..=5 => (cur_topic + 1) % N_TOPICS,
                6..=7 => (cur_topic + 3) % N_TOPICS,
                _ => (cur_topic + 7) % N_TOPICS,
            };
        }
        sequences.push(seq);
    }
    let markov = primd_core::predict::trainer::train_sequences(sequences, 3, 0.01);

    // Workload: utterances drawn from the same topic-Markov distribution.
    let mut wrng = StdRng::seed_from_u64(SEED ^ 0x4444);
    let mut utts = Vec::with_capacity(N_UTTERANCES);
    let mut cur_topic = wrng.random_range(0..N_TOPICS);
    for _ in 0..N_UTTERANCES {
        let event_in_topic = wrng.random_range(0..EVENTS_PER_TOPIC);
        let event_idx = (cur_topic * EVENTS_PER_TOPIC + event_in_topic) as usize;
        let final_sig = perturb(&event_centroids[event_idx], &mut wrng, QUERY_DRIFT);
        let partials = PARTIAL_DRIFTS
            .iter()
            .map(|&d| perturb(&final_sig, &mut wrng, d))
            .collect();
        utts.push(Utterance {
            partials,
            final_sig,
            target_topic: cur_topic,
        });
        let r = wrng.random_range(0..10);
        cur_topic = match r {
            0..=5 => (cur_topic + 1) % N_TOPICS,
            6..=7 => (cur_topic + 3) % N_TOPICS,
            _ => (cur_topic + 7) % N_TOPICS,
        };
    }

    let centroids_map: HashMap<EventId, BinarySignature> = event_centroids
        .into_iter()
        .enumerate()
        .map(|(i, sig)| (EventId(i as u32), sig))
        .collect();

    (index, markov, utts, centroids_map)
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
    served_windows: Vec<(usize, WindowedServed)>,
    /// Count of utterances where the top-1 hit's source doc is in the
    /// workload's intended topic. The content-quality metric the simple
    /// cache-hit rate hides.
    top1_topic_correct: usize,
    /// Count of utterances where ANY of the top-K hits' source docs are
    /// in the workload's intended topic. More lenient than top-1.
    topk_topic_present: usize,
    total_utterances: usize,
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
    fn top1_pct(&self) -> f64 {
        let n = self.total_utterances.max(1);
        100.0 * self.top1_topic_correct as f64 / n as f64
    }
    fn topk_pct(&self) -> f64 {
        let n = self.total_utterances.max(1);
        100.0 * self.topk_topic_present as f64 / n as f64
    }
    fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:>14}: finalize p50={:.2}us p95={:.2}us p99={:.2}us | cache-hit={:.1}% | top1-topic={:.1}% | top10-topic={:.1}%\n",
            self.name,
            self.finalize_p(50),
            self.finalize_p(95),
            self.finalize_p(99),
            self.served_overall.hit_pct(),
            self.top1_pct(),
            self.topk_pct(),
        ));
        for (boundary, w) in &self.served_windows {
            out.push_str(&format!(
                "{:>14}    window through n={:4}: cache-hit={:.1}% (spec={} delta={} shard={} full={})\n",
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
    let mut wi = 0;
    for (i, utt) in workload.iter().enumerate() {
        for &p in &utt.partials {
            ctx.observe_partial(index, p, TOP_K);
        }
        let t = Instant::now();
        let out = ctx.finalize(index, utt.final_sig, TOP_K);
        r.finalize_ns.push(t.elapsed().as_nanos());
        r.served_overall.record(out.served_by);
        cum.record(out.served_by);
        r.total_utterances += 1;

        // Content quality: did the top-1 hit's source doc come from the
        // intended topic? top-k presence is the more lenient version.
        if let Some(&(_, doc_idx)) = out.results.first()
            && topic_of_doc(doc_idx) == utt.target_topic
        {
            r.top1_topic_correct += 1;
        }
        if out
            .results
            .iter()
            .any(|&(_, doc_idx)| topic_of_doc(doc_idx) == utt.target_topic)
        {
            r.topk_topic_present += 1;
        }

        ctx.warm_next(index);
        if wi < WINDOWS.len() && i + 1 == WINDOWS[wi] {
            r.served_windows.push((WINDOWS[wi], cum.clone()));
            wi += 1;
        }
    }
    r
}

fn bench_paraphrase_ab(c: &mut Criterion) {
    let (index, markov, workload, centroids) = build_corpus_and_workload();

    eprintln!(
        "paraphrase workload: {} topics × {} events/topic = {} events; corpus={} docs",
        N_TOPICS, EVENTS_PER_TOPIC, VOCAB, TOTAL_SIGS
    );
    eprintln!(
        "within-topic drift {} bits, within-event drift {} bits, query drift {} bits",
        WITHIN_TOPIC_DRIFT, WITHIN_EVENT_DRIFT, QUERY_DRIFT
    );

    let mut group = c.benchmark_group("paraphrase_ab");
    group.sample_size(10);

    group.bench_function("markov", |b| {
        b.iter(|| {
            let _ = run_session("markov", &index, &workload, Box::new(markov.clone()));
        });
    });

    group.bench_function("low_rank_K32", |b| {
        b.iter(|| {
            let _ = run_session(
                "low-rank-K32",
                &index,
                &workload,
                Box::new(
                    LowRankSrPredictor::<32>::new(&centroids)
                        .with_warmup(40)
                        .with_gamma(0.9),
                ),
            );
        });
    });

    group.bench_function("low_rank_K64", |b| {
        b.iter(|| {
            let _ = run_session(
                "low-rank-K64",
                &index,
                &workload,
                Box::new(
                    LowRankSrPredictor::<64>::new(&centroids)
                        .with_warmup(40)
                        .with_gamma(0.9),
                ),
            );
        });
    });

    group.bench_function("low_rank_K128", |b| {
        b.iter(|| {
            let _ = run_session(
                "low-rank-K128",
                &index,
                &workload,
                Box::new(
                    LowRankSrPredictor::<128>::new(&centroids)
                        .with_warmup(40)
                        .with_gamma(0.9),
                ),
            );
        });
    });

    group.bench_function("hybrid_low_rank", |b| {
        b.iter(|| {
            let _ = run_session(
                "hybrid-LR",
                &index,
                &workload,
                Box::new(HybridPredictor::new(
                    SrPredictor::new().with_warmup(40).with_gamma(0.9),
                    markov.clone(),
                )),
            );
        });
    });

    group.finish();

    // Single comparison pass for the human-readable report. Run all four
    // predictors; the same RNG seeds make the comparison deterministic.
    let r_markov = run_session("markov", &index, &workload, Box::new(markov.clone()));
    let r_sr_tab = run_session(
        "sr-tabular",
        &index,
        &workload,
        Box::new(SrPredictor::new().with_warmup(40).with_gamma(0.9)),
    );
    let r_lr32 = run_session(
        "low-rank-K32",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<32>::new(&centroids)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    let r_lr64 = run_session(
        "low-rank-K64",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<64>::new(&centroids)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    let r_lr128 = run_session(
        "low-rank-K128",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<128>::new(&centroids)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    // PCA-projection variants at the K-sweep winner (K=64) and K=32 for
    // a fair head-to-head with the random projection at each K.
    let r_lr32_pca = run_session(
        "low-rank-K32-PCA",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<32>::with_pca(&centroids)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    let r_lr64_pca = run_session(
        "low-rank-K64-PCA",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<64>::with_pca(&centroids)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    // v0.2.8: PCA trained over the full doc corpus instead of just
    // event centroids. Tests the hypothesis that within-topic variation
    // in the training data closes the chance-level regression.
    let corpus_sigs: Vec<BinarySignature> = index.signatures().as_slice().to_vec();
    let r_lr32_pca_corpus = run_session(
        "low-rank-K32-PCA-corpus",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<32>::with_pca_over_corpus(&centroids, &corpus_sigs)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    let r_lr64_pca_corpus = run_session(
        "low-rank-K64-PCA-corpus",
        &index,
        &workload,
        Box::new(
            LowRankSrPredictor::<64>::with_pca_over_corpus(&centroids, &corpus_sigs)
                .with_warmup(40)
                .with_gamma(0.9),
        ),
    );
    let r_hybrid = run_session(
        "hybrid-LR(0.5)",
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

    println!();
    println!(
        "=== paraphrase_ab summary | corpus={} docs over {} events ({} topics × {}) | utterances={} | top_k={} ===",
        TOTAL_SIGS, VOCAB, N_TOPICS, EVENTS_PER_TOPIC, N_UTTERANCES, TOP_K
    );
    for r in [
        &r_markov,
        &r_sr_tab,
        &r_lr32,
        &r_lr64,
        &r_lr128,
        &r_lr32_pca,
        &r_lr64_pca,
        &r_lr32_pca_corpus,
        &r_lr64_pca_corpus,
        &r_hybrid,
    ] {
        print!("{}", r.report());
    }

    println!();
    println!("Top-1 topic correctness — the bench's primary content-quality metric:");
    println!("  markov:                  {:.1}%", r_markov.top1_pct());
    println!("  sr-tabular:              {:.1}%", r_sr_tab.top1_pct());
    println!("  low-rank-K32:            {:.1}%", r_lr32.top1_pct());
    println!("  low-rank-K64:            {:.1}%", r_lr64.top1_pct());
    println!("  low-rank-K128:           {:.1}%", r_lr128.top1_pct());
    println!("  low-rank-K32-PCA:        {:.1}%", r_lr32_pca.top1_pct());
    println!("  low-rank-K64-PCA:        {:.1}%", r_lr64_pca.top1_pct());
    println!(
        "  low-rank-K32-PCA-corpus: {:.1}%  ← v0.2.8 PCA over docs",
        r_lr32_pca_corpus.top1_pct()
    );
    println!(
        "  low-rank-K64-PCA-corpus: {:.1}%",
        r_lr64_pca_corpus.top1_pct()
    );
    println!("  hybrid-LR:               {:.1}%", r_hybrid.top1_pct());

    let markov_pct = r_markov.top1_pct();

    println!();
    println!("PCA vs random projection (same K, same workload):");
    println!(
        "  K=32:   random={:.1}%  pca={:.1}%  delta={:+.1}pp",
        r_lr32.top1_pct(),
        r_lr32_pca.top1_pct(),
        r_lr32_pca.top1_pct() - r_lr32.top1_pct()
    );
    println!(
        "  K=64:   random={:.1}%  pca={:.1}%  delta={:+.1}pp",
        r_lr64.top1_pct(),
        r_lr64_pca.top1_pct(),
        r_lr64_pca.top1_pct() - r_lr64.top1_pct()
    );

    let lr_best = [
        ("K=32 random", r_lr32.top1_pct()),
        ("K=64 random", r_lr64.top1_pct()),
        ("K=128 random", r_lr128.top1_pct()),
        ("K=32 PCA", r_lr32_pca.top1_pct()),
        ("K=64 PCA", r_lr64_pca.top1_pct()),
    ]
    .iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    .copied()
    .unwrap_or(("?", 0.0));
    let delta_vs_markov = lr_best.1 - markov_pct;
    println!();
    println!(
        "Best low-rank variant: {} at {:.1}% top-1 (vs Markov {:.1}%; delta {:+.1}pp)",
        lr_best.0, lr_best.1, markov_pct, delta_vs_markov,
    );
    if delta_vs_markov >= 5.0 {
        println!("✅ Low-rank SR beats Markov — ship as v0.2.6 default predictor.");
    } else if delta_vs_markov >= -5.0 {
        println!("Within 5pp of Markov — Hybrid wrapper deploys SR safely; consider per-corpus tuning.");
    } else {
        println!(
            "Still significantly below Markov — pivot to real production-conversation A/B \
             (depends on partnership)."
        );
    }
}

criterion_group!(paraphrase_ab_benches, bench_paraphrase_ab);
criterion_main!(paraphrase_ab_benches);
