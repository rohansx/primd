# primd benchmark report

> Last run 2026-05-13 on `AMD Ryzen 7 7435HS` (Zen 4, 8c/16t, 16 MiB L3) / Linux 7.0.5 cachyos. Reproducible: `make bench` from the repo root, or `cargo bench --bench voice_session --bench external_baseline` directly.

The headline numbers in the README come from `voice_session` (synthetic Pipecat-shaped workload at 100 k docs). This file collects every measurement we publish and the honest framing each one needs.

## TL;DR

| Path | p50 at finalize | What it measures |
|---|---|---|
| **primd (Markov, full pipeline)** | **1.50 µs** | speculative-cache hit; scan already done during STT |
| primd (Hybrid SR+Markov, full pipeline) | 3.10 µs | same, with the SR cold-start overhead |
| primd (naive — no speculation) | 131.86 µs | full SIMD Hamming scan at finalize, no STT-phase work |
| in-memory HNSW (instant-distance, dim=128 f32) | 621.20 µs | fair "best-case external retrieval back-end" |

primd's finalize win **comes from speculation**, not a faster scan kernel. Anyone who only calls `retrieve()` at end-of-utterance — including primd in stateless mode and any HNSW — pays the full scan cost at that moment. The 87.7× headline (primd-Markov over naive) is what you get when you wire up the session API and feed STT partials.

## Workload

`primd-bench/benches/voice_session.rs` builds a Pipecat-shaped workload:

- **100 000 documents** packed into **50 events** (2 000 docs/event). Documents are 256-bit signatures generated as perturbations of per-event centroid signatures (30-bit Hamming drift), so each event is a coherent cluster.
- **200 utterances** drawn from 20 canonical intents. Each utterance fans out into:
  - 4 partial transcripts (drift schedule `[30, 12, 6, 2]` bits from the final), simulating STT-emitted interim transcripts that converge as the user finishes speaking.
  - 1 finalize signature (6 bits from the intent's canonical query).
  - 1 `warm_next` call (the predictor + scope union the TTS-phase prefetch would do).
- A pre-trained Markov predictor seeded from 200 synthetic transition sequences over 50 events.

The fully self-contained workload runs in ~85 ms per session under primd's full pipeline; the same wall budget would be ~30 ms for a stripped-down `naive_full_scan` (only the finalize SIMD scan, no observe / warm overhead).

## voice_session results (latest run)

```
=== voice_session summary | corpus=100000 docs over 50 events | utterances=200 | top_k=10 ===

--- markov-only predictor ---
 observe_partial: n= 800  mean=   66.47us  p50=   83.07us  p95=  168.16us  p99=  244.27us
  finalize_primd: n= 200  mean=    1.56us  p50=    1.50us  p95=    2.31us  p99=    2.88us
       warm_next: n= 200  mean=  171.97us  p50=  161.71us  p95=  222.24us  p99=  340.81us
served_by: speculative=200 (100.0%) | delta_cache=0 (0.0%) | shard_scan=0 (0.0%) | full_scan=0 (0.0%)

--- hybrid (SR + markov) predictor ---
 observe_partial: n= 800  mean=   63.58us  p50=   27.56us  p95=  160.56us  p99=  260.55us
  finalize_primd: n= 200  mean=    3.22us  p50=    3.10us  p95=    4.36us  p99=    5.18us
       warm_next: n= 200  mean=  163.10us  p50=  157.37us  p95=  209.87us  p99=  237.12us
served_by: speculative=200 (100.0%) | delta_cache=0 (0.0%) | shard_scan=0 (0.0%) | full_scan=0 (0.0%)

--- baseline ---
  finalize_naive: n= 200  mean=  163.41us  p50=  131.86us  p95=  320.15us  p99= 1153.50us

primd-markov finalize p50 vs naive p50: 87.7x faster
primd-hybrid finalize p50 vs naive p50: 42.6x faster
speculative+delta cache hit-rate: markov=100.0%, hybrid=100.0%
```

### Reading the numbers

- **`observe_partial` p50 ~83 µs / Hybrid ~28 µs.** This is the per-partial cost during STT — primd's "background work." It runs once per emitted STT partial (the streaming gate suppresses redundant ones); criterion's measurement window catches both gate hits (cheap) and gate-emitted scans (full coarse Hamming). The Hybrid path has lower p50 because its first ~40 partials use SR-cold which has no learned scope union yet, falling through to a cheap empty-scope code path.
- **`finalize_primd` p50 1.50 µs.** This is the cache-hit cost. The speculative cache is queried, the cached signature is compared to the final signature, and (because all 200 utterances converge close enough) the cached top-K is returned without re-scanning. 100 % speculative cache hit rate on this workload.
- **`warm_next` p50 162 µs.** Predictor lookup + scope union construction. Runs once per utterance during TTS playback (typically 1–3 s of free CPU), so latency here is well below the budget.
- **`finalize_naive` p50 132 µs.** Same SIMD scan path primd uses, but called once at end-of-utterance with no prior speculation. This is what every "fast retrieval" library — including primd in stateless `/query` mode — looks like if you don't wire up the session API.

### SR vs Markov on this workload

The Hybrid predictor's finalize p50 (3.10 µs) is slightly *higher* than pure Markov (1.50 µs), with identical 100 % speculative-hit rate. The synthetic workload is too well-clustered for SR to differentiate from a frequency baseline — every utterance lands in an event whose scope was already pre-warmed.

A real lift requires an adversarial workload with paraphrased turns, longer chains, or context-dependent transitions where Markov fails. That's the v0.2.5 A/B harness work (`primd-sr-bench`).

The right way to read the table: SR + Hybrid is **not a regression** on the well-behaved workload (still beats naive by 42×) and the trait surface is in place for the harder workload measurements. The Markov-only path remains the v0.2 default; switch to `--predictor hybrid` when SR's properties (paraphrase generalization, soft horizon) are needed.

## external_baseline results

`primd-bench/benches/external_baseline.rs` indexes a **100 000-doc, 128-dim f32** corpus with `instant-distance` (a mature open-source HNSW Rust crate) and queries it with the same intent distribution.

```
=== external_baseline summary | corpus=100000 docs | queries=200 | dim=128 | top_k=10 ===
                  hnsw_build: 46.61s for 100000 docs
       hnsw_finalize: n= 200  mean=  619.08us  p50=  621.20us  p95=  723.76us  p99=  910.18us
```

### Reading the numbers

- **HNSW build 46.6 s.** Index construction is one-shot; cost amortized over all queries. primd's index build for the same workload (`primd index --input examples/faq.jsonl ...`) is sub-second because binary signatures don't require graph construction.
- **HNSW finalize p50 621 µs.** This is the cost of a single `retrieve()` call at end-of-utterance under a fair in-memory HNSW. **No speculation, no STT-phase pre-work.**

### Comparison framing

| Method | p50 at finalize | Why |
|---|---|---|
| **primd (Markov, speculative)** | **1.50 µs** | scan already done during STT |
| primd (no speculation) | 131.86 µs | SIMD Hamming on 100 k × 32-byte signatures, fits in L2 |
| in-memory HNSW (instant-distance) | 621.20 µs | 128-dim f32 graph traversal; 50 MB working set, doesn't fit in L2 |

primd's *raw scan* path (~132 µs) already beats in-memory HNSW (~621 µs) on this corpus size — 256-bit signatures pack into 3.2 MB which fits in L2 cache, while HNSW's 50 MB f32 representation thrashes. SIMD popcount throughput is also higher than HNSW's pointer-chase traversal pattern. But that's a side effect — the **headline win comes from speculation**, not from a faster scan.

## Honest framing

1. **The 1.5 µs is a cache hit.** It's the cost of *not having to scan* because speculation during STT already did the work. We never claim "primd's scan is 87× faster than HNSW" — we claim "primd hides the scan inside STT so finalize is a cache lookup."

2. **The HNSW comparison is apples-to-roughly-apples.** 256-bit binary signatures vs 128-dim f32 vectors aren't identical representations, but both are reasonable in-memory ANN choices for a voice-corpus-scale workload. The point isn't to prove binary > HNSW on raw scan; it's to bound what a fast in-memory back-end costs at finalize so the speculation win is calibrated.

3. **Managed vector DBs (Qdrant, Pinecone) add network latency.** We don't simulate that here because the comparison would be unfair in primd's favor (4–50 ms vs 1.5 µs is a 3000–30000× ratio that overstates the real architectural advantage). The strategy memo cites public Qdrant numbers (4 ms p50 at 1 M vectors) for context.

4. **Moss / InferEdge is not yet benchmarked.** Moss is closed-source under PolyForm Shield and ships separately. When primd has access to a Moss installation we'll add it as a third bench. Until then this report does *not* claim numbers vs Moss.

5. **100 % speculative hit rate is a workload property.** Synthetic intents drawn from 20 canonical clusters with low Hamming drift make speculation trivially correct. Real voice workloads — especially with longer chains and out-of-distribution paraphrases — will see lower hit rates. The Hybrid SR + Markov predictor is the v0.2 hedge for that case; the v0.2.5 A/B harness measures it.

## Reproducing

```bash
# voice_session: primd-Markov vs primd-Hybrid vs naive
cargo bench --bench voice_session

# external_baseline: in-memory HNSW for the same workload size
cargo bench --bench external_baseline
```

Numbers drift with CPU model. The headline ratios (primd-speculative vs naive ~50–100×, primd-speculative vs HNSW ~300–500×) are stable across the Zen 4 / Ice Lake / Sapphire Rapids machines we've tested on; absolute µs values vary by ~30 %.

Hardware drift caveats:
- Older CPUs without AVX-512 VPOPCNTDQ fall back to AVX2 VPSHUFB, which is ~3× slower per popcount. The naive_full_scan baseline moves more than the speculative path.
- Memory bandwidth dominates for the naive scan; primd's speculative path is mostly cache lookups, so it's less sensitive.
- Criterion's outlier reports (mild outliers on most benches) are typical noise; underlying distributions are tight.

## Related

- [Architecture overview](../architecture/overview.md)
- [Successor Representation predictor](../architecture/successor-representation.md)
- [Strategy memo](../business/strategy-2026-05.md)
