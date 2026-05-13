# Architecture Overview

primd is a four-layer retrieval runtime that trades compute-at-query-time for compute-during-idle-time. Each layer maps to a specific brain mechanism and eliminates a specific piece of latency.

## Design Principles

1. **Start earlier, not faster.** The dominant latency win comes from beginning retrieval before the user finishes speaking, not from a faster search algorithm.
2. **Do less work on the hot path.** Most queries are follow-ups or predicted pivots. Only novel queries pay the full retrieval cost.
3. **No LLM in the critical path.** Prediction uses a sparse Markov matrix (~10 microseconds), not a language model (~200-500ms).
4. **Co-located, not networked.** primd runs in-process with the voice agent. Zero network hops.
5. **Degrade gracefully.** Every layer is optional. If the Markov predictor misses, layer 2 handles it. If binary scan misses, full HNSW handles it. The system never fails — it just gets slightly slower.

## System Context

```
┌─────────────────────────────────────────────────────┐
│                   Voice Agent                        │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ STT      │──▶│ primd    │──▶│ LLM              │ │
│  │ (stream) │   │ (retrieve│   │ (generate)        │ │
│  │          │   │  + warm) │   │                   │ │
│  └──────────┘   └──────────┘   └──────┬───────────┘ │
│                       ▲                │             │
│                       │                ▼             │
│                       │          ┌──────────┐        │
│                       └──────────│ TTS      │        │
│                     warm_next()  │ (speak)  │        │
│                     during TTS   └──────────┘        │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Knowledge Base   │
│ (Qdrant/pgvector │
│  /parquet)       │
└─────────────────┘
```

primd sits between STT and LLM. It receives partial transcripts from the STT stream, retrieves relevant context, and feeds it to the LLM for generation. During TTS playback (3-10 seconds of free CPU time), it prefetches context for the predicted next question.

## The Four Layers

### Layer 1 — Streaming Partial-Query Retrieval

**Brain analogue**: Auditory cortex + predictive coding hierarchy (Caucheteux et al., 2023).

**Purpose**: Start retrieving before the user finishes speaking.

Traditional systems wait for end-of-utterance → embed → search. primd consumes partial transcripts from the STT stream and fires speculative retrievals every 2-3 tokens:

```
"so"                    → speculative retrieve (broad)
"so what about"         → speculative retrieve (narrowing)
"so what about pricing" → final retrieve (confirm or refine)
```

A lightweight incremental embedder produces rough embeddings from partials. These are used ONLY for candidate pre-selection (top-100 narrowing), NOT for final ranking. At end-of-utterance, a full MiniLM embedding is computed and used to rerank the pre-selected candidates.

**Latency saved**: ~150-300ms of perceived retrieval latency by overlapping retrieval with speech.

See: [layer-1-streaming.md](layer-1-streaming.md)

### Layer 2 — Event-Segmented Hierarchical Index

**Brain analogue**: Hippocampus + boundary/event cells (Zheng et al., 2022) + Sparse Distributed Memory (Kanerva, 1988).

**Purpose**: Use a two-stage retrieval that keeps the hot path in cache and bounds the rescan to a small event-scoped subset.

Two-stage flow:

**Stage 2a — Binary signature scan** *(shipped, v0.1)*: Every document has a 256-bit binary signature (sign-bit quantization of its dense embedding). 1M signatures = 32MB, fits in L2/L3 cache. SIMD-accelerated Hamming distance scan (AVX-512 VPOPCNTDQ → AVX2 VPSHUFB → scalar fallback) finds the coarse top-K in ~0.3-0.5ms.

**Stage 2b — Event-scoped subset rescan** *(shipped, v0.1)*: The coarse top-K is mapped to candidate events via the event catalog. The union of those events' document scopes is gathered into a contiguous buffer and rescanned with the same SIMD kernel to produce the final top-K. This is *not* HNSW — it's a SIMD gather + subset rescan. It works because event scopes are usually small (hundreds to low-thousands of docs) and contiguous, so the gather is cheap and the OS prefetcher does most of the work.

**Stage 2b — Real per-event HNSW shards** *(planned, v0.2)*: For corpora where the event scope itself is ≥ 5–10 k docs, the subset rescan starts to dominate. v0.2 adds an actual HNSW graph per event for that case. See [roadmap.md](../plan/roadmap.md).

**Combined (v0.1)**: ~0.3–0.7 ms for 100 k documents single-thread; the dominant cost is the coarse Hamming scan, not the subset rescan.

See: [layer-2-index.md](layer-2-index.md)

### Layer 3 — Predictive Co-Activation

**Brain analogue**: Prefrontal cortex + hebbian co-activation + priming.

**Purpose**: Prefetch the next likely answer during TTS playback.

After each user turn, primd looks up the current event signature in a learned conversation transition matrix and prefetches the top-5 predicted next events into cache. This happens during the TTS playback window (3-10 seconds of otherwise idle CPU).

The transition matrix is a sparse Markov model trained offline on domain-specific conversation transcripts. Lookup cost: ~10 microseconds.

When the prediction hits (projected 60-80% on structured domains), the next retrieval is a cache hit at ~0.5ms instead of a full 2-4ms search.

See: [layer-3-prediction.md](layer-3-prediction.md)

### Layer 4 — Predictive-Coding Delta Cache

**Brain analogue**: Friston's predictive coding + Rao-Ballard error correction.

**Purpose**: Skip retrieval entirely when the user is elaborating on the current topic.

40-50% of conversational turns are elaborations, not pivots. primd caches the current result set's centroid embedding + a learned topic radius. On the next query:

- Compute cosine distance to centroid (~10 microseconds)
- If within radius: return cached results ± a 1-hop delta search (<0.1ms)
- If outside radius: fall through to layers 1-3

See: [layer-4-coding.md](layer-4-coding.md)

## Composite Latency Profile

| Scenario | % of Turns (projected) | Latency | Path |
|---|---|---|---|
| Topic continuation | 40-50% | <0.1 ms | Layer 4 (delta cache) |
| Predicted pivot (cache hit) | 25-35% | 1.6 µs (measured) | Layer 3 → speculative cache |
| Novel query (full stack) | 15-25% | ~150–250 µs at 100 k docs | Layer 1 → 2a → 2b subset rescan |
| Cold start (first turn) | once | ~5–15 ms | Full embed + mmap fault |

**Measured at 100 k docs (`cargo bench --bench voice_session`): finalize p50 1.6 µs, p95 2.8 µs, naive p50 157.8 µs.**

At 1 M docs the v0.1 subset-rescan path will be ~10× slower than the 100 k numbers; v0.2 per-event HNSW closes that gap.

## Data Flow

```
Partial transcript arrives (from STT stream)
    │
    ▼
Layer 1: streaming gate → emit signature when partial stabilizes
    │
    ├── If Layer 4 cache hit (delta cache: scope_hash + signature match)
    │       └── Return cached results  ──▶  <0.1 ms  ──▶  Done
    │
    ├── If Layer 3 + speculation match (finalize signature ≈ speculative signature)
    │       └── Return speculative cache  ──▶  1.6 µs measured  ──▶  Done
    │
    └── Full path:
            Layer 2a: SIMD Hamming scan over signatures → coarse top-K
            Layer 2b: map coarse hits → candidate events → union scope
                       gather scope into contiguous buffer → SIMD rescan → final top-K
            ──▶  ~150–250 µs at 100 k docs  ──▶  Done

            (v0.2: replace gather+rescan with per-event HNSW when scope ≥ 5–10 k)

After retrieval:
    │
    ▼
Update Layer 4 centroid cache
Update Layer 3 transition context
    │
    ▼ (during TTS playback, async)
    │
Layer 3: Prefetch top-5 predicted next events into cache
```

## Deployment Modes

| Mode | Corpus Scale | Expected p50 | Use Case |
|---|---|---|---|
| Server (AVX-512) | 1M docs | <1ms | Production voice agents |
| Desktop (AVX2) | 1M docs | <2ms | Local dev, desktop apps |
| WASM (browser) | 100K docs | <10ms | In-app voice copilots |
| WASM (browser) | 10K docs | <3ms | Small knowledge bases |

## Integration Points

primd is designed to drop into existing voice agent pipelines:

- **Pipecat**: Ships as a `PrimdRetriever` processor that plugs into a Pipecat pipeline
- **LiveKit**: Ships as a LiveKit agent plugin
- **Generic**: Python and TypeScript SDKs for any framework
- **Knowledge base**: Reads from Qdrant, pgvector, raw parquet, or local files at index time
