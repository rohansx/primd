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

**Purpose**: Replace flat HNSW with a two-level cache-friendly structure.

Flat HNSW at 1M+ vectors becomes cache-hostile — every neighbor hop is a pointer chase. primd uses a two-stage approach:

**Stage 2a — Binary signature scan**: Every document has a 256-bit binary signature (sign-bit quantization of its dense embedding). 1M signatures = 32MB, fits in L2/L3 cache. SIMD-accelerated hamming distance scan finds the right neighborhood in ~0.3-0.5ms.

**Stage 2b — Per-event HNSW shards**: Documents are grouped into "events" (topically coherent clusters of 100-1000 chunks). Each event has a small HNSW graph. Searching a ~1K-node shard takes ~0.3-0.5ms.

**Combined**: ~0.5-1ms for 1M documents with better p99 than flat HNSW.

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
| Topic continuation | 40-50% | <0.1ms | Layer 4 |
| Predicted pivot (cache hit) | 25-35% | ~0.5-1ms | Layer 3 → Layer 2b |
| Novel query (full stack) | 15-25% | ~2-4ms | Layer 1 → 2a → 2b |
| Cold start (first turn) | once | ~5-15ms | Full embed + mmap fault |

**Projected p50: <1ms. Projected p99: <3ms.**

## Data Flow

```
Partial transcript arrives (from STT stream)
    │
    ▼
Layer 1: Incremental embedding → speculative candidate set (top-100)
    │
    ├── If Layer 4 cache hit (within topic radius)
    │       └── Return cached results + delta  ──▶  <0.1ms  ──▶  Done
    │
    ├── If Layer 3 cache hit (prefetched shard is warm)
    │       └── Search warm HNSW shard  ──▶  ~0.5-1ms  ──▶  Done
    │
    └── Full path:
            Layer 2a: Binary signature scan → top-256 events
            Layer 2b: HNSW search in matching shard → top-K results
            Rescore with full embeddings → final ranked results
            ──▶  ~2-4ms  ──▶  Done

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
