# Layer 3 — Predictive Co-Activation

**Brain analogue**: Prefrontal cortex + hebbian co-activation + associative priming.

**Core idea**: While the voice agent is speaking its current answer (3-10 seconds of free CPU), prefetch the most likely next answer into cache.

## The Problem It Solves

Voice agents have a hidden resource: idle CPU during TTS playback. Today, every voice agent wastes this time. primd uses it to predict and prefetch the next likely retrieval, so when the user asks a follow-up, the answer is already warm in cache.

VoiceAgentRAG (Salesforce, March 2026) does this with an LLM predictor — but that adds 200-500ms of compute per prediction. primd replaces the LLM with a sparse Markov transition matrix: ~10 microseconds per prediction.

## Architecture

### Offline: Training the Transition Matrix

The transition matrix is built from conversation transcripts:

```
Input: historical transcripts for the target domain
       (sales calls, support tickets, medical intake)

Process:
1. Segment each conversation into turns
2. For each turn, identify the matching event signature (layer 2)
3. Extract transitions: (current_event, previous_k_events) → next_event
4. Build sparse probability matrix: P(next_event | current_event, context)

Output: transitions.bin (CSR sparse matrix, ~MB scale)
```

**Data sources for training**:
- MultiWOZ (public dialogue dataset, 10K conversations, 7 domains)
- Synthetic SDR corpus (built from public Gong call transcripts)
- Customer-provided transcripts (primd ships a training pipeline)

**Minimum training data**: ~500 conversations for a usable matrix. Quality improves up to ~5K conversations, diminishing returns beyond that.

### Online: Prediction and Prefetch

After each user turn:

```rust
pub struct Predictor {
    transitions: SparseMatrix,  // CSR format
    context_window: VecDeque<EventId>,  // last k events
    prefetch_top_k: usize,  // default: 5
}

impl Predictor {
    /// Called after each retrieval completes
    pub fn predict_next(&self, current_event: EventId) -> Vec<(EventId, f32)> {
        // Sparse row lookup: O(nnz) where nnz ~ 20
        let row = self.transitions.row(current_event);

        // Weight by context (boost if we've seen related events recently)
        let scored = row.iter()
            .map(|(event, prob)| {
                let context_boost = self.context_boost(event);
                (*event, prob * context_boost)
            })
            .collect::<Vec<_>>();

        // Return top-5 most likely next events
        top_k(scored, self.prefetch_top_k)
    }

    /// Called during TTS playback (async, background)
    pub async fn warm_next(&self, index: &Index, current_event: EventId) {
        let predictions = self.predict_next(current_event);

        // Prefetch HNSW shards for predicted events into memory
        for (event_id, _prob) in predictions {
            index.prefetch_shard(event_id).await;
        }
    }
}
```

### Prefetch Mechanics

"Prefetching" means loading the HNSW shard data into OS page cache:

1. Identify which event shard files correspond to predicted events
2. Issue `madvise(MADV_WILLNEED)` on the mmap'd regions (Linux) or equivalent
3. The OS prefaults the pages into memory in the background
4. When the actual query arrives and hits one of these events, the data is already in RAM — no page fault

This is zero-copy and non-blocking. The TTS playback window (typically 3-10 seconds) is more than enough time for the OS to page in a few event shards (~6-60KB each).

### Timing

```
User turn 1: "What's your pricing?"
    │
    ├── primd retrieves pricing docs (2-4ms, full path)
    ├── LLM generates answer
    ├── TTS starts speaking (~3-8 seconds)
    │
    │   During TTS playback (background, async):
    │   ├── Predictor: P(next | pricing) →
    │   │     "discounts" (0.32)
    │   │     "payment_terms" (0.24)
    │   │     "included_features" (0.18)
    │   │     "enterprise_tier" (0.14)
    │   │     "competitors" (0.07)
    │   └── Prefetch all 5 shards into cache
    │
User turn 2: "Any discounts for annual billing?"
    │
    └── primd retrieves from warm "discounts" shard (~0.5ms, cache hit)
```

## Cache Hit Rate Projections

Based on VoiceAgentRAG's published numbers (75% overall, 45-95% by scenario) and adjusted for primd's simpler predictor:

| Domain | Expected Hit Rate | Notes |
|---|---|---|
| Sales calls (structured) | 70-80% | Highly predictable: discovery → demo → pricing → close |
| Customer support (FAQ-style) | 65-75% | Common question clusters, predictable follow-ups |
| Medical intake | 60-70% | Structured form-filling pattern |
| General knowledge Q&A | 40-55% | Less predictable, more topic pivots |
| Open-domain chat | 30-45% | Low predictability, Markov model struggles |

**Honest caveat**: These are projections, not measured. VoiceAgentRAG tested on 76 document chunks with 200 queries. Real-world hit rates at 1M documents may differ. primd will publish per-domain hit-rate tables from the benchmark suite.

## Why Not an LLM?

| | Markov Matrix | LLM Predictor (VoiceAgentRAG) |
|---|---|---|
| Prediction latency | ~10μs | 200-500ms |
| Compute cost | Negligible | GPU/API call |
| Accuracy (structured) | 70-80% | 75-95% |
| Accuracy (open domain) | 30-45% | 55-70% |
| Training data needed | ~500 conversations | Few-shot prompt |
| Explainability | Transparent (row lookup) | Black box |

The tradeoff: primd sacrifices ~10-15% hit rate vs an LLM predictor, but eliminates 200-500ms of compute. For voice AI where every millisecond matters, this is the right tradeoff.

**Upgrade path**: For domains where Markov accuracy is insufficient, a future version could use a tiny classifier (2M params, ~1ms inference) as a middle ground between Markov (~10μs) and full LLM (~200-500ms).

## Configuration

```toml
[layer3]
enabled = true
prefetch_top_k = 5            # number of events to prefetch
context_window = 3             # number of previous events for context
min_probability = 0.05         # don't prefetch events with P < this
prefetch_during_tts = true     # enable background prefetch

[layer3.training]
min_conversations = 500        # minimum training data
max_transitions_per_event = 20 # sparse matrix density cap
context_depth = 3              # k in P(next | current, last_k)
```

## On-Disk Format

```
corpus/
├── transitions.bin    # CSR sparse matrix
│                      # header: num_events, nnz
│                      # row_ptr: [u32; num_events + 1]
│                      # col_idx: [u32; nnz]
│                      # values:  [f32; nnz]
└── centroids.bin      # per-event centroid embeddings (used by layer 4)
```

## Limitations

- **Cold start**: First turn has no history, so no prediction is possible. Falls through to layer 2.
- **Domain specificity**: The transition matrix is trained per-domain. A matrix trained on sales calls will not help for medical intake. Customers need to train their own or use one of the shipped defaults.
- **Coarse taxonomy only**: Markov models work with event-level granularity (tens to hundreds of events), not document-level (millions). The prediction says "user will probably ask about pricing next," not "user will probably ask about the enterprise tier annual pricing with 20% discount."
- **Stale matrices**: If the knowledge base changes significantly, the transition matrix should be retrained. primd does not auto-update the matrix from live traffic in v1.
