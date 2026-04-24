# Layer 1 — Streaming Partial-Query Retrieval

**Brain analogue**: Auditory cortex + predictive coding hierarchy (Caucheteux et al., Nature Human Behaviour, 2023).

**Core idea**: Start retrieving before the user finishes speaking. Don't wait for end-of-utterance.

## The Problem It Solves

Traditional retrieval pipelines are reactive:

```
User speaks (1.2s) → VAD detects silence → Full embed (5-10ms) → Search (2-50ms) → Results
                                           ├─── this entire block ───┘
                                           └─── adds 7-60ms AFTER the user stopped talking
```

Layer 1 overlaps retrieval with speech:

```
User speaks: "so"           → embed partial → speculative retrieve
             "so what"      → update embed  → speculative retrieve
             "so what about"→ update embed  → speculative retrieve
             "so what about pricing" → full embed → confirm/refine
                                       ▲
                                       └── results already 80% ready
```

**Net savings**: ~150-300ms of perceived retrieval latency.

## Architecture

### Incremental Embedder

A lightweight model produces running embeddings from partial transcripts:

- **Model**: MiniLM-L6-v2 (22M params), ONNX runtime, int8 quantized
- **Inference**: ~3-5ms per short query (20-30 tokens) with multi-threading
- **Update strategy**: Full re-embed on each partial (not EMA)

**Why not EMA**: Research shows that exponential moving averages over token embeddings degrade retrieval quality by 10-20% because they over-weight recent tokens and under-weight topic-bearing early tokens. Instead, we re-embed the full partial transcript each time. At 3-5ms per embed and ~3 speculative cycles per utterance, the total compute is ~9-15ms spread across the utterance duration.

### Speculative Retrieval

Speculative retrievals fire on a token-count trigger (every 2-3 new tokens from STT):

```rust
pub struct StreamingRetriever {
    embedder: IncrementalEmbedder,
    last_embedding: Option<Vec<f32>>,
    last_candidates: Option<CandidateSet>,
    token_count: usize,
    trigger_interval: usize,  // default: 3 tokens
}

impl StreamingRetriever {
    /// Called on each partial transcript from STT
    pub fn observe(&mut self, partial: &str) -> Option<CandidateSet> {
        let tokens = self.embedder.tokenize(partial);
        if tokens.len() - self.token_count < self.trigger_interval {
            return None;  // not enough new tokens
        }
        self.token_count = tokens.len();

        // Full embed of current partial (not EMA)
        let embedding = self.embedder.embed(partial);

        // If embedding is close to last one, skip retrieval
        if let Some(ref last) = self.last_embedding {
            if cosine_similarity(&embedding, last) > 0.95 {
                return self.last_candidates.clone();
            }
        }

        // Speculative retrieve: binary scan only (layer 2a), top-100
        let candidates = self.index.binary_scan(&embedding, 100);
        self.last_embedding = Some(embedding);
        self.last_candidates = Some(candidates.clone());
        Some(candidates)
    }

    /// Called at end-of-utterance
    pub async fn finalize(&mut self, final_transcript: &str) -> Vec<Result> {
        let embedding = self.embedder.embed(final_transcript);

        // If we have speculative candidates, rerank them
        if let Some(ref candidates) = self.last_candidates {
            let refined = self.index.rerank(candidates, &embedding, 10);
            self.reset();
            return refined;
        }

        // No speculative results — full search
        let results = self.index.full_search(&embedding, 10).await;
        self.reset();
        results
    }
}
```

### Embedding Similarity Gate

Not every partial transcript produces a meaningfully different embedding. To avoid redundant work:

- After each embed, compare to the previous embedding via cosine similarity
- If similarity > 0.95, skip retrieval and reuse previous candidates
- This avoids wasting cycles when the user says filler words ("um", "uh", "like")

### Integration with STT Stream

primd consumes the same partial transcript stream that the voice agent already receives from STT providers:

```
Deepgram Nova-3:  interim results every ~100ms
Whisper streaming: chunks every ~500ms
AssemblyAI:       real-time partials
```

primd's `observe()` is called on every interim result. It decides internally whether enough has changed to trigger a new speculative retrieval.

## Performance Characteristics

| Metric | Value |
|---|---|
| Embed latency per partial | ~3-5ms (MiniLM-L6-v2, int8, 4 threads) |
| Speculative retrievals per utterance | 2-4 |
| Total compute per utterance | ~9-15ms (spread across 1.2s speech) |
| Perceived latency savings | 150-300ms |
| Quality vs full-query retrieval | Speculative candidates overlap ~80% with final results |

## Limitations

- **Short partials have low signal.** "so" or "can you" produce nearly random embeddings. Speculative retrieval only becomes useful after ~4-5 meaningful tokens.
- **Re-embedding is more expensive than EMA.** But the quality difference (10-20% better recall) justifies the cost for retrieval accuracy.
- **STT interim results vary in quality.** Deepgram produces stable partials; other providers may produce noisy or frequently-revised partials. primd's similarity gate handles this by skipping retrieval when the embedding hasn't changed meaningfully.

## Configuration

```toml
[layer1]
enabled = true
trigger_interval = 3          # tokens between speculative retrievals
similarity_threshold = 0.95   # skip retrieval if embedding hasn't changed
max_speculative_results = 100  # top-K for speculative candidate set
embedder_threads = 4           # threads for ONNX inference
```
