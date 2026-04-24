# Layer 4 — Predictive-Coding Delta Cache

**Brain analogue**: Friston's predictive coding framework + Rao-Ballard hierarchical prediction.

**Core idea**: When the user is continuing the same topic, don't search at all — return what you already have, adjusted by a cheap delta.

## The Problem It Solves

40-50% of conversational turns are elaborations on the current topic:

```
User: "What's the pricing for the pro plan?"     ← initial query
User: "And what's included in that?"              ← elaboration
User: "What about yearly billing?"                ← elaboration
User: "Can you tell me more about the API limits?"← elaboration
```

Running full retrieval for each of these is wasteful. The relevant documents are almost identical — the user is just asking for more detail about the same cluster of information.

Layer 4 detects this pattern and short-circuits retrieval entirely.

## Architecture

### Topic Centroid Cache

After each successful retrieval, primd computes and caches:

1. **Centroid embedding**: The mean of all retrieved document embeddings
2. **Topic radius**: A learned threshold for "same topic" (cosine distance)
3. **Result set**: The full set of retrieved documents

```rust
pub struct TopicCache {
    centroid: Vec<f32>,           // mean of result embeddings
    radius: f32,                  // cosine distance threshold
    results: Vec<ScoredDocument>, // cached result set
    turn_count: usize,            // how many turns this cache has been active
    max_turns: usize,             // force refresh after N turns (default: 5)
}

impl TopicCache {
    /// Check if new query is within the current topic
    pub fn is_continuation(&self, query_embedding: &[f32]) -> bool {
        if self.turn_count >= self.max_turns {
            return false;  // force refresh to prevent stale results
        }

        let distance = 1.0 - cosine_similarity(&self.centroid, query_embedding);
        distance < self.radius
    }

    /// Return cached results with optional delta adjustment
    pub fn retrieve_delta(
        &self,
        query_embedding: &[f32],
        index: &Index,
    ) -> Vec<ScoredDocument> {
        // Rescore cached results against new query
        let mut rescored: Vec<_> = self.results.iter()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                ScoredDocument { score, ..doc.clone() }
            })
            .collect();

        // 1-hop expansion: check immediate neighbors of top result
        if let Some(top) = rescored.first() {
            let neighbors = index.get_hnsw_neighbors(top.id, 10);
            for neighbor in neighbors {
                let score = cosine_similarity(query_embedding, &neighbor.embedding);
                rescored.push(ScoredDocument { score, ..neighbor });
            }
        }

        rescored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        rescored.truncate(10);
        rescored
    }
}
```

### Topic Radius Learning

The radius is not hardcoded — it's learned per-domain from training data:

1. From the same conversation transcripts used for layer 3, extract pairs of consecutive turns
2. For each pair, compute the cosine distance between their query embeddings
3. Classify pairs as "same topic" or "topic pivot" based on human annotation or heuristic (different event = pivot)
4. Set radius to the 85th percentile of "same topic" distances

**Default radius**: 0.3 cosine distance (works reasonably for most domains). Adjustable per deployment.

### Decision Flow

```
New query arrives → embed with full MiniLM
    │
    ▼
Is Layer 4 cache active?
    │
    ├── No → Fall through to layers 1-3
    │
    └── Yes → Compute cosine distance to centroid
              │
              ├── distance < radius AND turn_count < max_turns
              │       └── CACHE HIT: rescore + 1-hop delta (~0.05ms)
              │
              └── distance >= radius OR turn_count >= max_turns
                      └── CACHE MISS: invalidate cache, fall through
                          └── After retrieval: update cache with new centroid
```

## Performance

| Operation | Latency |
|---|---|
| Cosine distance to centroid | ~10μs (single vector dot product) |
| Rescore cached results (10 docs) | ~20μs |
| 1-hop HNSW neighbor lookup | ~20μs |
| **Total cache hit** | **~50μs (<0.1ms)** |

This is ~20-80x faster than a full layer 2 retrieval.

## When It Works Well

- **Sales calls**: "Tell me about the enterprise plan" → "What's included?" → "How about for teams of 50?" → "Any volume discounts?"
- **Support tickets**: "My login isn't working" → "I tried resetting my password" → "It still shows the error" → "What error code?"
- **Documentation Q&A**: "How do I set up webhooks?" → "What events are available?" → "Can I filter by type?"

## When It Doesn't Work

- **Rapid topic pivots**: "What's your pricing?" → "Who's your CEO?" (cosine distance exceeds radius, falls through correctly)
- **Subtle topic shifts**: "What's the enterprise plan?" → "What's the startup plan?" (might hit cache incorrectly if embeddings are too similar). Mitigated by `max_turns` forcing periodic refresh.
- **First turn**: No cache exists yet. Falls through to layers 1-3.

## Staleness Protection

Two mechanisms prevent stale results:

1. **max_turns**: Cache is forcibly invalidated after N turns on the same topic (default: 5). This prevents the cache from returning increasingly irrelevant results on long topic runs.
2. **Centroid drift**: On each cache hit, the centroid is updated with a weighted blend of old centroid and new query embedding (EMA, alpha=0.1). This allows the cache to "drift" with the conversation rather than being locked to the initial query.

## Configuration

```toml
[layer4]
enabled = true
default_radius = 0.3          # cosine distance threshold
max_turns = 5                  # force refresh after N cache hits
centroid_alpha = 0.1           # EMA weight for centroid drift
delta_neighbors = 10           # 1-hop expansion count
min_cache_results = 5          # don't cache if fewer results than this
```

## Limitations

- **False cache hits**: If two genuinely different topics have similar embeddings, the cache may return wrong results. The `max_turns` and `centroid_alpha` parameters mitigate this but don't eliminate it.
- **1-hop delta is shallow**: The delta search only looks at immediate HNSW neighbors of the top cached result. If the elaboration requires documents that are more than 1 hop away in the graph, the delta won't find them. These cases fall through on the next turn when the cache is invalidated.
- **Not suitable for multi-topic turns**: If the user asks about two topics in one utterance ("What's the pricing, and also how do I set up?"), the cache will match one topic but miss the other. This is rare in voice (people ask one thing at a time) but possible.
