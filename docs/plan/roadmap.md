# Roadmap

## 12-Week Solo MVP Plan

Three phases, each building on the last. Every phase ends with a measurable benchmark.

```
Week  1  2  3  4  5  6  7  8  9  10  11  12
      ├──────────┤  ├──────────┤  ├───────────┤
      Phase 1:      Phase 2:      Phase 3:
      Foundation    Intelligence  Integration
```

### Phase 1 — Foundation (Weeks 1-4)

Build the core retrieval engine. Prove that the hierarchical index beats flat HNSW.

| Week | Deliverable | Success Criteria |
|---|---|---|
| 1-2 | Layer 2a: Binary signatures + AVX-512 hamming scan | <0.5ms on 1M vectors (single-thread) |
| 3-4 | Layer 2b: Per-event HNSW shards + rescore pipeline | recall@10 >= 0.97 at 1M docs vs flat HNSW |

**Phase 1 milestone**: End-to-end retrieval at ~1ms with 97%+ recall. First benchmark vs FAISS published.

See: [phase-1-foundation.md](phase-1-foundation.md)

### Phase 2 — Intelligence (Weeks 5-8)

Add the predictive layers. Prove latency savings on real voice traces.

| Week | Deliverable | Success Criteria |
|---|---|---|
| 5-6 | Layer 1: Streaming partial-query embedder | Speculative retrieval within 100ms of first meaningful partial |
| 7-8 | Layer 3: Markov transition matrix + prefetch | >= 70% cache hit rate on structured domain corpus |

**Phase 2 milestone**: Conversational benchmark showing p50 <1ms, p99 <3ms on SDR dialogue corpus.

See: [phase-2-intelligence.md](phase-2-intelligence.md)

### Phase 3 — Integration (Weeks 9-12)

Build the delta cache, plugins, demo, and launch materials.

| Week | Deliverable | Success Criteria |
|---|---|---|
| 9-10 | Layer 4: Predictive-coding delta cache + end-to-end benchmarks | p50 <1ms, p99 <3ms on SDR dialogue corpus |
| 11 | Pipecat + LiveKit plugins + SDR bot demo | Working end-to-end voice agent with measurable improvement |
| 12 | arxiv preprint + launch: GitHub, HN, blog with benchmark repo | All baselines in `make bench` |

**Phase 3 milestone**: Public launch with reproducible benchmarks, working integrations, and a live demo.

See: [phase-3-integration.md](phase-3-integration.md)

## Key Dependencies

```
Layer 2a (binary scan)
    │
    ▼
Layer 2b (HNSW shards)  ← requires 2a for candidate filtering
    │
    ├── Layer 1 (streaming) ← can parallelize with layer 3
    │
    └── Layer 3 (prediction) ← requires event structure from 2b
            │
            ▼
        Layer 4 (delta cache) ← requires centroid data from layers 2-3
```

Layers 1 and 3 can be developed in parallel once layer 2 is complete.

## Risk Mitigations by Phase

| Phase | Primary Risk | Mitigation |
|---|---|---|
| 1 | Binary quantization recall too low for target domain | Ship fp16 fallback, configurable quantization |
| 1 | Event boundary detection produces poor clusters | Expose boundary quality metrics, iterative tuning |
| 2 | Streaming embedder latency too high | Reduce trigger frequency, use similarity gate aggressively |
| 2 | Markov predictor accuracy below 60% | Expand training data, consider tiny neural classifier |
| 3 | Pipecat/LiveKit API changes during development | Pin SDK versions, abstract plugin interface |
| 3 | Benchmark results don't match claims | Adjust claims to match reality (honest documentation) |

## Post-MVP Roadmap

See: [post-v1.md](post-v1.md)
