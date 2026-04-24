# Risks & Mitigations

Honest assessment of what could go wrong.

## Technical Risks

### Risk 1: Streaming Embedding Quality (HIGH)

**The problem**: EMA over token embeddings degrades retrieval quality by 10-20%. This was the original design — it doesn't work well enough.

**Mitigation**: Switched to full re-embed on each partial with a cosine similarity gate. More expensive (~3-5ms per partial vs ~0.3ms for EMA) but maintains retrieval quality. Acceptable because the compute is spread across the utterance duration (1.2 seconds).

**Residual risk**: Even with full re-embed, short partials ("so what") produce poor embeddings. Layer 1 only becomes useful after ~4-5 meaningful tokens.

**Status**: Mitigated by design change. Residual risk is acceptable.

### Risk 2: Cache Hit Rate Below Projections (MEDIUM-HIGH)

**The problem**: Projected 60-80% cache hit rate is based on VoiceAgentRAG's numbers (75%) and structural similarity. primd's Markov predictor is simpler than VoiceAgentRAG's LLM predictor — hit rate may be lower.

**VoiceAgentRAG hit rates by scenario**:
| Scenario | Hit Rate |
|---|---|
| Feature comparison | 95% |
| Pricing deep-dive | 85% |
| Overall | 75% |
| Mixed rapid-fire | 55% |
| Customer upgrade (worst) | 45% |

**Mitigation**:
- Publish per-domain hit-rate tables (no misleading aggregates)
- Ship pretrained matrices for top 5 verticals (sales, support, scheduling, intake, docs)
- Expose training pipeline so customers can train on their own data
- Layer 4 (delta cache) catches many follow-ups even when layer 3 misses

**Worst case**: 40-50% hit rate on semi-structured domains. Still better than zero (every competitor without prediction).

### Risk 3: Event Boundary Detection Quality (MEDIUM-HIGH)

**The problem**: Layer 2's entire advantage depends on well-chosen event boundaries. Bad boundaries = binary scan selects wrong events = retrieval misses relevant documents.

**Mitigation**:
- Default: k-means with silhouette-score-guided k selection
- Ship domain-specific boundary detectors as plugins
- Expose boundary quality metrics (`primd check` command)
- Allow manual boundary override for domain experts
- Recursive splitting/merging to keep event sizes in range

**Worst case**: Poor boundaries degrade layer 2 to ~flat HNSW performance (still works, just slower).

### Risk 4: WASM Performance Gap (MEDIUM)

**The problem**: WASM SIMD is 8-16x slower than native AVX-512. The original spec implied near-native WASM performance.

**Mitigation**:
- Tier the claims explicitly:
  - Server (AVX-512): 1M docs, <1ms
  - WASM (browser): 100K docs, <10ms / 10K docs, <3ms
- Position WASM as "good enough for in-browser copilots" not "production at scale"
- Server-side is the primary deployment target

**Worst case**: WASM is only viable for <10K docs. Still useful for small knowledge bases.

### Risk 5: Binary Quantization for Sensitive Domains (LOW-MEDIUM)

**The problem**: 1-3% recall loss (with rescoring) may be unacceptable for legal, medical, or financial retrieval where missing a relevant document has consequences.

**Mitigation**:
- Ship fp16 as a middle ground (half memory, ~0.5% recall loss, ~2x latency)
- Make quantization configurable via `primd.toml`
- Document recall/latency tradeoffs per quantization level

**Worst case**: Some customers need fp32 everywhere. primd is still faster than alternatives due to layers 3-4 (prediction, caching).

## Market Risks

### Risk 6: Moss Adds Prediction (HIGH)

**The problem**: Moss.dev is the closest competitor. They're backed by YC, partnered with Pipecat/LiveKit, and have paying customers. If they add predictive prefetch, primd's gap narrows significantly.

**Mitigation**:
- Move fast — 12-week MVP timeline
- Open source everything (Moss's core runtime is closed)
- Build the transition matrix training pipeline as a moat (domain-specific data flywheel)
- Focus on the four-capability combination, not any single feature

**Reality check**: Adding prediction to an existing system is non-trivial. It requires instrumenting the TTS playback loop, building a training pipeline, and managing cache invalidation. Moss would need 3-6 months to add this properly. primd's head start matters if execution is fast.

### Risk 7: Category Confusion (MEDIUM)

**The problem**: The market already has vector databases, retrieval frameworks, memory layers, and agent frameworks. Developers are confused about what goes where. "Retrieval runtime" is a new category that needs explanation.

**Mitigation**:
- Never say "vector database"
- Lead with the use case ("eliminate dead air in voice AI") not the category
- Pipecat/LiveKit plugin makes it tangible ("just add this to your pipeline")
- Benchmark comparisons show where primd fits relative to known systems

### Risk 8: Voice AI Market Concentration (MEDIUM)

**The problem**: If 2-3 platforms (Vapi, Retell, LiveKit) dominate and build their own retrieval optimization, primd's addressable market shrinks.

**Mitigation**:
- Target the open-source/self-hosted segment (Pipecat, LiveKit Agents)
- Build platform-agnostic SDKs (Python, TypeScript) that work with any framework
- Position as infrastructure that platforms should integrate, not replace

### Risk 9: Rust + WASM Is the Wrong Stack (LOW-MEDIUM)

**The problem**: If most users deploy server-side, Go or C++ might be easier to contribute to. The WASM bet only pays off if on-device voice AI grows.

**Mitigation**:
- Moonshine v2 (107ms STT, 26MB) validates on-device voice AI trajectory
- Privacy regulation is pushing toward on-premise/edge
- Rust has excellent cross-compilation and WASM support
- Keep core narrow enough that a C++ port is feasible if needed

## Execution Risks

### Risk 10: Solo Developer, 12-Week Timeline (HIGH)

**The problem**: Building a complete retrieval runtime with benchmarks, plugins, SDKs, and launch materials in 12 weeks is aggressive for one person.

**Mitigation**:
- Phase 1 (core engine) is self-contained and valuable alone
- Scope ruthlessly: skip WASM in MVP if needed
- Skip one SDK (TypeScript can wait for v0.2)
- The benchmark suite can be simplified (primd vs FAISS only, add others later)

**Minimum viable launch**: Layer 2 (binary scan + HNSW) + Layer 4 (delta cache) + Pipecat plugin + FAISS benchmark. Layers 1 and 3 can be v0.2.

### Risk 11: Training Data Availability (MEDIUM)

**The problem**: The Markov predictor needs domain-specific conversation transcripts. Real transcripts are hard to get at scale.

**Mitigation**:
- MultiWOZ is public (10K conversations, 7 domains)
- Build synthetic SDR corpus from Gong's public call patterns
- Ship the training pipeline so customers can use their own data
- Layer 3 degrades gracefully: without a good matrix, it simply doesn't prefetch (no harm, just no benefit)
