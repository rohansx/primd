# Feasibility Assessment

Honest evaluation of every major technical claim, based on published benchmarks and prior art.

## Summary

| # | Claim | Verdict | Confidence |
|---|---|---|---|
| 1 | AVX-512 hamming scan: 1M sigs in <0.3ms | Achievable with 2+ threads | High |
| 2 | Binary quantization: 1-3% recall loss | Achievable with rescoring | High |
| 3 | Streaming embeddings for speculative retrieval | Feasible (quality caveat) | High |
| 4 | MiniLM embedder: 2-3ms per query | Achievable for short queries | High |
| 5 | Markov predictor: 70-85% hit rate | Only in constrained domains | Medium |
| 6 | WASM SIMD deployment | Works, 8-16x slower than native | High |
| 7 | mmap index with no load phase | Production-proven | High |

---

## 1. AVX-512 Binary Hamming Scan

**Claim**: Scan 1M × 256-bit signatures in <0.3ms.

**Analysis**:
- 1M × 32 bytes = 32 MB data to scan
- Single-core L3 bandwidth: ~80 GB/s → ~0.4ms theoretical minimum
- `_mm512_popcnt_epi64` processes 8 × 64-bit popcounts per instruction
- FAISS binary index PR confirms 8x pathlength reduction with this instruction
- AWS OpenSearch benchmarks show 48% improvement on Sapphire Rapids with VPOPCNTDQ

**Verdict**: ~0.4-0.5ms single-threaded. <0.3ms with 2 threads via `rayon`. Achievable on Ice Lake or newer.

**Adjustment**: Spec says "<0.5ms single-thread, <0.3ms multi-threaded" to be accurate.

**Sources**: FAISS PR #4020, AWS OpenSearch binary vector blog, SimSIMD benchmarks.

---

## 2. Binary Quantization Recall

**Claim**: 1-3% recall loss at recall@10 = 0.95 using sign-bit quantization.

**Analysis**:
- Qdrant benchmarks on OpenAI embeddings: binary quantization + rescoring achieves 97-99% recall
- Elasticsearch BBQ: ~95% memory reduction with maintained ranking quality via rescoring
- Weaviate confirms similar patterns with 32x memory reduction
- Raw binary (without rescoring): only ~74-77% recall — unacceptable

**Verdict**: Achievable, but the architecture MUST include an oversampling + rescoring step. The flow is: binary scan → top-256 candidates → float32 rescore → final top-10. The rescore adds ~0.1ms.

**Sources**: Qdrant binary quantization blog, Elasticsearch BBQ blog, Jégou et al. (2016) polysemous codes.

---

## 3. Streaming Embeddings

**Claim**: Incremental embeddings via EMA at ~0.3ms per update, suitable for speculative retrieval.

**Analysis**:
- RAW-Embeddings (Yann Dubois) uses EMA over word embeddings as rolling sentence representation
- Reimers & Gurevych (2019): naive averaging of token embeddings without fine-tuning produces embeddings significantly worse than sentence transformers
- EMA over-weights recent tokens (end of sentence) and under-weights early tokens (topic/subject)
- Expected quality: 10-20% retrieval degradation vs proper sentence-level embedding

**Verdict**: EMA embeddings are too low-quality for final retrieval. However, they're acceptable for speculative candidate pre-selection (layer 1 only — narrowing from 1M to 100 candidates). primd uses full re-embedding at end-of-utterance for final ranking.

**Revised approach**: Full re-embed on each partial (~3-5ms) with a cosine similarity gate to skip redundant re-embeds. More expensive than EMA but retrieval quality is maintained.

**Sources**: RAW-Embeddings (GitHub), Sentence-BERT paper (Reimers & Gurevych, 2019).

---

## 4. MiniLM Embedder Latency

**Claim**: ~2-3ms per token in ONNX int8.

**Analysis**:
- MiniLM-L6-v2 fp32, single-thread, 128 tokens: ~20ms per sentence
- ONNX int8 quantization: 2-2.5x speedup → ~8-12ms per sentence
- With 4 threads: ~3-5ms per sentence
- Shorter queries (20-30 tokens, typical for voice): ~2-3ms

**Verdict**: Achievable for short voice queries with int8 + multi-threading. The original "per token" framing is incorrect — transformers process entire sequences, not individual tokens. Correct framing: "~3-5ms per query."

**Sources**: ONNX Runtime benchmarks, Sentence Transformers efficiency docs.

---

## 5. Markov Predictor Accuracy

**Claim**: 70-85% topic prediction accuracy without an LLM.

**Analysis**:
- Switchboard corpus (Stolcke et al., 2000): HMM dialog act tagging achieves 65-71% with ~42 dialog act types
- With coarser categories (5-10 broad topics): 75-85% achievable
- VoiceAgentRAG (LLM predictor): 75% overall, 45-95% by scenario
- Markov models are inherently limited by their inability to capture long-range dependencies or semantic nuance

**Verdict**: 70-85% is achievable ONLY for:
- Structured domains (sales, support, scheduling, intake)
- Coarse topic taxonomy (<15 categories)
- Regular conversation patterns

For open-domain with fine-grained taxonomy (50+ topics): expect 35-50%.

**Mitigation**: Ship pretrained matrices for top 5 verticals. Expose training pipeline for custom domains. Consider a tiny classifier (~2M params, ~1ms) as a middle-ground fallback.

**Sources**: Stolcke et al. (2000), VoiceAgentRAG (arXiv 2603.02206).

---

## 6. WASM SIMD Performance

**Claim**: Implied near-native performance via WASM SIMD.

**Analysis**:
- WASM has standardized 128-bit fixed-width SIMD (all major browsers)
- AVX-512 is 512-bit → 4x width disadvantage
- JIT compilation + sandbox overhead → additional 1.3-2x penalty
- Hashing benchmarks: WASM SIMD is ~4x slower than native AVX2 (256-bit)
- Against AVX-512: 6-10x slower for SIMD-heavy workloads, up to 16x worst-case
- Relaxed SIMD and Flexible Vectors proposals are in early stages (years from standardization)

**Verdict**: For 1M vectors at ~0.4ms native, expect ~3-6ms in WASM. This is still fast enough for a browser context but is a fundamentally different performance tier.

**Adjustment**: Position WASM deployment as:
- 10K docs: <3ms (good)
- 100K docs: <10ms (usable)
- 1M docs: not recommended in WASM

**Sources**: Emscripten SIMD docs, WASM Flexible Vectors proposal, wasm-performance benchmarks.

---

## 7. mmap-based Index

**Claim**: Near-instant startup with no explicit load phase.

**Analysis**:
- Production-proven pattern: Qdrant, Milvus, LMDB, RocksDB, SQLite all use mmap
- Startup is near-instant (file open + mmap setup)
- First queries incur page faults: ~10-100μs per 4KB page
- For 32MB binary index: ~8K pages → potentially 0.8-8ms of page fault overhead on first access
- After warm-up: performance equivalent to RAM

**Verdict**: This is the strongest claim. Well-understood, battle-tested pattern with known tradeoffs.

**Tradeoff**: "No load phase" trades startup time for first-query latency. Under memory pressure, pages may be evicted, causing sporadic latency spikes. Production deployments need sufficient RAM headroom.

**Sources**: Qdrant mmap documentation, Milvus mmap docs, CMU "Are You Sure You Want to Use MMAP?" (CIDR 2022).

---

## Overall Assessment

The technical approach is sound. The largest risks are:

1. **Streaming embedding quality** — addressed by switching from EMA to full re-embed with similarity gate
2. **Markov predictor accuracy** — bounded to structured domains, clearly communicated
3. **WASM performance gap** — tiered by deployment target, not over-promised

No individual claim is infeasible. The composite system is ambitious but achievable within the 12-week timeline if scope is managed carefully.
