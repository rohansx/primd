# Competitive Landscape

## Direct Competitors

### 1. Moss.dev (YC F25)

**What**: Local-first, real-time semantic search runtime in Rust + WASM.

**Positioning**: "Real-time semantic search for conversational AI." Explicitly targeting voice AI as primary use case.

**Published Performance**:
- Sub-10ms end-to-end retrieval (including embedding inference)
- Benchmarked on 100K documents
- Zero network hops

**Stage**: 3 paying customers, 6 enterprise design partners, ~100% WoW growth. Working directly with Pipecat (Daily.co) and LiveKit.

**Architecture**: Rust core → WASM for portability. Local embedding + indexing. Managed data layer for storage/sync.

**Strengths**:
- First mover in "co-located retrieval for voice"
- YC backing, Pipecat/LiveKit partnerships
- Production deployments (small scale)

**Weaknesses**:
- No predictive prefetch — purely reactive
- No streaming partial-query retrieval
- Core runtime is closed source (only SDKs are public)
- No published independent benchmarks, no p99 numbers, no recall figures
- 100K document scale only (10x smaller than primd's target)

**Threat Level**: HIGH. Closest competitor. Could add prediction features.

---

### 2. VoiceAgentRAG (Salesforce AI Research, March 2026)

**What**: Dual-agent architecture with LLM-powered predictive prefetch for voice RAG.

**Published Performance**:

| Metric | Value |
|---|---|
| Traditional RAG avg retrieval | 110.4ms |
| Cache-hit retrieval | 0.35ms |
| Speedup | 316x |
| Overall cache hit rate | 75% |
| Best scenario (feature comparison) | 95% |
| Worst scenario (customer upgrade) | 45% |

**Architecture**: "Slow Thinker" (background LLM agent predicts follow-up topics, prefetches into FAISS cache) + "Fast Talker" (foreground agent queries cache only). Semantic cache with FAISS IndexFlatIP, 2000-entry max, 300s TTL.

**Strengths**:
- Highest published cache hit rate (75-95%)
- Open source (Apache-2.0)
- Published academic validation

**Weaknesses**:
- LLM predictor adds 200-500ms background compute per turn
- Tested on tiny corpus (76 document chunks, 200 queries)
- No streaming partial-query retrieval
- Requires network access to vector database (Qdrant Cloud)
- Depends on LLM API (GPT-4o-mini) — cost and latency implications

**Threat Level**: MEDIUM. Research prototype, not production system. Different architectural bet (LLM-heavy vs LLM-free).

---

### 3. StreamRAG (Meta + CMU, October 2025)

**What**: Speculative retrieval during speech — issues retrieval queries from partial utterances before the user finishes speaking.

**Published Performance**:
- 34.2% accuracy (vs 26.3% standard RAG)
- ~20% tool use latency reduction
- Processes speech in 500ms blocks

**Strengths**:
- First published work on mid-utterance speculative retrieval
- Academic validation (Meta Research + CMU)

**Weaknesses**:
- Single-turn only — no cross-turn prediction
- Focused on accuracy improvement, not raw latency
- No local-first architecture
- 500ms block size is coarse

**Threat Level**: LOW. Research paper, not a product. Overlaps with primd's layer 1 only.

---

### 4. ContextCache (VLDB 2025)

**What**: Context-aware semantic cache for multi-turn queries. Two-stage retrieval with dialogue-context-aware matching.

**Published Performance**:
- +10.9% precision over GPTCache
- +14.8% recall over GPTCache
- ~10x faster than direct LLM invocation

**Strengths**:
- Published in VLDB (top database venue)
- Addresses multi-turn context awareness

**Weaknesses**:
- Not voice-specific
- Not predictive (reactive caching only)
- General-purpose semantic caching layer
- No streaming, no local-first, no SIMD optimization

**Threat Level**: LOW. Different scope — general-purpose caching, not voice retrieval.

---

## Adjacent Competitors (Not Direct)

### Vector Databases

These are primd's data sources, not competitors. primd reads FROM them.

| Database | p50 Latency | p99 Latency | Notes |
|---|---|---|---|
| FAISS HNSW (in-process) | ~2-3ms | varies | Standard ANN baseline |
| USearch | 2.54ms (exact) | — | 20x faster than FAISS exact search |
| Qdrant (self-hosted) | ~3ms | ~25ms | Most popular self-hosted |
| pgvector (HNSW) | ~5ms | ~75ms (50M) | Postgres-native |
| Milvus | ~4ms | ~11ms | GPU acceleration |
| Pinecone (managed) | ~12ms | ~48ms | Includes network latency |
| Weaviate | ~8ms | 10-123ms | Config-dependent |

### Local-First Retrieval

In the same "lineage" as primd but without voice-specific optimizations.

| Tool | Latency | Scale | Notes |
|---|---|---|---|
| sqlite-vec | <4ms (binary) | 100K | Brute-force only, no ANN. Portable. |
| vstash | 20.9ms median | 50K | Hybrid vector + FTS in SQLite. |
| USearch | 2.54ms | 1M | 131K QPS, C++11, many bindings. |

### Memory Layers

Complementary to primd, not competitive.

- **mem0**: Cross-session user memory. primd handles knowledge retrieval, mem0 handles "who is this user."
- **letta**: Long-term memory management for agents. Different scope.
- **supermemory**: Memory layer for AI apps. Not voice-specific.

### Agent Frameworks

Lower in the stack than primd.

- **Pipecat**: Voice agent orchestration. primd is a component within Pipecat.
- **LiveKit Agents**: Same relationship.
- **LangGraph**: Agent orchestration. primd could be a retriever node.

## Competitive Positioning

### One Sentence

primd is the only retrieval runtime that combines local-first co-location + predictive prefetch + streaming partials + LLM-free hot path.

### vs Moss

"Moss eliminates the network hop. primd eliminates the network hop AND predicts what you'll ask next."

### vs VoiceAgentRAG

"VoiceAgentRAG uses an LLM to predict. primd uses a Markov matrix — 20,000x faster, zero API cost, no GPU required."

### vs Everything Else

"Vector databases are for storage. primd is for retrieval at voice speed. You keep your Qdrant — primd sits in front of it."
