# Positioning & Go-to-Market

## Category

**Retrieval runtime** — a new category. Not "vector database," not "memory layer," not "agent framework."

A retrieval runtime is a specialized execution layer that sits between a voice agent and its knowledge base, optimized for the access patterns of real-time conversation: streaming input, predictable follow-ups, and zero tolerance for latency.

## Tagline

**"retrieval, already there when you ask."**

Alternatives:
- "the voice-ai retrieval runtime."
- "search that starts before you finish talking."
- "sub-millisecond retrieval for voice ai."

## One-Line Pitch

primd is the first retrieval system that doesn't just search faster — it searches *earlier*, because it's modeled on how the brain actually does recall.

## Three Bullets

1. **Sub-millisecond retrieval on the hot path** — predictive prefetch warms the cache during the TTS window
2. **No LLM in the critical path, no network hop** — Markov prediction + binary signature scan, pure CPU
3. **Apache-2.0, single Rust binary** — drops into Pipecat/LiveKit with one import

## Positioning Statements

### vs Vector Databases

"Vector databases are for storage. primd is for retrieval at voice speed. You keep your Qdrant — primd sits in front of it and makes it feel instant."

### vs Moss.dev

"Moss eliminates the network hop. primd eliminates the network hop AND predicts what you'll ask next — so the answer is cached before the question arrives."

### vs VoiceAgentRAG

"VoiceAgentRAG uses an LLM to predict follow-up questions. primd uses a Markov matrix — 20,000x faster, zero API cost, no GPU required."

### vs "Just use FAISS"

"FAISS gives you 2-5ms in-process. primd gives you <0.5ms on 80% of queries — because it was already done."

## Target Personas

### Primary: Voice AI Engineer

- Building on Pipecat or LiveKit
- Frustrated by dead air during RAG lookups
- Wants a drop-in solution, not a platform migration
- Evaluates by: latency benchmarks, ease of integration, open source

### Secondary: Voice AI Platform

- Building a managed voice agent product (Vapi, Retell, Bland)
- Needs to differentiate on latency
- Evaluates by: p99 impact, resource overhead, licensing

### Tertiary: Edge/On-Device Developer

- Building local voice assistants (privacy-sensitive, offline-capable)
- Needs small binary, WASM support
- Evaluates by: binary size, WASM performance, no cloud dependency

## Go-to-Market Strategy

### Phase 1: Developer Adoption (Launch)

**Channel**: Open source + community

1. **Pipecat plugin** — puts primd in front of the largest open-source voice agent community (10K+ stars)
2. **LiveKit plugin** — enterprise credibility (powers ChatGPT Voice)
3. **arxiv preprint** — technical credibility for engineers who read papers
4. **HN Show HN** — developer discovery
5. **Benchmark repo** — `make bench` reproducibility builds trust

**Success metric**: 500 GitHub stars in first month, 50 active installations.

### Phase 2: Use Case Validation (Months 2-4)

**Channel**: Direct outreach + content

1. **SDR bot case study** — partner with 1-2 AI SDR companies to measure impact
2. **Before/after latency comparison** — concrete numbers from production deployment
3. **Blog series**: "Voice AI Latency Deep Dive" (technical content marketing)
4. **Conference talks**: Voice AI meetups, Pipecat/LiveKit community events

**Success metric**: 3 production deployments, 1 published case study.

### Phase 3: Platform Partnerships (Months 5-8)

**Channel**: B2B partnerships

1. **Pipecat Cloud integration** — ship as a built-in option in Daily's managed service
2. **LiveKit marketplace** — listed as a recommended retrieval plugin
3. **Voice AI platform integrations** — Vapi, Retell, Bland
4. **Enterprise pilot** — 1-2 large deployments with SLA

**Success metric**: 1 platform partnership, $X ARR from enterprise pilots.

## Pricing Strategy (Future)

### Open Core

| Tier | Price | What You Get |
|---|---|---|
| **primd-core** | Free (Apache-2.0) | Everything for self-hosted deployment |
| **primd Cloud** | Usage-based | Managed service, pretrained domain matrices, SLA |
| **Enterprise** | Custom | Custom domain training, dedicated support, priority features |

### primd Cloud Pricing (Conceptual)

- Per-query pricing: $X per 1M queries
- Per-warm_next pricing: $X per 1M prefetch operations
- Storage: $X per GB/month for indexed corpus
- Free tier: 100K queries/month

## What NOT to Do

1. **Don't call it a vector database.** It's not. Calling it one invites comparison with Pinecone/Qdrant on dimensions where they're stronger (durability, scale, ecosystem).
2. **Don't market to general RAG users.** primd is specifically for voice/conversational latency-sensitive workloads. General RAG users should use their existing vector database.
3. **Don't compete on recall.** primd's recall is 97%+ (good) but not differentiated. The differentiation is latency and prediction.
4. **Don't build a platform.** Stay narrow: retrieval runtime. Don't become an agent framework, a memory layer, or a voice pipeline.
