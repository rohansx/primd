# Positioning & Go-to-Market

> Last updated 2026-05-13. Supersedes the v0.1 positioning that framed primd as "the voice-AI retrieval runtime." The new wedge is sharper and names the unit of work.

## Category

**Predictive turn-cache** — a sub-category of retrieval runtime. Not "vector database," not "memory layer," not "agent framework."

A predictive turn-cache is a specialized execution layer that sits between a voice agent's pipeline (or a full-duplex model's background-agent slot) and its knowledge source. It exists because a voice *turn* has timing structure — STT partials, end-of-speech, TTS playback — that a vector database cannot exploit.

## One-line pitch

**primd is the predictive turn-cache for real-time conversational AI: a 10 MB Apache-2.0 Rust runtime that hides retrieval latency inside the STT and TTS phases of your voice agent.**

This sentence is what a Pipecat developer can repeat to their VP Eng.

- *Unit of work* = a turn (not a query, not a session).
- *Mechanism* = hiding inside STT/TTS phases (not "faster scan").
- *Competitor it isn't* = Moss's general semantic search; Mem0/Letta's cross-session memory.
- *Format constraint* = 10 MB Rust binary, Apache-2.0 (vs Moss's PolyForm Shield).

## Three bullets

1. **Sub-200 µs at finalize** — speculative retrieval during STT means most user-visible queries return from a sub-microsecond cache. The 1.6 µs benchmark is a cache lookup, not magic; it's the cost of work already done.
2. **No LLM in the critical path** — predictor is currently a variable-order Markov chain with half-life decay, ~µs cost. v0.2 replaces this with a Successor Representation (TD(0)-trained low-rank predictive map). Both are local CPU work, no API calls.
3. **Apache-2.0, single Rust binary** — drops into Pipecat / LiveKit / Vapi with one import. Closed-source competitors (Moss, Cartesia, ElevenLabs) cannot match this licensing posture without restructuring their business.

## Positioning statements

### vs Moss / InferEdge (the direct competitor)

"Moss has lower raw scan latency on closed-source semantic search. primd has the entire predictive layer Moss doesn't — speculation during STT, pre-warming during TTS, delta cache for topic continuation. And primd is Apache-2.0; Moss is PolyForm Shield."

PolyForm Shield 1.0.0 forbids "production or competing commercial use" of Moss's free tier. For any voice startup planning to charge money, this is a structural obstacle primd doesn't impose.

### vs vector databases (Qdrant, Pinecone, pgvector, Weaviate)

"Vector databases are for storage. primd is for retrieval at voice speed. You keep your Qdrant — primd sits in front of it and makes it feel instant, by issuing the query before the user finishes speaking."

### vs Mem0 / Letta / Zep (the wrong category to compete in)

primd does not compete with chat memory. Mem0 / Letta / Zep handle who-this-user-is *across* sessions; primd handles what-they-need-right-now *within* the current turn. Both are needed in a production voice agent; they sit at different layers.

We have explicitly considered and rejected pivoting to agent memory:
- Mem0: $24 M total funding (Basis Set Series A Oct 2025), 41 k GitHub stars, AWS Agent SDK exclusive.
- Letta: $10 M seed at $70 M post (Felicis, Sept 2024), MemGPT pedigree from Berkeley.
- Letta's own benchmark shows a plain filesystem scoring 74 % on LoCoMo — the accuracy moat in this category is fragile.
- TML-Interaction-Small's "background model" surface is exactly where Mem0/Letta/Zep operate, so the category carries frontier-lab tail risk.

We will *integrate with* memory layers, never compete with them.

### vs MoshiRAG's default back-end (vLLM-served 27 B LLM)

"MoshiRAG's reference back-end is a generic LLM via vLLM with 1–3 s latency. primd is a drop-in replacement at sub-200 µs. We're building an OpenAI-compatible adapter so the swap is an env var change."

### vs full-duplex models (TML-Interaction-Small, future Moshi, Gemini Live 3)

"Full-duplex models eat STT/TTS as discrete pipeline phases, which removes our current hook points. But they still need a retrieval back-end — Kyutai's MoshiRAG keeps it explicitly separate, and TM names a 'background agent' for retrieval and tool use. primd evolves toward *model-phase* hooks (during generation pauses, during the background agent's slow-think window) rather than disappearing."

## Target personas

### Primary: Voice AI engineer at a Pipecat / LiveKit / Vapi shop

- Building on Pipecat, LiveKit Agents, or Vapi
- Frustrated by dead air during RAG lookups
- Wants a drop-in solution, not a platform migration
- Evaluates by: latency benchmarks at the *finalize* event, ease of integration, open source licensing
- Will install via `pip install pipecat-primd` or a similar single-line plugin

### Secondary: Open-source full-duplex model adopter

- Running Kyutai's MoshiRAG, evaluating TML-Interaction-Small when GA
- Pain: default vLLM retrieval back-end is 1–3 s
- Evaluates by: drop-in compatibility with MoshiRAG's text-in/text-out contract, latency

### Tertiary: Voice-AI platform (Vapi, Retell, Bland)

- Building a managed voice-agent product
- Needs to differentiate on latency at the p99
- Evaluates by: per-call cost impact, resource overhead, licensing

## Go-to-market strategy

### Phase 1 — Developer adoption (days 1–45)

1. **Ship the differentiated benchmark.** Public, reproducible bench vs Moss + Qdrant + Pinecone at the *finalize* event, not at `/query`. Open-source the harness. (Days 1–14.)
2. **Pipecat reference example.** Get `pipecat-primd` into Pipecat's official examples directory. (Days 1–14.)
3. **LiveKit Agents plugin.** Submit `livekit-primd` to LiveKit's plugin registry. (Days 15–45.)
4. **MoshiRAG back-end adapter.** OpenAI-compatible `/v1/chat/completions` endpoint on `primd serve` so MoshiRAG users can swap the vLLM call with one env var. (Days 15–45.)
5. **Technical post:** "Why your voice agent is slow even when your vector DB is fast." Frame around pipeline-phase scheduling; contrast with Moss's "10 ms semantic search" claim (which doesn't help if `retrieve()` is called after STT finalize).
6. **Hacker News + Show HN** with post + benchmark + 10 MB binary.

Success metric: 500 GitHub stars, 5 voice-AI shops running primd in pre-production by day 45.

### Phase 2 — Technical moat (days 46–90)

1. **Successor Representation predictor (v0.2 headline).** Replace variable-order Markov with TD(0)-trained low-rank SR (k=32) in a new `primd-sr` crate. Hybrid SR+Markov fallback for backward compatibility.
2. **Real per-event HNSW shards.** Close the credibility gap with the current "shard-local subset rescan" implementation.
3. **Hippocampus Signature DWM cold tier.** First-mover Rust port of arXiv:2602.13594 (paper has no public code release as of May 2026). Long-session memory in a single binary.
4. **Public LoCoMo / LongMemEval numbers** — not to compete with Hippocampus on memory, but to prove primd doesn't regress on long-horizon recall when the cold tier is engaged.

Success metric: SR predictor lifts speculative-cache hit rate by ≥15 % absolute on multi-turn flows; first design-partner letter from a voice-AI shop.

### Phase 3 — Distribution & enterprise (months 4–8)

1. **Pipecat Cloud / Daily integration** — ship as a built-in option in Daily's managed service.
2. **LiveKit marketplace listing.**
3. **Voice-platform integrations** — Vapi, Retell, Bland.
4. **Enterprise pilot** — 1–2 large deployments with SLA, basis for hosted "primd Cloud" pricing experimentation.

## Pricing strategy (future)

### Open core

| Tier | Price | What you get |
|---|---|---|
| **primd-core** | Free (Apache-2.0) | Everything for self-hosted deployment |
| **primd Cloud** | Usage-based | Managed service, per-domain pretrained predictors, SLA |
| **Enterprise** | Custom | Custom domain training, dedicated support, priority features |

### primd Cloud (conceptual)

- Per-query pricing: $X per 1 M queries
- Per-`warm_next` pricing: $X per 1 M prefetch operations
- Storage: $X per GB / month for indexed corpus
- Free tier: 100 k queries / month

## What we will NOT do

1. **Don't call it a vector database.** It's not. Calling it one invites comparison with Pinecone / Qdrant on dimensions where they're stronger (durability, scale, ecosystem). primd reads from them.
2. **Don't widen to agent memory.** Mem0 ($24 M) and Letta ($10 M seed at $70 M) own this category. We integrate; we don't compete.
3. **Don't market to general RAG users.** primd is specifically for real-time conversational latency. General RAG users should use their existing vector database directly.
4. **Don't compete on recall.** primd's recall is good (97 %+ with rescore) but not differentiated. The differentiation is *time-shifted* retrieval, not faster retrieval.
5. **Don't build a platform.** Stay narrow: predictive turn-cache. Don't become an agent framework, a memory layer, or a voice pipeline. Those layers already have winners.
6. **Don't ship closed-source modules.** Apache-2.0 across the board is a structural advantage no Moss marketing can erase.

## Decision thresholds that would change this positioning

| Signal | What it would mean | Response |
|---|---|---|
| > 20 paying voice-AI customers by day 90 | Option A validated | Raise seed on the predictive-turn-cache thesis |
| Moss raises Series A > $40 M before day 90 OR signs Pipecat / LiveKit as exclusive partner | License moat matters more than latency moat | Stay Apache-2.0; double-down on Pipecat/LiveKit upstream contributions |
| A full-duplex model with native retrieval ships before EOY 2026 (TML, Gemini Live 3) | STT/TTS hook points start to disappear | Evolve to model-phase hooks (during generation pauses, background-agent slow-think window) |
| CloakPipe (Rohan's other project) lands $1 M+ angel before primd hits 500 GitHub stars | Effort allocation question | Redirect primary effort to CloakPipe; keep primd as a credibility artifact (RustConf talk, maintained binary) |
