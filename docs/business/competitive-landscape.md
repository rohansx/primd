# Competitive Landscape

> Last updated 2026-05-13. Numbers in this doc are as-of that date; both voice-AI infra and agent-memory categories are moving fast, expect drift on every figure within 6 months.

## Market sizing

- Conversational AI: $11.58 B (2024) → $41.39 B (2030) at 23.7 % CAGR (Grand View Research)
- AI voice agents specifically: $2.4 B (2024) → $47.5 B (2034) at 34.8 % CAGR (Market.us)
- Voice-AI startup funding in 2025: $1.2 B (PitchBook), up from $680 M in 2024

primd targets the bottom-most layer of the voice stack (the retrieval back-end). At 0.5 % capture of a $4 B mid-2027 voice-AI TAM, that's a $20 M ARR company.

## Direct competitors

### 1. Moss / InferEdge Inc. (YC F25)

**Threat level: HIGH.** The only company shipping retrieval-shaped infrastructure into the same Pipecat/LiveKit slot as primd.

| Field | Value |
|---|---|
| What | Closed-source semantic-search runtime, Rust + WebAssembly core |
| Latency | Sub-10 ms end-to-end retrieval (including embedding inference) at 100 k docs |
| Stage | 3 paying customers, 6 enterprise design partners, ~100 % WoW growth claim |
| Backers | YC F25, San Francisco |
| Founders | Sri Raghu Malireddi (ex-ML Lead Grammarly + Microsoft), Harsha Nalluru (ex-Microsoft Azure SDK) |
| Partners | Pipecat (Daily.co), LiveKit — named in YC launch post |
| License | **PolyForm Shield 1.0.0** — forbids "production or competing commercial use" of the free tier |
| Distribution | npm package (250 k+ downloads ≠ enterprise installs; this is the JS SDK count) |

**Where Moss is stronger:** raw scan latency on cold queries, brand momentum from YC, design-partner relationships with Pipecat (Daily) and LiveKit.

**Where primd is structurally stronger:**

1. **Pipeline-phase scheduling** — Moss has no STT-partial or TTS-phase awareness. Their "10 ms" is measured at the point retrieval is *called*. By the time `retrieve()` runs in a Moss-based pipeline, the user has already finished speaking and the STT finalize event has already fired. primd has been retrieving for the previous 200 ms and is delivering a cache hit at 1.6 µs.
2. **Apache-2.0 vs PolyForm Shield** — Pipecat (MIT), LiveKit (Apache-2.0), and Vapi (mixed) all integrate Apache-licensed components without legal review. Moss's PolyForm explicitly forbids competing commercial use, which is a hard blocker for any voice-AI startup planning to monetize.
3. **Open benchmarks** — Moss publishes no independent benchmarks, no p99 numbers, no recall figures. primd ships `cargo bench --bench voice_session` with reproducible numbers.

### 2. VoiceAgentRAG (Salesforce AI Research, March 2026)

**Threat level: LOW (research prototype, not a product).**

| Metric | Value |
|---|---|
| Traditional RAG avg retrieval | 110.4 ms |
| Cache-hit retrieval | 0.35 ms |
| Cache hit rate (overall) | 75 % (95 % best case) |
| Architecture | LLM-powered predictor ("Slow Thinker") + foreground retrieval ("Fast Talker") |

Validates the dual-agent thesis primd ships against. Same idea space, different bet: VoiceAgentRAG uses an LLM (GPT-4o-mini, 200–500 ms per prediction) for prediction; primd uses Markov / SR (~µs). Tiny test corpus (76 chunks). Open-source (Apache-2.0).

### 3. StreamRAG (Meta + CMU, October 2025)

**Threat level: LOW.** Single-turn speculative retrieval during speech. Overlaps with primd's Layer 1 only. No cross-turn prediction, no local-first architecture, 500 ms block size is coarse.

### 4. ContextCache (VLDB 2025)

**Threat level: LOW.** General-purpose semantic cache, not voice-specific, no streaming, no SIMD optimization.

## Complementary (not competitive)

### Hippocampus / Dynamic Wavelet Matrix (arXiv:2602.13594, Feb 14 2026)

**Status:** Anonymized for MLSys double-blind review. Authors inferred from arXiv metadata as Yi Li, Lianjie Cao, Faraz Ahmed, Puneet Sharma (Hewlett Packard Labs) + Bingzhe Li (UT Dallas).

**What it solves:** Long-horizon agent memory at O(n·d/w) compressed-domain retrieval cost. Random-indexing-based signatures (zero LLM tokens at construction), Content + Signature DWMs, rank/select primitives. Builds memory in 6.70 min with 0 LLM tokens vs MemGPT 59.49 min (Appendix E).

**Why complementary to primd:** Hippocampus optimizes the *write* side at multi-day scale; primd optimizes the *wait* side of a voice turn. Both use ~256-bit binary signatures + Hamming scan — that's the only architectural overlap. Hippocampus has zero pipeline-phase awareness.

**primd's opening:** **No public code release as of May 2026.** Double-blind submission norms preclude release before the MLSys decision (likely June–August 2026). A clean Rust port of the Signature DWM kernel as `primd::cold::SignatureDWM` ships before the authors' reference implementation. v0.3 deliverable, time-sensitive. See [strategy-2026-05.md](strategy-2026-05.md) for full adopt/reject/extend analysis.

### Kyutai MoshiRAG (April 30, 2026)

**Status:** Open-source, Rust-implemented (rustymimi + Rust inference stack).

**Why complementary:** Defines a text-in/text-out retrieval contract; explicitly keeps the retrieval back-end as a separate component. **primd is a natural retrieval back-end for MoshiRAG.** Their default reference back-end is a generic LLM via vLLM at 1–3 s latency; primd's sub-200 µs response is a 4–5 order-of-magnitude improvement at the same interface.

**v0.2 deliverable:** OpenAI-compatible `/v1/chat/completions` adapter on `primd serve` so the swap is a single env var change.

## Adjacent (not direct)

### Vector databases

primd reads from these, doesn't compete.

| Database | p50 latency | p99 latency | Notes |
|---|---|---|---|
| FAISS HNSW (in-process) | ~2–3 ms | varies | Standard ANN baseline |
| USearch | 2.54 ms (exact) | — | 20× faster than FAISS exact search |
| Qdrant (self-hosted) | ~3 ms | ~25 ms | Most popular self-hosted |
| pgvector (HNSW) | ~5 ms | ~75 ms (50 M) | Postgres-native |
| Milvus | ~4 ms | ~11 ms | GPU acceleration |
| Pinecone (managed) | ~12 ms | ~48 ms | Includes network latency |
| Weaviate | ~8 ms | 10–123 ms | Config-dependent |

### Chat memory (the wrong category to compete in)

Complementary to primd; primd handles within-turn knowledge retrieval, these handle across-session user memory.

| Project | Funding | GitHub stars | Notes |
|---|---|---|---|
| Mem0 | $24 M total (Series A $20 M Basis Set, Oct 2025) | ~41 k | AWS Agent SDK exclusive; 186 M API calls / month in Q3 2025 |
| Letta | $10 M seed at $70 M post (Felicis, Sept 2024) | ~10 k | MemGPT pedigree, Berkeley founders Packer + Wooders |
| Zep | $3.3 M seed | ~24 k (Graphiti) | Bi-temporal knowledge graph; 94.8 % on Zep's own DMR bench |
| supermemory, MemMachine, MemPalace, Mnemos, Cognee, LangMem | various | mostly sub-1 k | Long tail; no traction signal visible in public sources |

**Why we don't pivot here:** Letta's own benchmark shows a plain filesystem scoring 74 % on LoCoMo, beating most specialized libraries. The accuracy moat is fragile. Mem0's distribution head start is decisive for any new entrant without $20 M+. And TML-Interaction-Small's "background model" surface is exactly where this category operates, carrying frontier-lab tail risk.

### Voice-AI orchestration & model layer

primd integrates with (not competes with):

| Company | Stage | Funding | Notes |
|---|---|---|---|
| Pipecat (Daily) | Mainstream | — | MIT, thousands of production deployments, primd ships `pipecat-primd` |
| LiveKit | Mainstream | — | Apache-2.0; powers ChatGPT Voice; v0.2 ships `livekit-primd` |
| Vapi | Series B | $72 M ($50 M B May 2026, Peak XV led, ~$500 M valuation) | 1 B+ calls, 1 M+ developers, named Amazon Ring, Intuit, New York Life as customers |
| Retell AI | Managed | — | Black-box stack, $0.07–0.31/min |
| Cartesia | Model layer | $122 M (Oct 2025 $100 M led by KP, Index, Lightspeed, NVIDIA) | Sonic-3 TTS at 90 ms TTFA |
| ElevenLabs | Model layer | $781 M total ($500 M D Feb 2026 at $11 B valuation, Sequoia led) | Andrew Reed on board; a16z 4x stake, ICONIQ 3x |
| Deepgram, AssemblyAI | STT model layer | — | primd reads their transcripts |

### Agent frameworks (lower in the stack)

- **LangGraph** — primd could be a retriever node
- **Pipecat / LiveKit Agents** — primd is a `FrameProcessor` / agent plugin within these

## The frontier-model threat: full-duplex models eating discrete STT/TTS

**TML-Interaction-Small** (Thinking Machines Lab, announced May 11 2026): 276 B parameter MoE / 12 B active, 0.40 s end-to-end latency, encoder-free early fusion. Explicitly architected as "interaction model + async background model."

**Time horizon:** TML is research-preview only, single-vendor (no API yet), and explicitly delegates retrieval/reasoning to a separate background agent. The 18–24 month threat is real; the 6-month threat is not.

**Implication for primd:**

- Short-term (12–18 mo): keep building pipeline-phase hooks against discrete STT/TTS. Production voice (Pipecat, LiveKit, Vapi) is mainstream and isn't going anywhere fast.
- Medium-term (12–24 mo): evolve hooks toward *model-phase* signals — generation pauses, background-agent slow-think windows. Kyutai's MoshiRAG keeps the retrieval back-end *explicitly separate* from the model; that interface is the model-phase hook primd targets.
- Long-term hedge: the SR predictor and Hippocampus cold tier are both useful even without STT/TTS phase signals, because they exploit conversation-structural priors (SR) and long-session memory (DWM). primd remains useful if discrete pipelines disappear.

## Competitive positioning summary

**One sentence:** primd is the only retrieval runtime in the voice stack that combines speculative scanning during STT, predictive pre-warming during TTS, repeat-query delta cache, *and* Apache-2.0 licensing.

**vs Moss:** "Moss has lower raw scan latency. primd has the entire predictive layer Moss doesn't — *and* an Apache-2.0 license that lets you ship to production."

**vs Mem0 / Letta / Zep:** "Different category. They handle user memory across sessions; primd handles knowledge retrieval within the current turn. Use both."

**vs vector DBs:** "Storage layer, not retrieval-runtime layer. You keep your Qdrant; primd sits in front of it."

**vs MoshiRAG's vLLM back-end:** "Four orders of magnitude faster at the same retrieval-back-end interface."

**vs full-duplex models:** "Today's hooks are STT/TTS. Tomorrow's hooks are model-phase. The predictive substrate (SR over event signatures, DWM cold tier) doesn't care which."
