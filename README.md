# primd

**The predictive turn-cache for real-time conversational AI.**

primd is a 10 MB Apache-2.0 Rust runtime that hides retrieval latency inside the STT and TTS phases of your voice agent. It starts retrieving while the user is still speaking, predicts the next turn during TTS playback, and serves repeats from a sub-microsecond cache.

```
   STT partials  ─►  observe_partial   (speculative scan, scoped by predicted events)
                          │
   end of speech ─►  finalize          (1.6 µs cache hit if speculation matched)
                          │
   TTS playback  ─►  warm_next         (predictor primes next turn's scope)
```

## The problem

Voice AI has a dead-air problem. Every other pipeline component has broken its latency wall:

| Component | Best-in-class (2026) |
|---|---|
| STT (Deepgram Nova-3) | <200 ms |
| TTS (Cartesia Sonic-3) | 40 ms |
| LLM TTFT (Groq) | <100 ms |
| **Retrieval** | **50–300 ms** |

A typical vector-DB query eats the entire ~200 ms voice budget before the LLM even starts generating. That's the pause that breaks the illusion of natural conversation.

## The wedge

primd is not "faster semantic search." It's a runtime that **time-shifts retrieval out of the critical path** by exploiting the timing structure of a voice turn:

- **STT emits partial transcripts** → start scanning before the user is done speaking
- **TTS playback creates a 1–3 s window** → pre-warm the next turn's scope
- **Conversations follow topic chains** → topic-continuation queries are zero-scan repeats

The underlying retrieval primitive is 256-bit binary signatures + SIMD Hamming distance (AVX-512 VPOPCNTDQ, AVX2 VPSHUFB nibble-lookup, scalar fallback). 100k docs scanned in ~100 µs. For high-recall workloads primd hands off to your existing vector DB; its job is to make that handoff happen as rarely as possible.

## Benchmarks

Reproducible: `cargo bench --bench voice_session`. Workload models a Pipecat session: 200 utterances over 20 canonical intents, 4 partial transcripts per turn, 100k-doc corpus across 50 events.

| Phase | What it does | primd p50 | primd p95 | Naive p50 |
|---|---|---|---|---|
| `observe_partial` | speculative scan during STT | 108 µs | 199 µs | — |
| **`finalize`** | **end-of-speech retrieval** | **1.6 µs** | **2.8 µs** | **157.8 µs** |
| `warm_next` | predictor + scope union during TTS | 222 µs | 289 µs | — |

**98× faster than a naive SIMD scan at the user-visible `finalize`.** 100% speculative-cache hit rate on this workload — every end-of-speech query was already answered before the user finished talking. The 1.6 µs is a cache lookup, not magic; it's the cost of work already done during `observe_partial`.

For reference: Qdrant's best-in-class managed vector DB reports **4 ms p50** at 1 M vectors. primd's `finalize` p50 is ~2,500× faster — because most of the retrieval has already happened before `finalize` is called.

## Where primd sits in the 2026 voice-AI stack

```
┌─────────────────────────────────────┐
│  THE MODEL                          │
│  Pipecat / LiveKit pipelines        │  ← Pipecat, LiveKit, Vapi, Retell
│  TML-Interaction-Small, Moshi,      │     (listens, speaks, decides when to retrieve)
│  GPT-Realtime, Gemini Live          │
└──────────────┬──────────────────────┘
               │ delegates retrieval
               ↓
┌─────────────────────────────────────┐
│  THE INTEGRATION PROTOCOL           │
│  MoshiRAG <ret> token, Pipecat      │  ← Kyutai, Pipecat, LiveKit
│  FrameProcessor, LiveKit agent      │     (knows when knowledge is needed)
│  plugins, TM's "background agent"   │
└──────────────┬──────────────────────┘
               │ sends query, awaits context
               ↓
┌─────────────────────────────────────┐
│  THE RETRIEVAL BACK-END             │
│  primd                              │  ← us
│                                     │     (actually returns docs, fast)
└─────────────────────────────────────┘
```

Pipecat's `FrameProcessor` API and Kyutai's MoshiRAG retrieval contract both leave the retrieval back-end open. primd fills exactly that slot, Apache-2.0, Rust-native, and shipping.

## Differentiation

| | primd | Moss / InferEdge | Qdrant / Pinecone | Mem0 / Letta |
|---|---|---|---|---|
| **Layer** | retrieval runtime | semantic search runtime | vector database | chat memory |
| **Latency target** | < 200 µs at finalize | < 10 ms | 4–50 ms | 100–500 ms |
| **STT-partial speculation** | yes | no | no | no |
| **TTS-phase pre-warming** | yes | no | no | no |
| **Repeat-query delta cache** | yes | no | no | no |
| **License** | Apache-2.0 | PolyForm Shield * | mixed | mixed |
| **Drops into Pipecat / LiveKit** | yes | yes (design-partner) | via wrappers | via wrappers |

\* PolyForm Shield forbids "production or competing commercial use" of the free tier.

**primd is not competing with vector DBs.** It reads from them. It's not competing with chat memory either — Mem0 / Letta handle who-this-user-is across sessions; primd handles what-they-need-right-now in the current turn.

The only direct overlap is **Moss / InferEdge** (YC F25, sub-10 ms semantic search, Pipecat/LiveKit design partners). Moss has lower raw scan latency on closed-source semantic search; primd has the entire predictive layer Moss doesn't — speculation during STT, pre-warming during TTS, delta cache for topic continuation — plus an Apache-2.0 license that lets you ship without a lawyer.

## Quick start

```bash
cargo build --release -p primd-cli

./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq \
  --embedder hashed

./target/release/primd serve \
  --index /tmp/primd-faq \
  --bind 127.0.0.1:8080
```

Stateless query (the slow path — doesn't use any of the predictive layers):

```bash
curl -s -X POST http://127.0.0.1:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"is there a free trial","top_k":3}'
```

OpenAI-compatible call (the drop-in path for MoshiRAG and any OpenAI-shaped client; see [docs/integrations/moshirag.md](docs/integrations/moshirag.md)):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"primd",
    "messages":[{"role":"user","content":"is there a free trial"}],
    "user":"session-id",
    "top_k":3
  }'
```

Session flow (the path that actually beats vector DBs):

```bash
# during STT — feed partial transcripts as they arrive
curl -X POST http://127.0.0.1:8080/session/demo/observe \
  -d '{"text":"what about pri","top_k":3}'

# end of speech — near-instant if speculation matched
curl -X POST http://127.0.0.1:8080/session/demo/finalize \
  -d '{"text":"what about pricing","top_k":3}'

# during TTS — pre-warm the next likely turn
curl -X POST http://127.0.0.1:8080/session/demo/warm -d '{}'
```

## Architecture

Four layers compose in `QueryContext` (`primd-core/src/query_context.rs`):

1. **Streaming partials** — speculative scan during STT (`observe_partial`)
2. **Binary signature index** — 256-bit signatures, SIMD Hamming scan, event-scoped gather + rescan
3. **Markov predictor + prefetch** — pre-warms next likely scope during TTS (`warm_next`)
4. **Predictive-coding delta cache** — sub-µs short-circuit for topic-continuation queries

Per-layer docs:
- [Layer 1 — streaming partials](docs/architecture/layer-1-streaming.md)
- [Layer 2 — binary signature index](docs/architecture/layer-2-index.md)
- [Layer 3 — Markov predictor + prefetch](docs/architecture/layer-3-prediction.md)
- [Layer 3 (v0.2) — Successor Representation predictor](docs/architecture/successor-representation.md)
- [Layer 4 — predictive coding delta cache](docs/architecture/layer-4-coding.md)
- [Overview](docs/architecture/overview.md)

## Status (v0.1.0)

Shipping:

- SIMD binary signature search (AVX-512 VPOPCNTDQ → AVX2 VPSHUFB → scalar) over event-scoped corpora
- `QueryContext` session runtime: observe / finalize / warm / reset
- Session-aware HTTP API
- Variable-order Markov predictor with half-life time decay, persistence, smoothing
- Predictive-coding delta cache
- `pipecat-primd` Python package (`FrameProcessor` + async client)
- Voice-realistic benchmark harness

Roadmap (v0.2 — in progress):

- ✅ `NextTurnPredictor` trait (foundation for swappable predictors)
- ✅ **MoshiRAG back-end adapter** — OpenAI-compatible `/v1/chat/completions` endpoint so MoshiRAG can swap its 3 s vLLM call for primd's sub-200 µs response with one env var change
- ✅ **Successor Representation predictor** in new `primd-sr` crate (tabular variant + Hybrid SR+Markov wrapper). Enable with `primd serve --predictor hybrid`.
- **Real per-event HNSW shards** (currently the event-scoped path is a SIMD gather + subset rescan, not HNSW; see [roadmap](docs/plan/roadmap.md))
- Public benchmark vs Moss + Qdrant + Pinecone at the *finalize* event
- A/B harness measuring SR vs Markov speculative-cache hit-rate lift

Roadmap (v0.3):

- `livekit-primd` packaged plugin
- **Hippocampus Signature DWM cold tier** (first-mover Rust port of arXiv:2602.13594)
- WASM / browser target
- Trust primitives — confidence scores, refusal-on-uncertainty

## What primd isn't

- **Not a vector database.** Reads from yours (Qdrant, pgvector, parquet files).
- **Not chat memory.** Use [Mem0](https://mem0.ai) or [Letta](https://letta.com) for cross-session user memory. primd retrieves knowledge for the *current* question.
- **Not an agent framework.** Lower in the stack than LangGraph or Pipecat itself.
- **Not a high-recall retrieval system at extreme scale.** Binary signatures trade some recall for speed; primd targets 10 k–1 M doc corpora where sub-millisecond response matters more than 99th-percentile semantic recall. For larger or recall-sensitive workloads, primd hands off to your vector DB.

## Why now

Three signals from the last six months, none of which existed when primd was first prototyped:

1. **Kyutai shipped moshi-rag** (April 2026) — open-sourced the full-duplex retrieval-injection protocol and left the back-end slot generic. Their own docs flag retrieval as the latency bottleneck.
2. **Thinking Machines announced interaction models** (May 11, 2026) — frontier validation of the background-agent architecture, with explicit acknowledgement that long-session context management is unsolved.
3. **Production voice (Pipecat, LiveKit, Vapi) is mainstream** — not research anymore. Pipecat alone has thousands of production deployments. All of them pause on retrieval.

The category is now legible. The back-end slot is empty. primd is the back-end.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Strategy 2026-05](docs/business/strategy-2026-05.md) — why the positioning is what it is
- [Roadmap](docs/plan/roadmap.md)
- [Competitive landscape](docs/business/competitive-landscape.md)
- [Positioning & GTM](docs/business/positioning.md)
- [Gap analysis](docs/business/gap-analysis.md)
- [MoshiRAG back-end adapter](docs/integrations/moshirag.md) — OpenAI-compatible `/v1/chat/completions` drop-in for MoshiRAG and any other OpenAI-shaped client

## Citing

If you use primd in research, cite both the underlying ideas (predictive map / hippocampal retrieval, dual-agent voice RAG) and this implementation:

```
@misc{salesforce2026voiceagentrag,
  title  = {VoiceAgentRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures},
  author = {Salesforce AI Research},
  year   = {2026},
  eprint = {2603.02206},
  archivePrefix = {arXiv}
}

@misc{kyutai2026moshirag,
  title  = {MoshiRAG: Real-Time RAG for Full-Duplex Speech Dialogue},
  author = {Kyutai},
  year   = {2026},
  eprint = {2604.12928},
  archivePrefix = {arXiv}
}
```

## License

Apache-2.0

---

*built by rohan. mumbai. [github](https://github.com/rohansx) · [x](https://x.com/rohansxd)*
