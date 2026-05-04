# primd

**open-source VoiceAgentRAG.** Sub-millisecond predictive retrieval for voice and conversational AI.

primd is a Rust runtime that starts retrieving while the user is still speaking, predicts the next likely answer during TTS, and serves repeat questions from a sub-microsecond cache. It implements the dual-agent *fast-talker / slow-thinker* architecture described in [Salesforce VoiceAgentRAG (arXiv:2603.02206, 2026)](https://arxiv.org/abs/2603.02206) — as a single ~10 MB binary you can drop into Pipecat or LiveKit today.

## The Problem

Voice AI has a dead-air problem. Every other pipeline component has broken its latency barrier:

| Component | Best-in-class (2026) |
|---|---|
| STT (Deepgram Nova-3) | <200 ms |
| TTS (Cartesia Sonic 3) | 40 ms |
| LLM TTFT (Groq) | <100 ms |
| **Retrieval** | **50–300 ms** |

Retrieval is the last unsolved latency wall. A typical vector DB query eats the entire ~200 ms voice budget before the LLM even starts generating. That's the pause that breaks the illusion of natural conversation.

## The Solution

primd eliminates that pause with four brain-inspired shortcuts:

1. **Start early** — speculative retrieval on partial transcripts during STT, not at end-of-utterance
2. **Search smart** — 256-bit binary signatures scan 100k+ docs in microseconds via SIMD, scoped by predicted event
3. **Predict next** — variable-order Markov predictor pre-warms the next likely answer during TTS playback
4. **Skip repeats** — predictive-coding delta cache short-circuits topic-continuation queries with zero scan

## Benchmarks

Reproducible: `cargo bench --bench voice_session`. Workload models a Pipecat session: 200 utterances over 20 canonical intents, 4 partial transcripts per turn, 100k-doc corpus across 50 events.

| Phase | What it does | primd p50 | primd p95 | naive p50 |
|---|---|---|---|---|
| `observe_partial` | speculative scan during STT | 108 µs | 199 µs | — |
| **`finalize`** | **end-of-speech retrieval** | **1.6 µs** | **2.8 µs** | **157.8 µs** |
| `warm_next` | predictor + scope union during TTS | 222 µs | 289 µs | — |

**98× faster than a naive SIMD scan at the user-visible finalize.** 100% speculative-cache hit rate on this workload — every end-of-speech query was already answered before the user finished talking.

For reference, the best-in-class managed vector DB (Qdrant) reports **4 ms p50** at 1M vectors. primd's finalize p50 is **~2,500× faster** than that — because most of the retrieval has already happened before finalize is called.

## Quick Start

```bash
cargo build --release -p primd-cli

./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq \
  --embedder hashed

./target/release/primd train \
  --corpus /tmp/primd-faq \
  --transcripts examples/transcripts.jsonl

./target/release/primd serve \
  --index /tmp/primd-faq \
  --bind 127.0.0.1:8080
```

Stateless query:

```bash
curl -s -X POST http://127.0.0.1:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"is there a free trial","top_k":3}'
```

Session flow (the path that actually beats vector DBs):

```bash
# during STT — feed partial transcripts as they arrive
curl -X POST http://127.0.0.1:8080/session/demo/observe \
  -d '{"text":"what about pri","top_k":3}'

# end of speech — return is near-instant if speculation matched
curl -X POST http://127.0.0.1:8080/session/demo/finalize \
  -d '{"text":"what about pricing","top_k":3}'

# during TTS — pre-warm the next likely turn
curl -X POST http://127.0.0.1:8080/session/demo/warm -d '{}'
```

## Architecture

```
   STT partials  ─►  observe_partial   (speculative scan, scoped by predicted events)
                          │
   end of speech ─►  finalize          (1.6µs cache hit if speculation matched)
                          │
   TTS playback  ─►  warm_next         (Markov predictor primes next turn's scope)
```

The four layers compose in `QueryContext` (`primd-core/src/query_context.rs`). Each is documented separately:

- [Layer 1 — streaming partials](docs/architecture/layer-1-streaming.md)
- [Layer 2 — binary signature index](docs/architecture/layer-2-index.md)
- [Layer 3 — Markov predictor + prefetch](docs/architecture/layer-3-prediction.md)
- [Layer 4 — predictive coding delta cache](docs/architecture/layer-4-coding.md)
- [Overview](docs/architecture/overview.md)

## What You Get

- A single Rust binary (~10 MB)
- A CLI for indexing, querying, serving, and training predictors
- A session-aware HTTP API
- A Pipecat reference integration with Sarvam (STT/LLM/TTS) and Daily transport
- Apache-2.0 license

## What It's For

- **SDR bots** — eliminate dead air after "what's your pricing?"
- **Customer support voice agents** — follow-up answers land instantly
- **In-app voice copilots** — runs as a Pipecat or LiveKit plugin
- **Healthcare intake, scheduling, dispatch** — any voice AI where pauses kill the conversation

## What It Isn't

- **Not a vector database.** Reads from yours (Qdrant, pgvector, parquet files). It's a runtime *on top* of one.
- **Not chat memory.** Use [Mem0](https://mem0.ai) or [Letta](https://letta.com) for cross-session user memory. primd retrieves knowledge for the current question.
- **Not an agent framework.** Lower in the stack than LangGraph or Pipecat itself.

## Status

v0.1.0 — voice retrieval runtime, ready to integrate.

Shipping in this release:
- SIMD binary signature search over event-scoped shards
- `QueryContext` session runtime (observe / finalize / warm)
- session-aware HTTP endpoints
- Markov predictor training and persistence
- predictive-coding delta cache
- Pipecat + Sarvam reference example
- voice-realistic benchmark harness

Planned next:
- packaged `pipecat-primd` and `livekit-primd` plugins
- Python and TypeScript SDKs
- per-event HNSW shards (currently shard-local subset rescans)
- WASM/browser target for in-page voice agents
- trust primitives — confidence scores, dataset freshness, refusal-on-uncertainty

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Technical specification](docs/tech-spec/technical-specification.md)
- [Roadmap](docs/plan/roadmap.md)
- [Gap analysis](docs/business/gap-analysis.md)
- [Competitive landscape](docs/business/competitive-landscape.md)

## Citing

If you use primd in research, cite both the Salesforce paper (which describes the architecture) and this implementation:

```
@misc{salesforce2026voiceagentrag,
  title  = {VoiceAgentRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures},
  author = {Salesforce AI Research},
  year   = {2026},
  eprint = {2603.02206},
  archivePrefix = {arXiv}
}
```

## License

Apache-2.0

---

*built by rohan. mumbai. [github](https://github.com/rohansx) · [x](https://x.com/rohansxd)*
