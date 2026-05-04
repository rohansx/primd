# primd

**retrieval, already there when you ask.**

primd is a sub-millisecond semantic retrieval runtime for voice and conversational AI. It starts answering the query before the user has finished speaking it — modeled on how the human brain does recall, not how databases do lookup.

## The Problem

Voice AI has a dead-air problem. Every other pipeline component has broken its latency barrier:

| Component | Best-in-class (2026) |
|---|---|
| STT (Deepgram Nova-3) | <200ms |
| TTS (Cartesia Sonic 3) | 40ms |
| LLM TTFT (Groq) | <100ms |
| **Retrieval** | **50-300ms** |

Retrieval is the last bottleneck. It's the pause that breaks the illusion of natural conversation.

## The Solution

primd eliminates that pause with four brain-inspired shortcuts:

1. **Start early** — begins retrieving on partial transcripts, not end-of-utterance
2. **Search smart** — 256-bit binary signatures scan 1M docs in <0.5ms using SIMD
3. **Predict next** — Markov transition matrix prefetches likely follow-up answers during TTS playback
4. **Skip repeats** — topic-continuation detection returns cached results in ~50 microseconds

### Target Latency

| Scenario | Latency |
|---|---|
| Topic continuation (follow-up question) | <0.1ms |
| Predicted topic pivot (cache hit) | ~0.5-1ms |
| Novel query (full retrieval) | ~2-4ms |

For context, a blink takes ~100ms. primd targets retrieval 100-1000x faster than a blink.

## What You Get

- A **single Rust binary** (~8-12 MB)
- A **CLI** for indexing, querying, serving, and training predictors
- A **session-aware HTTP API** for partials, finalize, and warm-next flows
- A **Pipecat example** wired to `primd serve`
- **Apache-2.0 licensed**

## Quick Example

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

Session query flow:

```bash
curl -s -X POST http://127.0.0.1:8080/session/demo/observe \
  -H 'Content-Type: application/json' \
  -d '{"text":"what about pri","top_k":3}'

curl -s -X POST http://127.0.0.1:8080/session/demo/finalize \
  -H 'Content-Type: application/json' \
  -d '{"text":"what about pricing","top_k":3}'

curl -s -X POST http://127.0.0.1:8080/session/demo/warm \
  -H 'Content-Type: application/json' \
  -d '{}'
```

## What It's For

- **SDR bots** — eliminate dead air after "what's your pricing?"
- **Customer support voice agents** — follow-up answers land instantly
- **In-app voice copilots** — runs in the browser via WASM
- **Healthcare intake, scheduling, dispatch** — any voice AI where pauses kill the conversation

## What It Isn't

- **Not a vector database.** Reads from yours (Qdrant, pgvector, parquet files). It's a runtime on top.
- **Not chat memory.** Use mem0 or letta for cross-session user memory. primd retrieves knowledge for the current question.
- **Not an agent framework.** Lower in the stack than LangGraph or Pipecat itself.

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Technical Specification](docs/tech-spec/technical-specification.md)
- [Roadmap & Phases](docs/plan/roadmap.md)
- [Gap Analysis](docs/business/gap-analysis.md)
- [Competitive Landscape](docs/business/competitive-landscape.md)

## Status

Alpha. The repo now includes:

- SIMD binary signature search
- event-scoped hierarchical rerank over shard scopes
- `QueryContext` session runtime in `primd-core`
- session-aware HTTP endpoints
- Markov predictor training and persistence
- Pipecat + Sarvam example integration

Still planned:

- true per-event HNSW shards instead of shard-local subset rescans
- packaged Python/TypeScript SDKs
- LiveKit plugin
- WASM/browser target

## License

Apache-2.0

---

*built by rohan. mumbai. [github](https://github.com/rohansx) · [x](https://x.com/rohansxd)*
