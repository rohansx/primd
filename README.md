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
- **Drop-in plugins** for Pipecat and LiveKit
- **Python and TypeScript SDKs**
- **WASM build** for in-browser deployment
- **Apache-2.0 licensed**

## Quick Example

```python
from primd import Index, QueryContext

idx = Index.open("/var/lib/primd/corpus")
ctx = idx.session()

# Feed partial transcripts as they arrive from STT
async for partial in stt_stream:
    ctx.observe(partial)

# By end-of-utterance, results are already waiting
results = await ctx.finalize()

# While TTS plays the response, prefetch next likely answers
await ctx.warm_next()
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

Pre-build. MVP targeting 12 weeks to public benchmark.

## License

Apache-2.0

---

*built by rohan. mumbai. [github](https://github.com/rohansx) · [x](https://x.com/rohansxd)*
