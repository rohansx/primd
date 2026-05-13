# livekit-primd

**Sub-millisecond predictive retrieval for LiveKit Agents.** Open-source [VoiceAgentRAG](https://arxiv.org/abs/2603.02206).

`livekit-primd` is the LiveKit-Agents-shaped client for [primd](https://github.com/rohansx/primd) — a Rust retrieval runtime that speculates on partial transcripts during STT and pre-warms next-turn answers during TTS.

End-of-utterance retrieval drops from ~157 µs (naive SIMD scan) to **~1.6 µs** when speculation matches — **98× faster**, deterministically, on a 100k-doc corpus.

## Install

```bash
pip install livekit-primd
```

You also need [`primd`](https://github.com/rohansx/primd) running locally:

```bash
cargo install primd-cli
primd index --input examples/faq.jsonl --out /tmp/primd-faq --embedder hashed
primd serve --index /tmp/primd-faq --bind 127.0.0.1:8080
```

## Use

Drop a `PrimdRetriever` next to your LiveKit `VoiceAssistant` (or any custom Agents pipeline) and attach the helper:

```python
from livekit.agents import VoiceAssistant
from livekit_primd import PrimdRetriever, attach_to_voice_assistant

retriever = PrimdRetriever(
    primd_url="http://127.0.0.1:8080",
    top_k=5,
    corpus_text={"faq-001": "We offer a 14-day free trial...", ...},
    on_context=async_context_handler,  # your LLM-context injection callback
)

assistant = VoiceAssistant(stt=..., llm=..., tts=...)
attach_to_voice_assistant(assistant, retriever)
assistant.start(room)
```

The helper wires:

* user `interim` / `started_speaking` events → `retriever.observe_partial(text)` so primd starts retrieving while the user is still talking
* `user_speech_committed` → `retriever.finalize(text)` — if the partial converged on the final, returns from a sub-microsecond cache
* `agent_started_speaking` → `retriever.warm()` so the next turn is already pre-scoped before STT lands

If your LiveKit release renames any of those events, the corresponding attachment is silently skipped — call the retriever methods directly from your own handlers instead.

## Standalone client

If you want primd retrieval without LiveKit (CLI tools, batch jobs, custom pipelines):

```python
import asyncio
from livekit_primd import PrimdClient

async def main() -> None:
    async with PrimdClient("http://127.0.0.1:8080") as primd:
        result = await primd.query("is there a free trial", top_k=3)
        for hit in result.hits:
            print(f"{hit.rank}: {hit.id} ({hit.event}) dist={hit.distance}")

asyncio.run(main())
```

The client also exposes the session methods directly: `observe`, `finalize`, `warm`, `reset`.

## Why session calls?

`/query` is convenient but doesn't unlock the latency win — primd has no idea what's coming until you call it. The session API lets primd:

1. **Speculate** on partial transcripts during STT (no critical-path cost).
2. **Short-circuit** `/finalize` if the partial converged on the final — cache hit, microseconds.
3. **Pre-warm** the predictor's scope during TTS so the next turn's `/observe` is already constrained.

This is the dual-agent fast-talker / slow-thinker pattern from [Salesforce VoiceAgentRAG](https://arxiv.org/abs/2603.02206), wired up to LiveKit Agents.

## OpenAI-compatible drop-in

If you'd rather treat primd as your LLM back-end (returning retrieved context in OpenAI Chat Completions shape), primd also exposes `POST /v1/chat/completions`. See [docs/integrations/moshirag.md](https://github.com/rohansx/primd/blob/main/docs/integrations/moshirag.md) for the swap.

## Compatibility

Tested against LiveKit Agents 0.10+. Event names occasionally shift between Pipecat releases — if your version renames `user_speech_committed` or `agent_started_speaking`, attach your own handlers calling `retriever.observe_partial` / `retriever.finalize` / `retriever.warm` directly.

## License

Apache-2.0
