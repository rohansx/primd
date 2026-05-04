# pipecat-primd

**Sub-millisecond predictive retrieval for Pipecat.** Open-source [VoiceAgentRAG](https://arxiv.org/abs/2603.02206).

`pipecat-primd` is a [Pipecat](https://github.com/pipecat-ai/pipecat) `FrameProcessor` that connects a voice pipeline to [primd](https://github.com/rohansx/primd) — a Rust retrieval runtime that speculates on partial transcripts during STT and pre-warms next-turn answers during TTS.

End-of-utterance retrieval drops from ~157 µs (naive SIMD scan) to **~1.6 µs** when speculation matches — **98× faster**, deterministically, on a 100k-doc corpus.

## Install

```bash
pip install pipecat-primd
```

You also need [`primd`](https://github.com/rohansx/primd) running locally:

```bash
cargo install primd-cli
primd index --input examples/faq.jsonl --out /tmp/primd-faq --embedder hashed
primd serve --index /tmp/primd-faq --bind 127.0.0.1:8080
```

## Use

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat_primd import PrimdRetriever

retriever = PrimdRetriever(
    primd_url="http://127.0.0.1:8080",
    top_k=5,
    corpus_text={"faq-001": "We offer a 14-day free trial...", ...},
)

pipeline = Pipeline([
    transport.input(),
    stt,
    retriever,                # ← interim → /observe, final → /finalize, bot speak → /warm
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

That's it. The retriever:

* feeds `InterimTranscriptionFrame` into primd's `/session/{id}/observe` so primd starts retrieving while the user is still speaking,
* hits `/session/{id}/finalize` on `TranscriptionFrame` — if the partial converged on the final, returns from a sub-microsecond cache,
* calls `/session/{id}/warm` on `BotStartedSpeakingFrame` so the next turn is already pre-scoped before STT lands.

## Standalone Client

If you want primd retrieval without Pipecat (CLI tools, batch jobs, custom pipelines):

```python
import asyncio
from pipecat_primd import PrimdClient

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

This is the dual-agent fast-talker / slow-thinker pattern from [Salesforce VoiceAgentRAG](https://arxiv.org/abs/2603.02206), wired up.

## Compatibility

Tested against Pipecat 0.0.49+. Frame names occasionally shift between Pipecat releases — if your version renames `InterimTranscriptionFrame`, `TranscriptionFrame`, or `BotStartedSpeakingFrame`, update the imports in `pipecat_primd/retriever.py`.

## License

Apache-2.0
