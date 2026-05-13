# LiveKit Agents integration

[`livekit-primd`](../../livekit-primd/) is the LiveKit-Agents-shaped client for primd. It mirrors [`pipecat-primd`](../../pipecat-primd/) in posture — same `PrimdClient` HTTP surface, same session-aware `observe`/`finalize`/`warm` lifecycle, same Apache-2.0 license — but exposes a framework-agnostic `PrimdRetriever` plus a thin attachment helper for LiveKit's `VoiceAssistant`.

## Why a separate package?

LiveKit Agents and Pipecat have different idioms (LiveKit uses event-driven `assistant.on(...)` callbacks; Pipecat uses an explicit `FrameProcessor` pipeline). Forcing both into one package leaks one framework's vocabulary into the other. Two packages, one client crate's worth of HTTP code, keeps each install clean.

The two packages duplicate ~140 lines of HTTP client code. A future shared `primd-py-client` package will consolidate; until then changes must land in both `pipecat-primd/pipecat_primd/client.py` and `livekit-primd/livekit_primd/client.py`.

## Install

```bash
pip install livekit-primd
# and for the actual LiveKit integration:
pip install "livekit-primd[livekit]"
```

## Wiring pattern

```python
from livekit.agents import VoiceAssistant
from livekit_primd import PrimdRetriever, attach_to_voice_assistant

async def on_context(prompt: str, qr) -> None:
    # Inject the formatted prompt into your LLM context aggregator.
    # In LiveKit Agents 0.10+, this typically means calling
    # `assistant.fnc_ctx` updates or appending to the chat-context messages.
    print(f"[primd] served_by={qr.served_by} hits={len(qr.hits)}")
    # Your context-aggregator update here.

retriever = PrimdRetriever(
    primd_url="http://127.0.0.1:8080",
    top_k=5,
    corpus_text={
        "faq-001": "We offer a 14-day free trial with no credit card required.",
        # ...
    },
    on_context=on_context,
)

assistant = VoiceAssistant(stt=..., llm=..., tts=...)
attach_to_voice_assistant(assistant, retriever)
assistant.start(room)
```

The attach helper attempts to listen for:

| LiveKit event | Calls |
|---|---|
| `user_speech_interim` / `user_started_speaking` | `retriever.observe_partial(text)` |
| `user_speech_committed` | `retriever.finalize(text)` |
| `agent_started_speaking` | `retriever.warm()` |

If your LiveKit release uses different event names, the attachment is silently skipped — wire the retriever methods into your own handlers instead. The retriever itself doesn't import livekit, so it remains testable without a live LiveKit room.

## OpenAI-compatible alternative

If you'd rather treat primd as your LLM back-end and let LiveKit's OpenAI plugin pull retrieved context as a "completion" string, point `livekit.plugins.openai.LLM` at `http://127.0.0.1:8080/v1`. See [moshirag.md](moshirag.md) — the same adapter works for any OpenAI-shaped client.

## Related

- [pipecat-primd integration](../../pipecat-primd/README.md)
- [Strategy memo](../business/strategy-2026-05.md)
- [Architecture overview](../architecture/overview.md)
