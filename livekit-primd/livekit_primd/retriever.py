"""Framework-agnostic primd retriever for LiveKit Agents.

LiveKit's Agents API surface shifts between releases (frame names, event
names, even module paths). To stay decoupled, :class:`PrimdRetriever`
exposes three plain async methods — ``observe_partial``, ``finalize``,
``warm`` — that LiveKit users call from their own pipeline event handlers.

A thin convenience helper :func:`attach_to_voice_assistant` wires the
common pattern (``user_speech_committed`` -> finalize, ``agent_started_speaking``
-> warm) but the retriever itself does not import livekit, so the package
is installable without livekit-agents present.

Example (LiveKit Agents 0.10+):

.. code-block:: python

    from livekit.agents import VoiceAssistant
    from livekit_primd import PrimdRetriever, attach_to_voice_assistant

    retriever = PrimdRetriever(
        primd_url="http://127.0.0.1:8080",
        corpus_text={...},  # optional: doc-id -> text for context injection
    )
    assistant = VoiceAssistant(stt=..., llm=..., tts=...)
    attach_to_voice_assistant(assistant, retriever)
    assistant.start(room)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Awaitable, Callable, Optional

from livekit_primd.client import PrimdClient, QueryResult

log = logging.getLogger("livekit_primd")

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer-support assistant. Answer using the retrieved "
    "context below. If the answer is not in the context, say you don't "
    "know. Keep answers under two sentences.\n\n"
    "Retrieved context:\n{context}"
)


ContextHandler = Callable[[str, QueryResult], Awaitable[None]]


class PrimdRetriever:
    """Session-aware primd retriever, framework-agnostic.

    Args:
        primd_url: Base URL of ``primd serve``.
        top_k: Number of hits to fetch per finalize.
        client: Pre-built :class:`PrimdClient` (otherwise one is created).
        system_prompt: Template; ``{context}`` is substituted with hit text.
        corpus_text: ``id`` -> original document text. Without it, context
            strings include only ids + events.
        min_chars: Skip queries shorter than this so the gate isn't hammered
            on stutters or interjections.
        session_id: Stable session id; if omitted a uuid is generated. Use
            the same id across the whole call to keep predictor state warm.
        on_context: Optional async callback fired with the formatted system
            prompt and the underlying :class:`QueryResult` after each
            finalize. Wire it into your LLM context aggregator.
    """

    def __init__(
        self,
        primd_url: str,
        *,
        top_k: int = 5,
        client: Optional[PrimdClient] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        corpus_text: Optional[dict[str, str]] = None,
        min_chars: int = 4,
        session_id: Optional[str] = None,
        on_context: Optional[ContextHandler] = None,
    ) -> None:
        self._owned_client = client is None
        self._client = client or PrimdClient(primd_url)
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.corpus_text = corpus_text or {}
        self.min_chars = min_chars
        self.session_id = session_id or f"livekit-{uuid.uuid4().hex[:12]}"
        self.on_context = on_context

    async def aclose(self) -> None:
        try:
            await self._client.reset(self.session_id)
        except Exception:  # noqa: BLE001
            pass
        if self._owned_client:
            await self._client.aclose()

    async def observe_partial(self, text: str) -> None:
        """Feed an STT interim transcript. Cheap; safe to call on every partial."""
        text = (text or "").strip()
        if len(text) < self.min_chars:
            return
        try:
            await self._client.observe(self.session_id, text, top_k=self.top_k)
        except Exception as e:  # noqa: BLE001
            log.debug("primd observe failed (non-fatal): %s", e)

    async def finalize(self, text: str) -> Optional[QueryResult]:
        """End-of-utterance retrieval. Returns the QueryResult, or None on error."""
        text = (text or "").strip()
        if len(text) < self.min_chars:
            return None
        try:
            qr = await self._client.finalize(self.session_id, text, top_k=self.top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("primd finalize failed; falling through: %s", e)
            return None

        log.info(
            "primd: top1=%s/%s served_by=%s embed=%dus scan=%dus net=%dus",
            qr.hits[0].id if qr.hits else "-",
            qr.hits[0].event if qr.hits else "-",
            qr.served_by,
            qr.embed_us,
            qr.scan_us,
            qr.network_us,
        )

        if self.on_context is not None:
            await self.on_context(self.format_context(qr), qr)
        return qr

    async def warm(self) -> None:
        """Prefetch the next likely turn's scope. Call when the assistant starts speaking."""
        try:
            await self._client.warm(self.session_id)
        except Exception as e:  # noqa: BLE001
            log.debug("primd warm failed (non-fatal): %s", e)

    def format_context(self, qr: QueryResult) -> str:
        """Render a system prompt embedding the retrieved context."""
        if not qr.hits:
            return self.system_prompt.format(context="(no relevant context found)")
        lines = []
        for h in qr.hits:
            text = self.corpus_text.get(h.id, "")
            if text:
                lines.append(f"- [{h.event}] {text}")
            else:
                lines.append(f"- [{h.event}] (id={h.id} dist={h.distance})")
        return self.system_prompt.format(context="\n".join(lines))


def attach_to_voice_assistant(assistant: Any, retriever: PrimdRetriever) -> None:
    """Wire a :class:`PrimdRetriever` into a LiveKit Agents ``VoiceAssistant``.

    LiveKit's voice-assistant emits ``user_speech_committed``,
    ``agent_started_speaking``, and (in newer releases) interim-transcript
    events. We attach handlers for the ones that fit primd's session
    lifecycle. If your LiveKit version doesn't expose one of these events,
    the corresponding attachment is silently skipped.

    The signature is ``Any`` rather than ``VoiceAssistant`` so this helper
    works across LiveKit releases without an import-time dependency.
    """

    def _attach(event_name: str, handler: Callable[..., Awaitable[None]]) -> bool:
        if not hasattr(assistant, "on"):
            return False
        try:
            assistant.on(event_name, handler)
            return True
        except Exception as e:  # noqa: BLE001
            log.debug("attach %s failed: %s", event_name, e)
            return False

    async def _on_user_committed(msg: Any) -> None:
        text = getattr(msg, "content", None) or getattr(msg, "text", None) or str(msg)
        await retriever.finalize(text)

    async def _on_agent_speaking(*_: Any) -> None:
        await retriever.warm()

    async def _on_interim(msg: Any) -> None:
        text = getattr(msg, "content", None) or getattr(msg, "text", None) or str(msg)
        await retriever.observe_partial(text)

    _attach("user_speech_committed", _on_user_committed)
    _attach("agent_started_speaking", _on_agent_speaking)
    # Interim transcripts: name varies across LiveKit releases. Attaching
    # multiple plausible names is harmless — only the matching one fires.
    _attach("user_speech_interim", _on_interim)
    _attach("user_started_speaking", _on_interim)
