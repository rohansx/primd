"""Pipecat ``FrameProcessor`` that wires primd into a voice pipeline.

Pipeline shape::

    audio_in → STT → PrimdRetriever → LLMContext → LLM → TTS → audio_out

The retriever uses primd's session API so the latency win is real:

* Interim transcripts (``InterimTranscriptionFrame``) trigger speculative
  retrieval during STT.
* Final transcripts (``TranscriptionFrame``) hit ``/finalize``; if
  speculation matched the user-visible retrieval is microseconds.
* When the bot starts speaking (``BotStartedSpeakingFrame``) primd warms
  the next likely turn's scope so the *next* observe is also pre-scoped.

Tested against Pipecat 0.0.49+. Frame names occasionally rename between
releases — the imports below are the only line that needs touching.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_primd.client import PrimdClient, QueryResult

log = logging.getLogger("pipecat_primd")

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer-support assistant. Answer using the retrieved "
    "context below. If the answer is not in the context, say you don't "
    "know. Keep answers under two sentences.\n\n"
    "Retrieved context:\n{context}"
)


class PrimdRetriever(FrameProcessor):
    """Session-aware primd retriever for Pipecat.

    Args:
        primd_url: Base URL of ``primd serve`` (e.g. ``http://localhost:8080``).
        top_k: Number of hits to fetch per finalize.
        client: Pre-built :class:`PrimdClient` (otherwise one is created).
        system_prompt: Template; ``{context}`` is substituted with hit text.
        corpus_text: ``id`` → original document text. Without it the system
            message will only show ids.
        min_chars: Skip queries shorter than this so the gate isn't hammered
            on stutters or interjections.
        session_id: Stable session id; if omitted a uuid is generated. Use
            the same id across the whole call to keep predictor state warm.
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
    ) -> None:
        super().__init__()
        self._owned_client = client is None
        self._client = client or PrimdClient(primd_url)
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.corpus_text = corpus_text or {}
        self.min_chars = min_chars
        self.session_id = session_id or f"pipecat-{uuid.uuid4().hex[:12]}"

    async def cleanup(self) -> None:
        if self._owned_client:
            try:
                await self._client.reset(self.session_id)
            except Exception:  # noqa: BLE001
                pass
            await self._client.aclose()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, InterimTranscriptionFrame):
                await self._observe(frame.text or "")
            elif isinstance(frame, TranscriptionFrame):
                await self._finalize(frame.text or "")
            elif isinstance(frame, BotStartedSpeakingFrame):
                await self._warm()

        await self.push_frame(frame, direction)

    async def _observe(self, text: str) -> None:
        text = text.strip()
        if len(text) < self.min_chars:
            return
        try:
            await self._client.observe(self.session_id, text, top_k=self.top_k)
        except Exception as e:  # noqa: BLE001
            log.debug("primd observe failed (non-fatal): %s", e)

    async def _finalize(self, text: str) -> None:
        text = text.strip()
        if len(text) < self.min_chars:
            return
        try:
            qr = await self._client.finalize(self.session_id, text, top_k=self.top_k)
        except Exception as e:  # noqa: BLE001
            log.warning("primd finalize failed; falling through: %s", e)
            return
        await self._inject_context(qr)

    async def _warm(self) -> None:
        try:
            await self._client.warm(self.session_id)
        except Exception as e:  # noqa: BLE001
            log.debug("primd warm failed (non-fatal): %s", e)

    async def _inject_context(self, qr: QueryResult) -> None:
        if not qr.hits:
            return
        lines = []
        for h in qr.hits:
            text = self.corpus_text.get(h.id, "")
            if text:
                lines.append(f"- [{h.event}] {text}")
            else:
                lines.append(f"- [{h.event}] (id={h.id} dist={h.distance})")
        message = {
            "role": "system",
            "content": self.system_prompt.format(context="\n".join(lines)),
        }

        log.info(
            "primd: top1=%s/%s served_by=%s embed=%dus scan=%dus net=%dus",
            qr.hits[0].id,
            qr.hits[0].event,
            qr.served_by,
            qr.embed_us,
            qr.scan_us,
            qr.network_us,
        )

        await self.push_frame(
            LLMMessagesAppendFrame(messages=[message]),
            FrameDirection.UPSTREAM,
        )
