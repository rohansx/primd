"""Pipecat FrameProcessor that injects primd retrieval results into the LLM context.

Pipeline shape:

    audio_in → STT → PrimdRetriever → LLMContext → LLM → TTS → audio_out
                            │
                            └─► POST /query → top-K hits → context injection

Plugs in like any other FrameProcessor. On each finalised user transcription:

    1. POST the text to `primd serve` and read the top-K hits.
    2. Format the hits as a system message ("Relevant context: ...").
    3. Push that message upstream so the LLM sees it before generating.
    4. Forward the original transcription frame so the rest of the pipeline
       behaves as usual.

Tested against Pipecat 0.0.49+. If your Pipecat version exposes different
frame names (it occasionally renames them), the constants near the top of
this module are the only thing that needs adjusting.
"""

from __future__ import annotations

import logging
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from primd_client import PrimdClient, QueryResult

log = logging.getLogger("primd_retriever")

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer-support assistant. Answer using the retrieved "
    "context below. If the answer is not in the context, say you don't "
    "know. Keep answers under two sentences.\n\n"
    "Retrieved context:\n{context}"
)


class PrimdRetriever(FrameProcessor):
    """Intercepts user transcripts, queries primd, injects retrieved docs."""

    def __init__(
        self,
        primd_url: str,
        *,
        top_k: int = 5,
        client: Optional[PrimdClient] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        corpus_text: Optional[dict[str, str]] = None,
        min_chars: int = 4,
    ) -> None:
        """
        Args:
            primd_url: Base URL of `primd serve` (e.g. "http://localhost:8080").
            top_k: How many hits to fetch per query.
            client: Optional pre-built PrimdClient (otherwise one is created).
            system_prompt: Template; `{context}` is replaced with the hit text.
            corpus_text: id → original document text. Required for the LLM to
                see actual answers; without it the system message will only
                show ids.
            min_chars: Skip queries shorter than this (avoids hammering primd
                on whitespace or short interjections like "yes" / "ok").
        """
        super().__init__()
        self._owned_client = client is None
        self._client = client or PrimdClient(primd_url)
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.corpus_text = corpus_text or {}
        self.min_chars = min_chars

    async def cleanup(self) -> None:
        if self._owned_client:
            await self._client.aclose()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = (frame.text or "").strip()
            if len(text) >= self.min_chars:
                try:
                    qr = await self._client.query(text, top_k=self.top_k)
                except Exception as e:  # noqa: BLE001
                    log.warning("primd query failed; falling through: %s", e)
                else:
                    await self._inject_context(qr)

        # Always forward the original frame so the rest of the pipeline
        # receives it (including the LLM, which sees the user's transcription).
        await self.push_frame(frame, direction)

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
        context = "\n".join(lines)
        message = {
            "role": "system",
            "content": self.system_prompt.format(context=context),
        }

        log.info(
            "primd: top1=%s/%s embed=%dus scan=%dus network=%dus",
            qr.hits[0].id,
            qr.hits[0].event,
            qr.embed_us,
            qr.scan_us,
            qr.network_us,
        )

        # Push the message into the conversation so the LLM sees it on the
        # next turn. `LLMMessagesAppendFrame` is the standard Pipecat pattern.
        await self.push_frame(
            LLMMessagesAppendFrame(messages=[message]),
            FrameDirection.UPSTREAM,
        )
