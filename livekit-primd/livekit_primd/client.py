"""Async HTTP client for ``primd serve``.

Stateless callers use :meth:`PrimdClient.query`. Voice pipelines use the
session methods (:meth:`observe`, :meth:`finalize`, :meth:`warm`,
:meth:`reset`) which let primd speculate during STT and prefetch during
TTS, where the latency win lives.

This client is intentionally duplicated from ``pipecat-primd``'s client to
keep ``livekit-primd`` installable without pulling in Pipecat. A future
shared ``primd-py-client`` package will consolidate the two; until then,
keep changes in sync between both copies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass(frozen=True)
class Hit:
    rank: int
    distance: int
    id: str
    event: str


@dataclass(frozen=True)
class QueryResult:
    """Top-K matches plus end-to-end and per-stage timings."""

    hits: tuple[Hit, ...]
    embedder: str
    embed_us: int
    scan_us: int
    corpus_size: int
    network_us: int
    served_by: str = ""
    predicted_events: tuple[str, ...] = field(default_factory=tuple)
    shard_scope_size: int = 0


class PrimdClient:
    """Async HTTP client for primd serve.

    The instance is safe to share across coroutines because it owns a single
    ``httpx.AsyncClient`` with HTTP/2 connection pooling.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout_s: float = 5.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def __aenter__(self) -> "PrimdClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def query(
        self,
        text: str,
        top_k: int = 5,
        parallel: bool = False,
    ) -> QueryResult:
        return await self._post_query("/query", text, top_k, parallel)

    async def observe(self, session_id: str, text: str, top_k: int = 5) -> None:
        """Feed an STT partial. Cheap; runs the streaming gate first."""
        await self._client.post(
            f"{self.base_url}/session/{session_id}/observe",
            json={"text": text, "top_k": top_k},
        )

    async def finalize(self, session_id: str, text: str, top_k: int = 5) -> QueryResult:
        """End-of-utterance retrieval. Returns cached result if speculation matched."""
        return await self._post_query(f"/session/{session_id}/finalize", text, top_k, False)

    async def warm(self, session_id: str) -> dict:
        """Prefetch likely next-turn scope. Call during TTS playback."""
        r = await self._client.post(
            f"{self.base_url}/session/{session_id}/warm",
            json={},
        )
        r.raise_for_status()
        return r.json()

    async def reset(self, session_id: str) -> None:
        await self._client.post(f"{self.base_url}/session/{session_id}/reset")

    async def _post_query(
        self,
        path: str,
        text: str,
        top_k: int,
        parallel: bool,
    ) -> QueryResult:
        wall_start = time.perf_counter()
        r = await self._client.post(
            f"{self.base_url}{path}",
            json={"text": text, "top_k": top_k, "parallel": parallel},
        )
        r.raise_for_status()
        wall_us = int((time.perf_counter() - wall_start) * 1_000_000)
        data = r.json()
        hits = tuple(
            Hit(rank=h["rank"], distance=h["distance"], id=h["id"], event=h["event"])
            for h in data.get("hits", [])
        )
        server_us = data.get("embed_us", 0) + data.get("scan_us", 0)
        return QueryResult(
            hits=hits,
            embedder=data.get("embedder", ""),
            embed_us=data.get("embed_us", 0),
            scan_us=data.get("scan_us", 0),
            corpus_size=data.get("corpus_size", 0),
            network_us=max(wall_us - server_us, 0),
            served_by=data.get("served_by", ""),
            predicted_events=tuple(data.get("predicted_events", [])),
            shard_scope_size=data.get("shard_scope_size", 0),
        )
