"""Async HTTP client for `primd serve`.

Used by the CLI demo and the Pipecat FrameProcessor. Stays minimal so it can
sit in a voice-agent's hot path: one POST per query, JSON in, JSON out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class Hit:
    rank: int
    distance: int
    id: str
    event: str


@dataclass
class QueryResult:
    """Top-K matches for a query plus end-to-end timing."""

    hits: list[Hit]
    embedder: str
    embed_us: int
    scan_us: int
    corpus_size: int
    network_us: int  # Round-trip excluding server-reported embed+scan


class PrimdClient:
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

    async def __aexit__(self, *_) -> None:
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
        wall_start = time.perf_counter()
        r = await self._client.post(
            f"{self.base_url}/query",
            json={"text": text, "top_k": top_k, "parallel": parallel},
        )
        r.raise_for_status()
        wall_us = int((time.perf_counter() - wall_start) * 1_000_000)
        data = r.json()
        hits = [
            Hit(
                rank=h["rank"],
                distance=h["distance"],
                id=h["id"],
                event=h["event"],
            )
            for h in data["hits"]
        ]
        server_us = data["embed_us"] + data["scan_us"]
        return QueryResult(
            hits=hits,
            embedder=data["embedder"],
            embed_us=data["embed_us"],
            scan_us=data["scan_us"],
            corpus_size=data["corpus_size"],
            network_us=max(wall_us - server_us, 0),
        )
