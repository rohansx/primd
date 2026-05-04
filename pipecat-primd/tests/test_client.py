"""Unit tests for PrimdClient. Uses an httpx MockTransport so no server is needed."""

from __future__ import annotations

import json

import httpx
import pytest

from pipecat_primd import Hit, PrimdClient, QueryResult


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def _make_client(handler):
    transport = httpx.MockTransport(handler)
    inner = httpx.AsyncClient(transport=transport, base_url="http://test")
    return PrimdClient("http://test", client=inner)


@pytest.mark.asyncio
async def test_query_parses_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/query"
        body = json.loads(request.content)
        assert body["text"] == "what's pricing"
        return _ok(
            {
                "embedder": "hashed",
                "embed_us": 12,
                "scan_us": 8,
                "corpus_size": 100,
                "hits": [
                    {"rank": 1, "distance": 14, "id": "faq-001", "event": "pricing"},
                    {"rank": 2, "distance": 22, "id": "faq-002", "event": "trial"},
                ],
                "served_by": "full_scan",
                "predicted_events": [],
                "shard_scope_size": 0,
            }
        )

    async with _make_client(handler) as primd:
        out = await primd.query("what's pricing", top_k=2)

    assert isinstance(out, QueryResult)
    assert out.embedder == "hashed"
    assert out.corpus_size == 100
    assert len(out.hits) == 2
    assert out.hits[0] == Hit(rank=1, distance=14, id="faq-001", event="pricing")


@pytest.mark.asyncio
async def test_session_observe_uses_session_path() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/observe"):
            return _ok({"status": "ok"})
        return _ok(
            {
                "embedder": "hashed",
                "embed_us": 1,
                "scan_us": 1,
                "corpus_size": 10,
                "hits": [],
                "served_by": "speculative",
                "predicted_events": ["pricing"],
                "shard_scope_size": 4,
            }
        )

    async with _make_client(handler) as primd:
        await primd.observe("sess-1", "what about pri")
        out = await primd.finalize("sess-1", "what about pricing")

    assert paths == ["/session/sess-1/observe", "/session/sess-1/finalize"]
    assert out.served_by == "speculative"
    assert out.predicted_events == ("pricing",)


@pytest.mark.asyncio
async def test_warm_returns_predictions() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _ok({"predicted_events": ["trial", "pricing"], "shard_scope_size": 8})

    async with _make_client(handler) as primd:
        out = await primd.warm("sess-1")

    assert out["predicted_events"] == ["trial", "pricing"]
    assert out["shard_scope_size"] == 8


@pytest.mark.asyncio
async def test_health_returns_bool() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"status":"ok"}')

    async with _make_client(handler) as primd:
        assert await primd.health() is True
