"""Tests for PrimdRetriever — verify the framework-agnostic surface."""

from __future__ import annotations

import json

import httpx
import pytest

from livekit_primd import PrimdClient, PrimdRetriever
from livekit_primd.retriever import attach_to_voice_assistant


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def _make_client(handler) -> PrimdClient:
    transport = httpx.MockTransport(handler)
    inner = httpx.AsyncClient(transport=transport, base_url="http://test")
    return PrimdClient("http://test", client=inner)


@pytest.mark.asyncio
async def test_finalize_invokes_on_context() -> None:
    received: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/finalize"):
            return _ok(
                {
                    "embedder": "hashed",
                    "embed_us": 1,
                    "scan_us": 1,
                    "corpus_size": 10,
                    "hits": [
                        {"rank": 1, "distance": 2, "id": "faq-1", "event": "pricing"}
                    ],
                    "served_by": "speculative",
                    "predicted_events": [],
                    "shard_scope_size": 0,
                }
            )
        return _ok({"status": "ok"})

    async def on_ctx(prompt: str, qr) -> None:
        received.append((prompt, qr.served_by))

    client = _make_client(handler)
    retriever = PrimdRetriever(
        primd_url="http://test",
        client=client,
        corpus_text={"faq-1": "We offer a 14-day trial."},
        on_context=on_ctx,
        session_id="t",
    )
    qr = await retriever.finalize("is there a free trial")
    await retriever.aclose()

    assert qr is not None
    assert qr.served_by == "speculative"
    assert len(received) == 1
    prompt, served_by = received[0]
    assert "We offer a 14-day trial." in prompt
    assert served_by == "speculative"


@pytest.mark.asyncio
async def test_observe_skips_short_text() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _ok({"status": "ok"})

    client = _make_client(handler)
    retriever = PrimdRetriever(primd_url="http://test", client=client, min_chars=5, session_id="t")
    await retriever.observe_partial("hi")  # below threshold
    await retriever.observe_partial("what about pricing")  # above threshold
    await retriever.aclose()

    assert calls == ["/session/t/observe", "/session/t/reset"]


@pytest.mark.asyncio
async def test_format_context_uses_corpus_text() -> None:
    client = _make_client(lambda _: _ok({"status": "ok"}))
    retriever = PrimdRetriever(
        primd_url="http://test",
        client=client,
        corpus_text={"faq-1": "Our trial is 14 days.", "faq-2": "Email support."},
        session_id="t",
    )

    class FakeQr:
        hits = [
            type("H", (), {"id": "faq-1", "event": "trial", "distance": 1})(),
            type("H", (), {"id": "faq-2", "event": "support", "distance": 2})(),
        ]

    out = retriever.format_context(FakeQr())  # type: ignore[arg-type]
    assert "Our trial is 14 days." in out
    assert "[trial]" in out
    assert "[support]" in out
    await retriever.aclose()


def test_attach_skips_missing_methods() -> None:
    # A plain object with no `on` method must not raise.
    class NoOp:
        pass

    client = _make_client(lambda _: _ok({"status": "ok"}))
    retriever = PrimdRetriever(primd_url="http://test", client=client, session_id="t")
    attach_to_voice_assistant(NoOp(), retriever)  # no-op, no exception
