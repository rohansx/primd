"""Standalone CLI demo: text in → primd retrieval → optional LLM answer.

Runs in any terminal. The point is to show, end-to-end:

    1. primd serve is exposing the index over HTTP.
    2. A Python client posts a question and gets top-K hits in microseconds.
    3. (Optional) Those hits become context for an LLM that drafts the answer.

Usage:

    # Make sure primd is serving in another terminal:
    #   primd serve --index /tmp/primd-faq --bind 127.0.0.1:8080

    # No LLM, just see what primd returns:
    python cli_demo.py

    # With Sarvam for the final answer:
    SARVAM_API_KEY=sk-... python cli_demo.py --llm --llm-provider sarvam

    # With OpenAI for the final answer:
    OPENAI_API_KEY=sk-... python cli_demo.py --llm --llm-provider openai

    # Single-shot:
    python cli_demo.py --text "is there a free trial"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# Allow `python cli_demo.py` from this directory without installing.
sys.path.insert(0, str(Path(__file__).parent))
from primd_client import PrimdClient, QueryResult  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8080"


def _format_hits(qr: QueryResult, faq_lookup: dict[str, str]) -> str:
    """Pretty-print the hits with their actual document text."""
    lines = []
    for h in qr.hits:
        text = faq_lookup.get(h.id, "(text not loaded)")
        lines.append(f"  [{h.rank}] dist={h.distance:>3}  {h.id}  ({h.event})")
        lines.append(f"      {text}")
    return "\n".join(lines)


def _load_faq_text(corpus_path: Optional[Path]) -> dict[str, str]:
    """Load id → text mapping from the same JSONL we indexed."""
    if not corpus_path or not corpus_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        out[entry["id"]] = entry["text"]
    return out


async def _draft_answer(
    question: str,
    qr: QueryResult,
    faq_lookup: dict[str, str],
    provider: str,
) -> Optional[str]:
    """Call the selected LLM provider with the retrieved hits as context."""
    if provider == "sarvam":
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key:
            return None
        url = "https://api.sarvam.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "api-subscription-key": api_key,
        }
        model = "sarvam-30b"
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        model = "gpt-4o-mini"

    context_lines = []
    for h in qr.hits:
        text = faq_lookup.get(h.id, "")
        if text:
            context_lines.append(f"[{h.event}] {text}")
    context = "\n".join(context_lines)

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a customer-support assistant. Answer using ONLY the "
                    "provided context. If the answer is not in the context, say so. "
                    "Keep answers under two sentences.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            url,
            json=body,
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


async def _ask_one(
    question: str,
    client: PrimdClient,
    faq_lookup: dict[str, str],
    use_llm: bool,
    llm_provider: str,
    top_k: int,
) -> None:
    print(f"\nyou: {question}")
    qr = await client.query(question, top_k=top_k)

    print(
        f"primd: embedder={qr.embedder} embed={qr.embed_us}us "
        f"scan={qr.scan_us}us network={qr.network_us}us "
        f"corpus={qr.corpus_size}"
    )
    print(_format_hits(qr, faq_lookup))

    if use_llm:
        llm_start = time.perf_counter()
        answer = await _draft_answer(question, qr, faq_lookup, llm_provider)
        llm_us = int((time.perf_counter() - llm_start) * 1_000_000)
        if answer is None:
            env_var = "SARVAM_API_KEY" if llm_provider == "sarvam" else "OPENAI_API_KEY"
            print(f"llm: (skipped — set {env_var} to enable {llm_provider})")
        else:
            print(f"llm[{llm_provider}]: {answer}  [took {llm_us / 1000:.0f}ms]")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="primd serve base URL")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parent.parent / "faq.jsonl",
        help="JSONL used to index, for showing matched text",
    )
    parser.add_argument("--top", type=int, default=3, help="results per query")
    parser.add_argument("--llm", action="store_true", help="also call an LLM to draft an answer")
    parser.add_argument(
        "--llm-provider",
        choices=["sarvam", "openai"],
        default="sarvam",
        help="LLM backend for the optional answer draft",
    )
    parser.add_argument("--text", help="single question; skip the REPL")
    args = parser.parse_args()

    faq_lookup = _load_faq_text(args.corpus)

    async with PrimdClient(args.url) as client:
        if not await client.health():
            print(f"primd not responding at {args.url}", file=sys.stderr)
            print("hint: in another terminal, run `primd serve --index ...`", file=sys.stderr)
            sys.exit(1)

        if args.text:
            await _ask_one(args.text, client, faq_lookup, args.llm, args.llm_provider, args.top)
            return

        print(f"primd cli demo  |  {args.url}  |  {len(faq_lookup)} corpus entries")
        print("type a question (or 'quit' to exit)")
        try:
            while True:
                try:
                    q = input("> ").strip()
                except EOFError:
                    return
                if not q:
                    continue
                if q.lower() in {"quit", "exit", "q"}:
                    return
                try:
                    await _ask_one(q, client, faq_lookup, args.llm, args.llm_provider, args.top)
                except httpx.HTTPError as e:
                    print(f"error: {e}")
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    asyncio.run(main())
