"""Reference Pipecat voice agent that uses primd as its retrieval layer.

This is a working skeleton — it shows where each piece plugs in but does
not include credentials. To run it you need accounts/keys for the services
below. The simplest single-vendor combo for Indian-language voice flows is:

    STT:  Sarvam Saaras v3
    LLM:  Sarvam 30B / 105B
    TTS:  Sarvam Bulbul v3
    WebRTC: Daily.co or LiveKit

Pipecat handles the framing; primd handles the retrieval; everything else
is provider integrations. Swap any of the three top boxes without
touching primd.

Setup:
    pip install -r requirements.txt
    export SARVAM_API_KEY=...
    export DAILY_API_KEY=...
    primd serve --index /tmp/primd-faq --bind 127.0.0.1:8080  # in another terminal
    python voice_agent.py --room https://yourdomain.daily.co/your-room

Then dial in to the Daily room from any browser; the agent will pick up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Allow running without installing.
sys.path.insert(0, str(Path(__file__).parent))

from primd_retriever import PrimdRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice_agent")


def _load_corpus_text(corpus_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not corpus_path.exists():
        return out
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        out[entry["id"]] = entry["text"]
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--room", required=True, help="Daily room URL")
    parser.add_argument("--primd", default="http://127.0.0.1:8080", help="primd serve base URL")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parent.parent / "faq.jsonl",
        help="Original JSONL — used to render document text in the system prompt",
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    # Imports are local so the script's --help works even if Pipecat is not
    # installed yet.
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.services.sarvam.llm import SarvamLLMService
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.transports.services.daily import DailyParams, DailyTransport

    corpus = _load_corpus_text(args.corpus)
    log.info("loaded %d corpus entries from %s", len(corpus), args.corpus)

    transport = DailyTransport(
        args.room,
        None,
        "primd-retriever",
        DailyParams(audio_out_enabled=True, vad_enabled=True),
    )

    stt = SarvamSTTService(api_key=os.environ["SARVAM_API_KEY"], mode="translate")
    llm = SarvamLLMService(api_key=os.environ["SARVAM_API_KEY"])
    tts = SarvamTTSService(api_key=os.environ["SARVAM_API_KEY"])

    retriever = PrimdRetriever(
        primd_url=args.primd,
        top_k=args.top,
        corpus_text=corpus,
    )

    initial_context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly customer-support agent on a phone call. "
                    "Speak in short, natural sentences. The user's question will "
                    "be augmented with retrieved context. Answer using only that. "
                    "If the caller speaks an Indian language, respond naturally in English unless told otherwise."
                ),
            }
        ],
    )
    context_aggregator = LLMContextAggregatorPair(initial_context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            retriever,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
