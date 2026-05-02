# primd × Pipecat

Three pieces here, each runnable in isolation:

| File | What it is | Runtime deps |
|---|---|---|
| `primd_client.py` | Async Python client for `primd serve` | `httpx` |
| `cli_demo.py` | Standalone REPL: text → primd → top-K (+ optional LLM) | `httpx` |
| `primd_retriever.py` | Pipecat `FrameProcessor` that injects primd hits into the LLM context | `pipecat-ai` |
| `voice_agent.py` | Reference end-to-end voice agent (Daily + Deepgram + OpenAI + Cartesia + primd) | full Pipecat stack |

## Step 1 — start primd

In one terminal, build a small index and serve it:

```bash
# From the repo root
cargo build --release -p primd-cli
./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq \
  --embedder hashed
./target/release/primd serve \
  --index /tmp/primd-faq \
  --bind 127.0.0.1:8080
```

Leave that running.

## Step 2 — run the CLI demo (text only)

In another terminal:

```bash
pip install httpx
python examples/pipecat/cli_demo.py --text "is there a free trial"
```

You should see something like:

```
you: is there a free trial
primd: embedder=hashed embed=27us scan=230us network=2546us corpus=30
  [1] dist= 16  faq-008  (trial)
      During the trial you have access to every feature on the premium plan with no limits.
  ...
```

For an LLM-drafted answer:

```bash
export OPENAI_API_KEY=sk-...
python examples/pipecat/cli_demo.py --llm
```

The REPL hands every question to primd first, then asks GPT-4o-mini to
answer using only the retrieved hits.

## Step 3 — run the voice agent (Pipecat)

```bash
pip install -r examples/pipecat/requirements.txt
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
export CARTESIA_API_KEY=...
export DAILY_API_KEY=...
python examples/pipecat/voice_agent.py \
  --room https://yourdomain.daily.co/your-room
```

Then open the Daily room in any browser and talk to the agent. The pipeline
is:

```
mic → Deepgram → PrimdRetriever → OpenAI → Cartesia → speaker
                       │
                       └─► POST :8080/query → top-K → LLM context
```

The `PrimdRetriever` adds zero perceptible latency (~3 ms round-trip on
localhost; the LLM call dominates) and keeps the LLM grounded in your
indexed corpus instead of hallucinating.

## How `PrimdRetriever` plugs in

```python
from primd_retriever import PrimdRetriever

retriever = PrimdRetriever(
    primd_url="http://localhost:8080",
    top_k=5,
    corpus_text={"faq-001": "...", "faq-002": "..."},  # for the LLM to read
)

pipeline = Pipeline([
    transport.input(),
    stt,
    retriever,                  # ← here
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

It listens for `TranscriptionFrame`s. On each finalised user utterance, it
posts to primd and pushes a fresh `LLMMessagesAppendFrame` so the LLM sees
the retrieved context on the next turn. The original transcription frame
is forwarded unchanged.

## Swapping the embedder

Nothing in the Pipecat layer cares which embedder primd is using. Reindex
with `--embedder local` (BGE-small) or `--embedder openai` and restart
`primd serve`; the Python side does not change.

## What's not wired up yet

- **Streaming partials.** Pipecat emits `InterimTranscriptionFrame` for
  mid-utterance text. The retriever currently only fires on finalised
  frames. Hooking partials into primd's streaming layer (layer 1) would
  let the speculative scan happen *during* user speech instead of after.
  That is the next obvious win.
- **Conversation state for the predictor.** primd's Markov predictor
  (layer 3) is unused over HTTP — the serve endpoint is a single-shot
  retrieval API. Exposing per-session conversation state would unlock
  the predictor's prefetch path. Plausible API:
  `POST /session/{id}/observe` + `POST /session/{id}/query`.
