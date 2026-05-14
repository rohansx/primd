# MoshiRAG back-end adapter

`primd serve` exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint that lets [Kyutai MoshiRAG](https://github.com/kyutai-labs/moshi-rag) (or any other OpenAI-compatible client) swap its 1–3 s vLLM-served LLM call for primd's sub-200 µs retrieval response with one env-var change.

primd does not generate. The endpoint returns the *retrieved context* as the assistant message's `content`, formatted as a numbered list. The model in the MoshiRAG loop still synthesizes the spoken response — primd just makes sure the context lands in microseconds instead of seconds.

## Why this exists

MoshiRAG's reference back-end is a generic LLM via vLLM (Kyutai's docs flag retrieval as the latency bottleneck and recommend running a 27 B LLM as the retrieval back-end). That's 1–3 s per turn just for context retrieval, on top of the audio model itself. primd answers the same OpenAI contract at sub-200 µs.

The same adapter works for any client that speaks OpenAI Chat Completions: LangChain, LlamaIndex, the Python `openai` SDK, custom curl pipelines, and Pipecat `OpenAILLMService` shims.

## Quick start

```bash
# 1. build the binary
cargo build --release -p primd-cli

# 2. index your knowledge base (texts are stored in the manifest so the
#    adapter can return real content, not just doc IDs)
./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq \
  --embedder hashed

# 3. serve
./target/release/primd serve \
  --index /tmp/primd-faq \
  --bind 127.0.0.1:8080
```

## Endpoint shape

`POST /v1/chat/completions`

### Request

Wire-compatible with the OpenAI Chat Completions API. Unknown fields are ignored.

```json
{
  "model": "primd",
  "messages": [
    {"role": "system", "content": "you are a support agent"},
    {"role": "user", "content": "is there a free trial"}
  ],
  "user": "session-id-here",
  "top_k": 5
}
```

- `messages` (required) — the latest user message becomes the retrieval query
- `user` (optional) — doubles as a primd session id. When present, drives the session lifecycle (`observe_partial` from prior calls + `finalize` here + speculative cache + delta cache), engaging the predictive layers. When absent, falls back to stateless retrieval.
- `top_k` (optional, primd extension) — defaults to 5
- `model`, `temperature`, `stream`, `max_tokens`, `stop`, `presence_penalty`, etc. are accepted and ignored

### Response

Standard OpenAI Chat Completions shape, plus a `primd` field with retrieval metadata.

```json
{
  "id": "chatcmpl-primd-1778675566",
  "object": "chat.completion",
  "created": 1778675566,
  "model": "primd",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "[1] (trial) During the trial you have access to every feature ...\n[2] (pricing) Enterprise pricing is custom; please contact sales ..."
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 5, "completion_tokens": 47, "total_tokens": 52},
  "primd": {
    "served_by": "speculative",
    "embed_us": 6,
    "scan_us": 3,
    "predicted_events": ["trial", "pricing"],
    "hits": [
      {"rank": 1, "distance": 23, "id": "faq-008", "event": "trial"},
      {"rank": 2, "distance": 25, "id": "faq-004", "event": "pricing"}
    ]
  }
}
```

`primd.served_by` is one of `full_scan` / `shard_scan` / `speculative` / `delta_cache` — useful for debugging whether the predictive layers fired.

## Engaging the speculative-cache wedge

A stateless call to `/v1/chat/completions` already beats vLLM by 3 orders of magnitude. To engage the sub-µs speculative path, drive primd's session lifecycle from your client:

```bash
# during STT — feed partial transcripts as they arrive
curl -X POST http://127.0.0.1:8080/session/turn-42/observe \
  -d '{"text":"what about pri","top_k":3}'

# end of speech — OpenAI-shaped finalize, served from speculative cache
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"primd",
    "messages":[{"role":"user","content":"what about pricing"}],
    "user":"turn-42",
    "top_k":3
  }'
# served_by: speculative, scan_us: ~3
```

The same `user` field that OpenAI uses for client identification becomes primd's session key. Set it once per turn (or per call, when interleaving `observe_partial` and final retrieval).

## Caveats

- **No streaming.** `stream: true` is accepted and ignored; the response is always returned as a single chat completion. For retrieval-as-completion, the content is short enough that streaming doesn't help.
- **Token counts are approximate.** `usage.prompt_tokens` and `completion_tokens` are word-counts, not BPE counts. primd doesn't run a tokenizer.
- **primd doesn't generate.** The `assistant` message's `content` is retrieved context. If your client expects natural-language answers, run the response through an LLM in your pipeline (which is how MoshiRAG works — the audio model consumes the retrieved context).
- **Indexes built before v0.2 don't carry doc texts.** Older `manifest.json` files lack the `texts` field; the adapter falls back to returning doc IDs in place of body text. Re-index to get full content.

## Per-user predictor persistence

Pass `--sr-state-dir <path>` to `primd serve` to persist each session's Markov predictor across restarts. On session `reset`, the current Markov state is written to `<path>/<sanitized-user>.markov.json`. On the next session create with the same `user` (or session id), primd warm-starts from that file instead of cold-starting.

## Cold-tier session memory (v0.4)

`--cold-tier-dir <path>` attaches a per-session DWM-backed cold tier to every voice session. Evicted signatures (from previous sessions or from out-of-band batch processing) live in `<path>/<sanitized-user>.cold.json` and load automatically when the session starts. Cold-tier hits surface in the `cold_hits` field of every finalize response — useful for spanning context across multi-day conversations without keeping the entire hot-path corpus warm.

```bash
./target/release/primd serve \
  --index /tmp/primd-faq \
  --predictor hybrid \
  --sr-state-dir /var/lib/primd/sessions \
  --cold-tier-dir /var/lib/primd/cold
```

Cold-tier results are independent of and additive to hot-tier hits — both are returned in the same response so the caller can merge or filter as appropriate. See [the architecture overview](../architecture/overview.md) for the layered design and `docs/plan/roadmap.md` Track F for the Hippocampus DWM context.

```bash
./target/release/primd serve \
  --index /tmp/primd-faq \
  --predictor hybrid \
  --sr-state-dir /var/lib/primd/sessions
```

Works for `--predictor markov` and `--predictor hybrid` (which carries a Markov fallback inside). Pure SR / low-rank predictors return `None` from the new `NextTurnPredictor::as_markov()` trait method and skip persistence — full SR-state persistence lands in v0.2.7.

User-ids are sanitized to alphanumeric + `. _ -` before being used as filenames, so `user="../../etc/passwd"` writes safely to `.._.._etc_passwd.markov.json` inside the configured directory and cannot escape it.

## Drop-in for MoshiRAG

MoshiRAG configures its retrieval back-end via an OpenAI-compatible base URL. The swap is:

```bash
# before — vLLM serving a 27 B LLM at 1–3 s/turn
export MOSHI_RAG_BACKEND_URL=http://localhost:8000/v1
export MOSHI_RAG_BACKEND_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

# after — primd at sub-200 µs/turn
export MOSHI_RAG_BACKEND_URL=http://localhost:8080/v1
export MOSHI_RAG_BACKEND_MODEL=primd
```

(Exact env-var names vary across MoshiRAG releases — check your version's docs.)

## Related

- [Roadmap v0.2](../plan/roadmap.md) — where this adapter sits in the release
- [Strategy memo](../business/strategy-2026-05.md) — why filling the MoshiRAG back-end slot is a primary v0.2 wedge
- [Layer-3 prediction](../architecture/layer-3-prediction.md) — what `served_by: speculative` means
