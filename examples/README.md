# primd examples

End-to-end demo: text in → primd retrieval → top-K matches.

## Quick start (deterministic, no network)

```bash
# Build the release binary once.
cargo build --release -p primd-cli

# Index the FAQ corpus with the bundled feature-hashing embedder.
./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq \
  --embedder hashed

# Query it.
./target/release/primd query \
  --index /tmp/primd-faq \
  --text "is there a free trial"
```

## Production retrieval (OpenAI embeddings)

For real semantic recall via API, swap the embedder. Requires `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=sk-...

./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq-openai \
  --embedder openai

./target/release/primd query \
  --index /tmp/primd-faq-openai \
  --text "what does the premium plan cost"
```

primd asks OpenAI for `text-embedding-3-small` with `dimensions=256`, so the
returned vector is direct-quantize ready and skips PCA.

## Production retrieval (fully local)

For real semantic recall *without* network or API keys, use the local
sentence-transformer backend (BGE-small by default, ~90MB ONNX model
downloaded once into `~/.cache/fastembed`).

```bash
./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq-local \
  --embedder local --local-model bge-small-en

./target/release/primd query \
  --index /tmp/primd-faq-local \
  --text "what does the premium plan cost"
```

Available `--local-model` values: `all-minilm` (384-dim, fastest),
`bge-small-en` (384-dim, default, recommended), `bge-base-en` (768-dim, best).
All three feed through a deterministic 384/768→256 random projection before
sign-bit quantization, preserving Hamming-distance fidelity by the
Johnson-Lindenstrauss lemma.

## HTTP service

Once an index exists, expose it as a retrieval microservice:

```bash
./target/release/primd serve \
  --index /tmp/primd-faq \
  --bind 127.0.0.1:8080
```

```bash
# In another terminal
curl -s http://localhost:8080/health
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"is there a free trial","top_k":3}'
curl -s http://localhost:8080/stats
```

This is the integration point for Pipecat / LiveKit / Vocode plugins — any
voice-agent framework that can POST to a URL can use primd as its retrieval
layer without linking the Rust crate.

## What's wired up

The current `primd` CLI ships with a deterministic feature-hashing embedder
(no ML dependency). It demonstrates the full pipeline:

```
text → tokenize → hash features → 256-dim float → sign-bit → 32-byte signature → SignatureIndex → top-K
```

Indexing throughput on a Ryzen 7 7435HS: **~150K docs/sec** for the 30-entry
FAQ. Per-query embed+scan latency: **~10 µs**.

## Embedder caveat

The hashed embedder is for proof-of-pipeline. It captures literal token
overlap (with bigrams + stop word removal) but does *not* understand
paraphrase. So queries like "how much is premium" and corpus entries like
"the cost of the premium plan" share only the literal token "premium" —
which produces noisy results on small corpora where hash collisions
dominate.

For production retrieval quality, swap the `HashedEmbedder` for a real
sentence-embedding model (Sentence-Transformers / BGE-small / OpenAI). The
`Embedder` trait in `primd-core/src/embed/embedder.rs` is the integration
point — implement it for any backend and pass it to `EmbeddingPipeline`.

## File format

`faq.jsonl` is one JSON object per line:

```json
{"id":"faq-001","event":"pricing","text":"Our basic plan costs nine dollars per month..."}
```

- `id`: stable identifier returned by `primd query`.
- `text`: the document content to embed.
- `event` *(optional)*: groups related entries. Used by the prefetch
  coordinator (layer 3) to narrow scope per anticipated event.
