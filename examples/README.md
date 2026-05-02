# primd examples

End-to-end demo: text in → primd retrieval → top-K matches.

## Quick start

```bash
# Build the release binary once.
cargo build --release -p primd-cli

# Index the FAQ corpus.
./target/release/primd index \
  --input examples/faq.jsonl \
  --out /tmp/primd-faq

# Query it.
./target/release/primd query \
  --index /tmp/primd-faq \
  --text "is there a free trial"
```

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
