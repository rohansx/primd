# Technical Specification

## Language & Runtime

- **Rust**, `no_std`-compatible core where possible
- Bindings: `pyo3` (Python), `napi-rs` (Node/TypeScript), `wasm-bindgen` (browser)
- Single binary, target ~8-12 MB
- Apache-2.0 license

## Crate Structure

```
primd/
├── primd-core/              # Pure Rust, no_std where possible
│   ├── embed/
│   │   ├── streaming.rs     # Full re-embed on partials, similarity gate
│   │   └── binary.rs        # float → 256-bit signature via sign-bit + PCA
│   ├── index/
│   │   ├── signatures.rs    # Packed binary array, AVX-512 hamming scan
│   │   ├── shards.rs        # Per-event HNSW using hnsw_rs or instant-distance
│   │   ├── events.rs        # Event boundary detection, clustering
│   │   └── loader.rs        # mmap + lazy load, NUMA-aware
│   ├── predict/
│   │   ├── markov.rs        # Sparse transition matrix (CSR format)
│   │   ├── trainer.rs       # Offline: transcripts → matrix
│   │   └── prefetch.rs      # Async cache warmer (madvise/prefault)
│   ├── coding/
│   │   ├── centroid.rs      # Topic-radius cache + centroid drift
│   │   └── delta.rs         # 1-hop delta search on continuation
│   └── lib.rs               # Public API: Index, QueryContext, Config
│
├── primd-py/                # pyo3 bindings
│   ├── src/lib.rs
│   └── pyproject.toml
│
├── primd-js/                # napi-rs bindings (Node.js)
│   ├── src/lib.rs
│   └── package.json
│
├── primd-wasm/              # wasm-bindgen (browser / edge)
│   ├── src/lib.rs
│   └── Cargo.toml
│
├── primd-cli/               # CLI: serve, index, train, bench
│   ├── src/
│   │   ├── main.rs
│   │   ├── cmd_index.rs     # primd index <corpus-dir>
│   │   ├── cmd_train.rs     # primd train <transcripts-dir>
│   │   ├── cmd_serve.rs     # primd serve --port 8080
│   │   └── cmd_bench.rs     # primd bench --baseline faiss
│   └── Cargo.toml
│
├── primd-bench/             # Benchmark suite
│   ├── benches/
│   │   ├── binary_scan.rs   # Layer 2a microbenchmark
│   │   ├── hnsw_shard.rs    # Layer 2b microbenchmark
│   │   ├── end_to_end.rs    # Full pipeline benchmark
│   │   └── baselines/       # FAISS, Qdrant, Moss configs
│   ├── corpora/             # Test corpora (or download scripts)
│   └── docker-compose.yml   # Stand up all competitors
│
├── primd-plugins/
│   ├── pipecat/             # Pipecat PrimdRetriever processor
│   │   ├── primd_pipecat/
│   │   │   └── retriever.py
│   │   └── pyproject.toml
│   ├── livekit/             # LiveKit agent adapter
│   │   ├── primd_livekit/
│   │   │   └── plugin.py
│   │   └── pyproject.toml
│   └── openai-compat/       # OpenAI-compatible RAG endpoint
│       └── src/main.rs
│
├── examples/
│   ├── sdr-bot/             # Full working SDR voice agent
│   ├── support-bot/         # Knowledge-base customer support
│   └── browser-demo/        # WASM in-browser demo
│
├── Cargo.toml               # Workspace root
├── Cargo.lock
├── Makefile                  # make build, make test, make bench
└── README.md
```

## Core API

### Rust

```rust
use primd::{Index, QueryContext, Config, SearchResult};

// Open an indexed corpus (mmap, near-instant)
let config = Config::from_file("primd.toml")?;
let idx = Index::open("/var/lib/primd/corpus", config)?;

// Create a per-conversation session
let mut ctx = QueryContext::new(&idx);

// Per-turn: feed partial transcripts from STT stream
ctx.observe_partial("so what about");
ctx.observe_partial("so what about pri");
ctx.observe_partial("so what about pricing");

// At end-of-utterance: get final results
let results: Vec<SearchResult> = ctx.finalize().await?;
// results[0].text = "Enterprise plan starts at $499/mo..."
// results[0].score = 0.94
// results[0].metadata = {"source": "pricing.md", "event": "pricing"}

// During TTS playback: prefetch next likely answers (non-blocking)
ctx.warm_next().await;

// Inspect what layer served the result
assert_eq!(results[0].served_by, Layer::PredictiveCache); // or BinaryScan, HnswShard, DeltaCache
```

### Python

```python
from primd import Index, QueryContext

idx = Index.open("/var/lib/primd/corpus")
ctx = idx.session()

# Streaming: feed partials as they arrive
async for partial in stt_stream:
    ctx.observe(partial)

# Finalize: get results
results = await ctx.finalize()

# Prefetch during TTS
await ctx.warm_next()

# Access results
for r in results:
    print(f"{r.score:.2f} | {r.text[:80]}...")
    print(f"  source: {r.metadata['source']}, layer: {r.served_by}")
```

### TypeScript

```typescript
import { Index, QueryContext } from '@primd/node';

const idx = await Index.open('/var/lib/primd/corpus');
const ctx = idx.session();

// Feed partials
for await (const partial of sttStream) {
  ctx.observe(partial);
}

// Get results
const results = await ctx.finalize();

// Prefetch
await ctx.warmNext();
```

## CLI Commands

```bash
# Index a corpus from source documents
primd index \
  --input ./documents/ \
  --output ./corpus/ \
  --model minilm-l6-v2 \
  --format parquet  # or: directory, qdrant-export, pgvector-dump

# Train transition matrix from conversation transcripts
primd train \
  --transcripts ./calls/ \
  --corpus ./corpus/ \
  --output ./corpus/transitions.bin \
  --min-conversations 500

# Check index quality
primd check ./corpus/
# → signatures: 1,000,000 docs, 32MB
# → events: 1,247 events, avg 802 docs/event
# → boundary quality: silhouette=0.41, coherence=0.73
# → transition matrix: 1,247 events, 18,432 transitions

# Run benchmarks
primd bench \
  --corpus ./corpus/ \
  --queries ./test-queries.jsonl \
  --baselines faiss,qdrant

# Serve as HTTP endpoint (OpenAI-compatible)
primd serve --corpus ./corpus/ --port 8080
```

## Dependencies

| Crate | Purpose | Why This One |
|---|---|---|
| `ort` | ONNX runtime for embedder inference | Most mature ONNX runtime in Rust |
| `candle-core` | Alternative: native Rust inference | No C++ dependency, better for WASM |
| `hnsw_rs` | Per-event HNSW shards | Pure Rust, no_std-friendly |
| `instant-distance` | Alternative HNSW | Simpler API, good for small graphs |
| `memmap2` | Memory-mapped index files | Standard mmap crate |
| `bytemuck` | Zero-copy binary operations | Safe transmutes for packed arrays |
| `rayon` | Parallel binary signature scan | Standard data parallelism |
| `tokio` | Async prefetch, HTTP server | Standard async runtime |
| `simsimd` | SIMD distance primitives (optional) | Hand-tuned AVX-512/NEON kernels |
| `arrow2` / `parquet2` | Read parquet corpora | Zero-copy parquet access |

## Index Format (On Disk)

```
corpus/
├── meta.toml              # Version, dims, model hash, domain, stats
├── signatures.bin         # Packed binary: [u8; 32] × N documents
├── pca_matrix.bin         # PCA projection: [f32; source_dim × 256]
├── events/
│   ├── 0000.hnsw          # Per-event HNSW adjacency lists
│   ├── 0000.vecs          # Per-event float32 vectors
│   ├── 0001.hnsw
│   ├── 0001.vecs
│   └── ...
├── event_map.bin          # doc_id → event_id: [u32; N]
├── chunks.parquet         # Document text + metadata (Apache Arrow format)
├── transitions.bin        # Sparse Markov matrix (CSR format)
└── centroids.bin          # Per-event centroid embeddings: [f32; dim × num_events]
```

### meta.toml Example

```toml
[index]
version = "0.1.0"
created = "2026-04-23T10:30:00Z"
num_documents = 1000000
num_events = 1247
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim = 384
binary_dim = 256
model_hash = "a1b2c3d4e5f6"

[corpus]
domain = "sales"
source = "qdrant-export"
boundary_detector = "semantic"

[quality]
silhouette_score = 0.41
intra_event_coherence = 0.73
inter_event_separation = 0.45
avg_event_size = 802
```

### File Size Estimates (1M Documents)

| File | Size | Notes |
|---|---|---|
| signatures.bin | 32 MB | 1M × 32 bytes, fits in L2/L3 |
| events/*.hnsw | ~200 MB total | ~1247 shards, ~160KB each |
| events/*.vecs | ~1.5 GB total | 1M × 384 × 4 bytes (float32) |
| chunks.parquet | ~2-5 GB | Depends on document length |
| transitions.bin | ~1 MB | Sparse, ~20 transitions per event |
| centroids.bin | ~1.9 MB | 1247 × 384 × 4 bytes |
| pca_matrix.bin | ~393 KB | 384 × 256 × 4 bytes |
| event_map.bin | ~4 MB | 1M × 4 bytes |
| **Total index** | **~1.7 GB** | Excluding corpus text |
| **Hot working set** | **~35 MB** | signatures.bin + active shards |

## Hardware Requirements

### Minimum (Development)

- 4 CPU cores
- 8 GB RAM
- AVX2 support (any x86 CPU from 2013+)

### Recommended (Production, 1M docs)

- 8+ CPU cores (Ice Lake or newer for AVX-512)
- 16 GB RAM (for full index in page cache)
- SSD (for mmap performance)

### WASM (Browser)

- Any modern browser with WASM SIMD support
- Practical limit: ~100K documents (memory constraints)
- ~3-10ms latency (vs <1ms native)

## SIMD Strategy

| Platform | SIMD Width | Instructions | Expected Performance |
|---|---|---|---|
| Ice Lake / Sapphire Rapids | 512-bit | AVX-512 + VPOPCNTDQ | ~0.3ms / 1M sigs (multi-threaded) |
| Zen 4 (AMD) | 256-bit (AVX-512 emulated) | AVX-512 | ~0.5ms / 1M sigs |
| Apple Silicon (M1-M3) | 128-bit | NEON | ~1ms / 1M sigs |
| WASM | 128-bit | WASM SIMD | ~3-6ms / 1M sigs |
| Fallback (no SIMD) | 64-bit | Scalar | ~5-10ms / 1M sigs |

primd uses runtime feature detection to select the best available SIMD path. The `signatures.rs` module has separate implementations for each tier, selected at startup.
