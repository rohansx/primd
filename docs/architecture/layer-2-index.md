# Layer 2 — Event-Segmented Hierarchical Index

**Brain analogue**: Hippocampus boundary/event cells (Zheng et al., Nature Neuroscience, 2022) + Sparse Distributed Memory (Kanerva, 1988).

**Core idea**: Replace flat HNSW with a two-level cache-friendly structure that keeps the hot path in CPU cache.

## The Problem It Solves

Flat HNSW at scale has two issues:

1. **Cache-hostile memory access.** Each neighbor hop in the graph is a pointer chase to a random memory location. At 1M+ vectors, the working set exceeds L2/L3 cache, causing frequent cache misses.
2. **p99 degradation.** While p50 stays at ~2-5ms, p99 can spike to hundreds of ms when graph traversal hits cold memory regions.

Layer 2 solves this by splitting retrieval into two stages with different memory characteristics.

## Architecture

### Stage 2a — Binary Signature Scan

Every document is summarized into a 256-bit binary signature:

```
Dense embedding (384 or 768 dims, float32)
    │
    ▼ sign-bit quantization (top-256 principal components)
    │
256-bit binary signature (32 bytes per document)
```

**Memory footprint**: 1M documents × 32 bytes = 32 MB. Fits in L2/L3 cache on any modern CPU.

**Scan operation**: Hamming distance between query signature and all stored signatures, using AVX-512 `VPOPCNTDQ`:

```rust
use std::arch::x86_64::*;

/// Scan all signatures, return top-K by hamming distance
pub unsafe fn hamming_scan_avx512(
    query: &[u8; 32],
    signatures: &[[u8; 32]],
    top_k: usize,
) -> Vec<(usize, u32)> {
    let q = _mm512_loadu_si512(query.as_ptr() as *const __m512i);
    // ... load remaining 256 bits

    let mut heap = MinHeap::with_capacity(top_k);

    for (i, sig) in signatures.iter().enumerate() {
        let s = _mm512_loadu_si512(sig.as_ptr() as *const __m512i);
        let xor = _mm512_xor_si512(q, s);
        let popcnt = _mm512_popcnt_epi64(xor);
        let dist = horizontal_sum_epi64(popcnt);

        heap.push_if_closer(i, dist);
    }

    heap.into_sorted_vec()
}
```

**Performance**: ~0.3-0.5ms for 1M signatures on a single thread (Ice Lake / Sapphire Rapids). Multi-threaded with `rayon`: ~0.15-0.25ms.

**Recall**: Binary scan alone produces ~74-77% recall@10. With oversampling (retrieve top-256) and float32 rescoring, recall recovers to 97-99%.

### Stage 2b — Per-Event HNSW Shards

Documents are grouped into "events" — topically coherent clusters of 100-1000 chunks:

```
Corpus (1M documents)
    │
    ├── Event 0:  "pricing & plans"     (342 chunks)
    ├── Event 1:  "product features"    (891 chunks)
    ├── Event 2:  "onboarding flow"     (156 chunks)
    ├── Event 3:  "billing & invoices"  (423 chunks)
    ├── ...
    └── Event N:  "competitor comparisons" (267 chunks)
```

Each event has its own small HNSW graph indexing the full-precision (float32) embeddings of its chunks. Searching a ~1K-node HNSW shard takes ~0.3-0.5ms with excellent cache behavior (the entire shard fits in L2).

### Combined Flow

```
Query embedding
    │
    ▼
Binary signature scan (2a) → top-256 event candidates (~0.3ms)
    │
    ▼
Identify top-3 matching events
    │
    ▼
HNSW search within matching event shards (2b) → top-K results (~0.3-0.5ms)
    │
    ▼
Float32 rescore → final ranked results (~0.1ms)
    │
    ▼
Total: ~0.5-1ms
```

## Event Boundary Detection

The quality of layer 2 depends entirely on how well documents are grouped into events. Bad boundaries = bad index.

### Default: Semantic Clustering

At index time, documents are clustered using k-means on their dense embeddings:

1. Embed all documents with the same model used for retrieval (MiniLM / bge-base)
2. Run k-means with silhouette-score-guided k selection
3. Target 100-1000 documents per event (adaptive based on corpus size)
4. Events with >1000 documents are recursively split

### Domain-Specific Boundary Detectors (Plugin API)

For structured domains, better boundaries are possible:

```rust
pub trait BoundaryDetector: Send + Sync {
    /// Given a document, return its event label
    fn detect_event(&self, doc: &Document) -> EventLabel;
}

// Built-in detectors:
// - SemanticCluster: k-means (default)
// - HeadingHierarchy: uses document headings (for docs/wikis)
// - IntentClassifier: uses intent labels (for support tickets)
// - SalesStage: discovery/demo/pricing/close (for sales corpora)
```

### Boundary Quality Metrics

primd ships a boundary quality benchmark alongside the retrieval benchmark:

- **Silhouette score**: How well-separated are events? (target: >0.3)
- **Intra-event coherence**: Average pairwise cosine similarity within events (target: >0.7)
- **Inter-event separation**: Average centroid distance between events (target: >0.4)
- **Size distribution**: Std deviation of event sizes (target: <2x mean)

## Binary Quantization Details

| Parameter | Value |
|---|---|
| Source embedding dim | 384 (MiniLM) or 768 (bge-base) |
| Binary signature dim | 256 bits |
| Quantization method | Sign-bit on top-256 principal components |
| Memory per document | 32 bytes (signature) + ~6KB (float32 in shard) |
| Recall without rescore | ~74-77% at recall@10 |
| Recall with rescore | ~97-99% at recall@10 |
| Rescore overhead | ~0.1ms (float32 cosine on 256 candidates) |

## On-Disk Format

```
corpus/
├── signatures.bin      # packed binary: [u8; 32] × N, mmap-able
├── events/
│   ├── 0000.hnsw       # per-event HNSW graph (mmap-able)
│   ├── 0000.vecs       # per-event float32 vectors (mmap-able)
│   ├── 0001.hnsw
│   ├── 0001.vecs
│   └── ...
├── event_map.bin       # doc_id → event_id mapping
└── pca_matrix.bin      # PCA projection matrix for binary quantization
```

All files are memory-mapped. No explicit load phase — the OS handles paging.

## Configuration

```toml
[layer2]
enabled = true
binary_dim = 256                    # bits in signature
oversample_factor = 256             # top-K from binary scan before rescore
max_event_size = 1000               # split events larger than this
min_event_size = 50                 # merge events smaller than this
boundary_detector = "semantic"      # or "heading", "intent", "sales_stage"
hnsw_ef_construction = 200          # HNSW build parameter
hnsw_ef_search = 64                 # HNSW search parameter
hnsw_m = 16                         # HNSW connections per node
```

## Limitations

- **Binary quantization is lossy.** For high-stakes domains (legal, medical) where 1-3% recall loss is unacceptable, use `quantization = "fp16"` for a middle ground (half memory, negligible recall loss, ~2x latency).
- **Event boundary quality is a bottleneck.** Poorly chosen boundaries degrade both binary scan precision (wrong events selected) and HNSW shard efficiency (shards too large or too heterogeneous). Always run `primd boundary-quality` after indexing.
- **PCA projection requires training data.** The top-256 principal components are computed from a sample of the corpus. If the corpus distribution shifts significantly after indexing, recompute with `primd reindex`.
