# Layer 2 — Event-Segmented Hierarchical Index

**Brain analogue**: Hippocampus boundary/event cells (Zheng et al., Nature Neuroscience, 2022) + Sparse Distributed Memory (Kanerva, 1988).

**Core idea**: Two-stage retrieval — a SIMD-friendly coarse scan over the whole corpus to locate candidate events, then a tighter rescan over the union of those events' document scopes. The hot path stays in cache; the rescan is bounded by the event structure.

## What ships in v0.1

`primd-core/src/index/shards.rs` implements `HierarchicalIndex::search` as:

1. **Coarse scan** — `SignatureIndex::scan_top_k_parallel` runs a SIMD Hamming scan over every signature in the corpus to find a coarse top-K (`SearchOptions::coarse_k`, default 64).
2. **Event candidate selection** — the top coarse hits are mapped to candidate events via `EventCatalog::candidate_events_from_docs`, capped at `max_candidate_events` (default 8).
3. **Scope union + subset rescan** — the union of the candidate events' doc indices is gathered into a contiguous `Vec<BinarySignature>` and SIMD-rescanned via `scan_top_k_subset_parallel` to produce the final top-K.

This is **not HNSW**. It's a deliberate SIMD-first design that works because:

- 256-bit signatures pack at 32 bytes/doc; 100 k docs = 3.2 MB, fits comfortably in L2/L3 cache
- AVX-512 VPOPCNTDQ does a 256-bit XOR + popcount per cycle
- Event scopes for a typical voice corpus are usually 200–2000 docs, so the gather is cheap and the rescan throughput is dominated by SIMD width, not pointer chasing

What it gives up vs HNSW: scaling beyond ~1 M docs. For 1 M+ corpora the linear coarse scan starts to dominate; v0.2 adds per-event HNSW shards for that regime.

## The Problem It Solves

Flat HNSW at scale has two issues:

1. **Cache-hostile memory access.** Each neighbor hop in the graph is a pointer chase to a random memory location. At 1M+ vectors, the working set exceeds L2/L3 cache, causing frequent cache misses.
2. **p99 degradation.** While p50 stays at ~2-5ms, p99 can spike to hundreds of ms when graph traversal hits cold memory regions.

Layer 2 solves this for the 10 k–1 M corpus range by splitting retrieval into a SIMD coarse scan + a bounded subset rescan that both have predictable, cache-resident memory access patterns.

## Architecture (shipped, v0.1)

### Stage 2a — Binary signature coarse scan

Every document is summarized into a 256-bit binary signature:

```
Dense embedding (384 or 768 dims, float32)
    │
    ▼ sign-bit quantization (top-256 principal components)
    │
256-bit binary signature (32 bytes per document)
```

**Memory footprint**: 1M documents × 32 bytes = 32 MB. Fits in L2/L3 cache on any modern CPU.

**Scan operation**: Hamming distance between query signature and all stored signatures. The implementation in `primd-core/src/index/signatures.rs` picks the best available kernel at startup:

- **AVX-512 path** (`scan_avx512`): uses `_mm256_popcnt_epi64` (VPOPCNTDQ) for a single-instruction 64-bit popcount. Available on Intel Ice Lake / Sapphire Rapids+ and AMD Zen 4+.
- **AVX2 path** (`scan_avx2`): nibble-lookup via `_mm256_shuffle_epi8` (VPSHUFB) for popcount on hardware without VPOPCNTDQ.
- **Scalar fallback**: portable `u64::count_ones()`.

A `OnceLock` caches the detected level; rayon chunks the work for parallel scans.

**Measured performance** (100 k corpus, `cargo bench --bench voice_session`):
- Coarse single-thread scan: ~100–200 µs
- Parallel scan: roughly linear with cores up to memory-bandwidth limit

**Recall**: Binary scan alone produces ~74-77% recall@10 on dense-embedding-derived signatures. With oversampling (`coarse_k` ≥ 4×`top_k`) and an optional float32 rescore step, recall recovers to 97-99%.

### Stage 2b — Event-scoped subset rescan

Documents are grouped into "events" — topically coherent clusters identified at index time:

```
Corpus (100 k documents)
    │
    ├── Event 0:  "pricing & plans"     (342 chunks)
    ├── Event 1:  "product features"    (891 chunks)
    ├── Event 2:  "onboarding flow"     (156 chunks)
    ├── Event 3:  "billing & invoices"  (423 chunks)
    ├── ...
    └── Event N:  "competitor comparisons" (267 chunks)
```

The coarse top-K hits are mapped to candidate events; the union of those events' doc indices is gathered into a contiguous scratch buffer and SIMD-rescanned. The cost is dominated by the gather + rescan over the union scope, which is typically a few hundred to a few thousand signatures — small enough that the OS prefetcher and L1 cache handle it well.

`SearchOptions` controls the cutoff:

| Field | Default | Meaning |
|---|---|---|
| `coarse_k` | 64 | Top-K from the global Hamming scan |
| `max_candidate_events` | 8 | Cap on how many distinct events the coarse hits can resolve to |
| `parallel` | true | Use rayon for the coarse and subset scans |

### Stage 2b v0.2 — Real per-event HNSW shards (planned)

For corpora where the union scope exceeds ~5–10 k docs (large multi-tenant deployments, very long sessions with broad scope unions), the subset rescan starts to dominate. v0.2 will add an actual HNSW graph per event, indexed by the float32 embeddings of the event's docs:

```
events/
├── 0000.hnsw       # per-event HNSW graph (mmap-able)
├── 0000.vecs       # per-event float32 vectors (mmap-able)
├── 0001.hnsw
├── 0001.vecs
└── ...
```

Searching a 1–10 k-node HNSW shard takes ~0.3–0.5 ms with excellent cache behavior. Likely impl: `instant-distance` or `hnsw_rs`. The trait surface in `shards.rs` already isolates the rescan step, so this is a localized change.

## Event boundary detection

The quality of layer 2 depends entirely on how well documents are grouped into events. Bad boundaries = bad index.

### Default: Semantic Clustering

At index time, documents are clustered using k-means on their dense embeddings:

1. Embed all documents with the same model used for retrieval (MiniLM / bge-base)
2. Run k-means with silhouette-score-guided k selection
3. Target 100-1000 documents per event (adaptive based on corpus size)
4. Events with >1000 documents are recursively split

### Domain-Specific Boundary Detectors (Plugin API — planned)

For structured domains, better boundaries are possible:

```rust
pub trait BoundaryDetector: Send + Sync {
    /// Given a document, return its event label
    fn detect_event(&self, doc: &Document) -> EventLabel;
}

// Planned detectors:
// - SemanticCluster: k-means (default, shipping)
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
| Memory per document | 32 bytes (signature); v0.2 HNSW adds ~6 KB float32 per doc |
| Recall without rescore | ~74-77% at recall@10 |
| Recall with rescore | ~97-99% at recall@10 |
| Rescore overhead | ~0.1 ms (float32 cosine on 256 candidates) |

## On-Disk Format (v0.1, shipped)

```
corpus/
├── signatures.bin      # packed binary: [u8; 32] × N, mmap-able
├── manifest.json       # embedder kind, event scope, doc ids
└── transitions.json    # Markov predictor state (optional)
```

v0.2 will add `events/NNNN.hnsw` + `events/NNNN.vecs` once per-event HNSW shards land. All files are memory-mapped via `memmap2`. No explicit load phase — the OS handles paging.

## Configuration

```toml
[layer2]
enabled = true
binary_dim = 256                    # bits in signature
coarse_k = 64                       # top-K from coarse SIMD Hamming scan
max_candidate_events = 8            # cap on candidate event count
max_event_size = 1000               # split events larger than this
min_event_size = 50                 # merge events smaller than this
boundary_detector = "semantic"      # or "heading", "intent", "sales_stage" (planned)

# v0.2 HNSW config (planned)
# hnsw_ef_construction = 200
# hnsw_ef_search = 64
# hnsw_m = 16
```

## Limitations

- **Binary quantization is lossy.** For high-stakes domains (legal, medical) where 1-3% recall loss is unacceptable, ship an fp16 rescore tier (planned). Workaround today: hand off to the user's vector DB after coarse scan.
- **Event boundary quality is a bottleneck.** Poorly chosen boundaries degrade both coarse-scan precision (wrong events selected) and the subset rescan's working-set size. Always run `primd boundary-quality` after indexing.
- **PCA projection requires training data.** The top-256 principal components are computed from a sample of the corpus. If the corpus distribution shifts significantly after indexing, recompute with `primd reindex`.
- **v0.1 subset rescan scales linearly with union-scope size.** At ≥ 5–10 k docs per union, the rescan dominates and v0.2 HNSW becomes necessary.
