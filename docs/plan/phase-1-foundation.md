# Phase 1 — Foundation (Weeks 1-4)

Build the core retrieval engine and prove it works.

## Goal

A working two-stage retrieval system (binary scan + per-event HNSW) that matches or beats flat HNSW in latency while maintaining 97%+ recall on 1M documents.

## Week 1-2: Binary Signature Scan

### Tasks

1. **Set up Rust workspace**
   - Initialize cargo workspace with `primd-core`, `primd-cli`, `primd-bench`
   - CI pipeline: `cargo test`, `cargo clippy`, `cargo fmt`
   - Basic project scaffolding

2. **Implement binary quantization** (`primd-core/embed/binary.rs`)
   - Load pre-trained PCA matrix (computed offline from corpus sample)
   - Float32 embedding → top-256 principal components → sign-bit → 256-bit signature
   - Unit tests: verify round-trip quality (cosine correlation between original and reconstructed)

3. **Implement packed signature storage** (`primd-core/index/signatures.rs`)
   - Packed `[u8; 32]` array, mmap-able
   - Write/read from `signatures.bin`
   - Unit tests: serialize/deserialize round-trip

4. **Implement AVX-512 hamming scan** (`primd-core/index/signatures.rs`)
   - SIMD-accelerated scan using `_mm512_popcnt_epi64`
   - Fallback path for AVX2 (256-bit) and NEON (128-bit)
   - Scalar fallback for platforms without SIMD
   - Runtime feature detection for SIMD path selection
   - Top-K min-heap for collecting nearest signatures

5. **Microbenchmark** (`primd-bench/benches/binary_scan.rs`)
   - Generate 1M random 256-bit signatures
   - Benchmark: single-thread and multi-thread (rayon, 2/4/8 threads)
   - Target: <0.5ms single-thread, <0.3ms with 2 threads on Ice Lake

### Deliverables

- `signatures.rs` with SIMD-accelerated hamming scan
- `binary.rs` with float32 → binary quantization
- Benchmark results: latency at 10K, 100K, 500K, 1M signatures
- CI passing

## Week 3-4: Per-Event HNSW Shards

### Tasks

1. **Implement event boundary detection** (`primd-core/index/events.rs`)
   - k-means clustering on dense embeddings (using `linfa` or custom impl)
   - Silhouette-score-guided k selection
   - Recursive splitting for events >1000 docs
   - Merging for events <50 docs
   - Boundary quality metrics: silhouette, coherence, separation

2. **Implement per-event HNSW shards** (`primd-core/index/shards.rs`)
   - Build small HNSW graph per event using `hnsw_rs` or `instant-distance`
   - Serialize each shard to `events/NNNN.hnsw` + `events/NNNN.vecs`
   - mmap-based loading (lazy, on-demand)

3. **Implement rescore pipeline**
   - Binary scan (layer 2a) → top-256 candidates
   - Map candidates to events → load relevant HNSW shards
   - HNSW search within shards → top-K
   - Float32 rescore → final ranked results

4. **Build indexing CLI** (`primd-cli/cmd_index.rs`)
   - `primd index --input <docs> --output <corpus> --model minilm-l6-v2`
   - Supports: directory of text files, parquet, JSONL
   - Outputs: complete corpus directory with all index files

5. **End-to-end benchmark** (`primd-bench/benches/end_to_end.rs`)
   - Index MS-MARCO (8.8M passages) or a 1M subset
   - Measure: p50, p95, p99 latency
   - Measure: recall@10 vs flat HNSW (FAISS) at equal recall
   - Compare with FAISS HNSW as baseline

### Deliverables

- Complete layer 2 (2a + 2b) implementation
- `primd index` CLI command working
- Benchmark: primd vs FAISS at 1M docs
- recall@10 >= 0.97
- p50 latency <= 1ms

### Exit Criteria

Phase 1 is complete when:
- [x] Binary scan <0.5ms on 1M signatures (single-thread)
- [x] Per-event HNSW search <0.5ms
- [x] Combined retrieval <1ms at p50
- [x] recall@10 >= 0.97 vs flat HNSW
- [x] `primd index` produces a valid corpus from parquet/text input
- [x] Benchmark results documented
