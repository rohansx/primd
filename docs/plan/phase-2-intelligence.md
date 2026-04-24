# Phase 2 — Intelligence (Weeks 5-8)

Add the brain-inspired predictive layers that make primd unique.

## Goal

A retrieval system that starts searching before the user finishes speaking and predicts what they'll ask next. Measured on real conversational workloads.

## Week 5-6: Streaming Partial-Query Embedder

### Tasks

1. **Integrate ONNX runtime** (`primd-core/embed/streaming.rs`)
   - Bundle MiniLM-L6-v2 (ONNX, int8 quantized) — or download on first use
   - `ort` crate for native, `candle-core` for WASM fallback
   - Embed function: `fn embed(text: &str) -> Vec<f32>` (384-dim)
   - Benchmark: latency per query at various sequence lengths

2. **Implement streaming retriever**
   - `observe_partial(text: &str)` — called on each STT interim result
   - Token-count trigger (every 3 tokens)
   - Cosine similarity gate (skip if embedding hasn't changed >0.05)
   - Speculative binary scan (layer 2a only) → top-100 candidates
   - `finalize(text: &str)` — full embed + rerank speculative candidates

3. **Build QueryContext** (`primd-core/lib.rs`)
   - Per-conversation state: streaming retriever + prediction context + topic cache
   - Public API: `observe_partial()`, `finalize()`, `warm_next()`
   - Thread-safe: `Send + Sync`

4. **Test on real voice traces**
   - Record partial transcript sequences from Deepgram Nova-3 on sample audio
   - Replay through primd's streaming retriever
   - Measure: how early do speculative results match final results?
   - Measure: total embed compute per utterance

5. **Python bindings** (`primd-py/`)
   - `pyo3` wrapping of `Index` and `QueryContext`
   - Async support via `pyo3-asyncio`
   - `pip install primd` (maturin build)

### Deliverables

- Streaming retriever with full re-embed + similarity gate
- MiniLM-L6-v2 embedded inference (ONNX int8)
- QueryContext public API
- Python SDK (`pip install primd`)
- Voice trace test results: speculative accuracy and timing

### Key Design Decision: Full Re-embed vs EMA

**Decision**: Full re-embed on each partial, NOT EMA.

**Rationale**: EMA over token embeddings degrades retrieval quality by 10-20% (Reimers & Gurevych, 2019). Full re-embed costs ~3-5ms per partial but maintains retrieval quality. With the similarity gate skipping redundant re-embeds, the actual compute per utterance is ~9-15ms spread across 1.2 seconds of speech — negligible.

## Week 7-8: Markov Transition Matrix

### Tasks

1. **Build training pipeline** (`primd-core/predict/trainer.rs`)
   - Input: conversation transcripts (JSONL, one turn per line)
   - Process: segment turns → match to events → extract transitions
   - Output: `transitions.bin` (CSR sparse matrix)
   - `primd train --transcripts <dir> --corpus <corpus>`

2. **Implement predictor** (`primd-core/predict/markov.rs`)
   - Load CSR matrix from `transitions.bin`
   - `predict_next(current_event, context)` → top-5 events with probabilities
   - Context window: last 3 events (configurable)

3. **Implement async prefetcher** (`primd-core/predict/prefetch.rs`)
   - `warm_next()` — called during TTS playback
   - Issues `madvise(MADV_WILLNEED)` on predicted event shard mmap regions
   - Non-blocking, fire-and-forget

4. **Build SDR dialogue corpus**
   - Source: MultiWOZ (public, 10K conversations) + synthetic from Gong patterns
   - 50K conversations, 20 turns average
   - Annotated with topic labels
   - Released under CC-BY-4.0

5. **Train and evaluate**
   - Train on 80% of SDR corpus, evaluate on 20%
   - Measure: cache hit rate (overall and per-scenario)
   - Measure: prediction latency
   - Compare with VoiceAgentRAG's published numbers

6. **TypeScript bindings** (`primd-js/`)
   - `napi-rs` wrapping of `Index` and `QueryContext`
   - `npm install @primd/node`

### Deliverables

- Training pipeline + CLI command
- Markov predictor with async prefetch
- SDR dialogue corpus (public, CC-BY-4.0)
- Cache hit rate benchmarks by domain
- TypeScript SDK (`npm install @primd/node`)

### Exit Criteria

Phase 2 is complete when:
- [ ] Streaming retriever fires speculative results within 100ms of first meaningful partial
- [ ] Speculative candidates overlap >= 80% with final results (by end of utterance)
- [ ] Markov predictor achieves >= 70% hit rate on structured SDR corpus
- [ ] Prediction latency < 100μs
- [ ] Prefetch completes within TTS window (measured)
- [ ] Python and TypeScript SDKs working with async API
- [ ] Conversational benchmark: p50 <1ms on the SDR dialogue corpus
