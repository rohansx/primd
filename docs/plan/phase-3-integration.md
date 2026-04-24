# Phase 3 — Integration (Weeks 9-12)

Build the delta cache, framework plugins, demo, and launch materials.

## Goal

A complete, benchmarked, documented, and demo-able product ready for public launch.

## Week 9-10: Delta Cache + End-to-End Benchmarks

### Tasks

1. **Implement topic cache** (`primd-core/coding/centroid.rs`)
   - Centroid computation from result set
   - Topic radius: configurable, default 0.3 cosine distance
   - Centroid drift via EMA (alpha=0.1)
   - `max_turns` staleness protection (default: 5)
   - `is_continuation(query_embedding)` → bool

2. **Implement delta search** (`primd-core/coding/delta.rs`)
   - Rescore cached results against new query
   - 1-hop HNSW neighbor expansion (top result → 10 neighbors)
   - Return reranked results

3. **Integrate all four layers**
   - Full data flow: layer 4 check → layer 3 cache → layer 1 streaming → layer 2 search
   - Diagnostic metadata: `served_by` field on each result (DeltaCache, PredictiveCache, BinaryScan, HnswShard)
   - Latency breakdown per layer in debug mode

4. **End-to-end benchmarks** (`primd-bench/`)
   - Scenario 1: Single-query latency (primd vs FAISS vs Qdrant vs Pinecone)
   - Scenario 2: Conversational sequence (primd vs VoiceAgentRAG)
   - Scenario 3: Scale sweep (10K → 1M → 5M)
   - Generate `bench-report.md`

5. **Configuration system** (`primd.toml`)
   - Per-layer enable/disable
   - All tuning parameters documented
   - Sensible defaults for sales, support, and general domains

### Deliverables

- Complete 4-layer retrieval pipeline
- End-to-end benchmark results vs all baselines
- Configuration system with documented defaults
- `bench-report.md` with reproducible numbers

## Week 11: Framework Plugins + Demo

### Tasks

1. **Pipecat plugin** (`primd-plugins/pipecat/`)
   - `PrimdRetriever` — a Pipecat processor that:
     - Receives interim transcripts from STT processor
     - Calls `ctx.observe()` on each partial
     - Calls `ctx.finalize()` at end-of-utterance
     - Emits retrieved context to the LLM processor
     - Calls `ctx.warm_next()` when TTS starts playing
   - Drop-in: `pipeline.add(PrimdRetriever(corpus_path="..."))`
   - `pip install primd-pipecat`

2. **LiveKit plugin** (`primd-plugins/livekit/`)
   - LiveKit agent plugin with same interface
   - Hooks into LiveKit's transcription events
   - `pip install primd-livekit`

3. **SDR bot demo** (`examples/sdr-bot/`)
   - Complete working voice agent using Pipecat + primd
   - Deepgram Nova-3 (STT) + GPT-4o-mini (LLM) + Cartesia (TTS)
   - Sample knowledge base: fictional SaaS pricing (50 docs)
   - Demonstrates: streaming retrieval, predictive prefetch, delta cache
   - Side-by-side comparison: same bot with vs without primd
   - Latency overlay showing which layer served each result

4. **Browser demo** (`examples/browser-demo/`)
   - WASM build of primd-core
   - Simple web page with text input (simulating voice)
   - Shows real-time retrieval with layer indicators
   - Deploys to GitHub Pages

### Deliverables

- `primd-pipecat` package (PyPI)
- `primd-livekit` package (PyPI)
- Working SDR bot demo with comparison mode
- Browser WASM demo on GitHub Pages

## Week 12: Launch

### Tasks

1. **arxiv preprint**
   - Title: "primd: Sub-Millisecond Predictive Retrieval for Real-Time Voice AI"
   - Sections: problem, architecture, benchmarks, related work
   - All benchmark numbers from week 9-10
   - Code and data availability statement

2. **GitHub repository**
   - Clean up codebase, ensure `make build && make test && make bench` works
   - README with quick start, benchmarks, architecture diagram
   - Contributing guide
   - Apache-2.0 license

3. **Launch blog post**
   - Problem → insight → architecture → benchmarks → how to use
   - Include comparison charts (primd vs FAISS vs Qdrant vs Pinecone vs VoiceAgentRAG)
   - Embed the browser demo

4. **Distribution**
   - Post to HN (Show HN)
   - Post to Twitter/X
   - Submit to Pipecat and LiveKit community channels
   - Publish crate to crates.io
   - Publish packages to PyPI and npm

### Deliverables

- arxiv preprint submitted
- GitHub repo public with Apache-2.0 license
- Blog post published
- HN/Twitter/X posts live
- Packages on crates.io, PyPI, npm

### Exit Criteria

Phase 3 (and the MVP) is complete when:
- [ ] All four layers integrated and working end-to-end
- [ ] p50 <1ms, p99 <3ms on SDR dialogue corpus
- [ ] Benchmark report with all baselines published and reproducible
- [ ] Pipecat and LiveKit plugins on PyPI
- [ ] SDR bot demo running and demonstrable
- [ ] WASM demo on GitHub Pages
- [ ] arxiv preprint submitted
- [ ] GitHub repo public
- [ ] Launch posts on HN and Twitter
