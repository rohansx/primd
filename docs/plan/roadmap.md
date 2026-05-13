# Roadmap

> Last updated 2026-05-13. Replaces the original 12-week MVP plan now that v0.1 has shipped. See [strategy-2026-05.md](../business/strategy-2026-05.md) for the strategic memo this roadmap derives from.

The shape of the work is now: **defend the wedge** (turn-cache positioning, real benchmarks vs Moss), **deepen the wedge** (real per-event HNSW, Successor Representation predictor), **extend the wedge** (Hippocampus DWM cold tier, full-duplex hooks).

## v0.1 — Shipped (Q2 2026)

- SIMD binary signature search (AVX-512 VPOPCNTDQ → AVX2 VPSHUFB → scalar)
- `QueryContext` session runtime: observe / finalize / warm / reset
- Session-aware HTTP API (`/query`, `/session/{id}/{observe,finalize,warm,reset}`, `/health`, `/stats`)
- Variable-order Markov predictor with half-life time decay, smoothing, persistence
- Predictive-coding delta cache
- `pipecat-primd` Python package (`FrameProcessor` + async client)
- Voice-realistic benchmark harness (`cargo bench --bench voice_session`)
- Hashed / OpenAI / fastembed local embedders
- crates.io-publish-ready (`primd-core`, `primd-cli`)

## v0.2 — Turn-cache moat (Q3 2026)

The headline release: ship the technical artifacts that make the "predictive turn-cache" positioning provable and unique.

### Track A — Predictor refactor & Successor Representation

1. **`NextTurnPredictor` trait** *(week 1, foundational)* — pull `MarkovPredictor` behind a trait so SR can slot in without an API break. Trait surface: `predict(ctx) -> Vec<(EventId, f32)>`, `observe(prev, next)`, `confidence() -> f32`.
2. **`primd-sr` crate** *(weeks 2–7, headline)* — Successor Representation predictor:
   - TD(0) update: `M[s_t, :] += η · (φ(s_{t+1}) + γ · M[s_{t+1}, :] − M[s_t, :])`
   - Low-rank reduction: `W: 256×K, M_low: K×K` with K=32 (matches a cache line × 4 SIMD lanes of f64)
   - Confidence signal: spectral gap of `M_low`
   - Hybrid wrapper: SR + Markov fallback gated on `sr.confidence() > τ` (default τ = 0.05)
   - Cold-start: pre-trained generic SR on synthetic FAQ + tool-use corpus, ~50 KB embedded
   - Online: TD(0) with η_t = 0.1 / (1 + 0.01 · t)
   - Persistence: session-local `.primd-sr` artifact for cross-session warm-start
3. **A/B harness** *(week 8)* — 1000-conversation eval corpus, measure speculative-cache hit rate of Hybrid vs pure Markov. Target: ≥ 15 % absolute lift on conversations with ≥ 5 turns.

References: Dayan 1993, Stachenfeld et al. 2017 (Nat Neurosci 20:1643–1653), Russek et al. 2017 (Nat Hum Behav 1:680–692), Gershman 2018 (J Neurosci 38:7193).

### Track B — Real per-event HNSW shards

The current event-scoped path is a SIMD gather + subset rescan, not HNSW. Works well up to ~1 M docs with small union scopes; degrades past that. v0.2 adds an actual per-event HNSW graph using `instant-distance` or `hnsw_rs`. The trait surface in `shards.rs` already isolates the rescan step, so the change is localized.

### Track C — MoshiRAG back-end adapter

OpenAI-compatible `/v1/chat/completions` endpoint on `primd serve` so MoshiRAG (and any other tool expecting an OpenAI-compatible retrieval back-end) can swap its 1–3 s vLLM call for primd's sub-200 µs response with one env-var change. Returns retrieved context as the completion string; the model in the loop still generates.

### Track D — Public benchmark vs Moss + Qdrant + Pinecone

Reproducible benchmark at the *finalize* event (not at `/query`), on a 100 k-doc voice corpus. Publish p50/p95/p99 latency and embed a 30-second GIF showing the speculative-retrieval timeline. Submit to Hacker News.

### Track E — LiveKit plugin

`livekit-primd` packaged plugin exposing `PrimdRetrievalAgent`. Submit PR to LiveKit's plugin registry.

### v0.2 exit criteria

- [ ] `NextTurnPredictor` trait shipped, Markov refactored behind it
- [ ] Hybrid SR + Markov predictor: ≥ 15 % absolute speculative-cache hit-rate lift over Markov-only on ≥ 5-turn conversations
- [ ] Per-event HNSW shards: p99 ≤ 2× v0.1 numbers at 1 M docs
- [ ] MoshiRAG adapter works as a drop-in for MoshiRAG's reference back-end
- [ ] Public benchmark vs Moss + Qdrant + Pinecone published; HN front-page attempt
- [ ] `livekit-primd` on PyPI

## v0.3 — Long-arc bets (Q4 2026 / Q1 2027)

### Track A — Hippocampus Signature DWM cold tier

First-mover Rust port of arXiv:2602.13594 (Yi Li, Cao, Ahmed, Sharma, Bingzhe Li — HPE Labs + UT Dallas, Feb 2026). **Time-sensitive:** the paper is anonymized for MLSys, no public code release as of May 2026; if MLSys accepts in June–August 2026 the authors will likely release code shortly after.

Adopt:
- Signature DWM as cold-tier evicted store (`primd::cold::SignatureDWM`)
- Random Indexing as `SignatureSource::RandomIndexing` (zero-LLM-token signature construction)
- Compressed-domain rank/select primitives for analytics queries

Reject:
- Replacing event-scoped HNSW on the hot path (DWM is O(n); HNSW shards are O(log k_event))
- Content DWM (forces tokenization, breaks model-agnosticism)

### Track B — Public LoCoMo / LongMemEval numbers

Run primd + cold tier on the standard long-horizon agent-memory benchmarks. Goal: prove primd doesn't regress on long-horizon recall when the cold tier is engaged. Not to win on memory — to credentialize that primd scales from a 1-minute voice turn to multi-day session memory in a single binary.

### Track C — WASM / browser target

`primd-core` compiled to WASM for in-page voice agents. Target: 10 k-doc corpus at < 10 ms p50 in the browser.

### Track D — Trust primitives

Confidence scores, dataset freshness, refusal-on-uncertainty. The SR predictor's spectral gap is the first concrete confidence signal; v0.3 extends this to per-result confidence based on coarse-scan margin and event-scope coherence.

### Track E — Streaming-query mode (paradigm-independent observe loop)

Refactor `observe_partial` so it works without discrete STT partial frames — a continuous-input `observe` that can be driven by a full-duplex model's audio embedding stream as well as Pipecat-style transcript frames. Hedges against the disappearance of STT/TTS phases.

## v0.5+ — Post-pipeline-phase world (2027+)

### Model-phase hooks for full-duplex models

If full-duplex models (TML-Interaction-Small, future Moshi, Gemini Live 3) commoditize discrete STT/TTS by EOY 2026, evolve primd's hooks toward model-phase signals:

- **Generation pauses** — analog of the TTS-playback hook; pre-warm during the model's brief silences
- **Background-agent slow-think windows** — TML's stated "async background model" surface is where retrieval / tool-use happens. Build a formal `BackgroundAgent` protocol adapter.

### Online learning for the predictor

The v0.2 SR is per-session online but doesn't cross sessions without explicit `user_id`. v0.5 adds cross-session SR updates with differential-privacy noise, plus per-domain pretrained SR matrices (sales, support, healthcare intake, dispatch).

### Hardware acceleration

- CUDA kernel for binary signature scan at > 10 M scale (target: < 0.1 ms for 10 M sigs on A100)
- Apple Neural Engine path for M-series
- Intel AMX for the rescore step

### Hosted primd Cloud

Managed deployment with per-domain pretrained predictors, SLA-backed p99, usage-based pricing.

## Key dependencies

```
v0.2 NextTurnPredictor trait
       │
       ├── primd-sr crate ───── A/B harness
       │
       ├── HNSW shards ──────── benchmark vs Moss/Qdrant/Pinecone
       │
       └── MoshiRAG adapter ─── pip + Pipecat + LiveKit integrations
                                       │
v0.3 Hippocampus DWM cold tier ────────┴── LoCoMo / LongMemEval

v0.5 model-phase hooks ←── reuses SR over event signatures
                          (paradigm-independent)
```

The SR work and the per-event HNSW work can proceed in parallel after the `NextTurnPredictor` trait lands. The MoshiRAG adapter is independent of both.

## Risk mitigations

| Risk | Mitigation |
|---|---|
| SR underperforms Markov on real corpora | Hybrid wrapper falls back to Markov when SR confidence is low — ship the engineering artifact regardless of A/B outcome |
| Hippocampus authors release code before our port | Race their release (4–6 mo from Feb 2026 → June–Aug 2026). If we miss it, the Rust port still has value as the open-source reference; reframe as compatibility, not first-mover |
| Per-event HNSW shards regress p50 at small scopes | Keep the v0.1 subset-rescan as fallback when union scope < 1 k docs |
| Full-duplex models eat STT/TTS before v0.5 | The SR + DWM substrate is useful even without STT/TTS hooks; pivot priority is model-phase hooks, not a positioning change |
| Moss raises Series A > $40 M before our v0.2 ships | License moat (Apache-2.0 vs PolyForm Shield) still holds; double-down on Pipecat / LiveKit upstream contributions |

## Decision thresholds

See [positioning.md § Decision thresholds](../business/positioning.md). The two that would materially change the roadmap:

- **CloakPipe lands $1 M+ angel before primd hits 500 GitHub stars** → redirect primary effort to CloakPipe; keep primd as RustConf-talk-grade credibility artifact rather than a fundable product. v0.2 SR work continues part-time; v0.3 long-arc bets are deferred.
- **A full-duplex model with native retrieval ships GA before EOY 2026** → fast-forward v0.5 model-phase hooks into v0.3; defer Hippocampus DWM until the new model surface stabilizes.
