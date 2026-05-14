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

## v0.2 — Turn-cache moat (Q3 2026, in progress)

The headline release: ship the technical artifacts that make the "predictive turn-cache" positioning provable and unique.

### Track A — Predictor refactor & Successor Representation

1. ✅ **`NextTurnPredictor` trait** (2026-05-13) — `MarkovPredictor` lives behind the trait, `QueryContext` holds `Box<dyn NextTurnPredictor>`.
2. ✅ **`primd-sr` crate, tabular variant** (2026-05-13) — Successor Representation predictor:
   - TD(0) update: `M[s_t, :] += η · (e_{s_t} + γ · M[s_{t+1}, :] − M[s_t, :])`
   - Self-visit initialization `M[s, s] = 1.0` on first sight so bootstrap is correct from the first observation
   - Sparse `HashMap<EventId, HashMap<EventId, f32>>` storage; correct for tens-to-hundreds of events
   - Confidence: linear warmth `min(1, t / warmup)`
   - Hybrid wrapper: SR + Markov; threshold 0.5 by default
   - Online TD(0) with η_t = η_base / (1 + 0.01 · t)
   - JSON persistence (`save_to_file` / `load_from_file`)
   - CLI flag: `primd serve --predictor {markov,sr,hybrid}` (default `markov` for v0.2 stability)
3. ✅ **Low-rank reduction** (2026-05-14) — `W: 256×32` random projection over signature bits; `M_low: 32×32` updated by TD(0). New `LowRankSrPredictor` in `primd-sr/src/low_rank.rs`. Identity initialization for correct bootstrap. 7 unit tests; integrated into the A/B bench.
4. ✅ **A/B harness** (2026-05-14) — `predictor_ab` bench measures Markov / SR-tabular / low-rank-SR / Hybrid side-by-side at 1000 utterances with windowed cumulative hit-rates. Validates Hybrid robustness (0 pp regression vs Markov).
5. ✅ **Paraphrase-aware adversarial workload** (2026-05-14) — `paraphrase_ab` bench, 10 topics × 10 paraphrases, with top-1 / top-K topic-correctness metrics. **Negative result:** low-rank SR underperforms Markov by 58.8 pp on top-1 topic correctness. The K=32 random projection over-pools and the M_low=I initialization biases toward current-feature-aligned (within-topic) prediction. See [bench-report.md § paraphrase_ab](../benchmarks/bench-report.md#paraphrase_ab-results) for the full analysis.
6. ✅ **Per-user predictor persistence** (2026-05-14) — `primd serve --sr-state-dir <path>` writes each session's Markov state on reset and warm-starts from disk on session create. New `NextTurnPredictor::as_markov()` trait method (default `None`; MarkovPredictor + HybridPredictor override). Session-id sanitization neutralizes path-traversal attempts. Works for `--predictor markov` and `--predictor hybrid`.
7. ✅ **LowRankSrPredictor save_to_file / load_from_file** (2026-05-14) — closes the v0.2.7 API gap: `LowRankSrPredictor` now has the same persistence interface as `SrPredictor` and `MarkovPredictor`. Serializes `M_low` + counters + config (~4 KB at K=32); projection matrix and event-centroid features are corpus-dependent and rebuilt from matching `event_centroids` at load. K-mismatch on load returns `InvalidData`. Auto-wiring into `cmd_serve` for SR/Hybrid sessions waits for stable Rust trait upcasting; manual `save_to_file` calls work today.

### v0.2.6 — iterate low-rank SR against the paraphrase bench

The paraphrase A/B surfaced a real architecture issue with v0.2.5's low-rank SR. Tracking progress:

- ✅ **K-sweep refactor** (2026-05-14) — `LowRankSrPredictor` is now generic over `const K: usize`. Type alias `LowRankSr = LowRankSrPredictor<32>` preserves the previous API.
- ✅ **K=32 vs K=64 vs K=128 on paraphrase_ab** (2026-05-14) — K=64 best on random projection (top-1 25–45 % with high variance due to HashMap iteration). K=128 *regresses* to 10.1 % due to overparameterization. Conclusion: **K=64 is the right default for typical voice corpora**; K=128 needs more training data.
- ✅ **PCA projection over corpus signatures** (2026-05-14) — implemented and benchmarked; regresses to chance-level (10.1 %) on the paraphrase workload due to a feature-magnitude mismatch with the `M_low = I` initialization. Diagnosed but not fixed; **needs v0.2.7 feature-normalization work**. The `pca` module itself is correct (5 unit tests verify eigenvector alignment).
- ✅ **Hybrid wrapper validates the SR thesis** (2026-05-14) — Hybrid (tabular SR + Markov, threshold 0.5) **beats Markov alone by +4–12 pp top-1 topic correctness** on `paraphrase_ab` across multiple runs. This is the v0.2.6 success story — the wrapper's confidence-gated ensemble exploits both predictors' complementary signals.
- ✅ **Spectral-gap confidence** (2026-05-14) — `LowRankSrPredictor::confidence` now blends warmth with the spectral gap of `M_low` (power iteration with deflation, refreshed every 25 observations). Replaces the v0.2.5 warmth-only proxy. New 23rd test exercises the blended semantics.
- ✅ **Sorted iteration everywhere** (2026-05-14) — switched `LowRankSrPredictor::event_features`, `MarkovPredictor::vocab`, and `SrPredictor::{m, vocab}` from `HashMap`/`HashSet` to `BTreeMap`/`BTreeSet`. **Two-run determinism verified**: paraphrase_ab produces identical top-1 numbers across consecutive bench runs for all 8 predictors. The previous ±5–20 pp variance is gone.
- ✅ **Ruled out: `M_low = 0` init** (2026-05-14) — breaks the SR bootstrap math. Identity is the SR-correct default.
- ❌ **Multi-step structured workload** — when the right prediction at step *t* depends on horizons longer than 1 step. SR's discount γ captures this; Markov-k needs k as a hyperparameter and sparsifies.
- ❌ **PCA feature normalization (v0.2.7)** — normalize PCA-projected features to unit norm so the `M_low = I` bootstrap term carries comparable magnitude to the random-projection variant. Until this lands, ship Hybrid with tabular SR, not low-rank SR.

**v0.2.6 ship decision:** Hybrid SR + Markov is the production-default predictor. Deterministic bench shows Hybrid beats Markov by **+13.3 pp top-1 topic correctness** on the paraphrase workload (reproducible across runs). Tabular SR alone beats Markov by **+9.9 pp**; the Hybrid wrapper compounds on top of that. Low-rank SR remains opt-in for research and post-v0.2.7 when PCA + feature normalization closes the regression.

References: Dayan 1993, Stachenfeld et al. 2017 (Nat Neurosci 20:1643–1653), Russek et al. 2017 (Nat Hum Behav 1:680–692), Gershman 2018 (J Neurosci 38:7193). See [successor-representation.md](../architecture/successor-representation.md).

### Track B — Real per-event HNSW shards

✅ Shipped 2026-05-14. New `primd-core::index::hnsw::EventHnswCache` builds per-event HNSW shards via `instant-distance` lazily on first query. `HierarchicalIndex::with_hnsw()` opts in; `search()` routes to HNSW when the union scope ≥ 1024 docs, else falls through to the v0.2 SIMD subset rescan. Gated behind the `hnsw` feature flag (on by default). 5 new unit tests cover build correctness, threshold gating, unknown-event fallback, and shard caching across queries. v0.3.1 will add on-disk persistence so shards survive `primd serve` restarts.

### Track C — MoshiRAG back-end adapter

✅ Shipped 2026-05-13. OpenAI-compatible `/v1/chat/completions` endpoint on `primd serve` lets MoshiRAG (and any other tool expecting an OpenAI-compatible retrieval back-end) swap its 1–3 s vLLM call for primd's sub-200 µs response with one env-var change. Returns retrieved context as the completion string; the model in the loop still generates. See [docs/integrations/moshirag.md](../integrations/moshirag.md).

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

✅ Shipped 2026-05-14. New `primd-wasm` crate (workspace member) wraps `primd-core` with wasm-bindgen for browser deployment. Builds against `wasm32-unknown-unknown` with `default-features = false` on primd-core (fastembed/openai/HNSW disabled in WASM — see [integration doc](../integrations/wasm.md) for what's available). JS-facing API: `PrimdIndex(signatures_flat, event_names, event_scopes)` constructor + `query` / `observe_partial` / `finalize` / `warm_next` methods. Output bundle is ~250 KB compressed via wasm-bindgen.

### Track D — Trust primitives

Confidence scores, dataset freshness, refusal-on-uncertainty. The SR predictor's spectral gap is the first concrete confidence signal; v0.3 extends this to per-result confidence based on coarse-scan margin and event-scope coherence.

### Track E — Streaming-query mode (paradigm-independent observe loop)

Refactor `observe_partial` so it works without discrete STT partial frames — a continuous-input `observe` that can be driven by a full-duplex model's audio embedding stream as well as Pipecat-style transcript frames. Hedges against the disappearance of STT/TTS phases.

### Track F — Hippocampus DWM cold tier

✅ **Foundation shipped 2026-05-14** — new `primd-dwm` workspace crate with:
- `BitVector` with O(1) `rank_1` / `select_1` primitives via two-level lookup tables (Jacobson 1989 + Clark 1996 succinct data-structures construction). ~6.25 % auxiliary overhead. Serializable for cold-tier persistence.
- `RandomIndexer` for zero-LLM-token signature construction per the Hippocampus paper (Appendix B). Sparse ternary base vectors + sliding-window aggregation + top-K sign-quantization → 256-bit signatures compatible with primd-core's existing pipeline.

**v0.4 work to complete the cold tier:**
- `SignatureDWM` — wavelet-matrix-backed cold-tier store layering many `BitVector`s. Append-only; supports compressed-domain Hamming-ball queries via XOR + popcount over the layered structure.
- Integration with `QueryContext` — events evicted from hot tier (HNSW shards) move into the DWM-backed store keyed by `session_id`. Cross-session memory for multi-day voice agents.
- Public LoCoMo / LongMemEval bench to credentialize the long-horizon recall story without competing with Hippocampus on its home turf.

First-mover status: the paper (arXiv:2602.13594) had no public reference code as of May 2026. The foundation we shipped today is the first open-source port of any portion of the algorithm. Strategy memo flagged this as time-sensitive; if MLSys accepts and the authors release reference code in June–August 2026, our framing pivots from "first-mover" to "Rust-idiomatic implementation" but the work still has reference value.

## Open items blocked on external dependencies

- **Full HNSW-graph persistence** — blocked on upstream `instant-distance` v0.6.x serde feature fix (currently references `BigArray` without declaring the dep). The v0.3 ship works around with eager-rebuild-from-warm-EventIds, but a real graph serialization is desirable when the upstream lands. Re-evaluate every release.
- **`cmd_serve` auto-wiring for SR / LowRank persistence** — blocked on stable Rust trait upcasting (currently nightly-only via `trait_upcasting` feature). Once stable, `Box<dyn NextTurnPredictor>` can `as_any()` into the concrete predictor for save/load. Manual `SrPredictor::save_to_file` and `LowRankSrPredictor::save_to_file` work today as opt-in APIs.
- **Multi-threaded WASM** — Atomics + SharedArrayBuffer require COOP/COEP headers that most demos can't serve. Worth doing once we have a paying customer who needs it.

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
