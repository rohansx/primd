# Strategy memo — May 2026

This is the strategic decision record for primd's v0.2 cycle. It captures *why* the positioning is what it is, *why* the v0.2 milestones look the way they do, and *which signals would change the plan*.

Three external inputs drive this memo:

1. The Hippocampus paper (arXiv:2602.13594, Feb 14 2026, HPE Labs + UT Dallas) — a new long-horizon agent-memory primitive.
2. Moss / InferEdge Inc. (YC F25) — the only company shipping retrieval-shaped infrastructure into the same Pipecat / LiveKit slot.
3. Kyutai MoshiRAG (April 30 2026) + Thinking Machines TML-Interaction-Small (announced May 11 2026) — frontier-lab signal that the *retrieval back-end* slot is recognized as separate from the model.

## TL;DR

- **Hippocampus is complementary, not competitive.** Long-horizon agent memory ≠ real-time turn cache. Port the Signature DWM as a *cold tier* in v0.3; keep event-scoped scanning on the hot path.
- **Stay voice-focused.** Don't pivot to agent memory (Mem0 / Letta own the category; the accuracy moat there is fragile). Sharpen the wedge to *"predictive turn-cache for real-time conversational AI."*
- **Replace variable-order Markov with Successor Representation** in v0.2's headline feature. Hybrid SR + Markov fallback keeps the old behavior available when SR confidence is low.

## 1. Hippocampus vs primd — complementary, with one overlap

Authors of arXiv:2602.13594 inferred from arXiv metadata: Yi Li, Lianjie Cao, Faraz Ahmed, Puneet Sharma (Hewlett Packard Labs) + Bingzhe Li (UT Dallas). Hardware: HPE DL380a Gen11, 4× H100, 1 TB DRAM. No public code release as of May 2026; the paper is anonymized for MLSys double-blind review.

### Axis-by-axis

| Axis | Hippocampus (DWM) | primd (binary signatures + event scope) |
|---|---|---|
| **Core data structure** | Dynamic Wavelet Matrix: ℓ co-indexed bit-vectors with constant-time rank/select. Content DWM (token-ID stream, lossless) + Signature DWM (binary, semantic). | 256-bit signature array + event catalog with per-event doc indices. SIMD Hamming scan; subset rescan over candidate-event union. v0.2 adds per-event HNSW shards. |
| **Indexing** | Random Indexing: sparse base vectors → sliding-context sum → sign-LSH signature. **Zero LLM tokens at construction.** Hippocampus builds memory in 6.70 min vs MemGPT 59.49 min. | Embedding pipeline (hashed / OpenAI / fastembed) → sign-LSH → push to corpus. Embedding cost is a real disadvantage vs random indexing. |
| **Query** | Compressed-domain Hamming-ball XOR + POPCOUNT, **O(n·d/w) — linear in n**. | SIMD coarse scan + gather-and-rescan over event union. Speculative variant fires during STT partial → 1.6 µs at finalize when speculation matches. |
| **Update** | O(log σ) append per symbol. **No embedding model call.** | Embedding model + sign-LSH + insertion. |
| **Footprint** | nℓ + o(nℓ) bits = O(n log σ). ~18 MB for 10 M tokens. | 32 bytes/sig + v0.2 HNSW overhead. Per-event shards keep working set in L2/L3. |
| **Latency** | ~1.08 s end-to-end on LoCoMo vs MemGPT ~33.6 s (**31× end-to-end including LLM judging**). | 1.6 µs cache hit, ~100–250 µs full scan at 100 k (**98× at finalize**). Different layer, not directly comparable. |
| **Accuracy** | LoCoMo F1: 34.36 / 31.97 / 38.30 / 48.38; beats MemOS on 3 of 4. | No published LoCoMo / LongMemEval; measurement gap to close in v0.2. |
| **Workload** | Long-horizon agent memory; write-once read-many; multi-session sessions. | Real-time voice turn; every-turn write+read; minutes-to-hours sessions. |
| **Pipeline-phase awareness** | None. | Core IP: speculation during STT, prewarming during TTS, delta cache for repeats. |

### Adopt / reject / extend

**Adopt (v0.3):**

1. **Signature DWM as cold tier** (`primd::cold::SignatureDWM`). Evicted event shards serialize into an append-only DWM keyed by `session_id`. Wins for long-session apps (customer support, therapy, tutoring) primd currently can't serve well.
2. **Random Indexing as no-GPU signature source.** `SignatureSource::RandomIndexing` alongside `SignatureSource::Embedding(model_id)`. ~1500 LoC using `bitvec` + custom rank/select.
3. **Compressed-domain rank/select primitives.** Exposed for analytics queries (`wm_rank`, `wm_select`).

**Reject:**

1. **Replacing event-scoped retrieval with DWM on the hot path.** DWM is O(n) in corpus; primd's per-event scope is O(k_event) where k_event ≪ n. Regresses p99.
2. **LoCoMo as primd's North Star.** Hippocampus compares against MemGPT / A-Mem / MemOS — none voice-optimized. Recreating those wins doesn't move primd's actual users.
3. **Token-ID Content DWM.** primd stores arbitrary documents, not LLM token streams. Forces tokenization; breaks model-agnosticism.

**Extend (primd's unique IP — double down):**

1. **Pipeline-phase awareness.** Push further: `observe_vad_silence` (start warming on 200 ms of VAD silence, before STT commits), `observe_interruption` (invalidate delta cache on user barge-in). Voice-specific primitives no agent-memory system can copy.
2. **Event-scoped sharding aligned with conversation structure.** Hippocampus has no event concept. Per-event shards become per-event *predictive contexts*; SR operates over event signatures naturally.
3. **Sub-µs SIMD Hamming kernel.** AVX-512 VPOPCNTDQ path is already shipped (`scan_avx512` in `primd-core/src/index/signatures.rs:381-418`). Remaining work: benchmark on Sapphire Rapids / Zen 4. Target: 0.5 µs cache hit.

### First-mover opportunity

The Hippocampus paper has **no public code as of May 2026**. If MLSys accepts in June–August 2026, authors will likely release shortly after. A clean Rust port of the Signature DWM kernel shipped before then makes primd the de facto open-source reference. **v0.3 deliverable; time-sensitive.**

## 2. Positioning decision — stay voice, sharpen the wedge

### Why option A (stay voice + sharpen) wins over B/C/D

- **B "brain-inspired retrieval — voice today, agent memory tomorrow"** is the trap. It signals "we don't know what we are." Worst of both worlds.
- **C "pivot to agent memory"** abandons primd's actual edge. STT-partial speculation and TTS-phase pre-warming don't carry over to async agent flows. Walks into Mem0's $24 M + Letta's Berkeley pedigree with no architectural advantage.
- **D "reposition as something else"** is what you do when A/B/C are bad. A is good.

### Why A wins

1. **Voice AI is the only category where primd's IP is structurally unique.** Moss has lower raw latency on closed-source semantic search but no pipeline-phase scheduling. Mem0 / Letta / Zep have richer memory models but zero voice-pipeline awareness. primd sits in the only empty quadrant: *real-time voice + predictive scheduling*.
2. **Market is large enough.** AI voice agents at $2.4 B → $47.5 B by 2034 is a $4 B mid-2027 TAM. 0.5 % capture = $20 M ARR.
3. **Apache-2.0 vs Moss's PolyForm Shield is a real, legal wedge.** Pipecat, LiveKit, Vapi, and Cartesia all integrate Apache-licensed components without legal review. Moss's PolyForm explicitly forbids "production or competing commercial use." That's how Pipecat itself beat closed alternatives.
4. **Founder fit.** Solo Rust developer in India with ~3 yr Rust experience can't out-distribute Mem0 (YC, AWS partnership). Can ship one technically obsessive open-source primitive that becomes a Pipecat dependency. Same playbook tokio, sqlx, qdrant, surrealdb executed with 1–2 person founding teams.
5. **CloakPipe diversification de-risks.** primd doesn't need to be everything. Deep technical project + RustConf talk + O-1A artifact is its own optimization function.

### The exact positioning sentence

> **"primd is the predictive turn-cache for real-time conversational AI: a 10 MB Apache-2.0 Rust runtime that hides retrieval latency inside the STT and TTS phases of your voice agent."**

This names the *unit of work* (a turn), the *competitor it's not* (Moss's general semantic search), and the *mechanism* (hiding inside STT/TTS phases). It's what a Pipecat developer can repeat to their VP Eng.

### What we explicitly will NOT do

- Compete with vector DBs on storage or scale
- Pivot to agent memory (Mem0 + Letta own that)
- Market to general RAG users
- Compete on recall (97 %+ is good but not differentiated)
- Build a platform
- Ship closed-source modules

## 3. Successor Representation — replacing variable-order Markov

### Why SR

The current `MarkovPredictor` (variable-order with half-life decay, smoothing, multi-order backoff in `primd-core/src/predict/markov.rs`) has three pathologies in primd's setting:

1. **No generalization across paraphrases.** "What's my balance" and "Tell me my balance" produce different 256-bit signatures. Markov treats them as unrelated states. SR's predictive map smooths over states with similar successors. Dayan (1993), Stachenfeld, Botvinick & Gershman (Nat Neurosci 20:1643–1653, 2017).
2. **Hard horizon.** Variable-order looks back k turns; cannot represent "user usually checks order status three turns after greeting" without sparse 3rd-order chains. SR's discount γ ≈ 0.85–0.95 encodes a *soft* horizon for typical 5–15-turn voice flows.
3. **No calibrated confidence for `warm_next`.** Markov gives `P(next | ctx)` but no "how confident should I be about prewarming?" SR exposes a continuous score plus eigenstructure (spectral gap of `M_low`).

### Math (adapted to primd's event graph)

Each event has signature `s_e ∈ {0,1}^256`. SR is:

```
M^π(s, s') = E_π[ Σ_{t=0}^∞ γ^t · 1[s_t = s'] | s_0 = s ]
M^π = (I − γT^π)^{-1}   (Stachenfeld et al. 2017 eq. 7 corrected, Nat Neurosci 21:895)
```

The signature space is 2^256, so we reduce via:

- **Eigenvector basis (k=32).** Top-32 eigenvectors of T^π; project signatures into ℝ^32. Eigenvectors of T^π and M coincide. k=32 = one cache line × 4 SIMD lanes of f64.
- **Linear function approximation** (Russek 2017). Weights `W: 256×K` so `ψ(s) = W^T s_bits`; `M(s, s') ≈ ψ(s) · M_low · ψ(s')^T`.

### Training options ranked by ship-readiness

1. **TD(0) over signature embeddings (ship in v0.2):**
   ```
   M[s_t, :] += η · (φ(s_{t+1}) + γ · M[s_{t+1}, :] − M[s_t, :])
   ```
   Low-rank form: `W += η · (φ(s_{t+1}) + γ·ψ(s_{t+1}) − ψ(s_t)) ⊗ s_t_bits`.
   Cost: O(256 × 32) ≈ 8 k FMAs per turn = 1–5 µs with SIMD. Trivial vs embedding forward pass.
2. **Eigenvector offline trainer (v0.3):** k-medoids cluster signatures, build empirical T, `faer::partial_eigh` for top-k eigenvectors.
3. **Neural SR (research roadmap):** small MLP (256 → 64 → 32) trained end-to-end. Defer until evidence linear TD plateaus.

### Crate structure

```
primd-sr/         # SR types, TD updater, eigenvector trainer
primd-sr-train/   # Offline trainer binary using faer
primd-core/predict/  # NextTurnPredictor trait + Markov + SR impls + Hybrid wrapper
```

Trait surface:

```rust
pub trait NextTurnPredictor: Send + Sync {
    fn predict(&self, ctx: &EventContext) -> Vec<(Signature, f32)>;
    fn update(&mut self, ctx: &EventContext, observed: &Signature);
    fn confidence(&self) -> f32; // NEW: SR exposes spectral gap; Markov returns 1.0
}
```

### A/B design

1000-conversation corpus, two arms:
- **A (control):** `warm_next` uses Markov
- **B (treatment):** `warm_next` uses SR (or Hybrid)

**Primary metric:** speculative-cache hit-rate = fraction of `finalize()` calls served from cache. **Hypothesis:** Arm B beats Arm A by ≥ 15 % absolute on conversations ≥ 5 turns. Power calc: n=1000 gives 95 % confidence at effect size 0.05.

**Secondary:** p99 latency at finalize, false-warm rate.

### Risks

| Risk | Mitigation |
|---|---|
| Non-stationary policy (chitchat → escalation) | Per-intent-cluster SR; context-reset signals from Pipecat |
| Eigenvector instability < 30 unique signatures | Cold-start generic SR + 50-turn warm-up gate |
| Memory blow-up if K grows | Cap at K ≤ 64; effective dim bounded by policy mixing time |
| Catastrophic forgetting across sessions | Per-user SR persistence when `user_id` provided |
| Adversarial inputs (deliberate topic noise) | Hybrid predictor's SR-confidence gate falls back to Markov |
| SR underperforms Markov on real corpora | Ship engineering artifact regardless — SR + Hybrid fallback gives narrative even on a null result |

### Implementation timeline (~8 weeks part-time)

- Week 1–2: `primd-sr` core with TD(0), ndarray-backed W and M_low, scalar
- Week 3: Offline trainer using `faer::partial_eigh`; synthetic ~10 k dialogue corpus
- Week 4: Refactor `MarkovPredictor` behind `NextTurnPredictor` trait; Hybrid wrapper
- Week 5: SIMD optimization (portable SIMD f32x8)
- Week 6: A/B infrastructure + 1000-conversation eval; publish results
- Week 7: Documentation, blog post draft, arXiv preprint
- Week 8: v0.2 release + Pipecat reference update + HN submission

### Papers to cite

1. Dayan 1993 — original SR formulation (Neural Computation 5:613–624)
2. Stachenfeld, Botvinick, Gershman 2017 — hippocampus as predictive map (Nat Neurosci 20:1643–1653, correction at 21:895)
3. Russek, Momennejad, Botvinick, Gershman, Daw 2017 — SR in human RL, TD(0) update (Nat Hum Behav 1:680–692)
4. Gershman 2018 — SR computational logic and neural substrates (J Neurosci 38:7193–7200)
5. Sherstan, Machado, Pilarski 2018 — accelerating learning with SR (arXiv:1803.09001)
6. Geerts, Stachenfeld, Burgess 2022 — STDP-style SR (eLife / bioRxiv 2022.04.20.488882)
7. Mahadevan & Maggioni 2007 — proto-value functions / Laplacian basis (JMLR 8:2169)
8. Hierarchical SR (arXiv:2602.12753, Feb 2026) — NMF for sparse SR bases
9. Machado, Bellemare, Bowling 2017 — deep SR option discovery (arXiv:1710.11089)

## 4. Decision thresholds that would change this strategy

| Signal | Impact | Response |
|---|---|---|
| > 20 paying voice-AI customers by day 90 | Validates option A | Raise seed on the turn-cache thesis |
| Moss raises Series A > $40 M before day 90 OR signs Pipecat / LiveKit exclusive | License moat shift | Stay Apache-2.0, double-down on Pipecat/LiveKit upstream contributions |
| Full-duplex model with native retrieval ships GA before EOY 2026 | STT/TTS hooks erode | Fast-forward v0.5 model-phase hooks to v0.3 |
| CloakPipe lands $1 M+ angel before primd hits 500 stars | Effort allocation | Redirect primary effort to CloakPipe; primd becomes credibility artifact |
| Hippocampus authors release reference code before our port | Lose first-mover narrative | Reframe as compatibility, not first-mover; the Rust port still has reference value |

## 5. Open caveats

- **Hippocampus paper is anonymized** — author affiliations (HPE Labs + UT Dallas) are inferred from arXiv metadata, dblp, and personal sites. If the paper is rejected at MLSys, an updated version may appear with different authors or results.
- **Hippocampus "31× end-to-end on LoCoMo" and primd's "98× at finalize" measure different layers against different baselines.** Honest framing: Hippocampus optimizes the *write side* of agent memory (token-free construction, sub-linear retrieval at scale); primd optimizes the *wait side* of voice turns (hide retrieval inside STT/TTS phases).
- **Moss's "250 k+ installs"** is the npm package download count, not enterprise installs.
- **TML-Interaction-Small** is 18–24 month horizon, not 6-month. The SR + DWM substrate hedges this without requiring a positioning change.
- All funding, customer, and benchmark numbers are as-of May 2026; voice-AI and agent-memory categories move fast.

## 6. v0.2 milestone summary

See [roadmap.md](../plan/roadmap.md) for the full track-by-track plan. Headline deliverables:

- `NextTurnPredictor` trait (foundation, week 1)
- `primd-sr` crate with Hybrid SR + Markov (weeks 2–8)
- Real per-event HNSW shards
- MoshiRAG `/v1/chat/completions` adapter
- Public benchmark vs Moss + Qdrant + Pinecone at finalize event
- `livekit-primd` packaged plugin

Exit criterion the entire memo collapses to: **a Hybrid SR + Markov predictor lifting speculative-cache hit rate by ≥ 15 % absolute on multi-turn voice flows, published in a reproducible benchmark, with the MoshiRAG adapter shipped so primd is a drop-in retrieval back-end for the only open-source full-duplex voice model in production.**
