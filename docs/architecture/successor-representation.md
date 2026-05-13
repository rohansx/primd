# Successor Representation predictor (`primd-sr`)

> v0.2 headline feature. Lives in [`primd-sr/src/lib.rs`](../../primd-sr/src/lib.rs).
> Strategy rationale in [strategy-2026-05.md](../business/strategy-2026-05.md).

primd's variable-order Markov chain (`primd-core/src/predict/markov.rs`) is sufficient as a baseline next-turn predictor, but it has three pathologies that limit cache hit rate on real voice flows:

1. **No generalization across paraphrases.** "What's my balance" and "Tell me my balance" produce different signatures; Markov treats them as unrelated states.
2. **Hard horizon.** Variable-order Markov looks back k turns; it cannot represent "user usually checks order status three turns after greeting" without an explicit 3rd-order chain (which sparsifies catastrophically).
3. **No calibrated confidence.** Markov returns `P(next | ctx)` but doesn't expose "how confident should I be about pre-warming?"

`primd-sr` replaces (or wraps) Markov with a **Successor Representation** predictor that addresses all three.

## The math

For events `s ∈ S` observed in conversation, the SR is the expected discounted future visit count:

```
M^π(s, s') = E_π[ Σ_{t=0}^∞ γ^t · 1[s_t = s'] | s_0 = s ]
```

Closed form: `M^π = (I − γT^π)^{-1}` (Stachenfeld et al. 2017 eq. 7 corrected; *Nat Neurosci* 21:895). γ is a discount in `(0, 1)` that controls the prediction horizon: γ ≈ 0.9 sees ~10 turns into the future; γ ≈ 0.5 sees ~2 turns.

### TD(0) update

On observed transition `s_t → s_{t+1}`, push `M[s_t, :]` toward `e_{s_t} + γ · M[s_{t+1}, :]`:

```
δ(s) = 1[s = s_t] + γ · M[s_{t+1}, s] − M[s_t, s]
M[s_t, s] ← M[s_t, s] + η · δ(s)   for all s in vocab
```

`e_{s_t}` is the one-hot indicator for `s_t` — the **t=0 self-visit term**. To make the bootstrap correct from the first observation, `primd-sr` initializes `M[s, s] = 1.0` on first sight of any event. Without this initialization, the analytical SR for a chain `A → B → C` would not match the TD-converged value.

### Why this beats Markov

| Property | Markov | SR |
|---|---|---|
| Horizon | k turns (fixed-order or backoff) | soft, controlled by γ |
| Generalization | none | yes — eigenstructure of M captures predictive similarity |
| Confidence signal | none (returns uniform when context unseen) | spectral gap of M_low (v0.2.5); warmth proxy (v0.2) |
| Memory per session | O(\|context\|^k) | O(N_events²) tabular; O(K²) low-rank |
| Update cost | ~200 ns | ~3 µs (with SIMD, ~µs without) |

The strategy memo's hypothesis: **≥ 15 % absolute speculative-cache hit rate lift on conversations with ≥ 5 turns.** A/B harness is on the roadmap for v0.2 close.

## What ships in v0.2

```rust
use primd_sr::{SrPredictor, HybridPredictor};
use primd_core::predict::NextTurnPredictor;

// Pure SR
let mut sr = SrPredictor::new()
    .with_gamma(0.9)       // discount factor — soft horizon
    .with_eta_base(0.1)    // base learning rate; decays with observations
    .with_warmup(50);      // observations before confidence saturates to 1.0

// Or hybrid: Markov serves cold-start, SR takes over once warm
let hybrid = HybridPredictor::new(SrPredictor::new(), MarkovPredictor::new())
    .with_threshold(0.5);  // switch to SR when SR.confidence() ≥ 0.5
```

Both impl `NextTurnPredictor`, so they slot directly into `QueryContext`:

```rust
let ctx = QueryContext::with_predictor(hybrid);
```

## CLI flag

```bash
primd serve --index /tmp/primd-faq --predictor hybrid   # SR + Markov fallback
primd serve --index /tmp/primd-faq --predictor sr       # pure SR, cold-starts uniform
primd serve --index /tmp/primd-faq --predictor markov   # v0.1 default (still default)
```

The `markov` default is preserved for v0.2 to keep behavior stable for existing deployments; switch to `hybrid` for the predictive-cache moat.

## What's tabular vs low-rank

v0.2 ships the **tabular** variant: SR is stored as a sparse `HashMap<EventId, HashMap<EventId, f32>>`. For voice corpora with tens to low hundreds of events this is correct and fast — confidence computations and TD updates are O(N²) worst case but typically O(N · degree).

v0.2.5 will add the **low-rank** variant from the strategy memo: `W: 256 × 32` projection and `M_low: 32 × 32` (K=32 = one cache line × 4 SIMD lanes of f64), with spectral-gap confidence and ~µs TD updates via portable SIMD. The trait surface stays the same; low-rank slots in as a third `NextTurnPredictor` impl.

## Persistence

```rust
sr.save_to_file(&Path::new("/tmp/sr.json"))?;
let loaded = SrPredictor::load_from_file(&Path::new("/tmp/sr.json"))?;
```

JSON format. Currently no on-disk integration with `primd serve`'s session manager — every new session starts with a fresh SR. v0.2.5 will add per-user persistence (`SrPredictor` artifact keyed by `user_id` from the OpenAI `user` field), warm-starting returning users.

## Risks (from the strategy memo)

| Risk | Mitigation |
|---|---|
| Non-stationary policy (user changes topic mid-session) | TD(0) has built-in recency bias; for adversarial topic noise the Hybrid wrapper's SR confidence drops and gates back to Markov |
| Eigenvector instability below ~30 unique signatures | Tabular variant doesn't compute eigenvectors; v0.2.5 adds a 30-event minimum gate for spectral-gap confidence |
| Memory blow-up if K grows (v0.2.5) | Cap K ≤ 64; effective dim is bounded by policy mixing time |
| Catastrophic forgetting across sessions | Per-user persistence (v0.2.5) when `user_id` provided |
| SR underperforms Markov on real corpora | Hybrid wrapper's SR confidence gating falls back to Markov — engineering artifact has narrative value regardless of A/B outcome |

## References

1. Dayan 1993, *Neural Computation* 5:613–624 — original SR formulation
2. Stachenfeld, Botvinick, Gershman 2017, *Nat Neurosci* 20:1643–1653 (correction 21:895) — hippocampal predictive map
3. Russek, Momennejad, Botvinick, Gershman, Daw 2017, *Nat Hum Behav* 1:680–692 — TD(0) update rule
4. Gershman 2018, *J Neurosci* 38:7193–7200 — review with implementation notes
5. Sherstan, Machado, Pilarski 2018, arXiv:1803.09001 — incremental SR with TD
6. Mahadevan & Maggioni 2007, *JMLR* 8:2169–2231 — Laplacian basis / proto-value functions
