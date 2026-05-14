# WebAssembly target

`primd-wasm` exposes the predictive-turn-cache runtime as a WASM module so primd can run in-browser for in-page voice agents (no Rust server needed). Roadmap-promised in v0.3.

## Build

`primd-wasm` builds against `wasm32-unknown-unknown`. Install the target:

```bash
# Arch / CachyOS
sudo pacman -S rust-wasm

# Or via rustup (any distro)
rustup target add wasm32-unknown-unknown
```

Then build:

```bash
cargo build -p primd-wasm --target wasm32-unknown-unknown --release
```

For a JS-ready package, install `wasm-bindgen-cli` and post-process:

```bash
cargo install wasm-bindgen-cli
wasm-bindgen target/wasm32-unknown-unknown/release/primd_wasm.wasm \
    --out-dir pkg --target web --no-typescript
```

The resulting `pkg/primd_wasm.js` and `pkg/primd_wasm_bg.wasm` (~250 KB compressed) can be loaded directly from a `<script type="module">` tag.

## Embedding model placement

WASM doesn't ship an embedding model. The `fastembed` / `local` and `openai` paths are disabled in `primd-wasm` (one pulls in tokenizer crates that don't compile to `wasm32-unknown-unknown`; the other needs the network stack). Embedding must happen in JS before calling into the module:

- Use a JS embedding model (`@xenova/transformers`, `tensorflow.js`, etc.) to compute a dense embedding
- Quantize to a 256-bit signature with a small JS sign-bit helper
- Pass the 32-byte `Uint8Array` into `index.query(...)`, `index.observe_partial(...)`, or `index.finalize(...)`

```javascript
import init, { PrimdIndex } from './pkg/primd_wasm.js';

await init();

// signatures_flat: Uint8Array of length n_docs * 32
// event_names: string[]
// event_scopes: Uint32Array[]
const index = new PrimdIndex(signatures_flat, event_names, event_scopes);

// Stateless query (32-byte signature)
const hits = index.query(querySig, 5);
// → [{ rank: 1, distance: 12, doc_idx: 42, event: "trial" }, ...]

// Session-aware speculative path
index.session_start();
index.observe_partial(partialSig, 5);                  // during STT
const final = index.finalize(finalSig, 5);             // end of utterance
// → { served_by: "speculative", hits: [...] }
index.warm_next();                                     // during TTS playback
```

## What's available vs disabled in WASM

| Feature | Native | WASM |
|---|---|---|
| SIMD Hamming scan | AVX-512 / AVX2 | scalar fallback (still ~10 ns / sig) |
| Event-scoped subset rescan | ✅ | ✅ |
| QueryContext (observe / finalize / warm) | ✅ | ✅ |
| Markov predictor | ✅ | ✅ |
| Per-event HNSW shards | ✅ | ❌ (instant-distance has WASM-unfriendly internals) |
| `fastembed` / `local` embedder | ✅ | ❌ (tokenizer crate doesn't build) |
| `openai` embedder | ✅ (HTTP) | ❌ (no network stack) |
| File / mmap signature loading | ✅ | ❌ (no filesystem) |

In WASM the corpus is passed in from JS at index construction time, kept in WASM memory, and queried in-place. There's no persistence layer — the user reloads the page, the corpus reloads.

## Limits

- Scalar Hamming popcount is ~3× slower than the AVX2 native path. At 10 k docs the cold scan is ~100 µs in WASM (still well under voice TTS budgets).
- WASM memory is bounded (typically 2 GB max). The realistic ceiling is ~100 k–500 k docs depending on how aggressively the browser limits the WASM heap. For larger corpora, host primd remotely and use `pipecat-primd` / `livekit-primd` instead.
- No threading: this is the single-threaded WASM target. Multi-threaded WASM is possible (`Atomics + SharedArrayBuffer`) but adds COOP/COEP header requirements that most demo deployments don't satisfy. v0.4 work.

## When to use this vs the native server

| Use the WASM build when… | Use `primd serve` (native) when… |
|---|---|
| Browser-only deployments, in-page voice agents | Pipecat / LiveKit / Vapi pipelines |
| Privacy-sensitive corpora that can't leave the device | Server-side voice agents |
| Demo / playground sites | Production deployments at scale |
| Edge / IoT (when you have Rust + WASM toolchain) | Multi-tenant workloads |

## Related

- [primd-wasm crate](../../primd-wasm/) — source
- [Architecture overview](../architecture/overview.md)
- [MoshiRAG integration](moshirag.md) — OpenAI-compatible adapter
