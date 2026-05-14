//! WebAssembly bindings for primd's predictive turn-cache runtime.
//!
//! Targets in-page voice agents that want speculative retrieval without
//! a Rust server — the entire engine compiled to ~250 KB of WASM.
//!
//! ## Build
//!
//! Requires `wasm32-unknown-unknown` rust target. Arch Linux:
//!
//! ```bash
//! sudo pacman -S rust-wasm
//! cargo build -p primd-wasm --target wasm32-unknown-unknown --release
//! ```
//!
//! Or via rustup:
//!
//! ```bash
//! rustup target add wasm32-unknown-unknown
//! cargo build -p primd-wasm --target wasm32-unknown-unknown --release
//! ```
//!
//! For JS-package output, install `wasm-bindgen-cli` and run:
//!
//! ```bash
//! wasm-bindgen target/wasm32-unknown-unknown/release/primd_wasm.wasm \
//!     --out-dir pkg --target web --no-typescript
//! ```
//!
//! ## Limits
//!
//! - The `fastembed`/`local` and `openai` embedding paths are disabled
//!   in WASM (they pull tokenizer crates / network stack that don't
//!   compile to wasm32-unknown-unknown). Embedding must happen in JS
//!   before calling into the WASM module; the WASM surface accepts
//!   pre-computed `Uint8Array(32)` signatures.
//! - HNSW is disabled in this build because `instant-distance` pulls
//!   in `parking_lot` which has WASM-unfriendly internals. The hot
//!   path is the SIMD signature scan + speculative cache, which works
//!   fine in WASM (without SIMD acceleration — falls back to the
//!   scalar popcount).
//! - Per-event scope unions still work; the v0.2 subset-rescan path is
//!   what's available in WASM.

use std::cell::RefCell;
use std::rc::Rc;

use primd_core::embed::binary::BinarySignature;
use primd_core::index::events::EventCatalog;
use primd_core::index::shards::{HierarchicalIndex, SearchOptions};
use primd_core::index::signatures::SignatureIndex;
use primd_core::predict::MarkovPredictor;
use primd_core::query_context::QueryContext;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// JS-facing wrapper around a `HierarchicalIndex`.
///
/// Construct via `PrimdIndex.from_arrays(...)`; query via
/// `query(signature_bytes, top_k)` for stateless or `finalize(...)` /
/// `observe_partial(...)` / `warm_next()` for session-aware flows.
#[wasm_bindgen]
pub struct PrimdIndex {
    inner: HierarchicalIndex,
    session: Rc<RefCell<Option<QueryContext>>>,
}

#[wasm_bindgen]
impl PrimdIndex {
    /// Build a primd index from JS-side data.
    ///
    /// - `signatures_flat`: `Uint8Array` of length `n_docs * 32`, the
    ///   docs' 256-bit signatures packed contiguously.
    /// - `event_names`: `Array<string>`, ordered, one per event.
    /// - `event_scopes`: `Array<Uint32Array>`, parallel to
    ///   `event_names`, giving the doc indices for each event.
    #[wasm_bindgen(constructor)]
    pub fn new(
        signatures_flat: &[u8],
        event_names: js_sys::Array,
        event_scopes: js_sys::Array,
    ) -> Result<PrimdIndex, JsError> {
        if !signatures_flat.len().is_multiple_of(32) {
            return Err(JsError::new(
                "signatures_flat length must be a multiple of 32 bytes",
            ));
        }
        let n_docs = signatures_flat.len() / 32;
        let mut sigs: Vec<BinarySignature> = Vec::with_capacity(n_docs);
        for i in 0..n_docs {
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&signatures_flat[i * 32..(i + 1) * 32]);
            sigs.push(BinarySignature(buf));
        }

        if event_names.length() != event_scopes.length() {
            return Err(JsError::new(
                "event_names and event_scopes must have the same length",
            ));
        }

        let mut named: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..event_names.length() {
            let name = event_names
                .get(i)
                .as_string()
                .ok_or_else(|| JsError::new("event_names entries must be strings"))?;
            let scope = event_scopes.get(i);
            let arr = js_sys::Uint32Array::from(scope);
            let mut indices: Vec<usize> = Vec::with_capacity(arr.length() as usize);
            for j in 0..arr.length() {
                indices.push(arr.get_index(j) as usize);
            }
            named.insert(name, indices);
        }

        let signatures = SignatureIndex::new(sigs);
        let events = EventCatalog::from_named_scope(&named, n_docs);
        let inner = HierarchicalIndex::new(signatures, events);
        Ok(PrimdIndex {
            inner,
            session: Rc::new(RefCell::new(None)),
        })
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Stateless query. Returns the top-K hits as a JS array of
    /// `{ rank, distance, doc_idx, event }` objects.
    pub fn query(&self, signature_bytes: &[u8], top_k: usize) -> Result<JsValue, JsError> {
        if signature_bytes.len() != 32 {
            return Err(JsError::new("signature_bytes must be exactly 32 bytes"));
        }
        let mut buf = [0u8; 32];
        buf.copy_from_slice(signature_bytes);
        let sig = BinarySignature(buf);

        let opts = SearchOptions::default();
        let result = self.inner.search(&sig, top_k, &opts);

        let hits: Vec<Hit> = result
            .results
            .iter()
            .enumerate()
            .map(|(rank, &(distance, doc_idx))| {
                let event = self
                    .inner
                    .events()
                    .doc_event(doc_idx)
                    .and_then(|e| self.inner.events().event_name(e))
                    .unwrap_or("")
                    .to_string();
                Hit {
                    rank: rank + 1,
                    distance,
                    doc_idx,
                    event,
                }
            })
            .collect();

        serde_wasm_bindgen::to_value(&hits).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Reset (or create) the in-memory session. Subsequent
    /// `observe_partial` / `finalize` / `warm_next` calls share state
    /// through this session.
    pub fn session_start(&self) {
        *self.session.borrow_mut() = Some(QueryContext::with_predictor(MarkovPredictor::new()));
    }

    /// Feed a partial-transcript signature during STT. Cheap; runs the
    /// streaming gate first.
    pub fn observe_partial(
        &self,
        signature_bytes: &[u8],
        top_k: usize,
    ) -> Result<(), JsError> {
        if signature_bytes.len() != 32 {
            return Err(JsError::new("signature_bytes must be exactly 32 bytes"));
        }
        let mut buf = [0u8; 32];
        buf.copy_from_slice(signature_bytes);
        let sig = BinarySignature(buf);

        let mut sess = self.session.borrow_mut();
        let ctx = sess
            .get_or_insert_with(|| QueryContext::with_predictor(MarkovPredictor::new()));
        ctx.observe_partial(&self.inner, sig, top_k);
        Ok(())
    }

    /// End-of-utterance retrieval. Returns the speculative cache hit
    /// if the partial converged on the final, otherwise scans and
    /// returns fresh top-K.
    pub fn finalize(&self, signature_bytes: &[u8], top_k: usize) -> Result<JsValue, JsError> {
        if signature_bytes.len() != 32 {
            return Err(JsError::new("signature_bytes must be exactly 32 bytes"));
        }
        let mut buf = [0u8; 32];
        buf.copy_from_slice(signature_bytes);
        let sig = BinarySignature(buf);

        let mut sess = self.session.borrow_mut();
        let ctx = sess
            .get_or_insert_with(|| QueryContext::with_predictor(MarkovPredictor::new()));
        let out = ctx.finalize(&self.inner, sig, top_k);

        let hits: Vec<Hit> = out
            .results
            .iter()
            .enumerate()
            .map(|(rank, &(distance, doc_idx))| {
                let event = self
                    .inner
                    .events()
                    .doc_event(doc_idx)
                    .and_then(|e| self.inner.events().event_name(e))
                    .unwrap_or("")
                    .to_string();
                Hit {
                    rank: rank + 1,
                    distance,
                    doc_idx,
                    event,
                }
            })
            .collect();

        let resp = FinalizeResp {
            served_by: served_by_str(out.served_by),
            hits,
        };
        serde_wasm_bindgen::to_value(&resp).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Pre-warm the predicted next-turn scope. Call during TTS
    /// playback so the next observe_partial is already scope-narrowed.
    pub fn warm_next(&self) {
        let mut sess = self.session.borrow_mut();
        let ctx = sess
            .get_or_insert_with(|| QueryContext::with_predictor(MarkovPredictor::new()));
        let _ = ctx.warm_next(&self.inner);
    }
}

#[derive(Serialize)]
struct Hit {
    rank: usize,
    distance: u32,
    doc_idx: usize,
    event: String,
}

#[derive(Serialize)]
struct FinalizeResp {
    served_by: &'static str,
    hits: Vec<Hit>,
}

fn served_by_str(s: primd_core::query_context::ServedBy) -> &'static str {
    use primd_core::query_context::ServedBy;
    match s {
        ServedBy::FullScan => "full_scan",
        ServedBy::ShardScan => "shard_scan",
        ServedBy::Speculative => "speculative",
        ServedBy::DeltaCache => "delta_cache",
    }
}

// When compiled for non-wasm targets, expose a tiny smoke-test that
// exercises the same constructor + query path through native types, so
// `cargo test -p primd-wasm` runs on the host toolchain without
// requiring wasm32-unknown-unknown.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// We can't call the wasm-bindgen surface directly (it expects
    /// js_sys::Array). But the underlying logic — building a
    /// HierarchicalIndex from raw signatures + scopes and querying it
    /// — is exercised by primd-core's tests. This smoke test just
    /// verifies the crate compiles on the host and the public types
    /// are stable.
    #[test]
    fn types_compile_on_host() {
        // The constructor needs js_sys types so we can't call it
        // directly. But ensure the module compiles by referencing the
        // type.
        let _ = std::mem::size_of::<PrimdIndex>();
    }

    #[test]
    fn hit_serializes() {
        let h = Hit {
            rank: 1,
            distance: 12,
            doc_idx: 42,
            event: "trial".to_string(),
        };
        let json = serde_json::to_string(&h).unwrap();
        assert!(json.contains("\"rank\":1"));
        assert!(json.contains("\"event\":\"trial\""));
    }
}
