use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::Args;
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use primd_core::embed::{
    EmbeddingPipeline, HashedEmbedder, LocalEmbedder, OpenAIEmbedder, random_projection,
};
use primd_core::{
    BinarySignature, EventCatalog, HierarchicalIndex, MarkovPredictor, PrimdError, QueryContext,
    QueryOutput, SearchOptions, ServedBy, SignatureIndex,
};

use crate::cmd_index::{EmbedderKind, LocalModelKind};

#[derive(Args, Debug)]
pub struct ServeArgs {
    #[arg(short, long)]
    pub index: PathBuf,

    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
}

#[derive(Deserialize, Clone)]
struct Manifest {
    embedder: EmbedderKind,
    dim: usize,
    use_bigrams: bool,
    #[serde(default)]
    openai_model: Option<String>,
    #[serde(default)]
    local_model: Option<LocalModelKind>,
    #[serde(default)]
    projection_seed: Option<u64>,
    #[serde(default)]
    source_dim: Option<usize>,
    n_entries: usize,
    ids: Vec<String>,
    /// Original doc text per entry, parallel to `ids`. Older indexes built
    /// before this field existed deserialize with an empty vec.
    #[serde(default)]
    texts: Vec<String>,
    #[serde(default)]
    scope: BTreeMap<String, Vec<usize>>,
}

#[derive(Deserialize)]
struct QueryRequest {
    text: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    parallel: bool,
}

#[derive(Deserialize)]
struct WarmRequest {}

fn default_top_k() -> usize {
    10
}

#[derive(Serialize)]
struct QueryHit {
    rank: usize,
    distance: u32,
    id: String,
    event: String,
}

#[derive(Serialize)]
struct QueryResponse {
    embedder: EmbedderKind,
    embed_us: u128,
    scan_us: u128,
    corpus_size: usize,
    hits: Vec<QueryHit>,
    served_by: &'static str,
    predicted_events: Vec<String>,
    shard_scope_size: usize,
}

#[derive(Serialize)]
struct SessionWarmResponse {
    predicted_events: Vec<String>,
    shard_scope_size: usize,
}

#[derive(Serialize)]
struct StatsResponse {
    corpus_size: usize,
    embedder: EmbedderKind,
    queries_served: u64,
    total_embed_us: u128,
    total_scan_us: u128,
    mean_embed_us: u128,
    mean_scan_us: u128,
    sessions: usize,
}

trait QueryEmbedder: Send + Sync {
    fn embed_query(&self, text: &str) -> Result<BinarySignature, PrimdError>;
}

impl<E: primd_core::embed::Embedder> QueryEmbedder for EmbeddingPipeline<E> {
    fn embed_query(&self, text: &str) -> Result<BinarySignature, PrimdError> {
        self.embed_to_signature(text)
    }
}

struct State {
    manifest: Manifest,
    index: HierarchicalIndex,
    event_for: BTreeMap<usize, String>,
    event_name_for_id: HashMap<u32, String>,
    /// Doc-id -> original text. Empty for indexes built before the manifest
    /// `texts` field shipped. The OpenAI adapter falls back to id-only output
    /// when this is empty.
    text_for_id: HashMap<String, String>,
    pipeline: Box<dyn QueryEmbedder>,
    predictor_path: Option<PathBuf>,
    sessions: Mutex<HashMap<String, QueryContext>>,
    queries_served: AtomicU64,
    total_embed_ns: AtomicU64,
    total_scan_ns: AtomicU64,
}

impl State {
    fn record_latency(&self, embed_ns: u128, scan_ns: u128) {
        self.queries_served.fetch_add(1, Ordering::Relaxed);
        self.total_embed_ns
            .fetch_add(embed_ns as u64, Ordering::Relaxed);
        self.total_scan_ns
            .fetch_add(scan_ns as u64, Ordering::Relaxed);
    }

    fn new_session(&self) -> QueryContext {
        let predictor = self
            .predictor_path
            .as_ref()
            .and_then(|path| MarkovPredictor::load_from_file(path).ok())
            .unwrap_or_default();
        QueryContext::with_predictor(predictor)
    }
}

pub fn run(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = args.index.join("manifest.json");
    let sigs_path = args.index.join("signatures.bin");
    let transitions_path = args.index.join("transitions.json");

    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    let signatures = SignatureIndex::from_file(&sigs_path)?;
    if signatures.len() != manifest.n_entries {
        return Err(format!(
            "manifest claims {} entries; signatures file has {}",
            manifest.n_entries,
            signatures.len()
        )
        .into());
    }

    let mut event_for = BTreeMap::new();
    let mut event_name_for_id = HashMap::new();
    for (i, (event_name, indices)) in manifest.scope.iter().enumerate() {
        event_name_for_id.insert(i as u32, event_name.clone());
        for &idx in indices {
            event_for.insert(idx, event_name.clone());
        }
    }

    let events = EventCatalog::from_named_scope(&manifest.scope, manifest.n_entries);
    let index = HierarchicalIndex::new(signatures, events);

    let mut text_for_id: HashMap<String, String> = HashMap::new();
    if manifest.texts.len() == manifest.ids.len() {
        for (id, text) in manifest.ids.iter().zip(manifest.texts.iter()) {
            text_for_id.insert(id.clone(), text.clone());
        }
    }

    eprintln!("loading embedder {:?}…", manifest.embedder);
    let pipeline = build_pipeline(&manifest)?;

    let state = Arc::new(State {
        manifest,
        index,
        event_for,
        event_name_for_id,
        text_for_id,
        pipeline,
        predictor_path: transitions_path.exists().then_some(transitions_path),
        sessions: Mutex::new(HashMap::new()),
        queries_served: AtomicU64::new(0),
        total_embed_ns: AtomicU64::new(0),
        total_scan_ns: AtomicU64::new(0),
    });

    let server =
        Server::http(&args.bind).map_err(|e| format!("failed to bind {}: {e}", args.bind))?;

    eprintln!(
        "primd serve | listening on http://{} | corpus={} embedder={:?}",
        args.bind,
        state.index.len(),
        state.manifest.embedder
    );
    eprintln!(
        "endpoints: POST /query | POST /v1/chat/completions | GET /health | GET /stats | POST /session/{{id}}/observe | POST /session/{{id}}/finalize | POST /session/{{id}}/warm | POST /session/{{id}}/reset"
    );

    for mut request in server.incoming_requests() {
        let path = request.url().to_string();
        let method = request.method().clone();
        let state = state.clone();

        let response_result = match (method, path.as_str()) {
            (Method::Get, "/health") => Ok(json_response(200, "{\"status\":\"ok\"}")),
            (Method::Get, "/stats") => stats_response(&state),
            (Method::Post, "/query") => {
                read_body(&mut request).and_then(|body| handle_query(&state, &body))
            }
            (Method::Post, "/v1/chat/completions") => {
                read_body(&mut request).and_then(|body| handle_chat_completions(&state, &body))
            }
            (Method::Post, _) if path.starts_with("/session/") => {
                read_body(&mut request).and_then(|body| handle_session(&state, &path, &body))
            }
            _ => Ok(json_response(404, "{\"error\":\"not found\"}")),
        };

        let response = match response_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("request error: {e}");
                json_response(400, format!("{{\"error\":\"{}\"}}", json_escape(&e)))
            }
        };

        if let Err(e) = request.respond(response) {
            eprintln!("respond error: {e}");
        }
    }

    Ok(())
}

fn read_body(request: &mut tiny_http::Request) -> Result<String, String> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    Ok(body)
}

fn handle_query(state: &State, body: &str) -> Result<Response<std::io::Cursor<Vec<u8>>>, String> {
    let req: QueryRequest = serde_json::from_str(body).map_err(|e| format!("bad json: {e}"))?;
    if req.text.trim().is_empty() {
        return Err("text is empty".into());
    }

    let embed_start = Instant::now();
    let query_sig = state
        .pipeline
        .embed_query(&req.text)
        .map_err(|e| e.to_string())?;
    let embed_elapsed = embed_start.elapsed();

    let scan_start = Instant::now();
    let search = state.index.search(
        &query_sig,
        req.top_k,
        &SearchOptions {
            parallel: req.parallel,
            ..SearchOptions::default()
        },
    );
    let scan_elapsed = scan_start.elapsed();
    state.record_latency(embed_elapsed.as_nanos(), scan_elapsed.as_nanos());

    let response = QueryResponse {
        embedder: state.manifest.embedder,
        embed_us: embed_elapsed.as_micros(),
        scan_us: scan_elapsed.as_micros(),
        corpus_size: state.index.len(),
        hits: map_hits(state, search.results),
        served_by: if search.used_shards {
            "shard_scan"
        } else {
            "full_scan"
        },
        predicted_events: search
            .candidate_events
            .into_iter()
            .filter_map(|event| state.event_name_for_id.get(&event.0).cloned())
            .collect(),
        shard_scope_size: search.shard_scope_size,
    };
    Ok(json_response(
        200,
        serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
    ))
}

fn handle_session(
    state: &State,
    path: &str,
    body: &str,
) -> Result<Response<std::io::Cursor<Vec<u8>>>, String> {
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    if parts.len() != 3 || parts[0] != "session" {
        return Err("expected /session/{id}/{action}".into());
    }
    let session_id = parts[1].to_string();
    let action = parts[2];

    match action {
        "observe" => {
            let req: QueryRequest =
                serde_json::from_str(body).map_err(|e| format!("bad json: {e}"))?;
            let sig = state
                .pipeline
                .embed_query(&req.text)
                .map_err(|e| e.to_string())?;
            let mut sessions = state.sessions.lock().map_err(|_| "session lock poisoned")?;
            let session = sessions
                .entry(session_id)
                .or_insert_with(|| state.new_session());
            session.observe_partial(&state.index, sig, req.top_k);
            Ok(json_response(200, "{\"status\":\"ok\"}"))
        }
        "finalize" => {
            let req: QueryRequest =
                serde_json::from_str(body).map_err(|e| format!("bad json: {e}"))?;
            let embed_start = Instant::now();
            let sig = state
                .pipeline
                .embed_query(&req.text)
                .map_err(|e| e.to_string())?;
            let embed_elapsed = embed_start.elapsed();
            let scan_start = Instant::now();
            let mut sessions = state.sessions.lock().map_err(|_| "session lock poisoned")?;
            let session = sessions
                .entry(session_id)
                .or_insert_with(|| state.new_session());
            let out = session.finalize(&state.index, sig, req.top_k);
            let scan_elapsed = scan_start.elapsed();
            state.record_latency(embed_elapsed.as_nanos(), scan_elapsed.as_nanos());
            Ok(json_response(
                200,
                serde_json::to_string(&session_query_response(
                    state,
                    out,
                    req.top_k,
                    embed_elapsed.as_micros(),
                    scan_elapsed.as_micros(),
                ))
                .unwrap_or_else(|_| "{}".to_string()),
            ))
        }
        "warm" => {
            let _: WarmRequest =
                serde_json::from_str(if body.trim().is_empty() { "{}" } else { body })
                    .map_err(|e| format!("bad json: {e}"))?;
            let mut sessions = state.sessions.lock().map_err(|_| "session lock poisoned")?;
            let session = sessions
                .entry(session_id)
                .or_insert_with(|| state.new_session());
            let predicted = session.warm_next(&state.index);
            let response = SessionWarmResponse {
                predicted_events: predicted
                    .into_iter()
                    .filter_map(|event| state.event_name_for_id.get(&event.0).cloned())
                    .collect(),
                shard_scope_size: session.predicted_scope_size(),
            };
            Ok(json_response(
                200,
                serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
            ))
        }
        "reset" => {
            let mut sessions = state.sessions.lock().map_err(|_| "session lock poisoned")?;
            sessions.remove(&session_id);
            Ok(json_response(200, "{\"status\":\"reset\"}"))
        }
        _ => Err("unknown session action".into()),
    }
}

fn session_query_response(
    state: &State,
    out: QueryOutput,
    _top_k: usize,
    embed_us: u128,
    scan_us: u128,
) -> QueryResponse {
    QueryResponse {
        embedder: state.manifest.embedder,
        embed_us,
        scan_us,
        corpus_size: state.index.len(),
        hits: map_hits(state, out.results),
        served_by: served_by_label(out.served_by),
        predicted_events: out
            .predicted_events
            .into_iter()
            .filter_map(|event| state.event_name_for_id.get(&event.0).cloned())
            .collect(),
        shard_scope_size: out.shard_scope_size,
    }
}

fn served_by_label(served_by: ServedBy) -> &'static str {
    match served_by {
        ServedBy::FullScan => "full_scan",
        ServedBy::ShardScan => "shard_scan",
        ServedBy::Speculative => "speculative",
        ServedBy::DeltaCache => "delta_cache",
    }
}

fn map_hits(state: &State, raw: Vec<(u32, usize)>) -> Vec<QueryHit> {
    raw.into_iter()
        .enumerate()
        .map(|(rank, (dist, idx))| QueryHit {
            rank: rank + 1,
            distance: dist,
            id: state
                .manifest
                .ids
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "?".to_string()),
            event: state
                .event_for
                .get(&idx)
                .cloned()
                .unwrap_or_else(|| "_default".to_string()),
        })
        .collect()
}

fn stats_response(state: &State) -> Result<Response<std::io::Cursor<Vec<u8>>>, String> {
    let queries = state.queries_served.load(Ordering::Relaxed);
    let total_embed = state.total_embed_ns.load(Ordering::Relaxed) as u128;
    let total_scan = state.total_scan_ns.load(Ordering::Relaxed) as u128;
    let denom = queries.max(1) as u128;
    let sessions = state
        .sessions
        .lock()
        .map_err(|_| "session lock poisoned")?
        .len();
    let body = StatsResponse {
        corpus_size: state.index.len(),
        embedder: state.manifest.embedder,
        queries_served: queries,
        total_embed_us: total_embed / 1000,
        total_scan_us: total_scan / 1000,
        mean_embed_us: (total_embed / 1000) / denom,
        mean_scan_us: (total_scan / 1000) / denom,
        sessions,
    };
    Ok(json_response(
        200,
        serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string()),
    ))
}

fn build_pipeline(manifest: &Manifest) -> Result<Box<dyn QueryEmbedder>, PrimdError> {
    match manifest.embedder {
        EmbedderKind::Hashed => {
            let mut e = HashedEmbedder::new(manifest.dim);
            if !manifest.use_bigrams {
                e = e.without_bigrams();
            }
            Ok(Box::new(EmbeddingPipeline::new_direct(e)?))
        }
        EmbedderKind::Openai => {
            let mut e = OpenAIEmbedder::from_env()?.with_dim(manifest.dim);
            if let Some(m) = &manifest.openai_model {
                e = e.with_model(m);
            }
            Ok(Box::new(EmbeddingPipeline::new_direct(e)?))
        }
        EmbedderKind::Local => {
            let kind = manifest
                .local_model
                .ok_or_else(|| PrimdError::Embedder("manifest missing local_model".into()))?;
            let seed = manifest
                .projection_seed
                .ok_or_else(|| PrimdError::Embedder("manifest missing projection_seed".into()))?;
            let source_dim = manifest
                .source_dim
                .ok_or_else(|| PrimdError::Embedder("manifest missing source_dim".into()))?;
            let local = LocalEmbedder::new(kind.into_local())?;
            let proj = random_projection(seed, source_dim);
            Ok(Box::new(EmbeddingPipeline::new_with_pca(local, proj)?))
        }
    }
}

fn json_response(status: u16, body: impl Into<String>) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(body.into())
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// OpenAI-compatible Chat Completions adapter (the MoshiRAG back-end slot).
//
// MoshiRAG's reference back-end is a generic LLM served via vLLM at 1–3 s
// latency. This adapter lets MoshiRAG (or any OpenAI-compatible client) point
// at primd instead and receive retrieved context as the "completion" string
// — sub-200 µs end-to-end. primd doesn't generate; it returns context, and the
// model in the loop synthesizes the spoken response.
//
// Wire-compatible with the OpenAI Chat Completions API surface: accepts
// `messages`, `model`, `user`, ignores temperature/stream/max_tokens. The
// `user` field doubles as a primd session id, so the speculative + delta-
// cache paths stay engaged across turns.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatCompletionsRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    user: Option<String>,
    /// primd extension. Not in standard OpenAI; clients can hint how many
    /// docs to retrieve. Defaults to 5.
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionsResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
    /// primd-specific metadata. Standard OpenAI clients ignore it; MoshiRAG
    /// and primd-aware clients can use it to inspect which retrieval path
    /// served the response.
    primd: PrimdMeta,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageOut,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatMessageOut {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct PrimdMeta {
    served_by: &'static str,
    embed_us: u128,
    scan_us: u128,
    predicted_events: Vec<String>,
    hits: Vec<QueryHit>,
}

const CHAT_DEFAULT_TOP_K: usize = 5;

/// Extract the most recent user message content. Returns `Err` with a
/// human-readable reason when the request doesn't contain a usable query.
fn extract_user_query(messages: &[ChatMessage]) -> Result<&str, String> {
    let last_user = messages.iter().rev().find(|m| m.role == "user");
    match last_user {
        Some(m) if !m.content.trim().is_empty() => Ok(m.content.as_str()),
        Some(_) => Err("latest user message is empty".into()),
        None => Err("messages must contain at least one user message".into()),
    }
}

/// Build the `content` string returned to the OpenAI client. Numbered list
/// because that's the format LLMs in MoshiRAG-style pipelines re-emit most
/// naturally back to TTS.
fn format_retrieved_context(hits: &[QueryHit], text_for_id: &HashMap<String, String>) -> String {
    if hits.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (i, hit) in hits.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let body = text_for_id
            .get(&hit.id)
            .map(|t| t.as_str())
            .unwrap_or(hit.id.as_str());
        out.push_str(&format!("[{}] ({}) {}", i + 1, hit.event, body));
    }
    out
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn handle_chat_completions(
    state: &State,
    body: &str,
) -> Result<Response<std::io::Cursor<Vec<u8>>>, String> {
    let req: ChatCompletionsRequest =
        serde_json::from_str(body).map_err(|e| format!("bad json: {e}"))?;

    let query = extract_user_query(&req.messages)?;
    let top_k = req.top_k.unwrap_or(CHAT_DEFAULT_TOP_K).max(1);

    // Embed the query once.
    let embed_start = Instant::now();
    let sig = state
        .pipeline
        .embed_query(query)
        .map_err(|e| e.to_string())?;
    let embed_elapsed = embed_start.elapsed();

    let scan_start = Instant::now();
    let (hits, served_by, predicted_events) = if let Some(session_id) = req.user.as_deref() {
        // Session-aware path: drives QueryContext finalize, engaging the
        // speculative + delta-cache + predictor surfaces.
        let mut sessions = state.sessions.lock().map_err(|_| "session lock poisoned")?;
        let session = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| state.new_session());
        let out = session.finalize(&state.index, sig, top_k);
        let hits = map_hits(state, out.results);
        let predicted = out
            .predicted_events
            .into_iter()
            .filter_map(|event| state.event_name_for_id.get(&event.0).cloned())
            .collect();
        (hits, served_by_label(out.served_by), predicted)
    } else {
        // Stateless path: equivalent to /query, no session state.
        let search = state.index.search(
            &sig,
            top_k,
            &SearchOptions {
                parallel: true,
                ..SearchOptions::default()
            },
        );
        let predicted = search
            .candidate_events
            .iter()
            .filter_map(|event| state.event_name_for_id.get(&event.0).cloned())
            .collect();
        let label = if search.used_shards {
            "shard_scan"
        } else {
            "full_scan"
        };
        (map_hits(state, search.results), label, predicted)
    };
    let scan_elapsed = scan_start.elapsed();
    state.record_latency(embed_elapsed.as_nanos(), scan_elapsed.as_nanos());

    let content = format_retrieved_context(&hits, &state.text_for_id);
    let response = ChatCompletionsResponse {
        id: format!("chatcmpl-primd-{}", unix_seconds()),
        object: "chat.completion",
        created: unix_seconds(),
        model: req.model.unwrap_or_else(|| "primd".to_string()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant",
                content: content.clone(),
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: query.split_whitespace().count(),
            completion_tokens: content.split_whitespace().count(),
            total_tokens: query.split_whitespace().count() + content.split_whitespace().count(),
        },
        primd: PrimdMeta {
            served_by,
            embed_us: embed_elapsed.as_micros(),
            scan_us: scan_elapsed.as_micros(),
            predicted_events,
            hits,
        },
    };

    Ok(json_response(
        200,
        serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn extract_user_query_returns_latest_user_message() {
        let messages = vec![
            msg("system", "you are a helpful assistant"),
            msg("user", "first question"),
            msg("assistant", "first answer"),
            msg("user", "what about pricing"),
        ];
        let q = extract_user_query(&messages).unwrap();
        assert_eq!(q, "what about pricing");
    }

    #[test]
    fn extract_user_query_errors_on_no_user_messages() {
        let messages = vec![msg("system", "hi"), msg("assistant", "hi back")];
        assert!(extract_user_query(&messages).is_err());
    }

    #[test]
    fn extract_user_query_errors_on_empty_user_message() {
        let messages = vec![msg("user", "   ")];
        assert!(extract_user_query(&messages).is_err());
    }

    #[test]
    fn format_retrieved_context_uses_text_when_available() {
        let mut texts = HashMap::new();
        texts.insert("faq-1".to_string(), "We offer a 14-day free trial.".to_string());
        let hits = vec![QueryHit {
            rank: 1,
            distance: 12,
            id: "faq-1".to_string(),
            event: "trial".to_string(),
        }];
        let out = format_retrieved_context(&hits, &texts);
        assert_eq!(out, "[1] (trial) We offer a 14-day free trial.");
    }

    #[test]
    fn format_retrieved_context_falls_back_to_id_when_text_absent() {
        let texts = HashMap::new();
        let hits = vec![QueryHit {
            rank: 1,
            distance: 12,
            id: "faq-unknown".to_string(),
            event: "x".to_string(),
        }];
        let out = format_retrieved_context(&hits, &texts);
        assert_eq!(out, "[1] (x) faq-unknown");
    }

    #[test]
    fn format_retrieved_context_numbers_multiple_hits() {
        let texts = HashMap::new();
        let hits = vec![
            QueryHit {
                rank: 1,
                distance: 1,
                id: "a".to_string(),
                event: "e1".to_string(),
            },
            QueryHit {
                rank: 2,
                distance: 2,
                id: "b".to_string(),
                event: "e2".to_string(),
            },
        ];
        let out = format_retrieved_context(&hits, &texts);
        assert_eq!(out, "[1] (e1) a\n[2] (e2) b");
    }

    #[test]
    fn chat_request_deserializes_with_unknown_fields() {
        // OpenAI clients send temperature, stream, max_tokens, etc. We must
        // ignore them rather than erroring.
        let body = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "stream": false,
            "max_tokens": 256,
            "user": "session-123"
        }"#;
        let req: ChatCompletionsRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.user.as_deref(), Some("session-123"));
        assert_eq!(req.model.as_deref(), Some("gpt-4o"));
    }
}
