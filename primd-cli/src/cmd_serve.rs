use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use clap::Args;
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use primd_core::embed::{
    EmbeddingPipeline, HashedEmbedder, LocalEmbedder, OpenAIEmbedder, random_projection,
};
use primd_core::index::signatures::SignatureIndex;
use primd_core::{BinarySignature, PrimdError};

use crate::cmd_index::{EmbedderKind, LocalModelKind};

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Index directory built by `primd index`.
    #[arg(short, long)]
    pub index: PathBuf,

    /// Bind address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
}

#[derive(Deserialize)]
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
    index: SignatureIndex,
    event_for: BTreeMap<usize, String>,
    pipeline: Box<dyn QueryEmbedder>,
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
}

pub fn run(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = args.index.join("manifest.json");
    let sigs_path = args.index.join("signatures.bin");

    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    let index = SignatureIndex::from_file(&sigs_path)?;
    if index.len() != manifest.n_entries {
        return Err(format!(
            "manifest claims {} entries; signatures file has {}",
            manifest.n_entries,
            index.len()
        )
        .into());
    }

    let mut event_for: BTreeMap<usize, String> = BTreeMap::new();
    for (event_name, indices) in &manifest.scope {
        for &i in indices {
            event_for.insert(i, event_name.clone());
        }
    }

    eprintln!("loading embedder {:?}…", manifest.embedder);
    let pipeline = build_pipeline(&manifest)?;

    let state = Arc::new(State {
        manifest,
        index,
        event_for,
        pipeline,
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
    eprintln!("endpoints:  POST /query  GET /health  GET /stats");

    for mut request in server.incoming_requests() {
        let path = request.url().to_string();
        let method = request.method().clone();
        let state = state.clone();

        let response_result = match (method, path.as_str()) {
            (Method::Get, "/health") => Ok(json_response(200, "{\"status\":\"ok\"}")),
            (Method::Get, "/stats") => stats_response(&state),
            (Method::Post, "/query") => {
                let mut body = String::new();
                if let Err(e) = request.as_reader().read_to_string(&mut body) {
                    Err(format!("read body: {e}"))
                } else {
                    handle_query(&state, &body)
                }
            }
            _ => Ok(json_response(404, "{\"error\":\"not found\"}")),
        };

        let response = match response_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("request error: {e}");
                let body = format!("{{\"error\":\"{}\"}}", json_escape(&e));
                json_response(400, body)
            }
        };

        if let Err(e) = request.respond(response) {
            eprintln!("respond error: {e}");
        }
    }

    Ok(())
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
    let raw = if req.parallel {
        state.index.scan_top_k_parallel(&query_sig, req.top_k)
    } else {
        state.index.scan_top_k(&query_sig, req.top_k)
    };
    let scan_elapsed = scan_start.elapsed();

    state.record_latency(embed_elapsed.as_nanos(), scan_elapsed.as_nanos());

    let hits: Vec<QueryHit> = raw
        .into_iter()
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
        .collect();

    let response = QueryResponse {
        embedder: state.manifest.embedder,
        embed_us: embed_elapsed.as_micros(),
        scan_us: scan_elapsed.as_micros(),
        corpus_size: state.index.len(),
        hits,
    };
    Ok(json_response(
        200,
        serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
    ))
}

fn stats_response(state: &State) -> Result<Response<std::io::Cursor<Vec<u8>>>, String> {
    let queries = state.queries_served.load(Ordering::Relaxed);
    let total_embed = state.total_embed_ns.load(Ordering::Relaxed) as u128;
    let total_scan = state.total_scan_ns.load(Ordering::Relaxed) as u128;
    let denom = queries.max(1) as u128;
    let body = StatsResponse {
        corpus_size: state.index.len(),
        embedder: state.manifest.embedder,
        queries_served: queries,
        total_embed_us: total_embed / 1000,
        total_scan_us: total_scan / 1000,
        mean_embed_us: (total_embed / 1000) / denom,
        mean_scan_us: (total_scan / 1000) / denom,
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
    let body = body.into();
    Response::from_string(body)
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
