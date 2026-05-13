use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

use primd_core::embed::{
    EmbeddingPipeline, HashedEmbedder, LocalEmbedder, LocalModel, OpenAIEmbedder, random_projection,
};
use primd_core::index::signatures::SignatureIndex;
use primd_core::{BinarySignature, PrimdError};

const LOCAL_PROJECTION_SEED: u64 = 0xA17F_C0DE_B007_5EED;

#[derive(Args, Debug)]
pub struct IndexArgs {
    /// Input JSONL with `{"id": "...", "text": "...", "event": "optional"}` per line.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory. Created if it does not exist.
    #[arg(short, long)]
    pub out: PathBuf,

    /// Embedder backend.
    #[arg(long, value_enum, default_value_t = EmbedderKind::Hashed)]
    pub embedder: EmbedderKind,

    /// Embedding dimension. 256 = direct quantize (no PCA needed).
    #[arg(long, default_value_t = 256)]
    pub dim: usize,

    /// Disable bigram features in the hashed embedder.
    #[arg(long)]
    pub no_bigrams: bool,

    /// OpenAI model name (default `text-embedding-3-small`).
    #[arg(long)]
    pub openai_model: Option<String>,

    /// Local model when `--embedder local` is selected.
    #[arg(long, value_enum, default_value_t = LocalModelKind::BgeSmallEn)]
    pub local_model: LocalModelKind,
}

#[derive(Copy, Clone, Debug, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LocalModelKind {
    AllMinilm,
    BgeSmallEn,
    BgeBaseEn,
}

impl LocalModelKind {
    pub fn into_local(self) -> LocalModel {
        match self {
            LocalModelKind::AllMinilm => LocalModel::AllMiniLM,
            LocalModelKind::BgeSmallEn => LocalModel::BgeSmallEn,
            LocalModelKind::BgeBaseEn => LocalModel::BgeBaseEn,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbedderKind {
    /// Deterministic feature-hashing baseline. No network, no model.
    Hashed,
    /// OpenAI text embeddings (`text-embedding-3-small` by default).
    /// Requires `OPENAI_API_KEY` in the environment.
    Openai,
    /// Local sentence-transformer model via fastembed (downloads on first run).
    Local,
}

#[derive(Deserialize, Debug)]
struct CorpusEntry {
    id: String,
    text: String,
    #[serde(default)]
    event: Option<String>,
}

#[derive(Serialize)]
struct Manifest {
    embedder: EmbedderKind,
    dim: usize,
    use_bigrams: bool,
    openai_model: Option<String>,
    local_model: Option<LocalModelKind>,
    projection_seed: Option<u64>,
    source_dim: Option<usize>,
    n_entries: usize,
    ids: Vec<String>,
    /// Original document text per entry, parallel to `ids`. Lets `primd serve`
    /// return real content (e.g. via the OpenAI-compatible `/v1/chat/completions`
    /// MoshiRAG adapter) instead of just IDs. Optional for backward compat.
    texts: Vec<String>,
    scope: BTreeMap<String, Vec<usize>>,
}

pub fn run(args: IndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.dim != 256 {
        return Err(format!(
            "dim={} not yet supported; only 256 (direct quantize) is wired up",
            args.dim
        )
        .into());
    }

    std::fs::create_dir_all(&args.out)?;
    let entries = read_jsonl(&args.input)?;

    let texts: Vec<&str> = entries.iter().map(|e| e.text.as_str()).collect();

    let started = Instant::now();
    let signatures = match args.embedder {
        EmbedderKind::Hashed => embed_hashed(args.dim, !args.no_bigrams, &texts)?,
        EmbedderKind::Openai => embed_openai(args.dim, args.openai_model.as_deref(), &texts)?,
        EmbedderKind::Local => embed_local(args.local_model, &texts)?,
    };
    let elapsed = started.elapsed();

    let throughput = entries.len() as f64 / elapsed.as_secs_f64();
    eprintln!(
        "indexed {} entries via {:?} in {:.1}ms ({:.0} docs/s)",
        entries.len(),
        args.embedder,
        elapsed.as_secs_f64() * 1000.0,
        throughput,
    );

    let mut scope: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, entry) in entries.iter().enumerate() {
        let key = entry
            .event
            .clone()
            .unwrap_or_else(|| "_default".to_string());
        scope.entry(key).or_default().push(i);
    }

    let sigs_path = args.out.join("signatures.bin");
    SignatureIndex::new(signatures).write_to_file(&sigs_path)?;

    let (local_model, projection_seed, source_dim) = match args.embedder {
        EmbedderKind::Local => (
            Some(args.local_model),
            Some(LOCAL_PROJECTION_SEED),
            Some(args.local_model.into_local().dim()),
        ),
        _ => (None, None, None),
    };

    let manifest = Manifest {
        embedder: args.embedder,
        dim: args.dim,
        use_bigrams: !args.no_bigrams,
        openai_model: args.openai_model,
        local_model,
        projection_seed,
        source_dim,
        n_entries: entries.len(),
        ids: entries.iter().map(|e| e.id.clone()).collect(),
        texts: entries.iter().map(|e| e.text.clone()).collect(),
        scope,
    };
    let manifest_path = args.out.join("manifest.json");
    let manifest_str = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_str)?;

    eprintln!(
        "wrote {} and {}",
        sigs_path.display(),
        manifest_path.display()
    );
    Ok(())
}

fn read_jsonl(path: &PathBuf) -> Result<Vec<CorpusEntry>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries: Vec<CorpusEntry> = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: CorpusEntry =
            serde_json::from_str(&line).map_err(|e| format!("line {}: {}", lineno + 1, e))?;
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err("input contained no entries".into());
    }
    Ok(entries)
}

fn embed_hashed(
    dim: usize,
    use_bigrams: bool,
    texts: &[&str],
) -> Result<Vec<BinarySignature>, PrimdError> {
    let mut e = HashedEmbedder::new(dim);
    if !use_bigrams {
        e = e.without_bigrams();
    }
    let pipe = EmbeddingPipeline::new_direct(e)?;
    pipe.embed_batch_to_signatures(texts)
}

fn embed_openai(
    dim: usize,
    model: Option<&str>,
    texts: &[&str],
) -> Result<Vec<BinarySignature>, PrimdError> {
    let mut e = OpenAIEmbedder::from_env()?.with_dim(dim);
    if let Some(m) = model {
        e = e.with_model(m);
    }
    let pipe = EmbeddingPipeline::new_direct(e)?;
    const BATCH: usize = 256;
    let mut out: Vec<BinarySignature> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH) {
        let part = pipe.embed_batch_to_signatures(chunk)?;
        out.extend(part);
    }
    Ok(out)
}

fn embed_local(kind: LocalModelKind, texts: &[&str]) -> Result<Vec<BinarySignature>, PrimdError> {
    let local = LocalEmbedder::new(kind.into_local())?;
    let proj = random_projection(LOCAL_PROJECTION_SEED, kind.into_local().dim());
    let pipe = EmbeddingPipeline::new_with_pca(local, proj)?;
    // fastembed handles its own internal batching but chunking guards memory.
    const BATCH: usize = 64;
    let mut out: Vec<BinarySignature> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH) {
        let part = pipe.embed_batch_to_signatures(chunk)?;
        out.extend(part);
    }
    Ok(out)
}
