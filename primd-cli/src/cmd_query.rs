use std::collections::BTreeMap;
use std::io::{Read, stdin};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use serde::Deserialize;

use primd_core::embed::{EmbeddingPipeline, HashedEmbedder, OpenAIEmbedder};
use primd_core::index::signatures::SignatureIndex;
use primd_core::{BinarySignature, PrimdError};

use crate::cmd_index::EmbedderKind;

#[derive(Args, Debug)]
pub struct QueryArgs {
    /// Path to an index directory produced by `primd index`.
    #[arg(short, long)]
    pub index: PathBuf,

    /// Query text. If omitted, reads from stdin.
    #[arg(short, long)]
    pub text: Option<String>,

    /// Number of results to return.
    #[arg(short = 'k', long, default_value_t = 5)]
    pub top: usize,

    /// Use parallel scan (rayon over corpus chunks).
    #[arg(long)]
    pub parallel: bool,
}

#[derive(Deserialize)]
struct Manifest {
    embedder: EmbedderKind,
    dim: usize,
    use_bigrams: bool,
    #[serde(default)]
    openai_model: Option<String>,
    n_entries: usize,
    ids: Vec<String>,
    #[serde(default)]
    scope: BTreeMap<String, Vec<usize>>,
}

pub fn run(args: QueryArgs) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = args.index.join("manifest.json");
    let sigs_path = args.index.join("signatures.bin");

    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;

    let index = SignatureIndex::from_file(&sigs_path)?;
    if index.len() != manifest.n_entries {
        return Err(format!(
            "manifest says {} entries but signatures.bin has {}",
            manifest.n_entries,
            index.len()
        )
        .into());
    }

    let query_text = match args.text {
        Some(t) => t,
        None => {
            let mut buf = String::new();
            stdin().read_to_string(&mut buf)?;
            buf.trim().to_string()
        }
    };
    if query_text.is_empty() {
        return Err("no query text provided (use --text or pipe to stdin)".into());
    }

    let embed_start = Instant::now();
    let query_sig = embed_query(&manifest, &query_text)?;
    let embed_elapsed = embed_start.elapsed();

    let scan_start = Instant::now();
    let results = if args.parallel {
        index.scan_top_k_parallel(&query_sig, args.top)
    } else {
        index.scan_top_k(&query_sig, args.top)
    };
    let scan_elapsed = scan_start.elapsed();

    eprintln!(
        "embedder: {:?} | embed: {:.0}us | scan: {:.0}us | corpus: {} sigs",
        manifest.embedder,
        embed_elapsed.as_secs_f64() * 1_000_000.0,
        scan_elapsed.as_secs_f64() * 1_000_000.0,
        index.len()
    );

    if results.is_empty() {
        println!("(no results)");
        return Ok(());
    }

    let mut event_for: BTreeMap<usize, String> = BTreeMap::new();
    for (event_name, indices) in &manifest.scope {
        for &i in indices {
            event_for.insert(i, event_name.clone());
        }
    }

    println!("rank  dist  id                    event");
    for (rank, (dist, idx)) in results.iter().enumerate() {
        let id = manifest.ids.get(*idx).map(|s| s.as_str()).unwrap_or("?");
        let event = event_for.get(idx).map(|s| s.as_str()).unwrap_or("_default");
        println!("{:>4}  {:>4}  {:<20}  {}", rank + 1, dist, id, event);
    }

    Ok(())
}

fn embed_query(manifest: &Manifest, text: &str) -> Result<BinarySignature, PrimdError> {
    match manifest.embedder {
        EmbedderKind::Hashed => {
            let mut e = HashedEmbedder::new(manifest.dim);
            if !manifest.use_bigrams {
                e = e.without_bigrams();
            }
            let pipe = EmbeddingPipeline::new_direct(e)?;
            pipe.embed_to_signature(text)
        }
        EmbedderKind::Openai => {
            let mut e = OpenAIEmbedder::from_env()?.with_dim(manifest.dim);
            if let Some(m) = &manifest.openai_model {
                e = e.with_model(m);
            }
            let pipe = EmbeddingPipeline::new_direct(e)?;
            pipe.embed_to_signature(text)
        }
    }
}
