use std::collections::BTreeMap;
use std::io::{Read, stdin};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use serde::Deserialize;

use primd_core::embed::{EmbeddingPipeline, HashedEmbedder};
use primd_core::index::signatures::SignatureIndex;

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
    embedder: String,
    dim: usize,
    use_bigrams: bool,
    n_entries: usize,
    ids: Vec<String>,
    #[serde(default)]
    scope: BTreeMap<String, Vec<usize>>,
}

pub fn run(args: QueryArgs) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = args.index.join("manifest.json");
    let sigs_path = args.index.join("signatures.bin");

    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    if manifest.embedder != "hashed" {
        return Err(format!(
            "this build only knows how to load 'hashed' indexes; manifest says '{}'",
            manifest.embedder
        )
        .into());
    }

    // Reconstruct the same embedder used at index time.
    let mut embedder = HashedEmbedder::new(manifest.dim);
    if !manifest.use_bigrams {
        embedder = embedder.without_bigrams();
    }
    let pipeline = EmbeddingPipeline::new_direct(embedder)?;
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
    let query_sig = pipeline.embed_to_signature(&query_text)?;
    let embed_elapsed = embed_start.elapsed();

    let scan_start = Instant::now();
    let results = if args.parallel {
        index.scan_top_k_parallel(&query_sig, args.top)
    } else {
        index.scan_top_k(&query_sig, args.top)
    };
    let scan_elapsed = scan_start.elapsed();

    eprintln!(
        "embed: {:.0}us | scan: {:.0}us | corpus: {} sigs",
        embed_elapsed.as_secs_f64() * 1_000_000.0,
        scan_elapsed.as_secs_f64() * 1_000_000.0,
        index.len()
    );

    if results.is_empty() {
        println!("(no results)");
        return Ok(());
    }

    // Reverse-lookup: corpus index → event name (if any).
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
