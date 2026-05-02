use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use serde::{Deserialize, Serialize};

use primd_core::embed::{Embedder, EmbeddingPipeline, HashedEmbedder};
use primd_core::index::signatures::SignatureIndex;

#[derive(Args, Debug)]
pub struct IndexArgs {
    /// Input JSONL with `{"id": "...", "text": "...", "event": "optional"}` per line.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory. Created if it does not exist.
    #[arg(short, long)]
    pub out: PathBuf,

    /// Embedding dimension. 256 = direct quantize (no PCA needed). Other
    /// values are accepted and would be used with a PCA projector down the line.
    #[arg(long, default_value_t = 256)]
    pub dim: usize,

    /// Disable bigram features in the hashed embedder.
    #[arg(long)]
    pub no_bigrams: bool,
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
    embedder: String,
    dim: usize,
    use_bigrams: bool,
    n_entries: usize,
    ids: Vec<String>,
    /// Map: event name → list of corpus indices belonging to that event. Used
    /// by the prefetch coordinator to narrow the search scope per event.
    scope: BTreeMap<String, Vec<usize>>,
}

pub fn run(args: IndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.dim != 256 {
        return Err(format!(
            "dim={} not yet supported; only 256 (direct quantize) is wired up. \
             Higher dims need a fitted PcaProjector — coming next iteration.",
            args.dim
        )
        .into());
    }

    std::fs::create_dir_all(&args.out)?;

    let file = File::open(&args.input)?;
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

    let mut embedder = HashedEmbedder::new(args.dim);
    if args.no_bigrams {
        embedder = embedder.without_bigrams();
    }
    let pipeline = EmbeddingPipeline::new_direct(embedder)?;

    let texts: Vec<&str> = entries.iter().map(|e| e.text.as_str()).collect();

    let started = Instant::now();
    let signatures = pipeline.embed_batch_to_signatures(&texts)?;
    let elapsed = started.elapsed();

    let throughput = entries.len() as f64 / elapsed.as_secs_f64();
    let dim = pipeline.embedder().dim();
    eprintln!(
        "indexed {} entries in {:.2}ms ({:.0} docs/s, dim={})",
        entries.len(),
        elapsed.as_secs_f64() * 1000.0,
        throughput,
        dim
    );

    // Group corpus indices by event for layer 3 prefetch scopes.
    let mut scope: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, entry) in entries.iter().enumerate() {
        let key = entry
            .event
            .clone()
            .unwrap_or_else(|| "_default".to_string());
        scope.entry(key).or_default().push(i);
    }

    // Write outputs.
    let sigs_path = args.out.join("signatures.bin");
    SignatureIndex::new(signatures).write_to_file(&sigs_path)?;

    let manifest = Manifest {
        embedder: "hashed".to_string(),
        dim: args.dim,
        use_bigrams: !args.no_bigrams,
        n_entries: entries.len(),
        ids: entries.iter().map(|e| e.id.clone()).collect(),
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
