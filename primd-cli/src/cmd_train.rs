use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Args;
use serde::Deserialize;

use primd_core::predict::{EventId, trainer::train_sequences};

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Transcript JSONL. Supported formats:
    /// {"events": ["pricing", "trial", ...]}
    /// {"session": "abc", "event": "pricing"}
    #[arg(long)]
    pub transcripts: PathBuf,

    /// Existing corpus/index directory containing manifest.json.
    #[arg(long)]
    pub corpus: PathBuf,

    /// Output predictor artifact.
    #[arg(long)]
    pub out: Option<PathBuf>,

    #[arg(long, default_value_t = 3)]
    pub max_order: usize,

    #[arg(long, default_value_t = 0.01)]
    pub smoothing: f32,
}

#[derive(Deserialize)]
struct Manifest {
    scope: BTreeMap<String, Vec<usize>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TranscriptLine {
    Sequence { events: Vec<String> },
    Turn { session: String, event: String },
}

pub fn run(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = args.corpus.join("manifest.json");
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;

    let mut event_ids = BTreeMap::new();
    for (i, name) in manifest.scope.keys().enumerate() {
        event_ids.insert(name.clone(), EventId(i as u32));
    }

    let file = std::fs::File::open(&args.transcripts)?;
    let reader = BufReader::new(file);
    let mut sequences: Vec<Vec<EventId>> = Vec::new();
    let mut sessions: HashMap<String, Vec<EventId>> = HashMap::new();

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: TranscriptLine =
            serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", lineno + 1))?;
        match parsed {
            TranscriptLine::Sequence { events } => {
                let seq: Vec<EventId> = events
                    .into_iter()
                    .filter_map(|name| event_ids.get(&name).copied())
                    .collect();
                if seq.len() >= 2 {
                    sequences.push(seq);
                }
            }
            TranscriptLine::Turn { session, event } => {
                if let Some(id) = event_ids.get(&event).copied() {
                    sessions.entry(session).or_default().push(id);
                }
            }
        }
    }

    sequences.extend(sessions.into_values().filter(|seq| seq.len() >= 2));
    if sequences.is_empty() {
        return Err("no valid event sequences found in transcripts".into());
    }

    let predictor = train_sequences(sequences, args.max_order, args.smoothing);
    let out = args
        .out
        .unwrap_or_else(|| args.corpus.join("transitions.json"));
    predictor.save_to_file(&out)?;
    eprintln!("wrote predictor to {}", out.display());
    Ok(())
}
