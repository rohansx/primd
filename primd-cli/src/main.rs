mod cmd_index;
mod cmd_query;
mod cmd_serve;
mod cmd_train;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "primd",
    version,
    about = "Sub-millisecond predictive retrieval runtime for voice AI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a signature index from a JSONL corpus.
    Index(cmd_index::IndexArgs),

    /// Query an existing index from text.
    Query(cmd_query::QueryArgs),

    /// Serve an index over HTTP for use as a retrieval microservice.
    Serve(cmd_serve::ServeArgs),

    /// Train a transition matrix from transcript JSONL.
    Train(cmd_train::TrainArgs),
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Index(args) => cmd_index::run(args),
        Command::Query(args) => cmd_query::run(args),
        Command::Serve(args) => cmd_serve::run(args),
        Command::Train(args) => cmd_train::run(args),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
