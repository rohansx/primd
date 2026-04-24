fn main() {
    let version = env!("CARGO_PKG_VERSION");
    println!("primd v{version}");
    println!();
    println!("sub-millisecond predictive retrieval for voice AI");
    println!();
    println!("commands (coming soon):");
    println!("  primd index   — index a corpus");
    println!("  primd train   — train transition matrix");
    println!("  primd serve   — serve as HTTP endpoint");
    println!("  primd bench   — run benchmarks");
    println!("  primd check   — check index quality");
}
