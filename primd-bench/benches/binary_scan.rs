use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use primd_core::embed::binary::BinarySignature;
use primd_core::index::signatures::SignatureIndex;
use rand::Rng;
use std::time::Duration;

fn random_sigs(n: usize) -> Vec<BinarySignature> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            let mut bytes = [0u8; 32];
            rng.fill(&mut bytes);
            BinarySignature(bytes)
        })
        .collect()
}

fn random_query() -> BinarySignature {
    let mut rng = rand::rng();
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    BinarySignature(bytes)
}

fn bench_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_scan_sequential");
    group.warm_up_time(Duration::from_secs(2));

    for &size in &[10_000, 100_000, 500_000, 1_000_000] {
        let sigs = random_sigs(size);
        let idx = SignatureIndex::new(sigs);
        let query = random_query();

        group.sample_size(if size >= 500_000 { 30 } else { 50 });
        group.measurement_time(if size >= 500_000 {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(5)
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| idx.scan_top_k(&query, 10));
        });
    }
    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_scan_parallel");
    group.warm_up_time(Duration::from_secs(2));

    for &size in &[10_000, 100_000, 500_000, 1_000_000] {
        let sigs = random_sigs(size);
        let idx = SignatureIndex::new(sigs);
        let query = random_query();

        group.sample_size(if size >= 500_000 { 30 } else { 50 });
        group.measurement_time(if size >= 500_000 {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(5)
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| idx.scan_top_k_parallel(&query, 10));
        });
    }
    group.finish();
}

fn bench_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_scan_varying_k");
    group.warm_up_time(Duration::from_secs(2));
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));

    let sigs = random_sigs(1_000_000);
    let idx = SignatureIndex::new(sigs);
    let query = random_query();

    for &k in &[1, 10, 50, 100, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| idx.scan_top_k(&query, k));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sequential, bench_parallel, bench_varying_k);
criterion_main!(benches);
