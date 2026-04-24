.PHONY: build test clippy fmt bench ci clean

# Enable AVX-512 for local dev if CPU supports it
RUSTFLAGS ?= -C target-cpu=native

build:
	RUSTFLAGS="$(RUSTFLAGS)" cargo build --workspace --release

test:
	RUSTFLAGS="$(RUSTFLAGS)" cargo test --workspace

clippy:
	cargo clippy --workspace -- -D warnings

fmt:
	cargo fmt --all -- --check

bench:
	RUSTFLAGS="$(RUSTFLAGS)" cargo bench --package primd-bench

ci: fmt clippy test

clean:
	cargo clean
