# Blazen

Rust workspace with 14 crates (LLM orchestration framework) + Python (PyO3) and Node.js (napi-rs) bindings.

## Setup

```bash
git config core.hooksPath .githooks
```

## Lint (MUST run before every commit)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-features -- -D warnings
```

ALWAYS use `--workspace --all-features` for clippy. Never run clippy on a single crate — lint the entire workspace every time.

## Build

```bash
cargo build --workspace --all-features
```

## Test

```bash
cargo nextest run --workspace --all-features
cargo test --workspace --doc --all-features
```

## Workspace lints

Clippy `all` + `pedantic` are enabled at warn level in root `Cargo.toml` under `[workspace.lints.clippy]`. CI runs with `-D warnings` (deny), so any warning is a hard error.

## Excluded crates

`blazen-wasm` and `blazen-wasm-sdk` are excluded from the workspace and built separately.
