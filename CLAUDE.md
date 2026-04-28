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

## Regenerate typegens

After any change to a binding (`blazen-py`, `blazen-node`, or `blazen-wasm-sdk`), regenerate ALL THREE typegens and commit drift:

```bash
# Python — regenerates crates/blazen-py/blazen.pyi
# NOTE: --features flags are required so feature-gated bindings (langfuse, otlp,
# prometheus, tract embedder, distributed peer types) actually appear in the stub.
# Without them, those symbols silently disappear from blazen.pyi.
cargo run --example stub_gen -p blazen-py --features langfuse,otlp,prometheus,tract,distributed

# Node — regenerates crates/blazen-node/index.d.ts (and runs the post-build error-classes shim)
pnpm --filter blazen run build

# WASM-SDK — regenerates crates/blazen-wasm-sdk/pkg/blazen_wasm_sdk.d.ts
wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release
```

CI's `audit-bindings` job runs all three regens and fails on drift, so committing stale typegens will block merges.

## Test

```bash
cargo nextest run --workspace --all-features
cargo test --workspace --doc --all-features
```

## Workspace lints

Clippy `all` + `pedantic` are enabled at warn level in root `Cargo.toml` under `[workspace.lints.clippy]`. CI runs with `-D warnings` (deny), so any warning is a hard error.

## Excluded crates

`blazen-wasm` and `blazen-wasm-sdk` are excluded from the workspace and built separately.
