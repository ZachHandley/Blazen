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
cargo clippy --workspace --all-features --tests -- -D warnings
```

ALWAYS use `--workspace --all-features` for clippy. Never run clippy on a single crate — lint the entire workspace every time. Both clippy invocations (with and without `--tests`) must be clean — CI runs both and `--tests` catches lints that only fire under the test profile.

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

The Rust workspace is only one of four test surfaces. ALL FOUR must pass before pushing:

```bash
# 1. Rust workspace (922+ tests in 14 crates)
cargo nextest run --workspace --all-features
cargo test --workspace --doc --all-features

# 2. Python bindings (blazen-py)
# Rebuild the extension first — pytest tests against the installed .so, not source.
# Without this, you'll see stale-binary failures even if the source is correct.
# --features must match the stub generator's set so feature-gated symbols compile in.
uv run --no-sync maturin develop -m crates/blazen-py/Cargo.toml \
    --features langfuse,otlp,prometheus,tract,distributed --release
uv run --no-sync pytest tests/python/ -p no:xdist     # serial — clearer tracebacks on first failure
uv run --no-sync pytest tests/python/ -n auto         # xdist — confirms no parallel-mode regressions

# 3. Node bindings (blazen-node)
# `pnpm --filter blazen run build` rebuilds the napi binary AND regenerates index.d.ts.
# Sync smoke check (this MUST work — confirms the binary loads + new bindings export):
#   node -e "const b = require('./crates/blazen-node/index.js'); console.log(b.version(), typeof b.ModelManager, typeof b.ModelRegistry);"
# Full async suite (`node --test tests/node/*.mjs`) currently hangs because the
# tests use `async () => {…}` and napi-rs's tokio runtime handle keeps the
# event loop alive past the test end — known issue, not the binding's fault.
# Convert tests to sync (Promise.then chains) before relying on this for CI.
pnpm --filter blazen run build
node -e "const b = require('./crates/blazen-node/index.js'); console.log('node binding ok:', b.version(), typeof b.ModelManager, typeof b.ModelRegistry);"

# 4. WASM SDK (blazen-wasm-sdk) — wasm-bindgen-test based
# Headless browser run; firefox or chrome must be installed.
wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release
wasm-pack test crates/blazen-wasm-sdk --headless --firefox     # or --chrome
```

ALWAYS use `uv run --no-sync` (NOT plain `uv run`) for Python after `maturin develop` — plain `uv run` re-syncs the venv and replaces the freshly-built wheel with a stale cache, silently undoing your build.

Skips are expected on optional surfaces (no API keys / no GPU / no models cached). Failures are not.

## Workspace lints

Clippy `all` + `pedantic` are enabled at warn level in root `Cargo.toml` under `[workspace.lints.clippy]`. CI runs with `-D warnings` (deny), so any warning is a hard error.

## Excluded crates

`blazen-wasm` and `blazen-wasm-sdk` are excluded from the workspace and built separately.
