# Blazen

Rust workspace with 14 crates (LLM orchestration framework) + Python (PyO3), Node.js (napi-rs), WASM (wasm-bindgen), and UniFFI bindings for Go, Swift, and Kotlin. Ruby UniFFI binding is in progress via a cbindgen-generated C ABI (see `bindings/ruby/`).

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

After any change to `crates/blazen-uniffi/` (the Go/Swift/Kotlin/Ruby UniFFI surface), regenerate the foreign-language bindings:

```bash
# Rebuild the release rlib (carries metadata for library-mode bindgen)
cargo build -p blazen-uniffi --release
# Regenerate all four FFI files (Go via uniffi-bindgen-go, Swift/Kotlin/Ruby via local uniffi-bindgen)
./scripts/regen-bindings.sh
```

The same `audit-bindings` job runs this script and fails on drift.

To rebuild the prebuilt native libs that the Go/Kotlin/Ruby bindings link against (linux amd64/arm64 today; macOS and Windows handled by CI release runs):

```bash
./scripts/build-uniffi-lib.sh
```

## Test

The Rust workspace is only one of seven test surfaces. ALL SEVEN must pass before pushing:

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

# 3. Node bindings (blazen-node) — uses ava (canonical napi-rs runner)
# Build the napi binary + regenerate index.d.ts:
pnpm --filter blazen run build
# Run the test suite from repo root (ava config is in root package.json):
pnpm exec ava --timeout 30s
# Or a single file: `pnpm exec ava tests/node/test_workflow.mjs`.

# 4. WASM SDK (blazen-wasm-sdk) — wasm-bindgen-test based
# Headless browser run; firefox or chrome must be installed.
wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release
wasm-pack test crates/blazen-wasm-sdk --headless --firefox     # or --chrome

# 5. Go binding (bindings/go) — requires cgo (gcc/clang) on host
# Rebuild the static lib and copy into internal/clib/ before testing on a fresh
# checkout. ./scripts/build-uniffi-lib.sh does both.
./scripts/build-uniffi-lib.sh linux_amd64
cd bindings/go && go vet ./... && go build ./... && go test ./...

# 6. Swift binding (bindings/swift) — requires Swift 5.10+ (swiftlang/swift container on Linux)
cd bindings/swift && swift build && swift test

# 7. Kotlin binding (bindings/kotlin) — requires JDK 17+ and Gradle
# Ensure the shared lib is at src/main/resources/<jna-platform>/ first.
./scripts/build-uniffi-lib.sh linux_amd64
cd bindings/kotlin && gradle test --no-daemon
```

ALWAYS use `uv run --no-sync` (NOT plain `uv run`) for Python after `maturin develop` — plain `uv run` re-syncs the venv and replaces the freshly-built wheel with a stale cache, silently undoing your build.

Skips are expected on optional surfaces (no API keys / no GPU / no models cached). Failures are not.

## Workspace lints

Clippy `all` + `pedantic` are enabled at warn level in root `Cargo.toml` under `[workspace.lints.clippy]`. CI runs with `-D warnings` (deny), so any warning is a hard error.

## Excluded crates

`blazen-wasm` and `blazen-wasm-sdk` are excluded from the workspace and built separately.
