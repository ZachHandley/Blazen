# Blazen — local CI mirror.
#
# Targets here run the EXACT command sequences from
# `.forgejo/workflows/ci.yaml` so that "green locally → green in CI" is
# a real guarantee, not a hope. Every public target is annotated with a
# `## name  Description` line; `make help` greps them.
#
# Quick reference:
#   make help          — list every target with a description
#   make ci-fast       — pre-push gate (no API keys, no Go/Swift/Kotlin/Ruby) ~10 min
#   make ci-bindings   — cross-language bindings test pass (~10 min)
#   make ci-smoke      — API-key-required smoke surface (costs $$)
#   make ci            — full ci.yaml mirror = ci-fast + ci-bindings + ci-smoke
#
#   make audit-bindings — regenerate all typegens + drift-check (matches
#                          CI's audit-bindings job)
#   make regen-all     — same regens, no drift check (use after edits)
#
# Override knobs:
#   TMPDIR             — default $(HOME)/.cache/blazen-tmp
#   CARGO_TARGET_DIR   — default target

SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

# Per-Blazen scratch, never /tmp. First target to touch it mkdir -p's it.
export TMPDIR ?= $(HOME)/.cache/blazen-tmp

# Cargo target dir. CI sets this; mirror it locally so warm caches share
# between `make test-rust` and `make test-go` (which both rebuild
# blazen-uniffi).
export CARGO_TARGET_DIR ?= target

# Local-machine UniFFI artifact triple. linux_amd64 is the only CI
# runner so this matches `.forgejo/workflows/ci.yaml`. Override if you
# need to stage for a different host arch.
UNIFFI_HOST_TRIPLE ?= linux_amd64
KOTLIN_JNA_DIR ?= linux-x86-64

# Default goal: show help.
.DEFAULT_GOAL := help

.PHONY: help \
        fmt fmt-check lint \
        test-rust test-doc \
        test-python test-python-smoke \
        build-node test-node test-node-smoke \
        test-go test-swift test-kotlin test-ruby \
        test-wasm test-smoke-compute \
        regen-py-stubs regen-node-types regen-wasm-types \
        regen-uniffi regen-cabi regen-all \
        audit-bindings \
        build-uniffi-libs \
        prune-pypi \
        ci-fast ci-bindings ci-smoke ci \
        tmpdir-info clean \
        _ensure-tmpdir _require-fal-keys

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

## help               Show this help message
help:
	@echo "Blazen — local CI mirror"
	@echo ""
	@echo "Lint / format:"
	@echo "  make fmt              Format the workspace (cargo fmt --all)"
	@echo "  make fmt-check        Verify formatting (ci.yaml lint step 1)"
	@echo "  make lint             clippy --workspace --all-features [-D warnings] x2"
	@echo ""
	@echo "Rust tests:"
	@echo "  make test-rust        nextest --workspace --all-features (ci.yaml test step 1)"
	@echo "  make test-doc         doctests (ci.yaml test step 2)"
	@echo ""
	@echo "Python binding:"
	@echo "  make test-python      uv sync + parallel pytest (no API keys)"
	@echo "  make test-python-smoke pytest with real HTTP (needs FAL/OPENROUTER keys)"
	@echo ""
	@echo "Node binding:"
	@echo "  make build-node       corepack + pnpm install + napi build"
	@echo "  make test-node        ava free tests (no API keys)"
	@echo "  make test-node-smoke  ava llm + fal tests (needs FAL/OPENROUTER keys)"
	@echo ""
	@echo "Cross-language bindings:"
	@echo "  make test-go          uniffi.a + go vet/build/test"
	@echo "  make test-swift       uniffi + swift build/test"
	@echo "  make test-kotlin      uniffi.so + gradle build test"
	@echo "  make test-ruby        cabi.so + bundle install + rspec"
	@echo ""
	@echo "WebAssembly:"
	@echo "  make test-wasm        wasm32 cargo check + wasm-pack build/test + cf-worker"
	@echo "  make test-smoke-compute fal compute smoke (python + node, slow, needs keys)"
	@echo ""
	@echo "Regeneration (matches audit-bindings):"
	@echo "  make regen-py-stubs   blazen.pyi"
	@echo "  make regen-node-types index.d.ts"
	@echo "  make regen-wasm-types blazen_wasm_sdk.d.ts"
	@echo "  make regen-uniffi     Go/Swift/Kotlin UniFFI bindings"
	@echo "  make regen-cabi       blazen.h via cbindgen"
	@echo "  make regen-all        all of the above (no drift check)"
	@echo "  make audit-bindings   regen-all + git diff --exit-code + parity audit"
	@echo ""
	@echo "Aggregate gates (pre-push):"
	@echo "  make ci-fast          fmt+lint+rust+py+node+wasm+audit (~10 min, no keys)"
	@echo "  make ci-bindings      go+swift+kotlin+ruby (~10 min, toolchains needed)"
	@echo "  make ci-smoke         API-key smoke tests (costs fal/openrouter credit)"
	@echo "  make ci               ci-fast + ci-bindings + ci-smoke (full ci.yaml)"
	@echo ""
	@echo "Utilities:"
	@echo "  make build-uniffi-libs ./scripts/build-uniffi-lib.sh $(UNIFFI_HOST_TRIPLE)"
	@echo "  make tmpdir-info      show effective TMPDIR + free space"
	@echo "  make clean            cargo clean"
	@echo ""
	@echo "TMPDIR           = $$TMPDIR"
	@echo "CARGO_TARGET_DIR = $$CARGO_TARGET_DIR"

## tmpdir-info        Print effective TMPDIR + free-space report
tmpdir-info: _ensure-tmpdir
	@echo "TMPDIR = $$TMPDIR"
	@df -h "$$TMPDIR"

_ensure-tmpdir:
	@mkdir -p "$$TMPDIR"

_require-fal-keys:
	@if [ -z "$${FAL_API_KEY:-}" ] || [ -z "$${OPENROUTER_API_KEY:-}" ]; then \
	  echo ""; \
	  echo "ERROR: FAL_API_KEY and OPENROUTER_API_KEY must both be set for smoke tests." >&2; \
	  echo "       Set them in your shell or skip via 'make ci-fast' / 'make ci-bindings'." >&2; \
	  exit 1; \
	fi

## clean              cargo clean (removes $(CARGO_TARGET_DIR))
clean:
	cargo clean

# -----------------------------------------------------------------------------
# Lint / format (ci.yaml `lint` job)
# -----------------------------------------------------------------------------

## fmt                Format the workspace
fmt:
	cargo fmt --all

## fmt-check          Verify formatting (ci.yaml lint step 1)
fmt-check:
	cargo fmt --all -- --check

## lint               clippy --workspace --all-features (with + without --tests, per CLAUDE.md)
lint:
	cargo clippy --workspace --all-features -- -D warnings
	cargo clippy --workspace --all-features --tests -- -D warnings

# -----------------------------------------------------------------------------
# Rust workspace tests (ci.yaml `test` job)
# -----------------------------------------------------------------------------

## test-rust          nextest --workspace --all-features --exclude blazen-py (ci.yaml test step 1)
test-rust: _ensure-tmpdir
	cargo nextest run --workspace --all-features --exclude blazen-py

## test-doc           doctests --workspace --all-features --exclude blazen-py (ci.yaml test step 2)
test-doc: _ensure-tmpdir
	cargo test --workspace --doc --all-features --exclude blazen-py

# -----------------------------------------------------------------------------
# Python binding (ci.yaml `test-python` job)
# -----------------------------------------------------------------------------

## test-python        uv sync + parallel pytest free set (ci.yaml test-python steps 1-4)
test-python: _ensure-tmpdir
	uv sync --group dev
	# Drift check: regenerate stubs and fail if they changed.
	cargo run --example stub_gen -p blazen-py --features langfuse,otlp,prometheus,tract,distributed
	@if ! git diff --exit-code crates/blazen-py/blazen.pyi; then \
	  echo "::error::blazen.pyi is stale. Regenerate via 'make regen-py-stubs' and commit." >&2; \
	  exit 1; \
	fi
	uv run pytest -v -n auto --timeout=30 \
	  tests/python/test_workflow.py \
	  tests/python/test_session_refs.py \
	  tests/python/test_e2e.py \
	  tests/python/test_model_manager_smoke.py

## test-python-smoke  Real-HTTP pytest (requires FAL_API_KEY + OPENROUTER_API_KEY)
test-python-smoke: _ensure-tmpdir _require-fal-keys
	uv sync --group dev
	uv run pytest -v --timeout=300 \
	  tests/python/test_fal_smoke.py \
	  tests/python/test_llm_smoke.py \
	  tests/python/test_capabilities_smoke.py \
	  tests/python/test_provider_smoke.py

# -----------------------------------------------------------------------------
# Node binding (ci.yaml `test-node` job)
# -----------------------------------------------------------------------------

## build-node         corepack + pnpm install + napi build (shared)
build-node:
	corepack enable
	corepack prepare pnpm@10 --activate
	pnpm install
	pnpm --filter blazen run build

## test-node          ava free tests (ci.yaml test-node steps 1-3)
test-node: build-node
	pnpm exec ava tests/node/test_workflow.mjs
	pnpm exec ava \
	  tests/node/test_session_refs.mjs \
	  tests/node/test_capabilities_smoke.mjs \
	  tests/node/test_e2e.mjs

## test-node-smoke    ava llm + fal tests (requires FAL_API_KEY + OPENROUTER_API_KEY)
test-node-smoke: build-node _require-fal-keys
	pnpm exec ava tests/node/test_llm_smoke.mjs tests/node/test_fal_smoke.mjs

# -----------------------------------------------------------------------------
# Cross-language bindings (ci.yaml `test-{go,swift,kotlin,ruby}` jobs)
# -----------------------------------------------------------------------------

## test-go            uniffi staticlib + go vet/build/test (ci.yaml test-go)
test-go:
	cargo build -p blazen-uniffi --release
	mkdir -p bindings/go/internal/clib/linux_amd64
	cp $(CARGO_TARGET_DIR)/release/libblazen_uniffi.a bindings/go/internal/clib/linux_amd64/
	cd bindings/go && go vet ./... && go build ./... && go test ./...

## test-swift         uniffi shared lib + swift build/test (ci.yaml test-swift)
test-swift:
	cargo build -p blazen-uniffi --release
	cd bindings/swift && \
	  LD_LIBRARY_PATH="$(CURDIR)/$(CARGO_TARGET_DIR)/release:$${LD_LIBRARY_PATH:-}" \
	  swift build && \
	  LD_LIBRARY_PATH="$(CURDIR)/$(CARGO_TARGET_DIR)/release:$${LD_LIBRARY_PATH:-}" \
	  swift test

## test-kotlin        uniffi .so + gradle build test (ci.yaml test-kotlin)
test-kotlin:
	cargo build -p blazen-uniffi --release
	mkdir -p bindings/kotlin/src/main/resources/$(KOTLIN_JNA_DIR)
	cp $(CARGO_TARGET_DIR)/release/libblazen_uniffi.so \
	   bindings/kotlin/src/main/resources/$(KOTLIN_JNA_DIR)/
	cd bindings/kotlin && gradle build test --no-daemon

## test-ruby          cabi shared lib + bundle install + rspec (ci.yaml test-ruby)
test-ruby:
	cargo build -p blazen-cabi --release
	mkdir -p bindings/ruby/ext/blazen/linux_amd64
	cp $(CARGO_TARGET_DIR)/release/libblazen_cabi.so \
	   bindings/ruby/ext/blazen/linux_amd64/
	cd bindings/ruby && \
	  gem install bundler --no-document --conservative && \
	  bundle install --jobs 4 --retry 3 && \
	  bundle exec rspec --format documentation

# -----------------------------------------------------------------------------
# WASM (ci.yaml `test-wasm` job)
# -----------------------------------------------------------------------------

## test-wasm          wasm32 cargo check + wasm-pack build/test + cf-worker vitest
test-wasm:
	RUSTFLAGS="-D warnings" cargo check --target wasm32-unknown-unknown \
	  -p blazen-core --no-default-features
	RUSTFLAGS="-D warnings" cargo check --target wasm32-unknown-unknown \
	  -p blazen-memory --no-default-features
	RUSTFLAGS="-D warnings" cargo check --target wasm32-unknown-unknown \
	  -p blazen-llm --no-default-features --features tsify
	wasm-pack build crates/blazen-wasm-sdk --target web --release
	wasm-pack test --node --release crates/blazen-wasm-sdk
	cd crates/blazen-wasm-sdk && cargo clippy --target wasm32-unknown-unknown
	corepack enable
	corepack prepare pnpm@10 --activate
	cd examples/cloudflare-worker && \
	  pnpm install --ignore-workspace --frozen-lockfile && \
	  pnpm test

# -----------------------------------------------------------------------------
# Compute smoke (ci.yaml `test-smoke-compute` job)
# -----------------------------------------------------------------------------

## test-smoke-compute fal compute smoke; python + node with BLAZEN_TEST_FAL_COMPUTE=1
test-smoke-compute: _ensure-tmpdir _require-fal-keys build-node
	uv sync --group dev
	BLAZEN_TEST_FAL_COMPUTE=1 uv run pytest -v --timeout=1200 tests/python/test_fal_smoke.py
	BLAZEN_TEST_FAL_COMPUTE=1 pnpm exec ava --timeout=2100s tests/node/test_fal_smoke.mjs

# -----------------------------------------------------------------------------
# Regeneration (matches ci.yaml `audit-bindings` job steps individually)
# -----------------------------------------------------------------------------

## regen-py-stubs     Regenerate crates/blazen-py/blazen.pyi
regen-py-stubs:
	cargo run --example stub_gen -p blazen-py \
	  --features langfuse,otlp,prometheus,tract,distributed

## regen-node-types   Regenerate crates/blazen-node/index.d.ts (napi build side-effect)
regen-node-types:
	corepack enable
	corepack prepare pnpm@10 --activate
	pnpm install
	pnpm --filter blazen run build

## regen-wasm-types   Regenerate crates/blazen-wasm-sdk/pkg/blazen_wasm_sdk.d.ts
regen-wasm-types:
	wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release

## regen-uniffi       Regenerate Go/Swift/Kotlin UniFFI bindings (rebuilds toolchain libs)
regen-uniffi:
	cargo build -p blazen-uniffi --release --bin uniffi-bindgen
	cargo build -p blazen-uniffi --release
	cargo build -p blazen-cabi --release
	./scripts/regen-bindings.sh

## regen-cabi         Regenerate bindings/ruby/ext/blazen/blazen.h via cbindgen
regen-cabi:
	cargo build -p blazen-cabi --release

## regen-all          regen-py-stubs + regen-node-types + regen-wasm-types + regen-uniffi + regen-cabi
regen-all: regen-py-stubs regen-node-types regen-wasm-types regen-uniffi regen-cabi

## audit-bindings     Full ci.yaml audit-bindings job: regenerate all + drift-check
audit-bindings:
	# 1. Python stubs
	cargo run --example stub_gen -p blazen-py \
	  --features langfuse,otlp,prometheus,tract,distributed
	@if ! git diff --exit-code crates/blazen-py/blazen.pyi; then \
	  echo "::error::blazen.pyi is stale; regenerate via 'make regen-py-stubs'" >&2; \
	  exit 1; \
	fi
	# 2. Node index.d.ts
	corepack enable
	corepack prepare pnpm@10 --activate
	pnpm install
	pnpm --filter blazen run build
	@if ! git diff --exit-code crates/blazen-node/index.d.ts; then \
	  echo "::error::index.d.ts is stale; regenerate via 'make regen-node-types'" >&2; \
	  exit 1; \
	fi
	# 3. WASM-SDK .d.ts
	wasm-pack build crates/blazen-wasm-sdk --target web --out-dir pkg --release
	@if ! git diff --exit-code crates/blazen-wasm-sdk/pkg/blazen_wasm_sdk.d.ts; then \
	  echo "::error::wasm-sdk pkg/blazen_wasm_sdk.d.ts is stale; regenerate via 'make regen-wasm-types'" >&2; \
	  exit 1; \
	fi
	# 4. Binding-parity audit
	uv run --no-project python3 tools/audit_bindings.py
	# 5. uniffi-bindgen-go (idempotent install at pinned tag)
	@if ! command -v uniffi-bindgen-go >/dev/null 2>&1; then \
	  cargo install uniffi-bindgen-go \
	    --tag v0.7.1+v0.31.0 \
	    --git https://github.com/NordSecurity/uniffi-bindgen-go; \
	fi
	# 6. UniFFI + cabi libs + regen bindings (Go/Swift/Kotlin/Ruby header)
	cargo build -p blazen-uniffi --release --bin uniffi-bindgen
	cargo build -p blazen-uniffi --release
	cargo build -p blazen-cabi --release
	./scripts/regen-bindings.sh
	@if ! git diff --exit-code bindings/; then \
	  echo "::error::bindings/ is stale; regenerate via 'make regen-uniffi' and commit" >&2; \
	  exit 1; \
	fi

# -----------------------------------------------------------------------------
# Build helpers
# -----------------------------------------------------------------------------

## build-uniffi-libs  Build libblazen_uniffi + libblazen_cabi for $(UNIFFI_HOST_TRIPLE)
build-uniffi-libs:
	./scripts/build-uniffi-lib.sh $(UNIFFI_HOST_TRIPLE)

# -----------------------------------------------------------------------------
# Maintenance
# -----------------------------------------------------------------------------

## prune-pypi         Preview PyPI releases to delete (frees project size quota).
##                    Dry-run by default. To apply (prompts for password+OTP —
##                    PyPI 2FA can't be headless, so run this yourself):
##                      PYPI_CLEANUP_PASSWORD=... make prune-pypi DO_IT=1 PYPI_USER=<user>
##                    Size-target mode: make prune-pypi TARGET_GB=8
prune-pypi:
	python3 scripts/prune-pypi.py \
	  $(if $(TARGET_GB),--target-gb $(TARGET_GB)) \
	  $(if $(PYPI_USER),-u $(PYPI_USER)) \
	  $(if $(DO_IT),--do-it)

# -----------------------------------------------------------------------------
# Aggregate CI gates
# -----------------------------------------------------------------------------

## ci-fast            Pre-push gate (~10 min, no API keys, no Go/Swift/Kotlin/Ruby)
ci-fast: fmt-check lint test-rust test-doc test-python test-node test-wasm audit-bindings
	@echo ""
	@echo "✓ ci-fast green — Rust/Python/Node/WASM/audit all pass."

## ci-bindings        Cross-language gate (~10 min, requires toolchains)
ci-bindings: test-go test-swift test-kotlin test-ruby
	@echo ""
	@echo "✓ ci-bindings green — Go/Swift/Kotlin/Ruby all pass."

## ci-smoke           API-key-required smoke pass (costs fal + openrouter credit)
ci-smoke: _require-fal-keys test-python-smoke test-node-smoke test-smoke-compute
	@echo ""
	@echo "✓ ci-smoke green — real-HTTP + compute smoke pass."

## ci                 Full ci.yaml mirror = ci-fast + ci-bindings + ci-smoke
ci: ci-fast ci-bindings ci-smoke
	@echo ""
	@echo "✓ FULL CI mirror green — safe to push."
