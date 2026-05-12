#!/usr/bin/env bash
# Regenerate the four foreign-language bindings from blazen-uniffi.
#
# Drives:
#   uniffi-bindgen-go    → bindings/go/internal/uniffi/blazen/
#   uniffi-bindgen swift → bindings/swift/Sources/UniFFIBlazen/
#   uniffi-bindgen kotlin → bindings/kotlin/src/commonMain/uniffi/
#   uniffi-bindgen ruby  → bindings/ruby/lib/uniffi/blazen/
#
# CI runs this followed by `git diff --exit-code` to catch unregenerated
# drift (see .forgejo/workflows/ci.yaml `audit-bindings` job).
#
# Usage:
#   ./scripts/regen-bindings.sh            # all four
#   ./scripts/regen-bindings.sh go         # subset: go|swift|kotlin|ruby (space-separated)
#
# Prerequisites (install with the helper at the bottom of this file):
#   uniffi-bindgen-go (NordSecurity) — for Go
#   uniffi-bindgen (mozilla/uniffi-rs) — for Swift / Kotlin / Ruby
#
# Both tools are pinned to uniffi-rs 0.31. Mismatched versions will produce
# scaffolding that doesn't match the runtime, with confusing link errors.
set -euo pipefail

cd "$(dirname "$0")/.."

UNIFFI_VERSION="0.31"

# ---- pick targets -----------------------------------------------------------
TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(go swift kotlin ruby)
fi

# ---- ensure native lib is built (bindgens read metadata from it) -----------
# UniFFI's library-mode bindgen reads scaffolding metadata (UNIFFI_META_*
# symbols) directly from the compiled artefact. Empirically on this workspace,
# all three release artefacts retain the full set of metadata symbols:
#   target/release/libblazen_uniffi.so   (cdylib)    — 138 UNIFFI_META symbols
#   target/release/libblazen_uniffi.a    (staticlib) — 138 UNIFFI_META symbols
#   target/release/libblazen_uniffi.rlib (rlib)      — 138 UNIFFI_META symbols
#
# The metadata is what powers every #[uniffi::export] proc-macro; without
# --library, bindgen falls back to UDL (which only declares `version()` here)
# and silently emits a near-empty surface. Always pass --library.
#
# We prefer the per-build rlib in target/release/deps/ (matches upstream
# uniffi-rs examples), but fall back to the top-level rlib or the staticlib,
# all of which are equivalent for metadata purposes.
DEPS_DIR="target/release/deps"
LIB_PATH=$(ls -t "$DEPS_DIR"/libblazen_uniffi-*.rlib 2>/dev/null | head -1)
if [[ -z "$LIB_PATH" || ! -f "$LIB_PATH" ]]; then
    if [[ -f "target/release/libblazen_uniffi.rlib" ]]; then
        LIB_PATH="target/release/libblazen_uniffi.rlib"
    else
        LIB_PATH="target/release/libblazen_uniffi.a"
    fi
fi

if [[ ! -f "$LIB_PATH" ]]; then
    echo "Building blazen-uniffi (release) so bindgens can read scaffolding metadata..."
    cargo build -p blazen-uniffi --release
    LIB_PATH=$(ls -t "$DEPS_DIR"/libblazen_uniffi-*.rlib 2>/dev/null | head -1)
    if [[ -z "$LIB_PATH" || ! -f "$LIB_PATH" ]]; then
        if [[ -f "target/release/libblazen_uniffi.rlib" ]]; then
            LIB_PATH="target/release/libblazen_uniffi.rlib"
        else
            LIB_PATH="target/release/libblazen_uniffi.a"
        fi
    fi
fi

echo "Using artefact for metadata: $LIB_PATH"

UDL="crates/blazen-uniffi/src/blazen.udl"
CFG="crates/blazen-uniffi/uniffi.toml"

regen_go() {
    if ! command -v uniffi-bindgen-go >/dev/null 2>&1; then
        echo "ERROR: uniffi-bindgen-go not installed."
        echo "  Install with: cargo install uniffi-bindgen-go --tag v$UNIFFI_VERSION.0+v$UNIFFI_VERSION.0 --git https://github.com/NordSecurity/uniffi-bindgen-go"
        return 1
    fi
    mkdir -p bindings/go/internal/uniffi
    uniffi-bindgen-go \
        --library "$LIB_PATH" \
        --config "$CFG" \
        --out-dir bindings/go/internal/uniffi
    echo "  ✓ Go bindings → bindings/go/internal/uniffi/"
}

regen_swift() {
    if ! command -v uniffi-bindgen >/dev/null 2>&1; then
        echo "ERROR: uniffi-bindgen not installed."
        echo "  Install with: cargo install uniffi_bindgen --version ^$UNIFFI_VERSION"
        return 1
    fi
    mkdir -p bindings/swift/Sources/UniFFIBlazen
    uniffi-bindgen generate \
        --library "$LIB_PATH" \
        --language swift \
        --config "$CFG" \
        --out-dir bindings/swift/Sources/UniFFIBlazen
    echo "  ✓ Swift bindings → bindings/swift/Sources/UniFFIBlazen/"
}

regen_kotlin() {
    if ! command -v uniffi-bindgen >/dev/null 2>&1; then
        echo "ERROR: uniffi-bindgen not installed."
        return 1
    fi
    # Output to src/main/kotlin so the generated file's package
    # (`dev.zorpx.blazen.uniffi`, set in uniffi.toml) lands at the path
    # `src/main/kotlin/dev/zorpx/blazen/uniffi/blazen.kt` that the hand-
    # written wrappers in `src/main/kotlin/dev/zorpx/blazen/` import.
    mkdir -p bindings/kotlin/src/main/kotlin
    uniffi-bindgen generate \
        --library "$LIB_PATH" \
        --language kotlin \
        --config "$CFG" \
        --out-dir bindings/kotlin/src/main/kotlin
    echo "  ✓ Kotlin bindings → bindings/kotlin/src/main/kotlin/dev/zorpx/blazen/uniffi/"
}

regen_ruby() {
    if ! command -v uniffi-bindgen >/dev/null 2>&1; then
        echo "ERROR: uniffi-bindgen not installed."
        return 1
    fi
    mkdir -p bindings/ruby/lib/uniffi/blazen
    uniffi-bindgen generate \
        --library "$LIB_PATH" \
        --language ruby \
        --config "$CFG" \
        --out-dir bindings/ruby/lib/uniffi/blazen
    echo "  ✓ Ruby bindings → bindings/ruby/lib/uniffi/blazen/"
}

# Silence unused-fn warnings — these are conditionally invoked below.
: "${UDL}"

echo "Regenerating bindings for: ${TARGETS[*]}"
for t in "${TARGETS[@]}"; do
    case "$t" in
        go) regen_go ;;
        swift) regen_swift ;;
        kotlin) regen_kotlin ;;
        ruby) regen_ruby ;;
        *) echo "Unknown target: $t (use go|swift|kotlin|ruby)" >&2; exit 1 ;;
    esac
done

echo "Done."
