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
# UniFFI's library-mode bindgen reads scaffolding metadata directly from the
# compiled cdylib, which avoids the UDL drift hazard.
LIB_DIR="target/release"
LIB_PATH="$LIB_DIR/libblazen_uniffi.so"
if [[ "$(uname -s)" == "Darwin" ]]; then
    LIB_PATH="$LIB_DIR/libblazen_uniffi.dylib"
fi

if [[ ! -f "$LIB_PATH" ]]; then
    echo "Building blazen-uniffi (release) so bindgens can read scaffolding metadata..."
    cargo build -p blazen-uniffi --release
fi

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
    mkdir -p bindings/kotlin/src/commonMain/uniffi
    uniffi-bindgen generate \
        --library "$LIB_PATH" \
        --language kotlin \
        --config "$CFG" \
        --out-dir bindings/kotlin/src/commonMain/uniffi
    echo "  ✓ Kotlin bindings → bindings/kotlin/src/commonMain/uniffi/"
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
