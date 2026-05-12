#!/usr/bin/env bash
# Build the blazen-uniffi native libraries that the four foreign-language
# bindings link against.
#
# Outputs (one host target, release profile):
#   target/release/libblazen_uniffi.a    — Go cgo, Swift XCFramework, Ruby fat-gem (static)
#   target/release/libblazen_uniffi.so   — Kotlin JNI, Ruby native ext (linux dynamic)
#   target/release/libblazen_uniffi.dylib — Kotlin JNI, Ruby native ext (macos dynamic)
#
# For cross-target builds (the prebuilt-lib matrix that feeds bindings/go/internal/clib/,
# bindings/swift/*.xcframework, bindings/kotlin/src/jvmMain/resources/, bindings/ruby/ext/),
# see .forgejo/workflows/build-artifacts.yaml. This script handles the host-only
# fast path for local development.
#
# Usage: ./scripts/build-uniffi-lib.sh
#
# Env:
#   BLAZEN_UNIFFI_FEATURES  comma-separated features (default: default features = local-all)
#   BLAZEN_UNIFFI_PROFILE   "release" (default) or "debug"
set -euo pipefail

cd "$(dirname "$0")/.."

FEATURES="${BLAZEN_UNIFFI_FEATURES:-}"
PROFILE="${BLAZEN_UNIFFI_PROFILE:-release}"

FEATURE_FLAGS=()
if [[ -n "$FEATURES" ]]; then
    FEATURE_FLAGS+=("--features" "$FEATURES")
fi

PROFILE_FLAG=()
if [[ "$PROFILE" == "release" ]]; then
    PROFILE_FLAG+=("--release")
elif [[ "$PROFILE" != "debug" ]]; then
    echo "BLAZEN_UNIFFI_PROFILE must be 'release' or 'debug', got '$PROFILE'" >&2
    exit 1
fi

echo "Building blazen-uniffi ($PROFILE, features=${FEATURES:-<default>})..."
cargo build -p blazen-uniffi "${PROFILE_FLAG[@]}" "${FEATURE_FLAGS[@]}"

OUT="target/$PROFILE"
echo "Artifacts:"
for ext in a so dylib dll lib; do
    for f in "$OUT"/*blazen_uniffi*."$ext"; do
        if [[ -f "$f" ]]; then
            echo "  $(du -h "$f" | cut -f1)  $f"
        fi
    done
done
