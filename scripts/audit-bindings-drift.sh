#!/usr/bin/env bash
# CI helper: regenerate all four UniFFI bindings and fail if anything changed.
#
# Used by .forgejo/workflows/ci.yaml's `audit-bindings` job. Local equivalent
# of the audit pyo3-stub-gen / napi build / wasm-pack build drift checks for
# the existing Python/Node/WASM bindings.
#
# Usage: ./scripts/audit-bindings-drift.sh
set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/regen-bindings.sh

BINDING_DIRS=(
    bindings/go/internal/uniffi
    bindings/swift/Sources/UniFFIBlazen
    bindings/kotlin/src/commonMain/uniffi
    bindings/ruby/lib/uniffi/blazen
)

if ! git diff --exit-code -- "${BINDING_DIRS[@]}"; then
    echo
    echo "ERROR: UniFFI-generated bindings are out of date." >&2
    echo "Run scripts/regen-bindings.sh locally and commit the diff." >&2
    exit 1
fi

echo "All four UniFFI bindings are in sync with blazen-uniffi."
