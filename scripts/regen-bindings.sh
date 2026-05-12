#!/usr/bin/env bash
# Regenerate the four foreign-language bindings from blazen-uniffi.
#
# Drives:
#   uniffi-bindgen-go    → bindings/go/internal/uniffi/blazen/
#   uniffi-bindgen swift → bindings/swift/Sources/UniFFIBlazen/
#   uniffi-bindgen kotlin → bindings/kotlin/src/main/kotlin/dev/zorpx/blazen/uniffi/
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

# Prefer the locally-built uniffi-bindgen CLI (target/release/uniffi-bindgen)
# over a globally-installed one. The local CLI is built from blazen-uniffi's
# bin/uniffi-bindgen.rs entrypoint, so its version is guaranteed to match the
# runtime scaffolding embedded in the cdylib. A mismatched globally-installed
# uniffi-bindgen would emit incompatible glue with cryptic link errors.
if [[ -x "target/release/uniffi-bindgen" ]]; then
    UNIFFI_BINDGEN="target/release/uniffi-bindgen"
elif command -v uniffi-bindgen >/dev/null 2>&1; then
    UNIFFI_BINDGEN="$(command -v uniffi-bindgen)"
else
    echo "ERROR: uniffi-bindgen not found. Build the local CLI first:"
    echo "  cargo build -p blazen-uniffi --release --bin uniffi-bindgen"
    exit 1
fi
echo "Using uniffi-bindgen: $UNIFFI_BINDGEN"

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
# Glob expansion under set -u/-e/pipefail: use a shell array so a missing
# pattern doesn't trigger pipeline failures.
LIB_PATH=""
shopt -s nullglob
deps_rlibs=("$DEPS_DIR"/libblazen_uniffi-*.rlib)
shopt -u nullglob
if (( ${#deps_rlibs[@]} > 0 )); then
    LIB_PATH="${deps_rlibs[0]}"
elif [[ -f "target/release/libblazen_uniffi.rlib" ]]; then
    LIB_PATH="target/release/libblazen_uniffi.rlib"
elif [[ -f "target/release/libblazen_uniffi.a" ]]; then
    LIB_PATH="target/release/libblazen_uniffi.a"
fi

if [[ -z "$LIB_PATH" || ! -f "$LIB_PATH" ]]; then
    echo "Building blazen-uniffi (release) so bindgens can read scaffolding metadata..."
    cargo build -p blazen-uniffi --release
    shopt -s nullglob
    deps_rlibs=("$DEPS_DIR"/libblazen_uniffi-*.rlib)
    shopt -u nullglob
    if (( ${#deps_rlibs[@]} > 0 )); then
        LIB_PATH="${deps_rlibs[0]}"
    elif [[ -f "target/release/libblazen_uniffi.rlib" ]]; then
        LIB_PATH="target/release/libblazen_uniffi.rlib"
    else
        LIB_PATH="target/release/libblazen_uniffi.a"
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
    # Patch a known uniffi-bindgen-go codegen bug: when a method takes a
    # parameter named `err` (e.g. CompletionStreamSink.on_error(err)), the
    # generated Go body declares `_, err := uniffiRustCallAsync(...)` which
    # collides with the parameter and fails `go vet` ("no new variables on
    # left side of :="). Rename the parameter so the local err declaration
    # is the only `err` in scope.
    if [[ -f bindings/go/internal/uniffi/blazen/blazen.go ]]; then
        python3 - <<'PY'
import re
from pathlib import Path
p = Path("bindings/go/internal/uniffi/blazen/blazen.go")
src = p.read_text()
pattern = re.compile(
    r'func \(_self \*CompletionStreamSinkImpl\) OnError\(err \*BlazenError\) error \{[\s\S]*?'
    r'FfiConverterBlazenErrorINSTANCE\.Lower\(err\)\)',
)
def fixup(match):
    block = match.group(0)
    block = block.replace("OnError(err *BlazenError)", "OnError(onErrorArg *BlazenError)", 1)
    block = block.replace("FfiConverterBlazenErrorINSTANCE.Lower(err))", "FfiConverterBlazenErrorINSTANCE.Lower(onErrorArg))", 1)
    return block
new_src, n = pattern.subn(fixup, src, count=1)
if n:
    p.write_text(new_src)
    print(f"  ↳ patched OnError parameter shadowing ({n} site)")
PY
    fi
    echo "  ✓ Go bindings → bindings/go/internal/uniffi/"
}

regen_swift() {
    mkdir -p bindings/swift/Sources/UniFFIBlazen
    "$UNIFFI_BINDGEN" generate \
        --library "$LIB_PATH" \
        --language swift \
        --config "$CFG" \
        --out-dir bindings/swift/Sources/UniFFIBlazen
    echo "  ✓ Swift bindings → bindings/swift/Sources/UniFFIBlazen/"
}

regen_kotlin() {
    # Output to src/main/kotlin so the generated file's package
    # (`dev.zorpx.blazen.uniffi`, set in uniffi.toml) lands at the path
    # `src/main/kotlin/dev/zorpx/blazen/uniffi/blazen.kt` that the hand-
    # written wrappers in `src/main/kotlin/dev/zorpx/blazen/` import.
    mkdir -p bindings/kotlin/src/main/kotlin
    "$UNIFFI_BINDGEN" generate \
        --library "$LIB_PATH" \
        --language kotlin \
        --config "$CFG" \
        --out-dir bindings/kotlin/src/main/kotlin
    # Patch a known uniffi-bindgen-kotlin codegen bug: each variant of the
    # sealed `BlazenException` class declares the constructor parameter as
    # plain `val message: kotlin.String` AND an override body
    # `override val message get() = "..."` — Kotlin rejects this as a
    # name collision ("Conflicting declarations" + "Overload resolution
    # ambiguity"). The fix is to promote the constructor parameter to
    # `override val message: kotlin.String` and drop the override body.
    # `Cancelled` has no constructor params and is left alone.
    local kt_file=bindings/kotlin/src/main/kotlin/dev/zorpx/blazen/uniffi/blazen.kt
    if [[ -f "$kt_file" ]]; then
        BLAZEN_KT_FILE="$kt_file" python3 - <<'PY'
import os
import re
from pathlib import Path
p = Path(os.environ["BLAZEN_KT_FILE"])
src = p.read_text()
pattern = re.compile(
    r'(class \w+\([^)]*?)(\n\s+)val `message`: kotlin\.String'
    r'([^)]*?\) : BlazenException\(\)) \{\s*'
    r'override val message\s*\n\s*get\(\) = "[^"]*"\s*\n\s*\}',
    re.MULTILINE,
)
def fixup(m):
    return f"{m.group(1)}{m.group(2)}override val `message`: kotlin.String{m.group(3)}"
new_src, n = pattern.subn(fixup, src)
if n:
    p.write_text(new_src)
    print(f"  ↳ patched BlazenException variant message collisions ({n} sites)")
PY
    fi
    echo "  ✓ Kotlin bindings → bindings/kotlin/src/main/kotlin/dev/zorpx/blazen/uniffi/"
}

regen_ruby() {
    mkdir -p bindings/ruby/lib/uniffi/blazen
    "$UNIFFI_BINDGEN" generate \
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
