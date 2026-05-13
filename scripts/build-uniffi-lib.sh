#!/usr/bin/env bash
# Build the blazen-uniffi native libraries for one or more targets and
# distribute the resulting archives/shared-libs into each binding's expected
# layout (Go cgo, Kotlin JNA, Ruby ffi).
#
# Strategy per target (in order):
#   1. Native rustup target + locally-available linker (cargo build --target …)
#   2. cargo-xwin (Windows MSVC: clang-cl + lld-link + xwin SDK cache)
#   3. cross via Docker/podman (only installed on demand)
#
# Where each binding loads the artefact:
#
#   Go (static .a / .lib, linked at cgo time)
#       bindings/go/internal/clib/<GOOS>_<GOARCH>/libblazen_uniffi.{a,lib}
#
#   Kotlin (dynamic, JNA loads at runtime)
#       bindings/kotlin/src/main/resources/<jna-platform>/{libblazen_uniffi.so|.dylib|blazen_uniffi.dll}
#       JNA platform tags: linux-x86-64, linux-aarch64, darwin-x86-64,
#       darwin-aarch64, win32-x86-64, win32-aarch64.
#
#   Ruby (dynamic, ffi gem loads at runtime)
#       bindings/ruby/ext/blazen/<GOOS>_<GOARCH>/{libblazen_uniffi.so|.dylib|blazen_uniffi.dll}
#
#   Swift XCFramework — macOS only, deferred to the macOS Forgejo runner
#   (see Phase 7 CI). Skipped locally with a warning.
#
# Usage:
#   ./scripts/build-uniffi-lib.sh                       # default-set
#   ./scripts/build-uniffi-lib.sh linux_amd64           # subset
#   ./scripts/build-uniffi-lib.sh linux_amd64 windows_amd64
#
# Targets:
#   linux_amd64    x86_64-unknown-linux-gnu
#   linux_arm64    aarch64-unknown-linux-gnu
#   windows_amd64  x86_64-pc-windows-msvc
#   windows_arm64  aarch64-pc-windows-msvc
#   darwin_amd64   x86_64-apple-darwin       (skipped on non-macOS hosts)
#   darwin_arm64   aarch64-apple-darwin      (skipped on non-macOS hosts)
#
# Default set: linux_amd64 linux_arm64 windows_amd64
# (darwin requires a macOS host; windows_arm64 must be opt-in because the
# msvc ARM64 toolchain is heavier than the x64 one.)
#
# Env:
#   BLAZEN_UNIFFI_FEATURES   comma-separated cargo features (default: crate default = local-all)
#   BLAZEN_UNIFFI_PROFILE    "release" (default) or "debug"
#   BLAZEN_INSTALL_CROSS     "1" to auto-install cross when needed (default: prompt-free skip)
set -euo pipefail

# ---- repo-relative paths ----------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_ROOT="$REPO_ROOT/target/cross-scratch"
mkdir -p "$SCRATCH_ROOT"
SCRATCH_DIR="$(mktemp -d -p "$SCRATCH_ROOT" build.XXXXXX)"
cleanup() { rm -rf "$SCRATCH_DIR"; }
trap cleanup EXIT

FEATURES="${BLAZEN_UNIFFI_FEATURES:-}"
PROFILE="${BLAZEN_UNIFFI_PROFILE:-release}"
INSTALL_CROSS="${BLAZEN_INSTALL_CROSS:-0}"

FEATURE_FLAGS=()
if [[ -n "$FEATURES" ]]; then
    FEATURE_FLAGS+=("--features" "$FEATURES")
fi

PROFILE_FLAG=()
case "$PROFILE" in
    release) PROFILE_FLAG+=("--release") ;;
    debug)   ;;
    *)
        echo "BLAZEN_UNIFFI_PROFILE must be 'release' or 'debug', got '$PROFILE'" >&2
        exit 1
        ;;
esac

# ---- target catalogue -------------------------------------------------------
# triple_for <name>           → rust target triple
# jna_for <name>              → kotlin resources/<jna-platform>
# go_subdir_for <name>        → bindings/go/internal/clib/<dir>
# ext_for <name>              → file extension of the shared lib (so/dylib/dll)
# static_ext_for <name>       → file extension of the static archive (a/lib)
# prefix_for <name>           → "lib" on unix, "" on windows
triple_for() {
    case "$1" in
        linux_amd64)    echo "x86_64-unknown-linux-gnu" ;;
        linux_arm64)    echo "aarch64-unknown-linux-gnu" ;;
        windows_amd64)  echo "x86_64-pc-windows-msvc" ;;
        windows_arm64)  echo "aarch64-pc-windows-msvc" ;;
        darwin_amd64)   echo "x86_64-apple-darwin" ;;
        darwin_arm64)   echo "aarch64-apple-darwin" ;;
        *) return 1 ;;
    esac
}

jna_for() {
    case "$1" in
        linux_amd64)    echo "linux-x86-64" ;;
        linux_arm64)    echo "linux-aarch64" ;;
        windows_amd64)  echo "win32-x86-64" ;;
        windows_arm64)  echo "win32-aarch64" ;;
        darwin_amd64)   echo "darwin-x86-64" ;;
        darwin_arm64)   echo "darwin-aarch64" ;;
    esac
}

ext_for() {
    case "$1" in
        linux_*)   echo "so" ;;
        windows_*) echo "dll" ;;
        darwin_*)  echo "dylib" ;;
    esac
}

static_ext_for() {
    case "$1" in
        windows_*) echo "lib" ;;
        *)         echo "a" ;;
    esac
}

prefix_for() {
    case "$1" in
        windows_*) echo "" ;;
        *)         echo "lib" ;;
    esac
}

# ---- build strategy ---------------------------------------------------------
# Probes the host environment for a workable toolchain. Returns one of:
#   native | xwin | cross | none
strategy_for() {
    local name="$1"
    local triple
    triple="$(triple_for "$name")"

    # rustup target must be installed first.
    if ! rustup target list --installed 2>/dev/null | grep -qx "$triple"; then
        # Try to add it silently. If that fails we're done.
        if ! rustup target add "$triple" >/dev/null 2>&1; then
            echo "none"
            return
        fi
    fi

    case "$name" in
        linux_amd64)
            # Native host (assumes Linux x86_64). On other hosts we'd need cross.
            if [[ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]]; then
                echo "native"
            else
                _have cross && echo "cross" || echo "none"
            fi
            ;;
        linux_arm64)
            if [[ "$(uname -s)" == "Linux" ]] && _have aarch64-linux-gnu-gcc; then
                echo "native"
            elif _have cross; then
                echo "cross"
            elif [[ "$INSTALL_CROSS" == "1" ]]; then
                echo "cross"
            else
                echo "none"
            fi
            ;;
        windows_amd64|windows_arm64)
            if _have cargo-xwin; then
                echo "xwin"
            elif _have cross; then
                echo "cross"
            elif [[ "$INSTALL_CROSS" == "1" ]]; then
                echo "cross"
            else
                echo "none"
            fi
            ;;
        darwin_*)
            if [[ "$(uname -s)" == "Darwin" ]]; then
                echo "native"
            else
                # No reliable open-source toolchain for darwin cross from Linux.
                echo "none"
            fi
            ;;
    esac
}

_have() { command -v "$1" >/dev/null 2>&1; }

# Install cross via cargo when requested. Returns 0 on success, 1 on failure.
ensure_cross() {
    if _have cross; then
        return 0
    fi
    if [[ "$INSTALL_CROSS" != "1" ]]; then
        return 1
    fi
    echo "Installing cross (BLAZEN_INSTALL_CROSS=1)..."
    cargo install cross --git https://github.com/cross-rs/cross --locked
}

# Run the target build for `name` using strategy `strat`.
# Outputs the path to the produced cdylib + staticlib via the BUILT_CDYLIB /
# BUILT_STATICLIB globals (bash has no return tuples).
BUILT_CDYLIB=""
BUILT_STATICLIB=""
build_target() {
    local name="$1"
    local strat="$2"
    local triple
    triple="$(triple_for "$name")"
    BUILT_CDYLIB=""
    BUILT_STATICLIB=""

    echo
    echo "============================================================"
    echo "Building $name  (triple=$triple, strategy=$strat, profile=$PROFILE)"
    echo "============================================================"

    case "$strat" in
        native)
            # Linker overrides for the non-host linux target. Cargo picks these
            # up via the standard CARGO_TARGET_*_LINKER env vars.
            local env_pairs=()
            if [[ "$name" == "linux_arm64" ]]; then
                env_pairs=(
                    "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc"
                    "CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc"
                    "CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++"
                    "AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar"
                )
            fi
            env "${env_pairs[@]}" cargo build \
                -p blazen-uniffi \
                --target "$triple" \
                "${PROFILE_FLAG[@]}" \
                "${FEATURE_FLAGS[@]}"
            ;;
        xwin)
            # cargo-xwin shadows `cargo build` with clang-cl + lld-link and the
            # MSVC SDK headers/libs it fetches into ~/.cache/cargo-xwin.
            cargo xwin build \
                -p blazen-uniffi \
                --target "$triple" \
                "${PROFILE_FLAG[@]}" \
                "${FEATURE_FLAGS[@]}"
            ;;
        cross)
            ensure_cross || {
                echo "cross not available; install with BLAZEN_INSTALL_CROSS=1" >&2
                return 1
            }
            cross build \
                -p blazen-uniffi \
                --target "$triple" \
                "${PROFILE_FLAG[@]}" \
                "${FEATURE_FLAGS[@]}"
            ;;
        *)
            echo "Unknown strategy: $strat" >&2
            return 1
            ;;
    esac

    # Locate the produced artefacts. cargo writes to target/<triple>/<profile>/.
    local out_dir="target/$triple/$PROFILE"
    local prefix ext sext
    prefix="$(prefix_for "$name")"
    ext="$(ext_for "$name")"
    sext="$(static_ext_for "$name")"

    BUILT_CDYLIB="$out_dir/${prefix}blazen_uniffi.${ext}"
    BUILT_STATICLIB="$out_dir/${prefix}blazen_uniffi.${sext}"

    if [[ ! -f "$BUILT_CDYLIB" ]]; then
        # Some windows toolchains emit `blazen_uniffi.dll` without the `lib`
        # prefix; some emit `libblazen_uniffi.dll`. Tolerate either.
        local alt="$out_dir/blazen_uniffi.${ext}"
        if [[ -f "$alt" ]]; then
            BUILT_CDYLIB="$alt"
        fi
    fi
    if [[ ! -f "$BUILT_STATICLIB" ]]; then
        local alt2="$out_dir/blazen_uniffi.${sext}"
        if [[ -f "$alt2" ]]; then
            BUILT_STATICLIB="$alt2"
        fi
    fi

    if [[ ! -f "$BUILT_CDYLIB" ]]; then
        echo "ERROR: expected cdylib not found: $BUILT_CDYLIB" >&2
        return 1
    fi
    if [[ ! -f "$BUILT_STATICLIB" ]]; then
        echo "ERROR: expected staticlib not found: $BUILT_STATICLIB" >&2
        return 1
    fi
}

# Distribute the freshly-built artefacts into the three binding layouts.
distribute() {
    local name="$1"
    local cdylib="$BUILT_CDYLIB"
    local staticlib="$BUILT_STATICLIB"

    local prefix ext sext jna go_subdir
    prefix="$(prefix_for "$name")"
    ext="$(ext_for "$name")"
    sext="$(static_ext_for "$name")"
    jna="$(jna_for "$name")"
    go_subdir="$name"

    # ---- Go (static) ---------------------------------------------------------
    # File name convention: keep the `lib` prefix for unix (.a) so cgo's
    # `-lblazen_uniffi` flag works. On windows-msvc we ship the .lib without
    # the `lib` prefix — cgo on windows uses MSVC-style `/LIBPATH:` and the
    # final go-toolchain selects the library by its base name.
    local go_dir="bindings/go/internal/clib/$go_subdir"
    mkdir -p "$go_dir"
    local go_dest
    case "$name" in
        windows_*) go_dest="$go_dir/blazen_uniffi.${sext}" ;;
        *)         go_dest="$go_dir/${prefix}blazen_uniffi.${sext}" ;;
    esac
    install -m 0644 "$staticlib" "$go_dest"

    # ---- Kotlin (dynamic, JNA) ----------------------------------------------
    local kt_dir="bindings/kotlin/src/main/resources/$jna"
    mkdir -p "$kt_dir"
    local kt_dest
    case "$name" in
        windows_*) kt_dest="$kt_dir/blazen_uniffi.${ext}" ;;
        *)         kt_dest="$kt_dir/${prefix}blazen_uniffi.${ext}" ;;
    esac
    install -m 0644 "$cdylib" "$kt_dest"

    # ---- Ruby (dynamic) ------------------------------------------------------
    local rb_dir="bindings/ruby/ext/blazen/$go_subdir"
    mkdir -p "$rb_dir"
    local rb_dest
    case "$name" in
        windows_*) rb_dest="$rb_dir/blazen_uniffi.${ext}" ;;
        *)         rb_dest="$rb_dir/${prefix}blazen_uniffi.${ext}" ;;
    esac
    install -m 0755 "$cdylib" "$rb_dest"

    echo "  → Go:     $go_dest    ($(du -h "$go_dest"  | cut -f1))"
    echo "  → Kotlin: $kt_dest   ($(du -h "$kt_dest"  | cut -f1))"
    echo "  → Ruby:   $rb_dest   ($(du -h "$rb_dest"  | cut -f1))"
}

# ---- host target ------------------------------------------------------------
# Used to decide whether a build failure is fatal (host) or a soft skip
# (cross). The host target *must* link, since every other piece of the
# workspace (regen-bindings, blazen-py, tests) consumes its artefact.
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)   HOST_TARGET="linux_amd64" ;;
    Linux-aarch64)  HOST_TARGET="linux_arm64" ;;
    Darwin-x86_64)  HOST_TARGET="darwin_amd64" ;;
    Darwin-arm64)   HOST_TARGET="darwin_arm64" ;;
    *)              HOST_TARGET="" ;;
esac

# ---- choose target set ------------------------------------------------------
DEFAULT_TARGETS=(linux_amd64 linux_arm64 windows_amd64)
TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

# Validate every requested name resolves to a triple.
for t in "${TARGETS[@]}"; do
    if ! triple_for "$t" >/dev/null 2>&1; then
        echo "Unknown target: $t" >&2
        echo "Supported: linux_amd64 linux_arm64 windows_amd64 windows_arm64 darwin_amd64 darwin_arm64" >&2
        exit 1
    fi
done

# ---- run --------------------------------------------------------------------
declare -A STATUS
declare -A REASON

for name in "${TARGETS[@]}"; do
    strat="$(strategy_for "$name")"
    if [[ "$strat" == "none" ]]; then
        STATUS["$name"]="skipped"
        REASON["$name"]="no working toolchain (rustup target / linker / cargo-xwin / cross)"
        echo
        echo "skipping $name — ${REASON["$name"]}"
        continue
    fi

    if build_target "$name" "$strat"; then
        if distribute "$name"; then
            STATUS["$name"]="built"
            REASON["$name"]="strategy=$strat"
        else
            STATUS["$name"]="failed"
            REASON["$name"]="distribute step errored"
        fi
    else
        # Optional / cross targets: downgrade a build failure to a warning so
        # one busted dep graph (e.g. ORT's DirectML.lib on windows-msvc, or
        # an aarch64 sysroot mismatch) doesn't tank the whole run. The host
        # target stays a hard failure because the rest of the workspace
        # depends on a working linux_amd64 artefact.
        case "$name" in
            "$HOST_TARGET")
                STATUS["$name"]="failed"
                REASON["$name"]="build step errored (strategy=$strat)"
                ;;
            *)
                STATUS["$name"]="skipped"
                REASON["$name"]="build failed under strategy=$strat (dep graph or toolchain mismatch)"
                echo
                echo "warning: $name failed to build under $strat; continuing (non-host target)"
                ;;
        esac
    fi
done

# ---- summary ----------------------------------------------------------------
echo
echo "============================================================"
echo "Summary"
echo "============================================================"
printf "%-16s %-10s %s\n" "target" "status" "detail"
printf "%-16s %-10s %s\n" "------" "------" "------"
exit_code=0
for name in "${TARGETS[@]}"; do
    s="${STATUS["$name"]:-unknown}"
    r="${REASON["$name"]:-}"
    mark="?"
    case "$s" in
        built)   mark="OK    " ;;
        skipped) mark="SKIP  " ;;
        failed)  mark="FAIL  "; exit_code=1 ;;
    esac
    printf "%-16s %-10s %s\n" "$name" "$mark" "$r"
done

exit "$exit_code"
