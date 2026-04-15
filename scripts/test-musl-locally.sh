#!/usr/bin/env bash
# Reproduce the musl wheel build locally using rust-musl-cross:x86_64-musl.
# Uses a read-only bind mount + in-container copy so host files are NEVER
# modified by the stamping or build steps. Mirrors the CI build-wheels job
# for x86_64-unknown-linux-musl.
#
# Usage: ./scripts/test-musl-locally.sh
#
# Requirements: docker (or podman with alias docker=podman)
set -euo pipefail

cd "$(dirname "$0")/.."

# Optional: capture built wheels somewhere on the host so the user can inspect.
# We use a sibling directory that git ignores.
OUT_DIR="${PWD}/target/musl-wheels"
mkdir -p "$OUT_DIR"

docker run --rm -i \
  -v "$(pwd):/src:ro,Z" \
  -v "$OUT_DIR:/wheels:Z" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_PYTHON_INSTALL_DIR=/tmp/uv-python \
  -e UV_TOOL_DIR=/tmp/uv-tools \
  -e UV_TOOL_BIN_DIR=/tmp/uv-tools/bin \
  ghcr.io/rust-cross/rust-musl-cross:x86_64-musl \
  bash -s <<'INNER'
set -euo pipefail

echo "==> Copying workspace into /build (host remains read-only)"
# cp -r preserves timestamps and preserves the tree structure.
# The ro bind mount at /src means any accidental write attempt fails loudly
# rather than quietly corrupting host files.
mkdir -p /build
cp -r /src/. /build/
cd /build

echo "==> Installing system deps"
apt-get update -qq
apt-get install -y -qq cmake ccache pkg-config curl jq perl make libssl-dev patchelf

echo "==> Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/tmp/uv-tools/bin:$PATH"

echo "==> Installing Python versions"
for v in 3.10 3.11 3.12 3.13 3.14; do
  uv python install "$v"
done

echo "==> Installing maturin (with patchelf, since this is Linux)"
uv tool install --with patchelf maturin

echo "==> Stamping version (inside container only — host files untouched)"
uv run --no-project python scripts/stamp-version.py 0.0.1

echo "==> Resolving interpreters"
INTERPRETERS=""
for V in 3.10 3.11 3.12 3.13 3.14; do
  INTERP=$(uv python find "$V" 2>/dev/null) && INTERPRETERS="$INTERPRETERS -i $INTERP" || echo "Python $V unavailable, skipping"
done
echo "Building for x86_64-unknown-linux-musl with: $INTERPRETERS"

echo "==> Building wheels"
maturin build --release \
  -m crates/blazen-py/Cargo.toml \
  --target x86_64-unknown-linux-musl \
  --manylinux musllinux_1_2 \
  $INTERPRETERS

echo "==> Copying produced wheels to host /wheels output"
if ls target/wheels/*.whl >/dev/null 2>&1; then
  cp target/wheels/*.whl /wheels/
  echo "Wheels copied:"
  ls -lh /wheels/
else
  echo "(no wheels produced)"
  exit 1
fi
INNER

echo ""
echo "==> Done. Host source files were NOT modified."
echo "==> Wheels are at: $OUT_DIR"
ls -lh "$OUT_DIR" 2>/dev/null || true
