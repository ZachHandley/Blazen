#!/usr/bin/env bash
# Stage a local @blazen-dev/blazen-wasm32-wasi from crates/blazen-node/ build
# outputs. Mirrors what .forgejo/workflows/release.yaml does at publish time
# (the inner exports map + files list + .wasm subpath) but reads from the
# local workspace instead of dist artifacts.
#
# Required because `blazen/workers` does
# `import wasm from '@blazen-dev/blazen-wasm32-wasi/blazen.wasm32-wasi.wasm'`,
# which only resolves if the wasm32-wasi subpackage exposes that subpath in
# its exports map. The published 0.1.x subpackage doesn't (yet) -- this
# staging mirrors what the release pipeline will publish so the example can
# verify locally against unpublished changes.
#
# Re-runs harmlessly: every file is overwritten from source each time.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$ROOT/crates/blazen-node"
DEST="$HERE/../local-wasi"

if [ ! -f "$SRC/blazen.wasi.cjs" ] || [ ! -f "$SRC/blazen.wasi-browser.js" ] || [ ! -f "$SRC/blazen.wasm32-wasi.wasm" ]; then
  echo "::error::missing wasi build outputs in $SRC -- run 'pnpm --filter blazen run build' first" >&2
  exit 1
fi

mkdir -p "$DEST"
cp "$SRC/blazen.wasi.cjs" "$DEST/"
cp "$SRC/blazen.wasi-browser.js" "$DEST/"
cp "$SRC/blazen.wasm32-wasi.wasm" "$DEST/"

cat > "$DEST/package.json" <<'EOF'
{
  "name": "@blazen-dev/blazen-wasm32-wasi",
  "version": "0.1.999-local",
  "type": "module",
  "main": "./blazen.wasi.cjs",
  "module": "./blazen.wasi-browser.js",
  "dependencies": {
    "@emnapi/core": "^1.10.0",
    "@emnapi/runtime": "^1.10.0",
    "@napi-rs/wasm-runtime": "^1.1.4",
    "@tybys/wasm-util": "^0.10.2"
  },
  "files": [
    "blazen.wasi.cjs",
    "blazen.wasi-browser.js",
    "blazen.wasm32-wasi.wasm"
  ],
  "exports": {
    ".": {
      "browser": "./blazen.wasi-browser.js",
      "module":  "./blazen.wasi-browser.js",
      "import":  "./blazen.wasi-browser.js",
      "require": "./blazen.wasi.cjs",
      "default": "./blazen.wasi.cjs"
    },
    "./blazen.wasm32-wasi.wasm": "./blazen.wasm32-wasi.wasm",
    "./package.json": "./package.json"
  }
}
EOF

echo "[stage-local-blazen] staged $DEST"
echo "[stage-local-blazen]   exposes ./blazen.wasm32-wasi.wasm subpath for the blazen/workers entry"
