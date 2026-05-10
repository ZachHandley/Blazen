#!/usr/bin/env bash
#
# End-to-end probe runner for the Blazen wasi sidecar on Cloudflare Workers.
#
# Builds the wasi binary, boots `wrangler dev` (capturing stderr to .dev/),
# runs the probe harness, then cleans up. Run from this directory.
#
# Usage:
#   ./dev-e2e.sh                            # build + run all probes
#   ./dev-e2e.sh --no-build                 # skip the napi build (use existing .wasm)
#   ./dev-e2e.sh --filter=workflow.run      # forwarded to run-tests.mjs
#   OPENAI_API_KEY=sk-... ./dev-e2e.sh      # include the openai probe
#   WRANGLER_PORT=8790 ./dev-e2e.sh         # use a different port
#
# After the run, .dev/wrangler.log contains the full wrangler/wasi stderr —
# useful for spotting Rust panics, eprintln! traces, or workerd diagnostics.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$EXAMPLE_DIR" rev-parse --show-toplevel)"
LOG_DIR="$EXAMPLE_DIR/.dev"
WRANGLER_LOG="$LOG_DIR/wrangler.log"
PORT="${WRANGLER_PORT:-8787}"
SKIP_BUILD=0
FORWARD_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-build) SKIP_BUILD=1 ;;
    *) FORWARD_ARGS+=("$arg") ;;
  esac
done

mkdir -p "$LOG_DIR"
: > "$WRANGLER_LOG"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "▸ building wasi binary…"
  cd "$REPO_ROOT"
  EMNAPI_LINK_DIR="$REPO_ROOT/node_modules/.pnpm/emnapi@1.9.0/node_modules/emnapi/lib/wasm32-wasip1-threads" \
    pnpm --filter blazen exec napi build --release \
      --target wasm32-wasip1-threads --no-default-features --features wasi \
      --js index.js --platform
fi

cd "$EXAMPLE_DIR"

# Free the port if a stale wrangler is still bound. Quiet on no-op.
fuser -k "${PORT}/tcp" 2>/dev/null || true

echo "▸ starting wrangler dev on :$PORT (logs → $WRANGLER_LOG)…"
npx wrangler dev --port "$PORT" --ip 127.0.0.1 --local >"$WRANGLER_LOG" 2>&1 &
WRANGLER_PID=$!

cleanup() {
  if kill -0 "$WRANGLER_PID" 2>/dev/null; then
    pkill -P "$WRANGLER_PID" 2>/dev/null || true
    kill -TERM "$WRANGLER_PID" 2>/dev/null || true
    wait "$WRANGLER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait up to 60s for `Ready on http`. Bail noisily if wrangler crashed first.
deadline=$((SECONDS + 60))
while ! grep -q "Ready on http" "$WRANGLER_LOG" 2>/dev/null; do
  if ! kill -0 "$WRANGLER_PID" 2>/dev/null; then
    echo "✘ wrangler exited before ready — last log:"
    tail -50 "$WRANGLER_LOG"
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "✘ wrangler did not become ready in 60s — last log:"
    tail -50 "$WRANGLER_LOG"
    exit 1
  fi
  sleep 0.3
done

echo "▸ running probes…"
set +e
WRANGLER_PORT="$PORT" SKIP_WRANGLER=1 node run-tests.mjs "${FORWARD_ARGS[@]}"
PROBE_EXIT=$?
set -e

# Stop wrangler first so its stdio buffers flush to the log before we grep.
cleanup
trap - EXIT INT TERM
sync 2>/dev/null || true

echo ""
echo "===== wrangler diagnostic lines ====="
if grep -E "ERROR|panic|Illegal|process\.exit|\[blazen|RuntimeError|unreachable" "$WRANGLER_LOG"; then
  :
else
  echo "(no diagnostic lines — clean run)"
fi
echo "===== full log: $WRANGLER_LOG ====="

exit "$PROBE_EXIT"
