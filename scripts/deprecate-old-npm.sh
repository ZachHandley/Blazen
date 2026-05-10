#!/usr/bin/env bash
#
# One-shot deprecation script for legacy Blazen npm package names.
#
# Run MANUALLY (NOT from CI), AFTER the first release that publishes the new
# scoped names (`@blazen-dev/wasm`, `@blazen-dev/wasi`, `@blazen-dev/blazen-wasm32-wasi`,
# and the existing `@blazen-dev/blazen-<triple>` native sidecars) succeeds.
#
# Usage:
#   npm login                       # authenticate with publish rights to all listed packages
#   bash scripts/deprecate-old-npm.sh
#
# What this does:
#   1. Marks every published version of the unscoped `blazen-wasm-sdk` package
#      as deprecated, redirecting users to `@blazen-dev/wasm`.
#   2. Marks every published version of the 6 stale unscoped platform binaries
#      from v0.1.143 (`blazen-linux-x64-gnu`, etc.) as deprecated, redirecting
#      to the canonical `blazen` umbrella package which auto-resolves the
#      scoped `@blazen-dev/blazen-<triple>` sidecars.
#
# `npm deprecate` is a metadata-only operation: existing installs continue to
# work; future installs see a console warning. There is no way to "un-publish"
# old versions, so deprecation is the standard rename path for npm.

set -euo pipefail

# Sanity: confirm logged in.
if ! npm whoami >/dev/null 2>&1; then
  echo "error: not logged in to npm. Run \`npm login\` first." >&2
  exit 1
fi

WASM_MSG="Renamed to @blazen-dev/wasm. See https://www.npmjs.com/package/@blazen-dev/wasm"
PLATFORM_MSG="Stale platform binary. Current binaries live under @blazen-dev/blazen-<triple> and are auto-resolved by 'blazen'. See https://www.npmjs.com/package/blazen"

echo "==> deprecating blazen-wasm-sdk"
npm deprecate "blazen-wasm-sdk@<*" "${WASM_MSG}"

PLATFORM_PKGS=(
  blazen-linux-x64-gnu
  blazen-linux-x64-musl
  blazen-linux-arm64-gnu
  blazen-linux-arm64-musl
  blazen-darwin-arm64
  blazen-win32-x64-msvc
)

for pkg in "${PLATFORM_PKGS[@]}"; do
  echo "==> deprecating ${pkg}"
  npm deprecate "${pkg}@<*" "${PLATFORM_MSG}"
done

echo
echo "Done. Verify with:"
echo "  npm view blazen-wasm-sdk deprecated"
for pkg in "${PLATFORM_PKGS[@]}"; do
  echo "  npm view ${pkg} deprecated"
done
