# Blazen ⨯ Astro 6 ⨯ Cloudflare Workers smoke test

End-to-end harness that proves `import { CompletionModel } from 'blazen'` works in
an `@astrojs/cloudflare` (workerd) build **without any consumer-side Vite/Rollup
workaround plugins**.

## What this exercises

Two longstanding pain points when bundling napi-rs packages for edge runtimes:

1. **The platform-detection chain.** napi-rs's auto-generated `index.js`
   statically `require()`s every `@scope/pkg-<platform>-<arch>-<libc>` subpackage.
   Rollup's commonjs plugin tries to parse the `.node` ELF binaries inside and
   crashes. Consumers typically work around this with a `resolveId` plugin that
   externalizes every platform subpackage — ugly and fragile.

2. **The wasi-browser shim.** napi-rs's generated `blazen.wasi-browser.js`
   does `new URL('./blazen.wasm32-wasi.wasm', import.meta.url).href` at module
   init. In workerd, `import.meta.url` is empty for inline-bundled chunks, so
   the top-level `new URL` throws `TypeError: Invalid URL string` before any
   user code runs.

The fix in Blazen:

- An `exports` map in `crates/blazen-node/package.json` routes the `workerd`
  and `browser` conditions to `blazen.workers.js`, a hand-written one-line
  re-export that has zero platform-detection code. Edge bundlers stop parsing
  `index.js` entirely.
- A post-build patch makes `blazen.wasi-browser.js` lazy: top-level wasm
  instantiation is deferred until first export access. Importing types/classes
  no longer trips the broken `new URL`.
- A workerd-specific entry (`blazen.workerd.js`) uses wrangler's static
  `import wasm from './foo.wasm'` to load the wasm `Module` without any URL or
  fetch at all.

If `astro build` succeeds AND `wrangler dev` returns `{"ok":"function"}` on
`/api/blazen-check.json`, every layer of the fix is working.

## Running

```bash
# From repo root: build Blazen so the local wasi shims exist.
pnpm --filter blazen run build

cd examples/astro-cloudflare
pnpm install        # links workspace `blazen` + stages local `@blazen-dev/blazen-wasm32-wasi`
pnpm build          # astro build → dist/_worker.js
pnpm preview        # wrangler dev on localhost:8787
# In another terminal:
curl http://localhost:8787/api/blazen-check.json
# Expected: {"ok":"function"}
```

## Why a local-wasi staging directory?

`scripts/stage-local-blazen.sh` copies the freshly-built `crates/blazen-node/blazen.wasi-*`
files into `local-wasi/` and writes a minimal `package.json` with the inner
`exports` map. The example then depends on `@blazen-dev/blazen-wasm32-wasi`
via `file:./local-wasi` so it picks up local changes to the wasi shims
without needing to publish a release. `pnpm install` and `pnpm build` both
re-run the stage script.

The released `@blazen-dev/blazen-wasm32-wasi` package goes through the same
shape via `.forgejo/workflows/release.yaml` at publish time.

## What this example deliberately does NOT have

- No Vite `resolveId` plugin to externalize `@blazen-dev/blazen-*` subpackages.
- No `ssr.external` / `ssr.noExternal` lists for `blazen`.
- No `optimizeDeps.exclude`.
- No `manualChunks` for Blazen.
- No `patch-package` patches.
- No `@cloudflare/vinext`-style auto-stubbing.

Any of these reappearing in this example means the fix has regressed.
