# Blazen Cloudflare Workers Smoke Test

End-to-end proof that the real Blazen workflow engine runs inside a Cloudflare
Worker via the `@blazen-dev/wasm` WebAssembly build (Rust crate
`blazen-wasm-sdk`, published to npm under the `@blazen-dev/wasm` name).

The worker imports `runSmokeWorkflow()` from the SDK and invokes it from a
`fetch` handler. The function drives the actual Blazen workflow engine to
completion and returns the `StopEvent` payload as JSON.

## Prerequisites

1. Build the WASM SDK (from the repo root):

   ```bash
   cd crates/blazen-wasm-sdk
   wasm-pack build --target bundler --release
   ```

   This produces `crates/blazen-wasm-sdk/pkg/` with `blazen_wasm_sdk.js`,
   `blazen_wasm_sdk_bg.wasm`, and the matching `.d.ts` files. The worker
   imports directly from that path.

2. Install Worker dependencies:

   ```bash
   cd examples/cloudflare-worker
   pnpm install   # or: npm install
   ```

## Alternative install paths

This example imports the WASM SDK from a local path (`../../crates/blazen-wasm-sdk/pkg/`)
so the worker exercises the bits that just came out of `wasm-pack build`. That is
deliberate: it gives end-to-end proof that the freshly-compiled artefacts work in
`workerd` before they ever reach npm.

For production deployments you do not need a local build at all. Two published
packages cover the same ground:

- **`npm install @blazen-dev/wasm`** — the same code this example imports
  locally, published. Drop-in: change the local-path import to
  `from "@blazen-dev/wasm"` and the rest of the worker is identical.
- **`npm install blazen`** — the full Node API surface. On Cloudflare Workers
  and Deno it transparently uses the `@blazen-dev/wasi` sidecar, so the same
  code that runs on a Node server runs on the edge with no rewrites. Pick this
  if you want API parity with your Node code; pick `@blazen-dev/wasm` if bundle
  size matters more than API breadth.

## Run locally

```bash
pnpm dev   # = wrangler dev
```

Wrangler starts a local Workers runtime on `http://localhost:8787`. Hit it:

```bash
curl http://localhost:8787
```

Expected response (the `StopEvent` payload from the real Blazen engine):

```json
{
  "greeting": "Hello, World!"
}
```

If you see that JSON, the Blazen workflow engine successfully ran inside the
V8 isolate that backs Cloudflare Workers.

## Deploy

```bash
pnpm deploy   # = wrangler deploy
```

You'll need to be logged in (`wrangler login`) and have a Workers-enabled
Cloudflare account.

## Known limitations

- **CPU time limits.** The Workers free tier caps wall/CPU time at ~10ms per
  request, which is too tight for any non-trivial Blazen workflow. The
  smoke workflow is designed to fit, but real multi-step workflows
  (especially anything calling out to LLM providers) need the paid tier
  (30s CPU) or should be split across multiple requests / Durable Objects /
  Workflows for orchestration that exceeds a single invocation.
- **No filesystem / native deps.** Anything in the SDK that touches the OS
  (filesystem, native crypto, threads) won't work — Workers run in a V8
  isolate. The WASM SDK is built specifically to avoid those.
- **WASM cold start.** The `--target bundler` build auto-initializes the
  module at import time, so the first request after a deploy pays the wasm
  instantiation cost (~tens of ms for ~1.8 MB of wasm). Subsequent requests
  on the same isolate reuse the initialized module.
