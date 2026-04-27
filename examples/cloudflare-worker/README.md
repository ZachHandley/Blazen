# Blazen Cloudflare Workers Smoke Test

End-to-end proof that the real Blazen workflow engine runs inside a Cloudflare
Worker via the `blazen-wasm-sdk` WebAssembly build.

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
