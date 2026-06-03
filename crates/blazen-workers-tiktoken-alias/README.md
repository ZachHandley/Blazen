# @blazen-dev/wasi-tiktoken

The exact-BPE Blazen package for Cloudflare Workers, Deno, and any other WASI host. Identical to [`@blazen-dev/wasi`](https://www.npmjs.com/package/@blazen-dev/wasi) except it re-exports [`@blazen-dev/blazen-wasm32-wasi-tiktoken`](https://www.npmjs.com/package/@blazen-dev/blazen-wasm32-wasi-tiktoken) — the WASI build that bundles `tiktoken-rs` so `TiktokenCounter` returns exact OpenAI BPE token counts on a Worker.

## Lean vs tiktoken

The default `@blazen-dev/wasi` (and the `@blazen-dev/blazen-wasm32-wasi` sidecar that `blazen/workers` uses) ships **without** tiktoken to stay small enough for the Cloudflare Workers bundle budget (~1.7 MiB gzipped vs ~4.8 MiB with tiktoken). On that lean build, token counting falls back to the always-available `EstimateCounter` heuristic, and `TiktokenCounter.forModel(...)` throws an actionable error.

Use this package **only** when you need exact counts on a Worker and have the bundle budget for the larger binary.

## Install

```bash
pnpm add @blazen-dev/wasi-tiktoken
# or
npm install @blazen-dev/wasi-tiktoken
```

```js
import { Model, TiktokenCounter, Workflow } from '@blazen-dev/wasi-tiktoken';

const counter = TiktokenCounter.forModel('gpt-4o');
counter.countTokens('Hello, world!'); // exact BPE count
```

The export surface is identical to `@blazen-dev/wasi` / the main `blazen` package; only the bundled wasm differs.

## Alternative: override the lean sidecar so `blazen/workers` uses tiktoken

If you import via `blazen/workers` (which pins the lean `@blazen-dev/blazen-wasm32-wasi` sidecar) and want exact counts without changing your imports, install the tiktoken sidecar and remap it with a package-manager override. The two packages share an identical export map and file layout, so this is a drop-in binary swap.

pnpm (in your `package.json`):

```json
{
  "pnpm": {
    "overrides": {
      "@blazen-dev/blazen-wasm32-wasi": "npm:@blazen-dev/blazen-wasm32-wasi-tiktoken@<version>"
    }
  }
}
```

npm:

```json
{
  "overrides": {
    "@blazen-dev/blazen-wasm32-wasi": "npm:@blazen-dev/blazen-wasm32-wasi-tiktoken@<version>"
  }
}
```

Pin `<version>` to the same release as your `blazen` package.

## Documentation

See the main [`blazen`](https://www.npmjs.com/package/blazen) package for full API docs and usage examples.
