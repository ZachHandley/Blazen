# @blazen-dev/wasi

The single-install Blazen package for Cloudflare Workers, Deno, and any other WASI host. Re-exports [`@blazen-dev/blazen-wasm32-wasi`](https://www.npmjs.com/package/@blazen-dev/blazen-wasm32-wasi) — the napi-rs WASI build of Blazen — and lists the wasm sidecar as a regular dependency so `pnpm add @blazen-dev/wasi` is all you need.

## Install

```bash
pnpm add @blazen-dev/wasi
# or
npm install @blazen-dev/wasi
```

The wasm sidecar (`@blazen-dev/blazen-wasm32-wasi`) is pulled in automatically as a transitive dependency. No second install, no Vite/Rollup workaround plugins, no `ssr.external` lists, no `manualChunks`. Works out of the box with wrangler ≥ 3.15 and `@cloudflare/vite-plugin`.

## Usage

```js
import {
  CompletionModel,
  EmbeddingModel,
  AnthropicProvider,
  Workflow,
} from '@blazen-dev/wasi';
```

The export surface is identical to the main `blazen` package; only the binding implementation differs (wasm32-wasi build instead of a per-platform `.node` binary).

## When to use this instead of `blazen`

The umbrella `blazen` package targets Node servers and installs a per-platform native `.node` binary. The wasm sidecar is an *optional* peer-dependency there, intentionally kept out of Node-only install footprints. Use this package when:

- You're deploying to Cloudflare Workers, Deno Deploy, Fastly Compute, or another WASI host where the native `.node` binary won't load.
- You want a single `pnpm add` line that pulls everything needed for the target runtime.
- You want the dependency graph to clearly declare your Workers/edge intent.

## Alternative: `blazen/workers` subpath

If you'd rather list `blazen` in your dependency tree (mirroring a Node deployment) and explicitly pin the wasm sidecar yourself:

```bash
pnpm add blazen @blazen-dev/blazen-wasm32-wasi
```

```js
import { CompletionModel } from 'blazen/workers';
```

Both installs are required because the wasm sidecar is an optional peer-dependency on `blazen`. The `blazen/workers` subpath uses the same wrangler-static `import wasm from '@blazen-dev/blazen-wasm32-wasi/blazen.wasm32-wasi.wasm'` loader under the hood as this alias.

## Documentation

See the main [`blazen`](https://www.npmjs.com/package/blazen) package for full API docs and usage examples.
