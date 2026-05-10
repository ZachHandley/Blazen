# @blazen-dev/wasi

Thin re-export alias for [`@blazen-dev/blazen-wasm32-wasi`](https://www.npmjs.com/package/@blazen-dev/blazen-wasm32-wasi) — the WASI build of the Blazen Node binding. Use it in Cloudflare Workers, Deno, or any other WASI host where you'd rather pin the WASI sidecar explicitly than rely on the umbrella `blazen` package's automatic platform resolution.

## Recommended path

Just install the umbrella package and let it resolve the right sidecar for your host:

```bash
npm install blazen
```

The `blazen` umbrella package already pulls in `@blazen-dev/blazen-wasm32-wasi` automatically when no native binary matches the host (e.g. Cloudflare Workers, Deno, WASI runtimes). You do **not** need this alias for that case.

## When to use this alias

Install `@blazen-dev/wasi` only when you want an explicit pin on the WASI build — e.g. you're authoring a library that must run on Workers and you want the dep graph to declare that intent unambiguously, or you want to skip umbrella resolution entirely.

```bash
npm install @blazen-dev/wasi
```

```js
import { /* your APIs */ } from '@blazen-dev/wasi';
```

The exports are identical to `@blazen-dev/blazen-wasm32-wasi`; this package only re-exports.

## Documentation

See the main [`blazen`](https://www.npmjs.com/package/blazen) package for full API docs and usage examples.
