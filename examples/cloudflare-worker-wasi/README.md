# Cloudflare Workers wasi smoke test

This example validates the napi-rs wasi sidecar
(`blazen.wasm32-wasi.wasm` + `@napi-rs/wasm-runtime`) end-to-end on
Cloudflare Workers via `wrangler dev` (`workerd` + `nodejs_compat`).

## Status: 12/12 probes pass (May 2026, wrangler 4.90, compat 2026-04-01)

`npm install blazen` works on Cloudflare Workers — every API exercised
by the probe set runs to completion. Latest harness run:

| Probe | Category | Status |
|---|---|---|
| `load.version` | sync | pass |
| `load.exports-shape` | sync | pass |
| `workflow.ctor` | sync | pass |
| `workflow.addStep` | sync | pass |
| `httpclient.fromCallback` | sync | pass |
| `httpclient.setDefault` | sync | pass |
| `upstash.create` | sync | pass |
| `httppeer.newHttp` | sync | pass |
| `workflow.run` | async | pass |
| `pipeline.run` | async | pass |
| `workflow.runStreaming` | async | pass |
| `workflow.runWithHandler` | async | pass |
| `openai.viaBlazen` | http | skip (no `OPENAI_API_KEY`) |

## What the harness exercises

- `import 'blazen'` resolves the wasi binary; module init wires the
  microtask-driven async dispatcher and the JS-`setTimeout`-driven sleeper
  (see `../../crates/blazen-node/src/wasi_async.rs` and
  `../../crates/blazen-core/src/runtime.rs`).
- All sync APIs: ctors, builders, factories
  (`new Workflow`, `addStep`, `HttpClient.fromCallback`,
  `setDefaultHttpClient`, `UpstashBackend.create`, `HttpPeerClient.newHttp`).
- `Workflow.run` / `Workflow.runStreaming` / `Workflow.runWithHandler` —
  every `#[napi] async fn` workflow entrypoint, including step handlers
  with `async` JS callbacks that round-trip through napi-rs's
  `ThreadsafeFunction` and `Promise<T>` types.
- `Pipeline.start().result()` — multi-stage pipeline with cross-stage
  data passing.

## Critical patches that make this work

The fixes live across four layers; running the dev harness rebuilds the
wasm and applies the JS patches automatically.

1. **Vendored napi-rs 3.8.6** (`crates/napi-patched/`) replaces
   `[patch.crates-io]` in the workspace. Two changes vs upstream:
   - `set_async_executor()` hook in `tokio_runtime.rs` so embedders can
     bypass napi-rs's hardcoded `std::thread::spawn(|| block_on(...))`,
     which traps on workerd's single-isolate WASI runtime.
   - `napi_call_function` `recv` arg switched from `napi_get_undefined`
     to `napi_get_null` (in `threadsafe_function.rs` and
     `bindgen_runtime/js_values/function.rs`) — workerd's strict-mode V8
     brand-checks host-bound methods, and `null` is the standard NAPI
     convention that other call sites in the same file already use.
2. **Custom wasi runtime polyfill** (`crates/blazen-core/src/runtime.rs`)
   provides `JoinHandle` / `JoinSet` / `sleep` / `timeout` whose
   semantics match `tokio::*` but route through `register_spawner` /
   `register_sleeper` hooks supplied by the host. The `JoinHandle`
   abort signal uses `tokio::sync::Notify` rather than
   `oneshot::Sender<()>` so dropping the handle does NOT abort the
   spawned task — matching native tokio's detached-on-drop behaviour.
   Without this, `runtime::spawn(execute_pipeline(...))` would silently
   kill the pipeline executor as soon as the JoinHandle went out of
   scope.
3. **JS-microtask dispatcher**
   (`crates/blazen-node/src/wasi_async.rs`) registers the spawner as a
   thread-local queue drained from a `Promise.resolve().then(...)`
   `ThreadsafeFunction`, plus a `setTimeout`-backed sleeper for
   `runtime::sleep` / `runtime::timeout`. Calls
   `napi::bindgen_prelude::set_async_executor` so napi-rs's own
   `#[napi] async fn` lowering uses the same dispatcher.
4. **emnapi runtime patch** (`patches/@emnapi+runtime+1.10.0.patch`,
   applied automatically via `patch-package` postinstall). Wraps
   `_setImmediate` so the `feature.setImmediate(cb)` call site invokes
   workerd's polyfilled `globalThis.setImmediate` with the right `this`
   binding instead of inheriting `feature` and tripping the V8
   brand-check.

The fix to abort the workflow's `usage_accumulator` task on wasm32 in
`crates/blazen-core/src/handler.rs::result` is also load-bearing: on
native the broadcast channel closes naturally once every `Sender`
clone drops with the event loop, but on wasi/Workers a step handler's
`JsContext` clone can outlive the event loop until JS GC reclaims it.
Without the explicit abort, each completed workflow leaves a
permanently-Pending accumulator task in the queue, re-scheduling
microtask drains forever and tripping workerd's "code had hung"
detector on the next request.

## Running the harness

```bash
cd examples/cloudflare-worker-wasi
npm install            # postinstall applies patches/@emnapi+runtime+1.10.0.patch
npm run e2e            # build wasi binary + boot wrangler dev + run all probes
npm run e2e:no-build   # skip the napi build (use the existing .wasm)
```

Direct invocations:

```bash
./dev-e2e.sh                            # build + run all
./dev-e2e.sh --no-build                 # skip build
./dev-e2e.sh --filter=workflow.run      # run a single probe
WRANGLER_PORT=8800 ./dev-e2e.sh         # change port
OPENAI_API_KEY=sk-... ./dev-e2e.sh      # include the openai probe
```

The wrangler stderr is captured to `.dev/wrangler.log` (gitignored) and
diagnostic lines (`ERROR`, `panic`, `Illegal`, `RuntimeError`,
`unreachable`, `[blazen` traces) are surfaced after the run.

## See also

- Working browser-SDK example: `../cloudflare-worker/`
- Node.js smoke (everything works):
  `node -e "const b = require('./crates/blazen-node/blazen.wasi.cjs'); console.log(b.version())"`
  from repo root.
