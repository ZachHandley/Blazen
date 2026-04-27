# @blazen/sdk

TypeScript/JavaScript SDK for Blazen via WebAssembly. Run the actual Blazen workflow engine — the same `blazen_core` Rust code that powers the native Python and Node bindings — directly in browsers, Node.js, Deno, and Cloudflare Workers. No native dependencies.

## Installation

```bash
npm install blazen-wasm-sdk
# pnpm add blazen-wasm-sdk
# yarn add blazen-wasm-sdk
```

## Quick start

```typescript
import { Workflow } from "blazen-wasm-sdk";

const wf = new Workflow("greeter");

wf.addStep("parse", ["blazen::StartEvent"], (event) => ({
  type: "GreetEvent",
  name: event?.data?.name ?? "World",
}));

wf.addStep("greet", ["GreetEvent"], (event) => ({
  type: "StopEvent",
  result: { greeting: `Hello, ${event.name}!` },
}));

const result = await wf.run({ name: "Zach" });
console.log(result); // { greeting: "Hello, Zach!" }
```

The `StartEvent` is the engine's built-in entry point and is interned as `"blazen::StartEvent"`. Returning an object with `type: "StopEvent"` ends the workflow; its `result` field is the value `run()` resolves to.

## Cloudflare Workers

A working end-to-end example lives at [`../../examples/cloudflare-worker/`](../../examples/cloudflare-worker/). The minimum:

```typescript
import { initSync, Workflow } from "blazen-wasm-sdk";
// @ts-expect-error - wasm import has no TS types
import wasmModule from "blazen-wasm-sdk/blazen_wasm_sdk_bg.wasm";

initSync({ module: wasmModule as WebAssembly.Module });

export default {
  async fetch(): Promise<Response> {
    const wf = new Workflow("hello");
    wf.addStep("go", ["blazen::StartEvent"], () => ({
      type: "StopEvent",
      result: { ok: true },
    }));
    return new Response(JSON.stringify(await wf.run({})));
  },
};
```

`wrangler.toml` needs:

```toml
[[rules]]
type = "CompiledWasm"
globs = ["**/*.wasm"]
fallthrough = true
```

Workerd's CPU-time caps (10ms free / 30s paid) mean long multi-step LLM workflows need the paid tier or Durable Objects to stretch across requests.

## Advanced: handler-based control

`runHandler` returns a live `WorkflowHandler` instead of awaiting the result, so you can pause, snapshot, stream events, or cancel mid-flight:

```typescript
const handler = await wf.runHandler(input);

// Stream events as they're emitted.
for (let ev = await handler.nextEvent(); ev !== null; ev = await handler.nextEvent()) {
  console.log(ev);
}

// Or pause and capture a snapshot for later resumption.
const snapshot = await handler.pause();

// Resume on a freshly-built workflow with the same step registrations
// (handlers can't be serialised, so they have to be re-registered).
const resumed = new Workflow("greeter");
resumed.addStep("parse", ["blazen::StartEvent"], parseHandler);
resumed.addStep("greet", ["GreetEvent"], greetHandler);

const resumedHandler = await resumed.resumeFromSnapshot(snapshot);
const result = await resumedHandler.awaitResult();
```

`handler.cancel()` tears the loop down. `handler.runId()` returns the run UUID.

## Context API

Step handlers receive `(event, context)` where `context` is a `WorkflowContext`. (The class is exported as `WorkflowContext` rather than `Context` because of a wasm-bindgen 0.2.118 cli-support bug that mangles class names containing `Wasm` — do not rename.)

```typescript
ctx.state.get(key, defaultValue);   // Persistable, JSON-typed.
ctx.state.set(key, value);
ctx.state.keys();

ctx.session.get(key, defaultValue); // In-process scratch space.
ctx.session.set(key, value);
ctx.session.keys();

ctx.metadata.get(key, defaultValue);
ctx.metadata.set(key, value);       // run_id / workflow_name are read-only.

ctx.sendEvent({ type: "MyEvent", payload: 1 }); // Routes through step registry.
ctx.writeEventToStream({ type: "Progress" });   // Broadcast-only, no routing.
ctx.runId;
ctx.workflowName;
```

All `Context` methods are synchronous on the JS side (no `await`). The WASM build is single-threaded, so the underlying `RwLock` polls without contention.

## v1 limitations

- **No `persist`-feature checkpointing.** Snapshot/resume works in-memory via `runHandler` + `pause()` + `resumeFromSnapshot()`, but the Rust `persist` feature's `CheckpointStore` integration is not enabled in the WASM build (it requires native dependencies). For crash-resilient checkpointing, use the native bindings or persist `pause()` snapshots to your own store.
- **JS reference identity is NOT preserved through `ctx.session`.** The wrapped `blazen_core::Context` stores session values as `serde_json::Value`, so `ctx.session.set(key, obj); ctx.session.get(key) !== obj`. The pre-stage-2 simplified `WasmWorkflow` preserved identity; the real engine does not. Keep live references in your own JS-side scope, not in the workflow context.
- **Parallel CPU-bound steps serialize.** The wasm32 `LocalJoinSet` polls all in-flight handlers on a single V8 isolate thread. Concurrent I/O (multiple `fetch` calls in parallel steps) still works fine; CPU-bound work in parallel steps does not.
- **Workers CPU-time caps.** Cloudflare Workers limits per-request CPU to 10ms (free) / 30s (paid). Multi-step LLM workflows can exhaust this — use Durable Objects or the native bindings for long flows.

## Build from source

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/). From `crates/blazen-wasm-sdk/`:

```bash
wasm-pack build --target web --release
```

Output is written to `pkg/`. We standardised on `--target web` because the `--target bundler` output's top-level `wasm.__wbindgen_start()` doesn't run under esbuild/Wrangler bundling.

## Docs

See [blazen.dev/docs](https://blazen.dev/docs/) or the source under `crates/blazen-wasm-sdk/src/`.

## License

AGPL-3.0-or-later
