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

## Workflow lifecycle: streaming, handlers, snapshots, HITL

The `Workflow` class exposes the full Blazen execution surface. The simple `wf.run(input)` shape is sugar over four orthogonal entry points:

```typescript
// 1. Fire-and-forget — resolves with the terminal result.
const result = await wf.run(input);

// 2. Forward each event to a callback as it is published.
const result = await wf.runStreaming(input, (ev) => {
  console.log(ev.event_type, ev.data);
});

// 3. Hand back a live handler for pause/snapshot/cancel.
const handler = await wf.runHandler(input);

// 4. Naming-parity alias for the Node binding's `runWithHandler`.
const handler = await wf.runWithHandler(input);
```

Every `Workflow` instance is single-shot — calling any `run*` method consumes the internal builder, so a second call rejects. Re-register the handlers on a fresh `Workflow` if you want to run again.

### `setSessionPausePolicy`

Live session refs (anything stored via `ctx.session.set(...)`) cannot be JSON-serialised. `setSessionPausePolicy` controls what happens when a workflow with live refs is paused or snapshotted:

```typescript
const wf = new Workflow("greeter");
wf.setSessionPausePolicy("pickle_or_serialize");
// "pickle_or_error" (default) | "pickle_or_serialize" | "warn_drop" | "hard_error"
```

The `PascalCase` Node spellings (`PickleOrError`, `PickleOrSerialize`, `WarnDrop`, `HardError`) are also accepted. The same setter exists on `WorkflowBuilder` for fluent configuration. With `pickle_or_serialize`, refs inserted via `ctx.insertSessionRefSerializable` are captured into the snapshot and can be restored via `resumeWithSerializableRefs`.

### `WorkflowHandler` API

`runHandler` / `runWithHandler` resolve to a `WorkflowHandler` wrapping the live engine loop:

```typescript
const handler = await wf.runHandler(input);

// Stream events without polling.
await handler.streamEvents((ev) => console.log(ev));

// Or pull one event at a time (resolves to null when the stream closes).
for (let ev = await handler.nextEvent(); ev !== null; ev = await handler.nextEvent()) {
  console.log(ev);
}

// Live snapshot without halting the loop (telemetry / introspection).
const live = await handler.snapshot();

// Quiescent snapshot: pause first, then capture, then resume in place.
const paused = await handler.pause();
await handler.resumeInPlace();

// Human-in-the-loop response to an InputRequestEvent.
handler.respondToInput(requestId, { confirmed: true });

// Stop the loop. abort() is an alias for cancel().
handler.abort();

// Final result (consumes the inner handler).
const result = await handler.awaitResult();
```

`handler.runId()` returns this run's UUID (it lazily snapshots on first call so the loop is not paused as a side effect).

### Resume + serializable refs

`resumeFromSnapshot` rehydrates a workflow whose session contained no live refs (or whose refs were dropped under `warn_drop`):

```typescript
const resumed = new Workflow("greeter");
resumed.addStep("parse", ["blazen::StartEvent"], parseHandler);
resumed.addStep("greet", ["GreetEvent"], greetHandler);
const handler2 = await resumed.resumeFromSnapshot(snapshot);
const result = await handler2.awaitResult();
```

`resumeWithSerializableRefs` is required when the original pause used `pickle_or_serialize`. JS callers pass a per-tag deserializer map; each callback receives the captured `Uint8Array` payload so the application can rebuild whatever live state the step handlers expect:

```typescript
const handler3 = await resumed.resumeWithSerializableRefs(snapshot, {
  "app::EmbeddingHandle": (bytes) => {
    rebuildEmbeddingHandleFromBytes(bytes);
  },
});
```

After resume, the same payloads are also retrievable from inside step handlers via `ctx.getSessionRefSerializable(key)` — the engine re-wraps them as opaque bytes so JS code keeps a bytes-in / bytes-out contract.

## Context API: bytes and serializable refs

In addition to the namespaces shown above, `Context` exposes raw-byte storage and a serializable-ref registry that integrates with `setSessionPausePolicy`:

```typescript
// Raw byte storage (survives snapshotting as StateValue::Bytes).
ctx.setBytes("blob", new Uint8Array([1, 2, 3]));
const bytes = ctx.getBytes("blob");

// Serializable session refs — bytes-in / bytes-out registry that
// integrates with setSessionPausePolicy("pickle_or_serialize").
const refKey = ctx.insertSessionRefSerializable(
  "app::EmbeddingHandle",
  serialiseHandle(handle),
);
const stored = ctx.getSessionRefSerializable(refKey);
// → { typeName: "app::EmbeddingHandle", bytes: Uint8Array(...) } or null
```

JS code is responsible for serialising into the `Uint8Array` it hands to `insertSessionRefSerializable` and for deserialising the bytes returned by `getSessionRefSerializable`. The engine treats the payload as opaque, which is what makes it compatible with `pickle_or_serialize` snapshotting and `resumeWithSerializableRefs`.

## Pipelines: stages, conditions, persistence

The `Pipeline` family lets you compose multiple workflows into a sequential or fan-out graph, with optional per-stage gating, input mapping, and a persist hook for IndexedDB-style checkpointing.

```typescript
import {
  PipelineBuilder,
  ParallelStage,
  Stage,
  Workflow,
  JoinStrategy,
} from "blazen-wasm-sdk";

const ingest = new Workflow("ingest");
ingest.addStep("go", ["blazen::StartEvent"], (ev) => ({
  type: "StopEvent",
  result: { rows: ev.data.rows ?? [] },
}));

const summarise = new Workflow("summarise");
summarise.addStep("go", ["blazen::StartEvent"], (ev) => ({
  type: "StopEvent",
  result: { summary: `${ev.data.rows.length} rows` },
}));

const pipeline = new PipelineBuilder("etl")
  .stage(new Stage("ingest", ingest))
  .stage(
    new Stage(
      "summarise",
      summarise,
      // input_mapper: shape the previous stage's output for this stage.
      (state) => ({ rows: state.lastOutput.rows }),
      // condition: skip the stage entirely when this returns false.
      (state) => state.lastOutput.rows.length > 0,
    ),
  )
  .timeoutPerStage(30)
  .onPersist(async (snapshot) => {
    await indexedDbPut("etl-checkpoint", snapshot);
  })
  .build();

const handler = await pipeline.start({ rows: [{ id: 1 }] });
handler.streamEvents((ev) => console.log(ev.stageName, ev.event));
const result = await handler.result();
```

`Stage` accepts two optional callbacks at construction:

- `input_mapper: (state: PipelineState) => unknown` — transform the rolling pipeline state into the workflow input. When omitted, the previous stage's output (or the pipeline input for the first stage) is forwarded directly.
- `condition: (state: PipelineState) => boolean` — when it returns `false`, the stage is skipped (`StageResult.skipped === true`, `output === null`).

Use `ParallelStage` to fan out across branches and join with `JoinStrategy.WaitAll` (default) or `JoinStrategy.FirstCompletes`.

### `onPersist` + IndexedDB

`PipelineBuilder.onPersist` and `onPersistJson` register a callback that fires after every stage. The first variant receives a typed `PipelineSnapshot` (decoded via `serde-wasm-bindgen`); the second receives a JSON string ready for stringly-typed stores. Both may return a `Promise` — the engine awaits it before continuing, so back-pressure is real.

```typescript
function openCheckpointStore(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("blazen-checkpoints", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("snapshots");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

const db = await openCheckpointStore();
const builder = new PipelineBuilder("ingestion")
  .stage(new Stage("fetch", fetchWorkflow))
  .onPersistJson(async (snapshotJson) => {
    const tx = db.transaction("snapshots", "readwrite");
    tx.objectStore("snapshots").put(snapshotJson, "ingestion");
    await new Promise((r) => (tx.oncomplete = r));
  });
```

`PipelineHandler` mirrors `WorkflowHandler` for live runs: `result()`, `pause()`, `streamEvents(callback)`, `snapshot()`, `resumeInPlace()`, `abort()`. Resuming from a stored snapshot is `pipeline.resume(snapshot)`.

## Memory, embeddings, and `TractEmbedModel`

`Memory` is Blazen's vector store; it works with any `EmbeddingModel` and any backend that implements the storage interface.

```typescript
import {
  EmbeddingModel,
  InMemoryBackend,
  Memory,
  MemoryResult,
  TractEmbedModel,
} from "blazen-wasm-sdk";

// Provider-agnostic embedders.
const cloudEmbedder = EmbeddingModel.openai();
const customEmbedder = EmbeddingModel.fromJsHandler(
  "all-MiniLM-L6-v2",
  384,
  async (texts) => {
    /* call transformers.js, ONNX Runtime Web, ... */
    return texts.map(() => new Float32Array(384));
  },
);
```

### `TractEmbedModel` — pure-WASM ONNX inference

`TractEmbedModel.create(modelUrl, tokenizerUrl, options?)` runs a `tract-onnx` graph entirely inside the WASM module — no JS-side ML runtime, no `hf-hub` (which is unavailable on `wasm32`). Both URLs must be served with permissive CORS:

```typescript
const tract = await TractEmbedModel.create(
  "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
  "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json",
);
const vectors = await tract.embed(["hello", "world"]);
console.log(tract.dimensions, tract.modelId);
```

`EmbeddingModel.tract(modelUrl, tokenizerUrl, options?)` is the equivalent static factory that produces a generic `EmbeddingModel`, so the result drops into anything that accepts `EmbeddingModel` (notably `Memory`).

### `InMemoryBackend` + `MemoryResult`

`InMemoryBackend` is a typed Rust-side backend that avoids per-call JS round-trips. Use the `Memory.fromBackend` / `Memory.localFromBackend` factories to wire it up, or stay with `new Memory(embedder)` for the default in-memory store:

```typescript
const backend = new InMemoryBackend();
const memory = Memory.fromBackend(cloudEmbedder, backend);

// Local-only (text SimHash; no embedding model required).
const localMemory = Memory.localFromBackend(new InMemoryBackend());

await memory.add("doc1", "Paris is the capital of France", { lang: "en" });
const hits = await memory.search("What is France's capital?", 5);
hits.forEach((hit) => console.log(hit.id, hit.score, hit.text));

// Construct MemoryResult instances directly when implementing a custom
// MemoryStore in JavaScript.
const synthetic = new MemoryResult("manual-1", "manual entry", 0.42, null);
```

`Memory.fromJsBackend(embedder, backend)` is still available when you want to back the store with IndexedDB, localStorage, or a remote API from JavaScript; see the d.ts comments on `Memory.fromJsBackend` for the required method shape.

## `ModelManager`

`ModelManager` wraps the real `blazen_manager::ModelManager` (LRU-backed VRAM accounting). The JS API is unchanged from the previous WASM stub, so existing code keeps working:

```typescript
import { ModelManager } from "blazen-wasm-sdk";

const manager = new ModelManager(8); // 8 GB budget
await manager.register("llama-3-8b", null, 6 * 1024 ** 3, {
  async load() { /* fetch weights, warm up runtime */ },
  async unload() { /* drop GPU buffers */ },
});

await manager.load("llama-3-8b");
console.log(await manager.isLoaded("llama-3-8b"));
const status = await manager.status();
await manager.unload("llama-3-8b");
```

Reads such as `availableBytes` / `usedBytes` are now `Promise<number>` because the upstream manager guards its budget behind a mutex; treat them as awaitables.

## OpenTelemetry: `OtlpConfig` + `initOtlp`

OTLP traces are exported over HTTP/protobuf via `WasmFetchHttpClient`, which sidesteps the wasm-incompat `tonic`/grpc stack. The exporter is gated behind the `otlp-http` Cargo feature; build the SDK with `wasm-pack build --features otlp-http` to enable it.

```typescript
import { OtlpConfig, initOtlp } from "blazen-wasm-sdk";

initOtlp(new OtlpConfig(
  "https://otel-collector.example.com/v1/traces",
  "blazen-frontend",
));
```

The WASM `OtlpConfig` constructor is positional (`new OtlpConfig(endpoint, serviceName)`), unlike the Node binding which accepts an options object. `initOtlp` installs a global `tracing-subscriber`, so call it exactly once at startup.

## Media types

`MediaSource` is a type alias for `ImageSource` — same `{ type: "url" | "base64" | "file"; ... }` discriminated union. Any API that accepts an `ImageSource` (provider request payloads, content parts, etc.) also accepts a `MediaSource`. The alias exists for symmetry with the native bindings, where media handling is unified across image / video / audio sources.

## v1 limitations

- **No `persist`-feature checkpointing.** Snapshot/resume works in-memory via `runHandler` + `pause()` + `resumeFromSnapshot()` (or `resumeWithSerializableRefs` for live refs), and `PipelineBuilder.onPersist` / `onPersistJson` give you a hook for IndexedDB-style stores. The Rust `persist` feature's `CheckpointStore` integration is not enabled in the WASM build (it requires native dependencies). For crash-resilient checkpointing, use the native bindings or persist `pause()` snapshots to your own store.
- **JS reference identity is NOT preserved through `ctx.session`.** The wrapped `blazen_core::Context` stores session values as `serde_json::Value`, so `ctx.session.set(key, obj); ctx.session.get(key) !== obj`. The pre-stage-2 simplified `WasmWorkflow` preserved identity; the real engine does not. Keep live references in your own JS-side scope, or push opaque bytes through `ctx.insertSessionRefSerializable` and pair with `setSessionPausePolicy("pickle_or_serialize")`.
- **Parallel CPU-bound steps serialize.** The wasm32 `LocalJoinSet` polls all in-flight handlers on a single V8 isolate thread. Concurrent I/O (multiple `fetch` calls in parallel steps) still works fine; CPU-bound work in parallel steps does not.
- **Workers CPU-time caps.** Cloudflare Workers limits per-request CPU to 10ms (free) / 30s (paid). Multi-step LLM workflows can exhaust this — use Durable Objects or the native bindings for long flows.
- **OTLP requires HTTP/protobuf collectors.** The grpc-tonic transport is unusable on `wasm32`; only collectors that accept HTTP/protobuf at their `/v1/traces` endpoint will work with `initOtlp`.

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
