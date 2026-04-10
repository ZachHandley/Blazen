# @blazen/sdk

TypeScript/JavaScript SDK for Blazen via WebAssembly. Run the full Blazen workflow engine, LLM completion, agent loop, and pipeline directly in the browser, Node.js, Deno, or Cloudflare Workers -- no native dependencies required.

## Installation

```bash
npm install @blazen/sdk
```

## Quick start

```typescript
import init, { CompletionModel, ChatMessage } from "@blazen/sdk";

await init(); // load the WASM module

// WASM SDK reads API keys from environment variables (e.g. OPENROUTER_API_KEY).
const model = CompletionModel.openrouter();
const response = await model.complete([ChatMessage.user("Hello!")]);
console.log(response.content);
```

### Workflows

```typescript
import init, { Workflow } from "@blazen/sdk";

await init();

const workflow = new Workflow("greeter");

workflow.addStep("parse", ["StartEvent"], async (event) => {
  return { type: "GreetEvent", name: event.name ?? "World" };
});

workflow.addStep("greet", ["GreetEvent"], async (event) => {
  return {
    type: "StopEvent",
    result: { greeting: `Hello, ${event.name}!` },
  };
});

const result = await workflow.run({ name: "Zach" });
console.log(result.data);
```

## State vs Session

The `Context` exposes two explicit namespaces:

- **`ctx.state`** -- persistable values (survives snapshotting when the WASM runner gains snapshot support).
- **`ctx.session`** -- live in-process JS references. **Identity IS preserved** within a run because the WASM runtime is single-threaded; this is a meaningful differentiator from the Node bindings, where session values are routed through `serde_json::Value` and identity is NOT preserved (due to napi-rs threading constraints).

```typescript
import init, { Workflow } from "@blazen/sdk";

await init();

const wf = new Workflow("example");
wf.addStep("setup", ["blazen::StartEvent"], (event, ctx) => {
  ctx.state.set("counter", 5);
  const liveObj = { tag: "live" };
  ctx.session.set("shared", liveObj);
  console.log(ctx.session.get("shared") === liveObj); // true -- identity preserved
  return { type: "blazen::StopEvent", result: {} };
});
```

All `Context` methods are **synchronous** in WASM (no `async`/`await` needed -- WASM has no tokio runtime). Session values are deliberately excluded from any snapshot.

## Supported platforms

- Browsers (Chrome, Firefox, Safari, Edge)
- Node.js (v16+)
- Deno
- Cloudflare Workers / edge runtimes
- Any environment with WebAssembly support

## Build from source

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```bash
# Bundler target (default -- works with Vite, webpack, etc.)
wasm-pack build --target bundler --release

# Web target (standalone ES module)
wasm-pack build --target web --release

# Node.js target (CommonJS)
wasm-pack build --target nodejs --release
```

Output is written to the `pkg/` directory.

## Docs

Full documentation at [blazen.dev/docs/getting-started/introduction](https://blazen.dev/docs/getting-started/introduction).

## License

AGPL-3.0-or-later
