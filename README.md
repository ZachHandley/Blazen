<p align="center">
  <h1 align="center">Blazen</h1>
  <p align="center">Event-driven AI workflow engine with first-class LLM integration.<br/>Written in Rust. Native bindings for Python, TypeScript, WebAssembly, Go, Swift, Kotlin, and Ruby.</p>
</p>

<p align="center">
  <a href="https://crates.io/crates/blazen"><img alt="crates.io" src="https://img.shields.io/crates/v/blazen.svg?style=flat-square&logo=rust&label=crates.io" /></a>
  <a href="https://pypi.org/project/blazen/"><img alt="PyPI" src="https://img.shields.io/pypi/v/blazen.svg?style=flat-square&logo=python&label=PyPI" /></a>
  <a href="https://www.npmjs.com/package/blazen"><img alt="npm" src="https://img.shields.io/npm/v/blazen.svg?style=flat-square&logo=npm&label=npm" /></a>
  <a href="https://www.npmjs.com/package/@blazen-dev/wasm"><img alt="npm wasm" src="https://img.shields.io/npm/v/@blazen-dev/wasm.svg?style=flat-square&logo=webassembly&label=wasm" /></a>
  <a href="https://github.com/ZachHandley/Blazen/blob/main/LICENSE"><img alt="License: MPL-2.0" src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" /></a>
</p>

---

## Features

- **Event-driven architecture** -- Type-safe events connect workflow steps with zero boilerplate via derive macros (Rust) or subclassing (Python) or plain objects (TypeScript)
- **15+ LLM providers** -- OpenAI, Anthropic, Gemini, Azure, OpenRouter, Groq, Together AI, Mistral, DeepSeek, Fireworks, Perplexity, xAI, Cohere, AWS Bedrock, and fal.ai -- with streaming, tool calling, structured output, and multimodal support
- **Content handles for tools** -- Tools accept multimodal inputs (image, audio, video, document, 3D, CAD) via typed content handles backed by a pluggable `ContentStore` (in-memory, local-file, OpenAI Files, Anthropic Files, Gemini Files, fal.ai storage, or your own). Tool *results* now carry multimodal payloads on every provider, not just Anthropic
- **Multi-workflow pipelines** -- Orchestrate sequential and parallel stages with pause/resume and per-workflow streaming
- **Branching and fan-out** -- Conditional branching, parallel fan-out, and real-time streaming within workflows
- **Native Python and TypeScript bindings** -- Python via PyO3/maturin, Node.js/TypeScript via napi-rs. Not wrappers around HTTP -- actual compiled Rust running in-process
- **WebAssembly SDK** -- Run Blazen in the browser, edge workers, Deno, and embedded runtimes via `@blazen-dev/wasm`. Same Rust core compiled to WASM
- **Prompt management** -- Versioned prompt templates with `{{variable}}` interpolation, YAML/JSON registries, and multimodal attachments
- **Persistence** -- Embedded persistence via redb, or bring-your-own via callbacks. Pause a workflow, serialize state to JSON, resume later
- **Identity-preserving live state** -- Pass DB connections, Pydantic models, and other live objects through events and the new `ctx.state` / `ctx.session` namespaces. `StopEvent(result=obj)` round-trips non-JSON Python values with `is`-identity preserved -- the engine no longer silently stringifies unpicklable results
- **Typed error hierarchies** -- Both Python and Node ship a full subclass tree (`BlazenError` plus ~87 leaves like `RateLimitError`, `LlamaCppError`, `MistralRsError`, `CandleLlmError`, `WhisperCppError`, `PiperError`, `DiffusionError`) so callers can write idiomatic `except RateLimitError` / `catch (e instanceof RateLimitError)` instead of string-matching messages
- **Bindings parity** -- `tools/audit_bindings.py` walks every public Rust symbol across all `blazen-*` crates and verifies the Python, Node, and WASM-SDK surfaces mirror it 1:1. The current report is `0 / 0 / 0` gaps, and CI fails on regression, so the bindings stay in lockstep with the Rust core
- **Observability** -- OpenTelemetry spans (OTLP gRPC and OTLP HTTP, the latter wasm-eligible), Prometheus metrics, and Langfuse all ship as opt-in features in `blazen-telemetry` -- enable an exporter, point it at your collector, and every step, LLM call, and pipeline stage is instrumented automatically

## Installation

**Rust:**

```bash
cargo add blazen
```

**Python** (requires Python 3.9+):

```bash
uv add blazen       # recommended
pip install blazen   # also works
```

**Node.js / TypeScript:**

```bash
# Umbrella package — auto-resolves the right artifact for your target.
# On Node it loads the native `.node` addon. On Deno (and other wasi
# hosts that polyfill node:wasi) it falls back to the wasi sidecar.
# Cloudflare Workers cannot load the wasi sidecar today (workerd's
# nodejs_compat doesn't polyfill node:wasi or expose MessageChannel —
# see the "Cloudflare Workers" section below); Workers users should
# install `@blazen-dev/wasm` instead.
pnpm add blazen
```

**WebAssembly — browser-style WASM SDK** (`wasm32-unknown-unknown`, wasm-pack output, no host imports beyond `fetch`):

```bash
npm install @blazen-dev/wasm
```

**WebAssembly — wasi sidecar** (`wasm32-wasip1`; explicit pin re-exporting `@blazen-dev/blazen-wasm32-wasi`):

```bash
npm install @blazen-dev/wasi
```

Most users want `blazen` (Node + auto-wasi resolution) or `@blazen-dev/wasm` (browser, Deno, ESM-only hosts). Pin `@blazen-dev/wasi` directly only if you need the wasi binary without the Node umbrella's resolution logic.

**Go** (requires Go 1.22+, with cgo enabled):

```bash
go get github.com/zachhandley/Blazen/bindings/go@latest
```

Ships a prebuilt static `libblazen_uniffi.a` under `internal/clib/<GOOS>_<GOARCH>/`. Linux amd64 and arm64 are shipped today; Windows and macOS land via CI release builds.

**Swift** (Linux 5.10+, macOS 13+):

```swift
// Package.swift dependencies
.package(url: "https://github.com/zachhandley/Blazen", from: "0.1.0")
// or pin via the bindings/swift/v* tag.
```

**Kotlin / JVM** (Java 17+):

```kotlin
// build.gradle.kts — Maven Central publishing pending; currently consume via git
implementation("dev.zorpx.blazen:blazen-kotlin:0.1.0")
```

**Ruby** (requires Ruby 3.1+ with the `ffi` and `async` gems):

```bash
gem install blazen
```

```ruby
require 'blazen'

Blazen.init

workflow = Blazen.workflow('hello') do |b|
  b.step('echo', accepts: ['blazen::StartEvent'], emits: ['blazen::StopEvent']) do |evt|
    Blazen::Workflow::StepOutput.single(
      Blazen::Workflow::Event.create(event_type: 'blazen::StopEvent', data: { msg: 'hello' })
    )
  end
end

result = workflow.run_blocking({})
```

Ships a prebuilt `libblazen_cabi` shared library under `ext/blazen/` matching the host's `<os>_<arch>` triple (linux amd64/arm64 today; macOS and Windows via CI release builds). The Ruby binding goes through a hand-written cbindgen-generated C ABI (`crates/blazen-cabi`) and the `ffi` gem, with `Fiber.scheduler`-aware async so workflows compose with the `async` gem out of the box. Full `StepHandler`, `ToolHandler`, and `CompletionStreamSink` callback support.

## Quick Start

### Rust

```rust
use blazen::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct GreetEvent {
    name: String,
}

#[step]
async fn parse_input(event: StartEvent, _ctx: Context) -> Result<GreetEvent, WorkflowError> {
    let name = event.data["name"].as_str().unwrap_or("World").to_string();
    Ok(GreetEvent { name })
}

#[step]
async fn greet(event: GreetEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    Ok(StopEvent {
        result: serde_json::json!({ "greeting": format!("Hello, {}!", event.name) }),
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let workflow = WorkflowBuilder::new("greeter")
        .step(parse_input_registration())
        .step(greet_registration())
        .build()?;

    let handler = workflow.run(serde_json::json!({ "name": "Zach" })).await?;
    let result = handler.result().await?;

    if let Some(stop) = result.event.downcast_ref::<StopEvent>() {
        println!("{}", stop.result); // {"greeting": "Hello, Zach!"}
    }
    Ok(())
}
```

### Python

```python
from blazen import Workflow, step, Event, StartEvent, StopEvent, Context

class GreetEvent(Event):
    name: str

@step
async def parse_input(ctx: Context, ev: StartEvent) -> GreetEvent:
    return GreetEvent(name=ev.name or "World")

@step
async def greet(ctx: Context, ev: GreetEvent) -> StopEvent:
    return StopEvent(result={"greeting": f"Hello, {ev.name}!"})

async def main():
    wf = Workflow("greeter", [parse_input, greet])
    handler = await wf.run(name="Zach")
    result = await handler.result()
    print(result.to_dict())  # {"result": {"greeting": "Hello, Zach!"}}

import asyncio
asyncio.run(main())
```

### TypeScript

```typescript
import { Workflow } from "blazen";

const workflow = new Workflow("greeter");

workflow.addStep("parse_input", ["blazen::StartEvent"], async (event, ctx) => {
  const name = event.name ?? "World";
  return { type: "GreetEvent", name };
});

workflow.addStep("greet", ["GreetEvent"], async (event, ctx) => {
  return {
    type: "blazen::StopEvent",
    result: { greeting: `Hello, ${event.name}!` },
  };
});

const result = await workflow.run({ name: "Zach" });
console.log(result.data); // { greeting: "Hello, Zach!" }
```

### Cloudflare Workers

Blazen runs the full workflow engine inside Cloudflare Workers via `@blazen-dev/wasm`. Multi-step LLM workflows, agents, and pipelines all execute on workerd -- Cloudflare's production runtime -- with no special configuration beyond `wasm-pack build --target web --release` and passing the compiled `WebAssembly.Module` to `initSync` at module load.

```typescript
import { initSync, Workflow } from "@blazen-dev/wasm";
// Wrangler resolves `*.wasm` imports as `WebAssembly.Module` instances.
import wasmModule from "@blazen-dev/wasm/blazen_wasm_sdk_bg.wasm";

initSync({ module: wasmModule as WebAssembly.Module });

export default {
  async fetch(): Promise<Response> {
    const wf = new Workflow("greeter");

    wf.addStep("parse", ["blazen::StartEvent"], (event: any) => ({
      type: "GreetEvent",
      name: event?.data?.name ?? "World",
    }));

    wf.addStep("greet", ["GreetEvent"], (event: any) => ({
      type: "StopEvent",
      result: { greeting: `Hello, ${event.name}!` },
    }));

    const result = await wf.run({});
    return Response.json(result);
  },
};
```

A complete runnable setup -- `wrangler.toml`, `vitest` integration test exercising the worker against a real `workerd` instance, and the `wasm-pack` build wiring -- lives in [`examples/cloudflare-worker/`](examples/cloudflare-worker/). CI builds and tests it on every push, so the Workers target is a supported deployment surface, not aspirational.

**Wasi sidecar on Workers (partial — sync APIs only, async still blocked upstream):** With compat date `2026-04-01` and `nodejs_compat` enabled, the wasi sidecar (`@blazen-dev/blazen-wasm32-wasi`, what `blazen` falls back to on non-Node hosts) loads cleanly on workerd and every sync API works — `new Workflow`, `addStep`, `HttpClient.fromCallback`, `setDefaultHttpClient`, `UpstashBackend.create`, `HttpPeerClient.newHttp`. We shipped a JS-microtask async dispatcher (`crates/blazen-node/src/wasi_async.rs` registers a custom spawner via `blazen_core::runtime::register_spawner`; the host wires `globalThis.__blazenDrainAsyncQueue = blazen.__blazenDrainAsyncQueue` after import so the scheduler can call back into Rust) and migrated every direct `tokio::spawn` call site onto it. **The dispatcher is verifiably installed**, but `Workflow.run()` / `Workflow.runWithHandler()` / `Pipeline.start()` still panic with `unreachable` from the same upstream wasm offsets — most likely tokio's time wheel (`tokio::time::sleep`/`timeout` is still re-exported from tokio in the wasi `runtime` impl) or napi-rs's `tokio_rt` glue around `#[napi] async fn`. `Workflow.runStreaming()` fails differently with an `Illegal invocation` on a TSfn → `ReadableByteStreamController.enqueue` callback. For full async workflow execution today, use `@blazen-dev/wasm`. The reproduction harness — `examples/cloudflare-worker-wasi/run-tests.mjs` — runs 13 probes (8 pass, 4 fail, 1 skip) and prints a pass/fail table.

Note: Cloudflare Workers cap CPU time per request (10ms on the free plan, up to 30s on paid plans). Long-running multi-call LLM flows should either fit within those limits, be split across requests using Blazen's pause/resume snapshots, or run on the WASIp2 component (`blazen-wasm`) for ZLayer edge deployment without the per-request cap.

#### Three install paths for non-native hosts

Blazen ships three npm artifacts so every JavaScript runtime has a first-class entry point:

| Package | Target | Use it when |
|---------|--------|-------------|
| `blazen` | Node-native + wasi auto-resolution | You want one install command that works on Node and on Deno (where `node:wasi` is implemented). On Node it loads the native `.node` addon; elsewhere it falls back to the wasi sidecar. **Does not currently work on Cloudflare Workers** — see "Cloudflare Workers" above; use `@blazen-dev/wasm` there. |
| `@blazen-dev/wasm` | Browser-style `wasm32-unknown-unknown` (wasm-pack output) | You're targeting a browser, **Cloudflare Workers**, Deno, or any host that talks to the SDK via `fetch` and `WebAssembly.Module`. Smallest surface, no wasi imports. |
| `@blazen-dev/wasi` | `wasm32-wasip1` sidecar (re-exports `@blazen-dev/blazen-wasm32-wasi`) | You want the wasi binary directly, without going through the `blazen` umbrella's resolution logic. Useful for Deno, StackBlitz/WebContainers, and Node fallback installs where prebuilt native binaries aren't available. |

#### wasi runtime requirement: register an `HttpClient`

The wasi build has no built-in HTTP stack — it cannot link `reqwest`'s wasm32 path because that pulls `wasm-bindgen`, which is incompatible with wasi. So **before any cloud LLM, OTLP exporter, Langfuse exporter, or peer-HTTP call**, the host must register a callback-backed `HttpClient` that proxies requests through the runtime's native `fetch` (or equivalent). Once registered, every Blazen subsystem that needs HTTP routes through it transparently.

```javascript
import { HttpClient, setDefaultHttpClient } from 'blazen';

const client = HttpClient.fromCallback(async (req) => {
  const res = await fetch(req.url, {
    method: req.method,
    headers: Object.fromEntries(req.headers),
    body: req.body,
  });
  return {
    status: res.status,
    headers: [...res.headers.entries()],
    body: new Uint8Array(await res.arrayBuffer()),
  };
});
setDefaultHttpClient(client);
```

Forgetting this step is the most common wasi-host failure mode: cloud-LLM constructors will succeed lazily, but the first `complete()` call returns an `HttpClientNotRegisteredError`.

#### What works on wasi (and what doesn't)

Available on wasi:

- Workflow engine, pipelines, agents, prompts — full feature parity with the native build.
- Cloud LLM providers (OpenAI, Anthropic, Gemini, Azure, OpenRouter, Groq, Together, Mistral, DeepSeek, Fireworks, Perplexity, xAI, Cohere, Bedrock, fal.ai) — all route HTTP through the registered `HttpClient`.
- Tiktoken token counting (`TiktokenCounter`).
- In-memory and Upstash Redis memory backends — `Memory.localUpstash(restUrl, restToken)` for a one-shot setup, or `Memory.withUpstash(embedder, backend)` for a custom embedder. The Upstash Redis REST API is HTTP-only, so it works wherever `setDefaultHttpClient` is wired up.
- Distributed peer (HTTP/JSON client only) — `HttpPeerClient.newHttp(baseUrl, nodeId)` for talking to a remote peer cluster from a Worker.
- OTLP-HTTP and Langfuse telemetry exporters — both route through the registered `HttpClient`. The OTLP gRPC exporter (`tonic`-based) is native-only.
- Custom memory backends via the `JsMemoryBackend` subclass — works on every target including wasi.

Not available on wasi:

- Local ONNX embeddings (`tract`) — pulled `wasm-bindgen` transitively via `reqwest`'s wasm32 path; deferred until upstream provides a pure-tokio HTTP path. Use `@blazen-dev/wasm` for in-browser tract embeddings, or call a remote embedding model through `setDefaultHttpClient`.
- Persistent file-backed checkpoint store — Workers and most wasi hosts have no filesystem. Use the in-memory checkpoint store, or persist snapshots out-of-band via the JSON snapshot API.
- Local C/C++ inference — `llama.cpp`, `whisper.cpp`, `piper`, `candle`, `fastembed`/`ort`. None compile to wasi.
- Native FS model cache — the wasi cache is in-memory only. Models redownload per worker instance unless you front them with a CDN/KV cache yourself.
- Prometheus exporter — there's no listening socket on Workers. Expose metrics via your own route handler, or use the OTLP-HTTP exporter and push to a collector instead.

#### Working example

A complete `wrangler` + `vitest` + `wasm-pack` setup using `@blazen-dev/wasm` lives in [`examples/cloudflare-worker/`](examples/cloudflare-worker/) and is exercised by CI on every push. A parallel example pinning `@blazen-dev/wasi` directly (rather than the browser-style SDK) may be added once the umbrella `blazen` package's wasi auto-resolution stabilizes across wrangler versions.

### WASM SDK feature parity

`@blazen-dev/wasm` is no longer the "lite" sibling. It now matches the Node binding for every workflow, pipeline, and handler primitive that makes sense in a browser or Worker:

- **Pipelines** -- `input_mapper`, `condition`, `onPersist`, and `onPersistJson` callbacks for sequential and parallel stages
- **Workflows** -- `setSessionPausePolicy`, `runStreaming(input, onEvent)`, `runWithHandler(input)`, and `resumeWithSerializableRefs(snapshot, refs)`
- **Handlers** -- `respondToInput`, `snapshot`, `resumeInPlace`, `streamEvents(callback)`, and `abort` on the returned handle
- **Context** -- session-ref serialization round-trips opaque host values across pause/resume the same way the Node and Python bindings do
- **In-browser embeddings** -- `TractEmbedModel.create(modelUrl, tokenizerUrl)` loads an ONNX embedding model and a HuggingFace tokenizer from URLs and runs inference on the CPU via `tract`, so RAG and semantic-memory flows work in the browser with no server round-trip

If a workflow runs against the Node binding, the same code path runs under `@blazen-dev/wasm` -- the only differences are the runtime-specific wiring (Node `fs` vs. browser `fetch`).

## LLM Integration

Every provider implements the same `CompletionModel` trait/interface. Switch providers by changing one line.

### Rust

```rust
use blazen_llm::{CompletionModel, CompletionRequest, ChatMessage};
use blazen_llm::providers::openai::OpenAiProvider;

let model = OpenAiProvider::new("sk-...");
let request = CompletionRequest::new(vec![
    ChatMessage::user("What is the meaning of life?"),
]);
let response = model.complete(request).await?;
println!("{}", response.content.unwrap_or_default());
```

Use any OpenAI-compatible provider with `OpenAiCompatProvider`:

```rust
use blazen_llm::providers::openai_compat::OpenAiCompatProvider;

let groq = OpenAiCompatProvider::groq("gsk-...");
let openrouter = OpenAiCompatProvider::openrouter("sk-or-...");
let together = OpenAiCompatProvider::together("...");
let deepseek = OpenAiCompatProvider::deepseek("...");
```

### Python

```python
from blazen import CompletionModel, ChatMessage, Role, CompletionResponse, ProviderOptions

model = CompletionModel.openai(options=ProviderOptions(api_key="sk-..."))
# or: CompletionModel.anthropic(options=ProviderOptions(api_key="sk-ant-..."))
# or: CompletionModel.groq(options=ProviderOptions(api_key="gsk-..."))
# or: CompletionModel.openrouter(options=ProviderOptions(api_key="sk-or-..."))
# or with env vars: CompletionModel.openai()

response: CompletionResponse = await model.complete([
    ChatMessage.system("You are helpful."),
    ChatMessage.user("What is the meaning of life?"),
])
print(response.content)        # typed attribute access
print(response.model)          # model name used
print(response.usage)          # TokenUsage with .prompt_tokens, .completion_tokens, .total_tokens
print(response.finish_reason)
```

### TypeScript

```typescript
import { CompletionModel, ChatMessage, Role } from "blazen";
import type { CompletionResponse } from "blazen";

const model = CompletionModel.openai({ apiKey: "sk-..." });
// or: CompletionModel.anthropic({ apiKey: "sk-ant-..." })
// or: CompletionModel.groq({ apiKey: "gsk-..." })
// or: CompletionModel.openrouter({ apiKey: "sk-or-..." })
// or with env vars: CompletionModel.openai()

const response: CompletionResponse = await model.complete([
  ChatMessage.system("You are helpful."),
  ChatMessage.user("What is the meaning of life?"),
]);
console.log(response.content);      // string
console.log(response.model);        // model name used
console.log(response.usage);        // { promptTokens, completionTokens, totalTokens }
console.log(response.finishReason);
```

## Multimodal Tool I/O

Tools can declare typed multimodal *inputs* via the `image_input`, `audio_input`, `file_input`, `three_d_input`, `cad_input`, and `video_input` schema helpers, and return multimodal *results* by emitting an `LlmPayload::Parts` value mixing text, images, audio, video, documents, 3D meshes, and CAD geometry. Result payloads round-trip through every provider, not just Anthropic.

Inputs flow through a pluggable `ContentStore`. You register a blob, URL, or remote-file reference with the store and receive a stable handle id; the model sees that id in the tool's JSON schema and emits it back in the tool call. Blazen's runner resolves the handle against the store and substitutes the typed content into the tool arguments before the user-supplied handler executes -- handlers never deal with raw blob plumbing.

### Rust

```rust
use blazen_llm::content::{InMemoryContentStore, ContentStore, ContentBody, ContentHint, ContentKind};
use blazen_llm::content::tool_input::image_input;
use blazen_llm::types::{ToolDefinition, CompletionRequest, ChatMessage};
use std::sync::Arc;

let store: Arc<dyn ContentStore> = Arc::new(InMemoryContentStore::new());

let handle = store
    .put(
        ContentBody::Url("https://example.com/cat.png".into()),
        ContentHint::default()
            .with_kind(ContentKind::Image)
            .with_mime_type("image/png"),
    )
    .await?;

// Declare a tool that accepts a content handle as its `photo` argument.
let tool = ToolDefinition {
    name: "describe_image".into(),
    description: "Describe what's in the photo".into(),
    parameters: image_input("photo", "The image to describe"),
};

// The model emits {"photo": "<handle-id>"} as a tool call;
// Blazen's runner substitutes the resolved image content before
// the tool handler runs.
```

### Python

```python
from blazen import ContentStore, ContentKind, image_input

store = ContentStore.in_memory()
handle = await store.put(b"...png bytes...", kind=ContentKind.Image, mime_type="image/png")

# Tool declaration uses image_input() to advertise a content-ref input:
schema = image_input("photo", "The image to describe")
# -> {"type": "object", "properties": {"photo": {"type": "string", ..., "x-blazen-content-ref": {"kind": "image"}}}, "required": ["photo"]}
```

### TypeScript

```typescript
import { ContentStore, imageInput } from "blazen";

const store = ContentStore.inMemory();
const handle = await store.put(Buffer.from(pngBytes), {
  kind: "image",
  mimeType: "image/png",
});

// Tool input schema:
const schema = imageInput("photo", "The image to describe");
```

See `docs/guides/tool-multimodal/` for the cross-cutting guide and `docs/guides/{rust,python,node,wasm}/multimodal/` for per-language details.

## Streaming

Steps can publish intermediate events to an external stream via `write_event_to_stream` on the context. Consumers subscribe before awaiting the final result.

### Rust

```rust
#[step]
async fn process(event: StartEvent, ctx: Context) -> Result<StopEvent, WorkflowError> {
    for i in 0..3 {
        ctx.write_event_to_stream(ProgressEvent { step: i }).await;
    }
    Ok(StopEvent { result: serde_json::json!({"done": true}) })
}

// Consumer side:
let handler = workflow.run(input).await?;
let mut stream = handler.stream_events();
while let Some(event) = stream.next().await {
    println!("got: {:?}", event.event_type_id());
}
let result = handler.result().await?;
```

### Python

```python
@step
async def process(ctx: Context, ev: StartEvent) -> StopEvent:
    for i in range(3):
        ctx.write_event_to_stream(Event("ProgressEvent", step=i))
    return StopEvent(result={"done": True})

# Consumer side:
handler = await wf.run(message="go")
async for event in handler.stream_events():
    print(event.event_type, event.to_dict())
result = await handler.result()
```

### TypeScript

```typescript
// Using runStreaming with a callback:
const result = await workflow.runStreaming({ message: "go" }, (event) => {
  console.log("stream:", event.type, event);
});

// Or using the handler API:
const handler = await workflow.runWithHandler({ message: "go" });
await handler.streamEvents((event) => {
  console.log("stream:", event.type, event);
});
const result = await handler.result();
```

## Crate / Package Structure

| Crate | Description |
|-------|-------------|
| `blazen` | Umbrella crate re-exporting everything |
| `blazen-events` | Core event traits, `StartEvent`, `StopEvent`, `DynamicEvent`, and derive macro support |
| `blazen-macros` | `#[derive(Event)]` and `#[step]` proc macros |
| `blazen-core` | Workflow engine, context, step registry, pause/resume, and snapshots |
| `blazen-llm` | LLM provider abstraction -- `CompletionModel`, `StructuredOutput`, `EmbeddingModel`, `Tool` |
| `blazen-pipeline` | Multi-workflow pipeline orchestrator with sequential/parallel stages |
| `blazen-prompts` | Prompt template management with versioning and YAML/JSON registries |
| `blazen-memory` | Memory and vector store with LSH-based approximate nearest-neighbor retrieval |
| `blazen-memory-valkey` | Valkey/Redis backend for `blazen-memory` |
| `blazen-persist` | Optional persistence layer (redb) |
| `blazen-telemetry` | Observability: OpenTelemetry spans, Prometheus metrics, Langfuse, and LLM call history |
| `blazen-py` | Python bindings via PyO3/maturin (published to PyPI as `blazen`) |
| `blazen-node` | Node.js/TypeScript bindings via napi-rs (published to npm as `blazen`) |
| [`blazen-wasm-sdk`](crates/blazen-wasm-sdk/) | TypeScript/JS client SDK via WebAssembly (published to npm as `@blazen-dev/wasm`) |
| [`blazen-wasm`](crates/blazen-wasm/) | WASIp2 WASM component for ZLayer edge deployment |
| `blazen-cli` | CLI tool for scaffolding projects (`blazen init`) |

## Supported LLM Providers

| Provider | Constructor | Default Model |
|----------|-------------|---------------|
| OpenAI | `OpenAiProvider::new` / `.openai()` | `gpt-4.1` |
| Anthropic | `AnthropicProvider::new` / `.anthropic()` | `claude-sonnet-4-5-20250929` |
| Google Gemini | `GeminiProvider::new` / `.gemini()` | `gemini-2.5-flash` |
| Azure OpenAI | `AzureOpenAiProvider::new` / `.azure()` | (deployment-specific) |
| OpenRouter | `.openrouter()` | `openai/gpt-4.1` |
| Groq | `.groq()` | `llama-3.3-70b-versatile` |
| Together AI | `.together()` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Mistral | `.mistral()` | `mistral-large-latest` |
| DeepSeek | `.deepseek()` | `deepseek-chat` |
| Fireworks | `.fireworks()` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| Perplexity | `.perplexity()` | `sonar-pro` |
| xAI (Grok) | `.xai()` | `grok-3` |
| Cohere | `.cohere()` | `command-a-08-2025` |
| AWS Bedrock | `.bedrock()` | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| fal.ai | `FalProvider::new` / `.fal()` | (image generation) |

All OpenAI-compatible providers are accessible through `OpenAiCompatProvider` in Rust, or through static factory methods on `CompletionModel` in Python and TypeScript.

## Typed Errors

Every error the engine, the LLM layer, or a backend can raise has a dedicated subclass in both Python and Node, so callers branch on type instead of parsing strings. The hierarchy is rooted at `BlazenError` (extending the host language's base `Error` / `Exception`) and fans out to ~87 leaves covering provider failures (`RateLimitError`, `AuthError`, `ContextLengthError`), local-inference backends (`LlamaCppError`, `MistralRsError`, `CandleLlmError`, `WhisperCppError`, `PiperError`, `DiffusionError`), persistence (`PersistError`, `SnapshotError`), and workflow control flow (`StepNotFoundError`, `EventTypeMismatchError`, `WorkflowAbortedError`).

```python
from blazen import CompletionModel, RateLimitError, AuthError, BlazenError

try:
    response = await model.complete(messages)
except RateLimitError as e:
    await asyncio.sleep(e.retry_after or 5)
except AuthError:
    rotate_api_key()
except BlazenError as e:
    log.exception("blazen failure", err=e)
```

```typescript
import { RateLimitError, AuthError, BlazenError } from "blazen";

try {
  const response = await model.complete(messages);
} catch (e) {
  if (e instanceof RateLimitError) await sleep(e.retryAfter ?? 5_000);
  else if (e instanceof AuthError) rotateApiKey();
  else if (e instanceof BlazenError) log.error("blazen failure", e);
  else throw e;
}
```

## Telemetry Exporters

`blazen-telemetry` ships four exporters as opt-in Cargo features. Enable the ones you need; the rest stay out of your binary.

| Exporter | Feature flag | Notes |
|----------|--------------|-------|
| OTLP gRPC | `otlp-grpc` | Standard tonic-based exporter for native deployments |
| OTLP HTTP | `otlp-http` | Pure-`reqwest` exporter; works under `wasm32` for browser/Worker telemetry |
| Langfuse | `langfuse` | Native Langfuse trace and observation API for LLM-call attribution |
| Prometheus | `prometheus` | Pull-based metrics endpoint for token counts, step latency, and pipeline stage timings |

All exporters share the same `TelemetryConfig` and per-exporter config structs (`OtlpConfig`, `LangfuseConfig`, `PrometheusConfig`), so swapping backends is a config change, not a code rewrite.

## Documentation

Full documentation, guides, and API reference are available at **[blazen.dev/docs/getting-started/introduction](https://blazen.dev/docs/getting-started/introduction)**.

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) (MPL-2.0).

## Author

Built by [Zach Handley](https://github.com/ZachHandley).
