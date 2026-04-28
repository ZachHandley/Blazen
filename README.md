<p align="center">
  <h1 align="center">Blazen</h1>
  <p align="center">Event-driven AI workflow engine with first-class LLM integration.<br/>Written in Rust. Native bindings for Python, TypeScript, and WebAssembly.</p>
</p>

<p align="center">
  <a href="https://crates.io/crates/blazen"><img alt="crates.io" src="https://img.shields.io/crates/v/blazen.svg?style=flat-square&logo=rust&label=crates.io" /></a>
  <a href="https://pypi.org/project/blazen/"><img alt="PyPI" src="https://img.shields.io/pypi/v/blazen.svg?style=flat-square&logo=python&label=PyPI" /></a>
  <a href="https://www.npmjs.com/package/blazen"><img alt="npm" src="https://img.shields.io/npm/v/blazen.svg?style=flat-square&logo=npm&label=npm" /></a>
  <a href="https://www.npmjs.com/package/@blazen/sdk"><img alt="npm wasm" src="https://img.shields.io/npm/v/@blazen/sdk.svg?style=flat-square&logo=webassembly&label=wasm" /></a>
  <a href="https://github.com/ZachHandley/Blazen/blob/main/LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" /></a>
</p>

---

## Features

- **Event-driven architecture** -- Type-safe events connect workflow steps with zero boilerplate via derive macros (Rust) or subclassing (Python) or plain objects (TypeScript)
- **15+ LLM providers** -- OpenAI, Anthropic, Gemini, Azure, OpenRouter, Groq, Together AI, Mistral, DeepSeek, Fireworks, Perplexity, xAI, Cohere, AWS Bedrock, and fal.ai -- with streaming, tool calling, structured output, and multimodal support
- **Multi-workflow pipelines** -- Orchestrate sequential and parallel stages with pause/resume and per-workflow streaming
- **Branching and fan-out** -- Conditional branching, parallel fan-out, and real-time streaming within workflows
- **Native Python and TypeScript bindings** -- Python via PyO3/maturin, Node.js/TypeScript via napi-rs. Not wrappers around HTTP -- actual compiled Rust running in-process
- **WebAssembly SDK** -- Run Blazen in the browser, edge workers, Deno, and embedded runtimes via `@blazen/sdk`. Same Rust core compiled to WASM
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
pnpm add blazen
```

**WebAssembly** (browser, edge, Deno, Cloudflare Workers):

```bash
npm install @blazen/sdk
```

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

Blazen runs the full workflow engine inside Cloudflare Workers via `@blazen/sdk`. Multi-step LLM workflows, agents, and pipelines all execute on workerd -- Cloudflare's production runtime -- with no special configuration beyond `wasm-pack build --target web --release` and passing the compiled `WebAssembly.Module` to `initSync` at module load.

```typescript
import { initSync, Workflow } from "@blazen/sdk";
// Wrangler resolves `*.wasm` imports as `WebAssembly.Module` instances.
import wasmModule from "@blazen/sdk/blazen_wasm_sdk_bg.wasm";

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

Note: Cloudflare Workers cap CPU time per request (10ms on the free plan, up to 30s on paid plans). Long-running multi-call LLM flows should either fit within those limits, be split across requests using Blazen's pause/resume snapshots, or run on the WASIp2 component (`blazen-wasm`) for ZLayer edge deployment without the per-request cap.

### WASM SDK feature parity

`@blazen/sdk` is no longer the "lite" sibling. It now matches the Node binding for every workflow, pipeline, and handler primitive that makes sense in a browser or Worker:

- **Pipelines** -- `input_mapper`, `condition`, `onPersist`, and `onPersistJson` callbacks for sequential and parallel stages
- **Workflows** -- `setSessionPausePolicy`, `runStreaming(input, onEvent)`, `runWithHandler(input)`, and `resumeWithSerializableRefs(snapshot, refs)`
- **Handlers** -- `respondToInput`, `snapshot`, `resumeInPlace`, `streamEvents(callback)`, and `abort` on the returned handle
- **Context** -- session-ref serialization round-trips opaque host values across pause/resume the same way the Node and Python bindings do
- **In-browser embeddings** -- `TractEmbedModel.create(modelUrl, tokenizerUrl)` loads an ONNX embedding model and a HuggingFace tokenizer from URLs and runs inference on the CPU via `tract`, so RAG and semantic-memory flows work in the browser with no server round-trip

If a workflow runs against the Node binding, the same code path runs under `@blazen/sdk` -- the only differences are the runtime-specific wiring (Node `fs` vs. browser `fetch`).

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
| [`blazen-wasm-sdk`](crates/blazen-wasm-sdk/) | TypeScript/JS client SDK via WebAssembly (published to npm as `@blazen/sdk`) |
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

This project is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) (AGPL-3.0).

## Author

Built by [Zach Handley](https://github.com/ZachHandley).
