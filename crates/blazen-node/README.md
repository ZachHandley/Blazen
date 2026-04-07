# Blazen

[![npm version](https://img.shields.io/npm/v/blazen.svg)](https://www.npmjs.com/package/blazen)
[![Node >= 18](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)

Event-driven AI workflow engine for Node.js and TypeScript, powered by a Rust core via napi-rs. Define workflows as a graph of async steps connected by typed events. Built-in LLM integration, streaming, pause/resume, and fan-out.

---

## Installation

```sh
pnpm add blazen
# or
npm install blazen
```

No native compilation required -- prebuilt binaries are provided for Linux (x86_64, aarch64) and macOS (x86_64, Apple Silicon).

---

## Quick Start

A workflow is a directed graph of **steps**. Each step listens for one or more event types and returns the next event. The reserved types `"blazen::StartEvent"` and `"blazen::StopEvent"` mark the entry and exit points.

Events are plain objects with a `type` field. All other fields are your data.

```typescript
import { Workflow, Context } from "blazen";
import type { JsWorkflowResult } from "blazen";

const wf = new Workflow("hello");

wf.addStep("parse", ["blazen::StartEvent"], async (event: Record<string, any>, ctx: Context) => {
  return { type: "GreetEvent", name: event.name || "World" };
});

wf.addStep("greet", ["GreetEvent"], async (event: Record<string, any>, ctx: Context) => {
  return { type: "blazen::StopEvent", result: { greeting: `Hello, ${event.name}!` } };
});

const result: JsWorkflowResult = await wf.run({ name: "Blazen" });
console.log(result.type); // "blazen::StopEvent"
console.log(result.data); // { greeting: "Hello, Blazen!" }
```

**Key concepts:**

- `addStep(name, eventTypes, handler)` -- `eventTypes` is a `string[]` of event types this step handles.
- The handler receives `(event, ctx)` and returns the next event object, an array of events, or `null`.
- `result.type` is the final event type (typically `"blazen::StopEvent"`).
- `result.data` is the payload you passed as `result` inside the `StopEvent`.

---

## Multi-Step Workflows

Steps communicate by emitting custom events. Any step whose `eventTypes` list includes a given event type will be invoked when that event fires.

```typescript
import { Workflow } from "blazen";

const wf = new Workflow("pipeline");

wf.addStep("extract", ["blazen::StartEvent"], async (event, ctx) => {
  const raw = event.text;
  await ctx.set("raw", raw);
  return { type: "CleanEvent", text: raw.trim().toLowerCase() };
});

wf.addStep("analyze", ["CleanEvent"], async (event, ctx) => {
  const wordCount = event.text.split(/\s+/).length;
  return { type: "SummarizeEvent", text: event.text, wordCount };
});

wf.addStep("summarize", ["SummarizeEvent"], async (event, ctx) => {
  const raw = await ctx.get("raw");
  return {
    type: "blazen::StopEvent",
    result: {
      original: raw,
      cleaned: event.text,
      wordCount: event.wordCount,
    },
  };
});

const result = await wf.run({ text: "  Hello World  " });
console.log(result.data);
// { original: "  Hello World  ", cleaned: "hello world", wordCount: 2 }
```

---

## Event Streaming

Steps can push intermediate events to external consumers via `ctx.writeEventToStream()`. Use `runStreaming(input, callback)` to receive them as they arrive.

```typescript
import { Workflow } from "blazen";

const wf = new Workflow("streaming");

wf.addStep("process", ["blazen::StartEvent"], async (event, ctx) => {
  await ctx.writeEventToStream({ type: "Progress", message: "Starting..." });

  // ... do work ...

  await ctx.writeEventToStream({ type: "Progress", message: "Halfway done." });

  // ... more work ...

  await ctx.writeEventToStream({ type: "Progress", message: "Complete." });

  return { type: "blazen::StopEvent", result: { status: "done" } };
});

const result = await wf.runStreaming({}, (event) => {
  // Called for every event published via ctx.writeEventToStream()
  console.log(`[stream] ${event.type}: ${event.message}`);
});

console.log(result.data); // { status: "done" }
```

`writeEventToStream` publishes to external consumers only. It does **not** route events through the internal step registry. Use `ctx.sendEvent()` for internal routing.

---

## LLM Integration

`CompletionModel` provides a unified interface to 15 LLM providers. Create a model instance with a static factory method and call `complete()` or `completeWithOptions()`. All messages and responses are fully typed.

### ChatMessage and Role

Build messages with the `ChatMessage` class and `Role` enum:

```typescript
import { CompletionModel, ChatMessage, Role } from "blazen";
import type { CompletionResponse, ToolCall, TokenUsage } from "blazen";

const model = CompletionModel.openrouter(process.env.OPENROUTER_API_KEY!);

// Using static factory methods (recommended)
const response: CompletionResponse = await model.complete([
  ChatMessage.system("You are helpful."),
  ChatMessage.user("What is 2+2?"),
]);

console.log(response.content);      // "4"
console.log(response.model);        // model name used
console.log(response.usage);        // TokenUsage: { promptTokens, completionTokens, totalTokens }
console.log(response.finishReason);  // "stop", "tool_calls", etc.
console.log(response.toolCalls);     // ToolCall[] | undefined
```

You can also construct messages with the `ChatMessage` constructor:

```typescript
const msg = new ChatMessage({ role: Role.User, content: "Hello" });
```

### Multimodal Messages

Send images alongside text using multimodal factory methods:

```typescript
// Image from URL
const msg = ChatMessage.userImageUrl("https://example.com/photo.jpg", "What's in this image?");

// Image from base64
const msg = ChatMessage.userImageBase64(base64Data, "image/png", "Describe this.");

// Multiple content parts
import type { ContentPart } from "blazen";
const msg = ChatMessage.userParts([
  { type: "text", text: "Compare these two images:" },
  { type: "image_url", imageUrl: { url: "https://example.com/a.jpg" } },
  { type: "image_url", imageUrl: { url: "https://example.com/b.jpg" } },
]);
```

### Advanced Options

Use `completeWithOptions` to control temperature, token limits, model selection, and tool definitions:

```typescript
import type { CompletionOptions } from "blazen";

const options: CompletionOptions = {
  temperature: 0.9,
  maxTokens: 256,
  topP: 0.95,
  model: "anthropic/claude-sonnet-4-20250514",
  tools: [/* tool definitions */],
};

const response = await model.completeWithOptions(
  [
    ChatMessage.system("You are a creative writer."),
    ChatMessage.user("Write a haiku about Rust."),
  ],
  options,
);
```

### All 15 Providers

| Factory Method | Provider |
|---|---|
| `CompletionModel.openai(apiKey)` | OpenAI |
| `CompletionModel.anthropic(apiKey)` | Anthropic |
| `CompletionModel.gemini(apiKey)` | Google Gemini |
| `CompletionModel.azure(apiKey, resourceName, deploymentName)` | Azure OpenAI |
| `CompletionModel.openrouter(apiKey)` | OpenRouter |
| `CompletionModel.groq(apiKey)` | Groq |
| `CompletionModel.together(apiKey)` | Together AI |
| `CompletionModel.mistral(apiKey)` | Mistral AI |
| `CompletionModel.deepseek(apiKey)` | DeepSeek |
| `CompletionModel.fireworks(apiKey)` | Fireworks AI |
| `CompletionModel.perplexity(apiKey)` | Perplexity |
| `CompletionModel.xai(apiKey)` | xAI / Grok |
| `CompletionModel.cohere(apiKey)` | Cohere |
| `CompletionModel.bedrock(apiKey, region)` | AWS Bedrock |
| `CompletionModel.fal(apiKey)` | fal.ai |

### Using LLMs Inside Workflows

```typescript
import { Workflow, CompletionModel, ChatMessage } from "blazen";

const model = CompletionModel.openai(process.env.OPENAI_API_KEY!);

const wf = new Workflow("llm-workflow");

wf.addStep("ask", ["blazen::StartEvent"], async (event, ctx) => {
  const response = await model.complete([
    ChatMessage.system("You are a helpful assistant."),
    ChatMessage.user(event.question),
  ]);
  return { type: "blazen::StopEvent", result: { answer: response.content } };
});

const result = await wf.run({ question: "What is the capital of France?" });
console.log(result.data.answer);
```

---

## Branching / Fan-Out

Return an array of events from a step handler to dispatch multiple events simultaneously. Each event routes to the step that handles its type.

```typescript
import { Workflow } from "blazen";

const wf = new Workflow("fan-out");

wf.addStep("split", ["blazen::StartEvent"], async (event, ctx) => {
  // Return an array to fan out into parallel branches
  return [
    { type: "BranchA", value: event.input },
    { type: "BranchB", value: event.input },
  ];
});

wf.addStep("handle_a", ["BranchA"], async (event, ctx) => {
  return { type: "blazen::StopEvent", result: { branch: "a", value: event.value } };
});

wf.addStep("handle_b", ["BranchB"], async (event, ctx) => {
  return { type: "blazen::StopEvent", result: { branch: "b", value: event.value } };
});

const result = await wf.run({ input: "data" });
// The first branch to produce a StopEvent wins
console.log(result.data);
```

---

## Side-Effect Steps

Return `null` from a step to perform side effects without emitting a return event. Use `ctx.sendEvent()` to manually route the next event through the internal step registry.

```typescript
import { Workflow } from "blazen";

const wf = new Workflow("side-effect");

wf.addStep("log_and_continue", ["blazen::StartEvent"], async (event, ctx) => {
  // Perform side effects
  await ctx.set("processed", true);

  // Manually send the next event
  await ctx.sendEvent({ type: "NextStep", data: event.input });

  // Return null -- no event emitted from the return value
  return null;
});

wf.addStep("finish", ["NextStep"], async (event, ctx) => {
  const processed = await ctx.get("processed");
  return { type: "blazen::StopEvent", result: { processed, data: event.data } };
});

const result = await wf.run({ input: "hello" });
console.log(result.data); // { processed: true, data: "hello" }
```

---

## Pause and Resume

`runWithHandler` returns a `WorkflowHandler` that gives you control over execution. Pause a workflow to serialize its full state as a JSON string, then resume it later -- even on a different machine.

```typescript
import { Workflow } from "blazen";
import { writeFileSync, readFileSync } from "fs";

const wf = new Workflow("pausable");

wf.addStep("work", ["blazen::StartEvent"], async (event, ctx) => {
  // ... expensive computation ...
  return { type: "blazen::StopEvent", result: { answer: 42 } };
});

// Start the workflow and get a handler
const handler = await wf.runWithHandler({ input: "data" });

// Pause and serialize the snapshot
const snapshot = await handler.pause();
writeFileSync("snapshot.json", snapshot);

// Later: resume from the snapshot
const saved = readFileSync("snapshot.json", "utf-8");
const resumedHandler = await wf.resume(saved);
const result = await resumedHandler.result();
console.log(result.data); // { answer: 42 }
```

**Important:** `handler.result()` and `handler.pause()` each consume the handler. You can only call one of them, and only once.

> **Note:** Values stored via `ctx.session.set(...)` are **excluded** from snapshots. The workflow's `session_pause_policy` (default `pickle_or_error`; other policies: `warn_drop`, `hard_error`) governs what happens to session entries at pause time -- see the Rust docs for policy details. For anything that must survive pause/resume, use `ctx.state.set(...)` (or the legacy `ctx.set(...)` shortcut).

### Human-in-the-Loop

Pause/resume is the foundation for human-in-the-loop workflows. Pause after a step to wait for human review, then resume when approved:

```typescript
const handler = await wf.runWithHandler({ document: rawText });

// Pause and persist until a human reviews
const snapshot = await handler.pause();
await db.saveSnapshot(jobId, snapshot);

// ... human reviews via UI ...

// Resume
const saved = await db.loadSnapshot(jobId);
const resumedHandler = await wf.resume(saved);
const result = await resumedHandler.result();
```

### Streaming with Handler

Use `handler.streamEvents()` to subscribe to intermediate events before calling `result()`:

```typescript
const handler = await wf.runWithHandler({ prompt: "Tell me a story." });

// Subscribe to stream events (must be called before result() or pause())
await handler.streamEvents((event) => {
  console.log("[stream]", event);
});

// Then await the final result
const result = await handler.result();
```

---

## Context API

Every step handler receives a `ctx` (Context) object. All methods are **async** and must be `await`ed.

```typescript
// Store a JSON-serializable value
await ctx.set("key", { any: "value" });

// Retrieve a stored value (returns null if not found)
const value = await ctx.get("key");

// Store raw binary data (no serialization requirement)
await ctx.setBytes("model-weights", buffer);

// Retrieve raw binary data (returns null if not found)
const data: Buffer | null = await ctx.getBytes("model-weights");

// Send an event through the internal step registry
await ctx.sendEvent({ type: "MyEvent", data: "..." });

// Publish an event to external streaming consumers (does NOT route internally)
await ctx.writeEventToStream({ type: "Progress", percent: 50 });

// Get the unique run ID for this workflow execution
const runId = await ctx.runId();
```

### Binary Storage

`setBytes` / `getBytes` let you store raw binary data in the context with no serialization requirement. Store any type by converting to bytes yourself (e.g., MessagePack, protobuf, or raw buffers). Binary data persists through pause/resume/checkpoint.

```typescript
// Store a raw buffer
const pixels = Buffer.from([0xff, 0x00, 0x00, 0xff]);
await ctx.setBytes("image-pixels", pixels);

// Retrieve it later in another step
const restored = await ctx.getBytes("image-pixels");
```

### State vs Session namespaces

The `Context` class exposes two explicit namespaces alongside the legacy smart-routing shortcuts (`ctx.set` / `ctx.get` / `ctx.setBytes` / `ctx.getBytes`):

- **`ctx.state`** -- persistable values. Routes through the same dispatch as `ctx.set` (bytes / JSON / pickle). Survives `pause()` / `resume()` and checkpoint stores.
- **`ctx.session`** -- in-process-only values. **Excluded from snapshots.** Use this for request IDs, rate-limit counters, ephemeral caches, and anything that should not survive pause/resume.

```typescript
wf.addStep("step", ["blazen::StartEvent"], async (event, ctx) => {
  // Persistable state
  await ctx.state.set("counter", 5);
  const count = await ctx.state.get("counter");

  // Bytes also work on the state namespace
  await ctx.state.setBytes("blob", Buffer.from([1, 2, 3]));
  const blob = await ctx.state.getBytes("blob");

  // In-process-only state
  await ctx.session.set("reqId", "abc123");
  const hasReq = await ctx.session.has("reqId");
  const reqId = await ctx.session.get("reqId");
  await ctx.session.remove("reqId");

  return { type: "blazen::StopEvent", result: { count, hasReq } };
});
```

**Important -- JS object identity is NOT preserved on Node.** Session values are routed through `serde_json::Value` because napi-rs's `Reference<T>` is `!Send` (its `Drop` must run on the v8 main thread). `await ctx.session.get("k")` returns a plain object equal to the one you passed in, not the same object. Session is still functionally distinct from state -- session values are excluded from snapshots, state values are not -- but for true identity preservation of live JS objects across steps you must use the Python or WASM bindings.

---

## Timeout

Set a workflow timeout in seconds. The default is 300 seconds (5 minutes). Set to 0 or negative to disable.

```typescript
const wf = new Workflow("my-workflow");
wf.setTimeout(60); // 60 second timeout
```

---

## TypeScript Support

Full TypeScript type definitions ship with the package -- no `@types` needed. All classes and interfaces are exported.

```typescript
import {
  Workflow, WorkflowHandler, Context, CompletionModel,
  ChatMessage, Role, version,
} from "blazen";
import type {
  JsWorkflowResult, CompletionResponse, CompletionOptions,
  ToolCall, TokenUsage, ContentPart, ImageContent, ImageSource,
} from "blazen";
```

---

## API Summary

| Export | Description |
|---|---|
| `Workflow` | Build and run event-driven workflows |
| `Workflow.addStep(name, eventTypes, handler)` | Register a step that handles specific event types |
| `Workflow.run(input)` | Run the workflow, returns `Promise<JsWorkflowResult>` |
| `Workflow.runStreaming(input, callback)` | Run with streaming, callback receives intermediate events |
| `Workflow.runWithHandler(input)` | Run and return a `WorkflowHandler` for pause/resume control |
| `Workflow.resume(snapshotJson)` | Resume a paused workflow from a JSON snapshot |
| `Workflow.setTimeout(seconds)` | Set workflow timeout in seconds |
| `WorkflowHandler` | Control handle for a running workflow |
| `WorkflowHandler.result()` | Await the final workflow result |
| `WorkflowHandler.pause()` | Pause and get a serialized snapshot string |
| `WorkflowHandler.streamEvents(callback)` | Subscribe to intermediate stream events |
| `Context` | Per-run shared state, event routing, and stream output |
| `Context.set(key, value)` | Store a JSON-serializable value (async) |
| `Context.get(key)` | Retrieve a value (async, returns null if missing) |
| `Context.setBytes(key, buffer)` | Store raw binary data (async) |
| `Context.getBytes(key)` | Retrieve raw binary data (async, returns null if missing) |
| `Context.sendEvent(event)` | Route an event to matching steps (async) |
| `Context.writeEventToStream(event)` | Publish to external stream consumers (async) |
| `Context.runId()` | Get the workflow run ID (async) |
| `Context.state` | `StateNamespace` getter -- persistable values (survives pause/resume) |
| `Context.session` | `SessionNamespace` getter -- in-process-only values (excluded from snapshots) |
| `StateNamespace.set / get / setBytes / getBytes` | Async persistable storage routed through the same dispatch as `ctx.set` |
| `SessionNamespace.set / get / has / remove` | Async in-process-only storage; values are routed through `serde_json::Value` (no JS identity preservation) |
| `CompletionModel` | Unified LLM client with 15 provider factory methods |
| `CompletionModel.complete(messages)` | Chat completion with typed `ChatMessage[]` input, returns `CompletionResponse` (async) |
| `CompletionModel.completeWithOptions(messages, opts)` | Chat completion with `CompletionOptions` (async) |
| `CompletionModel.modelId` | Getter for the current model ID |
| `ChatMessage` | Chat message class with static factories: `.system()`, `.user()`, `.assistant()`, `.tool()`, `.userImageUrl()`, `.userImageBase64()`, `.userParts()` |
| `Role` | String enum: `Role.System`, `Role.User`, `Role.Assistant`, `Role.Tool` |
| `CompletionResponse` | Interface: `{ content, toolCalls, usage, model, finishReason }` |
| `ToolCall` | Interface: `{ id, name, arguments }` |
| `TokenUsage` | Interface: `{ promptTokens, completionTokens, totalTokens }` |
| `CompletionOptions` | Interface: `{ temperature?, maxTokens?, topP?, model?, tools? }` |
| `ContentPart` / `ImageContent` / `ImageSource` | Types for multimodal message content |
| `JsWorkflowResult` | Interface: `{ type: string, data: any }` |
| `version()` | Returns the blazen library version string |

---

## Links

- [GitHub](https://github.com/ZachHandley/Blazen) -- source, issues, and advanced examples
- [blazen.dev](https://blazen.dev) -- documentation and guides

---

## License

[AGPL-3.0](https://opensource.org/licenses/AGPL-3.0)
