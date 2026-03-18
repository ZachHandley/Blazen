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

`CompletionModel` provides a unified interface to 15 LLM providers. Create a model instance with a static factory method and call `complete()` or `completeWithOptions()`.

```typescript
import { CompletionModel } from "blazen";

const model = CompletionModel.openrouter(process.env.OPENROUTER_API_KEY!);

const response = await model.complete([
  { role: "system", content: "You are helpful." },
  { role: "user", content: "What is 2+2?" },
]);

console.log(response.content);    // "4"
console.log(response.model);      // model name used
console.log(response.usage);      // { promptTokens, completionTokens, totalTokens }
console.log(response.finishReason);
```

### Advanced Options

Use `completeWithOptions` to control temperature, token limits, model selection, and tool definitions:

```typescript
const response = await model.completeWithOptions(
  [
    { role: "system", content: "You are a creative writer." },
    { role: "user", content: "Write a haiku about Rust." },
  ],
  {
    temperature: 0.9,
    maxTokens: 256,
    topP: 0.95,
    model: "anthropic/claude-sonnet-4-20250514",
    tools: [/* tool definitions */],
  }
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
import { Workflow, CompletionModel } from "blazen";

const model = CompletionModel.openai(process.env.OPENAI_API_KEY!);

const wf = new Workflow("llm-workflow");

wf.addStep("ask", ["blazen::StartEvent"], async (event, ctx) => {
  const response = await model.complete([
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: event.question },
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

// Send an event through the internal step registry
await ctx.sendEvent({ type: "MyEvent", data: "..." });

// Publish an event to external streaming consumers (does NOT route internally)
await ctx.writeEventToStream({ type: "Progress", percent: 50 });

// Get the unique run ID for this workflow execution
const runId = await ctx.runId();
```

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
import { Workflow, WorkflowHandler, Context, CompletionModel, version } from "blazen";
import type { JsWorkflowResult } from "blazen";
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
| `Context.set(key, value)` | Store a value (async) |
| `Context.get(key)` | Retrieve a value (async, returns null if missing) |
| `Context.sendEvent(event)` | Route an event to matching steps (async) |
| `Context.writeEventToStream(event)` | Publish to external stream consumers (async) |
| `Context.runId()` | Get the workflow run ID (async) |
| `CompletionModel` | Unified LLM client with 15 provider factory methods |
| `CompletionModel.complete(messages)` | Chat completion (async) |
| `CompletionModel.completeWithOptions(messages, opts)` | Chat completion with temperature, maxTokens, model, tools (async) |
| `CompletionModel.modelId` | Getter for the current model ID |
| `JsWorkflowResult` | Interface: `{ type: string, data: any }` |
| `version()` | Returns the blazen library version string |

---

## Links

- [GitHub](https://github.com/ZachHandley/Blazen) -- source, issues, and advanced examples
- [blazen.dev](https://blazen.dev) -- documentation and guides

---

## License

[AGPL-3.0](https://opensource.org/licenses/AGPL-3.0)
