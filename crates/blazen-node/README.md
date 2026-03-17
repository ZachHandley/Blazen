# Blazen

[![npm version](https://img.shields.io/npm/v/blazen.svg)](https://www.npmjs.com/package/blazen)
[![Node >= 18](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)

Event-driven AI workflow engine for Node.js, powered by a Rust core via napi-rs.

---

## Installation

```sh
pnpm add blazen
# or
npm install blazen
```

---

## Quick Start

Build a workflow by registering steps, each of which handles one or more event types and returns the next event. The reserved event types `blazen::StartEvent` and `blazen::StopEvent` mark the entry and exit points.

```typescript
import { Workflow } from "blazen";

const workflow = new Workflow("hello-world");

workflow.addStep("greet", ["blazen::StartEvent"], async (event, ctx) => {
  const name = event.name ?? "world";
  await ctx.set("greeted", name);
  return { type: "blazen::StopEvent", result: `Hello, ${name}!` };
});

const result = await workflow.run({ name: "Blazen" });
console.log(result.data); // "Hello, Blazen!"
```

---

## Multi-Step Workflows

Steps communicate by emitting events. Any step whose `eventTypes` list includes a given event type will be invoked when that event is fired.

```typescript
import { Workflow } from "blazen";

const workflow = new Workflow("pipeline");

workflow.addStep("extract", ["blazen::StartEvent"], async (event, ctx) => {
  await ctx.set("input", event.text);
  return { type: "analyze", payload: event.text };
});

workflow.addStep("analyze", ["analyze"], async (event, ctx) => {
  const input = await ctx.get("input");
  const summary = `Analyzed: ${input}`;
  return { type: "blazen::StopEvent", result: summary };
});

const result = await workflow.run({ text: "some content" });
console.log(result.data);
```

---

## LLM Integration

`CompletionModel` provides a unified interface to a wide range of LLM providers. Pass a model instance into your step's closure through a shared variable or via `ctx`.

```typescript
import { Workflow, CompletionModel } from "blazen";

const model = CompletionModel.openai(process.env.OPENAI_API_KEY!);

const workflow = new Workflow("llm-workflow");

workflow.addStep("chat", ["blazen::StartEvent"], async (event, ctx) => {
  const response = await model.complete([
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: event.question },
  ]);

  return { type: "blazen::StopEvent", result: response.content };
});

const result = await workflow.run({ question: "What is 2 + 2?" });
console.log(result.data);
```

### Supported Providers

| Factory method                                    | Provider         |
|---------------------------------------------------|------------------|
| `CompletionModel.openai(apiKey)`                  | OpenAI           |
| `CompletionModel.anthropic(apiKey)`               | Anthropic        |
| `CompletionModel.gemini(apiKey)`                  | Google Gemini    |
| `CompletionModel.azure(apiKey, resource, deploy)` | Azure OpenAI     |
| `CompletionModel.openrouter(apiKey)`              | OpenRouter       |
| `CompletionModel.groq(apiKey)`                    | Groq             |
| `CompletionModel.together(apiKey)`                | Together AI      |
| `CompletionModel.mistral(apiKey)`                 | Mistral AI       |
| `CompletionModel.deepseek(apiKey)`                | DeepSeek         |
| `CompletionModel.fireworks(apiKey)`               | Fireworks AI     |
| `CompletionModel.perplexity(apiKey)`              | Perplexity       |
| `CompletionModel.xai(apiKey)`                     | xAI / Grok       |
| `CompletionModel.cohere(apiKey)`                  | Cohere           |
| `CompletionModel.bedrock(apiKey, region)`         | AWS Bedrock      |
| `CompletionModel.fal(apiKey)`                     | fal.ai           |

You can also pass additional options such as `temperature`, `maxTokens`, `topP`, a model override, or tool definitions:

```typescript
const response = await model.completeWithOptions(messages, {
  model: "gpt-4o",
  temperature: 0.7,
  maxTokens: 1024,
});
```

---

## Streaming

Steps can push intermediate events to external consumers via `ctx.writeEventToStream()`. Use `runStreaming` to receive them as they arrive.

```typescript
import { Workflow, CompletionModel } from "blazen";

const model = CompletionModel.anthropic(process.env.ANTHROPIC_API_KEY!);

const workflow = new Workflow("streaming-workflow");

workflow.addStep("stream-tokens", ["blazen::StartEvent"], async (event, ctx) => {
  // Publish progress events consumers can observe in real time.
  await ctx.writeEventToStream({ type: "progress", message: "Starting..." });

  const response = await model.complete([
    { role: "user", content: event.prompt },
  ]);

  await ctx.writeEventToStream({ type: "progress", message: "Done." });

  return { type: "blazen::StopEvent", result: response.content };
});

const result = await workflow.runStreaming(
  { prompt: "Tell me a story." },
  (event) => {
    // Called for every event published via ctx.writeEventToStream().
    console.log("[stream]", event);
  }
);

console.log(result.data);
```

---

## Pause and Resume

`runWithHandler` gives you a `WorkflowHandler` that lets you pause execution and serialize the full workflow state to JSON. You can store the snapshot and resume it later -- on a different machine if needed.

```typescript
import { Workflow } from "blazen";
import { writeFileSync, readFileSync } from "fs";

const workflow = new Workflow("pausable");

workflow.addStep("long-task", ["blazen::StartEvent"], async (event, ctx) => {
  // ... expensive work ...
  return { type: "blazen::StopEvent", result: "done" };
});

// Start the workflow and immediately pause it.
const handler = await workflow.runWithHandler({ input: "data" });
const snapshot = await handler.pause();
writeFileSync("snapshot.json", snapshot);

// Later: restore the workflow and resume from the snapshot.
const snapshot2 = readFileSync("snapshot.json", "utf-8");
const resumedHandler = await workflow.resume(snapshot2);
const result = await resumedHandler.result();
console.log(result.data);
```

---

## Human-in-the-Loop

Pause/resume is the foundation for human-in-the-loop workflows. Pause after a step completes to wait for human review, inject updated data into the context, then call `resume` to continue:

```typescript
const handler = await workflow.runWithHandler({ document: rawText });

// Pause and persist state until a human reviews/approves.
const snapshot = await handler.pause();
await db.saveSnapshot(jobId, snapshot);

// ... human reviews and approves via your UI ...

// Resume with the same workflow instance (steps must be re-registered).
const savedSnapshot = await db.loadSnapshot(jobId);
const resumedHandler = await workflow.resume(savedSnapshot);
const result = await resumedHandler.result();
```

---

## Context API

Every step receives a `ctx` object for shared state and event routing.

```typescript
// Store and retrieve values across steps.
await ctx.set("key", { any: "json-serializable value" });
const value = await ctx.get("key");

// Route an event to a registered step by type.
await ctx.sendEvent({ type: "my-event", data: "..." });

// Publish an event to external streaming consumers.
await ctx.writeEventToStream({ type: "progress", pct: 50 });

// Get the current run ID.
const runId = await ctx.runId();
```

---

## TypeScript Support

Full TypeScript type definitions are included -- no `@types` package needed. All classes (`Workflow`, `WorkflowHandler`, `Context`, `CompletionModel`) and the `JsWorkflowResult` interface are exported from `index.d.ts`.

```typescript
import type { JsWorkflowResult } from "blazen";

const result: JsWorkflowResult = await workflow.run({ text: "hello" });
console.log(result.type); // "blazen::StopEvent"
console.log(result.data); // your result payload
```

---

## API Summary

| Export            | Description                                              |
|-------------------|----------------------------------------------------------|
| `Workflow`        | Build and run event-driven workflows                     |
| `WorkflowHandler` | Control handle returned by `runWithHandler` (pause/resume/stream) |
| `Context`         | Per-run key/value store, event routing, and stream sink  |
| `CompletionModel` | Unified LLM client with provider factory methods        |
| `version()`       | Returns the blazen library version string               |

---

## Full Documentation

See the [Blazen GitHub repository](https://github.com/ZachHandley/Blazen) for full documentation, advanced examples, and the Rust core source.

---

## License

[AGPL-3.0](https://opensource.org/licenses/AGPL-3.0)
