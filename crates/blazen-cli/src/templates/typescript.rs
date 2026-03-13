pub const TEMPLATE: &str = r#"# Blazen — TypeScript / Node.js Usage Guide

## Installation

```bash
npm install blazen --registry https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/npm/
```

## Core Concepts

- **Event** — A plain object with a `type` string and arbitrary data fields.
- **Step** — An async function added via `workflow.addStep()` that receives an event and context, returning a new event.
- **Workflow** — A pipeline of steps. Created with `new Workflow(name)`.
- **Context** — Shared key/value state accessible by all steps within a workflow run.

## Quick Start

```typescript
import { Workflow, Context, Event, WorkflowResult } from "blazen";

const workflow = new Workflow("greeter");

workflow.addStep(
  "parse_input",
  ["blazen::StartEvent"],
  async (event: Event, ctx: Context): Promise<Event> => {
    const name = event.name ?? "World";
    return { type: "GreetEvent", name };
  }
);

workflow.addStep(
  "greet",
  ["GreetEvent"],
  async (event: Event, ctx: Context): Promise<Event> => {
    return {
      type: "blazen::StopEvent",
      result: { greeting: `Hello, ${event.name}!` },
    };
  }
);

const result: WorkflowResult = await workflow.run({ name: "Zach" });
console.log(result.data); // { greeting: "Hello, Zach!" }
```

### How It Works

1. `blazen::StartEvent` is emitted automatically with your input object as data fields.
2. `parse_input` listens for `blazen::StartEvent`, extracts data, and emits `GreetEvent`.
3. `greet` listens for `GreetEvent` and emits `blazen::StopEvent` to end the workflow.
4. Event types are plain strings — use `"blazen::StartEvent"` and `"blazen::StopEvent"` for built-ins.

## Using Context

Steps can share state through the `Context`:

```typescript
workflow.addStep(
  "counter",
  ["blazen::StartEvent"],
  async (event: Event, ctx: Context): Promise<Event> => {
    const count = (await ctx.get("count")) ?? 0;
    await ctx.set("count", count + 1);
    return { type: "blazen::StopEvent", result: { count: count + 1 } };
  }
);
```

## Streaming

Use `runStreaming` to receive intermediate events published via `ctx.writeEventToStream()`:

```typescript
const result = await workflow.runStreaming(
  { message: "hello" },
  (event: Event) => {
    console.log("Stream event:", event);
  }
);
```

## LLM Integration

```typescript
import { CompletionModel, ChatMessage, CompletionResponse } from "blazen";

const model = CompletionModel.openai("your-api-key");

const response: CompletionResponse = await model.complete([
  { role: "user", content: "What is 2 + 2?" },
]);

console.log(response.content); // "4"
```

With options:

```typescript
const response = await model.completeWithOptions(
  [{ role: "user", content: "Tell me a joke" }],
  { temperature: 0.9, maxTokens: 200 }
);
```

### Supported Providers

| Provider | Factory |
|----------|---------|
| OpenAI | `CompletionModel.openai(apiKey)` |
| Anthropic | `CompletionModel.anthropic(apiKey)` |
| Google Gemini | `CompletionModel.gemini(apiKey)` |
| Azure OpenAI | `CompletionModel.azure(apiKey, resource, deployment)` |
| FAL.ai | `CompletionModel.fal(apiKey)` |
| OpenRouter | `CompletionModel.openrouter(apiKey)` |
| Groq | `CompletionModel.groq(apiKey)` |
| Together | `CompletionModel.together(apiKey)` |
| Mistral | `CompletionModel.mistral(apiKey)` |
| DeepSeek | `CompletionModel.deepseek(apiKey)` |
| Fireworks | `CompletionModel.fireworks(apiKey)` |
| Perplexity | `CompletionModel.perplexity(apiKey)` |
| xAI (Grok) | `CompletionModel.xai(apiKey)` |
| Cohere | `CompletionModel.cohere(apiKey)` |
| AWS Bedrock | `CompletionModel.bedrock(apiKey, region)` |

## TypeScript Types

Key interfaces exported from the package:

```typescript
interface Event { type: string; [key: string]: any; }
interface WorkflowResult { type: string; data: Record<string, any>; }
interface ChatMessage { role: "system" | "user" | "assistant" | "tool"; content: string; }
interface CompletionResponse { content: string | null; toolCalls: ToolCall[]; usage: TokenUsage | null; model: string; }
interface CompletionOptions { temperature?: number; maxTokens?: number; topP?: number; model?: string; }
```

## Package Structure

| Export | Description |
|--------|-------------|
| `Workflow` | Workflow builder and runner |
| `Context` | Shared state for steps |
| `CompletionModel` | LLM provider interface with factory methods |
| `version()` | Returns the library version string |
"#;
