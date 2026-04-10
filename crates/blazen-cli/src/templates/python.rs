pub const TEMPLATE: &str = r#"# Blazen — Python Usage Guide

## Installation

```bash
pip install blazen --index-url https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/pypi/simple/
```

## Core Concepts

- **Event** — A message passed between steps. Use the built-in `Event`, `StartEvent`, and `StopEvent` classes.
- **Step** — An async function decorated with `@step` that receives a context and event, returning a new event.
- **Workflow** — A pipeline of steps. Created with `Workflow(name, steps)`.
- **Context** — Shared key/value state accessible by all steps within a workflow run.

## Quick Start

```python
import asyncio
from blazen import Workflow, step, Event, StartEvent, StopEvent, Context

@step
async def parse_input(ctx: Context, ev: StartEvent) -> Event:
    name = ev.to_dict().get("name", "World")
    return Event(type="GreetEvent", name=name)

@step(event_types=["GreetEvent"])
async def greet(ctx: Context, ev: Event) -> StopEvent:
    name = ev.to_dict().get("name", "World")
    return StopEvent(result={"greeting": f"Hello, {name}!"})

async def main():
    wf = Workflow("greeter", [parse_input, greet])
    handler = await wf.run(name="Zach")
    result = await handler.result()
    print(result.to_dict())  # {"greeting": "Hello, Zach!"}

asyncio.run(main())
```

### How It Works

1. `StartEvent` is emitted automatically with your keyword arguments as data.
2. `parse_input` listens for `StartEvent`, extracts data, and emits a custom `Event`.
3. `greet` listens for `"GreetEvent"` and emits `StopEvent` to end the workflow.
4. The `@step` decorator registers the function with the workflow engine.

## Using Context

Steps can share state through the `Context`:

```python
@step
async def counter(ctx: Context, ev: StartEvent) -> StopEvent:
    count = await ctx.get("count") or 0
    await ctx.set("count", count + 1)
    return StopEvent(result={"count": count + 1})
```

## LLM Integration

```python
from blazen import CompletionModel, ChatMessage, ProviderOptions

# Pass an explicit key via typed options...
model = CompletionModel.openai(options=ProviderOptions(api_key="your-api-key"))

# ...or omit options to pick up the standard env var (OPENAI_API_KEY):
# model = CompletionModel.openai()

@step
async def ask_llm(ctx: Context, ev: StartEvent) -> StopEvent:
    response = await model.complete([
        ChatMessage(role="user", content="What is 2 + 2?")
    ])
    return StopEvent(result={"answer": response.content})
```

### Supported Providers

Every provider factory accepts a keyword-only `options` argument. Use
`ProviderOptions` for simple providers, or the dedicated `AzureOptions` /
`BedrockOptions` classes for providers with required extra fields. Omit
`options` entirely to fall back to the provider's standard environment
variable (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

| Provider | Factory |
|----------|---------|
| OpenAI | `CompletionModel.openai(options=ProviderOptions(api_key="..."))` |
| Anthropic | `CompletionModel.anthropic(options=ProviderOptions(api_key="..."))` |
| Google Gemini | `CompletionModel.gemini(options=ProviderOptions(api_key="..."))` |
| Azure OpenAI | `CompletionModel.azure(options=AzureOptions(resource_name="...", deployment_name="...", api_key="..."))` |
| OpenRouter | `CompletionModel.openrouter(options=ProviderOptions(api_key="..."))` |
| Groq | `CompletionModel.groq(options=ProviderOptions(api_key="..."))` |

## Package Structure

| Component | Description |
|-----------|-------------|
| `Workflow` | Workflow builder and runner |
| `Context` | Shared state for steps |
| `Event` | Generic event with type field |
| `StartEvent` | Kicks off a workflow |
| `StopEvent` | Terminates a workflow with a result |
| `CompletionModel` | LLM provider interface |
| `ChatMessage` | Message for LLM requests |
| `ProviderOptions` | Typed options (api_key, model, base_url) for most providers |
| `AzureOptions` | Typed options for Azure OpenAI |
| `BedrockOptions` | Typed options for AWS Bedrock |
| `@step` | Decorator to register step functions |
"#;
