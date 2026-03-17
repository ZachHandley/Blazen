# Blazen

**Event-driven AI workflow engine, powered by Rust.**

[![PyPI](https://img.shields.io/pypi/v/blazen)](https://pypi.org/project/blazen/)
[![Python](https://img.shields.io/pypi/pyversions/blazen)](https://pypi.org/project/blazen/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](https://github.com/ZachHandley/Blazen/blob/main/LICENSE)

Blazen lets you build multi-step AI workflows as composable, event-driven graphs. Define steps with a decorator, wire them together with typed events, and run everything on a native Rust engine with async Python bindings.

## Installation

```bash
pip install blazen
```

## Quick Start

```python
import asyncio
from blazen import Workflow, step, Event, StartEvent, StopEvent, Context

@step
async def greet(ctx: Context, ev: Event) -> Event:
    name = ev.to_dict().get("name", "world")
    return StopEvent(result=f"Hello, {name}!")

async def main():
    wf = Workflow("hello", [greet])
    handler = await wf.run(name="Blazen")
    result = await handler.result()
    print(result.to_dict())  # {"result": "Hello, Blazen!"}

asyncio.run(main())
```

## Multi-Step Workflows

Chain steps together using custom event types. Each step declares which events it accepts and emits.

```python
from blazen import Workflow, step, Event, StartEvent, StopEvent, Context

@step(emits=["AnalyzeEvent"])
async def fetch_data(ctx: Context, ev: Event) -> Event:
    url = ev.to_dict()["url"]
    # ... fetch data ...
    return Event("AnalyzeEvent", text="fetched content", source=url)

@step(accepts=["AnalyzeEvent"], emits=["StopEvent"])
async def analyze(ctx: Context, ev: Event) -> Event:
    text = ev.text
    return StopEvent(result={"summary": f"Analyzed: {text}"})

async def main():
    wf = Workflow("pipeline", [fetch_data, analyze])
    handler = await wf.run(url="https://example.com")
    result = await handler.result()
    print(result.to_dict())
```

## LLM Integration

Blazen includes a built-in multi-provider LLM client. Supports OpenAI, Anthropic, Gemini, Azure, OpenRouter, Groq, Together, Mistral, DeepSeek, Fireworks, Perplexity, xAI, Cohere, Bedrock, and fal.

```python
from blazen import CompletionModel, ChatMessage

model = CompletionModel.openai("sk-...")
# Or: CompletionModel.anthropic("sk-ant-...")
# Or: CompletionModel.gemini("AI...")

response = await model.complete(
    [
        ChatMessage.system("You are a helpful assistant."),
        ChatMessage.user("Explain quantum computing in one sentence."),
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response["content"])
# response also contains: model, tool_calls, usage, finish_reason
```

### Using LLMs in Workflows

```python
@step
async def ask_llm(ctx: Context, ev: Event) -> Event:
    model = CompletionModel.anthropic("sk-ant-...")
    response = await model.complete([
        ChatMessage.user(ev.to_dict()["prompt"]),
    ])
    return StopEvent(result=response["content"])
```

## Event Streaming

Stream intermediate events from a running workflow in real time.

```python
@step(emits=["ProgressEvent", "StopEvent"])
async def work(ctx: Context, ev: Event) -> Event:
    for i in range(3):
        ctx.write_event_to_stream(Event("ProgressEvent", step=i))
    return StopEvent(result="done")

async def main():
    wf = Workflow("streamer", [work])
    handler = await wf.run()

    async for event in handler.stream_events():
        print(event.event_type, event.to_dict())

    result = await handler.result()
```

## Pause and Resume

Snapshot a running workflow and resume it later -- useful for long-running processes, human-in-the-loop patterns, or persisting state across restarts.

```python
# Pause: capture workflow state as JSON
handler = await wf.run(prompt="Hello")
snapshot_json = await handler.pause()
# Save snapshot_json to disk, database, etc.

# Resume: restore from snapshot with the same steps
handler = await Workflow.resume(snapshot_json, [step1, step2])
result = await handler.result()
```

## Human-in-the-Loop

Combine pause/resume with custom events to build approval workflows where a human reviews intermediate results before the workflow continues.

## Shared Context

Steps share state through the `Context` object. All values must be JSON-serializable.

```python
@step(emits=["NextEvent"])
async def step_one(ctx: Context, ev: Event) -> Event:
    ctx.set("count", 0)
    return Event("NextEvent")

@step(accepts=["NextEvent"])
async def step_two(ctx: Context, ev: Event) -> Event:
    count = ctx.get("count")  # 0
    ctx.set("count", count + 1)
    return StopEvent(result=ctx.get("count"))
```

## API Reference

| Class / Function | Description |
|---|---|
| `Event(event_type, **kwargs)` | Dict-like event for inter-step communication |
| `StartEvent(**kwargs)` | Kicks off a workflow |
| `StopEvent(**kwargs)` | Terminates a workflow with a result |
| `Context` | Shared key/value store (`set`, `get`, `send_event`, `write_event_to_stream`, `run_id`) |
| `@step` | Decorator for workflow steps. Options: `accepts`, `emits`, `max_concurrency` |
| `Workflow(name, steps, timeout=None)` | Validated workflow. Call `.run(**kwargs)` to execute |
| `WorkflowHandler` | Handle to a running workflow (`.result()`, `.stream_events()`, `.pause()`) |
| `Workflow.resume(snapshot_json, steps)` | Static method to resume from a snapshot |
| `CompletionModel.openai(api_key)` | LLM provider (also: `.anthropic`, `.gemini`, `.azure`, `.openrouter`, `.groq`, `.together`, `.mistral`, `.deepseek`, `.fireworks`, `.perplexity`, `.xai`, `.cohere`, `.bedrock`, `.fal`) |
| `ChatMessage(role, content)` | Chat message (also: `.system()`, `.user()`, `.assistant()`, `.tool()`) |

## Documentation

Full docs and source: [github.com/ZachHandley/Blazen](https://github.com/ZachHandley/Blazen)

## License

AGPL-3.0 -- see [LICENSE](https://github.com/ZachHandley/Blazen/blob/main/LICENSE) for details.
