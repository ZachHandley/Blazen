# Blazen

**Event-driven AI workflow engine, powered by Rust.**

[![PyPI](https://img.shields.io/pypi/v/blazen)](https://pypi.org/project/blazen/)
[![Python](https://img.shields.io/pypi/pyversions/blazen)](https://pypi.org/project/blazen/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](https://github.com/ZachHandley/Blazen/blob/main/LICENSE)

Blazen lets you build multi-step AI workflows as composable, event-driven graphs. Define steps with a decorator, wire them together with typed events, and run everything on a native Rust engine with async Python bindings.

## Installation

```bash
# Recommended
uv add blazen

# Or with pip
pip install blazen
```

Requires Python 3.10+.

## Quick Start

```python
import asyncio
from blazen import Workflow, step, Event, StartEvent, StopEvent, Context

class GreetEvent(Event):
    name: str

@step
async def parse(ctx: Context, ev: Event):
    return GreetEvent(name=ev.name)

@step
async def greet(ctx: Context, ev: GreetEvent):
    return StopEvent(result={"greeting": f"Hello, {ev.name}!"})

async def main():
    wf = Workflow("hello", [parse, greet])
    handler = await wf.run(name="Blazen")
    result = await handler.result()
    print(result.result)  # {"greeting": "Hello, Blazen!"}

asyncio.run(main())
```

### How it works

- **`class GreetEvent(Event)`** -- Subclassing `Event` auto-sets `event_type` to the class name (`"GreetEvent"`). Annotations like `name: str` are for documentation only; at runtime all keyword arguments are stored as JSON.
- **`@step` reads type annotations** -- `ev: GreetEvent` on a step function automatically sets `accepts=["GreetEvent"]`. The step will only receive events of that type.
- **`@step` with no type hint or `ev: Event`** -- defaults to accepting `StartEvent` (the event emitted by `wf.run()`).
- **`ev.name`** -- Direct attribute access on events. No need for `ev.to_dict()["name"]`.
- **`wf.run(name="Blazen")`** -- Keyword arguments become the `StartEvent` payload. Steps that accept `StartEvent` receive an event where `ev.name == "Blazen"`.

## Multi-Step Workflows

Chain steps together using custom event subclasses. Each step declares which events it accepts via its type annotation.

```python
import asyncio
from blazen import Workflow, step, Event, StopEvent, Context

class FetchedEvent(Event):
    text: str
    source: str

class AnalyzedEvent(Event):
    summary: str

@step
async def fetch(ctx: Context, ev: Event):
    # ev is a StartEvent with url=...
    text = f"Content from {ev.url}"
    return FetchedEvent(text=text, source=ev.url)

@step
async def analyze(ctx: Context, ev: FetchedEvent):
    summary = f"Analysis of: {ev.text}"
    return AnalyzedEvent(summary=summary)

@step
async def report(ctx: Context, ev: AnalyzedEvent):
    return StopEvent(result={"summary": ev.summary})

async def main():
    wf = Workflow("pipeline", [fetch, analyze, report])
    handler = await wf.run(url="https://example.com")
    result = await handler.result()
    print(result.result)  # {"summary": "Analysis of: Content from https://example.com"}

asyncio.run(main())
```

## Event Streaming

Stream intermediate events from a running workflow in real time using `ctx.write_event_to_stream()`.

```python
import asyncio
from blazen import Workflow, step, Event, StopEvent, Context

class ProgressEvent(Event):
    step_num: int
    message: str

@step
async def work(ctx: Context, ev: Event):
    for i in range(3):
        ctx.write_event_to_stream(ProgressEvent(step_num=i, message=f"Processing {i}"))
    return StopEvent(result="done")

async def main():
    wf = Workflow("streamer", [work])
    handler = await wf.run()

    async for event in handler.stream_events():
        print(event.event_type, event.step_num, event.message)

    result = await handler.result()
    print(result.result)  # "done"

asyncio.run(main())
```

`write_event_to_stream()` publishes to an external broadcast stream. Consumers read it with `async for event in handler.stream_events()`. These events are **not** routed through the step graph -- they are for external observation only.

## LLM Integration

Blazen includes a built-in multi-provider LLM client. All providers share the same `CompletionModel` / `ChatMessage` interface. Responses are returned as typed `CompletionResponse` objects.

### ChatMessage, Role, and CompletionResponse

```python
import os
from blazen import CompletionModel, ChatMessage, Role, CompletionResponse

model = CompletionModel.openrouter(os.environ["OPENROUTER_API_KEY"])
response: CompletionResponse = await model.complete([
    ChatMessage.system("You are helpful."),
    ChatMessage.user("What is 2+2?"),
], temperature=0.7, max_tokens=256)

# Typed attribute access
print(response.content)        # "4"
print(response.model)          # model name used
print(response.finish_reason)  # "stop", "tool_calls", etc.
print(response.tool_calls)     # list[ToolCall] or None
print(response.usage)          # TokenUsage with .prompt_tokens, .completion_tokens, .total_tokens

# Dict-style access also works for backwards compatibility
print(response["content"])
```

### Role Enum

```python
from blazen import Role

Role.SYSTEM     # "system"
Role.USER       # "user"
Role.ASSISTANT  # "assistant"
Role.TOOL       # "tool"

# Use with ChatMessage constructor
msg = ChatMessage(role=Role.USER, content="Hello")
```

### Multimodal Messages

Send images alongside text using multimodal factory methods:

```python
from blazen import ChatMessage, ContentPart

# Image from URL
msg = ChatMessage.user_image_url("https://example.com/photo.jpg", "What's in this image?")

# Image from base64
msg = ChatMessage.user_image_base64(base64_data, "image/png", "Describe this.")

# Multiple content parts
msg = ChatMessage.user_parts([
    ContentPart.text(text="Compare these two images:"),
    ContentPart.image_url(url="https://example.com/a.jpg", media_type="image/jpeg"),
    ContentPart.image_url(url="https://example.com/b.jpg", media_type="image/jpeg"),
])
```

### Supported Providers

| Provider | Constructor | Default Model |
|---|---|---|
| OpenAI | `CompletionModel.openai(api_key, model=None)` | `gpt-4o` |
| Anthropic | `CompletionModel.anthropic(api_key, model=None)` | `claude-sonnet-4-20250514` |
| Google Gemini | `CompletionModel.gemini(api_key, model=None)` | `gemini-2.0-flash` |
| Azure OpenAI | `CompletionModel.azure(api_key, resource_name, deployment_name)` | (deployment) |
| OpenRouter | `CompletionModel.openrouter(api_key, model=None)` | -- |
| Groq | `CompletionModel.groq(api_key, model=None)` | -- |
| Together AI | `CompletionModel.together(api_key, model=None)` | -- |
| Mistral | `CompletionModel.mistral(api_key, model=None)` | -- |
| DeepSeek | `CompletionModel.deepseek(api_key, model=None)` | -- |
| Fireworks | `CompletionModel.fireworks(api_key, model=None)` | -- |
| Perplexity | `CompletionModel.perplexity(api_key, model=None)` | -- |
| xAI (Grok) | `CompletionModel.xai(api_key, model=None)` | -- |
| Cohere | `CompletionModel.cohere(api_key, model=None)` | -- |
| AWS Bedrock | `CompletionModel.bedrock(api_key, region, model=None)` | -- |
| fal.ai | `CompletionModel.fal(api_key, model=None)` | -- |

### Using LLMs in Workflows

```python
import os
from blazen import Workflow, step, Event, StopEvent, Context, CompletionModel, ChatMessage

class AnswerEvent(Event):
    answer: str

@step
async def ask_llm(ctx: Context, ev: Event):
    model = CompletionModel.anthropic(os.environ["ANTHROPIC_API_KEY"])
    response = await model.complete([
        ChatMessage.system("Answer concisely."),
        ChatMessage.user(ev.prompt),
    ], max_tokens=256)
    return AnswerEvent(answer=response.content)  # typed attribute access

@step
async def format_answer(ctx: Context, ev: AnswerEvent):
    return StopEvent(result={"answer": ev.answer})

async def main():
    wf = Workflow("llm-pipeline", [ask_llm, format_answer])
    handler = await wf.run(prompt="Explain gravity in one sentence.")
    result = await handler.result()
    print(result.result)
```

## Branching / Fan-Out

Return a list of events from a step to dispatch multiple events simultaneously. Each event is routed independently to steps that accept its type.

```python
from blazen import Workflow, step, Event, StopEvent, Context

class TaskEvent(Event):
    task_id: int
    payload: str

@step
async def fan_out(ctx: Context, ev: Event):
    return [
        TaskEvent(task_id=1, payload="first"),
        TaskEvent(task_id=2, payload="second"),
        TaskEvent(task_id=3, payload="third"),
    ]

@step
async def process_task(ctx: Context, ev: TaskEvent):
    # Called once per TaskEvent
    return StopEvent(result={"task_id": ev.task_id, "done": True})
```

## Side-Effect Steps

A step can return `None` and use `ctx.send_event()` to route events through the internal step graph without returning them. This is useful for steps that perform side effects (logging, saving state) before forwarding.

```python
from blazen import Workflow, step, Event, StopEvent, Context

class ProcessedEvent(Event):
    data: str

@step
async def log_and_forward(ctx: Context, ev: Event):
    ctx.set("received_at", "2025-01-01T00:00:00Z")
    ctx.send_event(ProcessedEvent(data=ev.payload))
    return None  # no direct return -- event sent via ctx

@step
async def finish(ctx: Context, ev: ProcessedEvent):
    received = ctx.get("received_at")
    return StopEvent(result={"data": ev.data, "received_at": received})
```

`ctx.send_event()` routes the event through the internal step registry (to steps whose `accepts` matches the event type). This is different from `ctx.write_event_to_stream()` which publishes to the external broadcast stream.

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

## Context API

Steps share state through the `Context` object. All values must be JSON-serializable (for `set`/`get`) or raw bytes (for `set_bytes`/`get_bytes`). Every method on `Context` is **synchronous** -- no `await` needed.

| Method | Description |
|---|---|
| `ctx.set(key, value)` | Store a JSON-serializable value. |
| `ctx.get(key)` | Retrieve a value (returns `None` if missing). |
| `ctx.set_bytes(key, data)` | Store raw binary data (bytes). No serialization requirement. |
| `ctx.get_bytes(key)` | Retrieve raw binary data (returns `None` if missing). |
| `ctx.send_event(event)` | Route an event through the internal step graph. |
| `ctx.write_event_to_stream(event)` | Publish an event to the external broadcast stream. |
| `ctx.run_id()` | Get the UUID string for the current workflow run. |

```python
@step
async def example(ctx: Context, ev: Event):
    ctx.set("counter", 42)              # synchronous
    val = ctx.get("counter")            # synchronous, returns 42
    run = ctx.run_id()                  # synchronous, returns UUID string
    ctx.send_event(SomeEvent(x=1))      # synchronous, routes internally
    ctx.write_event_to_stream(SomeEvent(x=1))  # synchronous, broadcasts externally
    return None
```

### Binary Storage

`set_bytes` / `get_bytes` let you store raw binary data with no serialization requirement. Any type can be stored by converting to bytes yourself (e.g., pickle, msgpack, protobuf). Binary data persists through pause/resume/checkpoint.

```python
import pickle

@step
async def store_model(ctx: Context, ev: Event):
    # Store arbitrary data as bytes
    model_data = pickle.dumps({"weights": [1.0, 2.0, 3.0]})
    ctx.set_bytes("model", model_data)
    return NextEvent()

@step
async def load_model(ctx: Context, ev: NextEvent):
    raw = ctx.get_bytes("model")
    model = pickle.loads(raw)
    return StopEvent(result=model)
```

## API Reference

| Class / Function | Description |
|---|---|
| `Event(event_type, **kwargs)` | Base event class. Subclass it: `class MyEvent(Event)` auto-sets `event_type` to class name. Direct attribute access: `ev.name`. Also has `ev.to_dict()` and `ev.event_type`. |
| `StartEvent(**kwargs)` | Emitted by `wf.run(**kwargs)`. Steps with `ev: Event` or no annotation accept this. |
| `StopEvent(**kwargs)` | Terminates the workflow. Access the result via `result.result`. |
| `Context` | Shared key/value store. Methods: `set`, `get`, `set_bytes`, `get_bytes`, `send_event`, `write_event_to_stream`, `run_id`. All synchronous. |
| `@step` | Decorator for workflow steps. Infers `accepts` from the `ev` parameter type annotation. Supports `async def` and plain `def`. May also be called as `@step(accepts=[...], emits=[...], max_concurrency=N)`. |
| `Workflow(name, steps, timeout=None)` | Validated workflow graph. `timeout` is in seconds (default: 300). |
| `await wf.run(**kwargs)` | Execute the workflow. Returns a `WorkflowHandler`. Kwargs become the `StartEvent` payload. |
| `WorkflowHandler` | Handle to a running workflow: `await handler.result()`, `async for ev in handler.stream_events()`, `await handler.pause()`. |
| `await Workflow.resume(snapshot_json, steps, timeout=None)` | Resume a paused workflow from a JSON snapshot. Returns a `WorkflowHandler`. |
| `CompletionModel.<provider>(api_key, ...)` | LLM provider. Providers: `openai`, `anthropic`, `gemini`, `azure`, `openrouter`, `groq`, `together`, `mistral`, `deepseek`, `fireworks`, `perplexity`, `xai`, `cohere`, `bedrock`, `fal`. |
| `await model.complete(messages, ...)` | Chat completion. Returns a typed `CompletionResponse`. |
| `ChatMessage(role=, content=, parts=)` | Chat message. Constructor with keyword args (role defaults to `"user"`). Static factories: `.system()`, `.user()`, `.assistant()`, `.tool()`, `.user_image_url()`, `.user_image_base64()`, `.user_parts()`. |
| `Role` | Role enum: `Role.SYSTEM`, `Role.USER`, `Role.ASSISTANT`, `Role.TOOL`. |
| `CompletionResponse` | Typed response: `.content`, `.model`, `.finish_reason`, `.tool_calls`, `.usage`. Also supports dict-style `response["content"]`. |
| `ToolCall` | Tool call object: `.id`, `.name`, `.arguments`. |
| `TokenUsage` | Token usage: `.prompt_tokens`, `.completion_tokens`, `.total_tokens`. |
| `ContentPart` | Multimodal content part: `.text(text=...)`, `.image_url(url=..., media_type=...)`, `.image_base64(data=..., media_type=...)`. |

## Documentation

Full docs: [blazen.dev](https://blazen.dev)

Source: [github.com/ZachHandley/Blazen](https://github.com/ZachHandley/Blazen)

## License

AGPL-3.0 -- see [LICENSE](https://github.com/ZachHandley/Blazen/blob/main/LICENSE) for details.

Author: Zach Handley
