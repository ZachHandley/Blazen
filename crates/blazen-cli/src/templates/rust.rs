pub const TEMPLATE: &str = r#"# Blazen — Rust Usage Guide

## Installation

```bash
cargo add blazen --registry forgejo
```

To enable all LLM providers and persistence:

```bash
cargo add blazen --registry forgejo --features all
```

You'll also need `tokio`, `serde`, and `anyhow`:

```bash
cargo add tokio --features full
cargo add serde --features derive
cargo add serde_json
cargo add anyhow
```

## Core Concepts

- **Event** — A typed message routed between steps. Derive with `#[derive(Event)]`.
- **Step** — An async function that receives an event and returns a new event. Annotate with `#[step]`.
- **Workflow** — A directed graph of steps connected by events. Built with `WorkflowBuilder`.
- **Context** — Shared key/value state accessible by all steps within a workflow run.

## Quick Start

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
    let result = handler.result().await?.event;

    if let Some(stop) = result.downcast_ref::<StopEvent>() {
        println!("Result: {}", stop.result);
    }

    Ok(())
}
```

### How It Works

1. `StartEvent` is emitted automatically with your input JSON.
2. `parse_input` listens for `StartEvent`, extracts data, and emits `GreetEvent`.
3. `greet` listens for `GreetEvent` and emits `StopEvent` to end the workflow.
4. The `#[step]` macro generates a `<name>_registration()` function for each step.

## Using Context

Steps can share state through the `Context`:

```rust
#[step]
async fn counter(event: StartEvent, ctx: Context) -> Result<StopEvent, WorkflowError> {
    let count: i64 = ctx.get("count").await.unwrap_or(0);
    ctx.set("count", count + 1).await;
    Ok(StopEvent { result: serde_json::json!({ "count": count + 1 }) })
}
```

## LLM Integration

Requires the `llm` feature (enabled by default) plus a provider feature:

```bash
cargo add blazen --registry forgejo --features openai
# or: --features anthropic, --features all
```

Each provider is a concrete struct under `blazen::llm::providers::<name>`
that implements the `blazen::llm::CompletionModel` trait. Construct one
directly with `ProviderName::new("your-api-key")` and fluently configure
the default model via `.with_model("...")`.

```rust
use blazen::llm::providers::openai::OpenAiProvider;
use blazen::llm::{CompletionModel, CompletionRequest, ChatMessage, Role};

#[step]
async fn ask_llm(event: StartEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    // Construct the provider directly. `.with_model()` is optional.
    let model = OpenAiProvider::new("your-api-key")
        .with_model("gpt-4.1-mini");

    let request = CompletionRequest {
        messages: vec![
            ChatMessage { role: Role::User, content: "What is 2 + 2?".into() },
        ],
        ..Default::default()
    };

    let response = model.complete(&request).await?;
    let answer = response.content.unwrap_or_default();

    Ok(StopEvent { result: serde_json::json!({ "answer": answer }) })
}
```

### Supported Providers

| Provider | Feature Flag | Constructor |
|----------|-------------|-------------|
| OpenAI | `openai` | `OpenAiProvider::new("key")` |
| Anthropic | `anthropic` | `AnthropicProvider::new("key")` |
| Google Gemini | `gemini` | `GeminiProvider::new("key")` |
| Azure OpenAI | `azure` | `AzureOpenAiProvider::new("key", "resource", "deployment")` |
| OpenRouter | `openai` | `OpenRouterProvider::new("key")` |
| Groq | `openai` | `GroqProvider::new("key")` |
| Together | `openai` | `TogetherProvider::new("key")` |
| DeepSeek | `openai` | `DeepSeekProvider::new("key")` |

All provider types live under `blazen::llm::providers::<name>`. Every
constructor is `fn new(api_key: impl Into<String>) -> Self`, and each
returned value exposes `.with_model("...")` and `.with_base_url("...")`
for further configuration.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `llm` (default) | LLM provider integrations |
| `openai` | OpenAI + OpenAI-compatible providers |
| `anthropic` | Anthropic Claude |
| `gemini` | Google Gemini |
| `persist` | Checkpoint storage via redb |
| `all` | Everything |

## Crate Structure

| Crate | Description |
|-------|-------------|
| `blazen` | Umbrella crate — start here |
| `blazen-events` | Core event traits and built-in events |
| `blazen-macros` | `#[derive(Event)]` and `#[step]` proc macros |
| `blazen-core` | Workflow engine, context, and step registry |
| `blazen-llm` | LLM provider integrations |
| `blazen-persist` | Optional persistence layer (redb) |
"#;
