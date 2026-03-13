# Blazen

Event-driven AI workflow engine with first-class LLM integration, written in Rust.

## Features

- **Event-driven architecture** - Type-safe events connect workflow steps with zero boilerplate via derive macros
- **LLM providers** - OpenAI, Anthropic, Gemini, Azure, FAL with structured output and streaming
- **Proc macro DSL** - `#[derive(Event)]` and `#[step]` macros for declarative workflow definitions
- **Branching and streaming** - Conditional branching, parallel fan-out, and real-time streaming workflows
- **Persistence** - Optional embedded persistence layer via redb
- **Language bindings** - Native Python (via maturin/PyO3) and Node.js/TypeScript (via napi-rs) packages

## Quick Example

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

    if let Some(stop) = result.downcast_ref::<StopEvent>() {
        println!("Result: {}", stop.result);
    }

    Ok(())
}
```

## Installation

**Rust:**
```bash
cargo add blazen --registry forgejo
```

**Python:**
```bash
pip install blazen --index-url https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/pypi/simple/
```

**Node.js / TypeScript:**
```bash
npm install blazen --registry https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/npm/
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `blazen-events` | Core event traits and derive macro support |
| `blazen-macros` | `#[derive(Event)]` and `#[step]` proc macros |
| `blazen-core` | Workflow engine, context, and step registry |
| `blazen-llm` | LLM provider integrations and structured output |
| `blazen-persist` | Optional persistence layer (redb) |
| `blazen` | Umbrella crate re-exporting everything |
| `blazen-py` | Python bindings via PyO3/maturin |
| `blazen-node` | Node.js bindings via napi-rs |

## License

MIT
