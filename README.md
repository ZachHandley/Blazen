# Blazen

Event-driven AI workflow engine with first-class LLM integration, written in Rust.

## Features

- **Event-driven architecture** - Type-safe events connect workflow steps with zero boilerplate via derive macros
- **Multi-workflow pipelines** - Orchestrate sequential and parallel stages with pause/resume, callback-based persistence, and per-workflow streaming
- **LLM providers** - OpenAI, Anthropic, Gemini, Azure, fal.ai with structured output, streaming, and multimodal support (images, files, multi-part messages)
- **Prompt management** - Versioned prompt templates with `{{variable}}` interpolation, YAML/JSON registries, and multimodal attachments
- **Proc macro DSL** - `#[derive(Event)]` and `#[step]` macros for declarative workflow definitions
- **Branching and streaming** - Conditional branching, parallel fan-out, and real-time streaming workflows
- **Persistence** - Optional embedded persistence layer via redb, or bring-your-own via callbacks
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

## Pipeline Example

```rust
use blazen::pipeline::*;

let pipeline = PipelineBuilder::new("analyze-and-summarize")
    .stage(Stage {
        name: "analyze".into(),
        workflow: analyze_workflow,
        input_mapper: None,  // uses pipeline input
        condition: None,
    })
    .stage(Stage {
        name: "summarize".into(),
        workflow: summarize_workflow,
        input_mapper: Some(Arc::new(|state| {
            // feed previous stage's output as this stage's input
            state.last_result().cloned().unwrap_or_default()
        })),
        condition: None,
    })
    .on_persist_json(Arc::new(|json| Box::pin(async move {
        tokio::fs::write("pipeline_state.json", json).await.map_err(|e| {
            PipelineError::PersistFailed(e.to_string())
        })
    })))
    .build()?;

let handler = pipeline.start(serde_json::json!({ "url": "https://..." })).await?;
let result = handler.result().await?;
```

## Installation

**Rust:**
```bash
cargo add blazen --registry forgejo
```

**Python:**
```bash
uv pip install blazen --index-url https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/pypi/simple/
```

**Node.js / TypeScript:**
```bash
pnpm add blazen --registry https://forge.blackleafdigital.com/api/packages/BlackLeafDigital/npm/
```

**CLI (scaffold new projects):**
```bash
cargo install blazen-cli --registry forgejo
blazen init workflow --lang rust
blazen init pipeline --lang typescript
blazen init prompts
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `blazen-events` | Core event traits and derive macro support |
| `blazen-macros` | `#[derive(Event)]` and `#[step]` proc macros |
| `blazen-core` | Workflow engine, context, and step registry |
| `blazen-llm` | LLM provider integrations, multimodal messages, and structured output |
| `blazen-pipeline` | Multi-workflow pipeline orchestrator with sequential/parallel stages |
| `blazen-prompts` | Prompt template management with versioning and YAML/JSON registries |
| `blazen-persist` | Optional persistence layer (redb/ValKey) |
| `blazen` | Umbrella crate re-exporting everything |
| `blazen-py` | Python bindings via PyO3/maturin |
| `blazen-node` | Node.js bindings via napi-rs |
| `blazen-cli` | CLI tool for scaffolding projects (`blazen init`) |

## License

MIT
