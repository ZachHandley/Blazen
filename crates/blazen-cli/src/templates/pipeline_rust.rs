pub const TEMPLATE: &str = r#"# Blazen Pipeline — Rust Usage Guide

## Installation

```bash
cargo add blazen --registry forgejo --features pipeline
```

You'll also need `tokio`, `serde`, and `anyhow`:

```bash
cargo add tokio --features full
cargo add serde --features derive
cargo add serde_json
cargo add anyhow
```

## Core Concepts

- **Pipeline** — An ordered sequence of stages, each wrapping a Workflow.
- **Stage** — A sequential step that runs a single workflow with optional input mapping and conditional execution.
- **ParallelStage** — Multiple workflow branches running concurrently with a configurable join strategy.
- **PipelineState** — Shared key/value state that flows between stages.
- **PersistFn** — Callback invoked after each stage with a serializable `PipelineSnapshot`.
- **PipelineHandler** — Handle for awaiting results, streaming events, or pausing the pipeline.

## Quick Start — Two-Stage Sequential Pipeline

```rust
use blazen::prelude::*;
use blazen::pipeline::{PipelineBuilder, Stage};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct ExtractedData {
    keywords: Vec<String>,
}

#[step]
async fn extract(event: StartEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    let text = event.data["text"].as_str().unwrap_or("");
    let keywords: Vec<String> = text.split_whitespace()
        .filter(|w| w.len() > 4)
        .map(String::from)
        .collect();
    Ok(StopEvent {
        result: serde_json::json!({ "keywords": keywords }),
    })
}

#[step]
async fn summarise(event: StartEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    let keywords = event.data["keywords"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();
    Ok(StopEvent {
        result: serde_json::json!({ "summary": format!("Found {} keywords", keywords.len()) }),
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let extract_wf = WorkflowBuilder::new("extract")
        .step(extract_registration())
        .build()?;

    let summarise_wf = WorkflowBuilder::new("summarise")
        .step(summarise_registration())
        .build()?;

    let pipeline = PipelineBuilder::new("text-pipeline")
        .stage(Stage {
            name: "extract".into(),
            workflow: extract_wf,
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "summarise".into(),
            workflow: summarise_wf,
            input_mapper: None, // receives previous stage's output
            condition: None,
        })
        .build()?;

    let handler = pipeline.start(serde_json::json!({
        "text": "Blazen provides powerful workflow orchestration for Rust applications"
    }));

    let result = handler.result().await?;
    println!("Pipeline output: {}", result.final_output);

    for stage in &result.stage_results {
        println!("  Stage '{}': {} ({}ms)", stage.name, stage.output, stage.duration_ms);
    }

    Ok(())
}
```

## Parallel Stage

Run multiple workflows concurrently and collect all results:

```rust
use blazen::pipeline::{PipelineBuilder, Stage, ParallelStage, JoinStrategy};

let pipeline = PipelineBuilder::new("analyze-pipeline")
    .stage(Stage {
        name: "preprocess".into(),
        workflow: preprocess_wf,
        input_mapper: None,
        condition: None,
    })
    .parallel(ParallelStage {
        name: "analyze".into(),
        branches: vec![
            Stage {
                name: "sentiment".into(),
                workflow: sentiment_wf,
                input_mapper: None,
                condition: None,
            },
            Stage {
                name: "entities".into(),
                workflow: entities_wf,
                input_mapper: None,
                condition: None,
            },
        ],
        join_strategy: JoinStrategy::WaitAll,
    })
    .build()?;
```

Use `JoinStrategy::FirstCompletes` to return as soon as the first branch finishes (remaining branches are cancelled).

## Conditional Stages & Input Mapping

Skip stages based on pipeline state, or transform the input before passing it to a workflow:

```rust
use blazen::pipeline::{Stage, ConditionFn, InputMapperFn};
use std::sync::Arc;

Stage {
    name: "optional-step".into(),
    workflow: optional_wf,
    input_mapper: Some(Arc::new(|state| {
        // Transform pipeline state into workflow input
        let keywords = state.stage_result("extract")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        serde_json::json!({ "data": keywords })
    })),
    condition: Some(Arc::new(|state| {
        // Only run if the extract stage produced keywords
        state.stage_result("extract")
            .and_then(|v| v.get("keywords"))
            .and_then(|v| v.as_array())
            .is_some_and(|arr| !arr.is_empty())
    })),
}
```

## Persistence Callback

Save a snapshot after each stage completes (for crash recovery or audit):

```rust
use blazen::pipeline::PipelineBuilder;
use std::sync::Arc;

let pipeline = PipelineBuilder::new("durable-pipeline")
    .stage(stage_a)
    .stage(stage_b)
    .on_persist_json(Arc::new(|json_string| {
        Box::pin(async move {
            tokio::fs::write("pipeline_snapshot.json", &json_string).await
                .map_err(|e| blazen::pipeline::PipelineError::PersistFailed(e.to_string()))?;
            println!("Snapshot saved ({} bytes)", json_string.len());
            Ok(())
        })
    }))
    .build()?;
```

## Streaming Events

Subscribe to intermediate events from all stages as they execute:

```rust
use tokio_stream::StreamExt;

let handler = pipeline.start(input);

// Stream events from a cloned handler reference
let mut stream = handler.stream_events();
tokio::spawn(async move {
    while let Some(event) = stream.next().await {
        println!("[{}] event: {:?}", event.stage_name, event.event);
    }
});

let result = handler.result().await?;
```

## Pause & Resume

Pause a running pipeline between stages and resume later:

```rust
let handler = pipeline.start(input);

// ... some time later ...
let snapshot = handler.pause().await?;
let json = snapshot.to_json()?;
tokio::fs::write("paused.json", &json).await?;

// Resume from snapshot (requires a new Pipeline with the same stages)
let json = tokio::fs::read_to_string("paused.json").await?;
let snapshot = blazen::pipeline::PipelineSnapshot::from_json(&json)?;
let handler = rebuilt_pipeline.resume(snapshot)?;
let result = handler.result().await?;
```

## Per-Stage Timeout

Set a maximum duration for each stage:

```rust
use std::time::Duration;

let pipeline = PipelineBuilder::new("timed-pipeline")
    .stage(stage_a)
    .stage(stage_b)
    .timeout_per_stage(Duration::from_secs(30))
    .build()?;
```
"#;
