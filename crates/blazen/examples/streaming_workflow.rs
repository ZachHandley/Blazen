//! Streaming workflow example: publishing progress events to external consumers.
//!
//! Demonstrates:
//! - Publishing intermediate events via `ctx.write_event_to_stream()`
//! - Subscribing to the event stream via `handler.stream_events()`
//! - Composing stream subscription with awaiting the final result
//!
//! Run with: `cargo run -p blazen --example streaming_workflow`

use blazen::prelude::*;
use tokio_stream::StreamExt;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// A progress event published to the stream (not routed through the workflow).
#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct ProgressEvent {
    step_name: String,
    percent: f32,
}

/// Intermediate processing event routed between steps.
#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct ProcessEvent {
    data: String,
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

#[step]
async fn step_one(event: StartEvent, ctx: Context) -> Result<ProcessEvent, WorkflowError> {
    ctx.write_event_to_stream(ProgressEvent {
        step_name: "step_one".into(),
        percent: 0.0,
    })
    .await;

    // Simulate some work.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    ctx.write_event_to_stream(ProgressEvent {
        step_name: "step_one".into(),
        percent: 50.0,
    })
    .await;

    let data = event.data["input"].as_str().unwrap_or("").to_uppercase();

    ctx.write_event_to_stream(ProgressEvent {
        step_name: "step_one".into(),
        percent: 100.0,
    })
    .await;

    Ok(ProcessEvent { data })
}

#[step]
async fn step_two(event: ProcessEvent, ctx: Context) -> Result<StopEvent, WorkflowError> {
    ctx.write_event_to_stream(ProgressEvent {
        step_name: "step_two".into(),
        percent: 0.0,
    })
    .await;

    // Simulate some work.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let result = format!("Processed: {}", event.data);

    ctx.write_event_to_stream(ProgressEvent {
        step_name: "step_two".into(),
        percent: 100.0,
    })
    .await;

    Ok(StopEvent {
        result: serde_json::json!({ "output": result }),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let workflow = WorkflowBuilder::new("streaming_example")
        .step(step_one_registration())
        .step(step_two_registration())
        .build()?;

    let handler = workflow
        .run(serde_json::json!({ "input": "hello world" }))
        .await?;

    // Subscribe to the event stream BEFORE consuming the handler.
    // `stream_events()` borrows &self, so we can call it first.
    let mut stream = handler.stream_events();

    // Spawn a task to consume stream events in the background.
    let stream_task = tokio::spawn(async move {
        while let Some(event) = stream.next().await {
            if let Some(progress) = event.downcast_ref::<ProgressEvent>() {
                println!("[{}: {:.0}%]", progress.step_name, progress.percent);
            }
        }
    });

    // Await the final result (consumes the handler).
    let result = handler.result().await?;

    // Give the stream task a moment to finish processing buffered events.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(100), stream_task).await;

    if let Some(stop) = result.downcast_ref::<StopEvent>() {
        println!("\nFinal result: {}", stop.result);
    }

    Ok(())
}
