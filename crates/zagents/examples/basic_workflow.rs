//! Basic workflow example: a simple 3-step sequential pipeline.
//!
//! Demonstrates:
//! - Defining custom events with `#[derive(Event)]`
//! - Writing step functions with `#[step]`
//! - Building and running a workflow with `WorkflowBuilder`
//! - Awaiting the final result
//!
//! Run with: `cargo run -p zagents --example basic_workflow`

// Steps must be async for the #[step] macro even when they don't await.
#![allow(clippy::unused_async)]

use zagents::prelude::*;

// ---------------------------------------------------------------------------
// Custom events
// ---------------------------------------------------------------------------

/// Intermediate event carrying a parsed name.
#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct GreetEvent {
    name: String,
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// Parse the raw JSON input into a typed `GreetEvent`.
#[step]
async fn parse_input(event: StartEvent, _ctx: Context) -> Result<GreetEvent, WorkflowError> {
    let name = event.data["name"]
        .as_str()
        .unwrap_or("World")
        .to_string();
    println!("[parse_input] Extracted name: {name}");
    Ok(GreetEvent { name })
}

/// Produce the final greeting and terminate the workflow.
#[step]
async fn greet(event: GreetEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    let greeting = format!("Hello, {}!", event.name);
    println!("[greet] {greeting}");
    Ok(StopEvent {
        result: serde_json::json!({ "greeting": greeting }),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build the workflow from step registrations.
    let workflow = WorkflowBuilder::new("greeter")
        .step(parse_input_registration())
        .step(greet_registration())
        .build()?;

    // Run the workflow with a JSON payload.
    let handler = workflow
        .run(serde_json::json!({ "name": "Zach" }))
        .await?;

    // Await the final result.
    let result = handler.result().await?;

    if let Some(stop) = result.downcast_ref::<StopEvent>() {
        println!("\nFinal result: {}", stop.result);
    }

    Ok(())
}
