//! Branching workflow example: conditional routing based on analysis.
//!
//! Demonstrates:
//! - Multiple custom event types
//! - Steps that return `StepOutput` for conditional routing
//! - `#[step(emits = [...])]` for declaring multiple possible output types
//! - Separate handlers for each branch
//!
//! Run with: `cargo run -p zagents --example branching_workflow`

// Steps must be async for the #[step] macro even when they don't await.
#![allow(clippy::unused_async)]

use zagents::prelude::*;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct AnalyzeEvent {
    text: String,
    sentiment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct PositiveEvent {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct NegativeEvent {
    text: String,
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// Analyze the input text and compute a simple sentiment score.
#[step]
async fn analyze(event: StartEvent, _ctx: Context) -> Result<AnalyzeEvent, WorkflowError> {
    let text = event.data["text"].as_str().unwrap_or("").to_string();
    let lower = text.to_lowercase();

    // Toy sentiment: positive if the text contains "good" or "great".
    let sentiment = if lower.contains("good") || lower.contains("great") {
        0.9
    } else {
        0.2
    };

    println!("[analyze] text={text:?}  sentiment={sentiment}");
    Ok(AnalyzeEvent { text, sentiment })
}

/// Route to the positive or negative branch based on the sentiment score.
#[step(emits = [PositiveEvent, NegativeEvent])]
async fn route(event: AnalyzeEvent, _ctx: Context) -> Result<StepOutput, WorkflowError> {
    if event.sentiment > 0.5 {
        println!("[route] -> PositiveEvent");
        Ok(StepOutput::Single(Box::new(PositiveEvent {
            text: event.text,
        })))
    } else {
        println!("[route] -> NegativeEvent");
        Ok(StepOutput::Single(Box::new(NegativeEvent {
            text: event.text,
        })))
    }
}

/// Handle the positive branch.
#[step]
async fn handle_positive(event: PositiveEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    let response = format!("Glad you feel positive about: {}", event.text);
    println!("[handle_positive] {response}");
    Ok(StopEvent {
        result: serde_json::json!({ "response": response }),
    })
}

/// Handle the negative branch.
#[step]
async fn handle_negative(event: NegativeEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    let response = format!("Sorry to hear about: {}", event.text);
    println!("[handle_negative] {response}");
    Ok(StopEvent {
        result: serde_json::json!({ "response": response }),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let workflow = WorkflowBuilder::new("sentiment_router")
        .step(analyze_registration())
        .step(route_registration())
        .step(handle_positive_registration())
        .step(handle_negative_registration())
        .build()?;

    // Test 1: positive input.
    println!("=== Positive Input ===");
    let result = workflow
        .run(serde_json::json!({ "text": "This is a great day!" }))
        .await?
        .result()
        .await?;
    println!("Result: {}\n", result.to_json());

    // Test 2: negative input.
    println!("=== Negative Input ===");
    let result = workflow
        .run(serde_json::json!({ "text": "This is terrible" }))
        .await?
        .result()
        .await?;
    println!("Result: {}", result.to_json());

    Ok(())
}
