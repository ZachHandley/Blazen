//! Integration tests for the Blazen workflow engine.
//!
//! These tests exercise the full workflow lifecycle: building workflows,
//! running them, verifying event routing, branching, fan-out, streaming,
//! context state sharing, and macro integration.

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use blazen_core::{
    Context, StepFn, StepOutput, StepRegistration, Workflow, WorkflowBuilder, WorkflowError,
    WorkflowSnapshot,
};
use blazen_events::{AnyEvent, Event, StartEvent, StopEvent};
use serde::{Deserialize, Serialize};

// ===========================================================================
// Custom event types (manual impls for simplicity)
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalyzeEvent {
    text: String,
    word_count: usize,
}

impl Event for AnalyzeEvent {
    fn event_type() -> &'static str {
        "test::AnalyzeEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::AnalyzeEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuyEvent {
    ticker: String,
    amount: f64,
}

impl Event for BuyEvent {
    fn event_type() -> &'static str {
        "test::BuyEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::BuyEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SellEvent {
    ticker: String,
    amount: f64,
}

impl Event for SellEvent {
    fn event_type() -> &'static str {
        "test::SellEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::SellEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepAEvent {
    value: i32,
}

impl Event for StepAEvent {
    fn event_type() -> &'static str {
        "test::StepAEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::StepAEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepBEvent {
    value: i32,
}

impl Event for StepBEvent {
    fn event_type() -> &'static str {
        "test::StepBEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::StepBEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepCEvent {
    value: i32,
}

impl Event for StepCEvent {
    fn event_type() -> &'static str {
        "test::StepCEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::StepCEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepDEvent {
    value: i32,
}

impl Event for StepDEvent {
    fn event_type() -> &'static str {
        "test::StepDEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::StepDEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

/// Stream-only event (published to external stream, not routed internally).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProgressEvent {
    message: String,
    percent: u8,
}

impl Event for ProgressEvent {
    fn event_type() -> &'static str {
        "test::ProgressEvent"
    }
    fn event_type_id(&self) -> &'static str {
        "test::ProgressEvent"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_boxed(&self) -> Box<dyn AnyEvent> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

// ===========================================================================
// Helper: build step registrations from closures
// ===========================================================================

fn make_step<I, O>(
    name: &str,
    handler: impl Fn(
        I,
        Context,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<O, WorkflowError>> + Send>,
    > + Send
    + Sync
    + 'static,
) -> StepRegistration
where
    I: Event + for<'de> Deserialize<'de> + 'static,
    O: Event + Serialize + 'static,
{
    let handler = Arc::new(handler);
    StepRegistration {
        name: name.to_string(),
        accepts: vec![I::event_type()],
        emits: vec![O::event_type()],
        handler: Arc::new(move |event: Box<dyn AnyEvent>, ctx: Context| {
            let handler = handler.clone();
            Box::pin(async move {
                let typed = event
                    .as_any()
                    .downcast_ref::<I>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: I::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();
                let result = handler(typed, ctx).await?;
                Ok(StepOutput::Single(Box::new(result)))
            })
        }),
        max_concurrency: 1,
    }
}

// ===========================================================================
// 1. Basic sequential workflow
// ===========================================================================

#[tokio::test]
async fn test_basic_sequential_workflow() {
    // StartEvent -> analyze -> AnalyzeEvent -> finalize -> StopEvent
    let analyze_step = make_step("analyze", |event: StartEvent, _ctx: Context| {
        Box::pin(async move {
            let text = event.data["text"].as_str().unwrap_or_default().to_string();
            let word_count = text.split_whitespace().count();
            Ok(AnalyzeEvent { text, word_count })
        })
    });

    let finalize_step = make_step("finalize", |event: AnalyzeEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StopEvent {
                result: serde_json::json!({
                    "text": event.text,
                    "word_count": event.word_count,
                }),
            })
        })
    });

    let workflow = WorkflowBuilder::new("basic-sequential")
        .step(analyze_step)
        .step(finalize_step)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"text": "hello world foo"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();

    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["text"], "hello world foo");
    assert_eq!(stop.result["word_count"], 3);
}

// ===========================================================================
// 2. Branching workflow
// ===========================================================================

#[tokio::test]
async fn test_branching_workflow() {
    // StartEvent -> decide -> BuyEvent OR SellEvent
    // BuyEvent -> handle_buy -> StopEvent
    // SellEvent -> handle_sell -> StopEvent

    let decide_step: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                let action = start.data["action"].as_str().unwrap_or("buy");
                let ticker = start.data["ticker"].as_str().unwrap_or("AAPL").to_string();
                let amount = start.data["amount"].as_f64().unwrap_or(100.0);

                if action == "buy" {
                    Ok(StepOutput::Single(Box::new(BuyEvent { ticker, amount })))
                } else {
                    Ok(StepOutput::Single(Box::new(SellEvent { ticker, amount })))
                }
            })
        });

        StepRegistration {
            name: "decide".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![BuyEvent::event_type(), SellEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let handle_buy = make_step("handle_buy", |event: BuyEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StopEvent {
                result: serde_json::json!({
                    "action": "bought",
                    "ticker": event.ticker,
                    "amount": event.amount,
                }),
            })
        })
    });

    let handle_sell = make_step("handle_sell", |event: SellEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StopEvent {
                result: serde_json::json!({
                    "action": "sold",
                    "ticker": event.ticker,
                    "amount": event.amount,
                }),
            })
        })
    });

    // Test the "buy" branch.
    let workflow = WorkflowBuilder::new("branching")
        .step(decide_step.clone())
        .step(handle_buy.clone())
        .step(handle_sell.clone())
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({
            "action": "buy",
            "ticker": "GOOG",
            "amount": 50.0,
        }))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["action"], "bought");
    assert_eq!(stop.result["ticker"], "GOOG");

    // Test the "sell" branch.
    let workflow2 = WorkflowBuilder::new("branching-sell")
        .step(decide_step)
        .step(handle_buy)
        .step(handle_sell)
        .build()
        .unwrap();

    let handler2 = workflow2
        .run(serde_json::json!({
            "action": "sell",
            "ticker": "MSFT",
            "amount": 25.0,
        }))
        .await
        .unwrap();
    let result2 = handler2.result().await.unwrap();
    let stop2 = result2.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop2.result["action"], "sold");
    assert_eq!(stop2.result["ticker"], "MSFT");
}

// ===========================================================================
// 3. Multi-step chain (4+ steps in sequence)
// ===========================================================================

#[tokio::test]
async fn test_multi_step_chain() {
    // StartEvent -> step_a -> StepAEvent -> step_b -> StepBEvent
    //            -> step_c -> StepCEvent -> step_d -> StepDEvent -> final -> StopEvent

    let step_a = make_step("step_a", |event: StartEvent, _ctx: Context| {
        Box::pin(async move {
            #[allow(clippy::cast_possible_truncation)]
            let value = event.data["value"].as_i64().unwrap_or(0) as i32;
            Ok(StepAEvent { value: value + 1 })
        })
    });

    let step_b = make_step("step_b", |event: StepAEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StepBEvent {
                value: event.value * 2,
            })
        })
    });

    let step_c = make_step("step_c", |event: StepBEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StepCEvent {
                value: event.value + 10,
            })
        })
    });

    let step_d = make_step("step_d", |event: StepCEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StepDEvent {
                value: event.value * 3,
            })
        })
    });

    let final_step = make_step("final", |event: StepDEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StopEvent {
                result: serde_json::json!({ "final_value": event.value }),
            })
        })
    });

    let workflow = WorkflowBuilder::new("multi-step")
        .step(step_a)
        .step(step_b)
        .step(step_c)
        .step(step_d)
        .step(final_step)
        .build()
        .unwrap();

    // value=5 -> +1=6 -> *2=12 -> +10=22 -> *3=66
    let handler = workflow.run(serde_json::json!({"value": 5})).await.unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["final_value"], 66);
}

// ===========================================================================
// 4. Fan-out: Step returns StepOutput::Multiple
// ===========================================================================

#[tokio::test]
async fn test_fan_out() {
    // StartEvent -> fan_out -> [BuyEvent, SellEvent]
    // BuyEvent -> handle_buy -> StopEvent
    // SellEvent -> handle_sell -> StopEvent
    //
    // The first StopEvent to arrive terminates the workflow.

    let fan_out_step: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                let ticker = start.data["ticker"].as_str().unwrap_or("AAPL").to_string();

                Ok(StepOutput::Multiple(vec![
                    Box::new(BuyEvent {
                        ticker: ticker.clone(),
                        amount: 100.0,
                    }),
                    Box::new(SellEvent {
                        ticker,
                        amount: 50.0,
                    }),
                ]))
            })
        });

        StepRegistration {
            name: "fan_out".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![BuyEvent::event_type(), SellEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let handle_buy = make_step("handle_buy", |event: BuyEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StopEvent {
                result: serde_json::json!({
                    "action": "bought",
                    "ticker": event.ticker,
                    "amount": event.amount,
                }),
            })
        })
    });

    let handle_sell = make_step("handle_sell", |event: SellEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(StopEvent {
                result: serde_json::json!({
                    "action": "sold",
                    "ticker": event.ticker,
                    "amount": event.amount,
                }),
            })
        })
    });

    let workflow = WorkflowBuilder::new("fan-out")
        .step(fan_out_step)
        .step(handle_buy)
        .step(handle_sell)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"ticker": "TSLA"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();

    // One of the two branches will win the race to produce a StopEvent.
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    let action = stop.result["action"].as_str().unwrap();
    assert!(
        action == "bought" || action == "sold",
        "unexpected action: {action}"
    );
    assert_eq!(stop.result["ticker"], "TSLA");
}

// ===========================================================================
// 5. Streaming: write_event_to_stream + stream_events()
// ===========================================================================

#[tokio::test]
async fn test_streaming() {
    use tokio_stream::StreamExt;

    // StartEvent -> process -> StopEvent
    // During processing, publish ProgressEvent to the stream.

    let process_step: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                // Publish progress events to the external stream.
                ctx.write_event_to_stream(ProgressEvent {
                    message: "Starting...".to_string(),
                    percent: 0,
                })
                .await;

                ctx.write_event_to_stream(ProgressEvent {
                    message: "Processing...".to_string(),
                    percent: 50,
                })
                .await;

                ctx.write_event_to_stream(ProgressEvent {
                    message: "Done!".to_string(),
                    percent: 100,
                })
                .await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: start.data.clone(),
                })))
            })
        });

        StepRegistration {
            name: "process".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let workflow = WorkflowBuilder::new("streaming")
        .step(process_step)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"input": "test"}))
        .await
        .unwrap();

    // Subscribe to the stream BEFORE awaiting the result.
    let mut stream = handler.stream_events();

    // Collect stream events with a timeout.
    let mut stream_events = Vec::new();
    let collect_task = tokio::spawn(async move {
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(2), stream.next()).await
        {
            stream_events.push(event);
        }
        stream_events
    });

    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["input"], "test");

    // Wait for stream collection to finish.
    let collected = collect_task.await.unwrap();

    // We should have received at least some progress events.
    // (Timing is non-deterministic, so just check we got at least one.)
    // The broadcast channel might miss early events if subscription happened
    // after they were sent, so we check >= 0 but ideally we'd get some.
    // In practice, the step runs on a spawned task that starts after we
    // subscribe, so we should get all events.
    let progress_events: Vec<_> = collected
        .iter()
        .filter(|e| e.event_type_id() == "test::ProgressEvent")
        .collect();

    // We should get at least one progress event.
    assert!(
        !progress_events.is_empty(),
        "expected at least one ProgressEvent in the stream, got {}",
        progress_events.len()
    );
}

// ===========================================================================
// 6. Context state sharing between steps
// ===========================================================================

#[tokio::test]
async fn test_context_state_sharing() {
    // StartEvent -> step_a (writes to ctx) -> StepAEvent
    //            -> step_b (reads from ctx) -> StopEvent

    let step_a: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                // Write shared state.
                let text = start.data["text"].as_str().unwrap_or_default().to_string();
                ctx.set("shared_text", text.clone()).await;
                ctx.set("shared_count", text.len() as u64).await;

                Ok(StepOutput::Single(Box::new(StepAEvent { value: 42 })))
            })
        });

        StepRegistration {
            name: "writer".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StepAEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let step_b: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                // Read shared state written by step_a.
                let text = ctx.get::<String>("shared_text").await.unwrap_or_default();
                let count = ctx.get::<u64>("shared_count").await.unwrap_or(0);

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "shared_text": text,
                        "shared_count": count,
                    }),
                })))
            })
        });

        StepRegistration {
            name: "reader".to_string(),
            accepts: vec![StepAEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let workflow = WorkflowBuilder::new("context-sharing")
        .step(step_a)
        .step(step_b)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"text": "hello world"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["shared_text"], "hello world");
    assert_eq!(stop.result["shared_count"], 11);
}

// ===========================================================================
// 7. Derive macro test: #[derive(Event)]
// ===========================================================================

// Use the derive macro to define a custom event.
#[derive(Debug, Clone, Serialize, Deserialize, blazen_macros::Event)]
struct DerivedEvent {
    message: String,
    score: f64,
}

#[tokio::test]
async fn test_derive_event_macro() {
    // Verify the derive macro generates a valid Event impl.
    let event = DerivedEvent {
        message: "hello".to_string(),
        score: 0.95,
    };

    // event_type() returns a string containing the struct name.
    let event_type = DerivedEvent::event_type();
    assert!(
        event_type.contains("DerivedEvent"),
        "event_type should contain the struct name, got: {event_type}"
    );

    // event_type_id() instance method should match.
    assert_eq!(Event::event_type_id(&event), event_type);

    // as_any() should allow downcasting.
    let any_ref = Event::as_any(&event);
    let downcasted = any_ref.downcast_ref::<DerivedEvent>().unwrap();
    assert_eq!(downcasted.message, "hello");

    // clone_boxed() should produce a valid boxed trait object.
    let boxed = Event::clone_boxed(&event);
    assert_eq!(boxed.event_type_id(), event_type);

    // to_json() should produce valid JSON.
    let json = Event::to_json(&event);
    assert_eq!(json["message"], "hello");
    assert!((json["score"].as_f64().unwrap() - 0.95).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_derive_event_in_workflow() {
    // Use the derived event in a real workflow.

    let produce_step = make_step("produce", |event: StartEvent, _ctx: Context| {
        Box::pin(async move {
            Ok(DerivedEvent {
                message: event.data["msg"].as_str().unwrap_or("").to_string(),
                score: 0.99,
            })
        })
    });

    // Since DerivedEvent's event_type uses module_path!() + struct name,
    // we need to match it for the consumer step's accepts.
    let derived_event_type = DerivedEvent::event_type();

    let consume_step: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let derived = event
                    .as_any()
                    .downcast_ref::<DerivedEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: DerivedEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "message": derived.message,
                        "score": derived.score,
                    }),
                })))
            })
        });

        StepRegistration {
            name: "consume".to_string(),
            accepts: vec![derived_event_type],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let workflow = WorkflowBuilder::new("derive-test")
        .step(produce_step)
        .step(consume_step)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"msg": "derived works"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["message"], "derived works");
    assert!((stop.result["score"].as_f64().unwrap() - 0.99).abs() < f64::EPSILON);
}

// ===========================================================================
// 8. #[step] macro test
// ===========================================================================

// Define events for macro test using derive.
#[derive(Debug, Clone, Serialize, Deserialize, blazen_macros::Event)]
struct MacroTestEvent {
    value: String,
}

// Use the #[step] attribute macro.
#[allow(clippy::unused_async)]
#[blazen_macros::step]
async fn macro_step_one(event: StartEvent, _ctx: Context) -> Result<MacroTestEvent, WorkflowError> {
    let text = event.data["text"]
        .as_str()
        .unwrap_or_default()
        .to_uppercase();
    Ok(MacroTestEvent { value: text })
}

#[allow(clippy::unused_async)]
#[blazen_macros::step]
async fn macro_step_two(event: MacroTestEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    Ok(StopEvent {
        result: serde_json::json!({ "processed": event.value }),
    })
}

#[tokio::test]
async fn test_step_macro_basic() {
    // The #[step] macro should have generated:
    // - macro_step_one_registration() -> StepRegistration
    // - macro_step_two_registration() -> StepRegistration

    let reg1 = macro_step_one_registration();
    assert_eq!(reg1.name, "macro_step_one");
    assert_eq!(reg1.accepts.len(), 1);
    assert_eq!(reg1.accepts[0], StartEvent::event_type());

    let reg2 = macro_step_two_registration();
    assert_eq!(reg2.name, "macro_step_two");
    assert_eq!(reg2.accepts.len(), 1);
    assert_eq!(reg2.accepts[0], MacroTestEvent::event_type());

    let workflow = WorkflowBuilder::new("step-macro-test")
        .step(reg1)
        .step(reg2)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"text": "hello macro"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["processed"], "HELLO MACRO");
}

// ===========================================================================
// 9. #[step] macro with StepOutput return (branching)
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize, blazen_macros::Event)]
struct HighValueEvent {
    amount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, blazen_macros::Event)]
struct LowValueEvent {
    amount: f64,
}

#[allow(clippy::unused_async)]
#[blazen_macros::step(emits = [HighValueEvent, LowValueEvent])]
async fn branching_step(event: StartEvent, _ctx: Context) -> Result<StepOutput, WorkflowError> {
    let amount = event.data["amount"].as_f64().unwrap_or(0.0);
    if amount > 1000.0 {
        Ok(StepOutput::Single(Box::new(HighValueEvent { amount })))
    } else {
        Ok(StepOutput::Single(Box::new(LowValueEvent { amount })))
    }
}

#[allow(clippy::unused_async)]
#[blazen_macros::step]
async fn handle_high(event: HighValueEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    Ok(StopEvent {
        result: serde_json::json!({ "tier": "high", "amount": event.amount }),
    })
}

#[allow(clippy::unused_async)]
#[blazen_macros::step]
async fn handle_low(event: LowValueEvent, _ctx: Context) -> Result<StopEvent, WorkflowError> {
    Ok(StopEvent {
        result: serde_json::json!({ "tier": "low", "amount": event.amount }),
    })
}

#[tokio::test]
async fn test_step_macro_branching() {
    let reg_branch = branching_step_registration();
    assert_eq!(reg_branch.name, "branching_step");
    // When emits = [A, B] is specified, those should appear in emits metadata.
    assert_eq!(reg_branch.emits.len(), 2);

    let workflow = WorkflowBuilder::new("macro-branching")
        .step(reg_branch.clone())
        .step(handle_high_registration())
        .step(handle_low_registration())
        .build()
        .unwrap();

    // High value path.
    let handler = workflow
        .run(serde_json::json!({"amount": 5000.0}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["tier"], "high");
    assert_eq!(stop.result["amount"], 5000.0);

    // Low value path.
    let workflow2 = WorkflowBuilder::new("macro-branching-low")
        .step(reg_branch)
        .step(handle_high_registration())
        .step(handle_low_registration())
        .build()
        .unwrap();

    let handler2 = workflow2
        .run(serde_json::json!({"amount": 50.0}))
        .await
        .unwrap();
    let result2 = handler2.result().await.unwrap();
    let stop2 = result2.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop2.result["tier"], "low");
    assert_eq!(stop2.result["amount"], 50.0);
}

// ===========================================================================
// 10. #[step] macro with context state sharing
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize, blazen_macros::Event)]
struct IntermediateEvent {
    data: String,
}

#[blazen_macros::step]
async fn ctx_writer_step(
    event: StartEvent,
    ctx: Context,
) -> Result<IntermediateEvent, WorkflowError> {
    let input = event.data["input"].as_str().unwrap_or_default().to_string();
    ctx.set("saved_input", input.clone()).await;
    Ok(IntermediateEvent { data: input })
}

#[blazen_macros::step]
async fn ctx_reader_step(
    event: IntermediateEvent,
    ctx: Context,
) -> Result<StopEvent, WorkflowError> {
    let saved: String = ctx.get::<String>("saved_input").await.unwrap_or_default();

    Ok(StopEvent {
        result: serde_json::json!({
            "from_event": event.data,
            "from_context": saved,
            "match": event.data == saved,
        }),
    })
}

#[tokio::test]
async fn test_step_macro_context_sharing() {
    let workflow = WorkflowBuilder::new("macro-ctx")
        .step(ctx_writer_step_registration())
        .step(ctx_reader_step_registration())
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"input": "shared data"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["from_event"], "shared data");
    assert_eq!(stop.result["from_context"], "shared data");
    assert_eq!(stop.result["match"], true);
}

// ===========================================================================
// 11. Pause: basic snapshot capture
// ===========================================================================

#[tokio::test]
async fn test_pause_captures_snapshot() {
    // A step that sets context state, sleeps briefly (to give us time to
    // send the pause signal), then emits a StopEvent.
    let slow_step: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                // Set context state that should appear in the snapshot.
                ctx.set("counter", 42_u64).await;
                ctx.set("message", "paused state".to_string()).await;

                // Sleep to keep the step in-flight when pause is requested.
                tokio::time::sleep(Duration::from_millis(200)).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: start.data.clone(),
                })))
            })
        });

        StepRegistration {
            name: "slow_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let workflow = WorkflowBuilder::new("pause-test")
        .step(slow_step)
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"input": "test_pause"}))
        .await
        .unwrap();

    // Pause immediately -- this will wait for the in-flight step to finish
    // (including its 200ms sleep), then drain pending events.
    let snapshot = handler.pause().await.unwrap();

    // Verify snapshot metadata.
    assert_eq!(snapshot.workflow_name, "pause-test");
    assert!(!snapshot.run_id.is_nil());

    // Verify context state was captured.
    assert_eq!(
        snapshot.context_state.get("counter"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(42)))
    );
    assert_eq!(
        snapshot.context_state.get("message"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(
            "paused state"
        )))
    );

    // The step should have produced a StopEvent that got drained as pending.
    assert!(
        !snapshot.pending_events.is_empty(),
        "expected at least one pending event (the StopEvent from the step)"
    );

    // Verify the snapshot is serializable.
    let json = snapshot.to_json().unwrap();
    let restored = WorkflowSnapshot::from_json(&json).unwrap();
    assert_eq!(restored.workflow_name, snapshot.workflow_name);
    assert_eq!(restored.run_id, snapshot.run_id);
    assert_eq!(restored.context_state, snapshot.context_state);
}

// ===========================================================================
// 12. Pause and resume: full round-trip
// ===========================================================================

#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn test_pause_and_resume() {
    // Two-step workflow:
    // Step 1: StartEvent -> sets counter=1 in context, emits AnalyzeEvent
    // Step 2: AnalyzeEvent -> reads counter from context, emits StopEvent
    //
    // We pause after step 1 runs, verify the snapshot, then resume and verify
    // the workflow completes correctly.

    let step_one: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let _start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                // Set shared state.
                ctx.set("counter", 1_u64).await;
                ctx.set("step_one_ran", true).await;

                // Delay so the pause signal arrives while we are in-flight.
                tokio::time::sleep(Duration::from_millis(100)).await;

                Ok(StepOutput::Single(Box::new(AnalyzeEvent {
                    text: "from step one".to_string(),
                    word_count: 3,
                })))
            })
        });

        StepRegistration {
            name: "step_one".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![AnalyzeEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    // Step 2 handles AnalyzeEvent using downcast_ref (works for fresh run).
    // For resume, events are reinjected as DynamicEvent, so we also handle that.
    let step_two: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                // Read the event data via to_json() for resume compatibility.
                let json = event.to_json();
                let text = json["text"].as_str().unwrap_or_default().to_string();
                let word_count = json["word_count"].as_u64().unwrap_or(0);

                // Read and update context state.
                let counter: u64 = ctx.get::<u64>("counter").await.unwrap_or(0);
                ctx.set("counter", counter + 1).await;
                ctx.set("step_two_ran", true).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "text": text,
                        "word_count": word_count,
                        "final_counter": counter + 1,
                    }),
                })))
            })
        });

        StepRegistration {
            name: "step_two".to_string(),
            accepts: vec![AnalyzeEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    // Build and run.
    let workflow = WorkflowBuilder::new("pause-resume-test")
        .step(step_one.clone())
        .step(step_two.clone())
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"input": "test"}))
        .await
        .unwrap();

    // Pause -- step_one will complete (after 100ms), producing an AnalyzeEvent
    // that becomes a pending event in the snapshot.
    let snapshot = handler.pause().await.unwrap();

    // Verify snapshot has the state from step_one.
    assert_eq!(
        snapshot.context_state.get("counter"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(1)))
    );
    assert_eq!(
        snapshot.context_state.get("step_one_ran"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(true)))
    );

    // The AnalyzeEvent from step_one should be pending.
    assert!(
        !snapshot.pending_events.is_empty(),
        "expected pending AnalyzeEvent"
    );
    let pending_types: Vec<_> = snapshot
        .pending_events
        .iter()
        .map(|e| e.event_type.as_str())
        .collect();
    assert!(
        pending_types.contains(&AnalyzeEvent::event_type()),
        "expected AnalyzeEvent in pending events, got: {pending_types:?}"
    );

    // Now resume with a new step set.
    // For resume, events come in as DynamicEvent. Step 2 needs to handle
    // the "test::AnalyzeEvent" event type. We register a DynamicEvent handler
    // that routes on the same event type string.
    let step_two_for_resume: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                // After resume, events arrive as DynamicEvent.
                let json = event.to_json();
                // DynamicEvent wraps data under a "data" key.
                let data = if json.get("data").is_some() && json.get("event_type").is_some() {
                    // This is a DynamicEvent envelope.
                    json["data"].clone()
                } else {
                    json
                };
                let text = data["text"].as_str().unwrap_or_default().to_string();
                let word_count = data["word_count"].as_u64().unwrap_or(0);

                let counter: u64 = ctx.get::<u64>("counter").await.unwrap_or(0);
                ctx.set("counter", counter + 1).await;
                ctx.set("step_two_ran", true).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "text": text,
                        "word_count": word_count,
                        "final_counter": counter + 1,
                    }),
                })))
            })
        });

        StepRegistration {
            name: "step_two".to_string(),
            // Use the interned event type so it matches the DynamicEvent's type.
            accepts: vec![AnalyzeEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let resumed_handler = Workflow::resume(
        snapshot,
        vec![step_one, step_two_for_resume],
        None, // no timeout
    )
    .await
    .unwrap();

    let result = resumed_handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["final_counter"], 2);
    assert_eq!(stop.result["word_count"], 3);
}

// ===========================================================================
// 13. Pause and resume via JSON serialization round-trip
// ===========================================================================

#[allow(clippy::similar_names)]
#[tokio::test]
async fn test_pause_resume_via_json() {
    // Single-step workflow that sets state, then emits StopEvent with a delay.
    let step: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let _start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                ctx.set("value", 999_u64).await;
                tokio::time::sleep(Duration::from_millis(50)).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"done": true}),
                })))
            })
        });

        StepRegistration {
            name: "json_test_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
        }
    };

    let workflow = WorkflowBuilder::new("json-roundtrip-test")
        .step(step.clone())
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow.run(serde_json::json!(null)).await.unwrap();

    let snapshot = handler.pause().await.unwrap();

    // Serialize to JSON and back.
    let json_str = snapshot.to_json().unwrap();
    let restored = WorkflowSnapshot::from_json(&json_str).unwrap();

    // Verify the round-trip preserved everything.
    assert_eq!(restored.workflow_name, "json-roundtrip-test");
    assert_eq!(
        restored.context_state.get("value"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(999)))
    );
    assert_eq!(restored.pending_events.len(), snapshot.pending_events.len());

    // Resume from the deserialized snapshot.
    let resumed_handler = Workflow::resume(restored, vec![step], None).await.unwrap();

    let result = resumed_handler.result().await.unwrap();
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["done"], true);
}
