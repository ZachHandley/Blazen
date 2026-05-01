//! Integration tests for the Blazen workflow engine.
//!
//! These tests exercise the full workflow lifecycle: building workflows,
//! running them, verifying event routing, branching, fan-out, streaming,
//! context state sharing, and macro integration.

use std::any::Any;
use std::sync::{Arc, Mutex};
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
        semaphore: None,
        timeout: None,
        retry_config: None,
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
    let result = handler.result().await.unwrap().event;

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
            semaphore: None,
            timeout: None,
            retry_config: None,
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
    let result = handler.result().await.unwrap().event;
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
    let result2 = handler2.result().await.unwrap().event;
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
    let result = handler.result().await.unwrap().event;
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
            semaphore: None,
            timeout: None,
            retry_config: None,
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
    let result = handler.result().await.unwrap().event;

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
            semaphore: None,
            timeout: None,
            retry_config: None,
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

    let result = handler.result().await.unwrap().event;
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
            semaphore: None,
            timeout: None,
            retry_config: None,
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
            semaphore: None,
            timeout: None,
            retry_config: None,
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
    let result = handler.result().await.unwrap().event;
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
            semaphore: None,
            timeout: None,
            retry_config: None,
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
    let result = handler.result().await.unwrap().event;
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
    let result = handler.result().await.unwrap().event;
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
    let result = handler.result().await.unwrap().event;
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
    let result2 = handler2.result().await.unwrap().event;
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
    let result = handler.result().await.unwrap().event;
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
            semaphore: None,
            timeout: None,
            retry_config: None,
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

    // Let the step start and write context state (ctx writes happen before the
    // 200ms sleep inside the step).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Park the event loop, then capture a snapshot.
    handler.pause().unwrap();
    let snapshot = handler.snapshot().await.unwrap();

    // Verify snapshot metadata.
    assert_eq!(snapshot.workflow_name, "pause-test");
    assert!(!snapshot.run_id.is_nil());

    // Verify context state was captured. The step writes these before its
    // internal sleep, so they should be present even though the step is
    // still in-flight.
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

    // Note: `pending_events` is empty because the in-place snapshot cannot
    // peek at the mpsc channel. This is expected with the new API.

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
            semaphore: None,
            timeout: None,
            retry_config: None,
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
            semaphore: None,
            timeout: None,
            retry_config: None,
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

    // Let step_one start and write context state (it writes before its
    // 100ms sleep). We pause while the step is still in-flight.
    tokio::time::sleep(Duration::from_millis(50)).await;

    handler.pause().unwrap();
    let mut snapshot = handler.snapshot().await.unwrap();
    handler.abort().unwrap();

    // Verify snapshot has the state from step_one.
    assert_eq!(
        snapshot.context_state.get("counter"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(1)))
    );
    assert_eq!(
        snapshot.context_state.get("step_one_ran"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(true)))
    );

    // The in-place snapshot cannot capture pending channel events, so we
    // manually inject the AnalyzeEvent that step_one emitted. This
    // simulates what a checkpoint store with drain semantics would do.
    snapshot.pending_events.push(blazen_core::SerializedEvent {
        event_type: AnalyzeEvent::event_type().to_owned(),
        data: serde_json::json!({
            "text": "from step one",
            "word_count": 3,
        }),
        source_step: Some("step_one".to_owned()),
    });

    // Resume with the same step set. With the flat DynamicEvent::to_json()
    // format, the same handler works for both fresh and resumed runs.
    let resumed_handler = Workflow::resume(
        snapshot,
        vec![step_one, step_two],
        None, // no timeout
    )
    .await
    .unwrap();

    let result = resumed_handler.result().await.unwrap().event;
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
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("json-roundtrip-test")
        .step(step.clone())
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow.run(serde_json::json!(null)).await.unwrap();

    // Let the step start and write context state (it writes before its 50ms sleep).
    tokio::time::sleep(Duration::from_millis(20)).await;

    handler.pause().unwrap();
    let mut snapshot = handler.snapshot().await.unwrap();
    handler.abort().unwrap();

    // Inject the StopEvent the step would have emitted so the resumed
    // workflow can complete. The in-place snapshot cannot peek at the
    // mpsc channel, so we simulate what a drain-capable checkpoint store
    // would provide.
    snapshot.pending_events.push(blazen_core::SerializedEvent {
        event_type: StopEvent::event_type().to_owned(),
        data: serde_json::json!({"done": true}),
        source_step: Some("json_test_step".to_owned()),
    });

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

    let result = resumed_handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["done"], true);
}

// ===========================================================================
// 14. Cross-step opaque object sharing
// ===========================================================================

#[tokio::test]
async fn test_cross_step_object_sharing() {
    // StartEvent -> obj_writer (stores Arc<Mutex<Vec<String>>>, pushes "step1")
    //            -> StepAEvent
    //            -> obj_reader (retrieves the same Arc, pushes "step2")
    //            -> StopEvent
    //
    // After completion, the shared Vec should contain ["step1", "step2"].

    let shared_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_for_writer = shared_log.clone();
    let log_for_verification = shared_log.clone();

    let obj_writer: StepRegistration = {
        let handler: StepFn = Arc::new(move |event: Box<dyn AnyEvent>, ctx: Context| {
            let log = log_for_writer.clone();
            Box::pin(async move {
                let _start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                // Store the live object in the context.
                ctx.set_object("log", log.clone()).await;

                // Push a value from step 1.
                log.lock().unwrap().push("step1".to_string());

                Ok(StepOutput::Single(Box::new(StepAEvent { value: 1 })))
            })
        });

        StepRegistration {
            name: "obj_writer".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StepAEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let obj_reader: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                // Retrieve the same live object from the context.
                let log = ctx
                    .get_object::<Arc<Mutex<Vec<String>>>>("log")
                    .await
                    .expect("shared log object should be present in context");

                // Push a value from step 2.
                log.lock().unwrap().push("step2".to_string());

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"done": true}),
                })))
            })
        });

        StepRegistration {
            name: "obj_reader".to_string(),
            accepts: vec![StepAEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("cross-step-object-sharing")
        .step(obj_writer)
        .step(obj_reader)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"trigger": true}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["done"], true);

    // Verify the shared object contains entries from both steps.
    let final_log = log_for_verification.lock().unwrap();
    assert_eq!(
        *final_log,
        vec!["step1".to_string(), "step2".to_string()],
        "expected both steps to have written to the shared object"
    );
}

// ===========================================================================
// 15. Pause then resume_in_place completes the workflow
// ===========================================================================

#[tokio::test]
async fn test_pause_resume_in_place() {
    // A step that sets context state, sleeps 200ms, then returns StopEvent.
    let step: StepRegistration = {
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

                ctx.set("key", "value".to_string()).await;
                tokio::time::sleep(Duration::from_millis(200)).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: start.data.clone(),
                })))
            })
        });

        StepRegistration {
            name: "pause_resume_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("pause-resume-in-place")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"test": "pause_resume"}))
        .await
        .unwrap();

    // Let the step start and set context state (before 200ms sleep).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Pause, then immediately resume in place.
    handler.pause().unwrap();
    handler.resume_in_place().unwrap();

    // The workflow should complete after the step finishes its sleep.
    let result = handler.result().await.unwrap();
    let stop_event = result.event.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop_event.result["test"], "pause_resume");
}

// ===========================================================================
// 16. Snapshot while running captures context state
// ===========================================================================

#[tokio::test]
async fn test_snapshot_while_running() {
    // A step that sets context state, sleeps, then returns StopEvent.
    let step: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                ctx.set("snapshot_key", "snapshot_value".to_string()).await;
                tokio::time::sleep(Duration::from_millis(200)).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"done": true}),
                })))
            })
        });

        StepRegistration {
            name: "snapshot_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("snapshot-running")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"input": "snap"}))
        .await
        .unwrap();

    // Wait for the step to set state before the 200ms sleep.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Pause, then take a snapshot.
    handler.pause().unwrap();
    let snapshot = handler.snapshot().await.unwrap();

    // The step wrote state before its sleep, so context_state should be non-empty.
    assert!(
        !snapshot.context_state.is_empty(),
        "snapshot context_state should be non-empty"
    );
    assert_eq!(
        snapshot.context_state.get("snapshot_key"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(
            "snapshot_value"
        )))
    );

    // Resume and let the workflow complete.
    handler.resume_in_place().unwrap();
    let result = handler.result().await.unwrap();
    let stop_event = result.event.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop_event.result["done"], true);
}

// ===========================================================================
// 17. Abort drops the workflow cleanly (doesn't hang)
// ===========================================================================

#[tokio::test]
async fn test_abort_drops_cleanly() {
    // A step that sleeps for 5 seconds — if abort doesn't work the test hangs.
    let step: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"should_not": "reach"}),
                })))
            })
        });

        StepRegistration {
            name: "slow_aborting_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("abort-test")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow.run(serde_json::json!(null)).await.unwrap();

    // Give the step a moment to start, then abort.
    tokio::time::sleep(Duration::from_millis(20)).await;
    handler.abort().unwrap();

    // result() should return an error (ChannelClosed), not hang for 5 seconds.
    let result = tokio::time::timeout(Duration::from_secs(2), handler.result()).await;
    assert!(
        result.is_ok(),
        "result() should resolve quickly after abort, not time out"
    );
    let inner = result.unwrap();
    assert!(inner.is_err(), "result() should be Err after abort, got Ok");
}

// ===========================================================================
// 18. WorkflowResult carries session refs from steps
// ===========================================================================

#[tokio::test]
async fn test_workflow_result_carries_registry() {
    // A step that inserts a value into the session ref registry.
    let step: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let registry = ctx.session_refs_arc().await;
                registry.insert(42_i32).await.unwrap();

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"inserted": true}),
                })))
            })
        });

        StepRegistration {
            name: "session_ref_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("session-ref-result")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();

    let handler = workflow.run(serde_json::json!(null)).await.unwrap();

    let result = handler.result().await.unwrap();

    // The session ref registry on the result should contain what the step inserted.
    assert_eq!(
        result.session_refs.len().await,
        1,
        "session_refs should have exactly 1 entry"
    );
}

// ===========================================================================
// 18b. Sub-workflow session ref handoff via a shared registry Arc
// ===========================================================================
// Regression test for `FUTURE_ROADMAP.md` §0 (Phase 0.3). When a parent
// workflow invokes a sub-workflow and passes its own `SessionRefRegistry`
// via `run_with_registry`, the sub-workflow's inserts must remain
// resolvable via the PARENT registry **after** the sub-workflow handler
// has finished. This is the whole point of Option A (parent-owned
// registry for sub-workflow results).

#[tokio::test]
async fn test_sub_workflow_session_ref_handoff_via_shared_registry() {
    use blazen_core::session_ref::RegistryKey;

    // Build the inner workflow first. Its one step inserts a value into
    // the *current* session ref registry (which will be the parent's Arc
    // once we invoke it via `run_with_registry`) and returns the
    // RegistryKey inside `StopEvent.result` as a JSON marker.
    let inner_step: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let registry = ctx.session_refs_arc().await;
                let key = registry
                    .insert("the-secret-value".to_string())
                    .await
                    .unwrap();
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "__blazen_session_ref__": key.to_string(),
                    }),
                })))
            })
        });

        StepRegistration {
            name: "inner_insert".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let inner_workflow = Arc::new(
        WorkflowBuilder::new("inner")
            .step(inner_step)
            .no_timeout()
            .build()
            .unwrap(),
    );

    // Build the outer workflow. Its step calls
    // `inner.run_with_registry(input, parent_registry).await`, awaits
    // the sub-handler's result, and re-emits the sub-result JSON as
    // its own `StopEvent.result` so we can inspect it from the test.
    let outer_step: StepRegistration = {
        let inner = Arc::clone(&inner_workflow);
        let handler: StepFn = Arc::new(move |_event: Box<dyn AnyEvent>, ctx: Context| {
            let inner = Arc::clone(&inner);
            Box::pin(async move {
                let parent_registry = ctx.session_refs_arc().await;
                let sub_handler = inner
                    .run_with_registry(serde_json::json!({}), Arc::clone(&parent_registry))
                    .await
                    .expect("sub-workflow run_with_registry should succeed");
                let sub_result_event = sub_handler
                    .result()
                    .await
                    .expect("sub result should be Ok")
                    .event;
                let sub_stop = sub_result_event
                    .downcast_ref::<StopEvent>()
                    .expect("inner workflow should emit a StopEvent");
                let sub_result = sub_stop.result.clone();
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: sub_result,
                })))
            })
        });

        StepRegistration {
            name: "outer_invoke".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let outer_workflow = WorkflowBuilder::new("outer")
        .step(outer_step)
        .no_timeout()
        .build()
        .unwrap();

    let outer_handler = outer_workflow.run(serde_json::json!(null)).await.unwrap();
    let outer_result = outer_handler.result().await.unwrap();
    let outer_stop_event = outer_result
        .event
        .downcast_ref::<StopEvent>()
        .expect("outer workflow should emit a StopEvent");

    // The outer result JSON should carry the session_ref marker the inner
    // step produced.
    let ref_str = outer_stop_event
        .result
        .get("__blazen_session_ref__")
        .and_then(serde_json::Value::as_str)
        .expect("outer result should carry __blazen_session_ref__ marker");
    let key = RegistryKey::parse(ref_str).expect("ref str is a valid UUID");

    // The actual object is STILL resolvable via the OUTER result's
    // registry even though the inner workflow handler has already
    // finished and its handle has been dropped. This is the whole point
    // of sharing the registry Arc across parent/child.
    let got: Arc<String> = outer_result
        .session_refs
        .get::<String>(key)
        .await
        .expect("inner-inserted session ref must survive on the parent registry");
    assert_eq!(*got, "the-secret-value");
}

// ===========================================================================
// 19. SessionPausePolicy::HardError prevents snapshot when refs exist
// ===========================================================================

#[tokio::test]
async fn test_session_pause_policy_hard_error() {
    use blazen_core::SessionPausePolicy;

    // A step that inserts a session ref, then sleeps so we can pause it.
    let step: StepRegistration = {
        let handler: StepFn = Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let registry = ctx.session_refs_arc().await;
                registry.insert(99_i32).await.unwrap();

                // Sleep to give the test time to pause.
                tokio::time::sleep(Duration::from_millis(500)).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"done": true}),
                })))
            })
        });

        StepRegistration {
            name: "hard_error_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    let workflow = WorkflowBuilder::new("hard-error-policy")
        .step(step)
        .no_timeout()
        .session_pause_policy(SessionPausePolicy::HardError)
        .build()
        .unwrap();

    let handler = workflow.run(serde_json::json!(null)).await.unwrap();

    // Wait for the step to insert the session ref.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Pause, then try to snapshot — should fail with SessionRefsNotSerializable.
    handler.pause().unwrap();
    let snapshot_result = handler.snapshot().await;

    assert!(
        snapshot_result.is_err(),
        "snapshot should fail with HardError policy when session refs exist"
    );

    match snapshot_result.unwrap_err() {
        WorkflowError::SessionRefsNotSerializable { keys } => {
            assert_eq!(keys.len(), 1, "should report exactly 1 session ref key");
        }
        other => panic!("expected SessionRefsNotSerializable, got: {other:?}"),
    }

    // Clean up: abort the workflow so the test doesn't hang.
    handler.abort().unwrap();
}

// ===========================================================================
// 20. Child SessionPausePolicy::HardError fires on parent-owned refs in a
//     shared registry
// ===========================================================================
// When a parent and child share one `SessionRefRegistry` via
// `run_with_registry`, the child's `SessionPausePolicy` is enforced against
// the ENTIRE shared registry — including refs the parent inserted before
// invoking the child. This test pins that current behavior: a child with
// `HardError` will reject a mid-run snapshot solely because the parent has
// a live ref in the registry. When `RefLifetime` (roadmap Phase 11.2)
// lands, the correct semantics will be "child HardError only applies to
// refs the child inserted" and this test will need to be updated.

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn test_sub_workflow_hard_error_policy_sees_parent_refs() {
    use blazen_core::SessionPausePolicy;
    use blazen_core::session_ref::RegistryKey;
    use tokio::sync::{Mutex, oneshot};

    // Child workflow: one slow step, inserts NO refs of its own, so any ref
    // its HardError policy sees must come from the parent's registry.
    let child_step: StepRegistration = StepRegistration {
        name: "child_slow_step".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(500)).await;
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"child_done": true}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
        timeout: None,
        retry_config: None,
    };
    let child_workflow = Arc::new(
        WorkflowBuilder::new("child-hard-error")
            .step(child_step)
            .no_timeout()
            .session_pause_policy(SessionPausePolicy::HardError)
            .build()
            .unwrap(),
    );

    // Parent step inserts a ref, launches the child, ships the child handle
    // to the test body, then waits for `resume_tx` before finishing.
    let (child_handle_tx, child_handle_rx) = oneshot::channel::<blazen_core::WorkflowHandler>();
    let (resume_tx, resume_rx) = oneshot::channel::<()>();
    let child_handle_tx = Arc::new(Mutex::new(Some(child_handle_tx)));
    let resume_rx = Arc::new(Mutex::new(Some(resume_rx)));

    let parent_step: StepRegistration = StepRegistration {
        name: "parent_invoke".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler: {
            let child = Arc::clone(&child_workflow);
            let tx_cell = Arc::clone(&child_handle_tx);
            let rx_cell = Arc::clone(&resume_rx);
            Arc::new(move |_event: Box<dyn AnyEvent>, ctx: Context| {
                let child = Arc::clone(&child);
                let tx_cell = Arc::clone(&tx_cell);
                let rx_cell = Arc::clone(&rx_cell);
                Box::pin(async move {
                    let registry = ctx.session_refs_arc().await;
                    let parent_key = registry.insert(7777_i32).await.unwrap();
                    let sub = child
                        .run_with_registry(serde_json::json!({}), Arc::clone(&registry))
                        .await
                        .unwrap();
                    let send_tx = tx_cell.lock().await.take().unwrap();
                    send_tx.send(sub).ok().unwrap();
                    let wait_rx = rx_cell.lock().await.take().unwrap();
                    wait_rx.await.unwrap();
                    Ok(StepOutput::Single(Box::new(StopEvent {
                        result: serde_json::json!({ "parent_ref_key": parent_key.to_string() }),
                    })))
                })
            })
        },
        max_concurrency: 1,
        semaphore: None,
        timeout: None,
        retry_config: None,
    };

    let parent_handler = WorkflowBuilder::new("parent-shared-registry")
        .step(parent_step)
        .no_timeout()
        .build()
        .unwrap()
        .run(serde_json::json!(null))
        .await
        .unwrap();

    // Grab the child handle and give the slow step time to start running.
    let child_handler = child_handle_rx.await.expect("child handle delivered");
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Pause + snapshot the CHILD. Its HardError policy fires against the
    // shared registry and raises SessionRefsNotSerializable — the offending
    // ref is the PARENT's 7777_i32 entry. The child inserted nothing.
    child_handler.pause().unwrap();
    let snapshot_result = child_handler.snapshot().await;
    match snapshot_result {
        Err(WorkflowError::SessionRefsNotSerializable { keys }) => {
            assert_eq!(keys.len(), 1, "shared registry holds only the parent's ref");
        }
        other => panic!("expected SessionRefsNotSerializable, got: {other:?}"),
    }

    // Clean up the child, release the parent step, then verify the parent's
    // ref is STILL resolvable on the returned WorkflowResult.
    child_handler.abort().unwrap();
    drop(child_handler);
    resume_tx.send(()).expect("parent step should be waiting");

    let parent_result = parent_handler.result().await.unwrap();
    let final_stop = parent_result
        .event
        .downcast_ref::<StopEvent>()
        .expect("parent emits StopEvent");
    let parent_key = RegistryKey::parse(
        final_stop.result["parent_ref_key"]
            .as_str()
            .expect("parent_ref_key present"),
    )
    .expect("valid uuid");
    assert_eq!(parent_result.session_refs.len().await, 1);
    let got: Arc<i32> = parent_result
        .session_refs
        .get::<i32>(parent_key)
        .await
        .expect("parent ref still resolvable after child abort");
    assert_eq!(*got, 7777);
}

// ===========================================================================
// 21. RefLifetime::UntilSnapshot — ephemeral refs are purged when a
//     snapshot is built mid-flight, while default-lifetime refs survive.
// ===========================================================================

#[tokio::test]
async fn test_until_snapshot_lifetime_purged_by_snapshot_walker() {
    use blazen_core::session_ref::{RefLifetime, RegistryKey};
    use tokio::sync::oneshot;

    // The step inserts two refs: one with the default lifetime
    // (UntilContextDrop), one with UntilSnapshot. It ships both keys
    // out via a oneshot so the test can verify post-snapshot state,
    // then sleeps long enough for the test to pause+snapshot.
    let (keys_tx, keys_rx) = oneshot::channel::<(RegistryKey, RegistryKey)>();
    let keys_tx = std::sync::Mutex::new(Some(keys_tx));

    let step: StepRegistration = {
        let handler: StepFn = Arc::new(move |_event: Box<dyn AnyEvent>, ctx: Context| {
            let tx = keys_tx
                .lock()
                .unwrap()
                .take()
                .expect("step handler called more than once");
            Box::pin(async move {
                let registry = ctx.session_refs_arc().await;

                // Default lifetime — should survive the snapshot.
                let durable_key = registry.insert(111_i32).await.unwrap();

                // UntilSnapshot lifetime — should be purged by the
                // snapshot walker before the configured pause policy
                // runs.
                let ephemeral_key = registry
                    .insert_with_lifetime("ephemeral".to_owned(), RefLifetime::UntilSnapshot)
                    .await
                    .unwrap();

                // Hand the keys back to the test body.
                tx.send((durable_key, ephemeral_key)).ok();

                // Sleep long enough for the test to pause+snapshot.
                tokio::time::sleep(Duration::from_millis(300)).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"done": true}),
                })))
            })
        });

        StepRegistration {
            name: "until_snapshot_step".to_string(),
            accepts: vec![StartEvent::event_type()],
            emits: vec![StopEvent::event_type()],
            handler,
            max_concurrency: 1,
            semaphore: None,
            timeout: None,
            retry_config: None,
        }
    };

    // Use WarnDrop so the snapshot walker doesn't error on the
    // surviving default-lifetime ref — we want it to complete
    // successfully and we'll verify the registry contents directly.
    let workflow = WorkflowBuilder::new("until-snapshot-purge")
        .step(step)
        .no_timeout()
        .session_pause_policy(blazen_core::SessionPausePolicy::WarnDrop)
        .build()
        .unwrap();

    let handler = workflow.run(serde_json::json!(null)).await.unwrap();

    // Wait for the step to insert both refs.
    let (durable_key, ephemeral_key) = keys_rx.await.expect("step delivered keys");

    // Both refs are live before the snapshot.
    let registry = handler.session_refs();
    assert!(
        registry.get::<i32>(durable_key).await.is_some(),
        "durable ref should be present before snapshot"
    );
    assert!(
        registry.get::<String>(ephemeral_key).await.is_some(),
        "ephemeral ref should be present before snapshot"
    );
    assert_eq!(
        registry.lifetime_of(ephemeral_key).await,
        Some(RefLifetime::UntilSnapshot),
    );

    // Pause and snapshot. The snapshot walker should purge the
    // UntilSnapshot ref BEFORE the WarnDrop policy looks at the
    // registry.
    handler.pause().unwrap();
    let snapshot = handler
        .snapshot()
        .await
        .expect("snapshot should succeed under WarnDrop");

    // Ephemeral ref must be gone (purged by the snapshot walker).
    assert!(
        registry.get::<String>(ephemeral_key).await.is_none(),
        "UntilSnapshot ref must be purged by the snapshot walker"
    );
    assert!(
        registry.lifetime_of(ephemeral_key).await.is_none(),
        "UntilSnapshot lifetime sidecar must be cleared too"
    );

    // Durable default-lifetime ref must still be live.
    assert!(
        registry.get::<i32>(durable_key).await.is_some(),
        "default-lifetime ref must survive a snapshot"
    );
    assert_eq!(
        registry.lifetime_of(durable_key).await,
        Some(RefLifetime::UntilContextDrop),
    );

    // Snapshot metadata must report the durable ref as dropped (it
    // wasn't pickled because no pickler is registered) but must NOT
    // mention the ephemeral key — the walker purged it before the
    // policy ran, so the policy never saw it.
    let dropped = snapshot
        .metadata
        .get("__blazen_dropped_session_refs")
        .expect("WarnDrop policy records dropped refs in snapshot metadata");
    let dropped_keys: Vec<String> = serde_json::from_value(dropped.clone()).unwrap();
    let durable_str = durable_key.to_string();
    let ephemeral_str = ephemeral_key.to_string();
    assert!(
        dropped_keys.contains(&durable_str),
        "durable key must appear in dropped refs metadata, got {dropped_keys:?}",
    );
    assert!(
        !dropped_keys.contains(&ephemeral_str),
        "ephemeral key must NOT appear in dropped refs metadata \
         (snapshot walker purged it before the policy ran), got {dropped_keys:?}",
    );

    // Clean up.
    handler.abort().unwrap();
}

// ===========================================================================
// Phase 12.2 — Step deserializer registry
// ===========================================================================
//
// These tests exercise `Workflow::new_from_registered_steps` end-to-end:
// a step builder is registered in the process-global
// `StepDeserializerRegistry`, the workflow is rebuilt by looking up the
// step ID, and the workflow runs to completion. Tests use unique,
// test-specific step IDs so they do not collide with each other or with
// the inline unit tests in `step_registry.rs`.

/// Step ID used by the `new_workflow_from_registered_steps` test.
const STEP_ID_ECHO: &str = "blazen_core::tests::phase_12_2::echo";

/// Step ID used by the `registered_step_ids_lists_all` test (second entry).
const STEP_ID_NOOP: &str = "blazen_core::tests::phase_12_2::noop";

fn registry_echo_step() -> StepRegistration {
    let handler: StepFn = Arc::new(|event, _ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("expected StartEvent");
            let stop = StopEvent {
                result: start.data.clone(),
            };
            Ok(StepOutput::Single(Box::new(stop)))
        })
    });

    StepRegistration {
        name: "registry_echo".into(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
        semaphore: None,
        timeout: None,
        retry_config: None,
    }
}

fn registry_noop_step() -> StepRegistration {
    let handler: StepFn = Arc::new(|_event, _ctx| Box::pin(async move { Ok(StepOutput::None) }));

    StepRegistration {
        name: "registry_noop".into(),
        // Accept a custom event so it does not collide with the echo
        // step on StartEvent dispatch if both were registered in the
        // same workflow.
        accepts: vec!["test::phase_12_2::NoopEvent"],
        emits: vec![],
        handler,
        max_concurrency: 0,
        semaphore: None,
        timeout: None,
        retry_config: None,
    }
}

#[tokio::test]
async fn new_workflow_from_registered_steps() {
    use blazen_core::register_step_builder;

    register_step_builder(STEP_ID_ECHO, registry_echo_step);

    let workflow =
        Workflow::new_from_registered_steps("phase-12-2-rebuilt", vec![STEP_ID_ECHO]).unwrap();

    let handler = workflow
        .run(serde_json::json!({"hello": "distributed"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    assert_eq!(result.event_type_id(), StopEvent::event_type());

    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result, serde_json::json!({"hello": "distributed"}));
}

#[tokio::test]
async fn unknown_step_id_errors() {
    let err = Workflow::new_from_registered_steps(
        "phase-12-2-missing",
        vec!["blazen_core::tests::phase_12_2::does_not_exist"],
    )
    .unwrap_err();

    match err {
        WorkflowError::UnknownStep { step_id } => {
            assert_eq!(step_id, "blazen_core::tests::phase_12_2::does_not_exist");
        }
        other => panic!("expected UnknownStep, got {other:?}"),
    }
}

#[tokio::test]
async fn registered_step_ids_lists_all() {
    use blazen_core::{register_step_builder, registered_step_ids};

    register_step_builder(STEP_ID_ECHO, registry_echo_step);
    register_step_builder(STEP_ID_NOOP, registry_noop_step);

    let ids = registered_step_ids();
    assert!(
        ids.contains(&STEP_ID_ECHO),
        "registered_step_ids() must include {STEP_ID_ECHO}, got {ids:?}",
    );
    assert!(
        ids.contains(&STEP_ID_NOOP),
        "registered_step_ids() must include {STEP_ID_NOOP}, got {ids:?}",
    );
}
