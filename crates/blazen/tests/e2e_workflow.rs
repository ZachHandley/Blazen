//! End-to-end tests exercising the full Blazen aggregator crate.
//!
//! These tests verify the public API surface that Rust consumers use,
//! importing from `blazen` (not `blazen-core` directly). Several tests
//! use `DynamicEvent` to mirror the cross-language event pattern used
//! by the Python and Node.js bindings.

use std::sync::Arc;
use std::time::Duration;

use blazen::prelude::*;
use blazen::{DynamicEvent, StepFn, intern_event_type};

// ===========================================================================
// 1. Single step echo: StartEvent -> echo -> StopEvent
// ===========================================================================

#[tokio::test]
async fn test_e2e_single_step_echo() {
    let echo = StepRegistration {
        name: "echo".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let start = event
                    .as_any()
                    .downcast_ref::<StartEvent>()
                    .ok_or(WorkflowError::EventDowncastFailed {
                        expected: StartEvent::event_type(),
                        got: event.event_type_id().to_string(),
                    })?
                    .clone();

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: start.data,
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let workflow = WorkflowBuilder::new("e2e-echo").step(echo).build().unwrap();

    let handler = workflow
        .run(serde_json::json!({"message": "hello e2e"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["message"], "hello e2e");
}

// ===========================================================================
// 2. Multi-step pipeline with DynamicEvent routing
// ===========================================================================

#[tokio::test]
async fn test_e2e_multi_step_dynamic_pipeline() {
    // StartEvent -> step_a (emits DynamicEvent "AnalyzeEvent")
    //            -> step_b (accepts "AnalyzeEvent", emits DynamicEvent "EnrichEvent")
    //            -> step_c (accepts "EnrichEvent", emits StopEvent)
    //
    // This mirrors how Python/Node bindings route events by string type name.

    let analyze_type = intern_event_type("AnalyzeEvent");
    let enrich_type = intern_event_type("EnrichEvent");

    let step_a = StepRegistration {
        name: "step_a".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![analyze_type],
        handler: Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let start = event.as_any().downcast_ref::<StartEvent>().unwrap().clone();
                let text = start.data["text"].as_str().unwrap_or_default().to_string();

                Ok(StepOutput::Single(Box::new(DynamicEvent {
                    event_type: "AnalyzeEvent".to_string(),
                    data: serde_json::json!({"text": text, "word_count": text.split_whitespace().count()}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let step_b = StepRegistration {
        name: "step_b".to_string(),
        accepts: vec![analyze_type],
        emits: vec![enrich_type],
        handler: Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let json = event.to_json();
                let text = json["text"].as_str().unwrap_or_default();
                let word_count = json["word_count"].as_u64().unwrap_or(0);

                Ok(StepOutput::Single(Box::new(DynamicEvent {
                    event_type: "EnrichEvent".to_string(),
                    data: serde_json::json!({
                        "text": text.to_uppercase(),
                        "word_count": word_count,
                        "enriched": true,
                    }),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let step_c = StepRegistration {
        name: "step_c".to_string(),
        accepts: vec![enrich_type],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let json = event.to_json();

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "text": json["text"],
                        "word_count": json["word_count"],
                        "enriched": json["enriched"],
                    }),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let workflow = WorkflowBuilder::new("e2e-dynamic-pipeline")
        .step(step_a)
        .step(step_b)
        .step(step_c)
        .build()
        .unwrap();

    let handler = workflow
        .run(serde_json::json!({"text": "hello world"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["text"], "HELLO WORLD");
    assert_eq!(stop.result["word_count"], 2);
    assert_eq!(stop.result["enriched"], true);
}

// ===========================================================================
// 3. Branching with DynamicEvent routing
// ===========================================================================

#[tokio::test]
async fn test_e2e_branching_dynamic_events() {
    let high_type = intern_event_type("HighPriority");
    let low_type = intern_event_type("LowPriority");

    let router = StepRegistration {
        name: "router".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![high_type, low_type],
        handler: Arc::new(|event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                let start = event.as_any().downcast_ref::<StartEvent>().unwrap().clone();
                let priority = start.data["priority"].as_str().unwrap_or("low");

                if priority == "high" {
                    Ok(StepOutput::Single(Box::new(DynamicEvent {
                        event_type: "HighPriority".to_string(),
                        data: start.data.clone(),
                    })))
                } else {
                    Ok(StepOutput::Single(Box::new(DynamicEvent {
                        event_type: "LowPriority".to_string(),
                        data: start.data.clone(),
                    })))
                }
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let handle_high = StepRegistration {
        name: "handle_high".to_string(),
        accepts: vec![high_type],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"handled": "high"}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let handle_low = StepRegistration {
        name: "handle_low".to_string(),
        accepts: vec![low_type],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"handled": "low"}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    // Test high priority branch.
    let wf = WorkflowBuilder::new("e2e-branch-high")
        .step(router.clone())
        .step(handle_high.clone())
        .step(handle_low.clone())
        .build()
        .unwrap();

    let handler = wf
        .run(serde_json::json!({"priority": "high"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop.result["handled"], "high");

    // Test low priority branch.
    let wf2 = WorkflowBuilder::new("e2e-branch-low")
        .step(router)
        .step(handle_high)
        .step(handle_low)
        .build()
        .unwrap();

    let handler2 = wf2
        .run(serde_json::json!({"priority": "low"}))
        .await
        .unwrap();
    let result2 = handler2.result().await.unwrap().event;
    let stop2 = result2.downcast_ref::<StopEvent>().unwrap();
    assert_eq!(stop2.result["handled"], "low");
}

// ===========================================================================
// 4. Context sharing across steps
// ===========================================================================

#[tokio::test]
async fn test_e2e_context_sharing() {
    #[derive(Debug, Clone, Serialize, Deserialize, Event)]
    struct MidEvent {
        value: i32,
    }

    let writer = StepRegistration {
        name: "writer".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![MidEvent::event_type()],
        handler: Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let start = event.as_any().downcast_ref::<StartEvent>().unwrap().clone();
                let text = start.data["text"].as_str().unwrap_or_default().to_string();

                ctx.set("shared_text", text.clone()).await;
                ctx.set("shared_len", text.len() as u64).await;

                Ok(StepOutput::Single(Box::new(MidEvent { value: 42 })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let reader = StepRegistration {
        name: "reader".to_string(),
        accepts: vec![MidEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let text = ctx.get::<String>("shared_text").await.unwrap_or_default();
                let len = ctx.get::<u64>("shared_len").await.unwrap_or(0);

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "shared_text": text,
                        "shared_len": len,
                    }),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let wf = WorkflowBuilder::new("e2e-context")
        .step(writer)
        .step(reader)
        .build()
        .unwrap();

    let handler = wf
        .run(serde_json::json!({"text": "hello world"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["shared_text"], "hello world");
    assert_eq!(stop.result["shared_len"], 11);
}

// ===========================================================================
// 5. Streaming events
// ===========================================================================

#[tokio::test]
async fn test_e2e_streaming_events() {
    use tokio_stream::StreamExt;

    let process: StepRegistration = {
        let handler: StepFn = Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let start = event.as_any().downcast_ref::<StartEvent>().unwrap().clone();

                // Publish progress events to external stream.
                for i in 0..3 {
                    ctx.write_event_to_stream(DynamicEvent {
                        event_type: "Progress".to_string(),
                        data: serde_json::json!({"step": i}),
                    })
                    .await;
                }

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: start.data,
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
        }
    };

    let wf = WorkflowBuilder::new("e2e-streaming")
        .step(process)
        .build()
        .unwrap();

    let handler = wf.run(serde_json::json!({"input": "test"})).await.unwrap();

    // Subscribe before awaiting result.
    let mut stream = handler.stream_events();

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

    let collected = collect_task.await.unwrap();
    let progress_events: Vec<_> = collected
        .iter()
        .filter(|e| e.event_type_id() == "Progress")
        .collect();

    assert!(
        !progress_events.is_empty(),
        "expected at least one Progress event in stream"
    );
}

// ===========================================================================
// 6. Pause and resume
// ===========================================================================

#[tokio::test]
async fn test_e2e_pause_and_resume() {
    #[derive(Debug, Clone, Serialize, Deserialize, Event)]
    struct SlowMidEvent {
        value: String,
    }

    let step_one = StepRegistration {
        name: "step_one".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![SlowMidEvent::event_type()],
        handler: Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let start = event.as_any().downcast_ref::<StartEvent>().unwrap().clone();
                ctx.set("step_one_ran", true).await;

                // Sleep to keep step in-flight when pause is requested.
                tokio::time::sleep(Duration::from_millis(100)).await;

                Ok(StepOutput::Single(Box::new(SlowMidEvent {
                    value: start.data["input"].as_str().unwrap_or_default().to_string(),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    // Step two reads via to_json() -- with the deserializer registry, events
    // are reconstructed as concrete types on resume, so to_json() always
    // returns the flat data regardless of fresh vs resumed.
    let step_two = StepRegistration {
        name: "step_two".to_string(),
        accepts: vec![SlowMidEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|event: Box<dyn AnyEvent>, ctx: Context| {
            Box::pin(async move {
                let json = event.to_json();
                let value = json["value"].as_str().unwrap_or_default().to_string();

                let step_one_ran = ctx.get::<bool>("step_one_ran").await.unwrap_or(false);
                ctx.set("step_two_ran", true).await;

                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "value": value,
                        "step_one_ran": step_one_ran,
                    }),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let wf = WorkflowBuilder::new("e2e-pause-resume")
        .step(step_one.clone())
        .step(step_two.clone())
        .no_timeout()
        .build()
        .unwrap();

    let handler = wf
        .run(serde_json::json!({"input": "paused_data"}))
        .await
        .unwrap();

    // Let step_one start and write context state (it writes before its
    // 100ms sleep). We pause while the step is still in-flight.
    tokio::time::sleep(Duration::from_millis(50)).await;

    handler.pause().unwrap();
    let mut snapshot = handler.snapshot().await.unwrap();
    handler.abort().unwrap();

    assert_eq!(
        snapshot.context_state.get("step_one_ran"),
        Some(&blazen_core::StateValue::Json(serde_json::json!(true)))
    );

    // The in-place snapshot cannot capture pending channel events, so we
    // manually inject the SlowMidEvent that step_one emitted.
    snapshot.pending_events.push(blazen_core::SerializedEvent {
        event_type: SlowMidEvent::event_type().to_owned(),
        data: serde_json::json!({ "value": "paused_data" }),
        source_step: Some("step_one".to_owned()),
    });

    // Resume with the same steps.
    let resumed_handler = Workflow::resume(snapshot, vec![step_one, step_two], None)
        .await
        .unwrap();

    let result = resumed_handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["value"], "paused_data");
    assert_eq!(stop.result["step_one_ran"], true);
}

// ===========================================================================
// 7. Fan-out: step returns multiple events
// ===========================================================================

#[tokio::test]
async fn test_e2e_fan_out() {
    let fan_type_a = intern_event_type("FanA");
    let fan_type_b = intern_event_type("FanB");

    let fan_out = StepRegistration {
        name: "fan_out".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![fan_type_a, fan_type_b],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                Ok(StepOutput::Multiple(vec![
                    Box::new(DynamicEvent {
                        event_type: "FanA".to_string(),
                        data: serde_json::json!({"branch": "a"}),
                    }),
                    Box::new(DynamicEvent {
                        event_type: "FanB".to_string(),
                        data: serde_json::json!({"branch": "b"}),
                    }),
                ]))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let handle_a = StepRegistration {
        name: "handle_a".to_string(),
        accepts: vec![fan_type_a],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"winner": "a"}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let handle_b = StepRegistration {
        name: "handle_b".to_string(),
        accepts: vec![fan_type_b],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"winner": "b"}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let wf = WorkflowBuilder::new("e2e-fan-out")
        .step(fan_out)
        .step(handle_a)
        .step(handle_b)
        .build()
        .unwrap();

    let handler = wf.run(serde_json::json!({})).await.unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    // One of the two branches wins the race.
    let winner = stop.result["winner"].as_str().unwrap();
    assert!(
        winner == "a" || winner == "b",
        "unexpected winner: {winner}"
    );
}

// ===========================================================================
// 8. Timeout behavior
// ===========================================================================

#[tokio::test]
async fn test_e2e_timeout() {
    let slow_step = StepRegistration {
        name: "slow".to_string(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler: Arc::new(|_event: Box<dyn AnyEvent>, _ctx: Context| {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({"done": true}),
                })))
            })
        }),
        max_concurrency: 1,
        semaphore: None,
    };

    let wf = WorkflowBuilder::new("e2e-timeout")
        .step(slow_step)
        .timeout(Duration::from_millis(100))
        .build()
        .unwrap();

    let handler = wf.run(serde_json::json!({})).await.unwrap();
    let result = handler.result().await;

    assert!(result.is_err(), "expected timeout error, got success");
    match result.unwrap_err() {
        WorkflowError::Timeout { .. } => {}
        other => panic!("expected WorkflowError::Timeout, got: {other:?}"),
    }
}

// ===========================================================================
// 9. Derive macros: #[derive(Event)] + #[step]
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct E2eCustomEvent {
    message: String,
    score: f64,
}

#[step]
#[allow(clippy::unused_async)]
async fn e2e_macro_step_one(
    event: StartEvent,
    _ctx: Context,
) -> Result<E2eCustomEvent, WorkflowError> {
    let msg = event.data["msg"]
        .as_str()
        .unwrap_or_default()
        .to_uppercase();
    Ok(E2eCustomEvent {
        message: msg,
        score: 0.95,
    })
}

#[step]
#[allow(clippy::unused_async)]
async fn e2e_macro_step_two(
    event: E2eCustomEvent,
    _ctx: Context,
) -> Result<StopEvent, WorkflowError> {
    Ok(StopEvent {
        result: serde_json::json!({
            "message": event.message,
            "score": event.score,
        }),
    })
}

#[tokio::test]
async fn test_e2e_derive_macros() {
    let reg1 = e2e_macro_step_one_registration();
    let reg2 = e2e_macro_step_two_registration();

    assert_eq!(reg1.name, "e2e_macro_step_one");
    assert_eq!(reg1.accepts[0], StartEvent::event_type());

    assert_eq!(reg2.name, "e2e_macro_step_two");
    assert_eq!(reg2.accepts[0], E2eCustomEvent::event_type());

    let wf = WorkflowBuilder::new("e2e-macros")
        .step(reg1)
        .step(reg2)
        .build()
        .unwrap();

    let handler = wf
        .run(serde_json::json!({"msg": "macro test"}))
        .await
        .unwrap();
    let result = handler.result().await.unwrap().event;
    let stop = result.downcast_ref::<StopEvent>().unwrap();

    assert_eq!(stop.result["message"], "MACRO TEST");
    assert!((stop.result["score"].as_f64().unwrap() - 0.95).abs() < f64::EPSILON);
}
