//! Integration tests for [`WorkflowBuilder`] timeout-related convenience
//! methods.
//!
//! These tests exercise the public boundary: chaining `.step_timeout(...)`
//! and `.no_step_timeout()` after `.step(...)` and verifying that the
//! resulting workflow builds successfully. The detailed field-level
//! assertions live in `crates/blazen-core/src/builder.rs`'s in-crate
//! `#[cfg(test)] mod tests` (where the private `steps` field is
//! accessible).

use std::sync::Arc;
use std::time::Duration;

use blazen_core::{StepFn, StepOutput, StepRegistration, WorkflowBuilder};
use blazen_events::{Event, StartEvent, StopEvent};

fn make_step(name: &str) -> StepRegistration {
    let handler: StepFn = Arc::new(|_event, _ctx| {
        Box::pin(async move {
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!(null),
            })))
        })
    });
    StepRegistration::new(
        name.to_owned(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        handler,
        0,
    )
}

#[test]
fn step_timeout_chain_builds_workflow() {
    let workflow = WorkflowBuilder::new("test")
        .step(make_step("a"))
        .step_timeout(Duration::from_millis(100))
        .build()
        .expect("workflow with step_timeout should build");

    assert_eq!(workflow.step_names(), vec!["a".to_string()]);
}

#[test]
fn step_timeout_then_no_step_timeout_builds_workflow() {
    let workflow = WorkflowBuilder::new("test")
        .step(make_step("a"))
        .step_timeout(Duration::from_secs(1))
        .no_step_timeout()
        .build()
        .expect("workflow with cleared step_timeout should build");

    assert_eq!(workflow.step_names(), vec!["a".to_string()]);
}

#[test]
fn step_timeout_via_with_timeout_builds_workflow() {
    // Canonical path: configure timeout on the StepRegistration itself
    // before handing it to the builder.
    let step = make_step("a").with_timeout(Duration::from_millis(250));
    let workflow = WorkflowBuilder::new("test")
        .step(step)
        .build()
        .expect("workflow with StepRegistration::with_timeout should build");

    assert_eq!(workflow.step_names(), vec!["a".to_string()]);
}

#[test]
#[should_panic(expected = "step_timeout() called before any step was registered")]
fn step_timeout_without_prior_step_panics() {
    let _ = WorkflowBuilder::new("test").step_timeout(Duration::from_millis(100));
}

#[test]
#[should_panic(expected = "no_step_timeout() called before any step was registered")]
fn no_step_timeout_without_prior_step_panics() {
    let _ = WorkflowBuilder::new("test").no_step_timeout();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn step_timeout_fires_when_handler_exceeds_duration() {
    use blazen_core::WorkflowError;

    // A step that sleeps longer than its timeout.
    let slow_handler: StepFn = Arc::new(|_ev, _ctx| {
        Box::pin(async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!("ok"),
            })))
        })
    });
    let mut step = StepRegistration::new(
        "slow".into(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        slow_handler,
        0,
    );
    step.timeout = Some(Duration::from_millis(50));

    let workflow = WorkflowBuilder::new("step_timeout_test")
        .step(step)
        .no_timeout() // disable workflow-level timeout so step timeout is the cause
        .build()
        .expect("workflow builds");

    let handler = workflow
        .run(serde_json::json!({}))
        .await
        .expect("workflow starts");
    let result = handler.result().await;

    match result {
        Err(WorkflowError::StepTimeout {
            step_name,
            elapsed_ms,
        }) => {
            assert_eq!(step_name, "slow");
            assert!(elapsed_ms >= 50);
        }
        other => panic!("expected StepTimeout, got {other:?}"),
    }
}

#[test]
fn workflow_builder_retry_config_sets_field() {
    use blazen_llm::retry::RetryConfig;
    let workflow = WorkflowBuilder::new("retry-test")
        .step(make_step("a"))
        .retry_config(RetryConfig {
            max_retries: 7,
            ..RetryConfig::default()
        })
        .build()
        .expect("builds");
    assert!(workflow.retry_config.is_some());
    assert_eq!(workflow.retry_config.as_ref().unwrap().max_retries, 7);
}

#[test]
fn workflow_builder_no_retry_sets_max_retries_zero() {
    let workflow = WorkflowBuilder::new("no-retry-test")
        .step(make_step("a"))
        .no_retry()
        .build()
        .expect("builds");
    assert_eq!(workflow.retry_config.as_ref().unwrap().max_retries, 0);
}

#[test]
fn workflow_builder_clear_retry_config_resets_to_none() {
    use blazen_llm::retry::RetryConfig;
    let workflow = WorkflowBuilder::new("clear-retry-test")
        .step(make_step("a"))
        .retry_config(RetryConfig::default())
        .clear_retry_config()
        .build()
        .expect("builds");
    assert!(workflow.retry_config.is_none());
}

#[test]
fn step_retry_chain_builds_workflow() {
    use blazen_llm::retry::RetryConfig;
    let workflow = WorkflowBuilder::new("step-retry-test")
        .step(make_step("a"))
        .step_retry(RetryConfig {
            max_retries: 9,
            ..RetryConfig::default()
        })
        .build()
        .expect("builds");
    assert_eq!(workflow.step_names(), vec!["a".to_string()]);
}

#[test]
fn no_step_retry_chain_builds_workflow() {
    let workflow = WorkflowBuilder::new("no-step-retry-test")
        .step(make_step("a"))
        .no_step_retry()
        .build()
        .expect("builds");
    assert_eq!(workflow.step_names(), vec!["a".to_string()]);
}

#[test]
#[should_panic(expected = "step_retry() called before any step was registered")]
fn step_retry_without_prior_step_panics() {
    use blazen_llm::retry::RetryConfig;
    let _ = WorkflowBuilder::new("test").step_retry(RetryConfig::default());
}

#[test]
#[should_panic(expected = "no_step_retry() called before any step was registered")]
fn no_step_retry_without_prior_step_panics() {
    let _ = WorkflowBuilder::new("test").no_step_retry();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workflow_retry_propagates_to_step_context() {
    use blazen_core::Context;
    use blazen_llm::retry::RetryConfig;
    use std::sync::atomic::{AtomicU32, Ordering};

    static OBSERVED_MAX_RETRIES: AtomicU32 = AtomicU32::new(0);

    let handler: StepFn = Arc::new(|_ev, ctx: Context| {
        Box::pin(async move {
            let resolved = ctx.resolved_retry(None);
            OBSERVED_MAX_RETRIES.store(resolved.max_retries, Ordering::SeqCst);
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!("ok"),
            })))
        })
    });
    let step = StepRegistration::new(
        "observer".into(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        handler,
        0,
    );

    let workflow = WorkflowBuilder::new("retry-prop-test")
        .step(step)
        .retry_config(RetryConfig {
            max_retries: 9,
            ..RetryConfig::default()
        })
        .no_timeout()
        .build()
        .expect("workflow builds");

    let h = workflow
        .run(serde_json::json!({}))
        .await
        .expect("workflow starts");
    let _ = h.result().await;
    assert_eq!(OBSERVED_MAX_RETRIES.load(Ordering::SeqCst), 9);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workflow_usage_event_emission_via_context() {
    use blazen_core::Context;
    use blazen_events::{Modality, UsageEvent};
    use tokio_stream::StreamExt;
    use uuid::Uuid;

    let handler: StepFn = Arc::new(|_ev, ctx: Context| {
        Box::pin(async move {
            ctx.emit_usage(UsageEvent {
                provider: "t".into(),
                model: "m".into(),
                modality: Modality::Llm,
                prompt_tokens: 7,
                completion_tokens: 13,
                total_tokens: 20,
                reasoning_tokens: 0,
                cached_input_tokens: 0,
                audio_input_tokens: 0,
                audio_output_tokens: 0,
                image_count: 0,
                audio_seconds: 0.0,
                video_seconds: 0.0,
                cost_usd: Some(0.001),
                latency_ms: 5,
                run_id: Uuid::new_v4(),
            })
            .await;
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!("ok"),
            })))
        })
    });
    let step = StepRegistration::new(
        "emit_usage_step".into(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        handler,
        0,
    );
    let wf = WorkflowBuilder::new("usage_emit")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();
    let h = wf.run(serde_json::json!({})).await.unwrap();

    // Subscribe to the stream BEFORE awaiting the result so we capture
    // every event the workflow emits.
    let mut stream = h.stream_events();
    let collect_task = tokio::spawn(async move {
        let mut events: Vec<Box<dyn blazen_events::AnyEvent>> = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(2), stream.next()).await
        {
            events.push(event);
        }
        events
    });

    let _ = h.result().await.expect("workflow completes");
    let collected = collect_task.await.unwrap();

    let saw_usage = collected
        .iter()
        .any(|boxed| boxed.as_any().downcast_ref::<UsageEvent>().is_some());
    assert!(
        saw_usage,
        "expected at least one UsageEvent on the workflow stream"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workflow_handler_aggregates_usage_totals_and_cost() {
    use blazen_core::Context;
    use blazen_events::{Modality, UsageEvent};
    use uuid::Uuid;

    // Step that emits two UsageEvents with known token counts and costs
    // before terminating the workflow with a StopEvent.
    let handler: StepFn = Arc::new(|_ev, ctx: Context| {
        Box::pin(async move {
            ctx.emit_usage(UsageEvent {
                provider: "t".into(),
                model: "m".into(),
                modality: Modality::Llm,
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                reasoning_tokens: 0,
                cached_input_tokens: 0,
                audio_input_tokens: 0,
                audio_output_tokens: 0,
                image_count: 0,
                audio_seconds: 0.0,
                video_seconds: 0.0,
                cost_usd: Some(0.002),
                latency_ms: 1,
                run_id: Uuid::new_v4(),
            })
            .await;
            ctx.emit_usage(UsageEvent {
                provider: "t".into(),
                model: "m".into(),
                modality: Modality::Embedding,
                prompt_tokens: 4,
                completion_tokens: 0,
                total_tokens: 4,
                reasoning_tokens: 0,
                cached_input_tokens: 0,
                audio_input_tokens: 0,
                audio_output_tokens: 0,
                image_count: 0,
                audio_seconds: 0.0,
                video_seconds: 0.0,
                // Deliberately None to verify it contributes zero to the
                // running cost total.
                cost_usd: None,
                latency_ms: 1,
                run_id: Uuid::new_v4(),
            })
            .await;
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!("ok"),
            })))
        })
    });
    let step = StepRegistration::new(
        "emit_two_usage_events".into(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        handler,
        0,
    );
    let wf = WorkflowBuilder::new("usage_totals")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();
    let h = wf.run(serde_json::json!({})).await.unwrap();
    let result = h.result().await.expect("workflow completes");

    // Token totals: 10+4=14 prompt, 5+0=5 completion, 15+4=19 total.
    assert_eq!(result.usage_total.prompt_tokens, 14);
    assert_eq!(result.usage_total.completion_tokens, 5);
    assert_eq!(result.usage_total.total_tokens, 19);
    // Cost: only the first event reported a cost.
    assert!((result.cost_total_usd - 0.002).abs() < 1e-9);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn step_timeout_does_not_fire_when_handler_finishes_first() {
    let fast_handler: StepFn = Arc::new(|_ev, _ctx| {
        Box::pin(async {
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!("ok"),
            })))
        })
    });
    let mut step = StepRegistration::new(
        "fast".into(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        fast_handler,
        0,
    );
    step.timeout = Some(Duration::from_secs(5));

    let workflow = WorkflowBuilder::new("fast_test")
        .step(step)
        .no_timeout()
        .build()
        .expect("workflow builds");

    let handler = workflow
        .run(serde_json::json!({}))
        .await
        .expect("workflow starts");
    let final_event = handler
        .result()
        .await
        .expect("workflow should complete")
        .event;
    let terminal = final_event
        .downcast_ref::<StopEvent>()
        .expect("terminal event is StopEvent");
    assert_eq!(terminal.result, serde_json::json!("ok"));
}
