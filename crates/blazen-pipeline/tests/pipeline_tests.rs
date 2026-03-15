//! Integration tests for the `blazen-pipeline` crate.

use std::sync::Arc;
use std::time::Duration;

use blazen_core::WorkflowBuilder;
use blazen_core::step::{StepFn, StepOutput, StepRegistration};
use blazen_events::{Event, StartEvent, StopEvent};
use blazen_pipeline::{JoinStrategy, ParallelStage, PipelineBuilder, PipelineError, Stage};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a workflow that echoes the `StartEvent` data as-is into a `StopEvent`.
fn echo_workflow() -> blazen_core::Workflow {
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

    let step = StepRegistration {
        name: "echo".into(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
    };

    WorkflowBuilder::new("echo")
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

/// Build a workflow that adds a prefix to the input's "text" field.
fn prefix_workflow(prefix: &'static str) -> blazen_core::Workflow {
    let handler: StepFn = Arc::new(move |event, _ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("expected StartEvent");
            let text = start
                .data
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let result = serde_json::json!({
                "text": format!("{prefix}{text}")
            });
            Ok(StepOutput::Single(Box::new(StopEvent { result })))
        })
    });

    let step = StepRegistration {
        name: format!("prefix-{prefix}"),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
    };

    WorkflowBuilder::new(format!("prefix-{prefix}"))
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

/// Build a workflow that adds a suffix to the input's "text" field.
fn suffix_workflow(suffix: &'static str) -> blazen_core::Workflow {
    let handler: StepFn = Arc::new(move |event, _ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("expected StartEvent");
            let text = start
                .data
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let result = serde_json::json!({
                "text": format!("{text}{suffix}")
            });
            Ok(StepOutput::Single(Box::new(StopEvent { result })))
        })
    });

    let step = StepRegistration {
        name: format!("suffix-{suffix}"),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
    };

    WorkflowBuilder::new(format!("suffix-{suffix}"))
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

/// Build a workflow that sleeps for `ms` milliseconds then echoes.
fn delayed_echo_workflow(ms: u64) -> blazen_core::Workflow {
    let handler: StepFn = Arc::new(move |event, _ctx| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(ms)).await;
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

    let step = StepRegistration {
        name: format!("delayed-echo-{ms}ms"),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
    };

    WorkflowBuilder::new(format!("delayed-echo-{ms}ms"))
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

/// Build a workflow that streams an event then completes.
fn streaming_echo_workflow() -> blazen_core::Workflow {
    let handler: StepFn = Arc::new(|event, ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("expected StartEvent");

            // Publish to the external stream.
            ctx.write_event_to_stream(StartEvent {
                data: serde_json::json!({ "streaming": true }),
            })
            .await;

            let stop = StopEvent {
                result: start.data.clone(),
            };
            Ok(StepOutput::Single(Box::new(stop)))
        })
    });

    let step = StepRegistration {
        name: "streaming-echo".into(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
    };

    WorkflowBuilder::new("streaming-echo")
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

/// Build a workflow that always fails.
fn failing_workflow() -> blazen_core::Workflow {
    let handler: StepFn = Arc::new(|_event, _ctx| {
        Box::pin(async move {
            Err(blazen_core::WorkflowError::Context(
                "intentional test failure".into(),
            ))
        })
    });

    let step = StepRegistration {
        name: "fail".into(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![],
        handler,
        max_concurrency: 0,
    };

    WorkflowBuilder::new("fail")
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_two_stage_sequential() {
    let pipeline = PipelineBuilder::new("two-stage")
        .stage(Stage {
            name: "add-prefix".into(),
            workflow: prefix_workflow("hello-"),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "add-suffix".into(),
            workflow: suffix_workflow("-world"),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "blazen"}));
    let result = handler.result().await.unwrap();

    assert_eq!(result.pipeline_name, "two-stage");
    assert_eq!(result.stage_results.len(), 2);
    assert_eq!(
        result.final_output,
        serde_json::json!({"text": "hello-blazen-world"})
    );
}

#[tokio::test]
async fn test_conditional_skip() {
    let pipeline = PipelineBuilder::new("conditional")
        .stage(Stage {
            name: "always-runs".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "skipped-stage".into(),
            workflow: prefix_workflow("SHOULD-NOT-SEE-"),
            input_mapper: None,
            condition: Some(Arc::new(|_state| false)),
        })
        .stage(Stage {
            name: "final".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "original"}));
    let result = handler.result().await.unwrap();

    assert_eq!(result.stage_results.len(), 3);

    // The skipped stage should be marked.
    let skipped = &result.stage_results[1];
    assert_eq!(skipped.name, "skipped-stage");
    assert!(skipped.skipped);
    assert_eq!(skipped.output, serde_json::Value::Null);

    // The final output should be from the first stage (skipped stage
    // didn't contribute, and the final stage echoed whatever was last).
    assert_eq!(result.final_output, serde_json::json!({"text": "original"}));
}

#[tokio::test]
async fn test_input_mapper() {
    let pipeline = PipelineBuilder::new("mapper")
        .stage(Stage {
            name: "produce".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "mapped".into(),
            workflow: prefix_workflow("mapped-"),
            input_mapper: Some(Arc::new(|state| {
                // Extract the "text" field from the previous result and
                // wrap it for the prefix workflow.
                let prev = state.last_result();
                let text = prev.get("nested").and_then(|v| v.get("value"));
                serde_json::json!({
                    "text": text.and_then(|v| v.as_str()).unwrap_or("default")
                })
            })),
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({
        "nested": { "value": "inner" }
    }));
    let result = handler.result().await.unwrap();

    assert_eq!(
        result.final_output,
        serde_json::json!({"text": "mapped-inner"})
    );
}

#[tokio::test]
async fn test_parallel_wait_all() {
    let pipeline = PipelineBuilder::new("parallel-wait-all")
        .parallel(ParallelStage {
            name: "parallel-group".into(),
            branches: vec![
                Stage {
                    name: "fast".into(),
                    workflow: delayed_echo_workflow(10),
                    input_mapper: Some(Arc::new(|_| serde_json::json!({"branch": "fast"}))),
                    condition: None,
                },
                Stage {
                    name: "slow".into(),
                    workflow: delayed_echo_workflow(50),
                    input_mapper: Some(Arc::new(|_| serde_json::json!({"branch": "slow"}))),
                    condition: None,
                },
            ],
            join_strategy: JoinStrategy::WaitAll,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!(null));
    let result = handler.result().await.unwrap();

    let output = &result.final_output;
    assert!(output.is_object());
    assert_eq!(output["fast"], serde_json::json!({"branch": "fast"}));
    assert_eq!(output["slow"], serde_json::json!({"branch": "slow"}));
}

#[tokio::test]
async fn test_parallel_first_completes() {
    let pipeline = PipelineBuilder::new("parallel-first")
        .parallel(ParallelStage {
            name: "race".into(),
            branches: vec![
                Stage {
                    name: "fast".into(),
                    workflow: delayed_echo_workflow(10),
                    input_mapper: Some(Arc::new(|_| serde_json::json!({"winner": "fast"}))),
                    condition: None,
                },
                Stage {
                    name: "slow".into(),
                    workflow: delayed_echo_workflow(2000),
                    input_mapper: Some(Arc::new(|_| serde_json::json!({"winner": "slow"}))),
                    condition: None,
                },
            ],
            join_strategy: JoinStrategy::FirstCompletes,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!(null));
    let result = handler.result().await.unwrap();

    let output = &result.final_output;
    assert!(output.is_object());
    // The fast branch should win.
    assert_eq!(output["fast"], serde_json::json!({"winner": "fast"}));
}

#[tokio::test]
async fn test_parallel_streaming() {
    let pipeline = PipelineBuilder::new("parallel-stream")
        .parallel(ParallelStage {
            name: "stream-group".into(),
            branches: vec![
                Stage {
                    name: "branch-a".into(),
                    workflow: streaming_echo_workflow(),
                    input_mapper: Some(Arc::new(|_| serde_json::json!({"from": "a"}))),
                    condition: None,
                },
                Stage {
                    name: "branch-b".into(),
                    workflow: streaming_echo_workflow(),
                    input_mapper: Some(Arc::new(|_| serde_json::json!({"from": "b"}))),
                    condition: None,
                },
            ],
            join_strategy: JoinStrategy::WaitAll,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!(null));

    // Subscribe to the event stream.
    let mut stream = handler.stream_events();

    // Collect events with a timeout.
    let mut events = Vec::new();
    let collect = async {
        while let Some(event) = stream.next().await {
            events.push(event);
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), collect).await;

    // Verify events arrived tagged with stage and branch names.
    let result = handler.result().await.unwrap();
    assert!(result.final_output.is_object());

    // Events should have stage_name "stream-group" and branch_name set.
    for event in &events {
        assert_eq!(event.stage_name, "stream-group");
        assert!(event.branch_name.is_some());
        let branch = event.branch_name.as_deref().unwrap();
        assert!(branch == "branch-a" || branch == "branch-b");
    }
}

#[tokio::test]
async fn test_pause_between_stages() {
    // Stage 1 is instant, stage 2 takes 2 seconds. We send pause after
    // 100ms -- by then stage 1 has completed and stage 2 is running.
    // The select! in the execution loop will catch the pause signal
    // during stage 2 execution.
    let pipeline = PipelineBuilder::new("pausable")
        .stage(Stage {
            name: "stage-1".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "stage-2".into(),
            workflow: delayed_echo_workflow(2000),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"value": 42}));

    // Give stage 1 time to complete, then pause during stage 2.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let snapshot = handler.pause().await.unwrap();

    assert_eq!(snapshot.pipeline_name, "pausable");
    // Stage 1 should have completed.
    assert!(!snapshot.completed_stages.is_empty());
    assert_eq!(snapshot.completed_stages[0].name, "stage-1");
    assert_eq!(snapshot.input, serde_json::json!({"value": 42}));
    // current_stage_index should point at stage 2 (index 1).
    assert_eq!(snapshot.current_stage_index, 1);

    // Resume with a fresh pipeline that has the same stages.
    let pipeline2 = PipelineBuilder::new("pausable")
        .stage(Stage {
            name: "stage-1".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "stage-2".into(),
            workflow: echo_workflow(), // Fast this time
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler2 = pipeline2.resume(snapshot).unwrap();
    let result = handler2.result().await.unwrap();
    assert_eq!(result.stage_results.len(), 2);
    assert_eq!(result.final_output, serde_json::json!({"value": 42}));
}

#[tokio::test]
async fn test_persist_json_callback() {
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    let persist_json = Arc::new(move |json: String| {
        let captured = captured_clone.clone();
        Box::pin(async move {
            captured.lock().await.push(json);
            Ok(())
        })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), PipelineError>> + Send>,
            >
    });

    let pipeline = PipelineBuilder::new("persist-json")
        .stage(Stage {
            name: "stage-a".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "stage-b".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .on_persist_json(persist_json)
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"test": true}));
    let _result = handler.result().await.unwrap();

    let snapshots = captured.lock().await;
    // Should have been called once per stage.
    assert_eq!(snapshots.len(), 2);

    // Each should be valid JSON.
    for json_str in snapshots.iter() {
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert!(parsed.get("pipeline_name").is_some());
        assert_eq!(parsed["pipeline_name"], "persist-json");
    }
}

#[tokio::test]
async fn test_persist_object_callback() {
    let captured: Arc<Mutex<Vec<blazen_pipeline::PipelineSnapshot>>> =
        Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    let persist_fn = Arc::new(move |snapshot: blazen_pipeline::PipelineSnapshot| {
        let captured = captured_clone.clone();
        Box::pin(async move {
            captured.lock().await.push(snapshot);
            Ok(())
        })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), PipelineError>> + Send>,
            >
    });

    let pipeline = PipelineBuilder::new("persist-object")
        .stage(Stage {
            name: "only-stage".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .on_persist(persist_fn)
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"data": "value"}));
    let _result = handler.result().await.unwrap();

    let snapshots = captured.lock().await;
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].pipeline_name, "persist-object");
    assert_eq!(snapshots[0].completed_stages.len(), 1);
    assert_eq!(snapshots[0].completed_stages[0].name, "only-stage");
}

#[tokio::test]
async fn test_no_state_noop() {
    // Pipeline without shared state, mappers, or conditions -- just plain
    // workflows chained together.
    let pipeline = PipelineBuilder::new("noop")
        .stage(Stage {
            name: "first".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!(42));
    let result = handler.result().await.unwrap();

    assert_eq!(result.final_output, serde_json::json!(42));
    assert_eq!(result.stage_results.len(), 1);
    assert!(!result.stage_results[0].skipped);
}

#[tokio::test]
async fn test_stage_failure_propagates() {
    let pipeline = PipelineBuilder::new("failure")
        .stage(Stage {
            name: "good-stage".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "bad-stage".into(),
            workflow: failing_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!(null));
    let result = handler.result().await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        PipelineError::StageFailed { stage_name, .. } => {
            assert_eq!(stage_name, "bad-stage");
        }
        other => panic!("expected StageFailed, got: {other}"),
    }
}

#[tokio::test]
async fn test_empty_pipeline_fails_validation() {
    let result = PipelineBuilder::new("empty").build();
    assert!(result.is_err());
    match result.unwrap_err() {
        PipelineError::ValidationFailed(msg) => {
            assert!(
                msg.contains("at least one stage"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected ValidationFailed, got: {other}"),
    }
}
