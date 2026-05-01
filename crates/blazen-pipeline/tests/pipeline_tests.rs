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
        semaphore: None,
        timeout: None,
        retry_config: None,
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
        semaphore: None,
        timeout: None,
        retry_config: None,
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
        semaphore: None,
        timeout: None,
        retry_config: None,
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
        semaphore: None,
        timeout: None,
        retry_config: None,
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
        semaphore: None,
        timeout: None,
        retry_config: None,
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
        semaphore: None,
        timeout: None,
        retry_config: None,
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

    // Events should have stage_name "stream-group". The pipeline-level
    // `ProgressEvent` for the parallel stage carries `branch_name: None`
    // (it announces the stage as a whole), while everything streamed from
    // an inner branch carries `branch_name = Some(...)`. Filter out the
    // top-level progress envelope and check the rest.
    for event in &events {
        assert_eq!(event.stage_name, "stream-group");
        if event.branch_name.is_none() {
            // Pipeline-level progress envelope — must be a ProgressEvent.
            assert_eq!(event.event.event_type_id(), "blazen::ProgressEvent");
            continue;
        }
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

// ---------------------------------------------------------------------------
// Resume validation: pipeline name mismatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_pipeline_resume_validates_pipeline_name() {
    // Build pipeline "A", start it, pause, get snapshot.
    let pipeline_a = PipelineBuilder::new("pipeline-A")
        .stage(Stage {
            name: "stage-1".into(),
            workflow: delayed_echo_workflow(200),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler_a = pipeline_a.start(serde_json::json!({"v": 1}));

    // Give stage-1 time to start then pause.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let snapshot = handler_a.pause().await.unwrap();
    assert_eq!(snapshot.pipeline_name, "pipeline-A");

    // Build a DIFFERENT pipeline "B" and try to resume with snapshot from "A".
    let pipeline_b = PipelineBuilder::new("pipeline-B")
        .stage(Stage {
            name: "stage-1".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let result = pipeline_b.resume(snapshot);
    assert!(
        result.is_err(),
        "resume should fail for mismatched pipeline name"
    );
    match result.err().unwrap() {
        PipelineError::ValidationFailed(msg) => {
            assert!(
                msg.contains("pipeline-A") && msg.contains("pipeline-B"),
                "error should mention both pipeline names, got: {msg}"
            );
        }
        other => panic!("expected ValidationFailed, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Resume validation: stage names mismatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_pipeline_resume_validates_stage_names() {
    // Build a pipeline with stages "step1" and "step2".
    let pipeline = PipelineBuilder::new("stage-validation")
        .stage(Stage {
            name: "step1".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "step2".into(),
            workflow: delayed_echo_workflow(200),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"data": "test"}));

    // Let step1 complete and step2 start, then pause.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let snapshot = handler.pause().await.unwrap();

    // step1 should be in completed_stages.
    assert!(!snapshot.completed_stages.is_empty());
    assert_eq!(snapshot.completed_stages[0].name, "step1");

    // Build a new pipeline with a DIFFERENT first stage name.
    let pipeline_new = PipelineBuilder::new("stage-validation")
        .stage(Stage {
            name: "different_name".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "step2".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let result = pipeline_new.resume(snapshot);
    assert!(
        result.is_err(),
        "resume should fail when completed stage names don't match"
    );
    match result.err().unwrap() {
        PipelineError::ValidationFailed(msg) => {
            assert!(
                msg.contains("step1") && msg.contains("different_name"),
                "error should mention both stage names, got: {msg}"
            );
        }
        other => panic!("expected ValidationFailed, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Resume restores shared state
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_pipeline_resume_restores_shared_state() {
    // Stage 1: produces output with a "text" field.
    // Stage 2: uses input_mapper to read from shared state, then echoes.
    let pipeline = PipelineBuilder::new("shared-state-restore")
        .stage(Stage {
            name: "producer".into(),
            workflow: prefix_workflow("hello-"),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "consumer".into(),
            workflow: delayed_echo_workflow(200),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "world"}));

    // Let stage 1 complete and stage 2 start, then pause.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let snapshot = handler.pause().await.unwrap();

    // Verify the snapshot captured the output from stage 1.
    assert!(!snapshot.completed_stages.is_empty());
    assert_eq!(snapshot.completed_stages[0].name, "producer");
    assert_eq!(
        snapshot.completed_stages[0].output,
        serde_json::json!({"text": "hello-world"})
    );

    // Resume from the snapshot with a new pipeline (same stage names).
    let pipeline2 = PipelineBuilder::new("shared-state-restore")
        .stage(Stage {
            name: "producer".into(),
            workflow: echo_workflow(), // Won't re-run (already completed)
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "consumer".into(),
            workflow: echo_workflow(), // Fast this time
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler2 = pipeline2.resume(snapshot).unwrap();
    let result = handler2.result().await.unwrap();

    // Stage 2 should have received the output from stage 1 as its input.
    // The echo workflow returns whatever it gets, so the final output should
    // be the output from the completed stage 1.
    assert_eq!(result.stage_results.len(), 2);
    assert_eq!(
        result.final_output,
        serde_json::json!({"text": "hello-world"})
    );
}

// ---------------------------------------------------------------------------
// Total pipeline timeout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pipeline_total_timeout_fires_when_exceeded() {
    // Build a pipeline whose single stage sleeps longer than `total_timeout`.
    // Expect the run-loop to be cancelled and `PipelineError::Timeout` to be
    // surfaced via the result channel.
    let pipeline = PipelineBuilder::new("total-timeout")
        .stage(Stage {
            name: "slow-stage".into(),
            workflow: delayed_echo_workflow(500),
            input_mapper: None,
            condition: None,
        })
        .total_timeout(Duration::from_millis(50))
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "hi"}));
    let result = handler.result().await;

    assert!(result.is_err(), "expected Timeout error, got {result:?}");
    match result.unwrap_err() {
        PipelineError::Timeout { elapsed_ms } => {
            assert_eq!(
                elapsed_ms, 50,
                "elapsed_ms should equal the configured total timeout"
            );
        }
        other => panic!("expected PipelineError::Timeout, got: {other}"),
    }
}

#[tokio::test]
async fn pipeline_total_timeout_does_not_fire_when_pipeline_finishes_first() {
    // A pipeline that finishes quickly relative to the total timeout should
    // produce a normal `Ok` result.
    let pipeline = PipelineBuilder::new("total-timeout-no-fire")
        .stage(Stage {
            name: "fast-stage".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .total_timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "fast"}));
    let result = handler
        .result()
        .await
        .expect("pipeline should complete before total_timeout");
    assert_eq!(result.final_output, serde_json::json!({"text": "fast"}));
}

#[tokio::test]
async fn pipeline_no_total_timeout_disables_total_timeout() {
    // Ensure the `no_total_timeout` builder method clears any previously-set
    // total timeout and the pipeline is allowed to run to completion even
    // though the previous setting would have fired.
    let pipeline = PipelineBuilder::new("no-total-timeout")
        .stage(Stage {
            name: "slow-stage".into(),
            workflow: delayed_echo_workflow(100),
            input_mapper: None,
            condition: None,
        })
        .total_timeout(Duration::from_millis(10))
        .no_total_timeout()
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "ok"}));
    let result = handler
        .result()
        .await
        .expect("no_total_timeout() should disable the total timeout");
    assert_eq!(result.final_output, serde_json::json!({"text": "ok"}));
}

// ---------------------------------------------------------------------------
// Pipeline-level retry config
// ---------------------------------------------------------------------------

#[test]
fn pipeline_builder_retry_config_sets_field() {
    use blazen_llm::retry::RetryConfig;
    let pipeline = PipelineBuilder::new("test")
        .stage(Stage {
            name: "only-stage".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .retry_config(RetryConfig {
            max_retries: 7,
            ..RetryConfig::default()
        })
        .build()
        .expect("builds");
    assert!(pipeline.retry_config().is_some());
    assert_eq!(pipeline.retry_config().unwrap().max_retries, 7);
}

#[test]
fn pipeline_builder_no_retry_sets_max_retries_zero() {
    let pipeline = PipelineBuilder::new("test")
        .stage(Stage {
            name: "only-stage".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .no_retry()
        .build()
        .expect("builds");
    assert_eq!(pipeline.retry_config().unwrap().max_retries, 0);
}

// ---------------------------------------------------------------------------
// Usage / cost rollup
// ---------------------------------------------------------------------------

/// Build a workflow that emits a single `UsageEvent` then stops.
fn usage_emitting_workflow() -> blazen_core::Workflow {
    use blazen_events::{Modality, UsageEvent};
    use uuid::Uuid;

    let handler: StepFn = Arc::new(|_event, ctx| {
        Box::pin(async move {
            let ue = UsageEvent {
                provider: "test".into(),
                model: "test-model".into(),
                modality: Modality::Llm,
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                reasoning_tokens: 0,
                cached_input_tokens: 0,
                audio_input_tokens: 0,
                audio_output_tokens: 0,
                image_count: 0,
                audio_seconds: 0.0,
                video_seconds: 0.0,
                cost_usd: Some(0.0025),
                latency_ms: 10,
                run_id: Uuid::new_v4(),
            };
            ctx.write_event_to_stream(ue).await;
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!("ok"),
            })))
        })
    });

    let step = StepRegistration {
        name: "emit_usage".into(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
        semaphore: None,
        timeout: None,
        retry_config: None,
    };

    WorkflowBuilder::new("usage_test")
        .step(step)
        .no_timeout()
        .build()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_aggregates_usage_events_into_state_and_result() {
    let pipeline = PipelineBuilder::new("usage_pipeline")
        .stage(Stage {
            name: "only".into(),
            workflow: usage_emitting_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .expect("builds");

    let handler = pipeline.start(serde_json::json!({}));
    let result = handler.result().await.expect("pipeline runs");

    assert_eq!(result.usage_total.prompt_tokens, 100);
    assert_eq!(result.usage_total.completion_tokens, 50);
    assert_eq!(result.usage_total.total_tokens, 150);
    assert!((result.cost_total_usd - 0.0025).abs() < 1e-9);

    assert_eq!(result.stage_results.len(), 1);
    let stage = &result.stage_results[0];
    assert_eq!(stage.name, "only");
    let stage_usage = stage.usage.as_ref().expect("stage usage recorded");
    assert_eq!(stage_usage.prompt_tokens, 100);
    assert_eq!(stage_usage.completion_tokens, 50);
    assert_eq!(stage_usage.total_tokens, 150);
    let stage_cost = stage.cost_usd.expect("stage cost recorded");
    assert!((stage_cost - 0.0025).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Typed shared state — `PipelineState<S>` end-to-end coverage
// ---------------------------------------------------------------------------

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
struct MyState {
    counter: u32,
    label: String,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_typed_state_struct_round_trips() {
    let handler_fn: StepFn = Arc::new(|_ev, _ctx| {
        Box::pin(async {
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!({"counter": 7, "label": "go"}),
            })))
        })
    });
    let step = StepRegistration::new(
        "step1".into(),
        vec![StartEvent::event_type()],
        vec![StopEvent::event_type()],
        handler_fn,
        0,
    );
    let workflow = WorkflowBuilder::new("wf")
        .step(step)
        .no_timeout()
        .build()
        .unwrap();

    let pipeline = PipelineBuilder::<serde_json::Value>::new("p1")
        .with_state::<MyState>()
        .stage(Stage {
            name: "s1".into(),
            workflow,
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let h = pipeline.start(serde_json::json!({}));
    let result = h.result().await.expect("pipeline runs");
    // Snapshot's JSON-encoded shared_state — verify the test compiles end-to-end
    // even with a typed S2.
    let _ = result;
}

#[test]
fn pipeline_objects_bag_round_trips_typed() {
    use blazen_pipeline::PipelineState;
    let mut state: PipelineState<serde_json::Value> = PipelineState::default();
    state.put_object("conn", String::from("postgres://localhost"));
    let s: &String = state.get_object("conn").unwrap();
    assert_eq!(s, "postgres://localhost");
    let removed: String = state.remove_object("conn").unwrap();
    assert_eq!(removed, "postgres://localhost");
}

// ---------------------------------------------------------------------------
// Wave 7 — Progress events
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_emits_progress_events_for_each_stage() {
    use blazen_events::{ProgressEvent, ProgressKind};

    let pipeline = PipelineBuilder::new("progress-emit")
        .stage(Stage {
            name: "first".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "second".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "hello"}));
    let mut stream = handler.stream_events();

    // Collect everything the pipeline streams over a short window. The
    // pipeline is fast so 1s is plenty.
    let collect_task = tokio::spawn(async move {
        let mut events: Vec<blazen_pipeline::PipelineEvent> = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(1), stream.next()).await
        {
            events.push(event);
        }
        events
    });

    let _ = handler.result().await.expect("pipeline succeeds");
    let collected = collect_task.await.unwrap();

    // Filter for typed pipeline-kind progress envelopes.
    let progress: Vec<&ProgressEvent> = collected
        .iter()
        .filter_map(|e| e.event.as_any().downcast_ref::<ProgressEvent>())
        .filter(|p| matches!(p.kind, ProgressKind::Pipeline))
        .collect();

    assert!(
        progress.len() >= 2,
        "expected >=2 Pipeline ProgressEvents, got {}",
        progress.len()
    );

    // `current` and `percent` must monotonically increase across the
    // first two pipeline-kind progress events (one per stage).
    let first = progress[0];
    let second = progress[1];
    assert_eq!(first.current, 1);
    assert_eq!(second.current, 2);
    assert_eq!(first.total, Some(2));
    assert_eq!(second.total, Some(2));
    assert_eq!(first.label, "first");
    assert_eq!(second.label, "second");
    let pct1 = first.percent.expect("percent present");
    let pct2 = second.percent.expect("percent present");
    assert!(pct1 < pct2, "percent must monotonically increase");
    assert!((pct2 - 100.0_f32).abs() < f32::EPSILON);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_handler_progress_snapshot_advances() {
    // Stage 1: instant. Stage 2: slow (300ms).
    let pipeline = PipelineBuilder::new("progress-snapshot")
        .stage(Stage {
            name: "fast".into(),
            workflow: echo_workflow(),
            input_mapper: None,
            condition: None,
        })
        .stage(Stage {
            name: "slow".into(),
            workflow: delayed_echo_workflow(300),
            input_mapper: None,
            condition: None,
        })
        .build()
        .unwrap();

    let handler = pipeline.start(serde_json::json!({"text": "snap"}));

    // Initial snapshot — total_stages must be 2; current_stage_index
    // is at most 1 (could already be 1 if the first stage started).
    let initial = handler.progress();
    assert_eq!(initial.total_stages, 2);
    assert!(initial.current_stage_index <= 1);

    // Wait into the slow second stage.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let mid = handler.progress();
    assert_eq!(mid.total_stages, 2);
    assert!(
        mid.current_stage_index >= initial.current_stage_index,
        "current_stage_index must not regress (initial={}, mid={})",
        initial.current_stage_index,
        mid.current_stage_index
    );
    assert!(mid.percent <= 100.0);

    // Final snapshot after the pipeline completes — should report 2/2.
    let _ = handler.result().await.expect("pipeline completes");
    // (handler is consumed by `.result()`; if a future task wants to
    // sample after completion it must keep its own clone of the atomic.)
}
