//! Integration tests for `StepKind::SubWorkflow` and
//! `StepKind::ParallelSubWorkflows` (Wave 5).
//!
//! These cases construct a parent workflow whose steps are themselves
//! workflows, exercise per-step timeouts and retries on the child run,
//! and validate both `JoinStrategy::WaitAll` and
//! `JoinStrategy::FirstCompletes` semantics.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use blazen_core::{
    JoinStrategy, ParallelSubWorkflowsStep, StepFn, StepOutput, StepRegistration, SubWorkflowStep,
    Workflow, WorkflowBuilder, WorkflowError,
};
use blazen_events::{Event, StartEvent, StopEvent};
use blazen_llm::retry::RetryConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a child workflow that turns its `StartEvent` into a `StopEvent`
/// whose `result` JSON is the input data with `{"child": "<tag>"}` merged.
fn child_workflow_tag(tag: &'static str) -> Workflow {
    let handler: StepFn = Arc::new(move |event, _ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("expected StartEvent in child workflow");
            let mut data = start.data.clone();
            if let Some(obj) = data.as_object_mut() {
                obj.insert(
                    "child".to_owned(),
                    serde_json::Value::String(tag.to_owned()),
                );
            } else {
                data = serde_json::json!({ "input": data, "child": tag });
            }
            Ok(StepOutput::Single(Box::new(StopEvent { result: data })))
        })
    });

    WorkflowBuilder::new(format!("child::{tag}"))
        .step(StepRegistration::new(
            format!("child_step_{tag}"),
            vec![StartEvent::event_type()],
            vec![StopEvent::event_type()],
            handler,
            0,
        ))
        .no_timeout()
        .build()
        .expect("child workflow must build")
}

/// Build a child workflow that sleeps for `d` before producing a
/// `StopEvent`. Used to exercise timeouts and `FirstCompletes`.
fn child_workflow_sleeping(name: &'static str, d: Duration) -> Workflow {
    let handler: StepFn = Arc::new(move |_event, _ctx| {
        Box::pin(async move {
            tokio::time::sleep(d).await;
            let slept_ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: serde_json::json!({ "name": name, "slept_ms": slept_ms }),
            })))
        })
    });

    WorkflowBuilder::new(format!("child::sleep::{name}"))
        .step(StepRegistration::new(
            format!("child_sleep_{name}"),
            vec![StartEvent::event_type()],
            vec![StopEvent::event_type()],
            handler,
            0,
        ))
        .no_timeout()
        .build()
        .expect("sleeping child workflow must build")
}

/// Build a child workflow that fails the first `fails_before_success`
/// runs and succeeds afterwards. Used to verify retry behaviour.
///
/// The counter is shared across handler invocations.
fn child_workflow_flaky(
    name: &'static str,
    counter: Arc<AtomicUsize>,
    fails_before_success: usize,
) -> Workflow {
    let handler: StepFn = Arc::new(move |_event, _ctx| {
        let counter = Arc::clone(&counter);
        Box::pin(async move {
            let attempts_so_far = counter.fetch_add(1, Ordering::SeqCst);
            if attempts_so_far < fails_before_success {
                Err(WorkflowError::Context(format!(
                    "{name} attempt {attempts_so_far} failed"
                )))
            } else {
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!({
                        "name": name,
                        "attempts": attempts_so_far + 1,
                    }),
                })))
            }
        })
    });

    WorkflowBuilder::new(format!("child::flaky::{name}"))
        .step(StepRegistration::new(
            format!("child_flaky_{name}"),
            vec![StartEvent::event_type()],
            vec![StopEvent::event_type()],
            handler,
            0,
        ))
        .no_timeout()
        .no_retry()
        .build()
        .expect("flaky child workflow must build")
}

/// Default input mapper: forwards the parent `StartEvent` payload as-is.
fn passthrough_input_mapper() -> blazen_core::SubWorkflowInputMapper {
    Arc::new(|event| {
        if let Some(start) = event.as_any().downcast_ref::<StartEvent>() {
            start.data.clone()
        } else {
            event.to_json()
        }
    })
}

/// Default output mapper: wraps the child's terminal JSON in a
/// `StopEvent` so the parent workflow exits cleanly.
fn stop_event_output_mapper() -> blazen_core::SubWorkflowOutputMapper {
    Arc::new(|json| Box::new(StopEvent { result: json }))
}

// ---------------------------------------------------------------------------
// SubWorkflowStep: happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn subworkflow_step_runs_child_and_emits_output() {
    let child = child_workflow_tag("alpha");

    let sub = SubWorkflowStep {
        name: "outer_sub".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let parent = WorkflowBuilder::new("parent")
        .add_subworkflow_step(sub)
        .no_timeout()
        .build()
        .expect("parent workflow must build");

    let handler = parent
        .run(serde_json::json!({ "n": 7 }))
        .await
        .expect("parent run() must succeed");

    let result = handler.result().await.expect("workflow must complete");
    let stop = result
        .event
        .as_any()
        .downcast_ref::<StopEvent>()
        .expect("terminal event must be StopEvent");

    assert_eq!(
        stop.result,
        serde_json::json!({ "n": 7, "child": "alpha" }),
        "parent should emit the child's mapped output",
    );
}

// ---------------------------------------------------------------------------
// SubWorkflowStep: timeout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn subworkflow_step_respects_its_own_timeout() {
    // Child sleeps 500ms; per-step timeout is 50ms.
    let child = child_workflow_sleeping("slow", Duration::from_millis(500));

    let sub = SubWorkflowStep {
        name: "slow_sub".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: Some(Duration::from_millis(50)),
        retry_config: Some(Arc::new(RetryConfig {
            max_retries: 0,
            ..RetryConfig::default()
        })),
    };

    let parent = WorkflowBuilder::new("parent")
        .add_subworkflow_step(sub)
        .timeout(Duration::from_secs(5))
        .build()
        .expect("parent workflow must build");

    let handler = parent
        .run(serde_json::json!({}))
        .await
        .expect("parent run() must succeed");

    let result = handler.result().await;
    let err = result.expect_err("parent should fail when sub-workflow times out");
    match err {
        WorkflowError::SubWorkflowFailed { step_name, message } => {
            assert_eq!(step_name, "slow_sub");
            assert!(
                message.contains("slow_sub") && message.contains("timed out"),
                "message should describe the inner step timeout, got: {message}",
            );
        }
        other => panic!("expected SubWorkflowFailed, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// SubWorkflowStep: retry config
// ---------------------------------------------------------------------------

#[tokio::test]
async fn subworkflow_step_respects_its_own_retry_config() {
    let counter = Arc::new(AtomicUsize::new(0));
    // First two runs fail, third succeeds. With `max_retries = 2` the
    // sub-workflow makes 3 attempts in total and ultimately succeeds.
    let child = child_workflow_flaky("flaky", Arc::clone(&counter), 2);

    let sub = SubWorkflowStep {
        name: "flaky_sub".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: Some(Arc::new(RetryConfig {
            max_retries: 2,
            initial_delay_ms: 1,
            max_delay_ms: 1,
            jitter: false,
            ..RetryConfig::default()
        })),
    };

    let parent = WorkflowBuilder::new("parent")
        .add_subworkflow_step(sub)
        .timeout(Duration::from_secs(5))
        .build()
        .expect("parent workflow must build");

    let handler = parent
        .run(serde_json::json!({}))
        .await
        .expect("parent run() must succeed");

    let result = handler
        .result()
        .await
        .expect("retry should eventually succeed");
    let stop = result
        .event
        .as_any()
        .downcast_ref::<StopEvent>()
        .expect("terminal event must be StopEvent");
    assert_eq!(stop.result["name"], serde_json::json!("flaky"));
    assert_eq!(
        stop.result["attempts"],
        serde_json::json!(3),
        "third attempt must be the one that succeeded"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        3,
        "child handler should have been invoked exactly 3 times"
    );
}

// ---------------------------------------------------------------------------
// ParallelSubWorkflowsStep: WaitAll
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parallel_subworkflows_wait_all() {
    let branch_a = SubWorkflowStep {
        name: "branch_a".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child_workflow_tag("a")),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let branch_b = SubWorkflowStep {
        name: "branch_b".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child_workflow_tag("b")),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let fanout = ParallelSubWorkflowsStep {
        name: "fanout".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        branches: vec![branch_a, branch_b],
        join_strategy: JoinStrategy::WaitAll,
    };

    let parent = WorkflowBuilder::new("parent")
        .add_parallel_subworkflows(fanout)
        .timeout(Duration::from_secs(5))
        .build()
        .expect("parent workflow must build");

    let handler = parent
        .run(serde_json::json!({ "n": 1 }))
        .await
        .expect("parent run() must succeed");

    // Both branches emit StopEvents; the first one through the channel
    // terminates the loop. Verify the result is one of the expected
    // child outputs (we don't depend on branch ordering here).
    let result = handler.result().await.expect("workflow must complete");
    let stop = result
        .event
        .as_any()
        .downcast_ref::<StopEvent>()
        .expect("terminal event must be StopEvent");
    let child_tag = stop
        .result
        .get("child")
        .and_then(|v| v.as_str())
        .expect("child tag must be present");
    assert!(
        child_tag == "a" || child_tag == "b",
        "expected child tag from one of the branches, got {child_tag}",
    );
    assert_eq!(stop.result["n"], serde_json::json!(1));
}

// ---------------------------------------------------------------------------
// ParallelSubWorkflowsStep: FirstCompletes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parallel_subworkflows_first_completes_aborts_losers() {
    let fast = SubWorkflowStep {
        name: "fast".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child_workflow_sleeping("fast", Duration::from_millis(20))),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let slow = SubWorkflowStep {
        name: "slow".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        workflow: Arc::new(child_workflow_sleeping("slow", Duration::from_mins(1))),
        input_mapper: passthrough_input_mapper(),
        output_mapper: stop_event_output_mapper(),
        timeout: None,
        retry_config: None,
    };

    let fanout = ParallelSubWorkflowsStep {
        name: "race".to_owned(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        branches: vec![fast, slow],
        join_strategy: JoinStrategy::FirstCompletes,
    };

    let parent = WorkflowBuilder::new("parent")
        .add_parallel_subworkflows(fanout)
        .timeout(Duration::from_secs(5))
        .build()
        .expect("parent workflow must build");

    let start = std::time::Instant::now();
    let handler = parent
        .run(serde_json::json!({}))
        .await
        .expect("parent run() must succeed");
    let result = handler.result().await.expect("workflow must complete");
    let elapsed = start.elapsed();

    let stop = result
        .event
        .as_any()
        .downcast_ref::<StopEvent>()
        .expect("terminal event must be StopEvent");
    assert_eq!(
        stop.result["name"],
        serde_json::json!("fast"),
        "FirstCompletes should resolve with the fast branch's output"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "FirstCompletes must abort the slow branch instead of waiting on it (elapsed = {elapsed:?})"
    );
}
