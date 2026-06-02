//! Phase 3: `status.*` event emission on queue state transitions.
//!
//! Each test spins up a `ControlPlaneServer`, registers a worker with a
//! purpose-built `AssignmentHandler`, submits a workflow via the
//! orchestrator-side `Client`, subscribes to the resulting run, and
//! asserts the terminal `status.*` event arrives with the expected
//! shape. Subscription happens **after** submit, so `status.running`
//! may have already fired and be missed by the per-run subscriber —
//! the assertions therefore only require the terminal event, but
//! verify ordering when `status.running` is observed.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;

use blazen_controlplane::client::Client;
use blazen_controlplane::protocol::Assignment;
use blazen_controlplane::server::ControlPlaneServer;
use blazen_controlplane::worker::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};

use blazen_core::distributed::{
    AdmissionMode, OrchestratorClient, RunEvent, RunEventStream, SubmitWorkflowRequest,
    WorkerCapability,
};

mod harness;

/// Bind to an ephemeral port and return the resolved address.
async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

/// Brief handler-side delay so the test code has time to attach the
/// per-run subscription after `submit_workflow` returns but before the
/// handler emits its terminal `status.*` event. Without this, the
/// broadcast bus drops the terminal event before any subscriber exists.
const HANDLER_DELAY: Duration = Duration::from_millis(250);

/// Trivial completion handler — pauses briefly so the subscriber can
/// attach, then returns a constant JSON object.
struct EchoOkHandler;

#[async_trait]
impl AssignmentHandler for EchoOkHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        tokio::time::sleep(HANDLER_DELAY).await;
        Ok(serde_json::json!({ "ok": true }))
    }
}

/// Always-fail handler. Pauses briefly, then returns
/// `AssignmentFailure { error: "boom" }`.
struct FailingHandler;

#[async_trait]
impl AssignmentHandler for FailingHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        tokio::time::sleep(HANDLER_DELAY).await;
        Err(AssignmentFailure {
            error: "boom".into(),
        })
    }
}

/// Pull events from `stream` until `predicate` returns true or `timeout`
/// elapses, returning everything collected so far (including the event
/// that satisfied the predicate).
async fn collect_until<F>(
    stream: &mut RunEventStream<'_>,
    timeout: Duration,
    mut predicate: F,
) -> Vec<RunEvent>
where
    F: FnMut(&RunEvent) -> bool,
{
    let mut acc = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let Ok(Some(item)) = tokio::time::timeout(remaining, stream.next()).await else {
            break;
        };
        let Ok(event) = item else { continue };
        let done = predicate(&event);
        acc.push(event);
        if done {
            break;
        }
    }
    acc
}

/// `status.completed` is emitted when a handler returns `Ok(_)`. Carries
/// `{"output": <handler return value>}` in `data`. If `status.running`
/// is observed (the subscriber races against submit), it must precede
/// `status.completed`.
#[tokio::test]
async fn status_completed_event_arrives_in_order() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-complete")
        .with_capability(WorkerCapability {
            kind: "workflow:complete-evt".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).expect("validate config");
    let worker_handle = tokio::spawn(async move { worker.run(EchoOkHandler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .expect("client connect");

    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "complete-evt".into(),
            workflow_version: None,
            input: serde_json::json!({ "x": 1 }),
            required_tags: vec![],
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: true,
            resource_hint: None,
        })
        .await
        .expect("submit");
    let run_id = snap.run_id;

    let mut stream = client
        .subscribe_run_events(run_id)
        .await
        .expect("subscribe");

    let events = collect_until(&mut stream, Duration::from_secs(3), |e| {
        e.event_type == "status.completed"
    })
    .await;

    let completed_idx = events
        .iter()
        .position(|e| e.event_type == "status.completed")
        .unwrap_or_else(|| {
            panic!("expected status.completed within 3s, got {events:?}");
        });
    let completed = &events[completed_idx];
    assert_eq!(completed.run_id, run_id, "event run_id matches submit");

    let data_obj = completed
        .data
        .as_object()
        .expect("status.completed.data is an object");
    assert!(
        data_obj.contains_key("output"),
        "status.completed.data missing 'output' key: {data_obj:?}",
    );

    // Ordering: if status.running showed up, it must precede status.completed.
    if let Some(running_idx) = events.iter().position(|e| e.event_type == "status.running") {
        assert!(
            running_idx < completed_idx,
            "status.running (idx {running_idx}) must precede status.completed (idx {completed_idx}): {events:?}",
        );
    }

    worker_handle.abort();
    server_handle.abort();
}

/// `status.failed` is emitted when a handler returns `Err(AssignmentFailure)`.
/// Carries `{"error": "<message>"}` in `data`.
#[tokio::test]
async fn status_failed_event_arrives() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-fail")
        .with_capability(WorkerCapability {
            kind: "workflow:fail-evt".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).expect("validate config");
    let worker_handle = tokio::spawn(async move { worker.run(FailingHandler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .expect("client connect");

    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "fail-evt".into(),
            workflow_version: None,
            input: serde_json::json!({}),
            required_tags: vec![],
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: true,
            resource_hint: None,
        })
        .await
        .expect("submit");
    let run_id = snap.run_id;

    let mut stream = client
        .subscribe_run_events(run_id)
        .await
        .expect("subscribe");

    let events = collect_until(&mut stream, Duration::from_secs(3), |e| {
        e.event_type == "status.failed"
    })
    .await;

    let failed = events
        .iter()
        .find(|e| e.event_type == "status.failed")
        .unwrap_or_else(|| {
            panic!("expected status.failed within 3s, got {events:?}");
        });
    assert_eq!(failed.run_id, run_id, "event run_id matches submit");

    let data_obj = failed
        .data
        .as_object()
        .expect("status.failed.data is an object");
    let error_msg = data_obj
        .get("error")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| {
            panic!("status.failed.data missing 'error' string: {data_obj:?}");
        });
    assert!(
        error_msg.contains("boom"),
        "expected error message to contain 'boom', got {error_msg:?}",
    );

    worker_handle.abort();
    server_handle.abort();
}

/// `status.cancelled` is emitted when `Client::cancel_workflow` reaches
/// an in-flight assignment. Uses `harness::SlowHandler` so the run is
/// guaranteed to still be running when we cancel.
#[tokio::test]
async fn status_cancelled_event_arrives() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-cancel")
        .with_capability(WorkerCapability {
            kind: "workflow:cancel-evt".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).expect("validate config");
    let cancel_seen = Arc::new(AtomicBool::new(false));
    let handler = harness::SlowHandler {
        cancel_seen: cancel_seen.clone(),
    };
    let worker_handle = tokio::spawn(async move { worker.run(handler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .expect("client connect");

    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "cancel-evt".into(),
            workflow_version: None,
            input: serde_json::json!({}),
            required_tags: vec![],
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: true,
            resource_hint: None,
        })
        .await
        .expect("submit");
    let run_id = snap.run_id;

    let mut stream = client
        .subscribe_run_events(run_id)
        .await
        .expect("subscribe");

    // Let the worker pick up the assignment so the cancel hits in-flight state.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let _ = client
        .cancel_workflow(run_id)
        .await
        .expect("cancel_workflow");

    let events = collect_until(&mut stream, Duration::from_secs(3), |e| {
        e.event_type == "status.cancelled"
    })
    .await;

    let cancelled = events
        .iter()
        .find(|e| e.event_type == "status.cancelled")
        .unwrap_or_else(|| {
            panic!("expected status.cancelled within 3s, got {events:?}");
        });
    assert_eq!(cancelled.run_id, run_id, "event run_id matches submit");

    worker_handle.abort();
    server_handle.abort();
}
