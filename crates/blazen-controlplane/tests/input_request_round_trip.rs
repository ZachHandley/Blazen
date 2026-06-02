//! B2: server→worker input-request round-trip, end-to-end over gRPC.
//!
//! A worker handler pauses on `ctx.request_input(...)`. The orchestrator
//! observes the `input.request` event, answers via `Client::respond_to_input`,
//! and the handler resumes and completes with the answer as its output.
//! A second test exercises the `timeout_ms` path (no answer → the
//! handler's `request_input` errors and the run fails).

use std::net::SocketAddr;
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
    AdmissionMode, OrchestratorClient, RunStatus, SubmitWorkflowRequest, WorkerCapability,
};

async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

/// Handler that asks for input (after a short window so the orchestrator
/// can subscribe) and echoes the answer back as its output.
struct AskingHandler {
    timeout_ms: Option<u64>,
}

#[async_trait]
impl AssignmentHandler for AskingHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        // Give the orchestrator a window to attach a subscriber before
        // we emit `input.request` (the broadcast bus drops events with
        // no live subscribers).
        tokio::time::sleep(Duration::from_millis(150)).await;
        let answer = ctx
            .request_input(
                "approve?",
                serde_json::json!({ "kind": "approval" }),
                self.timeout_ms,
            )
            .await
            .map_err(|e| AssignmentFailure::new(format!("input failed: {e}")))?;
        Ok(serde_json::json!({ "answer": answer }))
    }
}

/// Poll `describe_workflow` until the run reaches a terminal state or the
/// deadline elapses.
async fn wait_terminal(client: &Client, run_id: uuid::Uuid) -> blazen_core::distributed::RunStatus {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let snap = client.describe_workflow(run_id).await.expect("describe");
        if !matches!(snap.status, RunStatus::Pending | RunStatus::Running) {
            return snap.status;
        }
        if tokio::time::Instant::now() >= deadline {
            return snap.status;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn request_input_resolves_after_respond() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-ask")
        .with_capability(WorkerCapability {
            kind: "workflow:ask".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).unwrap();
    let worker_handle = tokio::spawn(async move {
        worker
            .run(AskingHandler {
                timeout_ms: Some(5_000),
            })
            .await
    });
    tokio::time::sleep(Duration::from_millis(75)).await;

    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .unwrap();

    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "ask".into(),
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

    // Subscribe and wait for the `input.request` event to learn the
    // request_id, then answer it.
    let mut stream = client
        .subscribe_run_events(run_id)
        .await
        .expect("subscribe");
    let mut answered = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        let Ok(Some(item)) = tokio::time::timeout(Duration::from_millis(500), stream.next()).await
        else {
            continue;
        };
        let event = item.expect("event ok");
        if event.event_type == "input.request" {
            let request_id = event.data["request_id"]
                .as_str()
                .expect("request_id present")
                .to_string();
            client
                .respond_to_input(run_id, request_id, serde_json::json!({ "approved": true }))
                .await
                .expect("respond_to_input");
            answered = true;
            break;
        }
    }
    assert!(answered, "never observed an input.request event");

    let status = wait_terminal(&client, run_id).await;
    assert_eq!(status, RunStatus::Completed, "run should complete");
    let final_snap = client.describe_workflow(run_id).await.expect("describe");
    assert_eq!(
        final_snap.output,
        Some(serde_json::json!({ "answer": { "approved": true } })),
        "handler should echo the answer",
    );

    worker_handle.abort();
    server_handle.abort();
}

#[tokio::test]
async fn request_input_times_out_without_answer() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-timeout")
        .with_capability(WorkerCapability {
            kind: "workflow:timeout".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).unwrap();
    let worker_handle = tokio::spawn(async move {
        worker
            .run(AskingHandler {
                // Short timeout; the test never answers.
                timeout_ms: Some(200),
            })
            .await
    });
    tokio::time::sleep(Duration::from_millis(75)).await;

    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .unwrap();
    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "timeout".into(),
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

    let status = wait_terminal(&client, run_id).await;
    assert_eq!(
        status,
        RunStatus::Failed,
        "run should fail when input is never answered",
    );
    let final_snap = client.describe_workflow(run_id).await.expect("describe");
    assert!(
        final_snap
            .error
            .as_deref()
            .is_some_and(|e| e.contains("timed out")),
        "error should mention the timeout, got {:?}",
        final_snap.error,
    );

    worker_handle.abort();
    server_handle.abort();
}
