//! Phase 3: `subscribe_all` with `required_tags` filter routes events
//! to only the subscribers whose predicate the run's worker satisfies.
//!
//! Two workers share the same `workflow:demo` capability but differ on
//! the `color` tag (`red` vs `blue`). A single orchestrator subscribes
//! to `subscribe_all` with `required_tags = ["color=red"]`, then submits
//! one workflow targeted at each worker via `required_tags` on the
//! submit request. The subscriber must see the red run reach
//! `status.completed` and must never see any event for the blue run.

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
    AdmissionMode, OrchestratorClient, RunEvent, SubmitWorkflowRequest, WorkerCapability,
};

async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

struct EchoHandler;

#[async_trait]
impl AssignmentHandler for EchoHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        Ok(serde_json::json!({ "ok": true }))
    }
}

#[tokio::test]
async fn subscribe_all_filters_by_required_tags() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Two workers — same capability, different `color` tag.
    let cfg_red = WorkerConfig::new(format!("http://{addr}"), "worker-red")
        .with_capability(WorkerCapability {
            kind: "workflow:demo".into(),
            version: 0,
        })
        .with_tag("color", "red")
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker_red = Worker::connect(cfg_red).unwrap();
    let red_handle = tokio::spawn(async move { worker_red.run(EchoHandler).await });

    let cfg_blue = WorkerConfig::new(format!("http://{addr}"), "worker-blue")
        .with_capability(WorkerCapability {
            kind: "workflow:demo".into(),
            version: 0,
        })
        .with_tag("color", "blue")
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker_blue = Worker::connect(cfg_blue).unwrap();
    let blue_handle = tokio::spawn(async move { worker_blue.run(EchoHandler).await });

    // Give both workers time to register before submitting.
    tokio::time::sleep(Duration::from_millis(150)).await;

    let client = Client::connect(format!("http://{addr}"), None)
        .await
        .unwrap();

    // Subscribe BEFORE submitting so we don't race the broadcast.
    let mut stream = client
        .subscribe_all(vec!["color=red".to_string()])
        .await
        .expect("subscribe_all");

    // Submit one run for the red worker and one for the blue worker.
    // Admission routes each based on the `color` tag predicate.
    let red_snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "demo".into(),
            workflow_version: None,
            input: serde_json::json!({}),
            required_tags: vec!["color=red".into()],
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: true,
            resource_hint: None,
        })
        .await
        .expect("submit red");
    let blue_snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "demo".into(),
            workflow_version: None,
            input: serde_json::json!({}),
            required_tags: vec!["color=blue".into()],
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: true,
            resource_hint: None,
        })
        .await
        .expect("submit blue");

    let red_id = red_snap.run_id;
    let blue_id = blue_snap.run_id;

    // Collect events for up to 3 seconds, stopping once the red run
    // reports `status.completed`.
    let mut seen: Vec<RunEvent> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let Ok(Some(item)) = tokio::time::timeout(remaining, stream.next()).await else {
            break;
        };
        if let Ok(event) = item {
            let done = event.run_id == red_id && event.event_type == "status.completed";
            seen.push(event);
            if done {
                break;
            }
        }
    }

    // Red must show up; blue must NOT.
    assert!(
        seen.iter()
            .any(|e| e.run_id == red_id && e.event_type == "status.completed"),
        "red run completion should appear in {seen:?}",
    );
    assert!(
        !seen.iter().any(|e| e.run_id == blue_id),
        "blue run events must be filtered out — saw {seen:?}",
    );

    red_handle.abort();
    blue_handle.abort();
    server_handle.abort();
}
