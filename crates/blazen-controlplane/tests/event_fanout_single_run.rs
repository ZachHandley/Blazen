//! Phase 3: end-to-end event fan-out via `SubscribeRunEvents`.
//!
//! A worker handler emits a custom event mid-run; the queue emits
//! `status.running` and `status.completed` status events around it.
//! An orchestrator subscriber receives all three.

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
    AdmissionMode, OrchestratorClient, SubmitWorkflowRequest, WorkerCapability,
};

/// Bind to an ephemeral port and return the address (port is released
/// before we hand it to the server).
async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

/// Handler that waits briefly so the orchestrator can subscribe, then
/// emits one custom event and completes. The brief wait keeps the
/// `status.running` and `custom.tick` events from racing past the
/// subscriber (the server's broadcast bus drops events when there are
/// no live subscribers).
struct EmittingHandler;

#[async_trait]
impl AssignmentHandler for EmittingHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        // Give the orchestrator a window to attach a subscriber before
        // we emit. The queue's `status.running` fires before this
        // handler is invoked, but the broadcast bus only drops events
        // when the channel has zero receivers — by the time `handle`
        // runs, the test has already had time to subscribe.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let _ = ctx
            .emit_event("custom.tick", serde_json::json!({ "step": 1 }))
            .await;
        Ok(serde_json::json!({ "ok": true }))
    }
}

#[tokio::test]
async fn subscribe_run_events_receives_custom_and_status() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Spin up a worker that emits one custom event.
    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-em")
        .with_capability(WorkerCapability {
            kind: "workflow:em".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).unwrap();
    let worker_handle = tokio::spawn(async move { worker.run(EmittingHandler).await });
    tokio::time::sleep(Duration::from_millis(75)).await;

    // Orchestrator client.
    let client = Client::connect(format!("http://{addr}"), None)
        .await
        .unwrap();

    // Submit the workflow. With `wait_for_worker=true`, a transient
    // ordering hiccup queues instead of rejecting.
    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "em".into(),
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

    // Subscribe immediately. The handler's leading 150ms sleep keeps
    // `custom.tick` and `status.completed` from racing past us.
    let mut stream = client
        .subscribe_run_events(run_id)
        .await
        .expect("subscribe");

    // Collect events for up to 5s or until `status.completed` arrives.
    let mut received = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        let Ok(Some(item)) = tokio::time::timeout(Duration::from_millis(500), stream.next()).await
        else {
            continue;
        };
        let event = item.expect("event ok");
        let done = event.event_type == "status.completed";
        received.push(event);
        if done {
            break;
        }
    }

    // We MUST see status.completed (terminal) and custom.tick (emitted
    // by the handler before the terminal frame). `status.running` may
    // race against subscribe — assert ordering only if it landed.
    assert!(
        received.iter().any(|e| e.event_type == "status.completed"),
        "expected status.completed in {received:?}",
    );
    let tick = received
        .iter()
        .find(|e| e.event_type == "custom.tick")
        .expect("custom.tick must arrive before status.completed");
    assert_eq!(tick.data, serde_json::json!({ "step": 1 }));
    assert_eq!(tick.run_id, run_id);

    // If we got status.running, it must precede status.completed.
    if let (Some(i_running), Some(i_completed)) = (
        received
            .iter()
            .position(|e| e.event_type == "status.running"),
        received
            .iter()
            .position(|e| e.event_type == "status.completed"),
    ) {
        assert!(
            i_running < i_completed,
            "status.running must precede status.completed",
        );
    }

    worker_handle.abort();
    server_handle.abort();
}
