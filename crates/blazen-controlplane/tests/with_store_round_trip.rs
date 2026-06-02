//! Phase 6: verify `ControlPlaneServer::with_store` plumbs an
//! injectable `AssignmentStore` end-to-end. Uses a `MemoryAssignmentStore`
//! constructed in test scope so we exercise the same code path Valkey
//! users will take, without requiring a Valkey instance.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use blazen_controlplane::client::Client;
use blazen_controlplane::protocol::Assignment;
use blazen_controlplane::server::{AssignmentStore, ControlPlaneServer, MemoryAssignmentStore};
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
async fn with_store_memory_round_trip() {
    let addr = ephemeral_addr().await;
    let store = Arc::new(MemoryAssignmentStore::default());
    let server = ControlPlaneServer::new("cp").with_store(store.clone());
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-x")
        .with_capability(WorkerCapability {
            kind: "workflow:echo".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).unwrap();
    let worker_handle = tokio::spawn(async move { worker.run(EchoHandler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .unwrap();
    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "echo".into(),
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

    // Poll until terminal.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut final_status = None;
    while tokio::time::Instant::now() < deadline {
        let s = client.describe_workflow(run_id).await.expect("describe");
        if matches!(
            s.status,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        ) {
            final_status = Some(s);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let final_status = final_status.expect("run reached terminal state");
    assert_eq!(final_status.status, RunStatus::Completed);

    // The store should now have a persisted RunStateSnapshot in
    // Completed state. The assignment record itself may or may not be
    // present — `mark_completed` deletes it via `store.delete_assignment`.
    let persisted_state = store
        .get_state(run_id)
        .await
        .expect("get_state ok")
        .expect("state should be persisted");
    assert!(
        matches!(persisted_state.status, RunStatus::Completed),
        "store state should reach Completed, got {:?}",
        persisted_state.status,
    );

    worker_handle.abort();
    server_handle.abort();
}
