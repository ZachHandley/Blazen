//! Phase 6: cold-start recovery from a populated `AssignmentStore`.
//!
//! Simulate the scenario where a previous server process died with an
//! in-flight assignment claimed by some session that no longer exists.
//! The store carries:
//! - the `Assignment` itself,
//! - the in-flight claim under a stale session id,
//! - a Running snapshot.
//!
//! On `serve`, `recover_from_store()` should see the stale session
//! (not in the alive set), re-queue the assignment into the pending
//! FIFO, reset the snapshot to Pending, and let a freshly-connected
//! worker pick it up.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use blazen_controlplane::client::Client;
use blazen_controlplane::protocol::{Assignment, ENVELOPE_VERSION};
use blazen_controlplane::server::{AssignmentStore, ControlPlaneServer, MemoryAssignmentStore};
use blazen_controlplane::worker::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};

use blazen_core::distributed::{
    AdmissionMode, OrchestratorClient, RunStateSnapshot, RunStatus, WorkerCapability,
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
        Ok(serde_json::json!({ "recovered": true }))
    }
}

#[tokio::test]
async fn cold_start_recovers_orphaned_inflight() {
    let addr = ephemeral_addr().await;

    // Pre-populate the store as if a prior process died mid-run.
    let store = Arc::new(MemoryAssignmentStore::default());
    let run_id = Uuid::new_v4();
    let stale_session = Uuid::new_v4();

    let assignment = Assignment {
        envelope_version: ENVELOPE_VERSION,
        run_id,
        parent_run_id: None,
        workflow_name: "recovered".into(),
        workflow_version: None,
        input_json: b"{}".to_vec(),
        deadline_ms: None,
        attempt: 0,
        resource_hint: None,
    };

    store
        .put_assignment(run_id, &assignment)
        .await
        .expect("put_assignment");
    store
        .put_state(
            run_id,
            &RunStateSnapshot {
                run_id,
                status: RunStatus::Running,
                started_at_ms: 0,
                completed_at_ms: None,
                assigned_to: Some("dead-worker".into()),
                last_event_at_ms: None,
                output: None,
                error: None,
            },
            None,
        )
        .await
        .expect("put_state");
    store
        .claim_inflight(run_id, stale_session)
        .await
        .expect("claim_inflight");

    // Hand the populated store to a fresh server.
    let server = ControlPlaneServer::new("cp-recover").with_store(store.clone());
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Bring up a worker with the matching capability.
    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-recover")
        .with_capability(WorkerCapability {
            kind: "workflow:recovered".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).unwrap();
    let worker_handle = tokio::spawn(async move { worker.run(EchoHandler).await });

    // Poll until the recovered run reaches a terminal state.
    let client = Client::connect(format!("http://{addr}"), None)
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut final_status: Option<RunStateSnapshot> = None;
    while tokio::time::Instant::now() < deadline {
        match client.describe_workflow(run_id).await {
            Ok(snap)
                if matches!(
                    snap.status,
                    RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
                ) =>
            {
                final_status = Some(snap);
                break;
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let final_status = final_status.expect("recovered run reached terminal state");
    assert_eq!(final_status.status, RunStatus::Completed);
    let output = final_status.output.as_ref().expect("output present");
    assert_eq!(output, &serde_json::json!({ "recovered": true }));

    worker_handle.abort();
    server_handle.abort();
}
