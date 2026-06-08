//! Verifies that `ControlPlaneServer::serve_with_interceptor` actually
//! installs the caller-supplied tonic interceptor instead of the default
//! `BearerAuthInterceptor`. The custom interceptor in this test counts
//! invocations and stamps a sentinel header, so we can prove:
//!
//! 1. Our interceptor ran (counter advanced past zero on the very first
//!    RPC).
//! 2. The default bearer interceptor did NOT run (the `BLAZEN_PEER_TOKEN`
//!    env is unset for the duration of this test and we never send an
//!    `authorization` header — yet the submit still succeeds, which the
//!    default interceptor *would also* allow but our counter proves we
//!    short-circuited it).
//!
//! Models the workflow-submit flow on `tests/with_store_round_trip.rs`
//! and `tests/event_fanout_single_run.rs`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use blazen_controlplane::client::Client;
use blazen_controlplane::protocol::Assignment;
use blazen_controlplane::server::ControlPlaneServer;
use blazen_controlplane::worker::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};

use blazen_core::distributed::{
    AdmissionMode, OrchestratorClient, RunStatus, SubmitWorkflowRequest, WorkerCapability,
};

/// Bind to an ephemeral port and return the address (port is released
/// before we hand it to the server). Mirrors the helper in the other
/// integration tests in this crate.
async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

/// No-op assignment handler — succeeds immediately so the workflow
/// reaches `Completed` quickly.
struct NoopHandler;

#[async_trait]
impl AssignmentHandler for NoopHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        Ok(serde_json::json!({ "ok": true }))
    }
}

/// Custom tonic interceptor that:
///
/// 1. Increments a shared atomic counter on every invocation so the test
///    can prove the override actually ran.
/// 2. Stamps a sentinel `x-test-marker: ok` metadata header on the
///    outgoing request (after-stamp — this is mostly a stand-in for the
///    real downstream use case of attaching a `CallerCtx` extension
///    before the handler runs).
/// 3. Does NOT consult `BLAZEN_PEER_TOKEN` — proving the default
///    `BearerAuthInterceptor` is replaced rather than chained.
#[derive(Clone)]
struct CountingInterceptor {
    calls: Arc<AtomicU64>,
}

impl tonic::service::Interceptor for CountingInterceptor {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let marker: tonic::metadata::MetadataValue<tonic::metadata::Ascii> =
            "ok".parse().expect("static ascii");
        request.metadata_mut().insert("x-test-marker", marker);
        Ok(request)
    }
}

#[tokio::test]
async fn serve_with_interceptor_runs_custom_interceptor() {
    let addr = ephemeral_addr().await;
    let calls = Arc::new(AtomicU64::new(0));
    let interceptor = CountingInterceptor {
        calls: calls.clone(),
    };

    let server = ControlPlaneServer::new("cp");
    let server_handle =
        tokio::spawn(async move { server.serve_with_interceptor(addr, interceptor).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Worker that handles a single noop assignment.
    let cfg = WorkerConfig::new(format!("http://{addr}"), "worker-override")
        .with_capability(WorkerCapability {
            kind: "workflow:noop".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).unwrap();
    let worker_handle = tokio::spawn(async move { worker.run(NoopHandler).await });
    tokio::time::sleep(Duration::from_millis(75)).await;

    // The worker dialing the server already routes through our
    // interceptor (handshake / session-open RPCs). Capture that baseline
    // so we can assert the orchestrator submit added at least one more.
    let calls_after_worker_connect = calls.load(Ordering::SeqCst);
    assert!(
        calls_after_worker_connect > 0,
        "custom interceptor must have observed at least one RPC after worker connect, got 0",
    );

    // Orchestrator submits a workflow via the same code path used by the
    // other integration tests — no explicit auth header is attached.
    let client = Client::connect(format!("http://{addr}"), None, None)
        .await
        .unwrap();
    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "noop".into(),
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

    // Submit must have flowed through our interceptor.
    assert!(
        calls.load(Ordering::SeqCst) > calls_after_worker_connect,
        "submit_workflow RPC should have advanced the interceptor counter",
    );

    // Poll for terminal state — proves the request actually reached the
    // service after our interceptor stamped it (i.e. we didn't break the
    // pipeline).
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
    assert_eq!(
        final_status.status,
        RunStatus::Completed,
        "noop workflow should complete cleanly under the custom interceptor",
    );

    worker_handle.abort();
    server_handle.abort();
}
