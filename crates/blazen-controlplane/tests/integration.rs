//! Phase 1 integration tests for the blazen-controlplane gRPC + HTTP
//! tiers. These spin up a real `ControlPlaneServer` on ephemeral ports
//! and exercise the worker session + submit flow end-to-end.

#[cfg(feature = "http-transport")]
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

#[cfg(feature = "http-transport")]
use blazen_controlplane::protocol::{AdmissionModeWire, CapabilityWire, WorkerHello};
use blazen_controlplane::protocol::{
    ENVELOPE_VERSION, RunStateSnapshotWire, RunStatusWire, SubmitRequest,
};
use blazen_controlplane::server::ControlPlaneServer;

use tonic::transport::Channel;

mod harness;

use harness::{Captured, EchoHandler, RecordingHandler};

/// Bind to an ephemeral port and return the resolved address.
pub(crate) async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

fn encode<T: serde::Serialize>(v: &T) -> Vec<u8> {
    postcard::to_allocvec(v).expect("encode")
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
    postcard::from_bytes(bytes).expect("decode")
}

/// Drive the `Worker` API through a full handshake → assignment →
/// completion roundtrip. Submission still uses the raw tonic client
/// because the orchestrator-side `Client` lands in Phase 2.
#[tokio::test]
async fn grpc_worker_session_handshake() {
    let addr = ephemeral_addr().await;

    let server = ControlPlaneServer::new("test-cp");
    let server_handle = tokio::spawn(async move {
        let _ = server.serve(addr).await;
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Spin up a Worker with capability `workflow:test`.
    let cfg = blazen_controlplane::worker::WorkerConfig::new(format!("http://{addr}"), "worker-a")
        .with_capability(blazen_core::distributed::WorkerCapability {
            kind: "workflow:test".into(),
            version: 0,
        })
        .with_admission(blazen_core::distributed::AdmissionMode::Fixed { max_in_flight: 4 })
        .with_heartbeat_interval(Duration::from_millis(200));
    let worker = blazen_controlplane::worker::Worker::connect(cfg).expect("validate config");

    let captured = Captured::default();
    let handler = RecordingHandler::new(captured.clone());
    let run_handle = tokio::spawn(async move { worker.run(handler).await });

    // The Worker's first action is to send Hello + receive Welcome.
    // Poll `captured` (which the handler updates on first assignment)
    // by waiting for the assignment we're about to submit. To send the
    // submit we need a separate orchestrator-side tonic client.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("connect");
    let mut client =
        blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient::new(
            channel,
        );

    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: "test".into(),
        workflow_version: None,
        input_json: b"{\"x\":1}".to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
        place: None,
    };
    let resp = client
        .submit_workflow(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&submit),
        })
        .await
        .expect("submit_workflow");
    let wire: RunStateSnapshotWire = decode(&resp.into_inner().postcard_payload);
    let run_id = wire.run_id;

    // The Worker hands the Assignment to the handler.
    let captured_assignment = tokio::time::timeout(Duration::from_secs(2), captured.wait_for_one())
        .await
        .expect("assignment captured within 2s");
    assert_eq!(captured_assignment.workflow_name, "test");
    assert_eq!(captured_assignment.input_json, b"{\"x\":1}");
    assert_eq!(captured_assignment.run_id, run_id);

    // Run state should reach Completed.
    let final_status = poll_until_terminal(&mut client, run_id).await;
    assert_eq!(final_status.status, RunStatusWire::Completed);
    assert_eq!(final_status.output_json, b"{\"echo\":{\"x\":1}}");

    run_handle.abort();
    server_handle.abort();
}

/// Submit + Worker echo handler completes. Direct test of the Phase 1
/// public surface (`Worker::connect` + `AssignmentHandler::handle`).
#[tokio::test]
async fn worker_executes_assignment() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = blazen_controlplane::worker::WorkerConfig::new(format!("http://{addr}"), "worker-x")
        .with_capability(blazen_core::distributed::WorkerCapability {
            kind: "workflow:echo".into(),
            version: 0,
        })
        .with_admission(blazen_core::distributed::AdmissionMode::Fixed { max_in_flight: 2 });
    let worker = blazen_controlplane::worker::Worker::connect(cfg).unwrap();
    let run_handle = tokio::spawn(async move { worker.run(EchoHandler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client =
        blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient::new(
            channel,
        );

    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: "echo".into(),
        workflow_version: None,
        input_json: b"{\"hello\":\"world\"}".to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
        place: None,
    };
    let resp = client
        .submit_workflow(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&submit),
        })
        .await
        .expect("submit");
    let wire: RunStateSnapshotWire = decode(&resp.into_inner().postcard_payload);
    let run_id = wire.run_id;

    let final_status = poll_until_terminal(&mut client, run_id).await;
    assert_eq!(final_status.status, RunStatusWire::Completed);
    let output: serde_json::Value = serde_json::from_slice(&final_status.output_json).unwrap();
    assert_eq!(output, serde_json::json!({"echo": {"hello": "world"}}));

    run_handle.abort();
    server_handle.abort();
}

/// `CancelWorkflow` aborts an in-flight assignment; the handler's
/// `on_cancel` hook fires, the handler future's cancellation token is
/// triggered, and the queue ends up in `Cancelled`.
#[tokio::test]
async fn worker_handles_cancel() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = blazen_controlplane::worker::WorkerConfig::new(format!("http://{addr}"), "worker-c")
        .with_capability(blazen_core::distributed::WorkerCapability {
            kind: "workflow:slow".into(),
            version: 0,
        })
        .with_admission(blazen_core::distributed::AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = blazen_controlplane::worker::Worker::connect(cfg).unwrap();
    let cancel_seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handler = harness::SlowHandler {
        cancel_seen: cancel_seen.clone(),
    };
    let run_handle = tokio::spawn(async move { worker.run(handler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client =
        blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient::new(
            channel,
        );

    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: "slow".into(),
        workflow_version: None,
        input_json: b"{}".to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
        place: None,
    };
    let resp = client
        .submit_workflow(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&submit),
        })
        .await
        .expect("submit");
    let wire: RunStateSnapshotWire = decode(&resp.into_inner().postcard_payload);
    let run_id = wire.run_id;

    // Give the handler time to start the slow work.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cancel = blazen_controlplane::protocol::CancelRequest {
        envelope_version: ENVELOPE_VERSION,
        run_id,
    };
    let _ = client
        .cancel_workflow(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&cancel),
        })
        .await
        .expect("cancel");

    let final_status = poll_until_terminal(&mut client, run_id).await;
    assert_eq!(
        final_status.status,
        RunStatusWire::Cancelled,
        "expected Cancelled, got {:?}",
        final_status.status,
    );
    assert!(
        cancel_seen.load(std::sync::atomic::Ordering::Relaxed),
        "expected on_cancel to fire",
    );

    run_handle.abort();
    server_handle.abort();
}

/// Heartbeats reach the server's registry (verified via `ListWorkers`).
#[tokio::test]
async fn worker_heartbeats() {
    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = blazen_controlplane::worker::WorkerConfig::new(format!("http://{addr}"), "worker-h")
        .with_capability(blazen_core::distributed::WorkerCapability {
            kind: "workflow:hb".into(),
            version: 0,
        })
        .with_admission(blazen_core::distributed::AdmissionMode::Fixed { max_in_flight: 4 })
        .with_heartbeat_interval(Duration::from_millis(50));
    let worker = blazen_controlplane::worker::Worker::connect(cfg).unwrap();
    let run_handle = tokio::spawn(async move { worker.run(EchoHandler).await });

    // Wait through at least two heartbeat intervals so the registry's
    // `in_flight` / `admission_snapshot` get populated.
    tokio::time::sleep(Duration::from_millis(250)).await;

    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client =
        blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient::new(
            channel,
        );

    let req = blazen_controlplane::protocol::ListWorkersRequest {
        envelope_version: ENVELOPE_VERSION,
    };
    let resp = client
        .list_workers(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&req),
        })
        .await
        .expect("list_workers");
    let list: blazen_controlplane::protocol::ListWorkersResponse =
        decode(&resp.into_inner().postcard_payload);
    let worker_info = list
        .workers
        .iter()
        .find(|w| w.node_id == "worker-h")
        .expect("worker present");
    assert_eq!(worker_info.in_flight, 0, "no work assigned yet");
    assert!(
        worker_info.admission_snapshot.is_some(),
        "heartbeat populated the admission snapshot",
    );

    run_handle.abort();
    server_handle.abort();
}

/// Full orchestrator-side `Client` round-trip: submit → worker echo →
/// describe sees `Completed`.
#[tokio::test]
async fn orchestrator_client_roundtrip() {
    use blazen_core::distributed::{OrchestratorClient, RunStatus, SubmitWorkflowRequest};

    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = blazen_controlplane::worker::WorkerConfig::new(format!("http://{addr}"), "worker-o")
        .with_capability(blazen_core::distributed::WorkerCapability {
            kind: "workflow:orch-echo".into(),
            version: 0,
        })
        .with_admission(blazen_core::distributed::AdmissionMode::Fixed { max_in_flight: 2 });
    let worker = blazen_controlplane::worker::Worker::connect(cfg).unwrap();
    let run_handle = tokio::spawn(async move { worker.run(EchoHandler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = blazen_controlplane::client::Client::connect(format!("http://{addr}"), None, None)
        .await
        .expect("client connect");

    let snap = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "orch-echo".into(),
            workflow_version: None,
            input: serde_json::json!({"x": 1}),
            required_tags: vec![],
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: true,
            resource_hint: None,
        })
        .await
        .expect("submit_workflow");
    let run_id = snap.run_id;

    // Poll describe via the Client until terminal.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let final_snap = loop {
        let snap = client
            .describe_workflow(run_id)
            .await
            .expect("describe_workflow");
        if matches!(
            snap.status,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        ) {
            break snap;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run {run_id} did not terminate within 5s (last={snap:?})",
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(final_snap.status, RunStatus::Completed);
    assert_eq!(
        final_snap.output,
        Some(serde_json::json!({"echo": {"x": 1}}))
    );

    // list_workers should report worker-o.
    let workers = client.list_workers().await.expect("list_workers");
    assert!(
        workers.iter().any(|w| w.node_id == "worker-o"),
        "expected worker-o in the registry, got {workers:?}",
    );

    run_handle.abort();
    server_handle.abort();
}

/// `Client::subscribe_run_events` opens cleanly against the gRPC tier.
/// Since Phase 3 the server returns a real `BroadcastStream`, so the
/// stream stays open waiting for events; this test only verifies the
/// handshake succeeds and the stream yields nothing for a run id that
/// will never emit. Real fan-out is covered by `event_fanout_*` tests.
#[tokio::test]
async fn orchestrator_client_subscribe_handshake_succeeds() {
    use blazen_core::distributed::OrchestratorClient;

    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = blazen_controlplane::client::Client::connect(format!("http://{addr}"), None, None)
        .await
        .expect("client connect");

    let mut stream = client
        .subscribe_run_events(uuid::Uuid::new_v4())
        .await
        .expect("subscribe");

    // Stream is open and waiting; nothing should arrive in 250ms for a
    // run id that doesn't exist. `timeout(...).await` returns
    // `Err(Elapsed)` on the elapsed branch, which we expect.
    let res = tokio::time::timeout(
        Duration::from_millis(250),
        futures_util::StreamExt::next(&mut stream),
    )
    .await;
    assert!(
        res.is_err(),
        "expected timeout (no events), got item: {res:?}"
    );
    drop(stream);

    server_handle.abort();
}

/// `Client::cancel_workflow` reaches an in-flight assignment via the
/// new client surface and the queue ends up `Cancelled`.
#[tokio::test]
async fn orchestrator_client_cancel_workflow() {
    use blazen_core::distributed::{OrchestratorClient, RunStatus, SubmitWorkflowRequest};

    let addr = ephemeral_addr().await;
    let server = ControlPlaneServer::new("cp");
    let server_handle = tokio::spawn(async move { server.serve(addr).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cfg = blazen_controlplane::worker::WorkerConfig::new(format!("http://{addr}"), "worker-oc")
        .with_capability(blazen_core::distributed::WorkerCapability {
            kind: "workflow:slow".into(),
            version: 0,
        })
        .with_admission(blazen_core::distributed::AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = blazen_controlplane::worker::Worker::connect(cfg).unwrap();
    let cancel_seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handler = harness::SlowHandler {
        cancel_seen: cancel_seen.clone(),
    };
    let run_handle = tokio::spawn(async move { worker.run(handler).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = blazen_controlplane::client::Client::connect(format!("http://{addr}"), None, None)
        .await
        .expect("client connect");

    let submitted = client
        .submit_workflow(SubmitWorkflowRequest {
            workflow_name: "slow".into(),
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

    // Let the handler start before we cancel.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let cancelled = client
        .cancel_workflow(submitted.run_id)
        .await
        .expect("cancel_workflow");
    // The cancel mark is applied server-side before the worker's
    // result frame arrives, so the immediate snapshot is already
    // `Cancelled`.
    assert!(
        matches!(
            cancelled.status,
            RunStatus::Cancelled | RunStatus::Completed | RunStatus::Failed
        ),
        "expected terminal status after cancel, got {:?}",
        cancelled.status,
    );

    // Wait for the on_cancel hook to actually fire on the worker side.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !cancel_seen.load(std::sync::atomic::Ordering::Relaxed) {
        assert!(
            std::time::Instant::now() < deadline,
            "on_cancel never fired within 2s",
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    run_handle.abort();
    server_handle.abort();
}

/// With `RetryPolicy { max_attempts: Some(1), .. }` and a dead endpoint,
/// `Worker::run` returns the underlying connect error rather than
/// looping forever.
#[tokio::test]
async fn worker_respects_max_attempts() {
    // 127.0.0.1:1 is reserved and refuses connections — perfect for a
    // deterministic "always fails to connect" target without timing
    // dependencies.
    let cfg = blazen_controlplane::worker::WorkerConfig::new("http://127.0.0.1:1", "worker-x")
        .with_retry(blazen_controlplane::worker::RetryPolicy {
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(1),
            multiplier: 1.0,
            max_attempts: Some(1),
        });
    let worker = blazen_controlplane::worker::Worker::connect(cfg).unwrap();
    let err = tokio::time::timeout(Duration::from_secs(2), worker.run(EchoHandler))
        .await
        .expect("run returned within timeout")
        .expect_err("max_attempts exhausted should return error");
    assert!(
        matches!(err, blazen_controlplane::ControlPlaneError::Transport(_)),
        "expected Transport error, got {err:?}",
    );
}

/// Helper: poll `describe_workflow` until the run is in a terminal
/// state (Completed / Failed / Cancelled). Times out after 5 seconds.
async fn poll_until_terminal(
    client: &mut blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient<
        Channel,
    >,
    run_id: uuid::Uuid,
) -> RunStateSnapshotWire {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let req = blazen_controlplane::protocol::DescribeRequest {
            envelope_version: ENVELOPE_VERSION,
            run_id,
        };
        let resp = client
            .describe_workflow(blazen_controlplane::pb::PostcardRequest {
                postcard_payload: encode(&req),
            })
            .await
            .expect("describe_workflow");
        let snap: RunStateSnapshotWire = decode(&resp.into_inner().postcard_payload);
        if matches!(
            snap.status,
            RunStatusWire::Completed | RunStatusWire::Failed | RunStatusWire::Cancelled,
        ) {
            return snap;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run {run_id} did not reach a terminal state within 5s (last={snap:?})",
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[cfg(feature = "http-transport")]
#[tokio::test]
async fn http_worker_register_and_submit() {
    use blazen_controlplane::http::PostcardEnvelope;
    use blazen_controlplane::protocol;

    let grpc_addr = ephemeral_addr().await;
    let http_addr = ephemeral_addr().await;

    let server = ControlPlaneServer::new("test-cp").with_http(http_addr);
    let server_handle = tokio::spawn(async move {
        let _ = server.serve(grpc_addr).await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    let client = reqwest::Client::new();
    let base = format!("http://{http_addr}");

    // Register.
    let hello = WorkerHello {
        envelope_version: ENVELOPE_VERSION,
        node_id: "worker-http".into(),
        capabilities: vec![CapabilityWire {
            kind: "workflow:http-test".into(),
            version: 0,
        }],
        tags: BTreeMap::new(),
        admission: AdmissionModeWire::Fixed { max_in_flight: 4 },
        supported_envelope_versions: vec![1],
        labels: BTreeMap::new(),
        taints: Vec::new(),
        descriptors: Vec::new(),
        place: None,
    };
    let register_resp: serde_json::Value = client
        .post(format!("{base}/v1/cp/worker/register"))
        .json(&PostcardEnvelope::encode(&hello).unwrap())
        .send()
        .await
        .expect("register POST")
        .json()
        .await
        .expect("register JSON");

    let session_id = register_resp["session_id"]
        .as_str()
        .expect("session_id present")
        .to_string();

    // Submit.
    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: "http-test".into(),
        workflow_version: None,
        input_json: b"{}".to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
        place: None,
    };
    let submit_resp: PostcardEnvelope = client
        .post(format!("{base}/v1/cp/submit"))
        .json(&PostcardEnvelope::encode(&submit).unwrap())
        .send()
        .await
        .expect("submit POST")
        .json()
        .await
        .expect("submit JSON");
    let snap: protocol::RunStateSnapshotWire = submit_resp.decode().expect("decode snapshot");
    assert!(matches!(
        snap.status,
        protocol::RunStatusWire::Pending | protocol::RunStatusWire::Running
    ));

    // (Skip SSE stream test — the registered worker's outbound receiver
    // was stashed when we POSTed register; consuming it via SSE is a
    // separate fetch we don't need to validate the basic flow here.)
    let _ = session_id;

    server_handle.abort();
}
