//! Phase 3: SSE event fan-out via the HTTP `/v1/cp/events/{run_id}` route.
//!
//! Mirrors `event_fanout_single_run.rs` but exercises the
//! browser-compatible HTTP / Server-Sent Events tier:
//!
//! 1. Boot `ControlPlaneServer` with BOTH a gRPC listener (for the
//!    worker bidi session) and an HTTP listener (for the orchestrator).
//! 2. Connect a worker over gRPC with an echo handler.
//! 3. Submit a run over HTTP (`POST /v1/cp/submit`) and subscribe via
//!    SSE (`GET /v1/cp/events/{run_id}`).
//! 4. Verify at least one event lands on the SSE stream within 5s and
//!    decodes back into a `RunEventWire` whose `run_id` matches the
//!    submitted run.
//!
//! The entire file is gated behind the `http-transport` feature because
//! the HTTP tier itself is feature-gated on the crate.

#![cfg(feature = "http-transport")]

use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;

use blazen_controlplane::http::PostcardEnvelope;
use blazen_controlplane::protocol::{
    Assignment, ENVELOPE_VERSION, RunEventWire, RunStateSnapshotWire, SubmitRequest,
};
use blazen_controlplane::server::ControlPlaneServer;
use blazen_controlplane::worker::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};

use blazen_core::distributed::{AdmissionMode, WorkerCapability};

/// Bind to an ephemeral port and return the address. The port is
/// released before we hand it to the server so the server-side
/// `bind` succeeds.
async fn ephemeral_addr() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

/// Locate the first occurrence of `needle` inside `haystack`. SSE
/// framing terminates each event with a blank line (`\n\n`), which we
/// scan for as we accumulate streaming bytes.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Try to extract one `RunEventWire` out of the byte buffer `buf`.
/// Drains every complete SSE frame (`...\n\n`) from `buf` and returns
/// the first event that decodes successfully, or `None` if nothing
/// complete is in the buffer yet.
fn drain_one_event(buf: &mut Vec<u8>) -> Option<RunEventWire> {
    while let Some(end_idx) = find_subslice(buf, b"\n\n") {
        let frame: Vec<u8> = buf.drain(..end_idx + 2).collect();
        let frame_str = String::from_utf8_lossy(&frame);
        for line in frame_str.lines() {
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            // axum's SSE serializer emits `data: <body>` (with a
            // leading space), but the SSE spec allows either, so
            // trim defensively.
            let payload = payload.trim_start();
            let Ok(env) = serde_json::from_str::<PostcardEnvelope>(payload) else {
                continue;
            };
            if let Ok(ev) = env.decode::<RunEventWire>() {
                return Some(ev);
            }
        }
    }
    None
}

/// Read SSE chunks from `resp` until we decode a `RunEventWire` or the
/// 5s deadline elapses.
async fn read_first_sse_event(mut resp: reqwest::Response) -> Option<RunEventWire> {
    let mut buf = Vec::<u8>::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let Ok(chunk_res) = tokio::time::timeout(remaining, resp.chunk()).await else {
            return None;
        };
        let Ok(Some(chunk)) = chunk_res else {
            return None;
        };
        buf.extend_from_slice(&chunk);
        if let Some(ev) = drain_one_event(&mut buf) {
            return Some(ev);
        }
    }
    None
}

/// Minimal handler used solely to keep the worker happy — the test
/// only cares that the server emits status events around the run, not
/// what the worker returns.
struct EchoHandler;

#[async_trait]
impl AssignmentHandler for EchoHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        // Brief sleep so the orchestrator's SSE subscription has time
        // to attach before the run reaches a terminal state — the
        // broadcast bus drops events when there are zero receivers.
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok(serde_json::json!({ "ok": true }))
    }
}

#[tokio::test]
async fn sse_events_one_streams_run_events() {
    // ----- Server: bind gRPC + HTTP on ephemeral ports -----
    let grpc_addr = ephemeral_addr().await;
    let http_addr = ephemeral_addr().await;

    let server = ControlPlaneServer::new("cp").with_http(http_addr);
    let server_handle = tokio::spawn(async move {
        let _ = server.serve(grpc_addr).await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    // ----- Worker: connect over gRPC -----
    let cfg = WorkerConfig::new(format!("http://{grpc_addr}"), "worker-sse")
        .with_capability(WorkerCapability {
            kind: "workflow:sse".into(),
            version: 0,
        })
        .with_admission(AdmissionMode::Fixed { max_in_flight: 1 });
    let worker = Worker::connect(cfg).expect("worker connect");
    let worker_handle = tokio::spawn(async move { worker.run(EchoHandler).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // ----- Orchestrator: submit via HTTP -----
    let http = reqwest::Client::new();
    let base = format!("http://{http_addr}");

    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: "sse".into(),
        workflow_version: None,
        input_json: b"{}".to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
    };
    let submit_envelope = PostcardEnvelope::encode(&submit).expect("encode submit");
    let submit_resp: PostcardEnvelope = http
        .post(format!("{base}/v1/cp/submit"))
        .json(&submit_envelope)
        .send()
        .await
        .expect("submit POST")
        .error_for_status()
        .expect("submit status")
        .json()
        .await
        .expect("submit body JSON");
    let snap: RunStateSnapshotWire = submit_resp.decode().expect("decode snapshot");
    let run_id = snap.run_id;

    // ----- Subscribe via SSE -----
    let sse_resp = http
        .get(format!("{base}/v1/cp/events/{run_id}"))
        .send()
        .await
        .expect("sse open");
    assert!(
        sse_resp.status().is_success(),
        "sse status: {}",
        sse_resp.status()
    );

    // Stream raw chunks and split on the SSE frame terminator
    // (`\n\n`). Each `data:` line carries a JSON-encoded
    // `PostcardEnvelope` that decodes to a `RunEventWire`.
    let event = read_first_sse_event(sse_resp)
        .await
        .expect("at least one SSE event arrived within 5s");
    assert_eq!(event.run_id, run_id, "event run_id matches submitted run");
    assert!(
        !event.event_type.is_empty(),
        "event_type should be set (got empty string)",
    );
    // Server-emitted status events are namespaced under `status.*`;
    // handler-emitted events use whatever prefix the worker chose.
    // For this echo handler we expect ONLY server-side status events,
    // so the prefix should always be `status.`.
    assert!(
        event.event_type.starts_with("status."),
        "expected a status.* event (got {})",
        event.event_type,
    );

    worker_handle.abort();
    server_handle.abort();
}
