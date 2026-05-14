//! Phase 1 integration tests for the blazen-controlplane gRPC + HTTP
//! tiers. These spin up a real `ControlPlaneServer` on ephemeral ports
//! and exercise the worker session + submit flow end-to-end.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

use blazen_controlplane::protocol::{
    AdmissionModeWire, CapabilityWire, ENVELOPE_VERSION, ServerToWorker, SubmitRequest,
    WorkerHello, WorkerToServer,
};
use blazen_controlplane::server::ControlPlaneServer;

use futures_util::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;

/// Bind to an ephemeral port and return the resolved address.
async fn ephemeral_addr() -> SocketAddr {
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

#[tokio::test]
async fn grpc_worker_session_handshake() {
    let addr = ephemeral_addr().await;

    // Start the server.
    let server = ControlPlaneServer::new("test-cp");
    let server_handle = tokio::spawn(async move {
        let _ = server.serve(addr).await;
    });

    // Give it a moment to bind.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect a raw tonic client.
    let endpoint = format!("http://{addr}");
    let channel = Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .expect("connect");
    let mut client =
        blazen_controlplane::pb::blazen_control_plane_client::BlazenControlPlaneClient::new(
            channel,
        );

    // Build outbound stream — start with a Hello.
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let hello = WorkerToServer::Hello(WorkerHello {
        envelope_version: ENVELOPE_VERSION,
        node_id: "worker-a".into(),
        capabilities: vec![CapabilityWire {
            kind: "workflow:test".into(),
            version: 0,
        }],
        tags: BTreeMap::new(),
        admission: AdmissionModeWire::Fixed { max_in_flight: 4 },
        supported_envelope_versions: vec![1],
    });
    tx.send(blazen_controlplane::pb::PostcardRequest {
        postcard_payload: encode(&hello),
    })
    .await
    .unwrap();

    let outbound = ReceiverStream::new(rx);
    let mut response_stream = client
        .worker_session(outbound)
        .await
        .expect("worker_session call")
        .into_inner();

    // Expect a Welcome frame.
    let first = response_stream
        .next()
        .await
        .expect("got first server frame")
        .expect("frame is Ok");
    let server_frame: ServerToWorker = decode(&first.postcard_payload);
    match server_frame {
        ServerToWorker::Welcome(w) => {
            assert_eq!(w.envelope_version, ENVELOPE_VERSION);
            assert_eq!(w.negotiated_envelope_version, 1);
            assert_ne!(w.session_id, uuid::Uuid::nil());
        }
        other => panic!("expected Welcome, got {other:?}"),
    }

    // Submit a workflow targeting `workflow:test`. The worker should
    // receive an Assignment frame.
    let submit = SubmitRequest {
        envelope_version: ENVELOPE_VERSION,
        workflow_name: "test".into(),
        workflow_version: None,
        input_json: b"{}".to_vec(),
        required_tags: vec![],
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker: true,
        resource_hint: None,
    };
    let _ = client
        .submit_workflow(blazen_controlplane::pb::PostcardRequest {
            postcard_payload: encode(&submit),
        })
        .await
        .expect("submit_workflow");

    // Worker should receive the Assignment.
    let second = tokio::time::timeout(Duration::from_secs(2), response_stream.next())
        .await
        .expect("assignment delivered within 2s")
        .expect("stream not closed")
        .expect("frame is Ok");
    let server_frame: ServerToWorker = decode(&second.postcard_payload);
    match server_frame {
        ServerToWorker::Assignment(a) => {
            assert_eq!(a.workflow_name, "test");
            assert_eq!(a.input_json, b"{}");
        }
        other => panic!("expected Assignment, got {other:?}"),
    }

    // Drop the outbound sender — closes the worker stream.
    drop(tx);
    drop(response_stream);

    server_handle.abort();
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
