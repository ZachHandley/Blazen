//! End-to-end integration test for the blazen-peer distributed workflow
//! layer.
//!
//! Spins up a [`BlazenPeerServer`] on a loopback port, connects a client,
//! and exercises the `InvokeSubWorkflow` RPC round-trip with both a
//! registered echo step and an unknown step.
//!
//! This test runs over plain TCP (no TLS) to keep the setup minimal.
//! mTLS integration tests are deferred to a follow-up.

#![cfg(all(feature = "server", feature = "client"))]

use std::sync::Arc;

use blazen_core::step::{StepFn, StepOutput, StepRegistration};
use blazen_core::step_registry::register_step_builder;
use blazen_events::{Event, StartEvent, StopEvent};
use blazen_peer::protocol::SubWorkflowRequest;
use blazen_peer::server::BlazenPeerServer;
use tonic::transport::server::TcpIncoming;

/// Step ID used by the echo step in these tests. Uses a fully-qualified
/// path to avoid collisions with any other test or production steps.
const ECHO_STEP_ID: &str = "blazen_peer::tests::e2e::echo";

/// Build an echo step: reads the `StartEvent` data and returns it
/// verbatim as `StopEvent.result`.
fn echo_step() -> StepRegistration {
    let handler: StepFn = Arc::new(|event, _ctx| {
        Box::pin(async move {
            let start = event
                .as_any()
                .downcast_ref::<StartEvent>()
                .expect("echo step expects a StartEvent");
            Ok(StepOutput::Single(Box::new(StopEvent {
                result: start.data.clone(),
            })))
        })
    });

    StepRegistration {
        name: "echo".into(),
        accepts: vec![StartEvent::event_type()],
        emits: vec![StopEvent::event_type()],
        handler,
        max_concurrency: 0,
    }
}

/// Register our echo step in the global step registry. Safe to call
/// multiple times (the registry is idempotent for the same fn pointer).
fn ensure_echo_registered() {
    register_step_builder(ECHO_STEP_ID, echo_step);
}

/// Helper: bind a `TcpIncoming` on port 0 and return the actual address
/// alongside the stream. This gives the test a race-free way to discover
/// the server port before starting the client.
fn bind_ephemeral() -> (std::net::SocketAddr, TcpIncoming) {
    let incoming =
        TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).expect("bind to ephemeral port");
    let addr = incoming.local_addr().expect("local_addr after bind");
    (addr, incoming)
}

/// Happy path: send a sub-workflow request with the echo step and verify
/// the response echoes the input back.
#[tokio::test]
async fn invoke_echo_step_returns_input() {
    ensure_echo_registered();

    let (addr, incoming) = bind_ephemeral();

    // Spawn the gRPC server in the background.
    let server = BlazenPeerServer::new("test-node-1");
    let svc = blazen_peer::pb::blazen_peer_server::BlazenPeerServer::new(server);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("server should not error");
    });

    // Give the server a moment to start accepting connections.
    tokio::task::yield_now().await;

    // Connect the client.
    let endpoint = format!("http://{addr}");
    let mut client = blazen_peer::client::BlazenPeerClient::connect(&endpoint, "test-client-1")
        .await
        .expect("client connect should succeed");

    // Build and send an echo request.
    let input = serde_json::json!({"hello": "world", "n": 42});
    let request = SubWorkflowRequest::new(
        "echo-workflow",
        vec![ECHO_STEP_ID.to_string()],
        &input,
        None,
    )
    .expect("request construction should not fail");

    let response = client
        .invoke_sub_workflow(request)
        .await
        .expect("invoke_sub_workflow should succeed");

    // The echo step puts the input straight into StopEvent.result, which
    // the server encodes as result_json.
    assert!(
        response.error.is_none(),
        "expected no error, got: {:?}",
        response.error
    );
    let result = response
        .result_value()
        .expect("result_json should be valid JSON")
        .expect("result_json should be Some");
    assert_eq!(result, input);

    // Shut down cleanly.
    let _ = shutdown_tx.send(());
    let _ = server_handle.await;
}

/// Error path: request an unknown step and verify the server returns an
/// error.
#[tokio::test]
async fn invoke_unknown_step_returns_error() {
    ensure_echo_registered();

    let (addr, incoming) = bind_ephemeral();

    let server = BlazenPeerServer::new("test-node-2");
    let svc = blazen_peer::pb::blazen_peer_server::BlazenPeerServer::new(server);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("server should not error");
    });

    tokio::task::yield_now().await;

    let endpoint = format!("http://{addr}");
    let mut client = blazen_peer::client::BlazenPeerClient::connect(&endpoint, "test-client-2")
        .await
        .expect("client connect should succeed");

    let input = serde_json::json!({"test": true});
    let request = SubWorkflowRequest::new(
        "unknown-workflow",
        vec!["does_not_exist::step".to_string()],
        &input,
        None,
    )
    .expect("request construction should not fail");

    let result = client.invoke_sub_workflow(request).await;

    // The server should return a transport error (tonic NOT_FOUND status)
    // because the step is not registered.
    assert!(result.is_err(), "expected error for unknown step");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("does_not_exist::step") || err_msg.contains("unknown step"),
        "error should mention the step id, got: {err_msg}"
    );

    let _ = shutdown_tx.send(());
    let _ = server_handle.await;
}
