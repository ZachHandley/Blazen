//! Orchestrator-side client for the control plane.
//!
//! Implements [`blazen_core::distributed::OrchestratorClient`] on top
//! of a tonic gRPC channel — every method postcard-encodes a typed
//! request, sends it as a [`pb::PostcardRequest`], and postcard-decodes
//! the response. Streaming RPCs ([`OrchestratorClient::subscribe_run_events`],
//! [`Client::subscribe_all`]) wrap the tonic `Streaming<PostcardResponse>`
//! in a [`blazen_core::distributed::RunEventStream`] that decodes each
//! frame as a core [`RunEvent`].
//!
//! ## Auth
//!
//! Like the [`super::worker::Worker`], the client reads
//! `BLAZEN_PEER_TOKEN` and injects an `authorization: Bearer <token>`
//! metadata header on every RPC.
//!
//! ## TLS
//!
//! Pass a [`tonic::transport::ClientTlsConfig`] to [`Client::connect`],
//! or use [`Client::with_mtls`] to load a client identity + CA from PEM
//! files via [`crate::tls::load_client_tls`].

// PR5: remote-mode ModelManager client. Gated behind the `model-client`
// feature so consumers of just the workflow control plane don't pay
// for the model-server symbols.
#[cfg(feature = "model-client")]
pub mod model_client;

#[cfg(feature = "model-client")]
pub use model_client::ModelClient;

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::Mutex;
use tonic::Request;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use uuid::Uuid;

use blazen_core::distributed::{
    OrchestratorClient, RunEvent, RunEventStream, RunStateSnapshot, SubmitWorkflowRequest,
    WorkerInfo,
};
use blazen_core::error::WorkflowError;

use crate::auth;
use crate::error::ControlPlaneError;
use crate::pb;
use crate::protocol::{
    CancelRequest, DescribeRequest, DrainWorkerRequest, ENVELOPE_VERSION, ListWorkersRequest,
    ListWorkersResponse, RunEventWire, RunStateSnapshotWire, SubmitRequest, SubscribeAllRequest,
    SubscribeRunRequest,
};

type GrpcClient = pb::blazen_control_plane_client::BlazenControlPlaneClient<Channel>;

/// gRPC client for the orchestrator side of the control plane.
///
/// Cheaply cloneable — the inner tonic client (which serialises calls
/// behind a `Mutex` because each gRPC method takes `&mut self`) is held
/// in an `Arc`. Use [`Client::connect`] or [`Client::with_mtls`] to
/// open a connection.
#[derive(Clone)]
pub struct Client {
    inner: Arc<Mutex<GrpcClient>>,
}

impl Client {
    /// Open a connection to the control plane at `endpoint`. Pass
    /// `tls = None` for plaintext (still uses the bearer token from
    /// `BLAZEN_PEER_TOKEN` if set).
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] if the endpoint URI is
    /// invalid or the TCP/HTTP-2 handshake fails. Returns
    /// [`ControlPlaneError::Tls`] if a non-`None` TLS config cannot be
    /// applied to the endpoint.
    pub async fn connect(
        endpoint: impl Into<String>,
        tls: Option<ClientTlsConfig>,
    ) -> Result<Self, ControlPlaneError> {
        let endpoint_str = endpoint.into();
        let mut endpoint = Endpoint::from_shared(endpoint_str)
            .map_err(|e| ControlPlaneError::Transport(format!("invalid endpoint URI: {e}")))?
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_keep_alive_interval(Duration::from_secs(20))
            .keep_alive_while_idle(true);
        if let Some(t) = tls {
            endpoint = endpoint
                .tls_config(t)
                .map_err(|e| ControlPlaneError::Tls(e.to_string()))?;
        }
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("connect: {e}")))?;
        let client = GrpcClient::new(channel);
        Ok(Self {
            inner: Arc::new(Mutex::new(client)),
        })
    }

    /// Convenience: load a client identity + CA from PEM files and
    /// open an mTLS connection.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Tls`] if any PEM file cannot be
    /// read or if the resulting TLS config is rejected by tonic. Any
    /// transport failure is surfaced as
    /// [`ControlPlaneError::Transport`].
    pub async fn with_mtls(
        endpoint: impl Into<String>,
        cert_pem: &Path,
        key_pem: &Path,
        ca_pem: &Path,
    ) -> Result<Self, ControlPlaneError> {
        let tls = crate::tls::load_client_tls(cert_pem, key_pem, ca_pem)
            .map_err(|e| ControlPlaneError::Tls(e.to_string()))?;
        Self::connect(endpoint, Some(tls)).await
    }

    /// Send a [`DrainInstruction`] to the named worker.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] for RPC failures and
    /// surfaces server-side `NotFound` / `FailedPrecondition` as
    /// `Transport` (the wire-level mapping at the moment).
    pub async fn drain_worker(
        &self,
        node_id: String,
        immediate: bool,
    ) -> Result<(), ControlPlaneError> {
        let wire = DrainWorkerRequest {
            envelope_version: ENVELOPE_VERSION,
            node_id,
            immediate,
        };
        let req = encode_request_cp(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_cp(&mut req_tonic)?;
        let _ = self
            .inner
            .lock()
            .await
            .drain_worker(req_tonic)
            .await
            .map_err(|s| ControlPlaneError::Transport(s.to_string()))?;
        Ok(())
    }

    /// Subscribe to events across all runs, optionally filtered by tag
    /// predicates. See [`OrchestratorClient::subscribe_run_events`] for
    /// per-run subscription.
    ///
    /// The server side currently returns an empty stream until the
    /// run-event fan-out lands; this client is already wire-compatible
    /// so consumers won't need to change once the server side is wired
    /// up.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] for RPC setup failures.
    pub async fn subscribe_all(
        &self,
        required_tags: Vec<String>,
    ) -> Result<RunEventStream<'_>, ControlPlaneError> {
        let wire = SubscribeAllRequest {
            envelope_version: ENVELOPE_VERSION,
            required_tags,
        };
        let req = encode_request_cp(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_cp(&mut req_tonic)?;
        let streaming = self
            .inner
            .lock()
            .await
            .subscribe_all(req_tonic)
            .await
            .map_err(|s| ControlPlaneError::Transport(s.to_string()))?
            .into_inner();
        Ok(wrap_event_stream(streaming))
    }
}

#[async_trait]
impl OrchestratorClient for Client {
    async fn submit_workflow(
        &self,
        request: SubmitWorkflowRequest,
    ) -> Result<RunStateSnapshot, WorkflowError> {
        let wire = SubmitRequest::from_core(&request)?;
        let req = encode_request_wf(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_wf(&mut req_tonic)?;
        let resp = self
            .inner
            .lock()
            .await
            .submit_workflow(req_tonic)
            .await
            .map_err(|s| status_to_workflow_err(&s))?;
        let snap: RunStateSnapshotWire = decode_wf(&resp.into_inner().postcard_payload)?;
        snap.to_core().map_err(WorkflowError::from)
    }

    async fn cancel_workflow(&self, run_id: Uuid) -> Result<RunStateSnapshot, WorkflowError> {
        let wire = CancelRequest {
            envelope_version: ENVELOPE_VERSION,
            run_id,
        };
        let req = encode_request_wf(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_wf(&mut req_tonic)?;
        let resp = self
            .inner
            .lock()
            .await
            .cancel_workflow(req_tonic)
            .await
            .map_err(|s| status_to_workflow_err(&s))?;
        let snap: RunStateSnapshotWire = decode_wf(&resp.into_inner().postcard_payload)?;
        snap.to_core().map_err(WorkflowError::from)
    }

    async fn describe_workflow(&self, run_id: Uuid) -> Result<RunStateSnapshot, WorkflowError> {
        let wire = DescribeRequest {
            envelope_version: ENVELOPE_VERSION,
            run_id,
        };
        let req = encode_request_wf(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_wf(&mut req_tonic)?;
        let resp = self
            .inner
            .lock()
            .await
            .describe_workflow(req_tonic)
            .await
            .map_err(|s| status_to_workflow_err(&s))?;
        let snap: RunStateSnapshotWire = decode_wf(&resp.into_inner().postcard_payload)?;
        snap.to_core().map_err(WorkflowError::from)
    }

    async fn subscribe_run_events<'a>(
        &'a self,
        run_id: Uuid,
    ) -> Result<RunEventStream<'a>, WorkflowError> {
        let wire = SubscribeRunRequest {
            envelope_version: ENVELOPE_VERSION,
            run_id,
        };
        let req = encode_request_wf(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_wf(&mut req_tonic)?;
        let streaming = self
            .inner
            .lock()
            .await
            .subscribe_run_events(req_tonic)
            .await
            .map_err(|s| status_to_workflow_err(&s))?
            .into_inner();
        Ok(wrap_event_stream(streaming))
    }

    async fn list_workers(&self) -> Result<Vec<WorkerInfo>, WorkflowError> {
        let wire = ListWorkersRequest {
            envelope_version: ENVELOPE_VERSION,
        };
        let req = encode_request_wf(&wire)?;
        let mut req_tonic = Request::new(req);
        inject_bearer_wf(&mut req_tonic)?;
        let resp = self
            .inner
            .lock()
            .await
            .list_workers(req_tonic)
            .await
            .map_err(|s| status_to_workflow_err(&s))?;
        let list: ListWorkersResponse = decode_wf(&resp.into_inner().postcard_payload)?;
        Ok(list.workers.iter().map(Into::into).collect())
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Postcard-encode `value` and wrap it in a `PostcardRequest`. Used by
/// inherent methods that surface [`ControlPlaneError`].
fn encode_request_cp<T: serde::Serialize>(
    value: &T,
) -> Result<pb::PostcardRequest, ControlPlaneError> {
    let payload = postcard::to_allocvec(value)?;
    Ok(pb::PostcardRequest {
        postcard_payload: payload,
    })
}

/// Postcard-encode `value` and wrap it in a `PostcardRequest`. Used by
/// `OrchestratorClient` trait methods that surface [`WorkflowError`].
fn encode_request_wf<T: serde::Serialize>(value: &T) -> Result<pb::PostcardRequest, WorkflowError> {
    let payload = postcard::to_allocvec(value)
        .map_err(|e| WorkflowError::Other(anyhow::anyhow!("postcard encode failed: {e}")))?;
    Ok(pb::PostcardRequest {
        postcard_payload: payload,
    })
}

/// Postcard-decode `bytes` into `T`, mapping any failure to
/// [`WorkflowError`].
fn decode_wf<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, WorkflowError> {
    postcard::from_bytes(bytes)
        .map_err(|e| WorkflowError::Other(anyhow::anyhow!("postcard decode failed: {e}")))
}

/// Inject the `authorization: Bearer <token>` metadata header on a
/// request when `BLAZEN_PEER_TOKEN` is set. No-op otherwise.
fn inject_bearer_cp<T>(req: &mut Request<T>) -> Result<(), ControlPlaneError> {
    if let Some(bearer) = auth::bearer_metadata_value() {
        let value = bearer
            .parse::<tonic::metadata::MetadataValue<_>>()
            .map_err(|e| {
                ControlPlaneError::Unauthenticated(format!("invalid bearer header: {e}"))
            })?;
        req.metadata_mut().insert("authorization", value);
    }
    Ok(())
}

/// `WorkflowError` flavour of [`inject_bearer_cp`].
fn inject_bearer_wf<T>(req: &mut Request<T>) -> Result<(), WorkflowError> {
    if let Some(bearer) = auth::bearer_metadata_value() {
        let value = bearer
            .parse::<tonic::metadata::MetadataValue<_>>()
            .map_err(|e| WorkflowError::Other(anyhow::anyhow!("invalid bearer header: {e}")))?;
        req.metadata_mut().insert("authorization", value);
    }
    Ok(())
}

/// Map a tonic `Status` (RPC-level failure) onto [`WorkflowError`].
/// Preserves the gRPC code in the formatted message so callers don't
/// lose triage information.
fn status_to_workflow_err(status: &tonic::Status) -> WorkflowError {
    WorkflowError::Other(anyhow::anyhow!(
        "control-plane RPC failed (code={:?}): {}",
        status.code(),
        status.message(),
    ))
}

/// Map a `Streaming<PostcardResponse>` from a subscribe RPC into the
/// trait's [`RunEventStream`]: decode each frame as a [`RunEventWire`],
/// convert to core [`RunEvent`], surface decode failures as stream
/// items.
fn wrap_event_stream<'a>(streaming: tonic::Streaming<pb::PostcardResponse>) -> RunEventStream<'a> {
    let mapped = streaming.map(|res| {
        let resp = res.map_err(|s| status_to_workflow_err(&s))?;
        let wire: RunEventWire = postcard::from_bytes(&resp.postcard_payload).map_err(|e| {
            WorkflowError::Other(anyhow::anyhow!("postcard decode of RunEventWire: {e}"))
        })?;
        wire.to_core().map_err(WorkflowError::from)
    });
    let pinned: Pin<Box<dyn futures_core::Stream<Item = Result<RunEvent, WorkflowError>> + Send>> =
        Box::pin(mapped);
    pinned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn connect_rejects_bad_endpoint() {
        match Client::connect("not a uri", None).await {
            Ok(_) => panic!("expected bad endpoint to be rejected"),
            Err(e) => assert!(
                matches!(e, ControlPlaneError::Transport(_)),
                "expected Transport error, got {e:?}",
            ),
        }
    }
}
