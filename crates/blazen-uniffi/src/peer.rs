//! Distributed peer surface for the UniFFI bindings.
//!
//! Wraps upstream [`blazen_peer::BlazenPeerServer`] and
//! [`blazen_peer::BlazenPeerClient`] so foreign callers can run Blazen
//! workflows across processes / machines via the peer gRPC transport.
//!
//! ## Deviations from `blazen-py`
//!
//! `blazen-py` mirrors the upstream `BlazenPeerClient::invoke_sub_workflow`
//! surface directly, including the full `SubWorkflowRequest` /
//! `SubWorkflowResponse` envelope (with state map and `remote_refs`). UniFFI
//! does not handle nested record graphs across four foreign runtimes as
//! cleanly as PyO3, so this layer flattens the response down to the bits a
//! workflow caller actually consumes — the terminal `StopEvent` payload —
//! and exposes it as the same [`crate::workflow::WorkflowResult`] type the
//! in-process [`crate::workflow::Workflow::run`] returns. Session-ref
//! proxies (`remote_refs`, `deref_session_ref`, `release_session_ref`) and
//! the streaming-events RPC are deferred to a follow-up; today's surface
//! covers fire-and-forget remote execution with a JSON result.
//!
//! ## Deviations from upstream
//!
//! Upstream [`blazen_peer::BlazenPeerServer`] does not have a
//! `register_workflow` API or a `start` / `stop` lifecycle — it dispatches
//! every invocation by step ID through the process-wide
//! [`blazen_core::step_registry`] and its `serve` method takes `self` by
//! value and runs until the bound socket errors. The originally-requested
//! `register_workflow` / `start` / `stop` shape therefore cannot be honoured
//! without inventing a parallel workflow catalog in this crate; instead,
//! [`PeerServer::serve`] (and its blocking sibling) is the single entry
//! point, matching upstream and matching the Python binding's
//! [`PyBlazenPeerServer::serve`] method.

use std::net::SocketAddr;
use std::sync::Arc;

use blazen_peer::{BlazenPeerClient, BlazenPeerServer, SubWorkflowRequest};
use tokio::sync::Mutex;

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;
use crate::workflow::{Event, WorkflowResult};

/// Node-local Blazen peer gRPC server.
///
/// Owns a stable `node_id` embedded in every
/// `RemoteRefDescriptor` this peer hands out, plus an in-process session-ref
/// registry. Construct with [`PeerServer::new`] and start the gRPC listener
/// with [`PeerServer::serve`] (async) or [`PeerServer::serve_blocking`].
///
/// Dispatched workflows are resolved at request time through the
/// process-wide [`blazen_core::step_registry`], so any workflow whose steps
/// have been registered in this process can be invoked remotely by name.
#[derive(uniffi::Object)]
pub struct PeerServer {
    inner: Mutex<Option<BlazenPeerServer>>,
}

#[uniffi::export]
impl PeerServer {
    /// Create a new peer server with a fresh, empty session-ref registry.
    ///
    /// `node_id` is the stable identifier that this server stamps onto every
    /// `RemoteRefDescriptor` it returns. Typical values are the hostname or
    /// a UUID picked at process startup.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(node_id: String) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Some(BlazenPeerServer::new(node_id))),
        })
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl PeerServer {
    /// Bind the gRPC server to `listen_address` and serve until the
    /// listener errors or the caller's async task is cancelled.
    ///
    /// `listen_address` must parse as a [`std::net::SocketAddr`] (for
    /// example `"0.0.0.0:50051"` or `"127.0.0.1:7443"`). This method
    /// consumes the underlying server; calling it twice on the same
    /// `PeerServer` returns [`BlazenError::Validation`].
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `listen_address` cannot be
    /// parsed or the server has already been consumed by a prior call, and
    /// [`BlazenError::Peer`] (`kind = "Transport"`) if the gRPC listener
    /// fails to bind or encounters a fatal I/O error while serving.
    pub async fn serve(self: Arc<Self>, listen_address: String) -> BlazenResult<()> {
        let addr: SocketAddr = listen_address
            .parse()
            .map_err(|e| BlazenError::Validation {
                message: format!("invalid peer listen address {listen_address:?}: {e}"),
            })?;
        let server = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or(BlazenError::Validation {
                message: "PeerServer already consumed by a prior serve() call".into(),
            })?
        };
        server.serve(addr).await.map_err(BlazenError::from)
    }
}

#[uniffi::export]
impl PeerServer {
    /// Synchronous variant of [`PeerServer::serve`] — blocks the current
    /// thread on the shared Tokio runtime until the server exits. Intended
    /// for foreign callers (Ruby scripts, Go `main`, Swift CLIs) that want
    /// a one-shot blocking bind without driving an async runtime.
    ///
    /// # Errors
    ///
    /// Same as [`PeerServer::serve`].
    pub fn serve_blocking(self: Arc<Self>, listen_address: String) -> BlazenResult<()> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.serve(listen_address).await })
    }
}

/// Client handle for invoking workflows on a remote [`PeerServer`].
///
/// Construct with [`PeerClient::connect`]. RPCs go out over a multiplexed
/// HTTP/2 channel held inside the client; multiple concurrent calls on the
/// same `PeerClient` are safe and share the connection.
#[derive(uniffi::Object)]
pub struct PeerClient {
    inner: Mutex<BlazenPeerClient>,
    node_id: String,
}

#[uniffi::export]
impl PeerClient {
    /// Open a connection to the peer at `address`.
    ///
    /// `address` must be a valid gRPC endpoint URI such as
    /// `"http://node-a.local:7443"`. `client_node_id` identifies *this* end
    /// of the connection in trace logs on both sides and is typically the
    /// local hostname or a process-startup UUID.
    ///
    /// This constructor is blocking — it drives the TCP connect on the
    /// shared Tokio runtime so foreign callers without an async story
    /// (Ruby, synchronous Go code) can still set up a client. The async
    /// connect path is internal to upstream `BlazenPeerClient` and is not
    /// re-exposed across UniFFI to avoid a constructor that returns a
    /// coroutine in every target language.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Peer`] (`kind = "Transport"`) if the
    /// endpoint URI is invalid or the TCP / HTTP/2 handshake fails.
    #[uniffi::constructor]
    pub fn connect(address: String, client_node_id: String) -> BlazenResult<Arc<Self>> {
        let node_id_for_client = client_node_id.clone();
        let client = runtime()
            .block_on(async move { BlazenPeerClient::connect(address, node_id_for_client).await })
            .map_err(BlazenError::from)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(client),
            node_id: client_node_id,
        }))
    }

    /// The node-id stamped into outgoing requests for tracing.
    #[must_use]
    pub fn node_id(self: Arc<Self>) -> String {
        self.node_id.clone()
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl PeerClient {
    /// Invoke a workflow on the connected peer and wait for its terminal
    /// result.
    ///
    /// - `workflow_name` is the symbolic name the remote peer's
    ///   [`blazen_core::step_registry`] knows the workflow as.
    /// - `step_ids` is the ordered list of step identifiers to execute.
    ///   Every entry must be registered on the remote peer's process or
    ///   the call fails with [`BlazenError::Peer`] (`kind = "UnknownStep"`).
    ///   This is required by the peer wire protocol — see
    ///   [`blazen_peer::SubWorkflowRequest`].
    /// - `input_json` is the JSON-encoded payload fed into the workflow's
    ///   entry step.
    /// - `timeout_secs` bounds the remote workflow's wall-clock execution.
    ///   `None` defers to the server's default deadline.
    ///
    /// The returned [`WorkflowResult`] carries the terminal `StopEvent`
    /// payload synthesised from the remote `SubWorkflowResponse`. Per-run
    /// LLM token usage and cost are not propagated over the wire and are
    /// reported as zero; foreign callers needing those should query the
    /// remote peer's telemetry directly.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Peer`] for encode / transport / envelope-
    /// version failures, or [`BlazenError::Workflow`] when the remote
    /// reports a workflow-execution error in `SubWorkflowResponse::error`.
    pub async fn run_remote_workflow(
        self: Arc<Self>,
        workflow_name: String,
        step_ids: Vec<String>,
        input_json: String,
        timeout_secs: Option<u64>,
    ) -> BlazenResult<WorkflowResult> {
        let input: serde_json::Value = serde_json::from_str(&input_json)?;
        let request = SubWorkflowRequest::new(workflow_name, step_ids, &input, timeout_secs)
            .map_err(BlazenError::from)?;
        let response = {
            let mut guard = self.inner.lock().await;
            guard
                .invoke_sub_workflow(request)
                .await
                .map_err(BlazenError::from)?
        };
        sub_workflow_response_to_result(response)
    }
}

#[uniffi::export]
impl PeerClient {
    /// Synchronous variant of [`PeerClient::run_remote_workflow`] — blocks
    /// the current thread on the shared Tokio runtime.
    ///
    /// # Errors
    ///
    /// Same as [`PeerClient::run_remote_workflow`].
    pub fn run_remote_workflow_blocking(
        self: Arc<Self>,
        workflow_name: String,
        step_ids: Vec<String>,
        input_json: String,
        timeout_secs: Option<u64>,
    ) -> BlazenResult<WorkflowResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move {
            this.run_remote_workflow(workflow_name, step_ids, input_json, timeout_secs)
                .await
        })
    }
}

/// Convert an upstream [`blazen_peer::SubWorkflowResponse`] into the
/// flattened [`WorkflowResult`] surface UniFFI exposes.
///
/// The remote `result_json` field (the terminal `StopEvent` payload) becomes
/// the `data_json` of a synthesised `StopEvent`-typed [`Event`]; an empty
/// remote result is mapped to the JSON literal `null`. Remote token usage
/// and cost are not carried in the peer wire format and default to zero.
fn sub_workflow_response_to_result(
    response: blazen_peer::SubWorkflowResponse,
) -> BlazenResult<WorkflowResult> {
    if let Some(error) = response.error {
        return Err(BlazenError::Workflow { message: error });
    }
    let result_value = response.result_value().map_err(BlazenError::from)?;
    let data_json = match result_value {
        Some(value) => value.to_string(),
        None => "null".to_string(),
    };
    Ok(WorkflowResult {
        event: Event {
            event_type: "StopEvent".to_string(),
            data_json,
        },
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_cost_usd: 0.0,
    })
}
