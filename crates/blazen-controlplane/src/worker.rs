//! Worker-side client for the bidi `WorkerSession` stream.
//!
//! A [`Worker`] connects outbound to a [`ControlPlaneServer`](crate::server::ControlPlaneServer),
//! advertises its capabilities, and runs assignments dispatched by the
//! server until the connection is drained or shut down. The worker
//! handles:
//!
//! - **Handshake.** Sends [`WorkerHello`](crate::protocol::WorkerHello)
//!   on connect; expects [`Welcome`](crate::protocol::Welcome) back.
//! - **Heartbeats.** Periodic [`WorkerHeartbeat`](crate::protocol::WorkerHeartbeat)
//!   frames so the server can track liveness and in-flight load.
//! - **Assignments.** Spawns a task per incoming
//!   [`Assignment`](crate::protocol::Assignment), bounded by an optional
//!   per-assignment deadline. The user-supplied [`AssignmentHandler`]
//!   returns a JSON value (success) or an [`AssignmentFailure`]
//!   (failure); either becomes an [`AssignmentResult`](crate::protocol::AssignmentResult)
//!   sent back to the server.
//! - **Reactive offers.** On [`Offer`](crate::protocol::Offer),
//!   [`AssignmentHandler::evaluate_offer`] decides Claim vs Decline.
//! - **Cancel / drain.** Per-run [`CancellationToken`]s let
//!   [`Cancel`](crate::protocol::CancelInstruction) abort an in-flight
//!   task; [`Drain`](crate::protocol::DrainInstruction) tells the worker
//!   to stop pulling new work.
//! - **Reconnect.** If the bidi stream drops, the worker reconnects
//!   under a [`RetryPolicy`] (exponential backoff with optional cap on
//!   attempts). Old in-flight tasks are abandoned — the server's
//!   `surrender_session` puts them back on the pending queue and a new
//!   session picks them up, with at-least-once semantics. Callers that
//!   need idempotency should use [`SubmitWorkflowRequest::idempotency_key`].
//!
//! ## TLS
//!
//! Set [`WorkerConfig::tls`] to a [`tonic::transport::ClientTlsConfig`]
//! to enable TLS / mTLS. Use [`WorkerConfig::with_mtls`] to load a
//! client identity + CA from PEM files via the
//! [`crate::tls::load_client_tls`] helper.
//!
//! ## Auth
//!
//! The worker reads `BLAZEN_PEER_TOKEN` from the environment via
//! [`crate::auth::bearer_metadata_value`] and injects it as the
//! `authorization: Bearer <token>` metadata header on the gRPC request.
//! Unset = no auth, which the server-side
//! [`BearerAuthInterceptor`](crate::server::interceptor::BearerAuthInterceptor)
//! accepts when the same env var is unset on its side.
//!
//! [`SubmitWorkflowRequest::idempotency_key`]: blazen_core::distributed::SubmitWorkflowRequest::idempotency_key
//! [`CancellationToken`]: tokio_util::sync::CancellationToken

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::Request;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use uuid::Uuid;

use blazen_core::distributed::{
    AdmissionMode, AdmissionSnapshot, RunEvent as CoreRunEvent, WorkerCapability,
    WorkerSessionSink, WorkerTaint,
};

use crate::auth;
use crate::error::ControlPlaneError;
use crate::pb;
use crate::protocol::{
    self, AdmissionSnapshotWire, AssignmentEvent, AssignmentResult, AssignmentStatus,
    ENVELOPE_VERSION, OfferDecision, OfferOutcome, ServerToWorker, WorkerHeartbeat, WorkerHello,
    WorkerToServer,
};

/// How long a per-assignment control-plane key pre-warm waits for each
/// provider's [`crate::protocol::KeyResponse`] before giving up on that
/// provider (and letting the resolver fall through to env). Bounded so a
/// stalled control plane can't block assignment start indefinitely.
const CP_KEY_PREWARM_TIMEOUT: Duration = Duration::from_secs(5);

// ===========================================================================
// Public types
// ===========================================================================

/// Configuration for a [`Worker`]. Use [`WorkerConfig::new`] for the base
/// and chain builder methods to add capabilities, tags, TLS, etc.
#[derive(Clone)]
pub struct WorkerConfig {
    /// gRPC endpoint URI, e.g. `"http://cp.internal:7445"` or
    /// `"https://cp.example.com"`.
    pub endpoint: String,
    /// Stable identifier of this worker. Used for routing, reconnect
    /// detection (the server evicts the prior session for the same
    /// `node_id`), and metrics.
    pub node_id: String,
    /// Capabilities advertised at handshake. See
    /// [`blazen_core::distributed::WorkerCapability`].
    pub capabilities: Vec<WorkerCapability>,
    /// Free-form `key=value` tags used by submission-time tag predicates.
    /// `BTreeMap` for deterministic encoding.
    pub tags: BTreeMap<String, String>,
    /// Worker-side scheduling labels surfaced in [`WorkerHello::labels`].
    /// Filtered against job-side [`crate::protocol::Assignment::selector`]
    /// inside admission. Empty by default.
    pub labels: BTreeMap<String, String>,
    /// Worker-local provider API keys, keyed by provider name (e.g.
    /// `"fal"`, `"openai"`). Installed as the FIRST link in the
    /// [`blazen_llm::KeyResolver`] cascade at worker start, so they take
    /// precedence over any control-plane-shared or environment keys. Not
    /// place-scoped — a worker serves whatever place the session declares.
    /// `BTreeMap` for deterministic order. Empty by default (the worker
    /// installs no resolver and the prior explicit-then-env behaviour holds).
    pub provider_keys: BTreeMap<String, String>,
    /// Self-reported tenant/place this worker serves, surfaced in
    /// [`WorkerHello::place`]. Advisory — the server's bearer-derived
    /// identity wins (anti-spoof). `None` selects the default place.
    pub place: Option<String>,
    /// Providers whose control-plane keys the worker pre-warms ahead of
    /// each assignment (P4). The synchronous
    /// [`blazen_llm::KeyResolver::resolve`] cannot fetch over the network,
    /// so the worker proactively requests these providers' keys from the
    /// control plane on assignment receipt and caches them, ensuring a
    /// later resolve hit is served from cache. Empty by default — a worker
    /// that lists nothing here relies solely on worker-local + env keys.
    /// `BTreeMap`-free ordered `Vec` so the pre-warm order is deterministic.
    pub prewarm_providers: Vec<String>,
    /// Worker-side taints surfaced in [`WorkerHello::taints`]. Jobs without
    /// a matching toleration are not scheduled here. Empty by default.
    pub taints: Vec<WorkerTaint>,
    /// Capability-descriptor manifest surfaced in
    /// [`WorkerHello::descriptors`]. One entry per node this worker is
    /// willing to host. Empty by default — workers that don't publish a
    /// descriptor catalogue keep the legacy behaviour.
    pub descriptors: Vec<protocol::NodeDescriptorWire>,
    /// Admission mode declared at handshake.
    pub admission: AdmissionMode,
    /// Cadence of [`WorkerHeartbeat`] frames.
    pub heartbeat_interval: Duration,
    /// Versions of the postcard envelope this worker can decode. The
    /// server negotiates the highest mutually-supported version on
    /// `Welcome`.
    pub envelope_versions: Vec<u32>,
    /// TLS configuration. `None` = plaintext (the default).
    pub tls: Option<ClientTlsConfig>,
    /// Reconnect / retry policy for the bidi stream.
    pub retry: RetryPolicy,
    /// Explicit bearer token sent as `authorization: Bearer <token>` on
    /// the handshake. When `None`, falls back to `BLAZEN_PEER_TOKEN` from
    /// the environment.
    pub bearer_token: Option<String>,
    /// Optional probe-handle that sources real `vram_free_mb` for the
    /// heartbeat's [`AdmissionSnapshot`]. When `None`, the synthesized
    /// snapshot reports `vram_free_mb = None` (today's behavior).
    ///
    /// The host-only `blazen-resource-probe` crate is not part of the
    /// wasm32/wasi dependency graph (it shells out to GPU tooling), so the
    /// field is absent on those targets.
    #[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
    pub probe_handle: Option<blazen_resource_probe::ProbeHandle>,
}

impl WorkerConfig {
    /// Build a config with the required `endpoint` and `node_id` and
    /// sensible defaults for everything else.
    #[must_use]
    pub fn new(endpoint: impl Into<String>, node_id: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            node_id: node_id.into(),
            capabilities: Vec::new(),
            tags: BTreeMap::new(),
            labels: BTreeMap::new(),
            provider_keys: BTreeMap::new(),
            place: None,
            prewarm_providers: Vec::new(),
            taints: Vec::new(),
            descriptors: Vec::new(),
            admission: AdmissionMode::Fixed { max_in_flight: 1 },
            heartbeat_interval: Duration::from_secs(5),
            envelope_versions: vec![ENVELOPE_VERSION],
            tls: None,
            retry: RetryPolicy::default(),
            bearer_token: None,
            #[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
            probe_handle: None,
        }
    }

    /// Attach a [`blazen_resource_probe::ProbeHandle`]. When set, the
    /// heartbeat ticker sources `vram_free_mb` from the probe's latest
    /// snapshot instead of leaving it `None`.
    ///
    /// Host-only: `blazen-resource-probe` is not in the wasm32/wasi graph.
    #[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
    #[must_use]
    pub fn with_probe_handle(mut self, handle: blazen_resource_probe::ProbeHandle) -> Self {
        self.probe_handle = Some(handle);
        self
    }

    /// Append a capability the worker advertises.
    #[must_use]
    pub fn with_capability(mut self, capability: WorkerCapability) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// Insert a tag.
    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Set an explicit bearer token for the handshake. Takes precedence
    /// over `BLAZEN_PEER_TOKEN`.
    #[must_use]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Insert a scheduling label surfaced in [`WorkerHello::labels`].
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Insert a worker-local provider API key (e.g. `("fal", "key-…")`).
    ///
    /// At worker start these are installed as the first link in the
    /// [`blazen_llm::KeyResolver`] cascade, so they win over
    /// control-plane-shared and environment keys. The key value is never
    /// logged.
    #[must_use]
    pub fn with_provider_key(
        mut self,
        provider: impl Into<String>,
        key: impl Into<String>,
    ) -> Self {
        self.provider_keys.insert(provider.into(), key.into());
        self
    }

    /// Declare the tenant/place this worker serves, surfaced in
    /// [`WorkerHello::place`]. Advisory: the server's bearer-derived
    /// identity wins over it.
    #[must_use]
    pub fn with_place(mut self, place: impl Into<String>) -> Self {
        self.place = Some(place.into());
        self
    }

    /// Append a provider whose control-plane key is pre-warmed before each
    /// assignment (P4). See [`Self::prewarm_providers`].
    #[must_use]
    pub fn with_prewarm_provider(mut self, provider: impl Into<String>) -> Self {
        self.prewarm_providers.push(provider.into());
        self
    }

    /// Append a worker-side taint surfaced in [`WorkerHello::taints`].
    #[must_use]
    pub fn with_taint(mut self, taint: WorkerTaint) -> Self {
        self.taints.push(taint);
        self
    }

    /// Append a capability descriptor surfaced in
    /// [`WorkerHello::descriptors`]. The descriptor is consumed verbatim
    /// — the control plane uses its `id` field as the catalogue key.
    #[must_use]
    pub fn with_descriptor(mut self, descriptor: protocol::NodeDescriptorWire) -> Self {
        self.descriptors.push(descriptor);
        self
    }

    /// Set the admission mode.
    #[must_use]
    pub fn with_admission(mut self, admission: AdmissionMode) -> Self {
        self.admission = admission;
        self
    }

    /// Override the heartbeat cadence.
    #[must_use]
    pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    /// Override the retry policy.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Attach a [`ClientTlsConfig`] directly. Use [`Self::with_mtls`]
    /// when loading PEM files from disk.
    #[must_use]
    pub fn with_tls(mut self, tls: ClientTlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Load a client identity + CA from PEM files and use them for mTLS.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Tls`] if any of the PEM files cannot
    /// be read.
    pub fn with_mtls(
        mut self,
        cert_pem: &Path,
        key_pem: &Path,
        ca_pem: &Path,
    ) -> Result<Self, ControlPlaneError> {
        let tls = crate::tls::load_client_tls(cert_pem, key_pem, ca_pem)
            .map_err(|e| ControlPlaneError::Tls(e.to_string()))?;
        self.tls = Some(tls);
        Ok(self)
    }
}

/// Reconnect / retry policy for the worker's bidi session.
///
/// Each disconnect waits `initial_backoff * multiplier^attempt`, capped
/// at `max_backoff`. Set `max_attempts = Some(n)` to bound the total
/// retry count; `None` retries forever.
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub multiplier: f32,
    pub max_attempts: Option<u32>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            multiplier: 2.0,
            max_attempts: None,
        }
    }
}

impl RetryPolicy {
    /// Skip retries — a single failed reconnect ends [`Worker::run`].
    #[must_use]
    pub fn none() -> Self {
        Self {
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            multiplier: 1.0,
            max_attempts: Some(1),
        }
    }

    /// Backoff duration for `attempt`-th retry, 0-indexed. Capped at
    /// `max_backoff`.
    #[allow(clippy::cast_precision_loss)]
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let mut secs = self.initial_backoff.as_secs_f64();
        for _ in 0..attempt {
            secs *= f64::from(self.multiplier);
        }
        let secs = secs.min(self.max_backoff.as_secs_f64());
        Duration::from_secs_f64(secs.max(0.0))
    }
}

/// Reason an [`AssignmentHandler`] returned a failure. Surfaced to the
/// server as [`AssignmentStatus::Failed`].
#[derive(Debug, Clone)]
pub struct AssignmentFailure {
    pub error: String,
}

impl AssignmentFailure {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
        }
    }
}

impl std::fmt::Display for AssignmentFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.error)
    }
}

impl std::error::Error for AssignmentFailure {}

/// User-supplied logic for what a worker actually does with an assignment.
///
/// Implement this trait to plug a workflow runner (or any other
/// per-assignment behaviour) into a [`Worker`].
#[async_trait]
pub trait AssignmentHandler: Send + Sync + 'static {
    /// Run an assignment to completion.
    ///
    /// Return `Ok(value)` to report `Completed` with `value` JSON-encoded
    /// as the run's output. Return `Err(AssignmentFailure)` to report
    /// `Failed`. Cancellation (via [`Self::on_cancel`]) and deadline
    /// timeout (via [`crate::protocol::Assignment::deadline_ms`]) are
    /// surfaced as `Cancelled` and `Failed` by the worker without the
    /// handler having to do anything special.
    async fn handle(
        &self,
        assignment: protocol::Assignment,
        ctx: AssignmentContext,
    ) -> Result<Value, AssignmentFailure>;

    /// Called when the server sends a [`CancelInstruction`](crate::protocol::CancelInstruction)
    /// for a run currently held by this handler. The worker has already
    /// fired the matching `CancellationToken`; this hook is for the
    /// handler to free any external resources tied to the run.
    async fn on_cancel(&self, _run_id: Uuid) {}

    /// Called when the server sends a [`DrainInstruction`](crate::protocol::DrainInstruction).
    /// The worker stops accepting new assignments either way; this hook
    /// lets the handler kick off graceful shutdown of its own subsystems.
    async fn on_drain(&self, _immediate: bool) {}

    /// Decide whether to claim a Reactive offer. Default: always Claim.
    async fn evaluate_offer(&self, _offer: &protocol::Offer) -> OfferOutcome {
        OfferOutcome::Claim
    }
}

/// Per-assignment context handed to [`AssignmentHandler::handle`].
///
/// Carries a sink for emitting non-terminal
/// [`AssignmentEvent`](crate::protocol::AssignmentEvent) frames back to
/// the server during a long-running assignment.
pub struct AssignmentContext {
    sink: Arc<WorkerOutbox>,
    run_id: Uuid,
    cancel: CancellationToken,
    /// Per-worker map of outstanding input requests, keyed by
    /// `request_id`. The inbound pump fulfils the matching oneshot when a
    /// [`ServerToWorker::InputResponse`](crate::protocol::ServerToWorker::InputResponse)
    /// frame arrives. Shared (cloned `Arc`) with [`WorkerState`].
    pending: Arc<DashMap<String, oneshot::Sender<Value>>>,
}

impl AssignmentContext {
    /// Emit a non-terminal event from the running assignment.
    ///
    /// `data` is JSON-encoded into the wire `AssignmentEvent::data_json`
    /// field by the [`WorkerSessionSink`] adapter on the way out.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] if the worker's outbound
    /// channel is closed (the bidi session disconnected).
    pub async fn emit_event(&self, event_type: &str, data: Value) -> Result<(), ControlPlaneError> {
        let event = CoreRunEvent {
            run_id: self.run_id,
            event_type: event_type.to_string(),
            data,
            timestamp_ms: now_ms(),
        };
        self.sink
            .emit_event(self.run_id, event)
            .await
            .map_err(|e| ControlPlaneError::Transport(e.to_string()))
    }

    /// Raise an `input.request` and block until the orchestrator answers
    /// via `respond_to_input` (or the assignment is cancelled / the
    /// optional `timeout_ms` elapses).
    ///
    /// Emits an `"input.request"` [`AssignmentEvent`](crate::protocol::AssignmentEvent)
    /// carrying `{request_id, prompt, metadata}`, then awaits the matching
    /// [`ServerToWorker::InputResponse`](crate::protocol::ServerToWorker::InputResponse)
    /// frame. The returned [`Value`] is the JSON the orchestrator passed
    /// back.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] if the event cannot be
    /// emitted, the assignment is cancelled while waiting, the worker
    /// disconnects, or `timeout_ms` elapses with no answer.
    pub async fn request_input(
        &self,
        prompt: &str,
        metadata: Value,
        timeout_ms: Option<u64>,
    ) -> Result<Value, ControlPlaneError> {
        let request_id = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        self.pending.insert(request_id.clone(), tx);

        // If emitting fails, drop the pending entry before returning.
        if let Err(e) = self
            .emit_event(
                "input.request",
                serde_json::json!({
                    "request_id": request_id,
                    "prompt": prompt,
                    "metadata": metadata,
                }),
            )
            .await
        {
            self.pending.remove(&request_id);
            return Err(e);
        }

        let recv = async {
            tokio::select! {
                () = self.cancel.cancelled() => {
                    Err(ControlPlaneError::Transport("cancelled awaiting input".into()))
                }
                v = rx => {
                    v.map_err(|_| ControlPlaneError::Transport("input channel dropped".into()))
                }
            }
        };
        let result = match timeout_ms {
            Some(ms) => tokio::time::timeout(Duration::from_millis(ms), recv)
                .await
                .unwrap_or_else(|_| {
                    Err(ControlPlaneError::Transport(
                        "timed out awaiting input".into(),
                    ))
                }),
            None => recv.await,
        };
        if result.is_err() {
            self.pending.remove(&request_id);
        }
        result
    }

    /// Cancellation token tied to this run. The handler can `.await`
    /// [`CancellationToken::cancelled`] and abort cleanly when the server
    /// (or a drain) fires it.
    #[must_use]
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancel
    }

    /// The run identifier this context belongs to.
    #[must_use]
    pub fn run_id(&self) -> Uuid {
        self.run_id
    }
}

// ===========================================================================
// Internal: outbound channel + WorkerSessionSink adapter
// ===========================================================================

/// Worker → server send-side wrapped as a [`WorkerSessionSink`] for
/// integration with workflow runners that already know that trait.
pub(crate) struct WorkerOutbox {
    tx: mpsc::Sender<WorkerToServer>,
}

impl WorkerOutbox {
    fn new(tx: mpsc::Sender<WorkerToServer>) -> Self {
        Self { tx }
    }

    async fn send(&self, frame: WorkerToServer) -> Result<(), ControlPlaneError> {
        self.tx
            .send(frame)
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("worker outbox closed: {e}")))
    }
}

#[async_trait]
impl WorkerSessionSink for WorkerOutbox {
    async fn emit_event(
        &self,
        run_id: Uuid,
        event: CoreRunEvent,
    ) -> Result<(), blazen_core::error::WorkflowError> {
        let data_json = serde_json::to_vec(&event.data)?;
        let frame = WorkerToServer::Event(AssignmentEvent {
            envelope_version: ENVELOPE_VERSION,
            run_id,
            event_type: event.event_type,
            data_json,
            timestamp_ms: event.timestamp_ms,
        });
        self.tx
            .send(frame)
            .await
            .map_err(|_| blazen_core::error::WorkflowError::ChannelClosed)
    }

    async fn report_result(
        &self,
        run_id: Uuid,
        result: Result<Value, String>,
    ) -> Result<(), blazen_core::error::WorkflowError> {
        let (status, output_json, error) = match result {
            Ok(v) => (AssignmentStatus::Completed, serde_json::to_vec(&v)?, None),
            Err(e) => (AssignmentStatus::Failed, Vec::new(), Some(e)),
        };
        let frame = WorkerToServer::Result(AssignmentResult {
            envelope_version: ENVELOPE_VERSION,
            run_id,
            output_json,
            status,
            error,
        });
        self.tx
            .send(frame)
            .await
            .map_err(|_| blazen_core::error::WorkflowError::ChannelClosed)
    }

    async fn heartbeat(
        &self,
        in_flight: u32,
        admission_snapshot: Option<AdmissionSnapshot>,
    ) -> Result<(), blazen_core::error::WorkflowError> {
        let frame = WorkerToServer::Heartbeat(WorkerHeartbeat {
            envelope_version: ENVELOPE_VERSION,
            in_flight,
            queue_depth: 0,
            mem_mb: 0,
            cpu_pct: 0.0,
            admission_snapshot: admission_snapshot.as_ref().map(AdmissionSnapshotWire::from),
        });
        self.tx
            .send(frame)
            .await
            .map_err(|_| blazen_core::error::WorkflowError::ChannelClosed)
    }
}

// ===========================================================================
// Worker
// ===========================================================================

/// Per-worker shared state. `Arc`'d so the inbound pump, heartbeat
/// ticker, and assignment tasks can all see live counters / cancel
/// tokens.
struct WorkerState {
    /// Number of assignments currently being executed by this worker.
    in_flight: AtomicU32,
    /// Per-run cancellation tokens. Inserted on Assignment dispatch,
    /// fired by Cancel or Drain immediate, removed on completion.
    running: DashMap<Uuid, CancellationToken>,
    /// Outstanding `request_input` calls keyed by `request_id`. Inserted
    /// by [`AssignmentContext::request_input`], fulfilled by the inbound
    /// pump on a [`ServerToWorker::InputResponse`](crate::protocol::ServerToWorker::InputResponse).
    /// `Arc` so each [`AssignmentContext`] shares the same map.
    pending_inputs: Arc<DashMap<String, oneshot::Sender<Value>>>,
    /// Shutdown signal for the whole worker — drops the inbound pump,
    /// heartbeat ticker, and outbound channel.
    shutdown: CancellationToken,
    /// Set once the worker-local provider-key resolver has been installed
    /// into the process-global [`blazen_llm`] cascade. The install happens
    /// at the first session start; this guard keeps reconnects from
    /// pushing the same resolver again. Survives reconnects because
    /// [`WorkerState`] is `Arc`'d for the worker's whole lifetime.
    keys_installed: AtomicBool,
    /// Worker-side control-plane key client (P4). Installed into the
    /// process-global [`blazen_llm`] cascade AFTER [`WorkerLocalKeys`] so
    /// the order is worker-local → control-plane → env terminal. Holds the
    /// per-provider key cache and the pending-request correlation map; the
    /// inbound pump routes [`ServerToWorker::KeyResponse`] frames into it,
    /// and the assignment path pre-warms it. Survives reconnects (the
    /// outbound sender is rebound per session); the install into the
    /// global chain is guarded by [`Self::cp_keys_installed`].
    cp_keys: crate::client::ControlPlaneKeyClient,
    /// Set once [`Self::cp_keys`] has been pushed into the process-global
    /// cascade. Mirrors [`Self::keys_installed`] — keeps reconnects from
    /// installing the same resolver twice.
    cp_keys_installed: AtomicBool,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            in_flight: AtomicU32::new(0),
            running: DashMap::new(),
            pending_inputs: Arc::new(DashMap::new()),
            shutdown: CancellationToken::new(),
            keys_installed: AtomicBool::new(false),
            cp_keys: crate::client::ControlPlaneKeyClient::disconnected(),
            cp_keys_installed: AtomicBool::new(false),
        }
    }
}

/// Outcome of a single connection attempt. Drives the reconnect loop.
enum SessionOutcome {
    /// Stream ended normally (server closed). Reconnect.
    Disconnected,
    /// Server drained us (graceful or immediate). Stop.
    Drained,
    /// `Worker::shutdown` was called. Stop.
    Shutdown,
}

/// Carrier for the pieces produced by [`Worker::do_handshake`] —
/// the inbound stream plus the outbound channel and its sink adapter.
/// Kept owned by `run_one_session` so the channel + adapter live for
/// the whole session and are torn down in lockstep.
struct HandshakeResult {
    inbound: tonic::Streaming<pb::PostcardResponse>,
    outbound_tx: mpsc::Sender<WorkerToServer>,
    outbox: Arc<WorkerOutbox>,
}

/// A worker connection to the control plane. Build a [`WorkerConfig`],
/// call [`Worker::connect`] to establish the first session and run the
/// handshake, then call [`Worker::run`] to drive the worker forever
/// (subject to [`WorkerConfig::retry`] and [`Worker::shutdown`]).
pub struct Worker {
    config: WorkerConfig,
    state: Arc<WorkerState>,
    /// Latest session id assigned by the server on the most recent
    /// `Welcome`. `None` until the first handshake completes.
    session_id: Arc<Mutex<Option<Uuid>>>,
}

impl Worker {
    /// Validate the config — does NOT open a network connection yet.
    ///
    /// The first network attempt happens inside [`Worker::run`] so the
    /// retry policy applies uniformly to the initial connection and to
    /// every reconnect. Use [`Worker::connect_and_run`] when you want
    /// the same future to also do the first connect.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] only when the endpoint
    /// URI itself cannot be parsed.
    pub fn connect(config: WorkerConfig) -> Result<Self, ControlPlaneError> {
        // Validate the endpoint URI eagerly so misconfiguration surfaces
        // before the caller starts a retry loop.
        Endpoint::from_shared(config.endpoint.clone())
            .map_err(|e| ControlPlaneError::Transport(format!("invalid endpoint URI: {e}")))?;
        Ok(Self {
            config,
            state: Arc::new(WorkerState::new()),
            session_id: Arc::new(Mutex::new(None)),
        })
    }

    /// Latest session id, populated after each successful handshake.
    /// `None` before the first connection completes.
    pub async fn session_id(&self) -> Option<Uuid> {
        *self.session_id.lock().await
    }

    /// Signal the worker to stop. The current `run` future will finish
    /// after the in-flight assignments either complete or have their
    /// cancellation tokens fired (drop semantics — the futures may
    /// still race to completion / clean up).
    ///
    /// Idempotent: calling `shutdown` more than once has no extra
    /// effect. The signal is delivered synchronously via
    /// [`CancellationToken`]s; no `.await` is required.
    pub fn shutdown(&self) {
        self.state.shutdown.cancel();
        // Fire every per-run token so handler futures see cancellation.
        for entry in &self.state.running {
            entry.value().cancel();
        }
    }

    /// Drive the worker forever (or until shutdown / drain). On
    /// connection drop, reconnects under the configured
    /// [`RetryPolicy`].
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] when the retry policy is
    /// exhausted. Returns the underlying error if the initial endpoint
    /// is fundamentally broken.
    pub async fn run<H: AssignmentHandler>(self, handler: H) -> Result<(), ControlPlaneError> {
        let handler = Arc::new(handler);
        let mut attempt: u32 = 0;

        loop {
            if self.state.shutdown.is_cancelled() {
                return Ok(());
            }

            match self.run_one_session(handler.clone()).await {
                Ok(SessionOutcome::Disconnected) => {
                    attempt += 1;
                    if let Some(max) = self.config.retry.max_attempts
                        && attempt >= max
                    {
                        return Err(ControlPlaneError::Transport(format!(
                            "reconnect exhausted after {max} attempts"
                        )));
                    }
                    let backoff = self.config.retry.delay_for_attempt(attempt - 1);
                    tracing::warn!(
                        attempt,
                        backoff_ms = duration_ms_lossy(backoff),
                        "worker disconnected; reconnecting",
                    );
                    tokio::select! {
                        () = tokio::time::sleep(backoff) => {}
                        () = self.state.shutdown.cancelled() => return Ok(()),
                    }
                }
                Ok(SessionOutcome::Drained | SessionOutcome::Shutdown) => return Ok(()),
                Err(e) => {
                    attempt += 1;
                    if let Some(max) = self.config.retry.max_attempts
                        && attempt >= max
                    {
                        return Err(e);
                    }
                    let backoff = self.config.retry.delay_for_attempt(attempt - 1);
                    tracing::warn!(
                        attempt,
                        error = %e,
                        backoff_ms = duration_ms_lossy(backoff),
                        "worker connection error; retrying",
                    );
                    tokio::select! {
                        () = tokio::time::sleep(backoff) => {}
                        () = self.state.shutdown.cancelled() => return Ok(()),
                    }
                }
            }
        }
    }

    /// One end-to-end session: connect, handshake, pump inbound +
    /// heartbeat ticker until the stream ends or shutdown fires.
    async fn run_one_session<H: AssignmentHandler>(
        &self,
        handler: Arc<H>,
    ) -> Result<SessionOutcome, ControlPlaneError> {
        let HandshakeResult {
            mut inbound,
            outbound_tx,
            outbox,
        } = self.do_handshake().await?;

        // Install worker-local provider keys as the first link in the
        // global resolver cascade — exactly once across reconnects. The
        // `keys_installed` guard lives on the `Arc`'d `WorkerState`, so a
        // later reconnect's session sees it already set and skips the push
        // (the process-global chain would otherwise accumulate duplicates).
        if !self.config.provider_keys.is_empty()
            && !self.state.keys_installed.swap(true, Ordering::AcqRel)
        {
            blazen_llm::push_key_resolver(Arc::new(WorkerLocalKeys::new(Arc::new(
                self.config.provider_keys.clone(),
            ))));
        }

        // Bind the control-plane key client to THIS session's outbound
        // channel, then install it into the global cascade exactly once —
        // AFTER any worker-local resolver, so the order is
        // worker-local → control-plane → env terminal. On reconnect the
        // guard skips the re-push and only the outbound rebind happens, so
        // the process-global chain holds a single, stable CP resolver whose
        // cache survives the reconnect.
        self.state.cp_keys.rebind_outbound(outbound_tx.clone());
        if !self.state.cp_keys_installed.swap(true, Ordering::AcqRel) {
            blazen_llm::push_key_resolver(Arc::new(self.state.cp_keys.clone()));
        }

        let session_done = CancellationToken::new();
        let hb_handle = self.spawn_heartbeat_ticker(
            Arc::clone(&outbox),
            self.config.heartbeat_interval,
            session_done.clone(),
        );

        let drained = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let outcome = self
            .pump_inbound(&mut inbound, &handler, &outbox, &drained)
            .await;

        // ---- Tear down ----
        session_done.cancel();
        drop(outbound_tx); // close outbound channel so request stream ends
        let _ = hb_handle.await;

        // Abandon any still-running assignments. The server's
        // surrender_session puts the work back on pending; the worker
        // will be re-dispatched after reconnect (with whatever attempt
        // counter the queue tracks).
        for entry in &self.state.running {
            entry.value().cancel();
        }
        self.state.running.clear();
        self.state.in_flight.store(0, Ordering::Relaxed);

        Ok(outcome)
    }

    /// Open the bidi gRPC stream, send `Hello`, await `Welcome`. Returns
    /// the inbound stream plus the outbound channel + sink adapter.
    async fn do_handshake(&self) -> Result<HandshakeResult, ControlPlaneError> {
        let channel = self.build_channel().await?;
        let mut client = pb::blazen_control_plane_client::BlazenControlPlaneClient::new(channel);

        let (outbound_tx, outbound_rx) = mpsc::channel::<WorkerToServer>(64);
        let outbox = Arc::new(WorkerOutbox::new(outbound_tx.clone()));

        // Postcard-encode each WorkerToServer frame into a PostcardRequest.
        // Encoding a well-typed enum through postcard cannot fail in
        // practice; if it does, log and drop the frame (the inbound
        // stream will end shortly when the server doesn't see expected
        // heartbeats).
        let req_stream = ReceiverStream::new(outbound_rx).filter_map(|frame| async move {
            match postcard::to_allocvec(&frame) {
                Ok(payload) => Some(pb::PostcardRequest {
                    postcard_payload: payload,
                }),
                Err(e) => {
                    tracing::error!(error = %e, "failed to encode WorkerToServer; dropping frame");
                    None
                }
            }
        });

        let hello = WorkerToServer::Hello(WorkerHello {
            envelope_version: ENVELOPE_VERSION,
            node_id: self.config.node_id.clone(),
            capabilities: self
                .config
                .capabilities
                .iter()
                .map(protocol::CapabilityWire::from)
                .collect(),
            tags: self.config.tags.clone(),
            admission: (&self.config.admission).into(),
            supported_envelope_versions: self.config.envelope_versions.clone(),
            labels: self.config.labels.clone(),
            taints: self
                .config
                .taints
                .iter()
                .cloned()
                .map(protocol::WorkerTaintWire::from)
                .collect(),
            descriptors: self.config.descriptors.clone(),
            place: self.config.place.clone(),
        });
        outbound_tx.send(hello).await.map_err(|_| {
            ControlPlaneError::Transport("outbound channel closed before Hello".into())
        })?;

        let mut req = Request::new(req_stream);
        if let Some(bearer) = auth::bearer_metadata_value_with(self.config.bearer_token.as_deref())
        {
            let value = bearer
                .parse::<tonic::metadata::MetadataValue<_>>()
                .map_err(|e| {
                    ControlPlaneError::Unauthenticated(format!("invalid bearer header: {e}"))
                })?;
            req.metadata_mut().insert("authorization", value);
        }

        let response = client
            .worker_session(req)
            .await
            .map_err(|s| match s.code() {
                tonic::Code::Unauthenticated => {
                    ControlPlaneError::Unauthenticated(s.message().to_string())
                }
                _ => ControlPlaneError::Transport(s.to_string()),
            })?;
        let mut inbound = response.into_inner();

        let first = inbound.next().await.ok_or_else(|| {
            ControlPlaneError::Transport("server closed stream before Welcome".into())
        })?;
        let first = first.map_err(|s| ControlPlaneError::Transport(s.to_string()))?;
        let server_frame: ServerToWorker = postcard::from_bytes(&first.postcard_payload)?;
        let session_id = match server_frame {
            ServerToWorker::Welcome(w) => {
                tracing::info!(
                    session_id = %w.session_id,
                    negotiated_envelope_version = w.negotiated_envelope_version,
                    "worker session established",
                );
                w.session_id
            }
            ServerToWorker::Reject { reason } => {
                return Err(ControlPlaneError::Unauthenticated(reason));
            }
            other => {
                return Err(ControlPlaneError::Transport(format!(
                    "expected Welcome as first server frame, got {other:?}"
                )));
            }
        };
        *self.session_id.lock().await = Some(session_id);

        Ok(HandshakeResult {
            inbound,
            outbound_tx,
            outbox,
        })
    }

    /// Spawn the heartbeat ticker tied to this session.
    fn spawn_heartbeat_ticker(
        &self,
        outbox: Arc<WorkerOutbox>,
        interval: Duration,
        session_done: CancellationToken,
    ) -> JoinHandle<()> {
        let state = Arc::clone(&self.state);
        let admission = self.config.admission.clone();
        #[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
        let probe = self.config.probe_handle.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Skip the immediate first tick — the server already has
            // fresh state from the Hello.
            ticker.tick().await;
            loop {
                tokio::select! {
                    () = session_done.cancelled() => return,
                    _ = ticker.tick() => {}
                }
                let in_flight = state.in_flight.load(Ordering::Relaxed);
                #[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
                let snapshot = synthesize_admission_snapshot(&admission, in_flight, probe.as_ref());
                #[cfg(any(target_os = "wasi", target_arch = "wasm32"))]
                let snapshot = synthesize_admission_snapshot(&admission, in_flight);
                if let Err(e) = outbox.heartbeat(in_flight, Some(snapshot)).await {
                    tracing::debug!(error = %e, "heartbeat send failed; session ending");
                    return;
                }
            }
        })
    }

    /// Pump server → worker frames until shutdown / disconnect / drain.
    async fn pump_inbound<H: AssignmentHandler>(
        &self,
        inbound: &mut tonic::Streaming<pb::PostcardResponse>,
        handler: &Arc<H>,
        outbox: &Arc<WorkerOutbox>,
        drained: &Arc<std::sync::atomic::AtomicBool>,
    ) -> SessionOutcome {
        loop {
            tokio::select! {
                () = self.state.shutdown.cancelled() => {
                    return SessionOutcome::Shutdown;
                }
                msg = inbound.next() => {
                    let Some(msg) = msg else { return SessionOutcome::Disconnected; };
                    let frame = match msg {
                        Ok(resp) => resp,
                        Err(s) => {
                            tracing::warn!(error = %s, "inbound stream error; disconnecting");
                            return SessionOutcome::Disconnected;
                        }
                    };
                    let server_frame: ServerToWorker =
                        match postcard::from_bytes(&frame.postcard_payload) {
                            Ok(f) => f,
                            Err(e) => {
                                tracing::warn!(error = %e, "discarding malformed ServerToWorker");
                                continue;
                            }
                        };
                    self.dispatch_server_frame(
                        server_frame,
                        Arc::clone(handler),
                        Arc::clone(outbox),
                        Arc::clone(drained),
                    )
                    .await;
                    if drained.load(Ordering::Relaxed)
                        && self.state.in_flight.load(Ordering::Relaxed) == 0
                    {
                        return SessionOutcome::Drained;
                    }
                }
            }
        }
    }

    /// Apply TLS / keep-alive settings to the Endpoint and open the
    /// channel.
    async fn build_channel(&self) -> Result<Channel, ControlPlaneError> {
        let mut endpoint = Endpoint::from_shared(self.config.endpoint.clone())
            .map_err(|e| ControlPlaneError::Transport(format!("invalid endpoint URI: {e}")))?
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_keep_alive_interval(Duration::from_secs(20))
            .keep_alive_while_idle(true);
        if let Some(tls) = self.config.tls.clone() {
            endpoint = endpoint
                .tls_config(tls)
                .map_err(|e| ControlPlaneError::Tls(e.to_string()))?;
        }
        endpoint
            .connect()
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("connect: {e}")))
    }

    /// Dispatch one inbound server frame.
    async fn dispatch_server_frame<H: AssignmentHandler>(
        &self,
        frame: ServerToWorker,
        handler: Arc<H>,
        outbox: Arc<WorkerOutbox>,
        drained: Arc<std::sync::atomic::AtomicBool>,
    ) {
        match frame {
            ServerToWorker::Welcome(_) | ServerToWorker::Reject { .. } => {
                tracing::warn!("ignoring late Welcome/Reject mid-session");
            }
            ServerToWorker::Assignment(a) => {
                // P4: ensure the LLM resolver chain knows the place this
                // worker serves, then pre-warm the control-plane key cache
                // so the (synchronous) resolver hit during the assignment
                // finds keys already fetched. The worker cannot read the
                // assignment's declared providers, so pre-warm from the
                // configured `prewarm_providers` list. Best-effort: a
                // pre-warm failure (no session, timeout) is non-fatal — the
                // resolver simply falls through to env.
                self.prewarm_keys_for_assignment().await;
                self.spawn_assignment(a, Arc::clone(&handler), Arc::clone(&outbox));
            }
            ServerToWorker::Offer(o) => {
                let decision = handler.evaluate_offer(&o).await;
                let frame = WorkerToServer::OfferDecision(OfferDecision {
                    envelope_version: ENVELOPE_VERSION,
                    run_id: o.assignment.run_id,
                    decision: decision.clone(),
                });
                if let Err(e) = outbox.send(frame).await {
                    tracing::warn!(error = %e, "failed to send OfferDecision");
                    return;
                }
                if matches!(decision, OfferOutcome::Claim) {
                    self.spawn_assignment(o.assignment, handler, outbox);
                }
            }
            ServerToWorker::Cancel(c) => {
                if let Some((_, token)) = self.state.running.remove(&c.run_id) {
                    token.cancel();
                }
                handler.on_cancel(c.run_id).await;
                // The assignment task is responsible for sending a
                // Cancelled result when its future returns; we don't
                // duplicate that here.
            }
            ServerToWorker::Drain(d) => {
                drained.store(true, Ordering::Relaxed);
                if d.immediate {
                    for entry in &self.state.running {
                        entry.value().cancel();
                    }
                }
                handler.on_drain(d.immediate).await;
            }
            ServerToWorker::InputResponse(r) => {
                if let Some((_, tx)) = self.state.pending_inputs.remove(&r.request_id) {
                    let value = serde_json::from_slice(&r.response_json).unwrap_or(Value::Null);
                    let _ = tx.send(value);
                } else {
                    tracing::debug!(
                        request_id = %r.request_id,
                        run_id = %r.run_id,
                        "received InputResponse with no pending request (late/duplicate?)",
                    );
                }
            }
            ServerToWorker::KeyResponse(resp) => {
                // P4: route the per-place key response into the CP key
                // client, which caches it (by the pending request's
                // provider) and unblocks the matching `pre_warm` waiter.
                // The key value is never logged (the frame redacts it).
                self.state.cp_keys.on_key_response(&resp);
            }
        }
    }

    /// Pre-warm the control-plane key cache for this worker's configured
    /// `prewarm_providers` ahead of running an assignment.
    ///
    /// Sets the LLM resolver's current place to the worker's configured
    /// place (so the cached key is attributed to the right tenant) and
    /// fetches each configured provider's key over the live session. A
    /// no-op when no providers are configured. Failures are swallowed —
    /// the resolver chain falls through to env if a key can't be warmed.
    async fn prewarm_keys_for_assignment(&self) {
        if self.config.prewarm_providers.is_empty() {
            return;
        }
        // The worker serves a single place for its whole lifetime; set it
        // on the process-global resolver scope so the CP resolver (and any
        // other place-aware resolver) sees it.
        blazen_llm::set_current_place(self.config.place.clone());

        let place = self.config.place.as_deref().unwrap_or_default();
        if let Err(e) = self
            .state
            .cp_keys
            .pre_warm(
                place,
                &self.config.prewarm_providers,
                CP_KEY_PREWARM_TIMEOUT,
            )
            .await
        {
            // Non-fatal: no bound session / closed channel. The resolver
            // falls through to env. Never log key material (there is none
            // on this path).
            tracing::debug!(error = %e, "control-plane key pre-warm failed; falling through to env");
        }
    }

    /// Spawn a per-assignment task. Honors `deadline_ms` and the
    /// per-run cancellation token; sends the terminal result back over
    /// the outbox.
    fn spawn_assignment<H: AssignmentHandler>(
        &self,
        assignment: protocol::Assignment,
        handler: Arc<H>,
        outbox: Arc<WorkerOutbox>,
    ) {
        let run_id = assignment.run_id;
        let token = CancellationToken::new();
        self.state.running.insert(run_id, token.clone());
        self.state.in_flight.fetch_add(1, Ordering::Relaxed);

        let state = Arc::clone(&self.state);
        let deadline = assignment.deadline_ms.map(Duration::from_millis);
        let outbox_for_ctx = Arc::clone(&outbox);

        tokio::spawn(async move {
            let ctx = AssignmentContext {
                sink: outbox_for_ctx,
                run_id,
                cancel: token.clone(),
                pending: Arc::clone(&state.pending_inputs),
            };
            let handler_fut = handler.handle(assignment, ctx);

            // We need three concurrent paths: handler completion, the
            // per-run cancel token firing, and an optional deadline.
            // `biased;` makes the select! evaluate arms in source order
            // — important because a cancelled handler may return Ok in
            // the same tick the cancel fires, and we want the Cancel
            // arm to win that race deterministically.
            let result: Result<Result<Value, AssignmentFailure>, AssignmentOutcomeError> =
                if let Some(d) = deadline {
                    tokio::select! {
                        biased;
                        () = token.cancelled() => Err(AssignmentOutcomeError::Cancelled),
                        () = tokio::time::sleep(d) => Err(AssignmentOutcomeError::DeadlineExceeded),
                        res = handler_fut => Ok(res),
                    }
                } else {
                    tokio::select! {
                        biased;
                        () = token.cancelled() => Err(AssignmentOutcomeError::Cancelled),
                        res = handler_fut => Ok(res),
                    }
                };

            let frame = match result {
                Ok(Ok(value)) => {
                    let output_json = serde_json::to_vec(&value).unwrap_or_else(|e| {
                        tracing::error!(
                            error = %e,
                            %run_id,
                            "failed to JSON-encode assignment output; reporting empty",
                        );
                        Vec::new()
                    });
                    WorkerToServer::Result(AssignmentResult {
                        envelope_version: ENVELOPE_VERSION,
                        run_id,
                        output_json,
                        status: AssignmentStatus::Completed,
                        error: None,
                    })
                }
                Ok(Err(failure)) => WorkerToServer::Result(AssignmentResult {
                    envelope_version: ENVELOPE_VERSION,
                    run_id,
                    output_json: Vec::new(),
                    status: AssignmentStatus::Failed,
                    error: Some(failure.error),
                }),
                Err(AssignmentOutcomeError::Cancelled) => {
                    WorkerToServer::Result(AssignmentResult {
                        envelope_version: ENVELOPE_VERSION,
                        run_id,
                        output_json: Vec::new(),
                        status: AssignmentStatus::Cancelled,
                        error: None,
                    })
                }
                Err(AssignmentOutcomeError::DeadlineExceeded) => {
                    WorkerToServer::Result(AssignmentResult {
                        envelope_version: ENVELOPE_VERSION,
                        run_id,
                        output_json: Vec::new(),
                        status: AssignmentStatus::Failed,
                        error: Some("deadline exceeded".into()),
                    })
                }
            };

            if let Err(e) = outbox.send(frame).await {
                tracing::warn!(error = %e, %run_id, "failed to send AssignmentResult");
            }

            state.running.remove(&run_id);
            state.in_flight.fetch_sub(1, Ordering::Relaxed);
        });
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Terminal-state input to assignment result framing. Local enum keeps
/// the outcome-collation logic exhaustive without leaking through the
/// public surface.
enum AssignmentOutcomeError {
    Cancelled,
    DeadlineExceeded,
}

fn now_ms() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

/// Best-effort `u64` ms from a [`Duration`]. Saturates rather than
/// truncates on overflow — keeps the log line honest when a caller
/// supplies a wildly large backoff.
fn duration_ms_lossy(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Synthesise a best-effort [`AdmissionSnapshot`] for the heartbeat
/// from the admission mode, current in-flight count, and (optionally) a
/// probe snapshot. When `probe` is `Some`, `vram_free_mb` is sourced from
/// the latest probe; otherwise it is left `None` to preserve the
/// pre-probe behavior.
#[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
fn synthesize_admission_snapshot(
    admission: &AdmissionMode,
    in_flight: u32,
    probe: Option<&blazen_resource_probe::ProbeHandle>,
) -> AdmissionSnapshot {
    let vram_free_mb = probe.and_then(|p| {
        let snap = p.latest();
        if snap.free_vram_is_complete() {
            Some(snap.total_free_vram_mb())
        } else {
            // Snapshot is incomplete — better to report None than a
            // known-undercounted value.
            None
        }
    });

    build_admission_snapshot(admission, in_flight, vram_free_mb)
}

/// wasm/wasi variant: the host-only resource probe is not in the dependency
/// graph, so `vram_free_mb` is always `None` (the pre-probe behavior).
#[cfg(any(target_os = "wasi", target_arch = "wasm32"))]
fn synthesize_admission_snapshot(admission: &AdmissionMode, in_flight: u32) -> AdmissionSnapshot {
    build_admission_snapshot(admission, in_flight, None)
}

/// Shared snapshot assembly, independent of where `vram_free_mb` was sourced.
fn build_admission_snapshot(
    admission: &AdmissionMode,
    in_flight: u32,
    vram_free_mb: Option<u64>,
) -> AdmissionSnapshot {
    match admission {
        AdmissionMode::Fixed { max_in_flight } => AdmissionSnapshot {
            capacity_score: fixed_capacity_score(*max_in_flight, in_flight),
            model_residency: std::collections::BTreeSet::new(),
            vram_free_mb,
            in_flight_vram_mb: 0,
        },
        AdmissionMode::VramBudget { .. } | AdmissionMode::Reactive => AdmissionSnapshot {
            capacity_score: 1.0,
            model_residency: std::collections::BTreeSet::new(),
            vram_free_mb,
            in_flight_vram_mb: 0,
        },
    }
}

/// 0.0 = saturated, 1.0 = fully idle.
#[allow(clippy::cast_precision_loss)] // u16-bounded inputs, ratio is a coarse rank.
fn fixed_capacity_score(max_in_flight: u32, in_flight: u32) -> f32 {
    if max_in_flight == 0 {
        return 0.0;
    }
    let used = f32::from(u16::try_from(in_flight).unwrap_or(u16::MAX));
    let cap = f32::from(u16::try_from(max_in_flight).unwrap_or(u16::MAX));
    (1.0 - (used / cap)).clamp(0.0, 1.0)
}

// ===========================================================================
// Worker-local provider keys
// ===========================================================================

/// A [`blazen_llm::KeyResolver`] backed by a worker's own provider keys.
///
/// Installed as the first link in the global cascade at worker start, so
/// worker-local keys win over control-plane-shared and environment keys.
/// Worker-local keys are NOT place-scoped — a worker serves whatever place
/// the session declares — so the `place` argument is ignored.
///
/// The key map is shared via `Arc` so installing the resolver does not
/// re-copy the keys. The key value is never logged.
struct WorkerLocalKeys {
    keys: Arc<BTreeMap<String, String>>,
}

impl WorkerLocalKeys {
    fn new(keys: Arc<BTreeMap<String, String>>) -> Self {
        Self { keys }
    }
}

impl blazen_llm::KeyResolver for WorkerLocalKeys {
    fn resolve(&self, provider: &str, _place: Option<&str>) -> Option<String> {
        self.keys.get(provider).cloned()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy_delay_grows_exponentially() {
        let p = RetryPolicy {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            multiplier: 2.0,
            max_attempts: None,
        };
        assert_eq!(p.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(p.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(p.delay_for_attempt(2), Duration::from_millis(400));
        // Caps at max_backoff.
        assert!(p.delay_for_attempt(100) <= Duration::from_secs(10));
    }

    #[test]
    fn retry_policy_none_caps_at_one_attempt() {
        let p = RetryPolicy::none();
        assert_eq!(p.max_attempts, Some(1));
    }

    #[test]
    fn worker_config_with_capability_appends() {
        let cfg = WorkerConfig::new("http://localhost:7445", "node-a").with_capability(
            WorkerCapability {
                kind: "workflow:hello".into(),
                version: 1,
            },
        );
        assert_eq!(cfg.capabilities.len(), 1);
        assert_eq!(cfg.capabilities[0].kind, "workflow:hello");
    }

    #[test]
    fn worker_config_with_tag_inserts() {
        let cfg =
            WorkerConfig::new("http://localhost:7445", "node-a").with_tag("region", "us-west");
        assert_eq!(cfg.tags.get("region").map(String::as_str), Some("us-west"));
    }

    // ---- Worker-local provider keys (P2) ----

    use blazen_llm::KeyResolver as _;

    /// The process-global resolver chain is shared across the whole test
    /// binary, so any test that installs into it must serialize and tear
    /// down. Pure-builder tests below don't touch the global chain and so
    /// don't need the lock.
    static KEYS_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn worker_local_keys_resolves_known_provider() {
        let mut map = BTreeMap::new();
        map.insert("fal".to_string(), "fal-key".to_string());
        let r = WorkerLocalKeys::new(Arc::new(map));
        assert_eq!(r.resolve("fal", None).as_deref(), Some("fal-key"));
        // Place is ignored (worker-local keys are not place-scoped).
        assert_eq!(r.resolve("fal", Some("acme")).as_deref(), Some("fal-key"));
    }

    #[test]
    fn worker_local_keys_missing_provider_is_none() {
        let mut map = BTreeMap::new();
        map.insert("fal".to_string(), "fal-key".to_string());
        let r = WorkerLocalKeys::new(Arc::new(map));
        assert!(r.resolve("openai", None).is_none());
    }

    #[test]
    fn worker_config_with_provider_key_populates_map() {
        let cfg = WorkerConfig::new("http://localhost:7445", "node-a")
            .with_provider_key("fal", "k")
            .with_provider_key("openai", "k2");
        assert_eq!(cfg.provider_keys.get("fal").map(String::as_str), Some("k"));
        assert_eq!(
            cfg.provider_keys.get("openai").map(String::as_str),
            Some("k2")
        );
        assert_eq!(cfg.provider_keys.len(), 2);
    }

    #[test]
    fn worker_config_provider_key_chains_with_other_builders() {
        let cfg = WorkerConfig::new("http://localhost:7445", "node-a")
            .with_capability(WorkerCapability {
                kind: "workflow:hello".into(),
                version: 1,
            })
            .with_tag("region", "us-west")
            .with_provider_key("fal", "k");
        assert_eq!(cfg.capabilities.len(), 1);
        assert_eq!(cfg.tags.get("region").map(String::as_str), Some("us-west"));
        assert_eq!(cfg.provider_keys.get("fal").map(String::as_str), Some("k"));
    }

    #[test]
    fn worker_local_keys_install_into_global_cascade() {
        let _g = KEYS_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        blazen_llm::clear_key_resolvers();

        let mut map = BTreeMap::new();
        map.insert("fal".to_string(), "worker-fal-key".to_string());
        blazen_llm::push_key_resolver(Arc::new(WorkerLocalKeys::new(Arc::new(map))));

        // `fal` has an env var (FAL_KEY); the worker-local resolver supplies
        // the key without one, proving it sits ahead of the env terminal.
        assert_eq!(
            blazen_llm::resolve_api_key("fal", None).unwrap(),
            "worker-fal-key"
        );

        blazen_llm::clear_key_resolvers();
    }

    #[test]
    fn synthesize_admission_snapshot_fixed_reports_capacity() {
        let snap =
            synthesize_admission_snapshot(&AdmissionMode::Fixed { max_in_flight: 4 }, 1, None);
        assert!((snap.capacity_score - 0.75).abs() < 0.001);
    }

    #[test]
    fn synthesize_admission_snapshot_fixed_reports_zero_at_capacity() {
        let snap =
            synthesize_admission_snapshot(&AdmissionMode::Fixed { max_in_flight: 4 }, 4, None);
        assert!(snap.capacity_score < f32::EPSILON);
    }

    #[test]
    fn synthesize_admission_snapshot_zero_cap_is_saturated() {
        let snap =
            synthesize_admission_snapshot(&AdmissionMode::Fixed { max_in_flight: 0 }, 0, None);
        assert!(snap.capacity_score < f32::EPSILON);
    }

    #[test]
    fn synthesize_admission_snapshot_without_probe_leaves_vram_none() {
        let snap =
            synthesize_admission_snapshot(&AdmissionMode::Fixed { max_in_flight: 4 }, 1, None);
        assert!(snap.vram_free_mb.is_none());
    }

    #[tokio::test]
    async fn synthesize_admission_snapshot_with_probe_sources_vram_when_complete() {
        // Use the probe's real probe_once() (the test host's CPU/RAM are
        // always probed; GPUs may or may not be present depending on
        // build target). What we assert is the contract: if the snapshot
        // is `free_vram_is_complete`, the result reflects the sum; if
        // not, the result is None (rather than a partial under-count).
        let handle =
            blazen_resource_probe::Probe::spawn(blazen_resource_probe::ProbeConfig::default())
                .await;
        let snap = synthesize_admission_snapshot(
            &AdmissionMode::VramBudget {
                max_vram_mb: 24_000,
            },
            0,
            Some(&handle),
        );
        let probed = handle.latest();
        if probed.free_vram_is_complete() {
            assert_eq!(snap.vram_free_mb, Some(probed.total_free_vram_mb()));
        } else {
            assert!(snap.vram_free_mb.is_none());
        }
    }

    #[test]
    fn worker_connect_rejects_bad_endpoint() {
        let cfg = WorkerConfig::new("not a uri", "node-a");
        // `Worker` itself is not `Debug` (it holds task handles and
        // streams), so don't reach for `unwrap_err`; pattern-match the
        // result instead.
        match Worker::connect(cfg) {
            Ok(_) => panic!("expected bad endpoint URI to be rejected"),
            Err(e) => assert!(
                matches!(e, ControlPlaneError::Transport(_)),
                "expected Transport error, got {e:?}",
            ),
        }
    }

    #[test]
    fn worker_config_with_mtls_threads_tls_config_through() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Dummy PEM content — the loaders just read bytes and hand them
        // to `tonic::Identity::from_pem` / `tonic::Certificate::from_pem`,
        // which defer real validation to handshake time. This mirrors
        // the test in `blazen_peer::tls::tests::load_tls_with_dummy_pem_files`.
        let mut cert = NamedTempFile::new().unwrap();
        let mut key = NamedTempFile::new().unwrap();
        let mut ca = NamedTempFile::new().unwrap();
        write!(
            cert,
            "-----BEGIN CERTIFICATE-----\nZHVtbXk=\n-----END CERTIFICATE-----",
        )
        .unwrap();
        write!(
            key,
            "-----BEGIN PRIVATE KEY-----\nZHVtbXk=\n-----END PRIVATE KEY-----",
        )
        .unwrap();
        write!(
            ca,
            "-----BEGIN CERTIFICATE-----\nZHVtbXk=\n-----END CERTIFICATE-----",
        )
        .unwrap();

        let cfg = WorkerConfig::new("https://localhost:7445", "node-tls")
            .with_mtls(cert.path(), key.path(), ca.path())
            .expect("with_mtls accepts the loaded ClientTlsConfig");
        assert!(
            cfg.tls.is_some(),
            "with_mtls must populate the tls field so connect() applies it",
        );
    }

    #[test]
    fn worker_config_with_mtls_errors_on_missing_pem() {
        let cfg = WorkerConfig::new("https://localhost:7445", "node-tls");
        let missing = std::path::Path::new("/nonexistent/blazen-test/missing.pem");
        // `WorkerConfig` is not `Debug`-printable (it holds optional
        // tonic types that we don't want to leak into derives), so
        // pattern-match the result rather than calling `unwrap_err`.
        match cfg.with_mtls(missing, missing, missing) {
            Ok(_) => panic!("expected with_mtls to error on missing PEM files"),
            Err(e) => assert!(
                matches!(e, ControlPlaneError::Tls(_)),
                "expected Tls error variant, got {e:?}",
            ),
        }
    }

    #[test]
    fn duration_ms_lossy_saturates_overflow() {
        // Anything past u64::MAX milliseconds saturates.
        assert_eq!(duration_ms_lossy(Duration::ZERO), 0);
        assert_eq!(duration_ms_lossy(Duration::from_millis(1234)), 1234);
        // The max `Duration` is roughly 5.85e11 years in ms, which
        // overflows u64; saturate.
        let huge = Duration::new(u64::MAX, 0);
        assert_eq!(duration_ms_lossy(huge), u64::MAX);
    }
}
