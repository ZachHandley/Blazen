//! Control-plane surface for the UniFFI bindings.
//!
//! Wraps [`blazen_controlplane::Client`] (orchestrator) and
//! [`blazen_controlplane::Worker`] (gRPC bidi worker session) so foreign
//! callers can submit workflows, observe runs, list workers, and drive
//! worker-side assignment loops without re-implementing the wire
//! protocol in every binding.
//!
//! ## Design vs. upstream
//!
//! Upstream `Client::submit_workflow` returns a
//! [`blazen_core::distributed::RunStateSnapshot`] whose `output` is a
//! [`serde_json::Value`]. UniFFI does not handle nested JSON values
//! cleanly across four foreign runtimes, so this layer flattens the
//! snapshot's `output` and `error` payloads to JSON-encoded strings via
//! [`ControlPlaneRunStateSnapshot`] (mirroring how
//! [`crate::peer`] flattens `SubWorkflowResponse`).
//!
//! Upstream [`blazen_controlplane::AssignmentHandler`] is `async_trait`,
//! takes the full [`blazen_controlplane::protocol::Assignment`], and is
//! invoked with an [`blazen_controlplane::AssignmentContext`] handle.
//! That shape leaks too much wire-protocol detail across the foreign
//! boundary; this module surfaces a simpler callback contract —
//! [`ControlPlaneAssignmentHandler`] receives `run_id`, `workflow_name`,
//! and `input_json` strings and returns either an `output_json` string
//! (success) or a stringly-typed error (failure). The adapter
//! [`AssignmentHandlerAdapter`] bridges this back into the
//! `async_trait` shape the Rust [`Worker`] expects.
//!
//! ## Async story across bindings
//!
//! Methods exposed under `#[uniffi::export(async_runtime = "tokio")]`
//! become language-native async: Swift `async`, Kotlin `suspend fun`,
//! Go blocking (the host's goroutine is the unit of async), Ruby
//! blocking. A blocking constructor sibling is provided for hosts that
//! cannot drive an async constructor (Go `main`, Ruby scripts).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use blazen_controlplane::protocol::Assignment;
use blazen_controlplane::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Client as CoreClient,
    Worker as CoreWorker, WorkerConfig,
};
use blazen_core::distributed::{
    AdmissionMode as CoreAdmissionMode, RunEvent as CoreRunEvent,
    RunStateSnapshot as CoreRunStateSnapshot, RunStatus as CoreRunStatus,
    SubmitWorkflowRequest as CoreSubmitWorkflowRequest, WorkerCapability as CoreWorkerCapability,
    WorkerInfo as CoreWorkerInfo,
};

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

// ===========================================================================
// Records
// ===========================================================================

/// Typed capability a worker advertises to the control plane.
///
/// `kind` follows the convention `"workflow:<name>"` /
/// `"step:<name>"` / `"provider:<id>"` / `"tag:<key>=<value>"`.
/// `version` lets the control plane gate routing on schema changes.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ControlPlaneWorkerCapability {
    pub kind: String,
    pub version: u32,
}

impl From<&ControlPlaneWorkerCapability> for CoreWorkerCapability {
    fn from(cap: &ControlPlaneWorkerCapability) -> Self {
        Self {
            kind: cap.kind.clone(),
            version: cap.version,
        }
    }
}

impl From<&CoreWorkerCapability> for ControlPlaneWorkerCapability {
    fn from(cap: &CoreWorkerCapability) -> Self {
        Self {
            kind: cap.kind.clone(),
            version: cap.version,
        }
    }
}

/// How a worker declares its admission policy to the control plane.
///
/// Carries the union of fields for the three flavours; consumers should
/// honour the discriminator in [`ControlPlaneAdmission::mode`].
#[derive(Debug, Clone, uniffi::Enum)]
pub enum ControlPlaneAdmissionMode {
    /// Hard concurrency cap.
    Fixed,
    /// Worker self-decides via offer/claim/decline.
    Reactive,
    /// VRAM-sum cap.
    VramBudget,
}

/// Bundle of admission-policy fields for a worker.
///
/// `max_in_flight` is meaningful when `mode == Fixed`, `total_mb` when
/// `mode == VramBudget`; both fields are ignored when `mode == Reactive`.
/// Either may be omitted to fall back to upstream defaults
/// (`Fixed { max_in_flight: 1 }`, `VramBudget { max_vram_mb: 0 }`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct ControlPlaneAdmission {
    pub mode: ControlPlaneAdmissionMode,
    pub max_in_flight: Option<u32>,
    pub total_mb: Option<u32>,
}

impl From<&ControlPlaneAdmission> for CoreAdmissionMode {
    fn from(a: &ControlPlaneAdmission) -> Self {
        match a.mode {
            ControlPlaneAdmissionMode::Fixed => Self::Fixed {
                max_in_flight: a.max_in_flight.unwrap_or(1),
            },
            ControlPlaneAdmissionMode::Reactive => Self::Reactive,
            ControlPlaneAdmissionMode::VramBudget => Self::VramBudget {
                max_vram_mb: u64::from(a.total_mb.unwrap_or(0)),
            },
        }
    }
}

impl From<&CoreAdmissionMode> for ControlPlaneAdmission {
    fn from(mode: &CoreAdmissionMode) -> Self {
        match *mode {
            CoreAdmissionMode::Fixed { max_in_flight } => Self {
                mode: ControlPlaneAdmissionMode::Fixed,
                max_in_flight: Some(max_in_flight),
                total_mb: None,
            },
            CoreAdmissionMode::Reactive => Self {
                mode: ControlPlaneAdmissionMode::Reactive,
                max_in_flight: None,
                total_mb: None,
            },
            CoreAdmissionMode::VramBudget { max_vram_mb } => Self {
                mode: ControlPlaneAdmissionMode::VramBudget,
                max_in_flight: None,
                total_mb: Some(u32::try_from(max_vram_mb).unwrap_or(u32::MAX)),
            },
        }
    }
}

/// Lifecycle state of a workflow run, mirrored across the UniFFI
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ControlPlaneRunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl From<CoreRunStatus> for ControlPlaneRunStatus {
    fn from(status: CoreRunStatus) -> Self {
        match status {
            CoreRunStatus::Pending => Self::Pending,
            CoreRunStatus::Running => Self::Running,
            CoreRunStatus::Completed => Self::Completed,
            CoreRunStatus::Failed => Self::Failed,
            CoreRunStatus::Cancelled => Self::Cancelled,
        }
    }
}

impl From<ControlPlaneRunStatus> for CoreRunStatus {
    fn from(status: ControlPlaneRunStatus) -> Self {
        match status {
            ControlPlaneRunStatus::Pending => Self::Pending,
            ControlPlaneRunStatus::Running => Self::Running,
            ControlPlaneRunStatus::Completed => Self::Completed,
            ControlPlaneRunStatus::Failed => Self::Failed,
            ControlPlaneRunStatus::Cancelled => Self::Cancelled,
        }
    }
}

/// Foreign-facing snapshot of a workflow run.
///
/// `run_id` is the canonical UUID string (`"550e8400-e29b-41d4-a716-446655440000"`);
/// `output_json` and `error` are flattened from the upstream snapshot's
/// `output: Option<serde_json::Value>` / `error: Option<String>` fields.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ControlPlaneRunStateSnapshot {
    pub run_id: String,
    pub status: ControlPlaneRunStatus,
    pub started_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub assigned_to: Option<String>,
    pub last_event_at_ms: Option<u64>,
    pub output_json: Option<String>,
    pub error: Option<String>,
}

impl From<CoreRunStateSnapshot> for ControlPlaneRunStateSnapshot {
    fn from(s: CoreRunStateSnapshot) -> Self {
        Self {
            run_id: s.run_id.to_string(),
            status: s.status.into(),
            started_at_ms: s.started_at_ms,
            completed_at_ms: s.completed_at_ms,
            assigned_to: s.assigned_to,
            last_event_at_ms: s.last_event_at_ms,
            output_json: s.output.as_ref().map(ToString::to_string),
            error: s.error,
        }
    }
}

/// Foreign-facing run event.
///
/// `data_json` is the upstream `data: serde_json::Value` serialized to a
/// JSON string for transport across the UniFFI boundary.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ControlPlaneRunEvent {
    pub run_id: String,
    pub event_type: String,
    pub data_json: String,
    pub timestamp_ms: u64,
}

impl From<CoreRunEvent> for ControlPlaneRunEvent {
    fn from(e: CoreRunEvent) -> Self {
        Self {
            run_id: e.run_id.to_string(),
            event_type: e.event_type,
            data_json: e.data.to_string(),
            timestamp_ms: e.timestamp_ms,
        }
    }
}

/// Foreign-facing summary of a connected worker.
///
/// Upstream [`CoreWorkerInfo`] carries an `admission_snapshot` and an
/// `admission` field; this surface omits the snapshot (foreign callers
/// who need it should query the control plane directly) and flattens
/// `tags` from a `BTreeMap` to a [`HashMap`] for UniFFI compatibility.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ControlPlaneWorkerInfo {
    pub node_id: String,
    pub capabilities: Vec<ControlPlaneWorkerCapability>,
    pub tags: HashMap<String, String>,
    pub in_flight: u32,
    pub connected_at_ms: u64,
}

impl From<&CoreWorkerInfo> for ControlPlaneWorkerInfo {
    fn from(w: &CoreWorkerInfo) -> Self {
        Self {
            node_id: w.node_id.clone(),
            capabilities: w.capabilities.iter().map(Into::into).collect(),
            tags: w.tags.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            in_flight: w.in_flight,
            connected_at_ms: w.connected_at_ms,
        }
    }
}

/// Foreign-facing workflow submission request.
///
/// Mirrors [`CoreSubmitWorkflowRequest`] except `input_json` carries the
/// initial input as a JSON-encoded string and `resource_hint` is omitted
/// (the UniFFI surface today targets Fixed/Reactive admission only;
/// VramBudget callers should target the native crate directly).
#[derive(Debug, Clone, uniffi::Record)]
pub struct ControlPlaneSubmitRequest {
    pub workflow_name: String,
    pub input_json: String,
    pub workflow_version: Option<u32>,
    pub required_tags: Vec<String>,
    pub idempotency_key: Option<String>,
    pub deadline_ms: Option<u64>,
    pub wait_for_worker: bool,
}

impl ControlPlaneSubmitRequest {
    fn into_core(self) -> Result<CoreSubmitWorkflowRequest, BlazenError> {
        let input: Value = serde_json::from_str(&self.input_json)?;
        Ok(CoreSubmitWorkflowRequest {
            workflow_name: self.workflow_name,
            workflow_version: self.workflow_version,
            input,
            required_tags: self.required_tags,
            idempotency_key: self.idempotency_key,
            deadline_ms: self.deadline_ms,
            wait_for_worker: self.wait_for_worker,
            resource_hint: None,
        })
    }
}

// ===========================================================================
// AssignmentContext handle
// ===========================================================================

/// Foreign-facing handle to the per-assignment
/// [`blazen_controlplane::AssignmentContext`].
///
/// Handed to [`ControlPlaneAssignmentHandler::handle`] so the foreign
/// handler can emit progress events and request human/operator input
/// mid-assignment. All methods are **synchronous** on the foreign side:
/// they block the calling (handler) thread on the shared Tokio runtime
/// until the underlying async operation resolves. This is safe because
/// the foreign `handle` callback runs on a [`tokio::task::spawn_blocking`]
/// thread — outside the async executor — so `runtime().block_on` does not
/// re-enter a running runtime.
#[derive(uniffi::Object)]
pub struct AssignmentContextHandle {
    inner: Arc<AssignmentContext>,
}

#[uniffi::export]
impl AssignmentContextHandle {
    /// The run identifier this assignment belongs to (UUID string).
    #[must_use]
    pub fn run_id(&self) -> String {
        self.inner.run_id().to_string()
    }

    /// Emit a non-terminal progress event from the running assignment.
    ///
    /// `data_json` is the event payload as a JSON-encoded string; an empty
    /// string is treated as JSON `null`.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `data_json` is non-empty but
    /// not valid JSON; [`BlazenError::Peer`] (`kind = "ControlPlace*"`) if
    /// the worker's outbound channel is closed.
    pub fn emit_event(&self, event_type: String, data_json: String) -> BlazenResult<()> {
        let data: serde_json::Value = if data_json.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&data_json).map_err(|e| BlazenError::Validation {
                message: format!("invalid data_json: {e}"),
            })?
        };
        runtime()
            .block_on(self.inner.emit_event(&event_type, data))
            .map_err(BlazenError::from)
    }

    /// Raise an `input.request` and block until the orchestrator answers
    /// (or the assignment is cancelled / `timeout_ms` elapses).
    ///
    /// `metadata_json` is an arbitrary JSON-encoded payload attached to
    /// the request; an empty string is treated as JSON `null`. Returns the
    /// orchestrator's answer as a JSON-encoded string.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `metadata_json` is non-empty
    /// but not valid JSON, or if the answer cannot be re-encoded;
    /// [`BlazenError::Peer`] for transport / cancellation / timeout.
    pub fn request_input(
        &self,
        prompt: String,
        metadata_json: String,
        timeout_ms: Option<u64>,
    ) -> BlazenResult<String> {
        let metadata: serde_json::Value = if metadata_json.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&metadata_json).map_err(|e| BlazenError::Validation {
                message: format!("invalid metadata_json: {e}"),
            })?
        };
        let resp = runtime()
            .block_on(self.inner.request_input(&prompt, metadata, timeout_ms))
            .map_err(BlazenError::from)?;
        serde_json::to_string(&resp).map_err(|e| BlazenError::Validation {
            message: format!("encode response: {e}"),
        })
    }
}

// ===========================================================================
// Callback interfaces
// ===========================================================================

/// Foreign-implementable handler invoked once per assignment a worker
/// receives.
///
/// `handle` is synchronous on the foreign side — foreign code that needs
/// to drive its own async work must spawn a goroutine / coroutine /
/// fiber inside the callback and return when that work resolves. The
/// returned `Ok(String)` is interpreted as the assignment output's JSON
/// representation; the returned `Err(String)` is surfaced to the control
/// plane as an assignment failure.
///
/// `on_cancel` and `on_drain` are best-effort notifications. The
/// underlying Rust worker has already fired the per-run cancellation
/// token / queue gate before invoking these; the foreign handler should
/// use them only to release external resources (open file handles,
/// network sockets, etc.).
#[uniffi::export(with_foreign)]
pub trait ControlPlaneAssignmentHandler: Send + Sync {
    /// Handle one assignment. Return `Ok(json)` for success or any
    /// [`BlazenError`] for failure — the error's `Display`
    /// representation is forwarded to the control plane as the
    /// assignment failure message.
    ///
    /// Use [`BlazenError::Tool`] for handler-side errors, or
    /// [`BlazenError::Workflow`] for workflow-level failures.
    fn handle(
        &self,
        run_id: String,
        workflow_name: String,
        input_json: String,
        ctx: Arc<AssignmentContextHandle>,
    ) -> Result<String, BlazenError>;

    /// Called when the server cancels an in-flight run. Foreign code
    /// should treat this as a notification; the underlying Rust worker
    /// has already fired the per-run cancellation token.
    fn on_cancel(&self, run_id: String);

    /// Called when the server initiates a drain. `immediate = true`
    /// means the worker must stop now; `false` means graceful drain.
    fn on_drain(&self, immediate: bool);
}

/// Foreign-implementable subscriber that observes a per-run event stream
/// opened by [`ControlPlaneClient::subscribe_run_events`].
///
/// Like [`ControlPlaneAssignmentHandler`], every method is synchronous
/// on the foreign side. The subscription pumps inbound events on the
/// shared Tokio runtime and invokes the callbacks in the order they
/// arrive; foreign callers wanting concurrent processing should spawn
/// from inside `on_event`.
#[uniffi::export(with_foreign)]
pub trait ControlPlaneRunEventSubscriber: Send + Sync {
    /// One event arrived from the run.
    fn on_event(&self, event: ControlPlaneRunEvent);
    /// Stream ended cleanly (the run reached a terminal state).
    fn on_close(&self);
    /// Stream errored. `error` is best-effort and may not survive a
    /// reconnect-then-retry cycle.
    fn on_error(&self, error: String);
}

// ===========================================================================
// Helpers
// ===========================================================================

fn parse_run_id(run_id: &str) -> Result<Uuid, BlazenError> {
    Uuid::parse_str(run_id).map_err(|e| BlazenError::Validation {
        message: format!("invalid run_id {run_id:?}: {e}"),
    })
}

// ===========================================================================
// ControlPlaneClient
// ===========================================================================

/// gRPC client for the orchestrator side of the control plane.
///
/// Construct with [`ControlPlaneClient::connect`] (async) or
/// [`ControlPlaneClient::connect_blocking`] (sync). All RPCs are
/// serialised behind an inner [`tokio::sync::Mutex`] held inside the
/// upstream [`CoreClient`]; concurrent calls on the same handle are safe
/// but each method holds the mutex for the duration of its RPC.
#[derive(uniffi::Object)]
pub struct ControlPlaneClient {
    inner: Arc<CoreClient>,
}

#[uniffi::export]
impl ControlPlaneClient {
    /// Synchronous constructor. Blocks the current thread on the shared
    /// Tokio runtime while the TCP/HTTP-2 handshake completes.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::ControlPlane`] (`kind = "Transport"`) if
    /// the endpoint URI is invalid or the handshake fails.
    #[uniffi::constructor]
    pub fn connect_blocking(
        endpoint: String,
        bearer_token: Option<String>,
    ) -> BlazenResult<Arc<Self>> {
        let inner = runtime()
            .block_on(async move { CoreClient::connect(endpoint, None, bearer_token).await })
            .map_err(BlazenError::from)?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl ControlPlaneClient {
    /// Async constructor. Use from Swift `async` / Kotlin `suspend`
    /// callers.
    ///
    /// # Errors
    ///
    /// Same as [`ControlPlaneClient::connect_blocking`].
    #[uniffi::constructor]
    pub async fn connect(
        endpoint: String,
        bearer_token: Option<String>,
    ) -> BlazenResult<Arc<Self>> {
        let inner = CoreClient::connect(endpoint, None, bearer_token)
            .await
            .map_err(BlazenError::from)?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Submit a workflow to the control plane.
    ///
    /// Returns the initial [`ControlPlaneRunStateSnapshot`] (status will
    /// usually be `Pending` or `Running` immediately after submission).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `request.input_json` is
    /// not valid JSON; [`BlazenError::Workflow`] for server-side errors.
    pub async fn submit_workflow(
        self: Arc<Self>,
        request: ControlPlaneSubmitRequest,
    ) -> BlazenResult<ControlPlaneRunStateSnapshot> {
        use blazen_core::distributed::OrchestratorClient;
        let core_req = request.into_core()?;
        let snap = self
            .inner
            .submit_workflow(core_req)
            .await
            .map_err(BlazenError::from)?;
        Ok(snap.into())
    }

    /// Cancel an in-flight run.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `run_id` is not a valid
    /// UUID; [`BlazenError::Workflow`] for server-side errors.
    pub async fn cancel_workflow(
        self: Arc<Self>,
        run_id: String,
    ) -> BlazenResult<ControlPlaneRunStateSnapshot> {
        use blazen_core::distributed::OrchestratorClient;
        let id = parse_run_id(&run_id)?;
        let snap = self
            .inner
            .cancel_workflow(id)
            .await
            .map_err(BlazenError::from)?;
        Ok(snap.into())
    }

    /// Look up the current state of a run.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `run_id` is not a valid
    /// UUID; [`BlazenError::Workflow`] for server-side errors.
    pub async fn describe_workflow(
        self: Arc<Self>,
        run_id: String,
    ) -> BlazenResult<ControlPlaneRunStateSnapshot> {
        use blazen_core::distributed::OrchestratorClient;
        let id = parse_run_id(&run_id)?;
        let snap = self
            .inner
            .describe_workflow(id)
            .await
            .map_err(BlazenError::from)?;
        Ok(snap.into())
    }

    /// List currently-connected workers.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Workflow`] for server-side errors.
    pub async fn list_workers(self: Arc<Self>) -> BlazenResult<Vec<ControlPlaneWorkerInfo>> {
        use blazen_core::distributed::OrchestratorClient;
        let workers = self.inner.list_workers().await.map_err(BlazenError::from)?;
        Ok(workers.iter().map(Into::into).collect())
    }

    /// Tell the control plane to drain `node_id`.
    ///
    /// `immediate = true` asks the worker to stop now; `false` lets
    /// it finish in-flight assignments before disconnecting.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::ControlPlane`] for RPC failures.
    pub async fn drain_worker(
        self: Arc<Self>,
        node_id: String,
        immediate: bool,
    ) -> BlazenResult<()> {
        self.inner
            .drain_worker(node_id, immediate)
            .await
            .map_err(BlazenError::from)
    }

    /// Subscribe to events for `run_id`, forwarding each event to
    /// `subscriber` until the stream terminates.
    ///
    /// Returns a [`ControlPlaneSubscription`] handle; call
    /// [`ControlPlaneSubscription::cancel`] to stop pumping events
    /// before the run completes. The pump task always invokes either
    /// `on_close` or `on_error` exactly once before exiting.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `run_id` is not a valid
    /// UUID; [`BlazenError::Workflow`] if the server rejects the
    /// subscription request itself.
    pub async fn subscribe_run_events(
        self: Arc<Self>,
        run_id: String,
        subscriber: Arc<dyn ControlPlaneRunEventSubscriber>,
    ) -> BlazenResult<Arc<ControlPlaneSubscription>> {
        let id = parse_run_id(&run_id)?;
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let client = Arc::clone(&self.inner);
        tokio::spawn(async move {
            use blazen_core::distributed::OrchestratorClient;
            // Open the subscription on the worker task itself so the
            // borrowed `RunEventStream<'a>` does not outlive the
            // `&self` reference — the stream borrows the client mutex
            // and must be dropped before the task ends.
            let mut stream = match client.subscribe_run_events(id).await {
                Ok(s) => s,
                Err(e) => {
                    subscriber.on_error(e.to_string());
                    return;
                }
            };
            loop {
                tokio::select! {
                    () = cancel_for_task.cancelled() => {
                        subscriber.on_close();
                        return;
                    }
                    next = stream.next() => match next {
                        Some(Ok(event)) => subscriber.on_event(event.into()),
                        Some(Err(e)) => {
                            subscriber.on_error(e.to_string());
                            return;
                        }
                        None => {
                            subscriber.on_close();
                            return;
                        }
                    },
                }
            }
        });
        Ok(Arc::new(ControlPlaneSubscription { cancel }))
    }

    /// Answer an outstanding `input.request` raised by a running
    /// assignment.
    ///
    /// `request_id` is the value carried in the `input.request` event's
    /// payload; `response_json` is the JSON-encoded value handed back to
    /// the worker's pending [`AssignmentContextHandle::request_input`].
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Validation`] if `run_id` is not a valid UUID
    /// or `response_json` is not valid JSON; [`BlazenError::Peer`]
    /// (`kind = "ControlPlane*"`) for RPC failures.
    pub async fn respond_to_input(
        self: Arc<Self>,
        run_id: String,
        request_id: String,
        response_json: String,
    ) -> BlazenResult<()> {
        let id = parse_run_id(&run_id)?;
        let response: serde_json::Value =
            serde_json::from_str(&response_json).map_err(|e| BlazenError::Validation {
                message: format!("invalid response_json: {e}"),
            })?;
        self.inner
            .respond_to_input(id, request_id, response)
            .await
            .map_err(BlazenError::from)
    }
}

/// Handle to an active run-event subscription. Drop the handle or call
/// [`ControlPlaneSubscription::cancel`] to stop pumping events.
#[derive(uniffi::Object)]
pub struct ControlPlaneSubscription {
    cancel: CancellationToken,
}

#[uniffi::export]
impl ControlPlaneSubscription {
    /// Cancel the subscription. Idempotent. After cancellation, the
    /// subscriber's `on_close` fires (best-effort) before the pump task
    /// exits.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

// ===========================================================================
// ControlPlaneWorker
// ===========================================================================

/// gRPC worker-side handle for the control plane.
///
/// Wraps [`CoreWorker`] behind an `Arc<Mutex<Option<...>>>` because
/// upstream [`CoreWorker::run`] consumes `self` by value. The first
/// successful call to [`ControlPlaneWorker::run`] takes the worker out
/// of the mutex; subsequent calls fail with [`BlazenError::Validation`].
/// [`ControlPlaneWorker::shutdown`] is exposed separately because it
/// needs to fire even while `run` is in flight.
#[derive(uniffi::Object)]
pub struct ControlPlaneWorker {
    inner: Mutex<Option<CoreWorker>>,
    shutdown: CancellationToken,
}

#[uniffi::export]
impl ControlPlaneWorker {
    /// Synchronous constructor.
    ///
    /// Builds a [`WorkerConfig`] with `Fixed { max_in_flight: 1 }`
    /// admission and the supplied `capabilities`, validates the endpoint
    /// URI, and returns a worker that has *not* yet opened the bidi
    /// stream — call [`ControlPlaneWorker::run`] to do that.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::ControlPlane`] (`kind = "Transport"`) if
    /// `endpoint` cannot be parsed as a URI.
    #[uniffi::constructor]
    pub fn new_blocking(
        endpoint: String,
        node_id: String,
        capabilities: Vec<ControlPlaneWorkerCapability>,
        bearer_token: Option<String>,
    ) -> BlazenResult<Arc<Self>> {
        let mut config = WorkerConfig::new(endpoint, node_id);
        for cap in &capabilities {
            config = config.with_capability(cap.into());
        }
        if let Some(t) = bearer_token {
            config = config.with_bearer_token(t);
        }
        let worker = CoreWorker::connect(config).map_err(BlazenError::from)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(Some(worker)),
            shutdown: CancellationToken::new(),
        }))
    }
}

#[uniffi::export]
impl ControlPlaneWorker {
    /// Signal the worker to stop. Returns immediately; any in-flight
    /// [`ControlPlaneWorker::run`] call will return cleanly once the
    /// in-flight assignments have been told to cancel.
    ///
    /// Idempotent.
    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl ControlPlaneWorker {
    /// Drive the worker session forever (or until shutdown / drain /
    /// retry exhaustion).
    ///
    /// Adapts `handler` to the upstream
    /// [`blazen_controlplane::AssignmentHandler`] trait and hands it to
    /// [`CoreWorker::run`]. Consumes the underlying worker — calling
    /// `run` twice on the same handle returns
    /// [`BlazenError::Validation`].
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::ControlPlane`] for transport / retry
    /// failures, or [`BlazenError::Validation`] if `run` is called more
    /// than once.
    pub async fn run(
        self: Arc<Self>,
        handler: Arc<dyn ControlPlaneAssignmentHandler>,
    ) -> BlazenResult<()> {
        let worker = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or(BlazenError::Validation {
                message: "ControlPlaneWorker already consumed by a prior run() call".into(),
            })?
        };
        let adapter = AssignmentHandlerAdapter { inner: handler };
        let shutdown = self.shutdown.clone();
        let run_fut = worker.run(adapter);
        tokio::select! {
            res = run_fut => res.map_err(BlazenError::from),
            () = shutdown.cancelled() => Ok(()),
        }
    }
}

// ===========================================================================
// AssignmentHandler adapter
// ===========================================================================

/// Bridges the synchronous foreign [`ControlPlaneAssignmentHandler`] to
/// the async upstream [`AssignmentHandler`] trait.
///
/// The foreign callback is invoked via [`tokio::task::spawn_blocking`]
/// so its synchronous body cannot starve the Tokio worker pool. Any
/// JSON-decode error on the returned output is surfaced as an
/// [`AssignmentFailure`] (so the control plane records a `Failed`
/// result rather than a malformed-output panic).
struct AssignmentHandlerAdapter {
    inner: Arc<dyn ControlPlaneAssignmentHandler>,
}

#[async_trait]
impl AssignmentHandler for AssignmentHandlerAdapter {
    async fn handle(
        &self,
        assignment: Assignment,
        ctx: AssignmentContext,
    ) -> Result<Value, AssignmentFailure> {
        let run_id = assignment.run_id.to_string();
        let workflow_name = assignment.workflow_name.clone();
        let input_json = match std::str::from_utf8(&assignment.input_json) {
            Ok(s) => s.to_string(),
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "assignment input_json was not valid UTF-8: {e}"
                )));
            }
        };
        // The upstream `AssignmentHandler::handle` hands us the context by
        // value; wrap it so the foreign handler can drive `emit_event` /
        // `request_input` against it through the synchronous handle.
        let handle = Arc::new(AssignmentContextHandle {
            inner: Arc::new(ctx),
        });
        let handler = Arc::clone(&self.inner);
        let result = tokio::task::spawn_blocking(move || {
            handler.handle(run_id, workflow_name, input_json, Arc::clone(&handle))
        })
        .await
        .map_err(|e| AssignmentFailure::new(format!("foreign handler join failed: {e}")))?;
        match result {
            Ok(raw) => {
                if raw.is_empty() {
                    Ok(Value::Null)
                } else {
                    serde_json::from_str(&raw).map_err(|e| {
                        AssignmentFailure::new(format!(
                            "foreign handler returned invalid JSON: {e}"
                        ))
                    })
                }
            }
            Err(err) => Err(AssignmentFailure::new(err.to_string())),
        }
    }

    async fn on_cancel(&self, run_id: Uuid) {
        let handler = Arc::clone(&self.inner);
        let id = run_id.to_string();
        let _ = tokio::task::spawn_blocking(move || handler.on_cancel(id)).await;
    }

    async fn on_drain(&self, immediate: bool) {
        let handler = Arc::clone(&self.inner);
        let _ = tokio::task::spawn_blocking(move || handler.on_drain(immediate)).await;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_round_trips_through_core() {
        let wire = ControlPlaneWorkerCapability {
            kind: "workflow:summarize".into(),
            version: 3,
        };
        let core: CoreWorkerCapability = (&wire).into();
        assert_eq!(core.kind, "workflow:summarize");
        assert_eq!(core.version, 3);
        let back: ControlPlaneWorkerCapability = (&core).into();
        assert_eq!(back.kind, wire.kind);
        assert_eq!(back.version, wire.version);
    }

    #[test]
    fn admission_mode_fixed_round_trip() {
        let wire = ControlPlaneAdmission {
            mode: ControlPlaneAdmissionMode::Fixed,
            max_in_flight: Some(4),
            total_mb: None,
        };
        let core: CoreAdmissionMode = (&wire).into();
        match core {
            CoreAdmissionMode::Fixed { max_in_flight } => assert_eq!(max_in_flight, 4),
            other => panic!("expected Fixed, got {other:?}"),
        }
        let back: ControlPlaneAdmission = (&core).into();
        assert!(matches!(back.mode, ControlPlaneAdmissionMode::Fixed));
        assert_eq!(back.max_in_flight, Some(4));
        assert_eq!(back.total_mb, None);
    }

    #[test]
    fn admission_mode_reactive_round_trip() {
        let wire = ControlPlaneAdmission {
            mode: ControlPlaneAdmissionMode::Reactive,
            max_in_flight: None,
            total_mb: None,
        };
        let core: CoreAdmissionMode = (&wire).into();
        assert!(matches!(core, CoreAdmissionMode::Reactive));
        let back: ControlPlaneAdmission = (&core).into();
        assert!(matches!(back.mode, ControlPlaneAdmissionMode::Reactive));
        assert!(back.max_in_flight.is_none());
        assert!(back.total_mb.is_none());
    }

    #[test]
    fn admission_mode_vram_budget_round_trip() {
        let wire = ControlPlaneAdmission {
            mode: ControlPlaneAdmissionMode::VramBudget,
            max_in_flight: None,
            total_mb: Some(16_384),
        };
        let core: CoreAdmissionMode = (&wire).into();
        match core {
            CoreAdmissionMode::VramBudget { max_vram_mb } => assert_eq!(max_vram_mb, 16_384),
            other => panic!("expected VramBudget, got {other:?}"),
        }
        let back: ControlPlaneAdmission = (&core).into();
        assert!(matches!(back.mode, ControlPlaneAdmissionMode::VramBudget));
        assert_eq!(back.total_mb, Some(16_384));
    }

    #[test]
    fn admission_mode_fixed_defaults_to_one_when_unspecified() {
        let wire = ControlPlaneAdmission {
            mode: ControlPlaneAdmissionMode::Fixed,
            max_in_flight: None,
            total_mb: None,
        };
        let core: CoreAdmissionMode = (&wire).into();
        match core {
            CoreAdmissionMode::Fixed { max_in_flight } => assert_eq!(max_in_flight, 1),
            other => panic!("expected Fixed, got {other:?}"),
        }
    }

    #[test]
    fn run_status_round_trip() {
        for status in [
            CoreRunStatus::Pending,
            CoreRunStatus::Running,
            CoreRunStatus::Completed,
            CoreRunStatus::Failed,
            CoreRunStatus::Cancelled,
        ] {
            let wire: ControlPlaneRunStatus = status.into();
            let back: CoreRunStatus = wire.into();
            assert_eq!(status, back);
        }
    }

    #[test]
    fn run_state_snapshot_flattens_output_to_json_string() {
        let id = Uuid::new_v4();
        let snap = CoreRunStateSnapshot {
            run_id: id,
            status: CoreRunStatus::Completed,
            started_at_ms: 100,
            completed_at_ms: Some(200),
            assigned_to: Some("worker-a".into()),
            last_event_at_ms: Some(150),
            output: Some(serde_json::json!({"answer": 42})),
            error: None,
        };
        let wire: ControlPlaneRunStateSnapshot = snap.into();
        assert_eq!(wire.run_id, id.to_string());
        assert!(matches!(wire.status, ControlPlaneRunStatus::Completed));
        assert_eq!(wire.assigned_to.as_deref(), Some("worker-a"));
        let output_json = wire.output_json.expect("output present");
        let parsed: Value = serde_json::from_str(&output_json).unwrap();
        assert_eq!(parsed, serde_json::json!({"answer": 42}));
        assert!(wire.error.is_none());
    }

    #[test]
    fn run_event_round_trips_through_string_uuid_and_json() {
        let id = Uuid::new_v4();
        let core = CoreRunEvent {
            run_id: id,
            event_type: "Progress".into(),
            data: serde_json::json!({"pct": 50}),
            timestamp_ms: 1234,
        };
        let wire: ControlPlaneRunEvent = core.into();
        assert_eq!(wire.run_id, id.to_string());
        assert_eq!(wire.event_type, "Progress");
        let parsed: Value = serde_json::from_str(&wire.data_json).unwrap();
        assert_eq!(parsed, serde_json::json!({"pct": 50}));
        assert_eq!(wire.timestamp_ms, 1234);
    }

    #[test]
    fn worker_info_flattens_btreemap_tags_to_hashmap() {
        let mut tags = std::collections::BTreeMap::new();
        tags.insert("region".to_string(), "us-west".to_string());
        tags.insert("tier".to_string(), "gpu".to_string());
        let core = CoreWorkerInfo {
            node_id: "node-a".into(),
            capabilities: vec![CoreWorkerCapability {
                kind: "workflow:hello".into(),
                version: 1,
            }],
            tags,
            admission: CoreAdmissionMode::Fixed { max_in_flight: 1 },
            in_flight: 0,
            admission_snapshot: None,
            connected_at_ms: 1,
        };
        let wire: ControlPlaneWorkerInfo = (&core).into();
        assert_eq!(wire.node_id, "node-a");
        assert_eq!(wire.capabilities.len(), 1);
        assert_eq!(wire.capabilities[0].kind, "workflow:hello");
        assert_eq!(wire.tags.get("region").map(String::as_str), Some("us-west"));
        assert_eq!(wire.tags.get("tier").map(String::as_str), Some("gpu"));
        assert_eq!(wire.tags.len(), 2);
    }

    #[test]
    fn submit_request_into_core_parses_input_json() {
        let req = ControlPlaneSubmitRequest {
            workflow_name: "summarize".into(),
            input_json: "{\"text\":\"hello\"}".into(),
            workflow_version: Some(2),
            required_tags: vec!["region=us-west".into()],
            idempotency_key: Some("dedupe-1".into()),
            deadline_ms: Some(60_000),
            wait_for_worker: true,
        };
        let core = req.into_core().expect("valid input JSON");
        assert_eq!(core.workflow_name, "summarize");
        assert_eq!(core.workflow_version, Some(2));
        assert_eq!(core.input, serde_json::json!({"text": "hello"}));
        assert_eq!(core.required_tags, vec!["region=us-west"]);
        assert_eq!(core.idempotency_key.as_deref(), Some("dedupe-1"));
        assert_eq!(core.deadline_ms, Some(60_000));
        assert!(core.wait_for_worker);
        assert!(core.resource_hint.is_none());
    }

    #[test]
    fn submit_request_into_core_rejects_invalid_json() {
        let req = ControlPlaneSubmitRequest {
            workflow_name: "x".into(),
            input_json: "{not json".into(),
            workflow_version: None,
            required_tags: Vec::new(),
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: false,
        };
        match req.into_core() {
            Ok(_) => panic!("expected invalid JSON to be rejected"),
            Err(e) => assert!(
                matches!(e, BlazenError::Validation { .. }),
                "expected Validation error, got {e:?}",
            ),
        }
    }

    #[test]
    fn parse_run_id_accepts_valid_uuid() {
        let id = Uuid::new_v4();
        let parsed = parse_run_id(&id.to_string()).expect("valid uuid");
        assert_eq!(parsed, id);
    }

    #[test]
    fn parse_run_id_rejects_garbage() {
        match parse_run_id("not-a-uuid") {
            Ok(_) => panic!("expected garbage to be rejected"),
            Err(e) => assert!(
                matches!(e, BlazenError::Validation { .. }),
                "expected Validation error, got {e:?}",
            ),
        }
    }
}
