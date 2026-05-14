//! Distributed workflow execution types (requires `distributed` feature).
//!
//! This module defines the [`PeerClient`] trait and the lightweight
//! request/response types that [`Workflow::run_remote`] uses to invoke a
//! workflow on a remote node. The types are transport-agnostic —
//! `blazen-peer` provides the canonical gRPC implementation of
//! [`PeerClient`] via `BlazenPeerClient`, but any transport (HTTP, NATS,
//! in-process mock, etc.) can implement the trait.
//!
//! ## Why a trait instead of a concrete client?
//!
//! `blazen-peer` depends on `blazen-core` (it needs [`crate::Context`],
//! [`crate::StepRegistration`], etc.). If `blazen-core` depended back on
//! `blazen-peer` there would be a cyclic crate dependency. Defining the
//! trait here lets `blazen-core` stay at the bottom of the dependency
//! graph while `blazen-peer` implements the trait from above.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use uuid::Uuid;

use crate::error::WorkflowError;
use crate::session_ref::RemoteRefDescriptor;

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// Transport-agnostic request for invoking a sub-workflow on a remote
/// peer.
///
/// Mirrors the fields of `blazen_peer::protocol::SubWorkflowRequest`
/// but without the postcard/serde wire encoding — the [`PeerClient`]
/// implementor is responsible for serializing this into whatever format
/// the transport requires.
#[derive(Debug, Clone)]
pub struct RemoteWorkflowRequest {
    /// Symbolic name of the workflow to invoke on the remote peer.
    pub workflow_name: String,
    /// Ordered list of step IDs to execute as part of this sub-workflow.
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step.
    pub input: serde_json::Value,
    /// Optional timeout in seconds. `None` means "use the server's
    /// default deadline".
    pub timeout_secs: Option<u64>,
}

/// Transport-agnostic response from a remote sub-workflow invocation.
///
/// Mirrors the useful parts of
/// `blazen_peer::protocol::SubWorkflowResponse` in a form that
/// [`Workflow::run_remote`] can consume without depending on
/// `blazen-peer`.
#[derive(Debug, Clone)]
pub struct RemoteWorkflowResponse {
    /// Optional terminal result. `None` when the workflow exited
    /// without producing one.
    pub result: Option<serde_json::Value>,
    /// Descriptors for any session refs the sub-workflow registered
    /// that the parent should be able to dereference remotely.
    pub remote_refs: HashMap<Uuid, RemoteRefDescriptor>,
    /// Error message if the sub-workflow failed. When `Some`, callers
    /// should ignore `result`.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Abstraction over a transport that can invoke a sub-workflow on a
/// remote node.
///
/// The canonical implementation lives in `blazen-peer` as
/// `BlazenPeerClient`. Tests can supply a mock implementation that
/// returns canned responses without standing up a gRPC server.
pub trait PeerClient: Send + Sync {
    /// Invoke a sub-workflow described by `request` on the remote peer.
    ///
    /// Implementations should serialize the request, send it over their
    /// transport, await the response, and deserialize it into a
    /// [`RemoteWorkflowResponse`].
    ///
    /// # Errors
    ///
    /// Returns a [`WorkflowError`] if the transport fails, the remote
    /// peer is unreachable, or the response cannot be decoded.
    fn invoke_sub_workflow<'a>(
        &'a self,
        request: RemoteWorkflowRequest,
    ) -> Pin<Box<dyn Future<Output = Result<RemoteWorkflowResponse, WorkflowError>> + Send + 'a>>;
}

// ---------------------------------------------------------------------------
// Control plane: shared types
// ---------------------------------------------------------------------------

/// Typed capability a worker advertises when it connects to the control
/// plane. The control plane uses the kind/version pair to route work to
/// the right worker.
///
/// Conventional kinds:
/// - `"workflow:<name>"` — worker can run this workflow end-to-end
/// - `"step:<name>"` — worker can host this individual step
/// - `"provider:<id>"` — worker hosts this LLM/compute provider
///   (built-in or `CustomProvider`)
/// - `"tag:<key>=<value>"` — free-form attribute used by tag predicates
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerCapability {
    pub kind: String,
    pub version: u32,
}

/// How a worker declares its capacity to the control plane. See the
/// architecture doc for when to use each.
#[derive(Debug, Clone)]
pub enum AdmissionMode {
    /// Hard count cap. Best for fungible CPU work where every job costs
    /// roughly the same.
    Fixed { max_in_flight: u32 },
    /// VRAM-sum cap. Best when GPU memory is the binding constraint.
    /// Every assignment routed to a `VramBudget` worker MUST carry a
    /// `resource_hint.vram_mb` estimate.
    VramBudget { max_vram_mb: u64 },
    /// Worker self-decides via offer/claim/decline negotiation. Best
    /// for multi-model GPUs, browser/WebGPU workers, and `CustomProvider`
    /// hosts with their own queueing.
    Reactive,
}

/// Optional resource estimate attached to an `Assignment`. Used by
/// `VramBudget` workers to track in-flight VRAM and by `Reactive`
/// workers as input to their decide-fn.
#[derive(Debug, Clone, Default)]
pub struct ResourceHint {
    pub vram_mb: Option<u64>,
    pub cpu_cores: Option<f32>,
    pub expected_seconds: Option<u32>,
}

/// Snapshot of a Reactive worker's live capacity. Carried in
/// heartbeats so the control plane can rank candidate workers without
/// pinging them per-decision.
#[derive(Debug, Clone)]
pub struct AdmissionSnapshot {
    /// 0.0 = at capacity, refuse new work; 1.0 = fully idle.
    pub capacity_score: f32,
    /// Models currently resident on this worker (used for affinity).
    pub model_residency: std::collections::BTreeSet<String>,
    /// Free VRAM in MB. `None` for workers without a GPU.
    pub vram_free_mb: Option<u64>,
    /// Sum of `resource_hint.vram_mb` across currently-assigned jobs.
    pub in_flight_vram_mb: u64,
}

/// Request to submit a workflow run to the control plane.
#[derive(Debug, Clone)]
pub struct SubmitWorkflowRequest {
    /// Symbolic name of the workflow to run.
    pub workflow_name: String,
    /// Workflow version. `None` = latest available.
    pub workflow_version: Option<u32>,
    /// JSON-encoded initial input for the workflow's first step.
    pub input: serde_json::Value,
    /// Required tags a matching worker must advertise. Each entry is
    /// `key=value` (or `key=*` wildcard). All entries are AND'd.
    pub required_tags: Vec<String>,
    /// Optional client-supplied dedupe key. The control plane will not
    /// schedule a second run with the same `(workflow_name, idempotency_key)`
    /// inside the dedupe TTL.
    pub idempotency_key: Option<String>,
    /// Optional deadline in milliseconds from submission. `None` = no
    /// timeout.
    pub deadline_ms: Option<u64>,
    /// If `true` and no worker matches at submit time, queue the
    /// request until a matching worker appears (or `deadline_ms`
    /// elapses). If `false` (default), unmatched submits return
    /// `FAILED_PRECONDITION` immediately.
    pub wait_for_worker: bool,
    /// Optional resource estimate. Required when targeting a
    /// `VramBudget` worker, advisory for `Reactive`, ignored by `Fixed`.
    pub resource_hint: Option<ResourceHint>,
}

/// Snapshot of a workflow run's state.
#[derive(Debug, Clone)]
pub struct RunStateSnapshot {
    pub run_id: uuid::Uuid,
    pub status: RunStatus,
    pub started_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    /// `Some(node_id)` once an assignment has been routed.
    pub assigned_to: Option<String>,
    pub last_event_at_ms: Option<u64>,
    /// Terminal output if `status == Completed`.
    pub output: Option<serde_json::Value>,
    /// Error message if `status == Failed`.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Event emitted during a run. Carried over `SubscribeRunEvents` and
/// `SubscribeAll` streams.
#[derive(Debug, Clone)]
pub struct RunEvent {
    pub run_id: uuid::Uuid,
    pub event_type: String,
    pub data: serde_json::Value,
    pub timestamp_ms: u64,
}

/// Summary of a connected worker, returned by `ListWorkers`.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub node_id: String,
    pub capabilities: Vec<WorkerCapability>,
    pub tags: std::collections::BTreeMap<String, String>,
    pub admission: AdmissionMode,
    /// Last reported in-flight assignment count.
    pub in_flight: u32,
    /// Last reported admission snapshot (Reactive/VramBudget only).
    pub admission_snapshot: Option<AdmissionSnapshot>,
    pub connected_at_ms: u64,
}

// ---------------------------------------------------------------------------
// Control plane: traits
// ---------------------------------------------------------------------------

/// Stream of run events delivered to an orchestrator subscription.
pub type RunEventStream<'a> =
    Pin<Box<dyn futures_core::Stream<Item = Result<RunEvent, WorkflowError>> + Send + 'a>>;

/// Orchestrator-side client: submit/cancel/observe workflows on a
/// control plane. The canonical implementation lives in
/// `blazen-controlplane::OrchestratorClient` (gRPC). Tests can supply
/// a mock that returns canned responses.
#[async_trait::async_trait]
pub trait OrchestratorClient: Send + Sync {
    /// Submit a workflow run. Returns the initial `RunStateSnapshot`
    /// (status will usually be `Pending` or `Running`).
    async fn submit_workflow(
        &self,
        request: SubmitWorkflowRequest,
    ) -> Result<RunStateSnapshot, WorkflowError>;

    /// Cancel an in-flight run.
    async fn cancel_workflow(&self, run_id: uuid::Uuid) -> Result<RunStateSnapshot, WorkflowError>;

    /// Look up the current state of a run.
    async fn describe_workflow(
        &self,
        run_id: uuid::Uuid,
    ) -> Result<RunStateSnapshot, WorkflowError>;

    /// Subscribe to events for a single run. The stream ends when the
    /// run terminates.
    async fn subscribe_run_events<'a>(
        &'a self,
        run_id: uuid::Uuid,
    ) -> Result<RunEventStream<'a>, WorkflowError>;

    /// List currently-connected workers.
    async fn list_workers(&self) -> Result<Vec<WorkerInfo>, WorkflowError>;
}

/// Sink delivered to a worker's session. The worker calls these methods
/// to forward events / results / heartbeats back to the control plane.
#[async_trait::async_trait]
pub trait WorkerSessionSink: Send + Sync {
    /// Emit an event from a running assignment.
    async fn emit_event(&self, run_id: uuid::Uuid, event: RunEvent) -> Result<(), WorkflowError>;

    /// Report terminal result of an assignment.
    async fn report_result(
        &self,
        run_id: uuid::Uuid,
        result: Result<serde_json::Value, String>,
    ) -> Result<(), WorkflowError>;

    /// Report a heartbeat — in-flight count and (for Reactive workers)
    /// an admission snapshot.
    async fn heartbeat(
        &self,
        in_flight: u32,
        admission_snapshot: Option<AdmissionSnapshot>,
    ) -> Result<(), WorkflowError>;
}

/// Server-side abstraction over the control plane itself. Used by
/// `Workflow::run_via_control_plane` to dispatch work.
///
/// Note: there is no separate `ControlPlaneClient` trait — orchestrator
/// clients implement `OrchestratorClient` above. This trait exists for
/// the *server* side to expose its own queue/registry to embedded
/// callers (e.g. when a workflow running on the same process as the
/// control plane wants to enqueue a sub-workflow).
#[async_trait::async_trait]
pub trait ControlPlane: Send + Sync {
    /// Submit a workflow for execution on a registered worker.
    async fn enqueue(
        &self,
        request: SubmitWorkflowRequest,
    ) -> Result<RunStateSnapshot, WorkflowError>;

    /// Cancel an in-flight run.
    async fn cancel(&self, run_id: uuid::Uuid) -> Result<RunStateSnapshot, WorkflowError>;
}
