//! Postcard-serializable wire types for the Blazen control plane.
//!
//! Every gRPC method on the [`crate::pb`] service takes and returns a
//! single `bytes` field whose contents are one of the structs defined
//! in this module, encoded with [`postcard`]. Versioning is handled
//! per-message by the [`ENVELOPE_VERSION`] field rather than by
//! evolving the proto schema, so adding a new field here is a
//! source-only change for both client and server.
//!
//! ## A note on `serde_json::Value`
//!
//! Postcard is a non-self-describing format and cannot round-trip
//! `serde_json::Value` directly — its untagged enum requires
//! `deserialize_any`, which postcard explicitly does not implement.
//! To stay compatible we carry JSON payloads as `Vec<u8>` of
//! pre-serialized JSON text. Helper constructors (e.g.
//! [`SubmitRequest::from_core`], [`Assignment::input_value`]) take and
//! return `serde_json::Value` for ergonomics, doing the JSON encode /
//! decode at the boundary so callers never have to think about the
//! byte representation.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use blazen_core::distributed as core;

/// Current envelope version. Bump this whenever you change the shape
/// of any struct in this module in a way that is not forward-compatible.
///
/// The convention is:
///
/// - **Adding a new optional field at the end** of a struct is
///   forward-compatible — postcard will skip unknown trailing bytes on
///   decode and `Option::None` for missing fields. No version bump.
/// - **Renaming, reordering, or removing fields** is *not*
///   forward-compatible. Bump this constant and update
///   [`crate::error::ControlPlaneError::EnvelopeVersion`] handling on
///   the server side.
pub const ENVELOPE_VERSION: u32 = 1;

/// Returns `Err` if `got` is greater than [`ENVELOPE_VERSION`] (i.e.
/// the payload was produced by a newer build of blazen-controlplane
/// than this one can decode). Older payloads are always accepted —
/// postcard will tolerate missing trailing fields.
///
/// # Errors
/// Returns [`crate::error::ControlPlaneError::EnvelopeVersion`] when
/// `got > ENVELOPE_VERSION`.
pub fn validate_envelope_version(got: u32) -> Result<(), crate::error::ControlPlaneError> {
    if got > ENVELOPE_VERSION {
        Err(crate::error::ControlPlaneError::EnvelopeVersion {
            got,
            supported: ENVELOPE_VERSION,
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared wire types — postcard-friendly mirrors of `blazen_core::distributed`
// ---------------------------------------------------------------------------

/// Wire-format mirror of [`core::AdmissionMode`]. Same shape — exists
/// so the control plane has a single `Serialize`/`Deserialize` type
/// independent of any future `serde` impls on the core type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdmissionModeWire {
    /// Hard count cap. Best for fungible CPU work.
    Fixed {
        /// Maximum number of concurrent in-flight assignments.
        max_in_flight: u32,
    },
    /// VRAM-sum cap. Best when GPU memory is the binding constraint.
    VramBudget {
        /// Maximum sum of `resource_hint.vram_mb` across in-flight assignments.
        max_vram_mb: u64,
    },
    /// Worker self-decides via offer/claim/decline negotiation.
    Reactive,
}

/// Wire-format mirror of [`core::ResourceHint`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResourceHintWire {
    /// VRAM estimate in MB. Required when targeting a `VramBudget` worker.
    pub vram_mb: Option<u64>,
    /// CPU-core estimate. Advisory.
    pub cpu_cores: Option<f32>,
    /// Expected runtime in seconds. Advisory; used by Reactive workers
    /// to anticipate queue pressure.
    pub expected_seconds: Option<u32>,
}

/// Wire-format mirror of [`core::WorkerCapability`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CapabilityWire {
    /// Capability kind tag (e.g. `"workflow:summarize"`, `"step:fetch"`).
    pub kind: String,
    /// Schema version of the capability. The control plane refuses to
    /// route work to a worker with a mismatched version.
    pub version: u32,
}

/// Wire-format mirror of [`core::AdmissionSnapshot`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionSnapshotWire {
    /// 0.0 = at capacity, refuse new work; 1.0 = fully idle.
    pub capacity_score: f32,
    /// Models currently resident on this worker (used for affinity).
    /// `BTreeSet` rather than `HashSet` so the wire encoding is
    /// deterministic across runs and platforms.
    pub model_residency: BTreeSet<String>,
    /// Free VRAM in MB. `None` for workers without a GPU.
    pub vram_free_mb: Option<u64>,
    /// Sum of `resource_hint.vram_mb` across currently-assigned jobs.
    pub in_flight_vram_mb: u64,
}

// ---------------------------------------------------------------------------
// Worker → server messages
// ---------------------------------------------------------------------------

/// First frame a worker sends when establishing a bidi session. The
/// server replies with [`Welcome`] (on accept) or [`ServerToWorker::Reject`]
/// (otherwise).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHello {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Stable identifier of the worker node, used for routing,
    /// reconnects, and metrics.
    pub node_id: String,
    /// Capabilities advertised by this worker. The control plane uses
    /// these to match incoming assignments to workers.
    pub capabilities: Vec<CapabilityWire>,
    /// Free-form key/value attributes used by tag predicates on
    /// [`SubmitRequest::required_tags`]. `BTreeMap` for deterministic
    /// encoding.
    pub tags: BTreeMap<String, String>,
    /// Admission strategy this worker uses. The server schedules to it
    /// accordingly.
    pub admission: AdmissionModeWire,
    /// Highest `envelope_version` this worker understands. Server uses
    /// this to negotiate down if it's older.
    pub supported_envelope_versions: Vec<u32>,
}

/// Periodic worker → server heartbeat. Drives liveness, in-flight
/// accounting, and (for Reactive workers) ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Number of assignments currently running on this worker.
    pub in_flight: u32,
    /// Number of assignments queued locally but not yet started.
    pub queue_depth: u32,
    /// Resident memory in MB.
    pub mem_mb: u64,
    /// Aggregate CPU utilisation as a percentage (0.0..=100.0 typical).
    pub cpu_pct: f32,
    /// Optional Reactive/VramBudget admission snapshot. `None` for
    /// `Fixed` workers since the server can derive their state from
    /// `in_flight` alone.
    pub admission_snapshot: Option<AdmissionSnapshotWire>,
}

/// Terminal result of an assignment, sent from worker back to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentResult {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the assignment this result belongs to.
    pub run_id: Uuid,
    /// JSON-encoded terminal output. Empty when `status != Completed`.
    /// See the module-level docs for why we use `Vec<u8>` instead of
    /// `serde_json::Value`.
    #[serde(with = "serde_bytes")]
    pub output_json: Vec<u8>,
    /// Outcome of the assignment.
    pub status: AssignmentStatus,
    /// Error description when `status == Failed`. `None` otherwise.
    pub error: Option<String>,
}

/// Outcome of an assignment as reported back by the worker.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssignmentStatus {
    /// Assignment ran to completion with a valid output.
    Completed,
    /// Assignment ran but failed; see [`AssignmentResult::error`].
    Failed,
    /// Assignment was cancelled before completion (by the orchestrator
    /// or by a drain instruction).
    Cancelled,
}

/// Non-terminal event emitted by a running assignment. Used to forward
/// step events, log lines, partial results, etc. to subscribers on the
/// orchestrator side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentEvent {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the assignment this event belongs to.
    pub run_id: Uuid,
    /// Free-form event kind. Conventional names mirror
    /// [`blazen_core::Event`] kinds (`"step.start"`, `"step.finish"`,
    /// `"workflow.error"`, …).
    pub event_type: String,
    /// JSON-encoded event payload. See the module-level docs for why
    /// we use `Vec<u8>` instead of `serde_json::Value`.
    #[serde(with = "serde_bytes")]
    pub data_json: Vec<u8>,
    /// Wall-clock event timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

/// Worker's response to a server [`Offer`] (Reactive admission only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfferDecision {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the assignment this decision is for. Matches the
    /// `run_id` of the [`Offer::assignment`] the worker is responding to.
    pub run_id: Uuid,
    /// Whether the worker is taking the offer.
    pub decision: OfferOutcome,
}

/// Outcome of a Reactive offer negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OfferOutcome {
    /// Worker is claiming the assignment and will execute it.
    Claim,
    /// Worker is declining; server should offer the assignment to the
    /// next candidate.
    Decline {
        /// Why the worker declined; used by the server for metrics and
        /// to avoid re-offering hopeless candidates.
        reason: DeclineReason,
    },
}

/// Reason a Reactive worker declined an offer. Carried in
/// [`OfferOutcome::Decline`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeclineReason {
    /// Worker does not have enough free VRAM to run this assignment.
    NotEnoughVram,
    /// Worker would accept but its decide-fn reports the run would
    /// exceed its deadline.
    TooSlow,
    /// Accepting would force an eviction the worker is unwilling to do
    /// (e.g. evicting a higher-priority resident model).
    EvictionConflict,
    /// Catch-all for declines that don't fit the structured variants.
    Other(String),
}

/// Top-level worker→server frame. Carried as the postcard payload of
/// any `PostcardRequest` the worker sends in the bidi stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerToServer {
    /// Initial handshake frame; first message on every new session.
    Hello(WorkerHello),
    /// Periodic liveness/capacity report.
    Heartbeat(WorkerHeartbeat),
    /// Terminal assignment result.
    Result(AssignmentResult),
    /// Non-terminal assignment event.
    Event(AssignmentEvent),
    /// Reactive admission decision in response to an [`Offer`].
    OfferDecision(OfferDecision),
}

// ---------------------------------------------------------------------------
// Server → worker messages
// ---------------------------------------------------------------------------

/// Server's response to a [`WorkerHello`] when the worker is accepted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Welcome {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Server-issued opaque session identifier. The worker echoes this
    /// in subsequent reconnect / resume requests.
    pub session_id: Uuid,
    /// Server-chosen `envelope_version` (intersection of client + server
    /// support). Both sides must encode subsequent frames using this
    /// version's shape.
    pub negotiated_envelope_version: u32,
}

/// Assignment dispatch from server to worker. The worker is expected
/// to run the named workflow with the provided input and report back
/// via [`AssignmentEvent`] / [`AssignmentResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of this assignment / run.
    pub run_id: Uuid,
    /// `Some` if this is a sub-workflow of an existing run.
    pub parent_run_id: Option<Uuid>,
    /// Symbolic name of the workflow to run.
    pub workflow_name: String,
    /// Workflow version. `None` = worker should use the latest it has.
    pub workflow_version: Option<u32>,
    /// JSON-encoded initial input for the workflow. See the module-level
    /// docs for why we use `Vec<u8>` instead of `serde_json::Value`.
    #[serde(with = "serde_bytes")]
    pub input_json: Vec<u8>,
    /// Optional deadline in milliseconds from when the worker received
    /// the assignment. `None` = no timeout.
    pub deadline_ms: Option<u64>,
    /// Attempt counter, starting at 1. Incremented when the server
    /// re-dispatches after a worker failure.
    pub attempt: u32,
    /// Optional resource estimate. Required when targeting a
    /// `VramBudget` worker, advisory for `Reactive`, ignored by `Fixed`.
    pub resource_hint: Option<ResourceHintWire>,
}

impl Assignment {
    /// Decode the inner `input_json` field back into a
    /// `serde_json::Value`.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `input_json` does not contain
    /// valid JSON.
    pub fn input_value(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_slice(&self.input_json)
    }
}

/// Server-initiated cancellation of an in-flight assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelInstruction {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the assignment to cancel.
    pub run_id: Uuid,
}

/// Server-initiated drain of a worker. Workers receiving this should
/// either refuse new assignments (immediate) or finish current work
/// then disconnect (graceful).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainInstruction {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// `true` = drain now, refuse new assignments immediately. `false` =
    /// graceful drain, finish in-flight then disconnect.
    pub immediate: bool,
}

/// Offer sent during Reactive admission negotiation. Worker must respond
/// with [`OfferDecision`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Offer {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// The assignment being offered. If the worker claims, the server
    /// promotes this to an active assignment without resending.
    pub assignment: Assignment,
}

/// Top-level server→worker frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerToWorker {
    /// Handshake response when the [`WorkerHello`] is accepted.
    Welcome(Welcome),
    /// Direct dispatch (Fixed/VramBudget workers, or Reactive after a
    /// successful claim).
    Assignment(Assignment),
    /// Offer phase of Reactive admission.
    Offer(Offer),
    /// Cancel an in-flight assignment.
    Cancel(CancelInstruction),
    /// Drain the worker.
    Drain(DrainInstruction),
    /// Server is rejecting the Hello (bad auth, version mismatch, …).
    Reject {
        /// Human-readable reason for the rejection.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Orchestrator unary RPC payload types
// ---------------------------------------------------------------------------

/// Orchestrator request to submit a new workflow run. Wire mirror of
/// [`core::SubmitWorkflowRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Symbolic name of the workflow to run.
    pub workflow_name: String,
    /// Workflow version. `None` = latest available.
    pub workflow_version: Option<u32>,
    /// JSON-encoded initial input. See the module-level docs for why
    /// we use `Vec<u8>` instead of `serde_json::Value`.
    #[serde(with = "serde_bytes")]
    pub input_json: Vec<u8>,
    /// Required tags a matching worker must advertise. AND-combined.
    pub required_tags: Vec<String>,
    /// Optional dedupe key. The control plane will not schedule a
    /// second run with the same `(workflow_name, idempotency_key)`
    /// inside the dedupe TTL.
    pub idempotency_key: Option<String>,
    /// Optional deadline in milliseconds from submission. `None` = no
    /// timeout.
    pub deadline_ms: Option<u64>,
    /// If `true` and no worker matches at submit time, queue the
    /// request until a matching worker appears (or `deadline_ms`
    /// elapses).
    pub wait_for_worker: bool,
    /// Optional resource estimate. Required when targeting a
    /// `VramBudget` worker.
    pub resource_hint: Option<ResourceHintWire>,
}

impl SubmitRequest {
    /// Build a [`SubmitRequest`] from the core
    /// [`core::SubmitWorkflowRequest`], handling the JSON encode for
    /// the caller.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `req.input` cannot be
    /// serialized to JSON.
    pub fn from_core(req: &core::SubmitWorkflowRequest) -> Result<Self, serde_json::Error> {
        Ok(Self {
            envelope_version: ENVELOPE_VERSION,
            workflow_name: req.workflow_name.clone(),
            workflow_version: req.workflow_version,
            input_json: serde_json::to_vec(&req.input)?,
            required_tags: req.required_tags.clone(),
            idempotency_key: req.idempotency_key.clone(),
            deadline_ms: req.deadline_ms,
            wait_for_worker: req.wait_for_worker,
            resource_hint: req.resource_hint.as_ref().map(Into::into),
        })
    }

    /// Decode this wire request back into the core
    /// [`core::SubmitWorkflowRequest`].
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `input_json` does not contain
    /// valid JSON.
    pub fn to_core(&self) -> Result<core::SubmitWorkflowRequest, serde_json::Error> {
        Ok(core::SubmitWorkflowRequest {
            workflow_name: self.workflow_name.clone(),
            workflow_version: self.workflow_version,
            input: serde_json::from_slice(&self.input_json)?,
            required_tags: self.required_tags.clone(),
            idempotency_key: self.idempotency_key.clone(),
            deadline_ms: self.deadline_ms,
            wait_for_worker: self.wait_for_worker,
            resource_hint: self.resource_hint.as_ref().map(Into::into),
        })
    }
}

/// Orchestrator request to cancel an in-flight run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run to cancel.
    pub run_id: Uuid,
}

/// Orchestrator request to describe a single run's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run to look up.
    pub run_id: Uuid,
}

/// Orchestrator request to subscribe to events for a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRunRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run to subscribe to.
    pub run_id: Uuid,
}

/// Orchestrator request to subscribe to events across many runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeAllRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Filter to runs whose tags AND-match these. Empty = subscribe to
    /// all runs.
    pub required_tags: Vec<String>,
}

/// Orchestrator request to list currently-connected workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListWorkersRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
}

/// Orchestrator request to drain a worker. Workers receiving this
/// will refuse new assignments (immediate) or finish current work
/// then disconnect (graceful).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainWorkerRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Node identifier of the worker to drain.
    pub node_id: String,
    /// `true` = drain now, refuse new assignments immediately. `false` =
    /// graceful drain, finish in-flight then disconnect.
    pub immediate: bool,
}

/// Wire mirror of [`core::RunStateSnapshot`]. Returned by `Submit`,
/// `Cancel`, and `Describe` orchestrator RPCs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStateSnapshotWire {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run.
    pub run_id: Uuid,
    /// Current lifecycle status.
    pub status: RunStatusWire,
    /// Wall-clock submission timestamp, milliseconds since the Unix
    /// epoch.
    pub started_at_ms: u64,
    /// Wall-clock completion timestamp. `None` until terminal.
    pub completed_at_ms: Option<u64>,
    /// `Some(node_id)` once an assignment has been routed.
    pub assigned_to: Option<String>,
    /// Timestamp of the most recent [`RunEventWire`] for this run.
    pub last_event_at_ms: Option<u64>,
    /// JSON-encoded output if `status == Completed`. Empty otherwise.
    /// See the module-level docs for why we use `Vec<u8>` instead of
    /// `serde_json::Value`.
    #[serde(with = "serde_bytes")]
    pub output_json: Vec<u8>,
    /// Error description when `status == Failed`. `None` otherwise.
    pub error: Option<String>,
}

impl RunStateSnapshotWire {
    /// Build a [`RunStateSnapshotWire`] from the core
    /// [`core::RunStateSnapshot`], JSON-encoding the optional output.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `snap.output` is `Some` and
    /// cannot be serialized to JSON.
    pub fn from_core(snap: &core::RunStateSnapshot) -> Result<Self, serde_json::Error> {
        let output_json = match &snap.output {
            Some(value) => serde_json::to_vec(value)?,
            None => Vec::new(),
        };
        Ok(Self {
            envelope_version: ENVELOPE_VERSION,
            run_id: snap.run_id,
            status: snap.status.into(),
            started_at_ms: snap.started_at_ms,
            completed_at_ms: snap.completed_at_ms,
            assigned_to: snap.assigned_to.clone(),
            last_event_at_ms: snap.last_event_at_ms,
            output_json,
            error: snap.error.clone(),
        })
    }

    /// Decode this wire snapshot back into the core
    /// [`core::RunStateSnapshot`].
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `output_json` is non-empty but
    /// does not contain valid JSON. An empty `output_json` decodes to
    /// `output: None`.
    pub fn to_core(&self) -> Result<core::RunStateSnapshot, serde_json::Error> {
        let output = if self.output_json.is_empty() {
            None
        } else {
            Some(serde_json::from_slice(&self.output_json)?)
        };
        Ok(core::RunStateSnapshot {
            run_id: self.run_id,
            status: self.status.into(),
            started_at_ms: self.started_at_ms,
            completed_at_ms: self.completed_at_ms,
            assigned_to: self.assigned_to.clone(),
            last_event_at_ms: self.last_event_at_ms,
            output,
            error: self.error.clone(),
        })
    }
}

/// Wire mirror of [`core::RunStatus`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunStatusWire {
    /// Submitted but not yet assigned to a worker.
    Pending,
    /// Assigned to a worker and executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cancelled (by the orchestrator or by drain).
    Cancelled,
}

/// Wire mirror of [`core::RunEvent`]. Delivered by `SubscribeRunEvents`
/// and `SubscribeAll`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEventWire {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run this event belongs to.
    pub run_id: Uuid,
    /// Free-form event kind; mirrors [`AssignmentEvent::event_type`].
    pub event_type: String,
    /// JSON-encoded event payload. See the module-level docs for why
    /// we use `Vec<u8>` instead of `serde_json::Value`.
    #[serde(with = "serde_bytes")]
    pub data_json: Vec<u8>,
    /// Wall-clock event timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

impl RunEventWire {
    /// Build a [`RunEventWire`] from the core [`core::RunEvent`],
    /// JSON-encoding the payload.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `event.data` cannot be
    /// serialized to JSON.
    pub fn from_core(event: &core::RunEvent) -> Result<Self, serde_json::Error> {
        Ok(Self {
            envelope_version: ENVELOPE_VERSION,
            run_id: event.run_id,
            event_type: event.event_type.clone(),
            data_json: serde_json::to_vec(&event.data)?,
            timestamp_ms: event.timestamp_ms,
        })
    }

    /// Decode this wire event back into the core [`core::RunEvent`].
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `data_json` does not contain
    /// valid JSON.
    pub fn to_core(&self) -> Result<core::RunEvent, serde_json::Error> {
        Ok(core::RunEvent {
            run_id: self.run_id,
            event_type: self.event_type.clone(),
            data: serde_json::from_slice(&self.data_json)?,
            timestamp_ms: self.timestamp_ms,
        })
    }
}

/// Wire mirror of [`core::WorkerInfo`]. Returned by `ListWorkers`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfoWire {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Stable identifier of the worker.
    pub node_id: String,
    /// Capabilities advertised by the worker.
    pub capabilities: Vec<CapabilityWire>,
    /// Free-form attributes. `BTreeMap` for deterministic encoding.
    pub tags: BTreeMap<String, String>,
    /// Admission strategy.
    pub admission: AdmissionModeWire,
    /// Last reported in-flight assignment count.
    pub in_flight: u32,
    /// Last reported admission snapshot (Reactive/VramBudget only).
    pub admission_snapshot: Option<AdmissionSnapshotWire>,
    /// Wall-clock connection timestamp, milliseconds since the Unix
    /// epoch.
    pub connected_at_ms: u64,
}

/// Response to a [`ListWorkersRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListWorkersResponse {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Currently-connected workers.
    pub workers: Vec<WorkerInfoWire>,
}

// ---------------------------------------------------------------------------
// Conversions: wire ↔ core (infallible — no JSON involved)
// ---------------------------------------------------------------------------

impl From<&core::WorkerCapability> for CapabilityWire {
    fn from(cap: &core::WorkerCapability) -> Self {
        Self {
            kind: cap.kind.clone(),
            version: cap.version,
        }
    }
}

impl From<&CapabilityWire> for core::WorkerCapability {
    fn from(cap: &CapabilityWire) -> Self {
        Self {
            kind: cap.kind.clone(),
            version: cap.version,
        }
    }
}

impl From<&core::AdmissionMode> for AdmissionModeWire {
    fn from(mode: &core::AdmissionMode) -> Self {
        match mode {
            core::AdmissionMode::Fixed { max_in_flight } => Self::Fixed {
                max_in_flight: *max_in_flight,
            },
            core::AdmissionMode::VramBudget { max_vram_mb } => Self::VramBudget {
                max_vram_mb: *max_vram_mb,
            },
            core::AdmissionMode::Reactive => Self::Reactive,
        }
    }
}

impl From<&AdmissionModeWire> for core::AdmissionMode {
    fn from(mode: &AdmissionModeWire) -> Self {
        match mode {
            AdmissionModeWire::Fixed { max_in_flight } => Self::Fixed {
                max_in_flight: *max_in_flight,
            },
            AdmissionModeWire::VramBudget { max_vram_mb } => Self::VramBudget {
                max_vram_mb: *max_vram_mb,
            },
            AdmissionModeWire::Reactive => Self::Reactive,
        }
    }
}

impl From<&core::ResourceHint> for ResourceHintWire {
    fn from(hint: &core::ResourceHint) -> Self {
        Self {
            vram_mb: hint.vram_mb,
            cpu_cores: hint.cpu_cores,
            expected_seconds: hint.expected_seconds,
        }
    }
}

impl From<&ResourceHintWire> for core::ResourceHint {
    fn from(hint: &ResourceHintWire) -> Self {
        Self {
            vram_mb: hint.vram_mb,
            cpu_cores: hint.cpu_cores,
            expected_seconds: hint.expected_seconds,
        }
    }
}

impl From<&core::AdmissionSnapshot> for AdmissionSnapshotWire {
    fn from(snap: &core::AdmissionSnapshot) -> Self {
        Self {
            capacity_score: snap.capacity_score,
            model_residency: snap.model_residency.clone(),
            vram_free_mb: snap.vram_free_mb,
            in_flight_vram_mb: snap.in_flight_vram_mb,
        }
    }
}

impl From<&AdmissionSnapshotWire> for core::AdmissionSnapshot {
    fn from(snap: &AdmissionSnapshotWire) -> Self {
        Self {
            capacity_score: snap.capacity_score,
            model_residency: snap.model_residency.clone(),
            vram_free_mb: snap.vram_free_mb,
            in_flight_vram_mb: snap.in_flight_vram_mb,
        }
    }
}

impl From<&core::WorkerInfo> for WorkerInfoWire {
    fn from(info: &core::WorkerInfo) -> Self {
        Self {
            envelope_version: ENVELOPE_VERSION,
            node_id: info.node_id.clone(),
            capabilities: info.capabilities.iter().map(Into::into).collect(),
            tags: info.tags.clone(),
            admission: (&info.admission).into(),
            in_flight: info.in_flight,
            admission_snapshot: info.admission_snapshot.as_ref().map(Into::into),
            connected_at_ms: info.connected_at_ms,
        }
    }
}

impl From<&WorkerInfoWire> for core::WorkerInfo {
    fn from(info: &WorkerInfoWire) -> Self {
        Self {
            node_id: info.node_id.clone(),
            capabilities: info.capabilities.iter().map(Into::into).collect(),
            tags: info.tags.clone(),
            admission: (&info.admission).into(),
            in_flight: info.in_flight,
            admission_snapshot: info.admission_snapshot.as_ref().map(Into::into),
            connected_at_ms: info.connected_at_ms,
        }
    }
}

impl From<core::RunStatus> for RunStatusWire {
    fn from(status: core::RunStatus) -> Self {
        match status {
            core::RunStatus::Pending => Self::Pending,
            core::RunStatus::Running => Self::Running,
            core::RunStatus::Completed => Self::Completed,
            core::RunStatus::Failed => Self::Failed,
            core::RunStatus::Cancelled => Self::Cancelled,
        }
    }
}

impl From<RunStatusWire> for core::RunStatus {
    fn from(status: RunStatusWire) -> Self {
        match status {
            RunStatusWire::Pending => Self::Pending,
            RunStatusWire::Running => Self::Running,
            RunStatusWire::Completed => Self::Completed,
            RunStatusWire::Failed => Self::Failed,
            RunStatusWire::Cancelled => Self::Cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T>(value: &T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        let bytes = postcard::to_allocvec(value).expect("postcard encode");
        postcard::from_bytes(&bytes).expect("postcard decode")
    }

    #[test]
    fn worker_hello_roundtrips() {
        let mut tags = BTreeMap::new();
        tags.insert("region".to_string(), "us-west".to_string());
        tags.insert("gpu".to_string(), "rtx-4090".to_string());

        let original = WorkerHello {
            envelope_version: ENVELOPE_VERSION,
            node_id: "worker-42".to_string(),
            capabilities: vec![
                CapabilityWire {
                    kind: "workflow:summarize".to_string(),
                    version: 1,
                },
                CapabilityWire {
                    kind: "step:fetch".to_string(),
                    version: 2,
                },
            ],
            tags,
            admission: AdmissionModeWire::VramBudget {
                max_vram_mb: 16_384,
            },
            supported_envelope_versions: vec![1],
        };

        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.node_id, original.node_id);
        assert_eq!(decoded.capabilities, original.capabilities);
        assert_eq!(decoded.tags, original.tags);
        assert_eq!(decoded.admission, original.admission);
        assert_eq!(
            decoded.supported_envelope_versions,
            original.supported_envelope_versions
        );
    }

    #[test]
    fn assignment_roundtrips() {
        let input = serde_json::json!({ "url": "https://example.com", "max_tokens": 256 });
        let original = Assignment {
            envelope_version: ENVELOPE_VERSION,
            run_id: Uuid::new_v4(),
            parent_run_id: Some(Uuid::new_v4()),
            workflow_name: "summarize".to_string(),
            workflow_version: Some(3),
            input_json: serde_json::to_vec(&input).unwrap(),
            deadline_ms: Some(30_000),
            attempt: 1,
            resource_hint: Some(ResourceHintWire {
                vram_mb: Some(2048),
                cpu_cores: Some(1.5),
                expected_seconds: Some(10),
            }),
        };

        let decoded = roundtrip(&original);
        assert_eq!(decoded.envelope_version, original.envelope_version);
        assert_eq!(decoded.run_id, original.run_id);
        assert_eq!(decoded.parent_run_id, original.parent_run_id);
        assert_eq!(decoded.workflow_name, original.workflow_name);
        assert_eq!(decoded.workflow_version, original.workflow_version);
        assert_eq!(decoded.deadline_ms, original.deadline_ms);
        assert_eq!(decoded.attempt, original.attempt);
        assert_eq!(decoded.resource_hint, original.resource_hint);
        assert_eq!(decoded.input_value().unwrap(), input);
    }

    #[test]
    fn submit_request_roundtrips_through_core() {
        let input = serde_json::json!({ "doc_id": "abc-123", "options": { "deep": true } });
        let core_req = core::SubmitWorkflowRequest {
            workflow_name: "ingest".to_string(),
            workflow_version: Some(2),
            input: input.clone(),
            required_tags: vec!["region=us-west".to_string()],
            idempotency_key: Some("dedupe-key-1".to_string()),
            deadline_ms: Some(60_000),
            wait_for_worker: true,
            resource_hint: Some(core::ResourceHint {
                vram_mb: Some(4096),
                cpu_cores: Some(2.0),
                expected_seconds: Some(20),
            }),
        };

        let wire = SubmitRequest::from_core(&core_req).expect("encode submit");
        let decoded_wire = roundtrip(&wire);
        let round = decoded_wire.to_core().expect("decode submit");

        assert_eq!(round.workflow_name, core_req.workflow_name);
        assert_eq!(round.workflow_version, core_req.workflow_version);
        assert_eq!(round.input, input);
        assert_eq!(round.required_tags, core_req.required_tags);
        assert_eq!(round.idempotency_key, core_req.idempotency_key);
        assert_eq!(round.deadline_ms, core_req.deadline_ms);
        assert_eq!(round.wait_for_worker, core_req.wait_for_worker);
        assert_eq!(
            round.resource_hint.as_ref().and_then(|h| h.vram_mb),
            Some(4096)
        );
    }

    #[test]
    fn validate_envelope_version_bounds() {
        assert!(validate_envelope_version(ENVELOPE_VERSION).is_ok());
        assert!(validate_envelope_version(0).is_ok());
        let err = validate_envelope_version(ENVELOPE_VERSION + 1)
            .expect_err("newer version must be rejected");
        match err {
            crate::error::ControlPlaneError::EnvelopeVersion { got, supported } => {
                assert_eq!(got, ENVELOPE_VERSION + 1);
                assert_eq!(supported, ENVELOPE_VERSION);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
