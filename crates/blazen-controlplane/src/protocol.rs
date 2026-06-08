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
///
/// Wire-format generation. Bumped from 1 → 2 to add `priority`,
/// `selector`, `tolerations` to [`Assignment`] and `labels`, `taints`,
/// `descriptors` to [`WorkerHello`], plus the bearer-token / input-request
/// round-trip surface. Postcard's binary format is positional so a v1
/// peer cannot decode v2 frames — the handshake negotiates the
/// intersection of `supported_envelope_versions` and rejects when the
/// versions don't overlap.
///
/// Bumped 2 → 3 to add the per-place (tenant) `place` field to
/// [`WorkerHello`], [`SubmitRequest`], [`WorkerInfoWire`], and
/// [`RunStateSnapshotWire`] as trailing `#[serde(default)]` fields.
///
/// Compatibility note (postcard is positional, non-self-describing): the
/// only direction that holds is FORWARD — an OLDER (v2) decoder reading a
/// NEWER (v3) frame ignores the trailing `place` bytes. The REVERSE — a
/// v3 struct decoding a shorter v2 frame — errors with
/// `DeserializeUnexpectedEnd`, because the missing trailing field has no
/// bytes to read. A trailing `#[serde(default)]` does NOT change that for
/// postcard. So, exactly as with the v1 → v2 bump, mixed-version wire
/// traffic is gated by the handshake's `supported_envelope_versions`
/// intersection, not by byte-level structural back-compat.
pub const ENVELOPE_VERSION: u32 = 3;

/// Sentinel tenant/place for single-tenant, legacy, and standalone
/// deployments. A `None` / empty `place` on the wire resolves to this
/// value server-side, so existing single-tenant clusters behave exactly
/// as before the per-place split.
pub const DEFAULT_PLACE: &str = "__default__";

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

/// Wire-format mirror of [`core::TaintEffect`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaintEffectWire {
    NoSchedule,
    PreferNoSchedule,
}

impl From<core::TaintEffect> for TaintEffectWire {
    fn from(t: core::TaintEffect) -> Self {
        match t {
            core::TaintEffect::NoSchedule => Self::NoSchedule,
            core::TaintEffect::PreferNoSchedule => Self::PreferNoSchedule,
        }
    }
}

impl From<TaintEffectWire> for core::TaintEffect {
    fn from(t: TaintEffectWire) -> Self {
        match t {
            TaintEffectWire::NoSchedule => Self::NoSchedule,
            TaintEffectWire::PreferNoSchedule => Self::PreferNoSchedule,
        }
    }
}

/// Wire-format mirror of [`core::WorkerTaint`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerTaintWire {
    pub key: String,
    pub value: Option<String>,
    pub effect: TaintEffectWire,
}

impl From<core::WorkerTaint> for WorkerTaintWire {
    fn from(t: core::WorkerTaint) -> Self {
        Self {
            key: t.key,
            value: t.value,
            effect: t.effect.into(),
        }
    }
}

impl From<WorkerTaintWire> for core::WorkerTaint {
    fn from(t: WorkerTaintWire) -> Self {
        Self {
            key: t.key,
            value: t.value,
            effect: t.effect.into(),
        }
    }
}

/// Wire-format mirror of [`core::TolerationSpec`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TolerationSpecWire {
    pub key: String,
    pub value: Option<String>,
    pub effect: TaintEffectWire,
}

impl From<core::TolerationSpec> for TolerationSpecWire {
    fn from(t: core::TolerationSpec) -> Self {
        Self {
            key: t.key,
            value: t.value,
            effect: t.effect.into(),
        }
    }
}

impl From<TolerationSpecWire> for core::TolerationSpec {
    fn from(t: TolerationSpecWire) -> Self {
        Self {
            key: t.key,
            value: t.value,
            effect: t.effect.into(),
        }
    }
}

/// Wire-format mirror of [`core::NodeSelector`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeSelectorWire {
    pub required: Vec<String>,
    pub forbidden: Vec<String>,
    pub preferred: Vec<String>,
}

impl From<core::NodeSelector> for NodeSelectorWire {
    fn from(s: core::NodeSelector) -> Self {
        Self {
            required: s.required,
            forbidden: s.forbidden,
            preferred: s.preferred,
        }
    }
}

impl From<NodeSelectorWire> for core::NodeSelector {
    fn from(s: NodeSelectorWire) -> Self {
        Self {
            required: s.required,
            forbidden: s.forbidden,
            preferred: s.preferred,
        }
    }
}

/// Wire-format description of one capability node a worker hosts.
///
/// Workers attach a `Vec<NodeDescriptorWire>` to their [`WorkerHello`]
/// so the control plane can build a live capability catalogue without
/// the workers having to depend on `zbrain-core` (or vice versa). The
/// schema bytes (`input_schema_json`, `output_schema_json`) are
/// pre-serialized JSON text so postcard can round-trip them without
/// going through `serde_json::Value` (which postcard cannot decode).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeDescriptorWire {
    /// Stable slug, e.g. `"trellis-image-to-mesh"`.
    pub id: String,
    /// Capability the executing worker must advertise.
    pub capability: CapabilityWire,
    /// JSON Schema (draft 2020-12) of the node's input shape, as
    /// pre-serialized JSON bytes.
    #[serde(with = "serde_bytes")]
    pub input_schema_json: Vec<u8>,
    /// JSON Schema (draft 2020-12) of the node's output shape, as
    /// pre-serialized JSON bytes.
    #[serde(with = "serde_bytes")]
    pub output_schema_json: Vec<u8>,
    /// Default resource estimate.
    pub default_resource_hint: ResourceHintWire,
    /// Default node selector.
    pub default_selector: NodeSelectorWire,
    /// Schema/contract version.
    pub version: u32,
    /// UI-only display name.
    pub display_name: String,
    /// Icon hint (emoji or named svg asset).
    pub icon: Option<String>,
    /// Tile color (CSS string).
    pub color: Option<String>,
    /// Palette category (e.g. `"3d"`, `"vision"`).
    pub category: Option<String>,
    /// Long-form description.
    pub description: Option<String>,
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
    /// Worker-side schedulable labels (e.g. `gpu:nvidia`, `host:beastpc`,
    /// `vram:>=24gb`). Filtered against [`Assignment::selector`] inside
    /// admission. Empty for legacy callers. Introduced in envelope v2.
    pub labels: BTreeMap<String, String>,
    /// Worker-side taints. Jobs must carry a matching toleration to land
    /// on a tainted worker. Empty for legacy callers. Introduced in
    /// envelope v2.
    pub taints: Vec<WorkerTaintWire>,
    /// Capability-descriptor manifest the worker publishes (one entry
    /// per node the worker is willing to host). The control plane
    /// merges these across every connected worker into the live
    /// capability catalogue. Empty for legacy callers; the field is
    /// appended at the end of the struct so postcard skips it cleanly
    /// when decoding older payloads.
    #[serde(default)]
    pub descriptors: Vec<NodeDescriptorWire>,
    /// Self-reported tenant/place this worker serves. Advisory only — the
    /// server-side [`crate::auth::PeerIdentity`] derived from the bearer
    /// token wins over this value (anti-spoof). `None` for legacy callers
    /// and standalone workers; the server treats `None` as the default
    /// place. Trailing `#[serde(default)]` (a v2 decoder ignores it;
    /// mixed-version traffic is gated by the handshake — see
    /// [`ENVELOPE_VERSION`]). Introduced in envelope v3.
    #[serde(default)]
    pub place: Option<String>,
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

/// Worker → server request for a per-place provider key over the
/// authenticated session. The server resolves the key against the
/// worker's server-authenticated place (the worker cannot name a place)
/// and replies with a [`ServerToWorker::KeyResponse`] carrying the same
/// `request_id`.
///
/// The request carries no secret. The key value travels only in the
/// response, which redacts it in `Debug`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Correlation id echoed back in the matching
    /// [`ServerToWorker::KeyResponse`].
    pub request_id: Uuid,
    /// Provider whose key the worker is requesting (e.g. `"openai"`,
    /// `"fal"`).
    pub provider: String,
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
    /// Request a per-place provider key over the authenticated session.
    ///
    /// Appended after [`OfferDecision`] so existing variant indices are
    /// preserved — older servers never decode this frame, and postcard
    /// decode of older payloads is unaffected (the negotiated
    /// `supported_envelope_versions` intersection gates whether a worker
    /// ever sends it).
    KeyRequest(KeyRequest),
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
    /// Scheduling priority — lower numeric value runs first. Default is
    /// [`core::DEFAULT_PRIORITY`] (128 = mid-band). Introduced in
    /// envelope v2.
    pub priority: u8,
    /// Node selector — `required` labels must all match the worker's
    /// labels, `forbidden` must none match, `preferred` adds to the
    /// tie-break score. Introduced in envelope v2.
    pub selector: NodeSelectorWire,
    /// Tolerations — allow the job to land on workers carrying a matching
    /// taint. Introduced in envelope v2.
    pub tolerations: Vec<TolerationSpecWire>,
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

/// Server→worker delivery of an answer to an earlier `input.request`
/// event raised by a running assignment via
/// [`AssignmentContext::request_input`](crate::worker::AssignmentContext::request_input).
///
/// The `request_id` correlates this response with the pending request the
/// worker is blocked on. `response_json` is the JSON-encoded answer (see
/// the module-level docs for why we use `Vec<u8>` instead of
/// `serde_json::Value`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputResponse {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run whose assignment raised the input request.
    pub run_id: Uuid,
    /// Correlation id echoed from the `input.request` event payload.
    pub request_id: String,
    /// JSON-encoded answer handed back to the worker's pending request.
    #[serde(with = "serde_bytes")]
    pub response_json: Vec<u8>,
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

/// Server → worker response to a [`WorkerToServer::KeyRequest`].
///
/// Carries the resolved provider key (or `None` when the server has no
/// key for the worker's place + provider), plus cache-control metadata.
/// The `key` field is a secret: this struct has a MANUAL [`std::fmt::Debug`]
/// impl that REDACTS it, so the key never reaches a log line, panic
/// message, or test assertion that formats the frame.
#[derive(Clone, Serialize, Deserialize)]
pub struct KeyResponse {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Correlation id echoed from the originating
    /// [`KeyRequest::request_id`].
    pub request_id: Uuid,
    /// The resolved key, or `None` when the server brokers no key for the
    /// worker's place + provider. NEVER logged — see the manual `Debug`
    /// impl below.
    pub key: Option<String>,
    /// Optional cache TTL in seconds for the worker-side cache. `None` =
    /// cache for the session lifetime.
    pub ttl_secs: Option<u64>,
    /// Monotonic version counter; a higher value supersedes a cached key
    /// for the same provider.
    pub version: u64,
}

impl std::fmt::Debug for KeyResponse {
    /// Redacts [`KeyResponse::key`] so the secret never reaches a log
    /// line, panic message, or test assertion that formats the frame.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyResponse")
            .field("envelope_version", &self.envelope_version)
            .field("request_id", &self.request_id)
            .field("key", &self.key.as_ref().map(|_| "<REDACTED>"))
            .field("ttl_secs", &self.ttl_secs)
            .field("version", &self.version)
            .finish()
    }
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
    /// Answer to an `input.request` raised by a running assignment.
    ///
    /// Appended after [`Reject`] so existing variant indices are
    /// preserved — older workers (envelope v1) never receive this frame,
    /// and postcard decode of older payloads is unaffected.
    InputResponse(InputResponse),
    /// Response to a [`WorkerToServer::KeyRequest`] carrying the resolved
    /// per-place provider key.
    ///
    /// Appended after [`InputResponse`] so existing variant indices are
    /// preserved. The carried [`KeyResponse`] redacts its key in `Debug`.
    KeyResponse(KeyResponse),
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
    /// Tenant/place this submission targets. Advisory — when the caller
    /// presents a bearer token, the server-side
    /// [`crate::auth::PeerIdentity`] place wins over this value
    /// (anti-spoof). `None` selects the default place. Trailing
    /// `#[serde(default)]` (a v2 decoder ignores it; mixed-version traffic
    /// is gated by the handshake — see [`ENVELOPE_VERSION`]). Introduced
    /// in envelope v3.
    #[serde(default)]
    pub place: Option<String>,
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
            // `place` has no core counterpart; the orchestrator sets it on
            // the wire struct directly (or leaves it `None` for default).
            place: None,
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

/// Orchestrator request to answer an outstanding `input.request` raised
/// by an in-flight assignment. Routed to the worker currently assigned
/// the run, delivered as a [`ServerToWorker::InputResponse`] frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespondToInputRequest {
    /// Envelope version of this payload. See [`ENVELOPE_VERSION`].
    pub envelope_version: u32,
    /// Identifier of the run whose assignment is awaiting input.
    pub run_id: Uuid,
    /// Correlation id from the `input.request` event payload.
    pub request_id: String,
    /// JSON-encoded answer to forward to the worker.
    #[serde(with = "serde_bytes")]
    pub response_json: Vec<u8>,
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
    /// Tenant/place this run belongs to. Empty string = the default
    /// place. Populated server-side from the submission's resolved place.
    /// Trailing `#[serde(default)]` (a v2 decoder ignores it; mixed-version
    /// traffic is gated by the handshake — see [`ENVELOPE_VERSION`]).
    /// Introduced in envelope v3.
    #[serde(default)]
    pub place: String,
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
            // `place` has no core counterpart; the server overwrites it with
            // the run's resolved place when building the response.
            place: String::new(),
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
    /// Tenant/place this worker serves. Empty string = the default place.
    /// Populated server-side from the worker's registry handle. Trailing
    /// `#[serde(default)]` (a v2 decoder ignores it; mixed-version traffic
    /// is gated by the handshake — see [`ENVELOPE_VERSION`]). Introduced
    /// in envelope v3.
    #[serde(default)]
    pub place: String,
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
            // `place` has no core counterpart; the server overwrites it from
            // the registry handle when building the `ListWorkers` response.
            place: String::new(),
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
            labels: BTreeMap::new(),
            taints: Vec::new(),
            descriptors: Vec::new(),
            place: Some("acme".to_string()),
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
        assert_eq!(decoded.place, original.place);
    }

    // NOTE on postcard envelope compatibility.
    //
    // Postcard is positional and non-self-describing. The compatibility
    // that actually holds (and that the handshake relies on) is the
    // FORWARD direction: a frame that a NEWER peer produced with MORE
    // trailing fields can be decoded by an OLDER peer's struct shape —
    // the older decoder reads exactly its fields and `take_from_bytes`
    // leaves the extra trailing bytes unconsumed. The REVERSE direction
    // — decoding OLDER (shorter) bytes into a NEWER struct that has an
    // extra trailing field — is NOT supported by postcard: it errors
    // with `DeserializeUnexpectedEnd` because the missing field has no
    // bytes to read (a trailing `#[serde(default)]` does not rescue a
    // positional format from EOF). This is exactly why the v1→v2 bump
    // already documents "a v1 peer cannot decode v2 frames — the
    // handshake negotiates the intersection". The same holds v2→v3:
    // wire-level mixing is gated by `supported_envelope_versions`, not by
    // byte-level structural back-compat. The test below pins the real,
    // working guarantee: a v3 frame decodes cleanly under a v2-shaped
    // (place-less) struct, with the trailing `place` bytes ignored.

    /// A v2 [`WorkerHello`] — identical field order minus the trailing
    /// `place` added in v3.
    #[derive(Serialize, Deserialize)]
    struct WorkerHelloV2 {
        envelope_version: u32,
        node_id: String,
        capabilities: Vec<CapabilityWire>,
        tags: BTreeMap<String, String>,
        admission: AdmissionModeWire,
        supported_envelope_versions: Vec<u32>,
        labels: BTreeMap<String, String>,
        taints: Vec<WorkerTaintWire>,
        #[serde(default)]
        descriptors: Vec<NodeDescriptorWire>,
    }

    #[test]
    fn v3_worker_hello_frame_decodes_under_v2_struct_ignoring_place() {
        // A v3 server/peer encodes a WorkerHello WITH a place set.
        let v3 = WorkerHello {
            envelope_version: 3,
            node_id: "modern-worker".to_string(),
            capabilities: Vec::new(),
            tags: BTreeMap::new(),
            admission: AdmissionModeWire::Reactive,
            supported_envelope_versions: vec![3],
            labels: BTreeMap::new(),
            taints: Vec::new(),
            descriptors: Vec::new(),
            place: Some("acme".to_string()),
        };
        let bytes = postcard::to_allocvec(&v3).expect("encode v3 WorkerHello");
        // An older (v2-shaped, place-less) decoder reads its fields and
        // leaves the trailing `place` bytes unconsumed.
        let (decoded, rest): (WorkerHelloV2, &[u8]) =
            postcard::take_from_bytes(&bytes).expect("v2 struct decodes a v3 frame");
        assert_eq!(decoded.node_id, "modern-worker");
        // The trailing `place` bytes were NOT consumed by the v2 struct.
        assert!(
            !rest.is_empty(),
            "the v3 `place` field should remain as trailing bytes"
        );
    }

    /// A v2 [`SubmitRequest`] — same fields minus the trailing v3 `place`.
    #[derive(Serialize, Deserialize)]
    struct SubmitRequestV2 {
        envelope_version: u32,
        workflow_name: String,
        workflow_version: Option<u32>,
        #[serde(with = "serde_bytes")]
        input_json: Vec<u8>,
        required_tags: Vec<String>,
        idempotency_key: Option<String>,
        deadline_ms: Option<u64>,
        wait_for_worker: bool,
        resource_hint: Option<ResourceHintWire>,
    }

    #[test]
    fn v3_submit_request_frame_decodes_under_v2_struct_ignoring_place() {
        let v3 = SubmitRequest {
            envelope_version: 3,
            workflow_name: "summarize".to_string(),
            workflow_version: Some(1),
            input_json: serde_json::to_vec(&serde_json::json!({"k": "v"})).unwrap(),
            required_tags: Vec::new(),
            idempotency_key: None,
            deadline_ms: None,
            wait_for_worker: false,
            resource_hint: None,
            place: Some("acme".to_string()),
        };
        let bytes = postcard::to_allocvec(&v3).expect("encode v3 SubmitRequest");
        let (decoded, rest): (SubmitRequestV2, &[u8]) =
            postcard::take_from_bytes(&bytes).expect("v2 struct decodes a v3 frame");
        assert_eq!(decoded.workflow_name, "summarize");
        assert!(
            !rest.is_empty(),
            "the v3 `place` field should remain as trailing bytes"
        );
    }

    #[test]
    fn submit_request_roundtrips_with_place() {
        let original = SubmitRequest {
            envelope_version: ENVELOPE_VERSION,
            workflow_name: "summarize".to_string(),
            workflow_version: Some(2),
            input_json: serde_json::to_vec(&serde_json::json!({"x": 1})).unwrap(),
            required_tags: vec!["gpu".to_string()],
            idempotency_key: Some("dedupe-1".to_string()),
            deadline_ms: Some(5_000),
            wait_for_worker: true,
            resource_hint: None,
            place: Some("acme".to_string()),
        };
        let decoded = roundtrip(&original);
        assert_eq!(decoded.workflow_name, original.workflow_name);
        assert_eq!(decoded.place, original.place);
        assert_eq!(decoded.required_tags, original.required_tags);
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
            priority: core::DEFAULT_PRIORITY,
            selector: NodeSelectorWire::default(),
            tolerations: Vec::new(),
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
    fn key_request_roundtrips() {
        let original = WorkerToServer::KeyRequest(KeyRequest {
            envelope_version: ENVELOPE_VERSION,
            request_id: Uuid::new_v4(),
            provider: "openai".to_string(),
        });
        let decoded = roundtrip(&original);
        match decoded {
            WorkerToServer::KeyRequest(req) => {
                assert_eq!(req.envelope_version, ENVELOPE_VERSION);
                assert_eq!(req.provider, "openai");
            }
            other => panic!("expected KeyRequest, got {other:?}"),
        }
    }

    #[test]
    fn key_response_roundtrips() {
        let request_id = Uuid::new_v4();
        let original = ServerToWorker::KeyResponse(KeyResponse {
            envelope_version: ENVELOPE_VERSION,
            request_id,
            key: Some("sk-roundtrip-secret".to_string()),
            ttl_secs: Some(300),
            version: 5,
        });
        let decoded = roundtrip(&original);
        match decoded {
            ServerToWorker::KeyResponse(resp) => {
                assert_eq!(resp.envelope_version, ENVELOPE_VERSION);
                assert_eq!(resp.request_id, request_id);
                assert_eq!(resp.key.as_deref(), Some("sk-roundtrip-secret"));
                assert_eq!(resp.ttl_secs, Some(300));
                assert_eq!(resp.version, 5);
            }
            other => panic!("expected KeyResponse, got {other:?}"),
        }
    }

    #[test]
    fn key_response_debug_redacts_key() {
        let resp = KeyResponse {
            envelope_version: ENVELOPE_VERSION,
            request_id: Uuid::new_v4(),
            key: Some("sk-never-log-me".to_string()),
            ttl_secs: None,
            version: 1,
        };
        let rendered = format!("{resp:?}");
        assert!(
            !rendered.contains("sk-never-log-me"),
            "KeyResponse Debug must not contain the secret, got: {rendered}"
        );
        assert!(
            rendered.contains("REDACT"),
            "KeyResponse Debug must signal redaction, got: {rendered}"
        );

        // A None key renders without a redaction marker and without any
        // secret (there is none).
        let empty = KeyResponse {
            envelope_version: ENVELOPE_VERSION,
            request_id: Uuid::new_v4(),
            key: None,
            ttl_secs: None,
            version: 0,
        };
        let rendered_empty = format!("{empty:?}");
        assert!(rendered_empty.contains("None"));
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
