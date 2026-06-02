//! `#[napi(object)]` wrappers around the [`blazen_controlplane`] +
//! [`blazen_core::distributed`] types exchanged by the JS-facing
//! control-plane client and worker.
//!
//! These mirror the native Rust structs but lean on plain JS-friendly
//! representations: UUIDs are exchanged as strings, JSON values become
//! `serde_json::Value`, byte vectors that carry pre-encoded JSON become
//! [`Buffer`]s, and admission modes are surfaced as a tagged-union
//! object so JS callers can pattern-match on `type`.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_controlplane::protocol::Assignment as WireAssignment;
use blazen_core::distributed::{
    AdmissionMode, RunEvent, RunStateSnapshot, RunStatus, WorkerCapability, WorkerInfo,
};

// ---------------------------------------------------------------------------
// WorkerCapability
// ---------------------------------------------------------------------------

/// Capability advertised by a worker at handshake time. Mirrors
/// [`blazen_core::distributed::WorkerCapability`].
#[napi(object)]
pub struct JsWorkerCapability {
    /// Capability tag, e.g. `"workflow:summarize"`.
    pub kind: String,
    /// Capability version. Workers and orchestrators only match when both
    /// values agree.
    pub version: u32,
}

impl From<JsWorkerCapability> for WorkerCapability {
    fn from(value: JsWorkerCapability) -> Self {
        Self {
            kind: value.kind,
            version: value.version,
        }
    }
}

impl From<&WorkerCapability> for JsWorkerCapability {
    fn from(value: &WorkerCapability) -> Self {
        Self {
            kind: value.kind.clone(),
            version: value.version,
        }
    }
}

// ---------------------------------------------------------------------------
// AdmissionMode
// ---------------------------------------------------------------------------

/// JS-facing label that drives [`JsAdmissionMode::r#type`]. Carries the
/// same three variants as [`AdmissionMode`] but as a plain string union
/// rather than a tagged enum (napi-rs `#[napi(object)]` does not yet
/// support discriminated-union codegen for enums with associated data).
#[napi(string_enum)]
pub enum JsAdmissionModeTag {
    Fixed,
    Reactive,
    VramBudget,
}

/// JS-facing admission mode. The `type` field discriminates the variant:
/// `'Fixed'` requires `maxInFlight`, `'VramBudget'` requires `totalMb`,
/// and `'Reactive'` has no payload fields.
#[napi(object)]
pub struct JsAdmissionMode {
    /// Discriminator: `'Fixed'`, `'Reactive'`, or `'VramBudget'`.
    #[napi(js_name = "type")]
    pub r#type: JsAdmissionModeTag,
    /// In-flight cap for `Fixed`. `None` for the other variants.
    #[napi(js_name = "maxInFlight")]
    pub max_in_flight: Option<u32>,
    /// VRAM budget in megabytes for `VramBudget`. `None` for the other
    /// variants.
    #[napi(js_name = "totalMb")]
    pub total_mb: Option<BigInt>,
}

impl JsAdmissionMode {
    /// Decode the JS-facing tagged union back into the native enum.
    ///
    /// # Errors
    /// Returns an error if the tag is `Fixed` without `maxInFlight` or
    /// `VramBudget` without `totalMb`.
    pub fn into_native(self) -> napi::Result<AdmissionMode> {
        match self.r#type {
            JsAdmissionModeTag::Fixed => Ok(AdmissionMode::Fixed {
                max_in_flight: self.max_in_flight.ok_or_else(|| {
                    napi::Error::from_reason(
                        "AdmissionMode { type: 'Fixed' } requires `maxInFlight`",
                    )
                })?,
            }),
            JsAdmissionModeTag::Reactive => Ok(AdmissionMode::Reactive),
            JsAdmissionModeTag::VramBudget => Ok(AdmissionMode::VramBudget {
                max_vram_mb: self
                    .total_mb
                    .ok_or_else(|| {
                        napi::Error::from_reason(
                            "AdmissionMode { type: 'VramBudget' } requires `totalMb`",
                        )
                    })?
                    .get_u64()
                    .1,
            }),
        }
    }
}

impl From<&AdmissionMode> for JsAdmissionMode {
    fn from(mode: &AdmissionMode) -> Self {
        match mode {
            AdmissionMode::Fixed { max_in_flight } => Self {
                r#type: JsAdmissionModeTag::Fixed,
                max_in_flight: Some(*max_in_flight),
                total_mb: None,
            },
            AdmissionMode::Reactive => Self {
                r#type: JsAdmissionModeTag::Reactive,
                max_in_flight: None,
                total_mb: None,
            },
            AdmissionMode::VramBudget { max_vram_mb } => Self {
                r#type: JsAdmissionModeTag::VramBudget,
                max_in_flight: None,
                total_mb: Some(BigInt::from(*max_vram_mb)),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// Worker-facing view of an assignment dispatched by the control plane.
/// JSON input is surfaced as a [`Buffer`] so handlers can decide whether
/// to decode (single-shot) or stream the bytes onward.
#[napi(object)]
pub struct JsAssignment {
    /// Run identifier, rendered as a UUID string.
    #[napi(js_name = "runId")]
    pub run_id: String,
    /// Symbolic workflow name.
    #[napi(js_name = "workflowName")]
    pub workflow_name: String,
    /// Optional workflow version. `None` lets the worker use whichever
    /// version it has registered.
    #[napi(js_name = "workflowVersion")]
    pub workflow_version: Option<u32>,
    /// JSON-encoded initial input as raw bytes.
    #[napi(js_name = "inputJson")]
    pub input_json: Buffer,
    /// Optional deadline in milliseconds. `None` = no timeout.
    #[napi(js_name = "deadlineMs")]
    pub deadline_ms: Option<BigInt>,
    /// 1-indexed attempt counter. Incremented on re-dispatch.
    pub attempt: u32,
}

impl From<&WireAssignment> for JsAssignment {
    fn from(value: &WireAssignment) -> Self {
        Self {
            run_id: value.run_id.to_string(),
            workflow_name: value.workflow_name.clone(),
            workflow_version: value.workflow_version,
            input_json: Buffer::from(value.input_json.clone()),
            deadline_ms: value.deadline_ms.map(BigInt::from),
            attempt: value.attempt,
        }
    }
}

// ---------------------------------------------------------------------------
// RunStateSnapshot
// ---------------------------------------------------------------------------

/// JS-facing run status. Mirrors [`RunStatus`].
#[napi(string_enum)]
pub enum JsRunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl From<RunStatus> for JsRunStatus {
    fn from(status: RunStatus) -> Self {
        match status {
            RunStatus::Pending => Self::Pending,
            RunStatus::Running => Self::Running,
            RunStatus::Completed => Self::Completed,
            RunStatus::Failed => Self::Failed,
            RunStatus::Cancelled => Self::Cancelled,
        }
    }
}

/// Snapshot of a workflow run's state. Mirrors [`RunStateSnapshot`].
#[napi(object)]
pub struct JsRunStateSnapshot {
    /// Run identifier, rendered as a UUID string.
    #[napi(js_name = "runId")]
    pub run_id: String,
    /// Current status.
    pub status: JsRunStatus,
    /// Wall-clock submission time in epoch milliseconds.
    #[napi(js_name = "startedAtMs")]
    pub started_at_ms: BigInt,
    /// Wall-clock completion time, `None` until terminal.
    #[napi(js_name = "completedAtMs")]
    pub completed_at_ms: Option<BigInt>,
    /// `Some(node_id)` once the run has been routed to a worker.
    #[napi(js_name = "assignedTo")]
    pub assigned_to: Option<String>,
    /// Wall-clock of the most recent event for this run, if any.
    #[napi(js_name = "lastEventAtMs")]
    pub last_event_at_ms: Option<BigInt>,
    /// Terminal output JSON value if `status == 'Completed'`.
    pub output: Option<serde_json::Value>,
    /// Error message if `status == 'Failed'`.
    pub error: Option<String>,
}

impl From<RunStateSnapshot> for JsRunStateSnapshot {
    fn from(value: RunStateSnapshot) -> Self {
        Self {
            run_id: value.run_id.to_string(),
            status: value.status.into(),
            started_at_ms: BigInt::from(value.started_at_ms),
            completed_at_ms: value.completed_at_ms.map(BigInt::from),
            assigned_to: value.assigned_to,
            last_event_at_ms: value.last_event_at_ms.map(BigInt::from),
            output: value.output,
            error: value.error,
        }
    }
}

// ---------------------------------------------------------------------------
// RunEvent
// ---------------------------------------------------------------------------

/// Event emitted during a run. Mirrors [`RunEvent`].
#[napi(object)]
pub struct JsRunEvent {
    /// Run identifier, rendered as a UUID string.
    #[napi(js_name = "runId")]
    pub run_id: String,
    /// Caller-supplied event type tag (e.g. `"step.completed"`).
    #[napi(js_name = "eventType")]
    pub event_type: String,
    /// Free-form JSON payload.
    pub data: serde_json::Value,
    /// Wall-clock emission time in epoch milliseconds.
    #[napi(js_name = "timestampMs")]
    pub timestamp_ms: BigInt,
}

impl From<RunEvent> for JsRunEvent {
    fn from(value: RunEvent) -> Self {
        Self {
            run_id: value.run_id.to_string(),
            event_type: value.event_type,
            data: value.data,
            timestamp_ms: BigInt::from(value.timestamp_ms),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerInfo
// ---------------------------------------------------------------------------

/// Summary of a connected worker returned by
/// [`crate::controlplane::client::JsControlPlaneClient::list_workers`].
#[napi(object)]
pub struct JsWorkerInfo {
    /// Stable identifier of the worker.
    #[napi(js_name = "nodeId")]
    pub node_id: String,
    /// Capabilities advertised by this worker at handshake.
    pub capabilities: Vec<JsWorkerCapability>,
    /// Free-form `key=value` tags this worker advertised.
    pub tags: HashMap<String, String>,
    /// Last reported in-flight assignment count.
    #[napi(js_name = "inFlight")]
    pub in_flight: u32,
    /// Wall-clock connection time in epoch milliseconds.
    #[napi(js_name = "connectedAtMs")]
    pub connected_at_ms: BigInt,
}

impl From<WorkerInfo> for JsWorkerInfo {
    fn from(value: WorkerInfo) -> Self {
        Self {
            node_id: value.node_id,
            capabilities: value.capabilities.iter().map(Into::into).collect(),
            tags: value.tags.into_iter().collect(),
            in_flight: value.in_flight,
            connected_at_ms: BigInt::from(value.connected_at_ms),
        }
    }
}

// ---------------------------------------------------------------------------
// mTLS options
// ---------------------------------------------------------------------------

/// PEM file paths for mTLS configuration. Used by
/// [`crate::controlplane::client::JsControlPlaneClient::connect`].
#[napi(object)]
pub struct JsMtlsOptions {
    /// Path to the client certificate PEM file.
    pub cert: String,
    /// Path to the client private-key PEM file.
    pub key: String,
    /// Path to the CA PEM file used to authenticate the server.
    pub ca: String,
}

/// Options bag passed to `ControlPlaneClient.connect`. The `mtls`
/// field, when present, swaps the connection into mTLS mode.
#[napi(object)]
pub struct JsClientConnectOptions {
    /// mTLS configuration. `None` = plaintext.
    pub mtls: Option<JsMtlsOptions>,
    /// Bearer token attached to control-plane requests. `None` = anonymous.
    pub bearer_token: Option<String>,
}

// ---------------------------------------------------------------------------
// SubmitWorkflow options
// ---------------------------------------------------------------------------

/// Options bag for
/// [`crate::controlplane::client::JsControlPlaneClient::submit_workflow`].
#[napi(object)]
pub struct JsSubmitWorkflowOptions {
    /// Symbolic name of the workflow to run.
    #[napi(js_name = "workflowName")]
    pub workflow_name: String,
    /// JSON-serializable initial input.
    pub input: serde_json::Value,
    /// Optional workflow version. `None` = latest.
    #[napi(js_name = "workflowVersion")]
    pub workflow_version: Option<u32>,
    /// Required tags. Each `key=value` (or `key=*`) is AND'd. `None`
    /// means no tag predicate.
    #[napi(js_name = "requiredTags")]
    pub required_tags: Option<Vec<String>>,
    /// Optional dedupe key for at-most-one-run-per-window semantics.
    #[napi(js_name = "idempotencyKey")]
    pub idempotency_key: Option<String>,
    /// Optional deadline in milliseconds from submission.
    #[napi(js_name = "deadlineMs")]
    pub deadline_ms: Option<BigInt>,
    /// If `true`, queue the submission until a matching worker appears.
    /// Defaults to `false`.
    #[napi(js_name = "waitForWorker")]
    pub wait_for_worker: Option<bool>,
}

// ---------------------------------------------------------------------------
// SubscribeAll options
// ---------------------------------------------------------------------------

/// Options bag for
/// [`crate::controlplane::client::JsControlPlaneClient::subscribe_all`].
#[napi(object)]
pub struct JsSubscribeAllOptions {
    /// Tag predicates the events must match.
    #[napi(js_name = "requiredTags")]
    pub required_tags: Option<Vec<String>>,
}
