//! `#[napi(object)]` wrappers around the postcard wire types in
//! [`blazen_peer::protocol`].
//!
//! These mirror the native Rust structs but use plain JS-friendly
//! representations: JSON values become `serde_json::Value`, byte vectors
//! that carry pre-encoded JSON become `serde_json::Value` at the
//! boundary, and UUIDs are exchanged as strings.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use uuid::Uuid;

use blazen_peer::protocol::{
    DerefRequest, DerefResponse, ENVELOPE_VERSION, ReleaseRequest, ReleaseResponse,
    RemoteRefDescriptor, SubWorkflowRequest, SubWorkflowResponse,
};

// ---------------------------------------------------------------------------
// SubWorkflowRequest / Response
// ---------------------------------------------------------------------------

/// Request to invoke a sub-workflow on a remote peer.
#[napi(object)]
pub struct JsSubWorkflowRequest {
    /// Symbolic name of the workflow to invoke on the remote peer.
    #[napi(js_name = "workflowName")]
    pub workflow_name: String,
    /// Ordered list of step IDs to execute. Empty means "use the
    /// remote workflow's default step set".
    #[napi(js_name = "stepIds")]
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step. Any
    /// JSON-serializable value is accepted.
    pub input: serde_json::Value,
    /// Optional wall-clock timeout in seconds.
    #[napi(js_name = "timeoutSecs")]
    pub timeout_secs: Option<u32>,
}

impl JsSubWorkflowRequest {
    /// Convert this JS object into a native [`SubWorkflowRequest`].
    #[allow(clippy::missing_errors_doc)]
    pub fn into_native(self) -> std::result::Result<SubWorkflowRequest, serde_json::Error> {
        SubWorkflowRequest::new(
            self.workflow_name,
            self.step_ids,
            &self.input,
            self.timeout_secs.map(u64::from),
        )
    }
}

/// Result of a remote sub-workflow invocation.
#[napi(object)]
pub struct JsSubWorkflowResponse {
    /// Envelope version of the wire payload.
    #[napi(js_name = "envelopeVersion")]
    pub envelope_version: u32,
    /// Public state values exported by the sub-workflow, decoded from
    /// JSON.
    #[napi(js_name = "stateJson")]
    pub state_json: HashMap<String, serde_json::Value>,
    /// Optional terminal result, decoded from JSON. `None` when the
    /// workflow exited without producing one.
    pub result: Option<serde_json::Value>,
    /// Descriptors for any session refs the sub-workflow registered
    /// that the parent should be able to dereference remotely. Keyed by
    /// the registry UUID rendered as a string.
    #[napi(js_name = "remoteRefs")]
    pub remote_refs: HashMap<String, JsPeerRemoteRefDescriptor>,
    /// Error message if the sub-workflow failed. When `Some`, callers
    /// should ignore `result` and `state_json`.
    pub error: Option<String>,
}

impl JsSubWorkflowResponse {
    /// Build a JS-shaped response from a native [`SubWorkflowResponse`].
    #[allow(clippy::missing_errors_doc)]
    pub fn from_native(value: SubWorkflowResponse) -> std::result::Result<Self, serde_json::Error> {
        let state_json = value.state_values()?;
        let result = value.result_value()?;
        let remote_refs = value
            .remote_refs
            .into_iter()
            .map(|(k, v)| (k.to_string(), JsPeerRemoteRefDescriptor::from_native(v)))
            .collect();
        Ok(Self {
            envelope_version: value.envelope_version,
            state_json,
            result,
            remote_refs,
            error: value.error,
        })
    }
}

// ---------------------------------------------------------------------------
// Deref / Release
// ---------------------------------------------------------------------------

/// Request to dereference a remote session ref.
#[napi(object)]
pub struct JsDerefRequest {
    /// Envelope version of the wire payload.
    #[napi(js_name = "envelopeVersion")]
    pub envelope_version: u32,
    /// UUID of the registry entry on the origin node, as a string.
    #[napi(js_name = "refUuid")]
    pub ref_uuid: String,
}

impl JsDerefRequest {
    /// Parse the UUID and convert into a native [`DerefRequest`].
    #[allow(clippy::missing_errors_doc)]
    pub fn into_native(self) -> std::result::Result<DerefRequest, uuid::Error> {
        Ok(DerefRequest {
            envelope_version: self.envelope_version,
            ref_uuid: Uuid::parse_str(&self.ref_uuid)?,
        })
    }
}

/// Response containing the dereferenced bytes for a session ref.
#[napi(object)]
pub struct JsDerefResponse {
    /// Envelope version of the wire payload.
    #[napi(js_name = "envelopeVersion")]
    pub envelope_version: u32,
    /// Raw payload returned by the origin node. Carried as a `Buffer`
    /// on the JS side.
    pub payload: Buffer,
}

impl JsDerefResponse {
    /// Build a JS-shaped response from a native [`DerefResponse`].
    #[must_use]
    pub fn from_native(value: DerefResponse) -> Self {
        Self {
            envelope_version: value.envelope_version,
            payload: Buffer::from(value.payload),
        }
    }
}

/// Request to release (drop) a remote session ref.
#[napi(object)]
pub struct JsReleaseRequest {
    /// Envelope version of the wire payload.
    #[napi(js_name = "envelopeVersion")]
    pub envelope_version: u32,
    /// UUID of the registry entry to drop on the origin node, as a
    /// string.
    #[napi(js_name = "refUuid")]
    pub ref_uuid: String,
}

impl JsReleaseRequest {
    /// Parse the UUID and convert into a native [`ReleaseRequest`].
    #[allow(clippy::missing_errors_doc)]
    pub fn into_native(self) -> std::result::Result<ReleaseRequest, uuid::Error> {
        Ok(ReleaseRequest {
            envelope_version: self.envelope_version,
            ref_uuid: Uuid::parse_str(&self.ref_uuid)?,
        })
    }
}

/// Acknowledgement for a [`JsReleaseRequest`].
#[napi(object)]
pub struct JsReleaseResponse {
    /// Envelope version of the wire payload.
    #[napi(js_name = "envelopeVersion")]
    pub envelope_version: u32,
    /// `true` if the registry entry was found and dropped, `false` if
    /// it was already gone.
    pub released: bool,
}

impl JsReleaseResponse {
    /// Build a JS-shaped response from a native [`ReleaseResponse`].
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_native(value: ReleaseResponse) -> Self {
        Self {
            envelope_version: value.envelope_version,
            released: value.released,
        }
    }
}

// ---------------------------------------------------------------------------
// RemoteRefDescriptor
// ---------------------------------------------------------------------------

/// Metadata describing a remote session ref handed back by an
/// `invokeSubWorkflow` call.
#[napi(object)]
pub struct JsPeerRemoteRefDescriptor {
    /// Stable identifier of the node that owns the underlying value.
    #[napi(js_name = "originNodeId")]
    pub origin_node_id: String,
    /// Type tag mirroring the Rust `SessionRefSerializable::blazen_type_tag`.
    #[napi(js_name = "typeTag")]
    pub type_tag: String,
    /// Wall-clock creation time on the origin node, in milliseconds
    /// since the Unix epoch.
    #[napi(js_name = "createdAtEpochMs")]
    pub created_at_epoch_ms: i64,
}

impl JsPeerRemoteRefDescriptor {
    /// Build a JS-shaped descriptor from a native [`RemoteRefDescriptor`].
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_native(value: RemoteRefDescriptor) -> Self {
        Self {
            origin_node_id: value.origin_node_id,
            type_tag: value.type_tag,
            created_at_epoch_ms: value.created_at_epoch_ms as i64,
        }
    }
}

/// Re-exported envelope version for parity with native callers that
/// want to assert on the version they are speaking.
#[must_use]
pub const fn envelope_version() -> u32 {
    ENVELOPE_VERSION
}
