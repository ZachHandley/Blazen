//! Typed Node binding for [`blazen_core::WorkflowSnapshot`].
//!
//! The Python and JS workflow APIs already expose `pause()` /
//! `resume()` paths that round-trip a snapshot through a JSON string.
//! [`JsWorkflowSnapshot`] is a strongly-typed handle around the same
//! data so JS callers can serialize to / from `MessagePack` as well as
//! JSON, and inspect the snapshot's metadata fields directly without
//! re-parsing the JSON envelope.

use blazen_core::WorkflowSnapshot;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::workflow_error_to_napi;

/// A serialized workflow state captured at a pause point. Mirrors
/// [`blazen_core::WorkflowSnapshot`].
///
/// Construct via [`JsWorkflowSnapshot::fromJson`] or
/// [`JsWorkflowSnapshot::fromMsgpack`], or obtain one from a workflow
/// handler. Convert back to bytes via [`JsWorkflowSnapshot::toJson`] /
/// [`JsWorkflowSnapshot::toMsgpack`] for storage or transport.
#[napi(js_name = "WorkflowSnapshot")]
pub struct JsWorkflowSnapshot {
    pub(crate) inner: WorkflowSnapshot,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsWorkflowSnapshot {
    /// Parse a snapshot from a JSON string.
    #[napi(factory, js_name = "fromJson")]
    pub fn from_json(json: String) -> Result<Self> {
        let inner = WorkflowSnapshot::from_json(&json).map_err(workflow_error_to_napi)?;
        Ok(Self { inner })
    }

    /// Parse a snapshot from MessagePack-encoded bytes.
    #[napi(factory, js_name = "fromMsgpack")]
    pub fn from_msgpack(bytes: Buffer) -> Result<Self> {
        let inner =
            WorkflowSnapshot::from_msgpack(bytes.as_ref()).map_err(workflow_error_to_napi)?;
        Ok(Self { inner })
    }

    /// Serialize this snapshot to a JSON string.
    #[napi(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String> {
        self.inner.to_json().map_err(workflow_error_to_napi)
    }

    /// Serialize this snapshot to a pretty-printed JSON string.
    #[napi(js_name = "toJsonPretty")]
    pub fn to_json_pretty(&self) -> Result<String> {
        self.inner.to_json_pretty().map_err(workflow_error_to_napi)
    }

    /// Serialize this snapshot to MessagePack-encoded bytes. Returns a
    /// `Buffer` so it survives the napi boundary unchanged.
    #[napi(js_name = "toMsgpack")]
    pub fn to_msgpack(&self) -> Result<Buffer> {
        let bytes = self.inner.to_msgpack().map_err(workflow_error_to_napi)?;
        Ok(Buffer::from(bytes))
    }

    /// Snapshot format version.
    #[napi(getter)]
    pub fn version(&self) -> u32 {
        self.inner.version
    }

    /// Name of the workflow that produced this snapshot.
    #[napi(getter, js_name = "workflowName")]
    pub fn workflow_name(&self) -> String {
        self.inner.workflow_name.clone()
    }

    /// Run ID of the workflow that produced this snapshot.
    #[napi(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// Wall-clock time when the snapshot was captured (RFC 3339).
    #[napi(getter)]
    pub fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    /// Number of context-state entries.
    #[napi(getter, js_name = "contextStateLen")]
    pub fn context_state_len(&self) -> u32 {
        u32::try_from(self.inner.context_state.len()).unwrap_or(u32::MAX)
    }

    /// Number of pending events still in the routing queue.
    #[napi(getter, js_name = "pendingEventsLen")]
    pub fn pending_events_len(&self) -> u32 {
        u32::try_from(self.inner.pending_events.len()).unwrap_or(u32::MAX)
    }

    /// Snapshot metadata as a JSON object. Includes the input-request
    /// stash under `__input_request` when the snapshot was captured at
    /// an `InputRequestEvent` pause.
    #[napi(getter)]
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::Value::Object(self.inner.metadata.clone().into_iter().collect())
    }
}

impl JsWorkflowSnapshot {
    #[allow(dead_code)]
    pub(crate) fn from_inner(inner: WorkflowSnapshot) -> Self {
        Self { inner }
    }

    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> WorkflowSnapshot {
        self.inner
    }
}
