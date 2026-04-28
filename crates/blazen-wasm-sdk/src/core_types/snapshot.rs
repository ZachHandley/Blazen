//! `wasm-bindgen` wrapper for [`blazen_core::WorkflowSnapshot`].
//!
//! `WorkflowSnapshot` is the on-the-wire representation of a paused
//! workflow. JS callers obtain one from
//! [`crate::handler::WasmWorkflowHandler::pause`] (currently surfaced as a
//! plain JS object) and can pass it back into
//! [`crate::workflow::WasmWorkflow::resume_from_snapshot`] to resume the
//! workflow. This module gives the snapshot a typed JS class so callers can
//! also serialise it to / deserialise it from JSON or `MessagePack` without
//! going through `serde-wasm-bindgen` themselves, and inspect snapshot
//! metadata without unpacking the whole tree.

use blazen_core::WorkflowSnapshot;
use serde::Serialize;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Marshalling helper
// ---------------------------------------------------------------------------

/// Convert a `Serialize` value into a `JsValue` shaped as a plain JS object.
fn marshal_to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&format!("marshal failed: {e}")))
}

// ---------------------------------------------------------------------------
// WasmWorkflowSnapshot
// ---------------------------------------------------------------------------

/// JS-facing handle for a [`blazen_core::WorkflowSnapshot`].
///
/// Holds the snapshot in its native Rust shape so the binding-free
/// to/from JSON and to/from `MessagePack` paths preserve byte-for-byte
/// fidelity. JS callers that only need the `serde-wasm-bindgen` JSON shape
/// can keep using a plain object — this wrapper is opt-in for users who
/// want the typed surface.
#[wasm_bindgen(js_name = "WorkflowSnapshot")]
pub struct WasmWorkflowSnapshot {
    inner: WorkflowSnapshot,
}

impl WasmWorkflowSnapshot {
    /// Wrap an existing [`WorkflowSnapshot`].
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn from_inner(inner: WorkflowSnapshot) -> Self {
        Self { inner }
    }

    /// Borrow the underlying [`WorkflowSnapshot`].
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &WorkflowSnapshot {
        &self.inner
    }

    /// Consume `self` and return the underlying [`WorkflowSnapshot`].
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> WorkflowSnapshot {
        self.inner
    }
}

#[wasm_bindgen(js_class = "WorkflowSnapshot")]
impl WasmWorkflowSnapshot {
    /// Reconstruct a snapshot from a plain JS object (typically the value
    /// returned by [`crate::handler::WasmWorkflowHandler::pause`]).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `value` does not match the snapshot
    /// schema.
    #[wasm_bindgen(js_name = "fromJsObject")]
    pub fn from_js_object(value: JsValue) -> Result<WasmWorkflowSnapshot, JsValue> {
        let inner: WorkflowSnapshot = serde_wasm_bindgen::from_value(value).map_err(|e| {
            JsValue::from_str(&format!("WorkflowSnapshot.fromJsObject failed: {e}"))
        })?;
        Ok(Self { inner })
    }

    /// Serialise the snapshot to a JSON string.
    ///
    /// Mirrors [`WorkflowSnapshot::to_json`].
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if serialisation fails.
    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.inner
            .to_json()
            .map_err(|e| JsValue::from_str(&format!("WorkflowSnapshot.toJson failed: {e}")))
    }

    /// Deserialise a snapshot from a JSON string.
    ///
    /// Mirrors [`WorkflowSnapshot::from_json`]. Validates the snapshot's
    /// version against [`blazen_core::SNAPSHOT_VERSION`] and rejects
    /// snapshots written by a newer reader.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JSON is malformed or the snapshot
    /// version is unsupported.
    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(json: &str) -> Result<WasmWorkflowSnapshot, JsValue> {
        WorkflowSnapshot::from_json(json)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&format!("WorkflowSnapshot.fromJson failed: {e}")))
    }

    /// Serialise the snapshot to `MessagePack` bytes.
    ///
    /// Mirrors [`WorkflowSnapshot::to_msgpack`]. Compact and especially
    /// efficient for snapshots that contain
    /// [`StateValue::Bytes`](blazen_core::StateValue::Bytes) payloads.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if serialisation fails.
    #[wasm_bindgen(js_name = "toMsgpack")]
    pub fn to_msgpack(&self) -> Result<js_sys::Uint8Array, JsValue> {
        let bytes = self
            .inner
            .to_msgpack()
            .map_err(|e| JsValue::from_str(&format!("WorkflowSnapshot.toMsgpack failed: {e}")))?;
        Ok(js_sys::Uint8Array::from(bytes.as_slice()))
    }

    /// Deserialise a snapshot from `MessagePack` bytes.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the bytes are malformed or the
    /// snapshot version is unsupported.
    #[wasm_bindgen(js_name = "fromMsgpack")]
    pub fn from_msgpack(bytes: &[u8]) -> Result<WasmWorkflowSnapshot, JsValue> {
        WorkflowSnapshot::from_msgpack(bytes)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&format!("WorkflowSnapshot.fromMsgpack failed: {e}")))
    }

    /// Marshal the snapshot to a plain JS object (the inverse of
    /// [`from_js_object`](Self::from_js_object)).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if marshalling fails.
    #[wasm_bindgen(js_name = "toJsObject")]
    pub fn to_js_object(&self) -> Result<JsValue, JsValue> {
        marshal_to_js(&self.inner)
    }

    /// Snapshot format version number.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn version(&self) -> u32 {
        self.inner.version
    }

    /// Name of the workflow that produced this snapshot.
    #[wasm_bindgen(getter, js_name = "workflowName")]
    #[must_use]
    pub fn workflow_name(&self) -> String {
        self.inner.workflow_name.clone()
    }

    /// Run id (UUID string) of the workflow that produced this snapshot.
    #[wasm_bindgen(getter, js_name = "runId")]
    #[must_use]
    pub fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// Wall-clock timestamp at which the snapshot was captured, formatted
    /// as RFC 3339.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }
}
