//! Python wrapper for [`blazen_core::snapshot::WorkflowSnapshot`].
//!
//! Snapshots have historically been threaded through the binding as
//! JSON strings (see `Workflow.resume(snapshot_json, ...)`); this module
//! adds a typed handle so Python callers can read/write the binary
//! `MessagePack` form too -- it's substantially more compact for
//! snapshots whose `context_state` carries `StateValue.Bytes` payloads.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::snapshot::WorkflowSnapshot;

use crate::convert::json_to_py;
use crate::error::BlazenPyError;

use super::value::PyStateValue;

// ---------------------------------------------------------------------------
// PyWorkflowSnapshot
// ---------------------------------------------------------------------------

/// Typed handle for a [`WorkflowSnapshot`].
///
/// Construct from a captured snapshot via the `Workflow.snapshot()`
/// helper (which still returns the JSON form for backwards
/// compatibility) by feeding the JSON into
/// `WorkflowSnapshot.from_json(json)`. The class also exposes the binary
/// `MessagePack` codec via `to_msgpack` / `from_msgpack`.
#[gen_stub_pyclass]
#[pyclass(name = "WorkflowSnapshot", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyWorkflowSnapshot {
    pub(crate) inner: WorkflowSnapshot,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWorkflowSnapshot {
    /// Snapshot format version. See
    /// [`blazen_core::snapshot::SNAPSHOT_VERSION`].
    #[getter]
    fn version(&self) -> u32 {
        self.inner.version
    }

    /// Symbolic name of the workflow that produced this snapshot.
    #[getter]
    fn workflow_name(&self) -> &str {
        &self.inner.workflow_name
    }

    /// Run identifier (UUID, as a string).
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    /// Wall-clock capture time as an ISO-8601 string.
    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    /// Decoded view of the captured `context_state` map. Each value is
    /// returned as a typed [`PyStateValue`] so callers can distinguish
    /// JSON, raw-bytes, and native-pickle entries.
    #[gen_stub(override_return_type(type_repr = "dict[str, StateValue]", imports = ()))]
    fn context_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.context_state {
            let py_val = Py::new(py, PyStateValue { inner: v.clone() })?;
            dict.set_item(k, py_val)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Fan-in collected events keyed by event-type identifier.
    #[gen_stub(override_return_type(type_repr = "dict[str, list[typing.Any]]", imports = ("typing",)))]
    fn collected_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, vs) in &self.inner.collected_events {
            let list = PyList::empty(py);
            for v in vs {
                list.append(json_to_py(py, v)?)?;
            }
            dict.set_item(k, list)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Pending events that were still in the routing channel at
    /// snapshot time. Each item is a `dict` with `event_type`, `data`,
    /// and optional `source_step` keys.
    #[gen_stub(override_return_type(type_repr = "list[dict[str, typing.Any]]", imports = ("typing",)))]
    fn pending_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = PyList::empty(py);
        for ev in &self.inner.pending_events {
            let entry = PyDict::new(py);
            entry.set_item("event_type", &ev.event_type)?;
            entry.set_item("data", json_to_py(py, &ev.data)?)?;
            entry.set_item("source_step", ev.source_step.clone())?;
            list.append(entry)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Free-form metadata captured alongside the snapshot. Includes
    /// `run_id`, `workflow_name`, and any user-defined keys set via
    /// `Context.set_metadata`.
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.metadata {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Serialize the snapshot to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| BlazenPyError::from(e).into())
    }

    /// Pretty-printed JSON form. Useful for debugging snapshots by
    /// hand.
    fn to_json_pretty(&self) -> PyResult<String> {
        self.inner
            .to_json_pretty()
            .map_err(|e| BlazenPyError::from(e).into())
    }

    /// Deserialize a snapshot from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = WorkflowSnapshot::from_json(json).map_err(BlazenPyError::from)?;
        Ok(Self { inner })
    }

    /// Serialize the snapshot to `MessagePack` bytes.
    ///
    /// `MessagePack` is more compact than JSON and avoids the per-byte
    /// overhead JSON imposes on `StateValue.Bytes` payloads.
    fn to_msgpack(&self) -> PyResult<Vec<u8>> {
        self.inner
            .to_msgpack()
            .map_err(|e| BlazenPyError::from(e).into())
    }

    /// Deserialize a snapshot from `MessagePack` bytes.
    #[staticmethod]
    fn from_msgpack(data: Vec<u8>) -> PyResult<Self> {
        let inner = WorkflowSnapshot::from_msgpack(&data).map_err(BlazenPyError::from)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkflowSnapshot(workflow_name={:?}, run_id={}, version={}, pending={}, refs={})",
            self.inner.workflow_name,
            self.inner.run_id,
            self.inner.version,
            self.inner.pending_events.len(),
            self.inner.context_state.len(),
        )
    }
}

impl From<WorkflowSnapshot> for PyWorkflowSnapshot {
    fn from(inner: WorkflowSnapshot) -> Self {
        Self { inner }
    }
}

impl From<PyWorkflowSnapshot> for WorkflowSnapshot {
    fn from(p: PyWorkflowSnapshot) -> Self {
        p.inner
    }
}
