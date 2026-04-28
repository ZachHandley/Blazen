//! Python wrappers for the postcard-serializable peer protocol types.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use uuid::Uuid;

use blazen_peer::{
    DerefRequest, DerefResponse, ENVELOPE_VERSION, ReleaseRequest, ReleaseResponse,
    RemoteRefDescriptor, SubWorkflowRequest, SubWorkflowResponse,
};

use crate::convert::{dict_to_json, json_to_py};
use crate::peer::error::PeerException;

/// Request to invoke a sub-workflow on a remote peer.
#[gen_stub_pyclass]
#[pyclass(name = "SubWorkflowRequest", from_py_object)]
#[derive(Clone)]
pub struct PySubWorkflowRequest {
    pub(crate) inner: SubWorkflowRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySubWorkflowRequest {
    /// Build a new sub-workflow request.
    ///
    /// Args:
    ///     workflow_name: Symbolic name of the workflow to invoke.
    ///     step_ids: Ordered list of step IDs (empty = workflow default).
    ///     input: Initial input value (passed as kwargs dict, encoded as JSON).
    ///     timeout_secs: Optional wall-clock timeout in seconds.
    #[new]
    #[pyo3(signature = (workflow_name, step_ids=None, input=None, timeout_secs=None))]
    fn new(
        workflow_name: String,
        step_ids: Option<Vec<String>>,
        input: Option<&Bound<'_, PyDict>>,
        timeout_secs: Option<u64>,
    ) -> PyResult<Self> {
        let input_value = match input {
            Some(d) => dict_to_json(d)?,
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        let inner = SubWorkflowRequest::new(
            workflow_name,
            step_ids.unwrap_or_default(),
            &input_value,
            timeout_secs,
        )
        .map_err(|e| PeerException::new_err(format!("invalid input JSON: {e}")))?;
        Ok(Self { inner })
    }

    /// Envelope version of this payload.
    #[getter]
    fn envelope_version(&self) -> u32 {
        self.inner.envelope_version
    }

    /// Symbolic name of the workflow to invoke.
    #[getter]
    fn workflow_name(&self) -> &str {
        &self.inner.workflow_name
    }

    /// Step IDs to execute as part of this sub-workflow.
    #[getter]
    fn step_ids(&self) -> Vec<String> {
        self.inner.step_ids.clone()
    }

    /// Optional wall-clock timeout in seconds.
    #[getter]
    fn timeout_secs(&self) -> Option<u64> {
        self.inner.timeout_secs
    }

    /// Decode the JSON-encoded `input_json` payload back into a Python value.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn input_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .input_value()
            .map_err(|e| PeerException::new_err(format!("invalid input JSON: {e}")))?;
        json_to_py(py, &value)
    }

    fn __repr__(&self) -> String {
        format!(
            "SubWorkflowRequest(workflow_name={:?}, steps={}, timeout_secs={:?})",
            self.inner.workflow_name,
            self.inner.step_ids.len(),
            self.inner.timeout_secs,
        )
    }
}

/// Result of a remote sub-workflow invocation.
#[gen_stub_pyclass]
#[pyclass(name = "SubWorkflowResponse", from_py_object)]
#[derive(Clone)]
pub struct PySubWorkflowResponse {
    pub(crate) inner: SubWorkflowResponse,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySubWorkflowResponse {
    /// Envelope version of this payload.
    #[getter]
    fn envelope_version(&self) -> u32 {
        self.inner.envelope_version
    }

    /// Error message if the sub-workflow failed; otherwise `None`.
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    /// Decode the JSON-encoded `result_json` payload, if any.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn result_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .result_value()
            .map_err(|e| PeerException::new_err(format!("invalid result JSON: {e}")))?;
        match value {
            Some(v) => json_to_py(py, &v),
            None => Ok(py.None()),
        }
    }

    /// Decode the entire `state_json` map back into a Python dict.
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn state_values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let map = self
            .inner
            .state_values()
            .map_err(|e| PeerException::new_err(format!("invalid state JSON: {e}")))?;
        let dict = PyDict::new(py);
        for (k, v) in &map {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Map from session-ref UUID (as string) to peer remote-ref descriptor.
    #[gen_stub(override_return_type(
        type_repr = "dict[str, PeerRemoteRefDescriptor]",
        imports = ()
    ))]
    fn remote_refs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (uuid, desc) in &self.inner.remote_refs {
            let py_desc = Py::new(
                py,
                PyPeerRemoteRefDescriptor {
                    inner: desc.clone(),
                },
            )?;
            dict.set_item(uuid.to_string(), py_desc)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "SubWorkflowResponse(refs={}, error={:?})",
            self.inner.remote_refs.len(),
            self.inner.error,
        )
    }
}

/// Metadata describing a remote session ref handed out by a peer.
///
/// Named `PeerRemoteRefDescriptor` to avoid colliding with the
/// `RemoteRefDescriptor` exposed by `blazen_core` for in-process refs.
#[gen_stub_pyclass]
#[pyclass(name = "PeerRemoteRefDescriptor", from_py_object)]
#[derive(Clone)]
pub struct PyPeerRemoteRefDescriptor {
    pub(crate) inner: RemoteRefDescriptor,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPeerRemoteRefDescriptor {
    /// Construct a new descriptor.
    #[new]
    fn new(origin_node_id: String, type_tag: String, created_at_epoch_ms: u64) -> Self {
        Self {
            inner: RemoteRefDescriptor {
                origin_node_id,
                type_tag,
                created_at_epoch_ms,
            },
        }
    }

    /// Stable identifier of the node that owns the underlying value.
    #[getter]
    fn origin_node_id(&self) -> &str {
        &self.inner.origin_node_id
    }

    /// Type tag mirroring `SessionRefSerializable::blazen_type_tag`.
    #[getter]
    fn type_tag(&self) -> &str {
        &self.inner.type_tag
    }

    /// Wall-clock creation time on the origin node, in ms since the Unix epoch.
    #[getter]
    fn created_at_epoch_ms(&self) -> u64 {
        self.inner.created_at_epoch_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "PeerRemoteRefDescriptor(origin_node_id={:?}, type_tag={:?}, created_at_epoch_ms={})",
            self.inner.origin_node_id, self.inner.type_tag, self.inner.created_at_epoch_ms,
        )
    }
}

/// Request to dereference a remote session ref.
#[gen_stub_pyclass]
#[pyclass(name = "DerefRequest", from_py_object)]
#[derive(Clone)]
pub struct PyDerefRequest {
    pub(crate) inner: DerefRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDerefRequest {
    /// Construct a new deref request from a session-ref UUID string.
    #[new]
    fn new(ref_uuid: &str) -> PyResult<Self> {
        let uuid = Uuid::parse_str(ref_uuid)
            .map_err(|e| PeerException::new_err(format!("invalid UUID: {e}")))?;
        Ok(Self {
            inner: DerefRequest {
                envelope_version: ENVELOPE_VERSION,
                ref_uuid: uuid,
            },
        })
    }

    /// Envelope version of this payload.
    #[getter]
    fn envelope_version(&self) -> u32 {
        self.inner.envelope_version
    }

    /// UUID of the registry entry on the origin node, as a string.
    #[getter]
    fn ref_uuid(&self) -> String {
        self.inner.ref_uuid.to_string()
    }

    fn __repr__(&self) -> String {
        format!("DerefRequest(ref_uuid={})", self.inner.ref_uuid)
    }
}

/// Response containing the dereferenced bytes for a session ref.
#[gen_stub_pyclass]
#[pyclass(name = "DerefResponse", from_py_object)]
#[derive(Clone)]
pub struct PyDerefResponse {
    pub(crate) inner: DerefResponse,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDerefResponse {
    /// Construct a new deref response from raw bytes.
    #[new]
    fn new(payload: Vec<u8>) -> Self {
        Self {
            inner: DerefResponse {
                envelope_version: ENVELOPE_VERSION,
                payload,
            },
        }
    }

    /// Envelope version of this payload.
    #[getter]
    fn envelope_version(&self) -> u32 {
        self.inner.envelope_version
    }

    /// Raw payload bytes returned by the origin node.
    #[getter]
    fn payload(&self) -> Vec<u8> {
        self.inner.payload.clone()
    }

    fn __repr__(&self) -> String {
        format!("DerefResponse(payload_len={})", self.inner.payload.len())
    }
}

/// Request to release (drop) a remote session ref.
#[gen_stub_pyclass]
#[pyclass(name = "ReleaseRequest", from_py_object)]
#[derive(Clone)]
pub struct PyReleaseRequest {
    pub(crate) inner: ReleaseRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReleaseRequest {
    /// Construct a new release request from a session-ref UUID string.
    #[new]
    fn new(ref_uuid: &str) -> PyResult<Self> {
        let uuid = Uuid::parse_str(ref_uuid)
            .map_err(|e| PeerException::new_err(format!("invalid UUID: {e}")))?;
        Ok(Self {
            inner: ReleaseRequest {
                envelope_version: ENVELOPE_VERSION,
                ref_uuid: uuid,
            },
        })
    }

    /// Envelope version of this payload.
    #[getter]
    fn envelope_version(&self) -> u32 {
        self.inner.envelope_version
    }

    /// UUID of the registry entry to drop on the origin node, as a string.
    #[getter]
    fn ref_uuid(&self) -> String {
        self.inner.ref_uuid.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ReleaseRequest(ref_uuid={})", self.inner.ref_uuid)
    }
}

/// Acknowledgement for a [`PyReleaseRequest`].
#[gen_stub_pyclass]
#[pyclass(name = "ReleaseResponse", from_py_object)]
#[derive(Clone)]
pub struct PyReleaseResponse {
    pub(crate) inner: ReleaseResponse,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReleaseResponse {
    /// Construct a new release response.
    #[new]
    fn new(released: bool) -> Self {
        Self {
            inner: ReleaseResponse {
                envelope_version: ENVELOPE_VERSION,
                released,
            },
        }
    }

    /// Envelope version of this payload.
    #[getter]
    fn envelope_version(&self) -> u32 {
        self.inner.envelope_version
    }

    /// `True` if the registry entry was found and dropped, `False` otherwise.
    #[getter]
    fn released(&self) -> bool {
        self.inner.released
    }

    fn __repr__(&self) -> String {
        format!("ReleaseResponse(released={})", self.inner.released)
    }
}

/// Convert a `HashMap<Uuid, RemoteRefDescriptor>` into a Python dict
/// keyed by UUID strings.
#[allow(dead_code)]
pub(crate) fn remote_refs_to_py(
    py: Python<'_>,
    refs: &HashMap<Uuid, RemoteRefDescriptor>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (uuid, desc) in refs {
        let py_desc = Py::new(
            py,
            PyPeerRemoteRefDescriptor {
                inner: desc.clone(),
            },
        )?;
        dict.set_item(uuid.to_string(), py_desc)?;
    }
    Ok(dict.into_any().unbind())
}
