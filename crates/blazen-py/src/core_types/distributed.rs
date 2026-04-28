//! Python wrappers for [`blazen_core::distributed`] (feature-gated).
//!
//! Only compiled when the `distributed` feature is enabled on this
//! crate, which in turn forwards to `blazen-core/distributed`.
//!
//! These mirror the *transport-agnostic* types that
//! `Workflow::run_remote` consumes -- not to be confused with the
//! gRPC-specific protocol types in [`crate::peer`], which are bound
//! separately under the `Peer*` Python class names.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_core::distributed::{PeerClient, RemoteWorkflowRequest, RemoteWorkflowResponse};
use blazen_core::session_ref::RemoteRefDescriptor;

use crate::convert::{dict_to_json, json_to_py};

use super::session_ref::PyRemoteRefDescriptor;

// ---------------------------------------------------------------------------
// PyRemoteWorkflowRequest
// ---------------------------------------------------------------------------

/// Python wrapper around [`RemoteWorkflowRequest`].
///
/// Held by callers that build sub-workflow invocations directly rather
/// than going through the higher-level `BlazenPeerClient` -- e.g. tests
/// that mock out a [`PyPeerClient`].
#[gen_stub_pyclass]
#[pyclass(name = "RemoteWorkflowRequest", from_py_object)]
#[derive(Clone)]
pub struct PyRemoteWorkflowRequest {
    pub(crate) inner: RemoteWorkflowRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRemoteWorkflowRequest {
    /// Build a request.
    ///
    /// Args:
    ///     workflow_name: Symbolic name of the workflow on the remote
    ///         peer.
    ///     step_ids: Ordered list of step IDs to execute.
    ///     input: Mapping passed as the workflow's initial input. Encoded
    ///         as JSON internally.
    ///     timeout_secs: Optional wall-clock timeout. `None` means "use
    ///         the server's default deadline".
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
        Ok(Self {
            inner: RemoteWorkflowRequest {
                workflow_name,
                step_ids: step_ids.unwrap_or_default(),
                input: input_value,
                timeout_secs,
            },
        })
    }

    #[getter]
    fn workflow_name(&self) -> &str {
        &self.inner.workflow_name
    }

    #[getter]
    fn step_ids(&self) -> Vec<String> {
        self.inner.step_ids.clone()
    }

    #[getter]
    fn timeout_secs(&self) -> Option<u64> {
        self.inner.timeout_secs
    }

    /// Decode the JSON-encoded input back to a Python value.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn input_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.input)
    }

    fn __repr__(&self) -> String {
        format!(
            "RemoteWorkflowRequest(workflow_name={:?}, steps={}, timeout_secs={:?})",
            self.inner.workflow_name,
            self.inner.step_ids.len(),
            self.inner.timeout_secs,
        )
    }
}

impl From<RemoteWorkflowRequest> for PyRemoteWorkflowRequest {
    fn from(inner: RemoteWorkflowRequest) -> Self {
        Self { inner }
    }
}

impl From<PyRemoteWorkflowRequest> for RemoteWorkflowRequest {
    fn from(p: PyRemoteWorkflowRequest) -> Self {
        p.inner
    }
}

// ---------------------------------------------------------------------------
// PyRemoteWorkflowResponse
// ---------------------------------------------------------------------------

/// Python wrapper around [`RemoteWorkflowResponse`].
#[gen_stub_pyclass]
#[pyclass(name = "RemoteWorkflowResponse", from_py_object)]
#[derive(Clone)]
pub struct PyRemoteWorkflowResponse {
    pub(crate) inner: RemoteWorkflowResponse,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRemoteWorkflowResponse {
    /// Build a response.
    ///
    /// Args:
    ///     result: Optional terminal result encoded as a Python value.
    ///     remote_refs: Mapping from session-ref UUID string to
    ///         in-process descriptor.
    ///     error: Error message if the sub-workflow failed.
    #[new]
    #[pyo3(signature = (result=None, remote_refs=None, error=None))]
    fn new(
        result: Option<&Bound<'_, PyAny>>,
        remote_refs: Option<&Bound<'_, PyDict>>,
        error: Option<String>,
    ) -> PyResult<Self> {
        let result_json = match result {
            Some(value) if !value.is_none() => Some(crate::convert::py_to_json(value.py(), value)?),
            _ => None,
        };

        let mut refs: HashMap<uuid::Uuid, RemoteRefDescriptor> = HashMap::new();
        if let Some(map) = remote_refs {
            for (k, v) in map.iter() {
                let key_str: String = k.extract()?;
                let uuid = uuid::Uuid::parse_str(&key_str).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid remote_refs key (not a UUID): {e}"
                    ))
                })?;
                let desc: PyRemoteRefDescriptor = v.extract()?;
                refs.insert(uuid, desc.inner);
            }
        }

        Ok(Self {
            inner: RemoteWorkflowResponse {
                result: result_json,
                remote_refs: refs,
                error,
            },
        })
    }

    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    /// Decode the optional result value back to a Python object.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn result_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.result {
            Some(v) => json_to_py(py, v),
            None => Ok(py.None()),
        }
    }

    /// Map of session-ref UUID strings to descriptors.
    #[gen_stub(override_return_type(
        type_repr = "dict[str, RemoteRefDescriptor]",
        imports = ()
    ))]
    fn remote_refs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (uuid, desc) in &self.inner.remote_refs {
            let py_desc = Py::new(
                py,
                PyRemoteRefDescriptor {
                    inner: desc.clone(),
                },
            )?;
            dict.set_item(uuid.to_string(), py_desc)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "RemoteWorkflowResponse(refs={}, error={:?})",
            self.inner.remote_refs.len(),
            self.inner.error,
        )
    }
}

impl From<RemoteWorkflowResponse> for PyRemoteWorkflowResponse {
    fn from(inner: RemoteWorkflowResponse) -> Self {
        Self { inner }
    }
}

impl From<PyRemoteWorkflowResponse> for RemoteWorkflowResponse {
    fn from(p: PyRemoteWorkflowResponse) -> Self {
        p.inner
    }
}

// ---------------------------------------------------------------------------
// PyPeerClient -- ABC subclassable from Python
// ---------------------------------------------------------------------------

/// Abstract base class mirroring the [`PeerClient`] trait.
///
/// Subclass this from Python and override `invoke_sub_workflow(request)`
/// to plug a custom transport into `Workflow.run_remote`. The subclass
/// method may return either a [`PyRemoteWorkflowResponse`] directly or a
/// coroutine that resolves to one.
///
/// Concrete bindings (such as the gRPC `BlazenPeerClient` exposed under
/// the peer module) implement [`PeerClient`] on the Rust side and bypass
/// this Python-facing shim.
#[gen_stub_pyclass]
#[pyclass(name = "PeerClient", subclass)]
pub struct PyPeerClient {
    /// Marker -- subclasses store their own state on the Python side.
    _private: (),
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPeerClient {
    #[new]
    fn new() -> Self {
        Self { _private: () }
    }

    /// Invoke a sub-workflow on a remote peer.
    ///
    /// The default implementation raises `NotImplementedError`. Override
    /// this in a subclass.
    #[pyo3(signature = (request))]
    fn invoke_sub_workflow(
        &self,
        request: &PyRemoteWorkflowRequest,
    ) -> PyResult<PyRemoteWorkflowResponse> {
        let _ = request;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "PeerClient.invoke_sub_workflow must be overridden by a subclass",
        ))
    }
}

/// Adapter that lets a Python-side [`PyPeerClient`] subclass satisfy the
/// Rust [`PeerClient`] trait. Used by the workflow runtime when a Python
/// caller passes a `PeerClient` subclass into `Workflow.run_remote`.
pub struct PyPeerClientAdapter {
    pub(crate) py_obj: Py<PyAny>,
}

impl PyPeerClientAdapter {
    /// Wrap a Python object that implements `invoke_sub_workflow`.
    #[must_use]
    pub fn new(py_obj: Py<PyAny>) -> Self {
        Self { py_obj }
    }
}

impl PeerClient for PyPeerClientAdapter {
    fn invoke_sub_workflow<'a>(
        &'a self,
        request: RemoteWorkflowRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<RemoteWorkflowResponse, blazen_core::WorkflowError>,
                > + Send
                + 'a,
        >,
    > {
        let py_req = PyRemoteWorkflowRequest { inner: request };
        Box::pin(async move {
            let response: Result<RemoteWorkflowResponse, blazen_core::WorkflowError> =
                Python::attach(|py| {
                    let bound = self.py_obj.bind(py);
                    let result = bound
                        .call_method1("invoke_sub_workflow", (py_req,))
                        .map_err(|e| {
                            blazen_core::WorkflowError::ValidationFailed(format!(
                                "PeerClient.invoke_sub_workflow raised: {e}"
                            ))
                        })?;
                    let py_resp: PyRemoteWorkflowResponse = result.extract().map_err(|e| {
                        blazen_core::WorkflowError::ValidationFailed(format!(
                            "PeerClient.invoke_sub_workflow returned a non-RemoteWorkflowResponse value: {e}"
                        ))
                    })?;
                    Ok(py_resp.inner)
                });
            response
        })
    }
}
