//! Python wrappers for compute job types.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::{self as compute_types};

// ---------------------------------------------------------------------------
// PyJobStatus
// ---------------------------------------------------------------------------

/// Job status constants.
///
/// Example:
///     >>> JobStatus.QUEUED    # "queued"
///     >>> JobStatus.RUNNING   # "running"
///     >>> JobStatus.COMPLETED # "completed"
///
/// NOTE: This is currently a flat set of string constants and does NOT
/// preserve the `error` payload carried by the core `JobStatus::Failed`
/// variant. A follow-up task will replace this with a proper pyclass enum
/// that retains the failure message. Known limitation.
#[pyclass(name = "JobStatus", frozen)]
pub struct PyJobStatus;

#[pymethods]
impl PyJobStatus {
    #[classattr]
    const QUEUED: &'static str = "queued";
    #[classattr]
    const RUNNING: &'static str = "running";
    #[classattr]
    const COMPLETED: &'static str = "completed";
    #[classattr]
    const FAILED: &'static str = "failed";
    #[classattr]
    const CANCELLED: &'static str = "cancelled";
}

// ---------------------------------------------------------------------------
// PyJobHandle
// ---------------------------------------------------------------------------

/// A handle to a submitted compute job.
///
/// Example:
///     >>> handle = await fal.submit(model="fal-ai/flux/dev", input={...})
///     >>> print(handle.id, handle.model)
#[gen_stub_pyclass]
#[pyclass(name = "JobHandle", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyJobHandle {
    pub(crate) inner: compute_types::JobHandle,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyJobHandle {
    /// The provider-assigned job identifier.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// The provider name (e.g. "fal").
    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    /// The model/endpoint that was invoked.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// When the job was submitted (ISO 8601 string).
    #[getter]
    fn submitted_at(&self) -> String {
        self.inner.submitted_at.to_rfc3339()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "id" => Ok(self.inner.id.clone().into_pyobject(py)?.into_any().unbind()),
            "provider" => Ok(self
                .inner
                .provider
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "model" => Ok(self
                .inner
                .model
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "submitted_at" => Ok(self
                .inner
                .submitted_at
                .to_rfc3339()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "JobHandle(id='{}', provider='{}', model='{}')",
            self.inner.id, self.inner.provider, self.inner.model,
        )
    }
}

// ---------------------------------------------------------------------------
// PyComputeRequest
// ---------------------------------------------------------------------------

/// Input for a raw compute job.
///
/// Example:
///     >>> req = ComputeRequest(model="fal-ai/flux/dev", input={"prompt": "a cat"})
#[gen_stub_pyclass]
#[pyclass(name = "ComputeRequest", from_py_object)]
#[derive(Clone)]
pub struct PyComputeRequest {
    pub(crate) inner: compute_types::ComputeRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyComputeRequest {
    #[new]
    #[pyo3(signature = (*, model, input, webhook=None))]
    fn new(
        py: Python<'_>,
        model: &str,
        input: &Bound<'_, PyAny>,
        webhook: Option<String>,
    ) -> PyResult<Self> {
        let input_json = crate::convert::py_to_json(py, input)?;
        Ok(Self {
            inner: compute_types::ComputeRequest {
                model: model.to_owned(),
                input: input_json,
                webhook,
            },
        })
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.input)
    }

    #[getter]
    fn webhook(&self) -> Option<String> {
        self.inner.webhook.clone()
    }

    fn __repr__(&self) -> String {
        format!("ComputeRequest(model='{}')", self.inner.model)
    }
}
