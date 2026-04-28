//! Python wrappers for compute job types.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::{self as compute_types, ComputeProvider};

use crate::compute::result_types::PyComputeResult;
use crate::error::blazen_error_to_pyerr;

// ---------------------------------------------------------------------------
// PyJobStatus
// ---------------------------------------------------------------------------

/// Status of a compute job.
///
/// Mirrors [`blazen_llm::compute::JobStatus`]. Carries both a discriminator
/// (``kind``) and the optional ``error`` message preserved from the core
/// ``JobStatus::Failed`` variant. Construct via the classmethod factories
/// (``queued()``, ``running()``, ``completed()``, ``failed(error)``,
/// ``cancelled()``) or via the legacy string constants.
///
/// Example:
///     >>> JobStatus.queued()
///     >>> JobStatus.failed("rate limited")
///     >>> JobStatus.QUEUED    # legacy string constant -- "queued"
#[gen_stub_pyclass]
#[pyclass(name = "JobStatus", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyJobStatus {
    pub(crate) inner: compute_types::JobStatus,
}

#[gen_stub_pymethods]
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

    /// Build the ``Queued`` variant.
    #[staticmethod]
    fn queued() -> Self {
        Self {
            inner: compute_types::JobStatus::Queued,
        }
    }

    /// Build the ``Running`` variant.
    #[staticmethod]
    fn running() -> Self {
        Self {
            inner: compute_types::JobStatus::Running,
        }
    }

    /// Build the ``Completed`` variant.
    #[staticmethod]
    fn completed() -> Self {
        Self {
            inner: compute_types::JobStatus::Completed,
        }
    }

    /// Build the ``Failed`` variant with an error message.
    #[staticmethod]
    fn failed(error: String) -> Self {
        Self {
            inner: compute_types::JobStatus::Failed { error },
        }
    }

    /// Build the ``Cancelled`` variant.
    #[staticmethod]
    fn cancelled() -> Self {
        Self {
            inner: compute_types::JobStatus::Cancelled,
        }
    }

    /// Discriminator: ``"queued"``, ``"running"``, ``"completed"``,
    /// ``"failed"``, or ``"cancelled"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            compute_types::JobStatus::Queued => "queued",
            compute_types::JobStatus::Running => "running",
            compute_types::JobStatus::Completed => "completed",
            compute_types::JobStatus::Failed { .. } => "failed",
            compute_types::JobStatus::Cancelled => "cancelled",
        }
    }

    /// The error message for ``Failed`` variants. ``None`` otherwise.
    #[getter]
    fn error(&self) -> Option<&str> {
        if let compute_types::JobStatus::Failed { error } = &self.inner {
            Some(error)
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            compute_types::JobStatus::Failed { error } => {
                format!("JobStatus.failed({error:?})")
            }
            other => format!(
                "JobStatus.{}",
                match other {
                    compute_types::JobStatus::Queued => "queued",
                    compute_types::JobStatus::Running => "running",
                    compute_types::JobStatus::Completed => "completed",
                    compute_types::JobStatus::Cancelled => "cancelled",
                    compute_types::JobStatus::Failed { .. } => unreachable!(),
                }
            ),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl From<compute_types::JobStatus> for PyJobStatus {
    fn from(inner: compute_types::JobStatus) -> Self {
        Self { inner }
    }
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

// ---------------------------------------------------------------------------
// PyCompute
// ---------------------------------------------------------------------------

/// Job-control front-end over any compute provider.
///
/// Wraps a [`ComputeProvider`] (for example a [`FalProvider`]) and exposes
/// the four-step async job lifecycle: ``submit`` -> ``status`` ->
/// ``await_completion`` (or ``cancel``).
///
/// Example:
///     >>> from blazen import Compute, FalProvider, FalOptions, ComputeRequest
///     >>> fal = FalProvider(options=FalOptions(api_key="fal-..."))
///     >>> compute = Compute.from_fal(fal)
///     >>> handle = await compute.submit(ComputeRequest(model="fal-ai/flux/dev",
///     ...                                              input={"prompt": "a cat"}))
///     >>> while (await compute.status(handle)) in ("queued", "running"):
///     ...     await asyncio.sleep(1)
///     >>> result = await compute.await_completion(handle)
#[gen_stub_pyclass]
#[pyclass(name = "Compute")]
pub struct PyCompute {
    pub(crate) inner: Arc<dyn ComputeProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompute {
    /// Build a ``Compute`` from a ``FalProvider``.
    #[staticmethod]
    fn from_fal(provider: PyRef<'_, crate::providers::fal::PyFalProvider>) -> Self {
        Self {
            inner: provider.compute_arc(),
        }
    }

    /// Submit a compute job and return a [`JobHandle`].
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, JobHandle]", imports = ("typing",)))]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        request: PyComputeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handle = ComputeProvider::submit(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyJobHandle { inner: handle })
        })
    }

    /// Poll the current status of a submitted job.
    ///
    /// Returns a status string: ``"queued"``, ``"running"``, ``"completed"``,
    /// ``"failed"``, or ``"cancelled"``.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.str]", imports = ("typing", "builtins")))]
    fn status<'py>(&self, py: Python<'py>, job: PyJobHandle) -> PyResult<Bound<'py, PyAny>> {
        let handle = job.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let status = ComputeProvider::status(inner.as_ref(), &handle)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let status_str = match status {
                compute_types::JobStatus::Queued => "queued",
                compute_types::JobStatus::Running => "running",
                compute_types::JobStatus::Completed => "completed",
                compute_types::JobStatus::Failed { .. } => "failed",
                compute_types::JobStatus::Cancelled => "cancelled",
            };
            Ok(status_str.to_owned())
        })
    }

    /// Cancel a running or queued job.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn cancel<'py>(&self, py: Python<'py>, job: PyJobHandle) -> PyResult<Bound<'py, PyAny>> {
        let handle = job.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            ComputeProvider::cancel(inner.as_ref(), &handle)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(())
        })
    }

    /// Wait for a job to complete and return the [`ComputeResult`].
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ComputeResult]", imports = ("typing",)))]
    fn await_completion<'py>(
        &self,
        py: Python<'py>,
        job: PyJobHandle,
    ) -> PyResult<Bound<'py, PyAny>> {
        let handle = job.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ComputeProvider::result(inner.as_ref(), handle)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyComputeResult { inner: result })
        })
    }
}
