//! Python wrappers for batch completion execution.
//!
//! Exposes [`complete_batch`](blazen_llm::batch::complete_batch) to Python
//! with [`PyBatchResult`] for inspecting per-request outcomes and
//! [`PyBatchConfig`] for configuring concurrency.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use blazen_llm::batch::{
    BatchConfig, BatchResult as RustBatchResult, complete_batch as rust_complete_batch,
};
use blazen_llm::types::ChatMessage;

use crate::providers::PyCompletionModel;
use crate::providers::completion_model::{PyCompletionOptions, build_request};
use crate::types::{PyChatMessage, PyCompletionResponse, PyTokenUsage};

// ---------------------------------------------------------------------------
// PyBatchConfig
// ---------------------------------------------------------------------------

/// Configuration for a batch completion run.
///
/// Args:
///     concurrency: Maximum number of concurrent requests. ``0`` (the default)
///         means unlimited.
///
/// Example:
///     >>> config = BatchConfig(concurrency=4)
///     >>> result = await complete_batch(model, requests, config=config)
#[gen_stub_pyclass]
#[pyclass(name = "BatchConfig", from_py_object)]
#[derive(Clone)]
pub struct PyBatchConfig {
    pub(crate) concurrency: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBatchConfig {
    #[new]
    #[pyo3(signature = (*, concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    /// Build a config with unlimited concurrency.
    #[staticmethod]
    fn unlimited() -> Self {
        Self { concurrency: 0 }
    }

    /// Maximum number of concurrent requests. ``0`` means unlimited.
    #[getter]
    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn __repr__(&self) -> String {
        format!("BatchConfig(concurrency={})", self.concurrency)
    }
}

impl PyBatchConfig {
    pub(crate) fn to_rust(&self) -> BatchConfig {
        BatchConfig::new(self.concurrency)
    }
}

// ---------------------------------------------------------------------------
// PyBatchResult
// ---------------------------------------------------------------------------

/// Result of a batch completion run.
///
/// Each input request maps to a positional entry in ``responses`` and
/// ``errors``. A successful request has a ``CompletionResponse`` in
/// ``responses`` and ``None`` in ``errors``; a failed request has ``None``
/// in ``responses`` and an error string in ``errors``.
///
/// Example:
///     >>> result = await complete_batch(model, [msgs1, msgs2], concurrency=4)
///     >>> for i, resp in enumerate(result.responses):
///     ...     if resp is not None:
///     ...         print(resp.content)
///     ...     else:
///     ...         print(f"Request {i} failed: {result.errors[i]}")
#[gen_stub_pyclass]
#[pyclass(name = "BatchResult")]
pub struct PyBatchResult {
    inner: RustBatchResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBatchResult {
    /// Per-request responses. ``None`` for requests that failed.
    #[getter]
    fn responses(&self) -> Vec<Option<PyCompletionResponse>> {
        self.inner
            .responses
            .iter()
            .map(|r| {
                r.as_ref().ok().map(|resp| PyCompletionResponse {
                    inner: resp.clone(),
                })
            })
            .collect()
    }

    /// Per-request error messages. ``None`` for requests that succeeded.
    #[getter]
    fn errors(&self) -> Vec<Option<String>> {
        self.inner
            .responses
            .iter()
            .map(|r| r.as_ref().err().map(ToString::to_string))
            .collect()
    }

    /// Aggregated token usage across all successful responses.
    #[getter]
    fn total_usage(&self) -> Option<PyTokenUsage> {
        self.inner.total_usage.as_ref().map(PyTokenUsage::from)
    }

    /// Aggregated cost across all successful responses.
    #[getter]
    fn total_cost(&self) -> Option<f64> {
        self.inner.total_cost
    }

    /// Number of requests that succeeded.
    #[getter]
    fn success_count(&self) -> usize {
        self.inner.responses.iter().filter(|r| r.is_ok()).count()
    }

    /// Number of requests that failed.
    #[getter]
    fn failure_count(&self) -> usize {
        self.inner.responses.iter().filter(|r| r.is_err()).count()
    }

    fn __repr__(&self) -> String {
        let ok = self.inner.responses.iter().filter(|r| r.is_ok()).count();
        let err = self.inner.responses.iter().filter(|r| r.is_err()).count();
        format!(
            "BatchResult(total={}, succeeded={ok}, failed={err}, cost={:?})",
            self.inner.responses.len(),
            self.inner.total_cost,
        )
    }

    fn __len__(&self) -> usize {
        self.inner.responses.len()
    }
}

// ---------------------------------------------------------------------------
// complete_batch function
// ---------------------------------------------------------------------------

/// Execute multiple completion requests in parallel with bounded concurrency.
///
/// Each element of ``requests`` is a list of ``ChatMessage`` objects
/// representing a single conversation. All conversations are dispatched to
/// the model concurrently (up to the ``concurrency`` limit) and results
/// are returned in the same order.
///
/// Args:
///     model: The completion model to use.
///     requests: A list of message lists. Each inner list is one conversation.
///     concurrency: Maximum number of concurrent requests. ``0`` (the default)
///         means unlimited.
///     options: Optional ``CompletionOptions`` applied to every request
///         (temperature, max_tokens, tools, etc.).
///
/// Returns:
///     A ``BatchResult`` with per-request responses and aggregated usage/cost.
///
/// Example:
///     >>> model = CompletionModel.openai()
///     >>> conversations = [
///     ...     [ChatMessage.user("What is 2+2?")],
///     ...     [ChatMessage.user("What is 3+3?")],
///     ... ]
///     >>> result = await complete_batch(model, conversations, concurrency=4)
///     >>> for resp in result.responses:
///     ...     if resp is not None:
///     ...         print(resp.content)
#[gen_stub_pyfunction]
#[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, BatchResult]", imports = ("typing",)))]
#[pyfunction]
#[pyo3(signature = (model, requests, *, config=None, concurrency=0, options=None))]
#[allow(clippy::map_unwrap_or)]
pub fn complete_batch<'py>(
    py: Python<'py>,
    model: Bound<'py, PyCompletionModel>,
    requests: Vec<Vec<PyRef<'py, PyChatMessage>>>,
    config: Option<PyRef<'py, PyBatchConfig>>,
    concurrency: usize,
    options: Option<PyRef<'py, PyCompletionOptions>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Build all CompletionRequests while we still hold the GIL.
    let rust_requests: Vec<blazen_llm::CompletionRequest> = requests
        .iter()
        .map(|msgs| {
            let rust_msgs: Vec<ChatMessage> = msgs.iter().map(|m| m.inner.clone()).collect();
            build_request(py, rust_msgs, options.as_deref())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let rust_config = config
        .as_deref()
        .map(PyBatchConfig::to_rust)
        .unwrap_or_else(|| BatchConfig::new(concurrency));
    let inner_model = crate::providers::completion_model::arc_from_bound(&model);

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = rust_complete_batch(inner_model.as_ref(), rust_requests, rust_config).await;
        Ok(PyBatchResult { inner: result })
    })
}
