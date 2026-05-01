//! Retry plumbing exposed to Python.
//!
//! Wraps [`RetryStack`](blazen_llm::retry::RetryStack), [`RetryEmbeddingModel`],
//! and [`RetryHttpClient`] so callers can layer retries onto every Blazen
//! surface, not just `CompletionModel`.
//!
//! `PyRetryConfig` itself (the typed config struct) lives next door in
//! `providers::config` because that is where `RetryCompletionModel` /
//! `with_retry` and the cache config already sit.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::retry::{RetryConfig, RetryEmbeddingModel, RetryHttpClient, RetryStack};

use crate::providers::config::PyRetryConfig;

// ---------------------------------------------------------------------------
// PyRetryStack
// ---------------------------------------------------------------------------

/// A snapshot of every scope's retry configuration.
///
/// Mirrors [`blazen_llm::retry::RetryStack`]. Build one to capture the
/// per-scope retry chain (provider / pipeline / workflow / step) and then
/// call [`resolve`](Self::resolve) with an optional per-call override to
/// produce the effective `RetryConfig` for a single LLM call.
///
/// Most users will not construct this directly — `Pipeline` / `Workflow`
/// / `Step` build it implicitly via the typed builder helpers — but the
/// type is exposed so custom orchestrators can drive the resolver
/// themselves.
#[gen_stub_pyclass]
#[pyclass(name = "RetryStack", from_py_object)]
#[derive(Clone, Default)]
pub struct PyRetryStack {
    pub(crate) inner: RetryStack,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryStack {
    /// Build an empty stack. Set fields via the per-scope getters/setters
    /// below or pass them as kwargs.
    #[new]
    #[pyo3(signature = (*, provider=None, pipeline=None, workflow=None, step=None))]
    fn new(
        provider: Option<PyRef<'_, PyRetryConfig>>,
        pipeline: Option<PyRef<'_, PyRetryConfig>>,
        workflow: Option<PyRef<'_, PyRetryConfig>>,
        step: Option<PyRef<'_, PyRetryConfig>>,
    ) -> Self {
        Self {
            inner: RetryStack {
                provider: provider.map(|c| Arc::new(c.inner.clone())),
                pipeline: pipeline.map(|c| Arc::new(c.inner.clone())),
                workflow: workflow.map(|c| Arc::new(c.inner.clone())),
                step: step.map(|c| Arc::new(c.inner.clone())),
            },
        }
    }

    /// Provider-level default (lowest priority).
    #[getter]
    fn provider(&self) -> Option<PyRetryConfig> {
        self.inner.provider.as_ref().map(|c| PyRetryConfig {
            inner: (**c).clone(),
        })
    }
    #[setter]
    fn set_provider(&mut self, v: Option<PyRef<'_, PyRetryConfig>>) {
        self.inner.provider = v.map(|c| Arc::new(c.inner.clone()));
    }

    /// Pipeline-level default.
    #[getter]
    fn pipeline(&self) -> Option<PyRetryConfig> {
        self.inner.pipeline.as_ref().map(|c| PyRetryConfig {
            inner: (**c).clone(),
        })
    }
    #[setter]
    fn set_pipeline(&mut self, v: Option<PyRef<'_, PyRetryConfig>>) {
        self.inner.pipeline = v.map(|c| Arc::new(c.inner.clone()));
    }

    /// Workflow-level override.
    #[getter]
    fn workflow(&self) -> Option<PyRetryConfig> {
        self.inner.workflow.as_ref().map(|c| PyRetryConfig {
            inner: (**c).clone(),
        })
    }
    #[setter]
    fn set_workflow(&mut self, v: Option<PyRef<'_, PyRetryConfig>>) {
        self.inner.workflow = v.map(|c| Arc::new(c.inner.clone()));
    }

    /// Step-level override (highest priority before per-call).
    #[getter]
    fn step(&self) -> Option<PyRetryConfig> {
        self.inner.step.as_ref().map(|c| PyRetryConfig {
            inner: (**c).clone(),
        })
    }
    #[setter]
    fn set_step(&mut self, v: Option<PyRef<'_, PyRetryConfig>>) {
        self.inner.step = v.map(|c| Arc::new(c.inner.clone()));
    }

    /// Resolve the stack against an optional per-call override and return
    /// the effective `RetryConfig`. Precedence: call > step > workflow >
    /// pipeline > provider; falls back to `RetryConfig()` defaults.
    #[pyo3(signature = (call=None))]
    fn resolve(&self, call: Option<PyRef<'_, PyRetryConfig>>) -> PyRetryConfig {
        let call_arc = call.map(|c| Arc::new(c.inner.clone()));
        let resolved = self.inner.resolve(call_arc.as_ref());
        PyRetryConfig {
            inner: (*resolved).clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryStack(provider={}, pipeline={}, workflow={}, step={})",
            self.inner.provider.is_some(),
            self.inner.pipeline.is_some(),
            self.inner.workflow.is_some(),
            self.inner.step.is_some(),
        )
    }
}

impl From<RetryStack> for PyRetryStack {
    fn from(inner: RetryStack) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyRetryEmbeddingModel
// ---------------------------------------------------------------------------

/// An [`EmbeddingModel`] decorator that retries transient failures with
/// exponential backoff.
///
/// Mirrors [`blazen_llm::retry::RetryEmbeddingModel`]. Wrap any embedding
/// model — built-in or `EmbeddingModel`-subclass from Python — with this
/// type to get rate-limit / timeout / 5xx retries with no extra plumbing.
///
/// Example:
///     >>> base = EmbeddingModel.openai(options=ProviderOptions(api_key="sk-..."))
///     >>> model = RetryEmbeddingModel(base, RetryConfig(max_retries=5))
///     >>> resp = await model.embed(["hello", "world"])
#[gen_stub_pyclass]
#[pyclass(name = "RetryEmbeddingModel")]
pub struct PyRetryEmbeddingModel {
    inner: Arc<dyn blazen_llm::traits::EmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryEmbeddingModel {
    /// Wrap an `EmbeddingModel` with automatic retry on transient failures.
    #[new]
    #[pyo3(signature = (model, config=None))]
    fn new(
        model: Bound<'_, crate::types::PyEmbeddingModel>,
        config: Option<PyRef<'_, PyRetryConfig>>,
    ) -> PyResult<Self> {
        let retry_config: RetryConfig = config.map(|c| c.inner.clone()).unwrap_or_default();
        let inner_model = model
            .borrow()
            .inner
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "RetryEmbeddingModel: source EmbeddingModel has no inner provider",
                )
            })?
            .clone();
        let wrapped = RetryEmbeddingModel::from_arc(inner_model, retry_config);
        Ok(Self {
            inner: Arc::new(wrapped),
        })
    }

    /// The model id reported by the underlying provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Output dimensionality.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Embed a list of texts with retries on transient failures.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = inner
                .embed(&texts)
                .await
                .map_err(crate::error::BlazenPyError::from)?;
            Ok(crate::types::PyEmbeddingResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryEmbeddingModel(model_id='{}', dimensions={})",
            self.inner.model_id(),
            self.inner.dimensions(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyRetryHttpClient (introspection-only handle)
// ---------------------------------------------------------------------------

/// Marker handle for a [`RetryHttpClient`].
///
/// The Rust [`RetryHttpClient`] decorator wraps an `Arc<dyn HttpClient>` to
/// add retry/backoff to raw HTTP. The dynamic HTTP-client trait does not
/// cross the Python FFI boundary as a callable transport (Python users plug
/// in HTTP via the [`HttpClient`](crate::types::PyHttpClient) abc class
/// instead), so this Python wrapper is intentionally a thin introspection
/// handle: it captures the [`RetryConfig`] you'd hand the Rust constructor
/// and is accepted by the standalone provider classes that take an
/// `Arc<dyn HttpClient>` internally.
#[gen_stub_pyclass]
#[pyclass(name = "RetryHttpClient", from_py_object)]
#[derive(Clone)]
pub struct PyRetryHttpClient {
    pub(crate) config: PyRetryConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryHttpClient {
    /// Capture a retry-config to layer onto an HTTP transport.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyRef<'_, PyRetryConfig>>) -> Self {
        Self {
            config: config.map(|c| c.clone()).unwrap_or_default(),
        }
    }

    /// The captured retry configuration.
    #[getter]
    fn config(&self) -> PyRetryConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryHttpClient(max_retries={}, initial_delay_ms={})",
            self.config.inner.max_retries, self.config.inner.initial_delay_ms,
        )
    }
}

impl PyRetryHttpClient {
    /// Build a Rust [`RetryHttpClient`] decorator from this handle, given
    /// the inner client to wrap.
    #[allow(dead_code)]
    pub(crate) fn build(&self, inner: Arc<dyn blazen_llm::http::HttpClient>) -> RetryHttpClient {
        RetryHttpClient::new(inner, self.config.inner.clone())
    }
}
