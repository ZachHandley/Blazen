//! Python bindings for the `blazen_llm::middleware` composable layer
//! system.
//!
//! Mirrors the Rust trait surface (`Middleware`, `MiddlewareStack`,
//! `RetryMiddleware`, `CacheMiddleware`) so Python callers can build
//! decorator chains the same way Rust callers can.
//!
//! Custom middleware can be implemented by subclassing `Middleware` and
//! overriding `apply(model, messages, options) -> response` (async). At
//! `apply(model)` time the stack constructs a Rust adapter that, on
//! every `complete()` call, dispatches into the Python subclass with
//! the inner model wrapped as a `CompletionModel` and the incoming
//! request reified as a list of `ChatMessage` and a `CompletionOptions`.

use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio_stream::Stream;

use blazen_llm::cache::{CacheConfig, CachedCompletionModel};
use blazen_llm::middleware::{
    CacheMiddleware as RsCacheMiddleware, Middleware as RsMiddleware,
    RetryMiddleware as RsRetryMiddleware,
};
use blazen_llm::retry::{RetryCompletionModel, RetryConfig};
use blazen_llm::{
    BlazenError, CompletionModel, CompletionRequest, CompletionResponse, ProviderConfig,
    StreamChunk,
};

use crate::providers::completion_model::{
    PyCompletionModel, arc_from_bound, build_py_options_from_request_helper,
};
use crate::providers::config::{PyCacheConfig, PyRetryConfig};
use crate::types::{PyChatMessage, PyCompletionResponse};

// ---------------------------------------------------------------------------
// Trait object enum: lets us hold heterogeneous middlewares in a Vec.
// ---------------------------------------------------------------------------

/// Internal handle to a middleware layer added to a `PyMiddlewareStack`.
///
/// Concrete variants reference the Rust implementations directly so that
/// re-applying the stack does not require crossing into Python; the
/// `Custom` variant boxes a Python-callable adapter.
enum LayerKind {
    Retry(RetryConfig),
    Cache(CacheConfig),
    Custom(Py<PyAny>),
}

impl LayerKind {
    /// Wrap `inner` with the layer's behaviour.
    ///
    /// Takes a `Python<'_>` token because the `Custom` variant must
    /// clone its `Py<PyAny>` reference, which requires the GIL.
    fn wrap(&self, py: Python<'_>, inner: Arc<dyn CompletionModel>) -> Arc<dyn CompletionModel> {
        match self {
            Self::Retry(cfg) => Arc::new(RetryCompletionModel::from_arc(inner, cfg.clone())),
            Self::Cache(cfg) => Arc::new(CachedCompletionModel::from_arc(inner, cfg.clone())),
            Self::Custom(py_obj) => Arc::new(PyMiddlewareAdapter::new(py_obj.clone_ref(py), inner)),
        }
    }
}

// ---------------------------------------------------------------------------
// PyMiddleware -- abstract base class for Python-implemented middleware.
// ---------------------------------------------------------------------------

/// Abstract base class for Python-implemented LLM middleware.
///
/// Subclass and override `apply(model, messages, options)` (async) to
/// add custom behaviour around a `CompletionModel`. The `model`
/// argument is the inner `CompletionModel` produced by the rest of the
/// middleware chain; call `await model.complete(messages, options)` to
/// forward the request, optionally inspecting or modifying the result
/// before returning it.
///
/// Example:
///     >>> class LoggingMiddleware(Middleware):
///     ...     async def apply(self, model, messages, options):
///     ...         print(f"calling {model.model_id}")
///     ...         response = await model.complete(messages, options)
///     ...         print(f"got {len(response.content or '')} chars")
///     ...         return response
///     >>>
///     >>> stack = MiddlewareStack().layer(LoggingMiddleware())
///     >>> wrapped = stack.apply(CompletionModel.openai())
#[gen_stub_pyclass]
#[pyclass(name = "Middleware", subclass)]
#[derive(Default)]
pub struct PyMiddleware;

#[gen_stub_pymethods]
#[pymethods]
impl PyMiddleware {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Wrap a `complete()` invocation with custom behaviour.
    ///
    /// Subclasses must override this method. The default implementation
    /// raises `NotImplementedError`.
    ///
    /// Args:
    ///     model: The inner `CompletionModel` to forward to.
    ///     messages: The list of `ChatMessage` objects from the request.
    ///     options: Optional `CompletionOptions` (sampling parameters,
    ///         tools, response format).
    ///
    /// Returns:
    ///     A `CompletionResponse`.
    #[pyo3(signature = (model, messages, options=None))]
    #[allow(unused_variables)]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyAny>,
        messages: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Middleware subclass must override apply(model, messages, options)",
        ))
    }
}

// ---------------------------------------------------------------------------
// Adapter: bridges a Python Middleware subclass into a CompletionModel.
// ---------------------------------------------------------------------------

/// Wraps a Python `Middleware` subclass instance and the inner
/// `CompletionModel` so the resulting object satisfies `CompletionModel`.
///
/// On every `complete()` call the adapter:
///   1. Wraps the inner `Arc<dyn CompletionModel>` as a `PyCompletionModel`.
///   2. Reifies the request into Python `ChatMessage` / `CompletionOptions`.
///   3. Calls `subclass.apply(model, messages, options)`.
///   4. Awaits the returned coroutine on the Python event loop.
///   5. Decodes the resulting `CompletionResponse`.
///
/// `stream()` and `provider_config()` always pass through to the inner
/// model -- Python middleware can only intercept `complete()`.
struct PyMiddlewareAdapter {
    py_obj: Py<PyAny>,
    inner: Arc<dyn CompletionModel>,
}

impl PyMiddlewareAdapter {
    fn new(py_obj: Py<PyAny>, inner: Arc<dyn CompletionModel>) -> Self {
        Self { py_obj, inner }
    }
}

#[async_trait]
impl CompletionModel for PyMiddlewareAdapter {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        // Phase 1: under the GIL, wrap inputs and invoke `apply`.
        let (fut, locals) = tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<(_, TaskLocals)> {
                let model_py = Py::new(
                    py,
                    PyCompletionModel {
                        inner: Some(self.inner.clone()),
                        local_model: None,
                        config: None,
                    },
                )?;

                let messages_py: Vec<Py<PyChatMessage>> = request
                    .messages
                    .iter()
                    .map(|m| Py::new(py, PyChatMessage { inner: m.clone() }))
                    .collect::<PyResult<_>>()?;

                let options_py = build_py_options_from_request_helper(py, &request)?;

                let host = self.py_obj.bind(py);
                let coro = match options_py {
                    Some(opts) => host.call_method1("apply", (model_py, messages_py, opts)),
                    None => host.call_method1("apply", (model_py, messages_py)),
                }
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Middleware.apply() raised before yielding a coroutine: {e}"
                    ))
                })?;

                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                let fut =
                    pyo3_async_runtimes::into_future_with_locals(&locals, coro).map_err(|e| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "Middleware.apply() must be an async def returning a coroutine: {e}"
                        ))
                    })?;
                Ok((fut, locals))
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider("middleware", format!("dispatch setup failed: {e}"))
        })?;

        // Phase 2: drive the coroutine on the asyncio loop.
        let py_result = pyo3_async_runtimes::tokio::scope(locals, fut)
            .await
            .map_err(|e: PyErr| {
                BlazenError::provider("middleware", format!("Middleware.apply() raised: {e}"))
            })?;

        // Phase 3: decode result back into CompletionResponse.
        tokio::task::block_in_place(|| {
            Python::attach(|py| -> PyResult<CompletionResponse> {
                let bound = py_result.bind(py);
                if let Ok(resp) = bound.extract::<PyRef<'_, PyCompletionResponse>>() {
                    return Ok(resp.inner.clone());
                }
                let response: CompletionResponse = pythonize::depythonize(bound).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Middleware.apply() must return CompletionResponse or a compatible dict: {e}"
                    ))
                })?;
                Ok(response)
            })
        })
        .map_err(|e: PyErr| {
            BlazenError::provider(
                "middleware",
                format!("failed to decode Middleware.apply() result: {e}"),
            )
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // Streaming bypasses Python middleware -- the inner model handles it.
        self.inner.stream(request).await
    }

    fn provider_config(&self) -> Option<&ProviderConfig> {
        self.inner.provider_config()
    }
}

// ---------------------------------------------------------------------------
// PyRetryMiddleware -- concrete Rust RetryMiddleware wrapper.
// ---------------------------------------------------------------------------

/// Middleware that wraps a model with `RetryCompletionModel`.
///
/// This is the standalone Python-facing equivalent of
/// `blazen_llm::middleware::RetryMiddleware`.
#[gen_stub_pyclass]
#[pyclass(name = "RetryMiddleware")]
pub struct PyRetryMiddleware {
    pub(crate) config: RetryConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryMiddleware {
    /// Build a retry middleware layer.
    ///
    /// Args:
    ///     config: Optional typed `RetryConfig`. Defaults to
    ///         `RetryConfig()` (3 retries, 1s initial, 30s max).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyRef<'_, PyRetryConfig>>) -> Self {
        Self {
            config: config.map(|c| c.inner.clone()).unwrap_or_default(),
        }
    }

    /// Apply this middleware directly to a `CompletionModel`, returning
    /// a new `CompletionModel` with retry behaviour.
    fn wrap(&self, model: Bound<'_, PyCompletionModel>) -> PyCompletionModel {
        let local_model = model.borrow().local_model.clone();
        let inner = arc_from_bound(&model);
        let mw = RsRetryMiddleware {
            config: self.config.clone(),
        };
        PyCompletionModel {
            inner: Some(mw.wrap(inner)),
            local_model,
            config: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryMiddleware(max_retries={}, initial_delay_ms={}, max_delay_ms={})",
            self.config.max_retries, self.config.initial_delay_ms, self.config.max_delay_ms
        )
    }
}

// ---------------------------------------------------------------------------
// PyCacheMiddleware -- concrete Rust CacheMiddleware wrapper.
// ---------------------------------------------------------------------------

/// Middleware that wraps a model with `CachedCompletionModel`.
///
/// This is the standalone Python-facing equivalent of
/// `blazen_llm::middleware::CacheMiddleware`.
#[gen_stub_pyclass]
#[pyclass(name = "CacheMiddleware")]
pub struct PyCacheMiddleware {
    pub(crate) config: CacheConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCacheMiddleware {
    /// Build a cache middleware layer.
    ///
    /// Args:
    ///     config: Optional typed `CacheConfig`. Defaults to
    ///         `CacheConfig()` (content-hash strategy, 300s TTL,
    ///         1000 entries).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyRef<'_, PyCacheConfig>>) -> Self {
        Self {
            config: config.map(|c| c.inner.clone()).unwrap_or_default(),
        }
    }

    /// Apply this middleware directly to a `CompletionModel`, returning
    /// a new `CompletionModel` with caching behaviour.
    fn wrap(&self, model: Bound<'_, PyCompletionModel>) -> PyCompletionModel {
        let local_model = model.borrow().local_model.clone();
        let inner = arc_from_bound(&model);
        let mw = RsCacheMiddleware {
            config: self.config.clone(),
        };
        PyCompletionModel {
            inner: Some(mw.wrap(inner)),
            local_model,
            config: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheMiddleware(ttl_seconds={}, max_entries={})",
            self.config.ttl_seconds, self.config.max_entries
        )
    }
}

// ---------------------------------------------------------------------------
// PyMiddlewareStack -- Python builder mirroring blazen_llm::MiddlewareStack.
// ---------------------------------------------------------------------------

/// A composable stack of `Middleware` layers.
///
/// Layers are added with `layer()` (or the convenience helpers
/// `with_retry()` / `with_cache()`); the first layer added becomes the
/// outermost wrapper at `apply_to()` time. Calling `apply_to(model)`
/// returns a new `CompletionModel` that runs every layer around the
/// supplied model.
///
/// Example:
///     >>> stack = (MiddlewareStack()
///     ...     .with_retry(RetryConfig(max_retries=5))
///     ...     .with_cache(CacheConfig(ttl_seconds=600)))
///     >>> wrapped = stack.apply(CompletionModel.openai())
#[gen_stub_pyclass]
#[pyclass(name = "MiddlewareStack")]
pub struct PyMiddlewareStack {
    layers: Arc<StdMutex<Vec<LayerKind>>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMiddlewareStack {
    /// Create an empty middleware stack.
    #[new]
    fn new() -> Self {
        Self {
            layers: Arc::new(StdMutex::new(Vec::new())),
        }
    }

    /// Add a middleware layer.
    ///
    /// Accepts any of:
    ///   * a `RetryMiddleware` instance,
    ///   * a `CacheMiddleware` instance,
    ///   * a Python `Middleware` subclass instance.
    ///
    /// The first layer added becomes the **outermost** wrapper; it
    /// executes first on the request path and last on the response
    /// path.
    ///
    /// Returns ``self`` so calls can be chained.
    fn layer<'py>(
        slf: Bound<'py, Self>,
        middleware: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let kind = if let Ok(retry) = middleware.extract::<PyRef<'_, PyRetryMiddleware>>() {
            LayerKind::Retry(retry.config.clone())
        } else if let Ok(cache) = middleware.extract::<PyRef<'_, PyCacheMiddleware>>() {
            LayerKind::Cache(cache.config.clone())
        } else {
            // Treat anything else as a Python Middleware subclass and
            // verify it has an `apply` attribute.
            if !middleware.hasattr("apply")? {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "layer() expects a RetryMiddleware, CacheMiddleware, or Middleware subclass with an `apply` method",
                ));
            }
            LayerKind::Custom(middleware.unbind())
        };

        slf.borrow_mut()
            .layers
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .push(kind);
        Ok(slf)
    }

    /// Convenience: add a `RetryMiddleware` layer.
    #[pyo3(signature = (config=None))]
    fn with_retry<'py>(
        slf: Bound<'py, Self>,
        config: Option<PyRef<'_, PyRetryConfig>>,
    ) -> PyResult<Bound<'py, Self>> {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        slf.borrow_mut()
            .layers
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .push(LayerKind::Retry(cfg));
        Ok(slf)
    }

    /// Convenience: add a `CacheMiddleware` layer.
    #[pyo3(signature = (config=None))]
    fn with_cache<'py>(
        slf: Bound<'py, Self>,
        config: Option<PyRef<'_, PyCacheConfig>>,
    ) -> PyResult<Bound<'py, Self>> {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        slf.borrow_mut()
            .layers
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .push(LayerKind::Cache(cfg));
        Ok(slf)
    }

    /// Apply every registered layer to `model` and return the fully
    /// wrapped `CompletionModel`.
    ///
    /// The first layer added becomes the outermost wrapper. Layers are
    /// applied in reverse insertion order so that the first layer added
    /// runs first on the request path.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyCompletionModel>,
    ) -> PyResult<PyCompletionModel> {
        let local_model = model.borrow().local_model.clone();
        let mut wrapped: Arc<dyn CompletionModel> = arc_from_bound(&model);
        let layers = self
            .layers
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        // Mirror `MiddlewareStack::apply`: iterate in reverse so the
        // first-added layer is the outermost wrapper.
        for layer in layers.iter().rev() {
            wrapped = layer.wrap(py, wrapped);
        }
        Ok(PyCompletionModel {
            inner: Some(wrapped),
            local_model,
            config: None,
        })
    }

    /// The number of layers currently registered.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self
            .layers
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .len())
    }

    fn __repr__(&self) -> String {
        let n = self.layers.lock().map(|l| l.len()).unwrap_or(0);
        format!("MiddlewareStack(layers={n})")
    }
}
