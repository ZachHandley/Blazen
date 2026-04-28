//! Standalone Python wrappers for the LLM decorator types.
//!
//! These classes mirror the `CompletionModel.with_retry` /
//! `with_fallback` / `with_cache` decorator methods but are exposed as
//! first-class Python types so that users who prefer explicit
//! construction can build them directly.
//!
//! Each wrapper holds an `Arc<dyn CompletionModel>` over the underlying
//! Rust decorator and implements `complete()` / `stream()` /
//! `model_id` so it behaves like a fully-fledged `CompletionModel`.

use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::{Stream, StreamExt};

use blazen_llm::cache::CachedCompletionModel;
use blazen_llm::fallback::FallbackModel;
use blazen_llm::retry::RetryCompletionModel;
use blazen_llm::{BlazenError, ChatMessage, CompletionModel, StreamChunk};

use crate::error::BlazenPyError;
use crate::providers::completion_model::{
    LazyStreamState, PendingStream, PyCompletionModel, PyCompletionOptions, PyLazyCompletionStream,
    arc_from_bound, build_request,
};
use crate::types::{PyChatMessage, PyCompletionResponse};

/// Type alias for the pinned boxed stream returned by `CompletionModel::stream`.
type PinnedChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>;

// ---------------------------------------------------------------------------
// PyRetryCompletionModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` decorator that retries transient failures with
/// exponential backoff.
///
/// This is the standalone equivalent of
/// `CompletionModel.with_retry(config)` and exposes the same
/// `complete()` / `stream()` surface.
///
/// Example:
///     >>> base = CompletionModel.openai()
///     >>> model = RetryCompletionModel(base, RetryConfig(max_retries=5))
///     >>> response = await model.complete([ChatMessage.user("Hi")])
#[gen_stub_pyclass]
#[pyclass(name = "RetryCompletionModel")]
pub struct PyRetryCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRetryCompletionModel {
    /// Wrap a `CompletionModel` with automatic retry on transient failures.
    ///
    /// Args:
    ///     model: The `CompletionModel` to wrap.
    ///     config: Optional typed `RetryConfig`. Defaults to
    ///         `RetryConfig()` (3 retries, 1s initial, 30s max).
    #[new]
    #[pyo3(signature = (model, config=None))]
    fn new(
        model: Bound<'_, PyCompletionModel>,
        config: Option<PyRef<'_, crate::providers::config::PyRetryConfig>>,
    ) -> Self {
        let retry_config = config.map(|c| c.inner.clone()).unwrap_or_default();
        let inner = arc_from_bound(&model);
        let wrapped = RetryCompletionModel::from_arc(inner, retry_config);
        Self {
            inner: Arc::new(wrapped),
        }
    }

    /// Convert this decorator into a `CompletionModel` so it can be passed
    /// to APIs that expect a `CompletionModel` (`run_agent`,
    /// `complete_batch`, further decorators, etc.).
    fn as_model(&self) -> PyCompletionModel {
        PyCompletionModel {
            inner: Some(self.inner.clone()),
            local_model: None,
            config: None,
        }
    }

    /// The model id reported by the underlying provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion, retrying transient failures.
    #[pyo3(signature = (messages, options=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        complete_via_arc(py, self.inner.clone(), messages, options)
    }

    /// Stream a chat completion, retrying the initial connection on
    /// transient failures.
    #[pyo3(signature = (messages, on_chunk=None, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Option<Py<PyAny>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        stream_via_arc(py, self.inner.clone(), messages, on_chunk, options)
    }

    fn __repr__(&self) -> String {
        format!("RetryCompletionModel(model_id='{}')", self.inner.model_id())
    }
}

// ---------------------------------------------------------------------------
// PyCachedCompletionModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` decorator that caches non-streaming responses.
///
/// Standalone equivalent of `CompletionModel.with_cache(config)`.
/// Streaming requests are never cached and always pass through.
///
/// Example:
///     >>> base = CompletionModel.openai()
///     >>> model = CachedCompletionModel(base, CacheConfig(ttl_seconds=600))
///     >>> response = await model.complete([ChatMessage.user("Hi")])
#[gen_stub_pyclass]
#[pyclass(name = "CachedCompletionModel")]
pub struct PyCachedCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCachedCompletionModel {
    /// Wrap a `CompletionModel` with response caching.
    ///
    /// Args:
    ///     model: The `CompletionModel` to wrap.
    ///     config: Optional typed `CacheConfig`. Defaults to
    ///         `CacheConfig()` (content-hash strategy, 300s TTL,
    ///         1000 entries).
    #[new]
    #[pyo3(signature = (model, config=None))]
    fn new(
        model: Bound<'_, PyCompletionModel>,
        config: Option<PyRef<'_, crate::providers::config::PyCacheConfig>>,
    ) -> Self {
        let cache_config = config.map(|c| c.inner.clone()).unwrap_or_default();
        let inner = arc_from_bound(&model);
        let wrapped = CachedCompletionModel::from_arc(inner, cache_config);
        Self {
            inner: Arc::new(wrapped),
        }
    }

    /// Convert this decorator into a `CompletionModel`.
    fn as_model(&self) -> PyCompletionModel {
        PyCompletionModel {
            inner: Some(self.inner.clone()),
            local_model: None,
            config: None,
        }
    }

    /// The model id reported by the underlying provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion, served from cache when possible.
    #[pyo3(signature = (messages, options=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        complete_via_arc(py, self.inner.clone(), messages, options)
    }

    /// Stream a chat completion (always passes through; never cached).
    #[pyo3(signature = (messages, on_chunk=None, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Option<Py<PyAny>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        stream_via_arc(py, self.inner.clone(), messages, on_chunk, options)
    }

    fn __repr__(&self) -> String {
        format!(
            "CachedCompletionModel(model_id='{}')",
            self.inner.model_id()
        )
    }
}

// ---------------------------------------------------------------------------
// PyFallbackModel
// ---------------------------------------------------------------------------

/// A `CompletionModel` that tries a primary provider and falls back to a
/// secondary on retryable failures.
///
/// Standalone equivalent of
/// `CompletionModel.with_fallback([primary, fallback])` for the common
/// two-provider case. For chains of three or more providers use
/// `CompletionModel.with_fallback(...)`.
///
/// Example:
///     >>> primary = CompletionModel.openai()
///     >>> backup = CompletionModel.anthropic()
///     >>> model = FallbackModel(primary, backup)
///     >>> response = await model.complete([ChatMessage.user("Hi")])
#[gen_stub_pyclass]
#[pyclass(name = "FallbackModel")]
pub struct PyFallbackModel {
    inner: Arc<dyn CompletionModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFallbackModel {
    /// Build a fallback chain from a primary and a fallback provider.
    ///
    /// Args:
    ///     primary: The first provider to try.
    ///     fallback: The provider to invoke when `primary` fails with
    ///         a retryable error.
    #[new]
    fn new(primary: Bound<'_, PyCompletionModel>, fallback: Bound<'_, PyCompletionModel>) -> Self {
        let providers: Vec<Arc<dyn CompletionModel>> =
            vec![arc_from_bound(&primary), arc_from_bound(&fallback)];
        let model = FallbackModel::new(providers);
        Self {
            inner: Arc::new(model),
        }
    }

    /// Convert this decorator into a `CompletionModel`.
    fn as_model(&self) -> PyCompletionModel {
        PyCompletionModel {
            inner: Some(self.inner.clone()),
            local_model: None,
            config: None,
        }
    }

    /// The model id reported by the primary provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion against the primary provider, falling
    /// back to the secondary on retryable failures.
    #[pyo3(signature = (messages, options=None))]
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        complete_via_arc(py, self.inner.clone(), messages, options)
    }

    /// Stream a chat completion against the primary provider, falling
    /// back to the secondary on retryable failures during the initial
    /// connection.
    #[pyo3(signature = (messages, on_chunk=None, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Option<Py<PyAny>>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        stream_via_arc(py, self.inner.clone(), messages, on_chunk, options)
    }

    fn __repr__(&self) -> String {
        format!("FallbackModel(model_id='{}')", self.inner.model_id())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers for invoking complete()/stream() through an Arc.
// ---------------------------------------------------------------------------

/// Drive `complete()` on an `Arc<dyn CompletionModel>`.
fn complete_via_arc<'py>(
    py: Python<'py>,
    inner: Arc<dyn CompletionModel>,
    messages: Vec<PyRef<'py, PyChatMessage>>,
    options: Option<PyRef<'py, PyCompletionOptions>>,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    let request = build_request(py, rust_messages, options.as_deref())?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let response = inner.complete(request).await.map_err(BlazenPyError::from)?;
        Ok(PyCompletionResponse { inner: response })
    })
}

/// Drive `stream()` on an `Arc<dyn CompletionModel>` with the same
/// callback / async-iterator semantics as `PyCompletionModel::stream`.
fn stream_via_arc<'py>(
    py: Python<'py>,
    inner: Arc<dyn CompletionModel>,
    messages: Vec<PyRef<'py, PyChatMessage>>,
    on_chunk: Option<Py<PyAny>>,
    options: Option<PyRef<'py, PyCompletionOptions>>,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
    let request = build_request(py, rust_messages, options.as_deref())?;

    if let Some(callback) = on_chunk {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream: PinnedChunkStream = inner
                .stream(request)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let mut stream = std::pin::pin!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        tokio::task::block_in_place(|| {
                            Python::attach(|py| {
                                let py_chunk = pythonize::pythonize(py, &chunk).map_err(|e| {
                                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                                })?;
                                callback.call1(py, (py_chunk,))?;
                                Ok::<_, PyErr>(())
                            })
                        })?;
                    }
                    Err(e) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()));
                    }
                }
            }
            Ok(())
        })
    } else {
        let lazy = PyLazyCompletionStream {
            state: Arc::new(Mutex::new(LazyStreamState::NotStarted(Box::new(
                PendingStream {
                    model: inner,
                    request: Some(request),
                },
            )))),
        };
        Ok(lazy.into_pyobject(py)?.into_any())
    }
}
