//! Synchronous (blocking) Python wrappers for the async-only Rust surface.
//!
//! Python users frequently want a `BlockingFoo` ergonomic where every method
//! is synchronous — no `await`, no asyncio loop required. We provide that by
//! reusing the same Rust types as the async wrappers and `block_on`-ing each
//! call on a process-global Tokio runtime.
//!
//! The runtime is a multi-threaded `tokio::runtime::Runtime` started lazily
//! on first use via `OnceCell`. Inside a `block_on` we cannot call back into
//! Python's asyncio loop, so blocking siblings are NOT compatible with code
//! paths that rely on Python `async def` callbacks (custom tools, custom
//! HTTP clients, custom providers). For those cases callers must continue
//! to use the regular async API.

use std::sync::Arc;
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::{ChatMessage, CompletionModel};

use crate::error::BlazenPyError;
use crate::providers::completion_model::{
    PyCompletionModel, PyCompletionOptions, arc_from_bound, build_request,
};
use crate::types::{PyChatMessage, PyCompletionResponse, PyEmbeddingModel, PyEmbeddingResponse};

/// Shared multi-thread Tokio runtime used to drive every blocking method.
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("blazen-blocking")
            .build()
            .expect("failed to start blazen-blocking tokio runtime")
    })
}

// ---------------------------------------------------------------------------
// PyBlockingCompletionModel
// ---------------------------------------------------------------------------

/// Blocking sibling of [`CompletionModel`].
///
/// Wrap any `CompletionModel` with `BlockingCompletionModel(model)` to drive
/// `complete()` synchronously from non-async Python contexts (CLIs, REPLs,
/// scripts, sync libraries). Streaming is not exposed here — use the async
/// API or pass an `on_chunk` callback to the regular model when you need
/// partial chunks.
#[gen_stub_pyclass]
#[pyclass(name = "BlockingCompletionModel")]
pub struct PyBlockingCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlockingCompletionModel {
    /// Wrap a `CompletionModel` for synchronous dispatch.
    #[new]
    fn new(model: Bound<'_, PyCompletionModel>) -> Self {
        Self {
            inner: arc_from_bound(&model),
        }
    }

    /// The model id reported by the wrapped provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Synchronously perform a chat completion. Blocks the calling thread.
    #[pyo3(signature = (messages, options=None))]
    fn complete(
        &self,
        py: Python<'_>,
        messages: Vec<PyRef<'_, PyChatMessage>>,
        options: Option<PyRef<'_, PyCompletionOptions>>,
    ) -> PyResult<PyCompletionResponse> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();
        let response = py
            .detach(|| shared_runtime().block_on(async move { inner.complete(request).await }))
            .map_err(BlazenPyError::from)?;
        Ok(PyCompletionResponse { inner: response })
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockingCompletionModel(model_id='{}')",
            self.inner.model_id()
        )
    }
}

// ---------------------------------------------------------------------------
// PyBlockingEmbeddingModel
// ---------------------------------------------------------------------------

/// Blocking sibling of [`EmbeddingModel`].
#[gen_stub_pyclass]
#[pyclass(name = "BlockingEmbeddingModel")]
pub struct PyBlockingEmbeddingModel {
    inner: Arc<dyn blazen_llm::EmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlockingEmbeddingModel {
    /// Wrap an `EmbeddingModel` for synchronous dispatch.
    #[new]
    fn new(model: Bound<'_, PyEmbeddingModel>) -> PyResult<Self> {
        let inner = model
            .borrow()
            .inner
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "BlockingEmbeddingModel: source EmbeddingModel has no inner provider",
                )
            })?
            .clone();
        Ok(Self { inner })
    }

    /// The model id reported by the wrapped embedding provider.
    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Output dimensionality.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Synchronously embed a list of texts.
    fn embed(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<PyEmbeddingResponse> {
        let inner = self.inner.clone();
        let response = py
            .detach(|| shared_runtime().block_on(async move { inner.embed(&texts).await }))
            .map_err(BlazenPyError::from)?;
        Ok(PyEmbeddingResponse { inner: response })
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockingEmbeddingModel(model_id='{}', dimensions={})",
            self.inner.model_id(),
            self.inner.dimensions(),
        )
    }
}
