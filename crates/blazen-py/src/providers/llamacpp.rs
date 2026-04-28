//! Python wrapper for the local llama.cpp LLM provider.
//!
//! Exposes [`LlamaCppProvider`](blazen_llm::LlamaCppProvider) to Python with
//! ``complete``, ``stream``, and load/unload control. For the unified
//! provider-agnostic factory, see
//! :meth:`CompletionModel.llamacpp <crate::providers::completion_model::PyCompletionModel>`
//! (when wired up); this class is the typed standalone wrapper.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio_stream::StreamExt;

use crate::error::LlamaCppError;
use crate::providers::completion_model::{PyCompletionOptions, build_request};
use crate::providers::options::PyLlamaCppOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::LlamaCppProvider;
use blazen_llm::traits::{CompletionModel, LocalModel};

// ---------------------------------------------------------------------------
// PyLlamaCppProvider
// ---------------------------------------------------------------------------

/// A local llama.cpp completion provider.
///
/// Runs LLM inference fully on-device using the llama.cpp engine. No API
/// key is required. The first inference call lazily loads the model;
/// callers can pre-warm by awaiting :meth:`load`.
///
/// Example:
///     >>> opts = LlamaCppOptions(model_path="/models/llama-3.2-1b-q4_k_m.gguf")
///     >>> provider = LlamaCppProvider(options=opts)
///     >>> response = await provider.complete([ChatMessage.user("Hello!")])
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppProvider", from_py_object)]
#[derive(Clone)]
pub struct PyLlamaCppProvider {
    inner: Arc<LlamaCppProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppProvider {
    /// Create a new llama.cpp provider.
    ///
    /// Args:
    ///     options: Optional :class:`LlamaCppOptions` for model path,
    ///         device, quantization, context length, GPU layer count, and
    ///         cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyLlamaCppOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider = crate::convert::block_on_context(LlamaCppProvider::from_options(opts))
            .map_err(|e| LlamaCppError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Get the model identifier (typically the GGUF path or HF model id).
    #[getter]
    fn model_id(&self) -> String {
        CompletionModel::model_id(self.inner.as_ref()).to_owned()
    }

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of :class:`ChatMessage` objects.
    ///     options: Optional :class:`CompletionOptions` for sampling params.
    ///
    /// Returns:
    ///     A :class:`CompletionResponse` with content, model, usage,
    ///     finish_reason, and timing.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    #[pyo3(signature = (messages, options=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyChatMessage>,
        options: Option<Py<PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.into_iter().map(|m| m.inner).collect();
        let opts_borrow = options.as_ref().map(|o| o.borrow(py));
        let request = build_request(py, rust_messages, opts_borrow.as_deref())?;

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = CompletionModel::complete(inner.as_ref(), request)
                .await
                .map_err(|e| LlamaCppError::new_err(e.to_string()))?;
            Ok(PyCompletionResponse { inner: response })
        })
    }

    /// Stream a chat completion, calling a callback for each chunk.
    ///
    /// Args:
    ///     messages: A list of :class:`ChatMessage` objects.
    ///     on_chunk: Callback function receiving each chunk as a dict.
    ///     options: Optional :class:`CompletionOptions` for sampling params.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    #[pyo3(signature = (messages, on_chunk, options=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Py<PyAny>,
        options: Option<PyRef<'py, PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = build_request(py, rust_messages, options.as_deref())?;
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = CompletionModel::stream(inner.as_ref(), request)
                .await
                .map_err(|e| LlamaCppError::new_err(e.to_string()))?;

            let mut stream = std::pin::pin!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        let chunk_json = serde_json::json!({
                            "delta": chunk.delta,
                            "finish_reason": chunk.finish_reason,
                            "tool_calls": chunk.tool_calls.iter().map(|tc| {
                                serde_json::json!({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
                            }).collect::<Vec<_>>(),
                        });

                        tokio::task::block_in_place(|| {
                            Python::attach(|py| {
                                let py_val = crate::convert::json_to_py(py, &chunk_json)?;
                                on_chunk.call1(py, (py_val,))?;
                                Ok::<_, PyErr>(())
                            })
                        })?;
                    }
                    Err(e) => {
                        return Err(LlamaCppError::new_err(e.to_string()));
                    }
                }
            }
            Ok(())
        })
    }

    /// Load the model weights into memory / VRAM.
    ///
    /// Idempotent; calling on an already-loaded provider is a no-op.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::load(inner.as_ref())
                .await
                .map_err(|e| LlamaCppError::new_err(e.to_string()))
        })
    }

    /// Drop the loaded model and free its memory / VRAM.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::unload(inner.as_ref())
                .await
                .map_err(|e| LlamaCppError::new_err(e.to_string()))
        })
    }

    /// Whether the model is currently loaded.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.bool]", imports = ("typing", "builtins")))]
    fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(LocalModel::is_loaded(inner.as_ref()).await)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LlamaCppProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}
