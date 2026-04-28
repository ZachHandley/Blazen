//! Python wrapper for the local mistral.rs LLM provider.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio_stream::StreamExt;

use crate::error::MistralRsError;
use crate::providers::completion_model::{PyCompletionOptions, build_request};
use crate::providers::options::PyMistralRsOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::MistralRsProvider;
use blazen_llm::traits::{CompletionModel, LocalModel};

// ---------------------------------------------------------------------------
// PyMistralRsProvider
// ---------------------------------------------------------------------------

/// A local mistral.rs LLM completion provider.
///
/// Runs LLM inference fully on-device using the mistral.rs engine. Supports
/// vision-capable models when configured via the underlying Rust struct.
/// No API key is required.
///
/// Example:
///     >>> opts = MistralRsOptions("mistralai/Mistral-7B-Instruct-v0.3")
///     >>> provider = MistralRsProvider(options=opts)
///     >>> response = await provider.complete([ChatMessage.user("Hello!")])
#[gen_stub_pyclass]
#[pyclass(name = "MistralRsProvider", from_py_object)]
#[derive(Clone)]
pub struct PyMistralRsProvider {
    inner: Arc<MistralRsProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMistralRsProvider {
    /// Create a new mistral.rs provider.
    ///
    /// Args:
    ///     options: Required :class:`MistralRsOptions` with the model id.
    #[new]
    #[pyo3(signature = (*, options))]
    fn new(options: PyRef<'_, PyMistralRsOptions>) -> PyResult<Self> {
        let opts = options.inner.clone();
        let provider = MistralRsProvider::from_options(opts)
            .map_err(|e| MistralRsError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Get the model identifier.
    #[getter]
    fn model_id(&self) -> String {
        CompletionModel::model_id(self.inner.as_ref()).to_owned()
    }

    /// Perform a chat completion.
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
                .map_err(|e| MistralRsError::new_err(e.to_string()))?;
            Ok(PyCompletionResponse { inner: response })
        })
    }

    /// Stream a chat completion, calling a callback for each chunk.
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
                .map_err(|e| MistralRsError::new_err(e.to_string()))?;

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
                    Err(e) => return Err(MistralRsError::new_err(e.to_string())),
                }
            }
            Ok(())
        })
    }

    /// Load the model weights into memory / VRAM. Idempotent.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::load(inner.as_ref())
                .await
                .map_err(|e| MistralRsError::new_err(e.to_string()))
        })
    }

    /// Drop the loaded model and free its memory / VRAM.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::unload(inner.as_ref())
                .await
                .map_err(|e| MistralRsError::new_err(e.to_string()))
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
            "MistralRsProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}
