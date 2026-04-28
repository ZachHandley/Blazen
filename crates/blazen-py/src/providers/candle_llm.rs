//! Python wrapper for the local candle LLM provider.
//!
//! Exposes [`CandleLlmProvider`](blazen_llm::CandleLlmProvider) (with the
//! [`CandleLlmCompletionModel`](blazen_llm::CandleLlmCompletionModel) trait
//! bridge) to Python with ``complete``, ``stream``, and load/unload control.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio_stream::StreamExt;

use crate::error::CandleLlmError;
use crate::providers::completion_model::{PyCompletionOptions, build_request};
use crate::providers::options::PyCandleLlmOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::{CandleLlmCompletionModel, CandleLlmProvider};

// ---------------------------------------------------------------------------
// PyCandleLlmProvider
// ---------------------------------------------------------------------------

/// A local candle LLM completion provider.
///
/// Runs LLM inference fully on-device using the candle (HuggingFace) engine.
/// No API key is required.
///
/// Example:
///     >>> opts = CandleLlmOptions(model_id="meta-llama/Llama-3.2-1B")
///     >>> provider = CandleLlmProvider(options=opts)
///     >>> response = await provider.complete([ChatMessage.user("Hello!")])
#[gen_stub_pyclass]
#[pyclass(name = "CandleLlmProvider", from_py_object)]
#[derive(Clone)]
pub struct PyCandleLlmProvider {
    /// Trait-object wrapper used for ``complete`` / ``stream`` (it adapts
    /// the inherent ``model_id() -> Option<&str>`` to the trait-required
    /// ``&str``).
    completion: Arc<CandleLlmCompletionModel>,
    /// Direct provider used for ``load`` / ``unload`` / ``is_loaded`` via
    /// the [`LocalModel`] trait impl.
    local: Arc<CandleLlmProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCandleLlmProvider {
    /// Create a new candle LLM provider.
    ///
    /// Args:
    ///     options: Optional :class:`CandleLlmOptions` for model id, device,
    ///         quantization, revision, context length, and cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyCandleLlmOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        // `CandleLlmProvider` is not `Clone` and exposes no options accessor,
        // so we keep the options by value and build two independent provider
        // instances: one for the `CompletionModel` trait bridge wrapper, one
        // for direct `LocalModel` calls. They each hold their own engine
        // mutex; load/unload on the second is honored by the load/unload
        // methods exposed here.
        let local = CandleLlmProvider::from_options(opts.clone())
            .map_err(|e| CandleLlmError::new_err(e.to_string()))?;
        let for_completion = CandleLlmProvider::from_options(opts)
            .map_err(|e| CandleLlmError::new_err(e.to_string()))?;
        let completion = Arc::new(CandleLlmCompletionModel::new(for_completion));
        Ok(Self {
            completion,
            local: Arc::new(local),
        })
    }

    /// Get the model identifier.
    #[getter]
    fn model_id(&self) -> String {
        CompletionModel::model_id(self.completion.as_ref()).to_owned()
    }

    /// Perform a chat completion.
    ///
    /// Args:
    ///     messages: A list of :class:`ChatMessage` objects.
    ///     options: Optional :class:`CompletionOptions` for sampling params.
    ///
    /// Returns:
    ///     A :class:`CompletionResponse`.
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

        let inner = self.completion.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = CompletionModel::complete(inner.as_ref(), request)
                .await
                .map_err(|e| CandleLlmError::new_err(e.to_string()))?;
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
        let inner = self.completion.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = CompletionModel::stream(inner.as_ref(), request)
                .await
                .map_err(|e| CandleLlmError::new_err(e.to_string()))?;

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
                    Err(e) => return Err(CandleLlmError::new_err(e.to_string())),
                }
            }
            Ok(())
        })
    }

    /// Load the model weights into memory / VRAM. Idempotent.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn load<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::load(local.as_ref())
                .await
                .map_err(|e| CandleLlmError::new_err(e.to_string()))
        })
    }

    /// Drop the loaded model and free its memory / VRAM.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn unload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            LocalModel::unload(local.as_ref())
                .await
                .map_err(|e| CandleLlmError::new_err(e.to_string()))
        })
    }

    /// Whether the model is currently loaded.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.bool]", imports = ("typing", "builtins")))]
    fn is_loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let local = self.local.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(LocalModel::is_loaded(local.as_ref()).await)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CandleLlmProvider(model_id='{}')",
            CompletionModel::model_id(self.completion.as_ref())
        )
    }
}
