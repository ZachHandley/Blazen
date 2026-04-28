//! Python wrapper for the local llama.cpp LLM provider.
//!
//! Exposes [`LlamaCppProvider`](blazen_llm::LlamaCppProvider) to Python with
//! ``complete``, ``stream``, and load/unload control. For the unified
//! provider-agnostic factory, see
//! :meth:`CompletionModel.llamacpp <crate::providers::completion_model::PyCompletionModel>`
//! (when wired up); this class is the typed standalone wrapper.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use crate::error::LlamaCppError;
use crate::providers::completion_model::{PyCompletionOptions, build_request};
use crate::providers::options::PyLlamaCppOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::LlamaCppProvider;
use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::{
    LlamaCppChatMessageInput, LlamaCppChatRole, LlamaCppInferenceChunk,
    LlamaCppInferenceChunkStream, LlamaCppInferenceResult, LlamaCppInferenceUsage,
};

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

// ---------------------------------------------------------------------------
// PyLlamaCppChatRole
// ---------------------------------------------------------------------------

/// Chat message role for llama.cpp inference inputs.
#[gen_stub_pyclass_enum]
#[pyclass(name = "LlamaCppChatRole", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyLlamaCppChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl From<PyLlamaCppChatRole> for LlamaCppChatRole {
    fn from(r: PyLlamaCppChatRole) -> Self {
        match r {
            PyLlamaCppChatRole::System => Self::System,
            PyLlamaCppChatRole::User => Self::User,
            PyLlamaCppChatRole::Assistant => Self::Assistant,
            PyLlamaCppChatRole::Tool => Self::Tool,
        }
    }
}

impl From<LlamaCppChatRole> for PyLlamaCppChatRole {
    fn from(r: LlamaCppChatRole) -> Self {
        match r {
            LlamaCppChatRole::System => Self::System,
            LlamaCppChatRole::User => Self::User,
            LlamaCppChatRole::Assistant => Self::Assistant,
            LlamaCppChatRole::Tool => Self::Tool,
        }
    }
}

// ---------------------------------------------------------------------------
// PyLlamaCppChatMessageInput
// ---------------------------------------------------------------------------

/// A single chat message for llama.cpp inference.
///
/// llama.cpp messages are text-only; multimodal inputs are not supported by
/// this backend.
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppChatMessageInput", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyLlamaCppChatMessageInput {
    pub(crate) inner: LlamaCppChatMessageInput,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppChatMessageInput {
    /// Construct a chat message input.
    #[new]
    #[pyo3(signature = (*, role, text))]
    fn new(role: PyLlamaCppChatRole, text: String) -> Self {
        Self {
            inner: LlamaCppChatMessageInput {
                role: role.into(),
                text,
            },
        }
    }

    /// Build a chat message via the typed factory on the underlying type.
    #[staticmethod]
    #[pyo3(name = "create")]
    fn create(role: PyLlamaCppChatRole, text: String) -> Self {
        Self {
            inner: LlamaCppChatMessageInput::new(role.into(), text),
        }
    }

    /// The role that produced this message.
    #[getter]
    fn role(&self) -> PyLlamaCppChatRole {
        PyLlamaCppChatRole::from(self.inner.role)
    }

    /// The textual content of the message.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    fn __repr__(&self) -> String {
        format!(
            "LlamaCppChatMessageInput(role={:?}, text_len={})",
            self.inner.role,
            self.inner.text.len()
        )
    }
}

impl From<LlamaCppChatMessageInput> for PyLlamaCppChatMessageInput {
    fn from(inner: LlamaCppChatMessageInput) -> Self {
        Self { inner }
    }
}

impl From<&LlamaCppChatMessageInput> for PyLlamaCppChatMessageInput {
    fn from(inner: &LlamaCppChatMessageInput) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyLlamaCppInferenceUsage
// ---------------------------------------------------------------------------

/// Token usage statistics from a llama.cpp inference call.
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppInferenceUsage", frozen, from_py_object)]
#[derive(Clone, Default)]
pub struct PyLlamaCppInferenceUsage {
    pub(crate) inner: LlamaCppInferenceUsage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppInferenceUsage {
    /// Construct usage stats.
    #[new]
    #[pyo3(signature = (*, prompt_tokens=0, completion_tokens=0, total_tokens=0, total_time_sec=0.0))]
    fn new(
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        total_time_sec: f32,
    ) -> Self {
        Self {
            inner: LlamaCppInferenceUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_time_sec,
            },
        }
    }

    /// Tokens in the prompt.
    #[getter]
    fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Tokens generated.
    #[getter]
    fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    /// Total tokens (prompt + completion).
    #[getter]
    fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    /// Total wall-clock inference time in seconds.
    #[getter]
    fn total_time_sec(&self) -> f32 {
        self.inner.total_time_sec
    }

    fn __repr__(&self) -> String {
        format!(
            "LlamaCppInferenceUsage(prompt={}, completion={}, total={}, time_sec={})",
            self.inner.prompt_tokens,
            self.inner.completion_tokens,
            self.inner.total_tokens,
            self.inner.total_time_sec
        )
    }
}

impl From<LlamaCppInferenceUsage> for PyLlamaCppInferenceUsage {
    fn from(inner: LlamaCppInferenceUsage) -> Self {
        Self { inner }
    }
}

impl From<&LlamaCppInferenceUsage> for PyLlamaCppInferenceUsage {
    fn from(inner: &LlamaCppInferenceUsage) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyLlamaCppInferenceResult
// ---------------------------------------------------------------------------

/// Result of a single non-streaming llama.cpp inference call.
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppInferenceResult", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyLlamaCppInferenceResult {
    pub(crate) inner: Arc<LlamaCppInferenceResult>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppInferenceResult {
    /// Construct an inference result.
    #[new]
    #[pyo3(signature = (*, finish_reason, model, usage, content=None))]
    fn new(
        finish_reason: String,
        model: String,
        usage: PyLlamaCppInferenceUsage,
        content: Option<String>,
    ) -> Self {
        Self {
            inner: Arc::new(LlamaCppInferenceResult {
                content,
                finish_reason,
                model,
                usage: usage.inner,
            }),
        }
    }

    /// The generated text content, if any.
    #[getter]
    fn content(&self) -> Option<&str> {
        self.inner.content.as_deref()
    }

    /// Why the model stopped generating.
    #[getter]
    fn finish_reason(&self) -> &str {
        &self.inner.finish_reason
    }

    /// The model identifier that produced this result.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Token usage statistics.
    #[getter]
    fn usage(&self) -> PyLlamaCppInferenceUsage {
        PyLlamaCppInferenceUsage::from(self.inner.usage.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "LlamaCppInferenceResult(model={:?}, finish_reason={:?})",
            self.inner.model, self.inner.finish_reason
        )
    }
}

impl From<LlamaCppInferenceResult> for PyLlamaCppInferenceResult {
    fn from(inner: LlamaCppInferenceResult) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

// ---------------------------------------------------------------------------
// PyLlamaCppInferenceChunk
// ---------------------------------------------------------------------------

/// A single chunk from streaming llama.cpp inference.
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppInferenceChunk", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyLlamaCppInferenceChunk {
    pub(crate) inner: LlamaCppInferenceChunk,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppInferenceChunk {
    /// Construct an inference chunk.
    #[new]
    #[pyo3(signature = (*, delta=None, finish_reason=None))]
    fn new(delta: Option<String>, finish_reason: Option<String>) -> Self {
        Self {
            inner: LlamaCppInferenceChunk {
                delta,
                finish_reason,
            },
        }
    }

    /// Incremental text content.
    #[getter]
    fn delta(&self) -> Option<&str> {
        self.inner.delta.as_deref()
    }

    /// Set on the final chunk when generation stops.
    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "LlamaCppInferenceChunk(delta={:?}, finish_reason={:?})",
            self.inner.delta, self.inner.finish_reason
        )
    }
}

impl From<LlamaCppInferenceChunk> for PyLlamaCppInferenceChunk {
    fn from(inner: LlamaCppInferenceChunk) -> Self {
        Self { inner }
    }
}

impl From<&LlamaCppInferenceChunk> for PyLlamaCppInferenceChunk {
    fn from(inner: &LlamaCppInferenceChunk) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyLlamaCppInferenceChunkStream -- async iterator over streamed chunks
// ---------------------------------------------------------------------------

/// Async iterator over streamed llama.cpp inference chunks.
///
/// Implements the Python ``__aiter__`` / ``__anext__`` protocol so the stream
/// can be consumed with ``async for chunk in stream: ...``. Yields
/// ``LlamaCppInferenceChunk`` items and terminates with ``StopAsyncIteration``
/// once the underlying engine stream is exhausted.
#[gen_stub_pyclass]
#[pyclass(name = "LlamaCppInferenceChunkStream")]
pub struct PyLlamaCppInferenceChunkStream {
    pub(crate) inner: Arc<Mutex<Option<LlamaCppInferenceChunkStream>>>,
}

impl PyLlamaCppInferenceChunkStream {
    /// Wrap a raw llama.cpp chunk stream.
    pub fn new(stream: LlamaCppInferenceChunkStream) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(stream))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLlamaCppInferenceChunkStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, LlamaCppInferenceChunk]", imports = ("typing",)))]
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let Some(stream) = guard.as_mut() else {
                return Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                    "stream exhausted",
                ));
            };
            match stream.next().await {
                Some(Ok(chunk)) => Ok(PyLlamaCppInferenceChunk::from(chunk)),
                Some(Err(e)) => {
                    *guard = None;
                    Err(LlamaCppError::new_err(e.to_string()))
                }
                None => {
                    *guard = None;
                    Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                        "stream exhausted",
                    ))
                }
            }
        })
    }
}
