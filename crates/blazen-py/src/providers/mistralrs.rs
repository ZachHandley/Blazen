//! Python wrapper for the local mistral.rs LLM provider.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use crate::error::MistralRsError;
use crate::providers::completion_model::{PyCompletionOptions, build_request};
use crate::providers::options::PyMistralRsOptions;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::MistralRsProvider;
use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::{
    ChatMessageInput, ChatRole, InferenceChunk, InferenceChunkStream, InferenceImage,
    InferenceImageSource, InferenceResult, InferenceToolCall, InferenceUsage,
};

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

// ---------------------------------------------------------------------------
// PyChatRole
// ---------------------------------------------------------------------------

/// Chat message role for mistral.rs inference inputs.
#[gen_stub_pyclass_enum]
#[pyclass(name = "ChatRole", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl From<PyChatRole> for ChatRole {
    fn from(r: PyChatRole) -> Self {
        match r {
            PyChatRole::System => Self::System,
            PyChatRole::User => Self::User,
            PyChatRole::Assistant => Self::Assistant,
            PyChatRole::Tool => Self::Tool,
        }
    }
}

impl From<ChatRole> for PyChatRole {
    fn from(r: ChatRole) -> Self {
        match r {
            ChatRole::System => Self::System,
            ChatRole::User => Self::User,
            ChatRole::Assistant => Self::Assistant,
            ChatRole::Tool => Self::Tool,
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceImageSource
// ---------------------------------------------------------------------------

/// Source of an image payload for a multimodal mistral.rs chat message.
///
/// Use the static factories ``bytes(...)`` or ``path(...)``. Inspect via
/// ``kind`` (``"bytes"`` or ``"path"``) plus the per-variant getters.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceImageSource", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyInferenceImageSource {
    pub(crate) inner: InferenceImageSource,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceImageSource {
    /// Build a bytes-backed image source from raw encoded bytes.
    #[staticmethod]
    fn bytes(data: Vec<u8>) -> Self {
        Self {
            inner: InferenceImageSource::Bytes(data),
        }
    }

    /// Build a path-backed image source from a local file path.
    #[staticmethod]
    fn path(path: PathBuf) -> Self {
        Self {
            inner: InferenceImageSource::Path(path),
        }
    }

    /// Variant tag: ``"bytes"`` or ``"path"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            InferenceImageSource::Bytes(_) => "bytes",
            InferenceImageSource::Path(_) => "path",
        }
    }

    /// Raw image bytes for the ``Bytes`` variant; ``None`` otherwise.
    #[getter]
    fn data(&self) -> Option<Vec<u8>> {
        match &self.inner {
            InferenceImageSource::Bytes(b) => Some(b.clone()),
            InferenceImageSource::Path(_) => None,
        }
    }

    /// Local file path for the ``Path`` variant; ``None`` otherwise.
    #[getter]
    fn path_value(&self) -> Option<PathBuf> {
        match &self.inner {
            InferenceImageSource::Path(p) => Some(p.clone()),
            InferenceImageSource::Bytes(_) => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("InferenceImageSource(kind={:?})", self.kind())
    }
}

impl From<InferenceImageSource> for PyInferenceImageSource {
    fn from(inner: InferenceImageSource) -> Self {
        Self { inner }
    }
}

impl From<&InferenceImageSource> for PyInferenceImageSource {
    fn from(inner: &InferenceImageSource) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceImage
// ---------------------------------------------------------------------------

/// An image payload attached to a ``ChatMessageInput`` for multimodal
/// mistral.rs inference.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceImage", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyInferenceImage {
    pub(crate) inner: InferenceImage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceImage {
    /// Construct an image from an explicit source.
    #[new]
    #[pyo3(signature = (*, source))]
    fn new(source: PyInferenceImageSource) -> Self {
        Self {
            inner: InferenceImage {
                source: source.inner,
            },
        }
    }

    /// Build an image from raw encoded bytes (PNG, JPEG, WebP, ...).
    #[staticmethod]
    fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            inner: InferenceImage::from_bytes(bytes),
        }
    }

    /// Build an image from a local file path.
    #[staticmethod]
    fn from_path(path: PathBuf) -> Self {
        Self {
            inner: InferenceImage::from_path(path),
        }
    }

    /// The underlying image source.
    #[getter]
    fn source(&self) -> PyInferenceImageSource {
        PyInferenceImageSource::from(&self.inner.source)
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceImage(source_kind={:?})",
            match &self.inner.source {
                InferenceImageSource::Bytes(_) => "bytes",
                InferenceImageSource::Path(_) => "path",
            }
        )
    }
}

impl From<InferenceImage> for PyInferenceImage {
    fn from(inner: InferenceImage) -> Self {
        Self { inner }
    }
}

impl From<&InferenceImage> for PyInferenceImage {
    fn from(inner: &InferenceImage) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyChatMessageInput
// ---------------------------------------------------------------------------

/// A single chat message for mistral.rs inference, optionally carrying image
/// attachments. Text-only messages leave ``images`` empty; vision messages
/// supply one or more ``InferenceImage`` entries.
#[gen_stub_pyclass]
#[pyclass(name = "ChatMessageInput", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyChatMessageInput {
    pub(crate) inner: ChatMessageInput,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyChatMessageInput {
    /// Construct a chat message input.
    #[new]
    #[pyo3(signature = (*, role, text, images=None))]
    fn new(role: PyChatRole, text: String, images: Option<Vec<PyInferenceImage>>) -> Self {
        let images = images
            .map(|v| v.into_iter().map(|i| i.inner).collect())
            .unwrap_or_default();
        Self {
            inner: ChatMessageInput {
                role: role.into(),
                text,
                images,
            },
        }
    }

    /// Build a text-only chat message.
    #[staticmethod]
    #[pyo3(name = "text_only")]
    fn text_only(role: PyChatRole, text: String) -> Self {
        Self {
            inner: ChatMessageInput::text(role.into(), text),
        }
    }

    /// Build a chat message with images attached.
    #[staticmethod]
    fn with_images(role: PyChatRole, text: String, images: Vec<PyInferenceImage>) -> Self {
        let imgs = images.into_iter().map(|i| i.inner).collect();
        Self {
            inner: ChatMessageInput::with_images(role.into(), text, imgs),
        }
    }

    /// The role that produced this message.
    #[getter]
    fn role(&self) -> PyChatRole {
        PyChatRole::from(self.inner.role)
    }

    /// The textual content of the message.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Image attachments on this message.
    #[getter]
    fn images(&self) -> Vec<PyInferenceImage> {
        self.inner
            .images
            .iter()
            .map(PyInferenceImage::from)
            .collect()
    }

    /// Whether this message has any image attachments.
    fn has_images(&self) -> bool {
        self.inner.has_images()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatMessageInput(role={:?}, text_len={}, images={})",
            self.inner.role,
            self.inner.text.len(),
            self.inner.images.len()
        )
    }
}

impl From<ChatMessageInput> for PyChatMessageInput {
    fn from(inner: ChatMessageInput) -> Self {
        Self { inner }
    }
}

impl From<&ChatMessageInput> for PyChatMessageInput {
    fn from(inner: &ChatMessageInput) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceToolCall
// ---------------------------------------------------------------------------

/// A tool call returned by the mistral.rs engine.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceToolCall", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyInferenceToolCall {
    pub(crate) inner: InferenceToolCall,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceToolCall {
    /// Construct a tool call.
    #[new]
    #[pyo3(signature = (*, id, name, arguments))]
    fn new(id: String, name: String, arguments: String) -> Self {
        Self {
            inner: InferenceToolCall {
                id,
                name,
                arguments,
            },
        }
    }

    /// Provider-assigned call identifier.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Tool function name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Tool arguments as a JSON string.
    #[getter]
    fn arguments(&self) -> &str {
        &self.inner.arguments
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceToolCall(id={:?}, name={:?})",
            self.inner.id, self.inner.name
        )
    }
}

impl From<InferenceToolCall> for PyInferenceToolCall {
    fn from(inner: InferenceToolCall) -> Self {
        Self { inner }
    }
}

impl From<&InferenceToolCall> for PyInferenceToolCall {
    fn from(inner: &InferenceToolCall) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceUsage
// ---------------------------------------------------------------------------

/// Token usage statistics from a mistral.rs inference call.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceUsage", frozen, from_py_object)]
#[derive(Clone, Default)]
pub struct PyInferenceUsage {
    pub(crate) inner: InferenceUsage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceUsage {
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
            inner: InferenceUsage {
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
            "InferenceUsage(prompt={}, completion={}, total={}, time_sec={})",
            self.inner.prompt_tokens,
            self.inner.completion_tokens,
            self.inner.total_tokens,
            self.inner.total_time_sec
        )
    }
}

impl From<InferenceUsage> for PyInferenceUsage {
    fn from(inner: InferenceUsage) -> Self {
        Self { inner }
    }
}

impl From<&InferenceUsage> for PyInferenceUsage {
    fn from(inner: &InferenceUsage) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceResult
// ---------------------------------------------------------------------------

/// Result of a single non-streaming mistral.rs inference call.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceResult", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyInferenceResult {
    pub(crate) inner: Arc<InferenceResult>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceResult {
    /// Construct an inference result.
    #[new]
    #[pyo3(signature = (
        *,
        finish_reason,
        model,
        usage,
        content=None,
        reasoning_content=None,
        tool_calls=None,
    ))]
    fn new(
        finish_reason: String,
        model: String,
        usage: PyInferenceUsage,
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<PyInferenceToolCall>>,
    ) -> Self {
        let tool_calls = tool_calls
            .map(|v| v.into_iter().map(|tc| tc.inner).collect())
            .unwrap_or_default();
        Self {
            inner: Arc::new(InferenceResult {
                content,
                reasoning_content,
                tool_calls,
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

    /// Reasoning / chain-of-thought content, if the model exposes it.
    #[getter]
    fn reasoning_content(&self) -> Option<&str> {
        self.inner.reasoning_content.as_deref()
    }

    /// Tool calls requested by the model.
    #[getter]
    fn tool_calls(&self) -> Vec<PyInferenceToolCall> {
        self.inner
            .tool_calls
            .iter()
            .map(PyInferenceToolCall::from)
            .collect()
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
    fn usage(&self) -> PyInferenceUsage {
        PyInferenceUsage::from(self.inner.usage.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceResult(model={:?}, finish_reason={:?})",
            self.inner.model, self.inner.finish_reason
        )
    }
}

impl From<InferenceResult> for PyInferenceResult {
    fn from(inner: InferenceResult) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceChunk
// ---------------------------------------------------------------------------

/// A single chunk from streaming mistral.rs inference.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceChunk", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyInferenceChunk {
    pub(crate) inner: InferenceChunk,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceChunk {
    /// Construct an inference chunk.
    #[new]
    #[pyo3(signature = (*, delta=None, reasoning_delta=None, tool_calls=None, finish_reason=None))]
    fn new(
        delta: Option<String>,
        reasoning_delta: Option<String>,
        tool_calls: Option<Vec<PyInferenceToolCall>>,
        finish_reason: Option<String>,
    ) -> Self {
        let tool_calls = tool_calls
            .map(|v| v.into_iter().map(|tc| tc.inner).collect())
            .unwrap_or_default();
        Self {
            inner: InferenceChunk {
                delta,
                reasoning_delta,
                tool_calls,
                finish_reason,
            },
        }
    }

    /// Incremental text content.
    #[getter]
    fn delta(&self) -> Option<&str> {
        self.inner.delta.as_deref()
    }

    /// Incremental reasoning content.
    #[getter]
    fn reasoning_delta(&self) -> Option<&str> {
        self.inner.reasoning_delta.as_deref()
    }

    /// Tool calls completed in this chunk.
    #[getter]
    fn tool_calls(&self) -> Vec<PyInferenceToolCall> {
        self.inner
            .tool_calls
            .iter()
            .map(PyInferenceToolCall::from)
            .collect()
    }

    /// Set on the final chunk when generation stops.
    #[getter]
    fn finish_reason(&self) -> Option<&str> {
        self.inner.finish_reason.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceChunk(delta={:?}, finish_reason={:?})",
            self.inner.delta, self.inner.finish_reason
        )
    }
}

impl From<InferenceChunk> for PyInferenceChunk {
    fn from(inner: InferenceChunk) -> Self {
        Self { inner }
    }
}

impl From<&InferenceChunk> for PyInferenceChunk {
    fn from(inner: &InferenceChunk) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyInferenceChunkStream -- async iterator over streamed chunks
// ---------------------------------------------------------------------------

/// Async iterator over streamed mistral.rs inference chunks.
///
/// Implements the Python ``__aiter__`` / ``__anext__`` protocol so the stream
/// can be consumed with ``async for chunk in stream: ...``. Yields
/// ``InferenceChunk`` items and terminates with ``StopAsyncIteration`` once
/// the underlying engine stream is exhausted.
#[gen_stub_pyclass]
#[pyclass(name = "InferenceChunkStream")]
pub struct PyInferenceChunkStream {
    pub(crate) inner: Arc<Mutex<Option<InferenceChunkStream>>>,
}

impl PyInferenceChunkStream {
    /// Wrap a raw mistral.rs chunk stream.
    pub fn new(stream: InferenceChunkStream) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(stream))),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInferenceChunkStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, InferenceChunk]", imports = ("typing",)))]
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
                Some(Ok(chunk)) => Ok(PyInferenceChunk::from(chunk)),
                Some(Err(e)) => {
                    *guard = None;
                    Err(MistralRsError::new_err(e.to_string()))
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
