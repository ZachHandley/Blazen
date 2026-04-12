//! Python wrapper for the fal.ai compute + LLM provider.
//!
//! Exposes [`FalProvider`](blazen_llm::providers::fal::FalProvider) to Python
//! with all compute, media generation, and LLM completion capabilities.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio_stream::StreamExt;

use crate::compute::job::{PyComputeRequest, PyJobHandle};
use crate::compute::request_types::{
    PyBackgroundRemovalRequest, PyImageRequest, PyMusicRequest, PySpeechRequest, PyThreeDRequest,
    PyTranscriptionRequest, PyUpscaleRequest, PyVideoRequest,
};
use crate::compute::result_types::{
    PyAudioResult, PyComputeResult, PyImageResult, PyThreeDResult, PyTranscriptionResult,
    PyVideoResult,
};
use crate::error::{BlazenPyError, blazen_error_to_pyerr};
use crate::providers::completion_model::PyCompletionOptions;
use crate::providers::options::PyFalOptions;
use crate::types::embedding::PyEmbeddingResponse;
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ComputeProvider, ImageGeneration, ThreeDGeneration,
    Transcription, VideoGeneration,
};
use blazen_llm::providers::fal::{FalEmbeddingModel, FalProvider};
use blazen_llm::traits::{CompletionModel, EmbeddingModel};
use blazen_llm::types::{CompletionRequest, ToolDefinition};

// ---------------------------------------------------------------------------
// PyFalProvider
// ---------------------------------------------------------------------------

/// A fal.ai provider that supports LLM completions AND compute operations.
///
/// This is the unified entry point for all fal.ai capabilities:
/// - Image generation, upscaling, and background removal
/// - Video generation (text-to-video, image-to-video)
/// - Audio generation (TTS, music, sound effects)
/// - Audio transcription
/// - 3D model generation
/// - Text embeddings (via the OpenAI-compatible router)
/// - Raw compute job submission
/// - LLM chat completions (default endpoint:
///   ``openrouter/router/openai/v1/chat/completions`` -- ``OpenAiChat``)
///
/// Example:
///     >>> fal = FalProvider(options=FalOptions(api_key="fal-key-..."))
///     >>> result = await fal.generate_image(ImageRequest(prompt="a cat in space"))
///     >>> response = await fal.complete([ChatMessage.user("Hello!")])
#[gen_stub_pyclass]
#[pyclass(name = "FalProvider", from_py_object)]
#[derive(Clone)]
pub struct PyFalProvider {
    inner: Arc<FalProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFalProvider {
    /// Create a new fal.ai provider.
    ///
    /// Args:
    ///     options: Optional [`FalOptions`] for model, endpoint, enterprise,
    ///         and auto-routing configuration.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyFalOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(FalProvider::from_options(opts).map_err(blazen_error_to_pyerr)?),
        })
    }

    // -----------------------------------------------------------------
    // Image methods
    // -----------------------------------------------------------------

    /// Generate images from a text prompt.
    ///
    /// Args:
    ///     request: An [`ImageRequest`] with prompt, dimensions, etc.
    ///
    /// Returns:
    ///     An [`ImageResult`] with images, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn generate_image<'py>(
        &self,
        py: Python<'py>,
        request: PyImageRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ImageGeneration::generate_image(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    /// Upscale an image.
    ///
    /// Args:
    ///     request: An [`UpscaleRequest`] with image_url and scale factor.
    ///
    /// Returns:
    ///     An [`ImageResult`] with the upscaled image, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn upscale_image<'py>(
        &self,
        py: Python<'py>,
        request: PyUpscaleRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ImageGeneration::upscale_image(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    /// Upscale an image via the aura-sr model.
    ///
    /// Args:
    ///     request: An [`UpscaleRequest`] with image_url and scale factor.
    ///
    /// Returns:
    ///     An [`ImageResult`] with the upscaled image, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn upscale_image_aura<'py>(
        &self,
        py: Python<'py>,
        request: PyUpscaleRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .upscale_image_aura(rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    /// Upscale an image via the clarity-upscaler model.
    ///
    /// Args:
    ///     request: An [`UpscaleRequest`] with image_url and scale factor.
    ///
    /// Returns:
    ///     An [`ImageResult`] with the upscaled image, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn upscale_image_clarity<'py>(
        &self,
        py: Python<'py>,
        request: PyUpscaleRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .upscale_image_clarity(rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    /// Upscale an image via the creative-upscaler model.
    ///
    /// Args:
    ///     request: An [`UpscaleRequest`] with image_url and scale factor.
    ///
    /// Returns:
    ///     An [`ImageResult`] with the upscaled image, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn upscale_image_creative<'py>(
        &self,
        py: Python<'py>,
        request: PyUpscaleRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .upscale_image_creative(rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    /// Remove the background from an image.
    ///
    /// Args:
    ///     request: A [`BackgroundRemovalRequest`] with image_url and optional model.
    ///
    /// Returns:
    ///     An [`ImageResult`] with the matted image, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn remove_background<'py>(
        &self,
        py: Python<'py>,
        request: PyBackgroundRemovalRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = BackgroundRemoval::remove_background(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyImageResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model from a text prompt or source image.
    ///
    /// Args:
    ///     request: A [`ThreeDRequest`] with prompt and/or image_url.
    ///
    /// Returns:
    ///     A [`ThreeDResult`] with the generated 3D model, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ThreeDResult]", imports = ("typing",)))]
    fn generate_3d<'py>(
        &self,
        py: Python<'py>,
        request: PyThreeDRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ThreeDGeneration::generate_3d(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyThreeDResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Embeddings
    // -----------------------------------------------------------------

    /// Build a [`FalEmbeddingModel`] sharing this provider's HTTP client and API key.
    ///
    /// Returns:
    ///     A FalEmbeddingModel that can be used to embed text via fal's
    ///     OpenAI-compatible router.
    fn embedding_model(&self) -> PyFalEmbeddingModel {
        PyFalEmbeddingModel {
            inner: Arc::new(self.inner.embedding_model()),
        }
    }

    // -----------------------------------------------------------------
    // Video methods
    // -----------------------------------------------------------------

    /// Generate a video from a text prompt.
    ///
    /// Args:
    ///     request: A [`VideoRequest`] with prompt and optional parameters.
    ///
    /// Returns:
    ///     A [`VideoResult`] with videos, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, VideoResult]", imports = ("typing",)))]
    fn text_to_video<'py>(
        &self,
        py: Python<'py>,
        request: PyVideoRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = VideoGeneration::text_to_video(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyVideoResult { inner: result })
        })
    }

    /// Generate a video from a source image and prompt.
    ///
    /// Args:
    ///     request: A [`VideoRequest`] with prompt and image_url.
    ///
    /// Returns:
    ///     A [`VideoResult`] with videos, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, VideoResult]", imports = ("typing",)))]
    fn image_to_video<'py>(
        &self,
        py: Python<'py>,
        request: PyVideoRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = VideoGeneration::image_to_video(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyVideoResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Audio methods
    // -----------------------------------------------------------------

    /// Synthesize speech from text.
    ///
    /// Args:
    ///     request: A [`SpeechRequest`] with text and optional voice/language.
    ///
    /// Returns:
    ///     An [`AudioResult`] with audio clips, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        request: PySpeechRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::text_to_speech(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }

    /// Generate music from a prompt.
    ///
    /// Args:
    ///     request: A [`MusicRequest`] with prompt and optional duration.
    ///
    /// Returns:
    ///     An [`AudioResult`] with audio clips, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn generate_music<'py>(
        &self,
        py: Python<'py>,
        request: PyMusicRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::generate_music(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }

    /// Generate sound effects from a prompt.
    ///
    /// Args:
    ///     request: A [`MusicRequest`] with prompt and optional duration.
    ///
    /// Returns:
    ///     An [`AudioResult`] with audio clips, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AudioResult]", imports = ("typing",)))]
    fn generate_sfx<'py>(
        &self,
        py: Python<'py>,
        request: PyMusicRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::generate_sfx(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyAudioResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio to text.
    ///
    /// Args:
    ///     request: A [`TranscriptionRequest`] with audio_url and options.
    ///
    /// Returns:
    ///     A [`TranscriptionResult`] with text, segments, language, timing,
    ///     cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, TranscriptionResult]", imports = ("typing",)))]
    fn transcribe<'py>(
        &self,
        py: Python<'py>,
        request: PyTranscriptionRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = Transcription::transcribe(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyTranscriptionResult { inner: result })
        })
    }

    // -----------------------------------------------------------------
    // Raw compute
    // -----------------------------------------------------------------

    /// Submit a compute job and wait for the result.
    ///
    /// Args:
    ///     request: A [`ComputeRequest`] with model and input.
    ///
    /// Returns:
    ///     A [`ComputeResult`] with output, timing, cost, and metadata.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ComputeResult]", imports = ("typing",)))]
    fn run<'py>(&self, py: Python<'py>, request: PyComputeRequest) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ComputeProvider::run(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyComputeResult { inner: result })
        })
    }

    /// Submit a compute job without waiting.
    ///
    /// Args:
    ///     request: A [`ComputeRequest`] with model and input.
    ///
    /// Returns:
    ///     A [`JobHandle`] with id, provider, model, and submitted_at.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, JobHandle]", imports = ("typing",)))]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        request: PyComputeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req = request.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handle = ComputeProvider::submit(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyJobHandle { inner: handle })
        })
    }

    /// Poll the status of a submitted job.
    ///
    /// Args:
    ///     job: The [`JobHandle`] returned by [`submit`].
    ///
    /// Returns:
    ///     A status string: "queued", "running", "completed", "failed", or "cancelled".
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, builtins.str]", imports = ("typing", "builtins")))]
    fn status<'py>(&self, py: Python<'py>, job: PyJobHandle) -> PyResult<Bound<'py, PyAny>> {
        let handle = job.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let status = ComputeProvider::status(inner.as_ref(), &handle)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let status_str = match status {
                blazen_llm::compute::JobStatus::Queued => "queued",
                blazen_llm::compute::JobStatus::Running => "running",
                blazen_llm::compute::JobStatus::Completed => "completed",
                blazen_llm::compute::JobStatus::Failed { .. } => "failed",
                blazen_llm::compute::JobStatus::Cancelled => "cancelled",
            };
            Ok(status_str.to_owned())
        })
    }

    /// Cancel a running or queued job.
    ///
    /// Args:
    ///     job: The [`JobHandle`] returned by [`submit`].
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, None]", imports = ("typing",)))]
    fn cancel<'py>(&self, py: Python<'py>, job: PyJobHandle) -> PyResult<Bound<'py, PyAny>> {
        let handle = job.inner;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            ComputeProvider::cancel(inner.as_ref(), &handle)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(())
        })
    }

    // -----------------------------------------------------------------
    // LLM completion (delegates to CompletionModel trait)
    // -----------------------------------------------------------------

    /// Perform a chat completion via fal-ai/any-llm.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     options: Optional [`CompletionOptions`] for sampling parameters,
    ///         tools, and response format.
    ///
    /// Returns:
    ///     A CompletionResponse with content, model, tool_calls, usage, etc.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, CompletionResponse]", imports = ("typing",)))]
    #[pyo3(signature = (messages, options=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyChatMessage>,
        options: Option<Py<PyCompletionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Building the request requires GIL access because `PyCompletionOptions`
        // holds `tools` / `response_format` as `Py<PyAny>` and we need to
        // convert them with `crate::convert::py_to_json`. We use the `py` param
        // directly, build the fully-owned `CompletionRequest`, and then release
        // the GIL inside future_into_py for the HTTP call.
        let rust_messages: Vec<ChatMessage> = messages.into_iter().map(|m| m.inner).collect();
        let opts_borrow = options.as_ref().map(|o| o.borrow(py));
        let request = build_completion_request(py, rust_messages, opts_borrow.as_deref())?;

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = CompletionModel::complete(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            Ok(PyCompletionResponse { inner: response })
        })
    }

    /// Stream a chat completion, calling a callback for each chunk.
    ///
    /// Args:
    ///     messages: A list of ChatMessage objects.
    ///     on_chunk: Callback function receiving each chunk as a dict.
    ///     options: Optional [`CompletionOptions`] for sampling parameters,
    ///         tools, and response format.
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
        let request = build_completion_request(py, rust_messages, options.as_deref())?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = CompletionModel::stream(inner.as_ref(), request)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()));
                    }
                }
            }
            Ok(())
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[getter]
    fn model_id(&self) -> &str {
        CompletionModel::model_id(self.inner.as_ref())
    }

    fn __repr__(&self) -> String {
        format!(
            "FalProvider(model_id='{}')",
            CompletionModel::model_id(self.inner.as_ref())
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [`CompletionRequest`] from messages and optional [`PyCompletionOptions`].
///
/// Mirrors the helper in [`crate::providers::completion_model`] so that
/// [`PyFalProvider::complete`] and [`PyFalProvider::stream`] accept the same
/// typed options object as [`PyCompletionModel`].
fn build_completion_request(
    py: Python<'_>,
    messages: Vec<ChatMessage>,
    options: Option<&PyCompletionOptions>,
) -> PyResult<CompletionRequest> {
    let mut request = CompletionRequest::new(messages);

    if let Some(opts) = options {
        if let Some(t) = opts.temperature {
            request = request.with_temperature(t);
        }
        if let Some(mt) = opts.max_tokens {
            request = request.with_max_tokens(mt);
        }
        if let Some(tp) = opts.top_p {
            request = request.with_top_p(tp);
        }
        if let Some(ref m) = opts.model {
            request = request.with_model(m.clone());
        }
        if let Some(ref tools_py) = opts.tools {
            let tools_bound = tools_py.bind(py);
            let tools_list: &Bound<'_, pyo3::types::PyList> = tools_bound.cast()?;
            let tool_vec: Vec<Bound<'_, PyAny>> = tools_list.iter().collect();
            let rust_tools = extract_tool_definitions(py, &tool_vec)?;
            request = request.with_tools(rust_tools);
        }
        if let Some(ref fmt) = opts.response_format {
            let schema = crate::convert::py_to_json(py, fmt.bind(py))?;
            request = request.with_response_format(schema);
        }
    }

    Ok(request)
}

/// Extract a list of [`ToolDefinition`] from Python dicts (or dict-like objects).
fn extract_tool_definitions(
    py: Python<'_>,
    tool_list: &[Bound<'_, PyAny>],
) -> PyResult<Vec<ToolDefinition>> {
    let mut rust_tools = Vec::with_capacity(tool_list.len());
    for tool in tool_list {
        let name: String = tool.get_item("name")?.extract()?;
        let description: String = tool.get_item("description")?.extract()?;
        let parameters = crate::convert::py_to_json(py, &tool.get_item("parameters")?)?;
        rust_tools.push(ToolDefinition {
            name,
            description,
            parameters,
        });
    }
    Ok(rust_tools)
}

// ---------------------------------------------------------------------------
// PyFalEmbeddingModel
// ---------------------------------------------------------------------------

/// A fal.ai embedding model.
///
/// Wraps [`FalEmbeddingModel`] and exposes the
/// [`EmbeddingModel`](blazen_llm::traits::EmbeddingModel) interface to
/// Python. Constructed via [`FalProvider::embedding_model`].
///
/// Example:
///     >>> fal = FalProvider(options=FalOptions(api_key="fal-..."))
///     >>> em = fal.embedding_model()
///     >>> resp = await em.embed(["hello", "world"])
///     >>> print(len(resp.embeddings))  # 2
#[gen_stub_pyclass]
#[pyclass(name = "FalEmbeddingModel", from_py_object)]
#[derive(Clone)]
pub struct PyFalEmbeddingModel {
    inner: Arc<FalEmbeddingModel>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFalEmbeddingModel {
    /// Get the underlying embedding model id.
    #[getter]
    fn model_id(&self) -> &str {
        EmbeddingModel::model_id(self.inner.as_ref())
    }

    /// Get the dimensionality of the produced embedding vectors.
    #[getter]
    fn dimensions(&self) -> usize {
        EmbeddingModel::dimensions(self.inner.as_ref())
    }

    /// Embed one or more texts.
    ///
    /// Args:
    ///     texts: A list of strings to embed.
    ///
    /// Returns:
    ///     An EmbeddingResponse with embeddings, model, usage, and cost.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, EmbeddingResponse]", imports = ("typing",)))]
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = EmbeddingModel::embed(inner.as_ref(), &texts)
                .await
                .map_err(BlazenPyError::from)?;
            Ok(PyEmbeddingResponse { inner: response })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "FalEmbeddingModel(model_id='{}', dimensions={})",
            EmbeddingModel::model_id(self.inner.as_ref()),
            EmbeddingModel::dimensions(self.inner.as_ref()),
        )
    }
}
