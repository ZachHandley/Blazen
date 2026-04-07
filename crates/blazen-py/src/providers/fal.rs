//! Python wrapper for the fal.ai compute + LLM provider.
//!
//! Exposes [`FalProvider`](blazen_llm::providers::fal::FalProvider) to Python
//! with all compute, media generation, and LLM completion capabilities.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pythonize::depythonize;
use tokio_stream::StreamExt;

use crate::convert::JsonValue;
use crate::error::{BlazenPyError, blazen_error_to_pyerr};
use crate::types::{PyChatMessage, PyCompletionResponse};
use blazen_llm::ChatMessage;
use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, BackgroundRemovalRequest, ComputeProvider, ComputeRequest,
    ImageGeneration, ImageRequest, MusicRequest, SpeechRequest, ThreeDGeneration, ThreeDRequest,
    Transcription, TranscriptionRequest, UpscaleRequest, VideoGeneration, VideoRequest,
};
use blazen_llm::providers::fal::{FalEmbeddingModel, FalProvider};
use blazen_llm::traits::{CompletionModel, EmbeddingModel};
use blazen_llm::types::CompletionRequest;

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
///     >>> fal = FalProvider(api_key="fal-key-...")
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
    ///     api_key: Your fal.ai API key.
    ///     options: Optional dict of options (model, endpoint, enterprise,
    ///         auto_route_modality) -- deserialized into core ``FalOptions``.
    #[new]
    #[pyo3(signature = (*, api_key, options=None))]
    fn new(api_key: &str, options: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut provider = FalProvider::new(api_key);
        if let Some(opts_raw) = options {
            let opts: blazen_llm::types::provider_options::FalOptions = depythonize(opts_raw)?;
            if let Some(m) = opts.base.model {
                provider = provider.with_llm_model(m);
            }
            if let Some(ep) = opts.endpoint {
                use blazen_llm::providers::fal::FalLlmEndpoint;
                use blazen_llm::types::provider_options::FalLlmEndpointKind;
                let endpoint = match ep {
                    FalLlmEndpointKind::OpenAiChat => FalLlmEndpoint::OpenAiChat,
                    FalLlmEndpointKind::OpenAiResponses => FalLlmEndpoint::OpenAiResponses,
                    FalLlmEndpointKind::OpenAiEmbeddings => FalLlmEndpoint::OpenAiEmbeddings,
                    FalLlmEndpointKind::OpenRouter => FalLlmEndpoint::OpenRouter {
                        enterprise: opts.enterprise,
                    },
                    FalLlmEndpointKind::AnyLlm => FalLlmEndpoint::AnyLlm {
                        enterprise: opts.enterprise,
                    },
                };
                provider = provider.with_llm_endpoint(endpoint);
            } else if opts.enterprise {
                provider = provider.with_enterprise();
            }
            provider = provider.with_auto_route_modality(opts.auto_route_modality);
        }
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    // -----------------------------------------------------------------
    // Image methods
    // -----------------------------------------------------------------

    /// Generate images from a text prompt.
    ///
    /// Args:
    ///     request: An ImageRequest with prompt, dimensions, etc.
    ///
    /// Returns:
    ///     A dict with images, timing, cost, and metadata.
    fn generate_image<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: ImageRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ImageGeneration::generate_image(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Upscale an image.
    ///
    /// Args:
    ///     request: An UpscaleRequest with image_url and scale factor.
    ///
    /// Returns:
    ///     A dict with the upscaled image, timing, cost, and metadata.
    fn upscale_image<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: UpscaleRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ImageGeneration::upscale_image(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Upscale an image via the aura-sr model.
    fn upscale_image_aura<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: UpscaleRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .upscale_image_aura(rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Upscale an image via the clarity-upscaler model.
    fn upscale_image_clarity<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: UpscaleRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .upscale_image_clarity(rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Upscale an image via the creative-upscaler model.
    fn upscale_image_creative<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: UpscaleRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .upscale_image_creative(rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Remove the background from an image.
    ///
    /// Args:
    ///     image_url: URL of the source image.
    ///     model: Optional model id override.
    ///
    /// Returns:
    ///     A dict with the matted image, timing, cost, and metadata.
    #[pyo3(signature = (*, image_url, model=None))]
    fn remove_background<'py>(
        &self,
        py: Python<'py>,
        image_url: String,
        model: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let request = BackgroundRemovalRequest {
                image_url,
                model,
                parameters: serde_json::Value::Null,
            };
            let result = BackgroundRemoval::remove_background(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    // -----------------------------------------------------------------
    // 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model from a text prompt or source image.
    ///
    /// Args:
    ///     request: A ThreeDRequest with prompt and/or image_url.
    ///
    /// Returns:
    ///     A dict with the generated 3D model, timing, cost, and metadata.
    fn generate_3d<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: ThreeDRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = ThreeDGeneration::generate_3d(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
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
    ///     request: A VideoRequest with prompt and optional parameters.
    ///
    /// Returns:
    ///     A dict with videos, timing, cost, and metadata.
    fn text_to_video<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: VideoRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = VideoGeneration::text_to_video(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Generate a video from a source image and prompt.
    ///
    /// Args:
    ///     request: A VideoRequest with prompt and image_url.
    ///
    /// Returns:
    ///     A dict with videos, timing, cost, and metadata.
    fn image_to_video<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: VideoRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = VideoGeneration::image_to_video(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    // -----------------------------------------------------------------
    // Audio methods
    // -----------------------------------------------------------------

    /// Synthesize speech from text.
    ///
    /// Args:
    ///     request: A SpeechRequest with text and optional voice/language.
    ///
    /// Returns:
    ///     A dict with audio clips, timing, cost, and metadata.
    fn text_to_speech<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: SpeechRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::text_to_speech(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Generate music from a prompt.
    ///
    /// Args:
    ///     request: A MusicRequest with prompt and optional duration.
    ///
    /// Returns:
    ///     A dict with audio clips, timing, cost, and metadata.
    fn generate_music<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: MusicRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::generate_music(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Generate sound effects from a prompt.
    ///
    /// Args:
    ///     request: A MusicRequest with prompt and optional duration.
    ///
    /// Returns:
    ///     A dict with audio clips, timing, cost, and metadata.
    fn generate_sfx<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: MusicRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = AudioGeneration::generate_sfx(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio to text.
    ///
    /// Args:
    ///     request: A TranscriptionRequest with audio_url and options.
    ///
    /// Returns:
    ///     A dict with text, segments, language, timing, cost, and metadata.
    fn transcribe<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_req: TranscriptionRequest = depythonize(request)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = Transcription::transcribe(inner.as_ref(), rust_req)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    // -----------------------------------------------------------------
    // Raw compute
    // -----------------------------------------------------------------

    /// Submit a compute job and wait for the result.
    ///
    /// Args:
    ///     model: The fal.ai model endpoint (e.g. "fal-ai/flux/dev").
    ///     input: Input parameters as a dict.
    ///
    /// Returns:
    ///     A dict with output, timing, cost, and metadata.
    #[pyo3(signature = (*, model, input))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        model: String,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input_json = crate::convert::py_to_json(py, input)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let request = ComputeRequest {
                model,
                input: input_json,
                webhook: None,
            };
            let result = ComputeProvider::run(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Submit a compute job without waiting (returns a job handle dict).
    ///
    /// Args:
    ///     model: The fal.ai model endpoint.
    ///     input: Input parameters as a dict.
    ///
    /// Returns:
    ///     A dict with id, provider, model, and submitted_at.
    #[pyo3(signature = (*, model, input))]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        model: String,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input_json = crate::convert::py_to_json(py, input)?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let request = ComputeRequest {
                model,
                input: input_json,
                webhook: None,
            };
            let handle = ComputeProvider::submit(inner.as_ref(), request)
                .await
                .map_err(blazen_error_to_pyerr)?;
            let json = serde_json::to_value(&handle)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(JsonValue(json))
        })
    }

    /// Poll the status of a submitted job.
    ///
    /// Args:
    ///     job_id: The job identifier returned by submit().
    ///     model: The model endpoint the job was submitted to.
    ///
    /// Returns:
    ///     A status string: "queued", "running", "completed", "failed", or "cancelled".
    #[pyo3(signature = (*, job_id, model))]
    fn status<'py>(
        &self,
        py: Python<'py>,
        job_id: String,
        model: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handle = blazen_llm::compute::JobHandle {
                id: job_id,
                provider: "fal".to_owned(),
                model,
                submitted_at: chrono::Utc::now(),
            };
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
    ///     job_id: The job identifier returned by submit().
    ///     model: The model endpoint the job was submitted to.
    #[pyo3(signature = (*, job_id, model))]
    fn cancel<'py>(
        &self,
        py: Python<'py>,
        job_id: String,
        model: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let handle = blazen_llm::compute::JobHandle {
                id: job_id,
                provider: "fal".to_owned(),
                model,
                submitted_at: chrono::Utc::now(),
            };
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
    ///     temperature: Optional sampling temperature (0.0-2.0).
    ///     max_tokens: Optional maximum tokens to generate.
    ///     model: Optional model override for this request.
    ///     response_format: Optional JSON schema dict for structured output.
    ///
    /// Returns:
    ///     A CompletionResponse with content, model, tool_calls, usage, etc.
    #[pyo3(signature = (messages, *, temperature=None, max_tokens=None, model=None, response_format=None))]
    fn complete<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
        response_format: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

        let mut request = CompletionRequest::new(rust_messages);
        if let Some(t) = temperature {
            request = request.with_temperature(t);
        }
        if let Some(mt) = max_tokens {
            request = request.with_max_tokens(mt);
        }
        if let Some(m) = model {
            request = request.with_model(m);
        }
        if let Some(fmt) = response_format {
            let schema = crate::convert::py_to_json(py, fmt)?;
            request = request.with_response_format(schema);
        }

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
    ///     temperature: Optional sampling temperature (0.0-2.0).
    ///     max_tokens: Optional maximum tokens to generate.
    ///     model: Optional model override for this request.
    #[pyo3(signature = (messages, on_chunk, *, temperature=None, max_tokens=None, model=None))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<PyRef<'py, PyChatMessage>>,
        on_chunk: Py<PyAny>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rust_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();

        let mut request = CompletionRequest::new(rust_messages);
        if let Some(t) = temperature {
            request = request.with_temperature(t);
        }
        if let Some(mt) = max_tokens {
            request = request.with_max_tokens(mt);
        }
        if let Some(m) = model {
            request = request.with_model(m);
        }

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
// PyFalEmbeddingModel
// ---------------------------------------------------------------------------

/// A fal.ai embedding model.
///
/// Wraps [`FalEmbeddingModel`] and exposes the
/// [`EmbeddingModel`](blazen_llm::traits::EmbeddingModel) interface to
/// Python. Constructed via [`FalProvider::embedding_model`].
///
/// Example:
///     >>> fal = FalProvider(api_key="fal-...")
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
    fn embed<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = EmbeddingModel::embed(inner.as_ref(), &texts)
                .await
                .map_err(BlazenPyError::from)?;
            Python::attach(|py| -> PyResult<Py<PyAny>> {
                let obj = pythonize::pythonize(py, &response)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(obj.unbind())
            })
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
