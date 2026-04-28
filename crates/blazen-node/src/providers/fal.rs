//! JavaScript bindings for the fal.ai compute platform provider.
//!
//! Exposes [`JsFalProvider`] as a NAPI class with factory constructor,
//! compute methods (image, video, audio, transcription), raw compute
//! (run/submit/status/cancel), and LLM completion support.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::CompletionModel;
use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ComputeProvider, ImageGeneration, ThreeDGeneration,
    Transcription, VideoGeneration,
};
use blazen_llm::providers::fal::{FalEmbeddingModel, FalProvider};
use blazen_llm::types::{ChatMessage, CompletionRequest};

use crate::compute::JsJobStatus;
use crate::error::blazen_error_to_napi;
use crate::generated::{
    JsAudioResult, JsBackgroundRemovalRequest, JsComputeRequest, JsComputeResult, JsFalOptions,
    JsImageRequest, JsImageResult, JsJobHandle, JsMusicRequest, JsSpeechRequest, JsThreeDRequest,
    JsThreeDResult, JsTranscriptionRequest, JsTranscriptionResult, JsUpscaleRequest,
    JsVideoRequest, JsVideoResult,
};
use crate::types::{JsChatMessage, JsCompletionResponse, build_response};

// ---------------------------------------------------------------------------
// FalProvider NAPI class
// ---------------------------------------------------------------------------

/// A fal.ai compute platform provider with image, video, audio, transcription,
/// and LLM capabilities.
///
/// ```typescript
/// const fal = FalProvider.create();
/// const result = await fal.generateImage({ prompt: "a sunset" });
/// const response = await fal.complete([ChatMessage.user("Hi")]);
/// ```
#[napi(js_name = "FalProvider")]
pub struct JsFalProvider {
    inner: Arc<FalProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsFalProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new fal.ai provider.
    ///
    /// `options` optionally configures the LLM model, endpoint family,
    /// enterprise tier, and modality auto-routing. Defaults to the
    /// OpenAI-compatible chat-completions endpoint (`OpenAiChat`).
    #[napi(factory)]
    pub fn create(options: Option<JsFalOptions>) -> Result<Self> {
        let opts: blazen_llm::types::provider_options::FalOptions =
            options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(FalProvider::from_options(opts).map_err(blazen_error_to_napi)?),
        })
    }

    // -----------------------------------------------------------------
    // Model ID getter
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Compute methods -- Image
    // -----------------------------------------------------------------

    /// Generate images from a text prompt.
    #[napi(js_name = "generateImage")]
    pub async fn generate_image(&self, request: JsImageRequest) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::ImageRequest = request.into();
        let result = self
            .inner
            .generate_image(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Upscale an image.
    #[napi(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, request: JsUpscaleRequest) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::UpscaleRequest = request.into();
        let result = self
            .inner
            .upscale_image(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Upscale an image via the aura-sr model.
    #[napi(js_name = "upscaleImageAura")]
    pub async fn upscale_image_aura(&self, request: JsUpscaleRequest) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::UpscaleRequest = request.into();
        let result = self
            .inner
            .upscale_image_aura(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Upscale an image via the clarity-upscaler model.
    #[napi(js_name = "upscaleImageClarity")]
    pub async fn upscale_image_clarity(&self, request: JsUpscaleRequest) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::UpscaleRequest = request.into();
        let result = self
            .inner
            .upscale_image_clarity(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Upscale an image via the creative-upscaler model.
    #[napi(js_name = "upscaleImageCreative")]
    pub async fn upscale_image_creative(&self, request: JsUpscaleRequest) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::UpscaleRequest = request.into();
        let result = self
            .inner
            .upscale_image_creative(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Remove the background from an image.
    #[napi(js_name = "removeBackground")]
    pub async fn remove_background(
        &self,
        request: JsBackgroundRemovalRequest,
    ) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::BackgroundRemovalRequest = request.into();
        let result = BackgroundRemoval::remove_background(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Compute methods -- 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model from a text prompt or source image.
    #[napi(js_name = "generate3d")]
    pub async fn generate_3d(&self, request: JsThreeDRequest) -> Result<JsThreeDResult> {
        let rust_req: blazen_llm::compute::ThreeDRequest = request.into();
        let result = ThreeDGeneration::generate_3d(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Embeddings
    // -----------------------------------------------------------------

    /// Build a [`JsFalEmbeddingModel`] sharing this provider's HTTP client
    /// and API key.
    #[napi(js_name = "embeddingModel")]
    pub fn embedding_model(&self) -> JsFalEmbeddingModel {
        JsFalEmbeddingModel {
            inner: Arc::new(self.inner.embedding_model()),
        }
    }

    // -----------------------------------------------------------------
    // Compute methods -- Video
    // -----------------------------------------------------------------

    /// Generate a video from a text prompt.
    #[napi(js_name = "textToVideo")]
    pub async fn text_to_video(&self, request: JsVideoRequest) -> Result<JsVideoResult> {
        let rust_req: blazen_llm::compute::VideoRequest = request.into();
        let result = self
            .inner
            .text_to_video(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate a video from a source image and prompt.
    #[napi(js_name = "imageToVideo")]
    pub async fn image_to_video(&self, request: JsVideoRequest) -> Result<JsVideoResult> {
        let rust_req: blazen_llm::compute::VideoRequest = request.into();
        let result = self
            .inner
            .image_to_video(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Compute methods -- Audio
    // -----------------------------------------------------------------

    /// Synthesize speech from text.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsSpeechRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::SpeechRequest = request.into();
        let result = self
            .inner
            .text_to_speech(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate music from a prompt.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(&self, request: JsMusicRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::MusicRequest = request.into();
        let result = self
            .inner
            .generate_music(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate sound effects from a prompt.
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(&self, request: JsMusicRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::MusicRequest = request.into();
        let result = self
            .inner
            .generate_sfx(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Compute methods -- Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio to text.
    #[napi]
    pub async fn transcribe(
        &self,
        request: JsTranscriptionRequest,
    ) -> Result<JsTranscriptionResult> {
        let rust_req: blazen_llm::compute::TranscriptionRequest = request.into();
        let result = self
            .inner
            .transcribe(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Raw compute methods
    // -----------------------------------------------------------------

    /// Run a model synchronously (submit + wait for result).
    #[napi]
    pub async fn run(&self, request: JsComputeRequest) -> Result<JsComputeResult> {
        let rust_req: blazen_llm::compute::ComputeRequest = request.into();
        let result = self
            .inner
            .run(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Submit a job to the queue and return a job handle.
    #[napi]
    pub async fn submit(&self, request: JsComputeRequest) -> Result<JsJobHandle> {
        let rust_req: blazen_llm::compute::ComputeRequest = request.into();
        let handle = self
            .inner
            .submit(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(handle.into())
    }

    /// Get the status of a submitted job.
    #[napi]
    pub async fn status(&self, handle: JsJobHandle) -> Result<JsJobStatus> {
        let core_handle: blazen_llm::compute::JobHandle = handle.into();
        let status = self
            .inner
            .status(&core_handle)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(match status {
            blazen_llm::compute::JobStatus::Queued => JsJobStatus::Queued,
            blazen_llm::compute::JobStatus::Running => JsJobStatus::Running,
            blazen_llm::compute::JobStatus::Completed => JsJobStatus::Completed,
            blazen_llm::compute::JobStatus::Failed { .. } => JsJobStatus::Failed,
            blazen_llm::compute::JobStatus::Cancelled => JsJobStatus::Cancelled,
        })
    }

    /// Cancel a submitted job.
    #[napi]
    pub async fn cancel(&self, handle: JsJobHandle) -> Result<()> {
        let core_handle: blazen_llm::compute::JobHandle = handle.into();
        self.inner
            .cancel(&core_handle)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(())
    }

    /// Wait for a submitted job to complete and return its result.
    ///
    /// Equivalent to the Rust [`ComputeProvider::result`] method. Use this
    /// after [`submit`](Self::submit) when you want to perform additional
    /// work between submission and retrieval; otherwise prefer
    /// [`run`](Self::run) which submits and waits in one call.
    #[napi(js_name = "awaitCompletion")]
    pub async fn await_completion(&self, handle: JsJobHandle) -> Result<JsComputeResult> {
        let core_handle: blazen_llm::compute::JobHandle = handle.into();
        let result = self
            .inner
            .result(core_handle)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Alias for [`awaitCompletion`](Self::await_completion).
    #[napi]
    pub async fn result(&self, handle: JsJobHandle) -> Result<JsComputeResult> {
        let core_handle: blazen_llm::compute::JobHandle = handle.into();
        let result = self
            .inner
            .result(core_handle)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // LLM completion
    // -----------------------------------------------------------------

    /// Perform a chat completion via fal.ai's `any-llm` proxy.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(blazen_error_to_napi)?;

        Ok(build_response(response))
    }
}

// ---------------------------------------------------------------------------
// JsFalEmbeddingModel
// ---------------------------------------------------------------------------

/// A fal.ai embedding model.
///
/// Wraps [`FalEmbeddingModel`] and exposes the
/// [`EmbeddingModel`](blazen_llm::EmbeddingModel) interface to JavaScript.
/// Constructed via [`JsFalProvider::embedding_model`].
///
/// ```typescript
/// const fal = FalProvider.create();
/// const em = fal.embeddingModel();
/// const vectors = await em.embed(["hello", "world"]);
/// console.log(vectors.length); // 2
/// ```
#[napi(js_name = "FalEmbeddingModel")]
pub struct JsFalEmbeddingModel {
    inner: Arc<FalEmbeddingModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsFalEmbeddingModel {
    /// Get the underlying embedding model id.
    #[napi(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        use blazen_llm::EmbeddingModel;
        self.inner.model_id().to_owned()
    }

    /// Get the dimensionality of the produced embedding vectors.
    #[napi(getter)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dimensions(&self) -> u32 {
        use blazen_llm::EmbeddingModel;
        self.inner.dimensions() as u32
    }

    /// Embed one or more texts.
    ///
    /// Returns a list of embedding vectors (one per input text).
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        use blazen_llm::EmbeddingModel;
        let inner = Arc::clone(&self.inner);
        let response = inner.embed(&texts).await.map_err(|e| {
            napi::Error::new(napi::Status::GenericFailure, format!("fal embed: {e}"))
        })?;
        // Convert Vec<Vec<f32>> to Vec<Vec<f64>> for JS compatibility
        // (JS Number is f64).
        Ok(response
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect())
            .collect())
    }
}
