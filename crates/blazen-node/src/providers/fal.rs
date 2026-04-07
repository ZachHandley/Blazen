//! JavaScript bindings for the fal.ai compute platform provider.
//!
//! Exposes [`JsFalProvider`] as a NAPI class with factory constructor,
//! compute methods (image, video, audio, transcription), raw compute
//! (run/submit/status/cancel), and LLM completion support.

use std::sync::Arc;

use chrono::Utc;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::CompletionModel;
use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, BackgroundRemovalRequest, ComputeProvider, ComputeRequest,
    ImageGeneration, ThreeDGeneration, Transcription, VideoGeneration,
};
use blazen_llm::providers::fal::{FalEmbeddingModel, FalProvider};
use blazen_llm::types::{ChatMessage, CompletionRequest};

use crate::compute::{
    JsImageRequest, JsMusicRequest, JsSpeechRequest, JsThreeDRequest, JsTranscriptionRequest,
    JsUpscaleRequest, JsVideoRequest,
};
use crate::error::blazen_error_to_napi;
use crate::types::{JsChatMessage, JsCompletionResponse, build_response};

// ---------------------------------------------------------------------------
// JsFalLlmEndpoint
// ---------------------------------------------------------------------------

/// The fal.ai LLM endpoint family.
///
/// Selects which fal LLM URL/body schema to use. Defaults to `OpenAiChat`
/// (the OpenAI-compatible chat-completions surface on fal).
#[napi(string_enum, js_name = "FalLlmEndpoint")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsFalLlmEndpoint {
    /// OpenAI-compatible chat-completions endpoint (default).
    #[napi(value = "openai_chat")]
    OpenAiChat,
    /// OpenAI-compatible Responses endpoint.
    #[napi(value = "openai_responses")]
    OpenAiResponses,
    /// OpenAI-compatible embeddings endpoint.
    #[napi(value = "openai_embeddings")]
    OpenAiEmbeddings,
    /// `OpenRouter` proxy via fal.
    #[napi(value = "openrouter")]
    OpenRouter,
    /// `OpenRouter` proxy via fal (enterprise/SOC2 tier).
    #[napi(value = "openrouter_enterprise")]
    OpenRouterEnterprise,
    /// fal-ai/any-llm proxy.
    #[napi(value = "any_llm")]
    AnyLlm,
    /// fal-ai/any-llm proxy (enterprise/SOC2 tier).
    #[napi(value = "any_llm_enterprise")]
    AnyLlmEnterprise,
}

impl From<JsFalLlmEndpoint> for blazen_llm::providers::fal::FalLlmEndpoint {
    fn from(j: JsFalLlmEndpoint) -> Self {
        use blazen_llm::providers::fal::FalLlmEndpoint as F;
        match j {
            JsFalLlmEndpoint::OpenAiChat => F::OpenAiChat,
            JsFalLlmEndpoint::OpenAiResponses => F::OpenAiResponses,
            JsFalLlmEndpoint::OpenAiEmbeddings => F::OpenAiEmbeddings,
            JsFalLlmEndpoint::OpenRouter => F::OpenRouter { enterprise: false },
            JsFalLlmEndpoint::OpenRouterEnterprise => F::OpenRouter { enterprise: true },
            JsFalLlmEndpoint::AnyLlm => F::AnyLlm { enterprise: false },
            JsFalLlmEndpoint::AnyLlmEnterprise => F::AnyLlm { enterprise: true },
        }
    }
}

// ---------------------------------------------------------------------------
// JsFalOptions
// ---------------------------------------------------------------------------

/// Configuration options for [`JsFalProvider`].
///
/// ```typescript
/// const fal = FalProvider.create("fal-key-...", {
///   model: "anthropic/claude-sonnet-4.5",
///   endpoint: "openai_chat",
///   enterprise: false,
///   autoRouteModality: true,
/// });
/// ```
#[napi(object)]
pub struct JsFalOptions {
    /// Underlying LLM model name (e.g. `"anthropic/claude-sonnet-4.5"`).
    pub model: Option<String>,
    /// The fal endpoint family to target. Defaults to `OpenAiChat`.
    pub endpoint: Option<JsFalLlmEndpoint>,
    /// Promote the endpoint to its enterprise / SOC2-eligible variant.
    pub enterprise: Option<bool>,
    /// Auto-route the request to the matching vision/audio/video variant
    /// when the message content contains media. Defaults to `true`.
    #[napi(js_name = "autoRouteModality")]
    pub auto_route_modality: Option<bool>,
}

/// Apply [`JsFalOptions`] to a [`FalProvider`] builder.
pub(crate) fn apply_fal_options(
    mut provider: FalProvider,
    options: Option<JsFalOptions>,
) -> FalProvider {
    if let Some(opts) = options {
        if let Some(model) = opts.model {
            provider = provider.with_llm_model(model);
        }
        if let Some(ep) = opts.endpoint {
            provider = provider.with_llm_endpoint(ep.into());
        }
        if opts.enterprise.unwrap_or(false) {
            provider = provider.with_enterprise();
        }
        if let Some(arm) = opts.auto_route_modality {
            provider = provider.with_auto_route_modality(arm);
        }
    }
    provider
}

// ---------------------------------------------------------------------------
// Converters: Js*Request -> Rust request types
// ---------------------------------------------------------------------------

fn js_to_image_request(req: JsImageRequest) -> blazen_llm::compute::ImageRequest {
    let mut r = blazen_llm::compute::ImageRequest::new(&req.prompt);
    if let (Some(w), Some(h)) = (req.width, req.height) {
        r = r.with_size(w, h);
    }
    if let Some(n) = req.num_images {
        r = r.with_count(n);
    }
    if let Some(np) = req.negative_prompt {
        r = r.with_negative_prompt(np);
    }
    if let Some(m) = req.model {
        r = r.with_model(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

fn js_to_upscale_request(req: JsUpscaleRequest) -> blazen_llm::compute::UpscaleRequest {
    #[allow(clippy::cast_possible_truncation)]
    let mut r = blazen_llm::compute::UpscaleRequest::new(&req.image_url, req.scale as f32);
    if let Some(m) = req.model {
        r = r.with_model(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

fn js_to_video_request(req: JsVideoRequest) -> blazen_llm::compute::VideoRequest {
    let mut r = if let Some(ref image_url) = req.image_url {
        blazen_llm::compute::VideoRequest::for_image(image_url, &req.prompt)
    } else {
        blazen_llm::compute::VideoRequest::new(&req.prompt)
    };
    #[allow(clippy::cast_possible_truncation)]
    if let Some(d) = req.duration_seconds {
        r = r.with_duration(d as f32);
    }
    if let (Some(w), Some(h)) = (req.width, req.height) {
        r = r.with_size(w, h);
    }
    if let Some(m) = req.model {
        r = r.with_model(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

fn js_to_speech_request(req: JsSpeechRequest) -> blazen_llm::compute::SpeechRequest {
    let mut r = blazen_llm::compute::SpeechRequest::new(&req.text);
    if let Some(v) = req.voice {
        r = r.with_voice(v);
    }
    if let Some(vu) = req.voice_url {
        r = r.with_voice_url(vu);
    }
    if let Some(l) = req.language {
        r = r.with_language(l);
    }
    #[allow(clippy::cast_possible_truncation)]
    if let Some(s) = req.speed {
        r = r.with_speed(s as f32);
    }
    if let Some(m) = req.model {
        r = r.with_model(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

fn js_to_music_request(req: JsMusicRequest) -> blazen_llm::compute::MusicRequest {
    let mut r = blazen_llm::compute::MusicRequest::new(&req.prompt);
    #[allow(clippy::cast_possible_truncation)]
    if let Some(d) = req.duration_seconds {
        r = r.with_duration(d as f32);
    }
    if let Some(m) = req.model {
        r = r.with_model(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

fn js_to_three_d_request(req: JsThreeDRequest) -> blazen_llm::compute::ThreeDRequest {
    let mut r = if let Some(ref prompt) = req.prompt {
        blazen_llm::compute::ThreeDRequest::new(prompt)
    } else {
        blazen_llm::compute::ThreeDRequest::new("")
    };
    if let Some(image_url) = req.image_url {
        r.image_url = Some(image_url);
    }
    if let Some(format) = req.format {
        r.format = Some(format);
    }
    if let Some(m) = req.model {
        r.model = Some(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

fn js_to_transcription_request(
    req: JsTranscriptionRequest,
) -> blazen_llm::compute::TranscriptionRequest {
    let mut r = blazen_llm::compute::TranscriptionRequest::new(&req.audio_url);
    if let Some(l) = req.language {
        r = r.with_language(l);
    }
    if let Some(d) = req.diarize {
        r = r.with_diarize(d);
    }
    if let Some(m) = req.model {
        r = r.with_model(m);
    }
    if let Some(p) = req.parameters {
        r.parameters = p;
    }
    r
}

// ---------------------------------------------------------------------------
// FalProvider NAPI class
// ---------------------------------------------------------------------------

/// A fal.ai compute platform provider with image, video, audio, transcription,
/// and LLM capabilities.
///
/// ```typescript
/// const fal = FalProvider.create("fal-key-...");
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
    pub fn create(api_key: String, options: Option<JsFalOptions>) -> Self {
        let provider = apply_fal_options(FalProvider::new(api_key), options);
        Self {
            inner: Arc::new(provider),
        }
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
    pub async fn generate_image(&self, request: JsImageRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_image_request(request);
        let result = self
            .inner
            .generate_image(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Upscale an image.
    #[napi(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, request: JsUpscaleRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_upscale_request(request);
        let result = self
            .inner
            .upscale_image(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Upscale an image via the aura-sr model.
    #[napi(js_name = "upscaleImageAura")]
    pub async fn upscale_image_aura(&self, request: JsUpscaleRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_upscale_request(request);
        let result = self
            .inner
            .upscale_image_aura(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Upscale an image via the clarity-upscaler model.
    #[napi(js_name = "upscaleImageClarity")]
    pub async fn upscale_image_clarity(
        &self,
        request: JsUpscaleRequest,
    ) -> Result<serde_json::Value> {
        let rust_req = js_to_upscale_request(request);
        let result = self
            .inner
            .upscale_image_clarity(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Upscale an image via the creative-upscaler model.
    #[napi(js_name = "upscaleImageCreative")]
    pub async fn upscale_image_creative(
        &self,
        request: JsUpscaleRequest,
    ) -> Result<serde_json::Value> {
        let rust_req = js_to_upscale_request(request);
        let result = self
            .inner
            .upscale_image_creative(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Remove the background from an image.
    ///
    /// `imageUrl` is the URL of the source image. `model` optionally
    /// overrides the model id.
    #[napi(js_name = "removeBackground")]
    pub async fn remove_background(
        &self,
        image_url: String,
        model: Option<String>,
    ) -> Result<serde_json::Value> {
        let request = BackgroundRemovalRequest {
            image_url,
            model,
            parameters: serde_json::Value::Null,
        };
        let result = BackgroundRemoval::remove_background(self.inner.as_ref(), request)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    // -----------------------------------------------------------------
    // Compute methods -- 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model from a text prompt or source image.
    #[napi(js_name = "generate3d")]
    pub async fn generate_3d(&self, request: JsThreeDRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_three_d_request(request);
        let result = ThreeDGeneration::generate_3d(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
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
    pub async fn text_to_video(&self, request: JsVideoRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_video_request(request);
        let result = self
            .inner
            .text_to_video(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Generate a video from a source image and prompt.
    #[napi(js_name = "imageToVideo")]
    pub async fn image_to_video(&self, request: JsVideoRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_video_request(request);
        let result = self
            .inner
            .image_to_video(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    // -----------------------------------------------------------------
    // Compute methods -- Audio
    // -----------------------------------------------------------------

    /// Synthesize speech from text.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsSpeechRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_speech_request(request);
        let result = self
            .inner
            .text_to_speech(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Generate music from a prompt.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(&self, request: JsMusicRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_music_request(request);
        let result = self
            .inner
            .generate_music(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Generate sound effects from a prompt.
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(&self, request: JsMusicRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_music_request(request);
        let result = self
            .inner
            .generate_sfx(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    // -----------------------------------------------------------------
    // Compute methods -- Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio to text.
    #[napi]
    pub async fn transcribe(&self, request: JsTranscriptionRequest) -> Result<serde_json::Value> {
        let rust_req = js_to_transcription_request(request);
        let result = self
            .inner
            .transcribe(rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    // -----------------------------------------------------------------
    // Raw compute methods
    // -----------------------------------------------------------------

    /// Run a model synchronously (submit + wait for result).
    #[napi]
    pub async fn run(&self, model: String, input: serde_json::Value) -> Result<serde_json::Value> {
        let request = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self
            .inner
            .run(request)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Submit a job to the queue and return a job handle.
    #[napi]
    pub async fn submit(
        &self,
        model: String,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let request = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let handle = self
            .inner
            .submit(request)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&handle).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Get the status of a submitted job.
    #[napi]
    pub async fn status(&self, job_id: String, model: String) -> Result<serde_json::Value> {
        let handle = blazen_llm::compute::JobHandle {
            id: job_id,
            provider: "fal".to_owned(),
            model,
            submitted_at: Utc::now(),
        };
        let status = self
            .inner
            .status(&handle)
            .await
            .map_err(blazen_error_to_napi)?;
        serde_json::to_value(&status).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Cancel a submitted job.
    #[napi]
    pub async fn cancel(&self, job_id: String, model: String) -> Result<()> {
        let handle = blazen_llm::compute::JobHandle {
            id: job_id,
            provider: "fal".to_owned(),
            model,
            submitted_at: Utc::now(),
        };
        self.inner
            .cancel(&handle)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(())
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
/// const fal = FalProvider.create("fal-key-...");
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
