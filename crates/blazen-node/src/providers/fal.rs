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
    AudioGeneration, ComputeProvider, ComputeRequest, ImageGeneration, Transcription,
    VideoGeneration,
};
use blazen_llm::providers::fal::FalProvider;
use blazen_llm::types::{ChatMessage, CompletionRequest};

use crate::compute::{
    JsImageRequest, JsMusicRequest, JsSpeechRequest, JsTranscriptionRequest, JsUpscaleRequest,
    JsVideoRequest,
};
use crate::error::blazen_error_to_napi;
use crate::types::{JsChatMessage, JsCompletionResponse, build_response};

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
    /// `model` optionally sets the LLM model name used by the `any-llm`
    /// proxy (e.g. `"anthropic/claude-sonnet-4.5"`, `"openai/gpt-4o"`).
    ///
    /// `endpoint` optionally overrides the fal.ai endpoint path
    /// (default: `fal-ai/any-llm`).
    #[napi(factory)]
    pub fn create(api_key: String, model: Option<String>, endpoint: Option<String>) -> Self {
        let mut provider = FalProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_llm_model(m);
        }
        if let Some(e) = endpoint {
            provider = provider.with_endpoint(e);
        }
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
