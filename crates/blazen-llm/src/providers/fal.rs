//! fal.ai compute platform provider.
//!
//! fal.ai is fundamentally different from typical LLM providers -- it is a
//! compute platform with a queue/poll/webhook execution model. It supports
//! 600+ models for various tasks including LLMs (via its `fal-ai/any-llm`
//! proxy), image generation, video, and audio.
//!
//! Key differences:
//! - Auth: `Authorization: Key <FAL_API_KEY>` (note `Key` prefix, not `Bearer`)
//! - Queue mode: submit -> poll status -> get result
//! - Sync mode: submit and wait (timeout risk for long jobs)
//! - Webhook mode: submit with callback URL
//!
//! For LLM specifically, fal.ai proxies through `OpenRouter` via `fal-ai/any-llm`.
//!
//! This module implements all media generation traits:
//! - [`CompletionModel`] for LLM chat completions (via `fal-ai/any-llm`)
//! - [`ComputeProvider`] for generic compute job submission/polling
//! - [`ImageGeneration`] for typed image generation and upscaling
//! - [`VideoGeneration`] for text-to-video and image-to-video
//! - [`AudioGeneration`] for TTS, music, and sound effects
//! - [`Transcription`] for speech-to-text (Whisper)

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use futures_util::Stream;
use futures_util::stream;
use serde::Deserialize;
use tracing::{debug, warn};

use super::openai_format::parse_retry_after;
use crate::compute::{
    AudioGeneration, AudioResult, ComputeProvider, ComputeRequest, ComputeResult, ImageGeneration,
    ImageRequest, ImageResult, JobHandle, JobStatus, MusicRequest, SpeechRequest, Transcription,
    TranscriptionRequest, TranscriptionResult, TranscriptionSegment, UpscaleRequest,
    VideoGeneration, VideoRequest, VideoResult,
};
use crate::error::{BlazenError, ComputeErrorKind};
use crate::http::{HttpClient, HttpRequest};
use crate::media::{GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput, MediaType};
use crate::types::{
    CompletionRequest, CompletionResponse, MessageContent, RequestTiming, StreamChunk, TokenUsage,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FAL_QUEUE_URL: &str = "https://queue.fal.run";
const FAL_SYNC_URL: &str = "https://fal.run";

/// Default poll interval for queue-based execution.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Maximum number of poll iterations before giving up.
const MAX_POLL_ITERATIONS: u32 = 600; // 10 minutes at 1s intervals

/// Default image generation model.
const DEFAULT_IMAGE_MODEL: &str = "fal-ai/flux/schnell";

/// Default upscaling model.
const DEFAULT_UPSCALE_MODEL: &str = "fal-ai/esrgan";

/// Default text-to-video model.
const DEFAULT_TEXT_TO_VIDEO_MODEL: &str = "fal-ai/minimax/video-01";

/// Default image-to-video model.
const DEFAULT_IMAGE_TO_VIDEO_MODEL: &str = "fal-ai/kling-video/v2.1/pro/image-to-video";

/// Default text-to-speech model.
const DEFAULT_TTS_MODEL: &str = "fal-ai/chatterbox/text-to-speech";

/// Default music generation model.
const DEFAULT_MUSIC_MODEL: &str = "fal-ai/stable-audio";

/// Default sound effect generation model.
const DEFAULT_SFX_MODEL: &str = "fal-ai/stable-audio";

/// Default transcription model.
const DEFAULT_TRANSCRIPTION_MODEL: &str = "fal-ai/whisper";

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// How to execute requests on fal.ai.
#[derive(Debug, Clone)]
pub enum FalExecutionMode {
    /// Synchronous -- wait for result (timeout risk for long jobs).
    Sync,
    /// Queue-based -- submit, poll for result.
    Queue {
        /// How often to poll for completion.
        poll_interval: Duration,
    },
    /// Webhook -- submit, receive result at the given URL.
    Webhook {
        /// The URL to receive the result.
        url: String,
    },
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A fal.ai compute platform provider.
///
/// For LLM usage, this provider uses the `fal-ai/any-llm` model which
/// proxies through `OpenRouter` and accepts a simple prompt-based format.
///
/// For compute usage, this provider implements [`ComputeProvider`] with
/// queue-based job submission, status polling, and result retrieval.
///
/// For image generation and upscaling, this provider implements [`ImageGeneration`].
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::fal::FalProvider;
///
/// let provider = FalProvider::new("fal-key-...")
///     .with_endpoint("fal-ai/any-llm");
/// ```
pub struct FalProvider {
    client: Arc<dyn HttpClient>,
    api_key: String,
    endpoint: String,
    /// The underlying LLM model to use when proxying through `fal-ai/any-llm`.
    llm_model: String,
    execution_mode: FalExecutionMode,
    base_queue_url: String,
    base_sync_url: String,
}

impl std::fmt::Debug for FalProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalProvider")
            .field("endpoint", &self.endpoint)
            .field("llm_model", &self.llm_model)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

impl Clone for FalProvider {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            api_key: self.api_key.clone(),
            endpoint: self.endpoint.clone(),
            llm_model: self.llm_model.clone(),
            execution_mode: self.execution_mode.clone(),
            base_queue_url: self.base_queue_url.clone(),
            base_sync_url: self.base_sync_url.clone(),
        }
    }
}

impl FalProvider {
    /// Create a new fal.ai provider with the given API key.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            api_key: api_key.into(),
            endpoint: "fal-ai/any-llm".to_owned(),
            llm_model: "anthropic/claude-sonnet-4.5".to_owned(),
            execution_mode: FalExecutionMode::Queue {
                poll_interval: DEFAULT_POLL_INTERVAL,
            },
            base_queue_url: FAL_QUEUE_URL.to_owned(),
            base_sync_url: FAL_SYNC_URL.to_owned(),
        }
    }

    /// Create a new fal.ai provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            endpoint: "fal-ai/any-llm".to_owned(),
            llm_model: "anthropic/claude-sonnet-4.5".to_owned(),
            execution_mode: FalExecutionMode::Queue {
                poll_interval: DEFAULT_POLL_INTERVAL,
            },
            base_queue_url: FAL_QUEUE_URL.to_owned(),
            base_sync_url: FAL_SYNC_URL.to_owned(),
        }
    }

    /// Override the fal.ai endpoint path (e.g. `fal-ai/any-llm`).
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Deprecated: use [`with_endpoint`](Self::with_endpoint) instead.
    #[deprecated(since = "0.2.0", note = "renamed to `with_endpoint`")]
    #[must_use]
    pub fn with_model(self, model: impl Into<String>) -> Self {
        self.with_endpoint(model)
    }

    /// Override the base queue URL (default: `https://queue.fal.run`).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_queue_url = url.into();
        self
    }

    /// Set the underlying LLM model used by `fal-ai/any-llm`.
    ///
    /// This is the model name passed in the request body (e.g.
    /// `"anthropic/claude-sonnet-4.5"`, `"openai/gpt-4o"`).
    #[must_use]
    pub fn with_llm_model(mut self, model: impl Into<String>) -> Self {
        self.llm_model = model.into();
        self
    }

    /// Set the execution mode.
    #[must_use]
    pub fn with_execution_mode(mut self, mode: FalExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    // -----------------------------------------------------------------------
    // Auth helper
    // -----------------------------------------------------------------------

    /// Apply fal.ai authentication (`Authorization: Key <key>`) to an [`HttpRequest`].
    fn apply_auth(&self, request: HttpRequest) -> HttpRequest {
        request.header("Authorization", format!("Key {}", self.api_key))
    }

    // -----------------------------------------------------------------------
    // LLM body builder
    // -----------------------------------------------------------------------

    /// Build the JSON request body for the `fal-ai/any-llm` endpoint.
    ///
    /// fal-ai/any-llm is text-only. Non-text content (images, files) is
    /// dropped with a warning.
    fn build_llm_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let llm_model = request.model.as_deref().unwrap_or(&self.llm_model);

        // Concatenate all messages into a prompt string.
        // fal-ai/any-llm expects `prompt` and optionally `system_prompt`.
        let mut system_parts: Vec<String> = Vec::new();
        let mut conversation_parts: Vec<String> = Vec::new();

        for msg in &request.messages {
            let text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                other => {
                    // fal-ai/any-llm is text-only; extract what text we can.
                    if !matches!(other, MessageContent::Text(_)) {
                        warn!(
                            "fal.ai provider is text-only; non-text content parts will be dropped"
                        );
                    }
                    other.text_content().unwrap_or_default()
                }
            };
            match msg.role {
                crate::types::Role::System => {
                    system_parts.push(text);
                }
                crate::types::Role::User => {
                    conversation_parts.push(format!("User: {text}"));
                }
                crate::types::Role::Assistant => {
                    conversation_parts.push(format!("Assistant: {text}"));
                }
                crate::types::Role::Tool => {
                    conversation_parts.push(format!("Tool result: {text}"));
                }
            }
        }

        let mut body = serde_json::json!({
            "model": llm_model,
            "prompt": conversation_parts.join("\n\n"),
        });

        if !system_parts.is_empty() {
            body["system_prompt"] = serde_json::Value::String(system_parts.join("\n\n"));
        }

        // Pass through optional LLM parameters.
        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(ref response_format) = request.response_format {
            body["response_format"] = response_format.clone();
        }

        body
    }

    /// Resolve the fal.ai endpoint path to use.
    fn resolve_endpoint(&self) -> &str {
        &self.endpoint
    }

    // -----------------------------------------------------------------------
    // Shared HTTP helpers
    // -----------------------------------------------------------------------

    /// Map an HTTP error response to the appropriate `BlazenError`.
    fn map_http_error(status: u16, body: &str, retry_after_ms: Option<u64>) -> BlazenError {
        match status {
            401 => BlazenError::auth("authentication failed"),
            429 => BlazenError::RateLimit { retry_after_ms },
            _ => BlazenError::request(format!("HTTP {status}: {body}")),
        }
    }

    // -----------------------------------------------------------------------
    // Sync execution (for CompletionModel)
    // -----------------------------------------------------------------------

    /// Execute synchronously: POST to fal.run and wait for the response.
    async fn execute_sync(
        &self,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, BlazenError> {
        let model = self.resolve_endpoint();
        let url = format!("{}/{model}", self.base_sync_url);

        let request = self.apply_auth(HttpRequest::post(&url).json_body(body)?);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let retry_after_ms = parse_retry_after(&response.headers);
            let error_body = response.text();
            return Err(Self::map_http_error(
                response.status,
                &error_body,
                retry_after_ms,
            ));
        }

        response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Webhook execution (for CompletionModel)
    // -----------------------------------------------------------------------

    /// Execute via webhook: submit with webhook URL.
    async fn execute_webhook(
        &self,
        body: &serde_json::Value,
        webhook_url: &str,
    ) -> Result<serde_json::Value, BlazenError> {
        let model = self.resolve_endpoint();
        let submit_url = format!("{}/{model}?fal_webhook={webhook_url}", self.base_queue_url);

        let request = self.apply_auth(HttpRequest::post(&submit_url).json_body(body)?);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "HTTP {}: {error_body}",
                response.status
            )));
        }

        // Webhook mode returns the queue submission response. The actual
        // result will be delivered to the webhook URL.
        response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Shared queue polling logic
    // -----------------------------------------------------------------------

    /// Submit a request to the fal.ai queue and return the queue response.
    async fn queue_submit(
        &self,
        model: &str,
        body: &serde_json::Value,
        webhook: Option<&str>,
    ) -> Result<FalQueueSubmitResponse, BlazenError> {
        let mut submit_url = format!("{}/{model}", self.base_queue_url);
        if let Some(wh) = webhook {
            submit_url = format!("{submit_url}?fal_webhook={wh}");
        }

        let request = self.apply_auth(HttpRequest::post(&submit_url).json_body(body)?);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let retry_after_ms = parse_retry_after(&response.headers);
            let error_body = response.text();
            return Err(Self::map_http_error(
                response.status,
                &error_body,
                retry_after_ms,
            ));
        }

        response
            .json::<FalQueueSubmitResponse>()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Poll the fal.ai queue status endpoint.
    async fn queue_poll_status(
        &self,
        model: &str,
        request_id: &str,
    ) -> Result<FalStatusResponse, BlazenError> {
        let status_url = format!(
            "{}/{model}/requests/{request_id}/status",
            self.base_queue_url
        );

        let request = self.apply_auth(HttpRequest::get(&status_url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "status poll failed: {error_body}"
            )));
        }

        response
            .json::<FalStatusResponse>()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Fetch the result of a completed queue job.
    async fn queue_get_result(
        &self,
        model: &str,
        request_id: &str,
    ) -> Result<serde_json::Value, BlazenError> {
        let result_url = format!("{}/{model}/requests/{request_id}", self.base_queue_url);

        let request = self.apply_auth(HttpRequest::get(&result_url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "result fetch failed: {error_body}"
            )));
        }

        response
            .json()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Poll until a queue job completes and return the result JSON plus timing.
    ///
    /// This is the shared polling logic used by both [`ComputeProvider::result`]
    /// and [`CompletionModel::complete`] (queue mode).
    ///
    /// When `status_url` and `response_url` are provided (from the queue submit
    /// response), they are used directly instead of constructing URLs from the
    /// model and request ID. This avoids 405 errors with multi-segment model
    /// IDs where manual URL construction produces incorrect paths.
    async fn poll_until_complete(
        &self,
        model: &str,
        request_id: &str,
        poll_interval: Duration,
        status_url: Option<&str>,
        response_url: Option<&str>,
    ) -> Result<(serde_json::Value, serde_json::Value, RequestTiming), BlazenError> {
        let start = Instant::now();
        let mut in_progress_at: Option<Instant> = None;

        for _ in 0..MAX_POLL_ITERATIONS {
            crate::sleep::sleep(poll_interval).await;

            let status_body = if let Some(url) = status_url {
                self.get_json_from_url(url).await?
            } else {
                self.queue_poll_status(model, request_id).await?
            };

            match status_body.status.as_str() {
                "COMPLETED" => {
                    // Check for error in COMPLETED status.
                    if let Some(ref error) = status_body.error {
                        return Err(BlazenError::Compute(ComputeErrorKind::JobFailed {
                            message: error.clone(),
                            error_type: None,
                            retryable: false,
                        }));
                    }

                    // Build timing from metrics.
                    let inference_time =
                        status_body.metrics.as_ref().and_then(|m| m.inference_time);
                    let timing = build_timing(start, in_progress_at, inference_time);

                    // Serialize status for metadata before moving on.
                    let status_json =
                        serde_json::to_value(&status_body).unwrap_or(serde_json::Value::Null);

                    // Fetch the result using the server-provided URL if available.
                    let result = if let Some(url) = response_url {
                        self.get_json_value_from_url(url).await?
                    } else {
                        self.queue_get_result(model, request_id).await?
                    };

                    return Ok((result, status_json, timing));
                }
                "IN_PROGRESS" => {
                    if in_progress_at.is_none() {
                        in_progress_at = Some(Instant::now());
                    }
                }
                // IN_QUEUE -- keep polling.
                _ => {}
            }
        }

        Err(BlazenError::Timeout {
            elapsed_ms: millis_u64(start.elapsed()),
        })
    }

    /// GET a URL and deserialize the response as [`FalStatusResponse`].
    async fn get_json_from_url(&self, url: &str) -> Result<FalStatusResponse, BlazenError> {
        let request = self.apply_auth(HttpRequest::get(url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "status poll failed: {error_body}"
            )));
        }

        response
            .json::<FalStatusResponse>()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// GET a URL and deserialize the response as a generic JSON value.
    async fn get_json_value_from_url(&self, url: &str) -> Result<serde_json::Value, BlazenError> {
        let request = self.apply_auth(HttpRequest::get(url));
        let response = self.client.send(request).await?;

        if !response.is_success() {
            let error_body = response.text();
            return Err(BlazenError::request(format!(
                "result fetch failed: {error_body}"
            )));
        }

        response
            .json()
            .map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Execute via queue: submit, poll, return result. Used by `CompletionModel`.
    async fn execute_queue_llm(
        &self,
        body: &serde_json::Value,
        poll_interval: Duration,
    ) -> Result<(serde_json::Value, RequestTiming), BlazenError> {
        let model = self.resolve_endpoint();

        let submit_response = self.queue_submit(model, body, None).await?;

        let request_id = &submit_response.request_id;
        debug!(request_id = %request_id, "fal.ai LLM job submitted to queue");

        let (result, _status, timing) = self
            .poll_until_complete(
                model,
                request_id,
                poll_interval,
                submit_response.status_url.as_deref(),
                submit_response.response_url.as_deref(),
            )
            .await?;

        Ok((result, timing))
    }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/// Safely convert a `Duration` to milliseconds as `u64`, saturating at `u64::MAX`.
fn millis_u64(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Build a [`RequestTiming`] from measured instants and fal.ai metrics.
fn build_timing(
    start: Instant,
    in_progress_at: Option<Instant>,
    inference_time_secs: Option<f64>,
) -> RequestTiming {
    let total_ms = Some(millis_u64(start.elapsed()));

    let queue_ms = in_progress_at.map(|t| millis_u64(t.duration_since(start)));

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    let execution_ms = inference_time_secs
        .map(|s| {
            let ms = (s * 1000.0).max(0.0);
            if ms > u64::MAX as f64 {
                u64::MAX
            } else {
                ms as u64
            }
        })
        .or_else(|| {
            // Fallback: if we know when IN_PROGRESS started, compute
            // execution as total minus queue time.
            match (total_ms, queue_ms) {
                (Some(total), Some(queue)) => Some(total.saturating_sub(queue)),
                _ => None,
            }
        });

    RequestTiming {
        queue_ms,
        execution_ms,
        total_ms,
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Response from the fal.ai queue submit endpoint.
#[derive(Debug, Deserialize)]
struct FalQueueSubmitResponse {
    request_id: String,
    #[serde(default)]
    response_url: Option<String>,
    #[serde(default)]
    status_url: Option<String>,
    #[serde(default)]
    #[allow(dead_code)] // Cancel uses its own URL construction in cancel().
    cancel_url: Option<String>,
}

/// Response from the fal.ai queue status endpoint.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct FalStatusResponse {
    status: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    metrics: Option<FalMetrics>,
    #[serde(default)]
    #[allow(dead_code)]
    queue_position: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    response_url: Option<String>,
}

/// Metrics returned by fal.ai in COMPLETED status.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct FalMetrics {
    /// Inference time in seconds.
    inference_time: Option<f64>,
}

/// Response from `fal-ai/any-llm`.
#[derive(Debug, Deserialize)]
struct FalLlmResponse {
    output: Option<String>,
    error: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    partial: Option<bool>,
}

/// A single image from fal.ai image generation output.
#[derive(Debug, Deserialize)]
struct FalImage {
    url: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    content_type: Option<String>,
}

/// Image generation output from fal.ai.
#[derive(Debug, Deserialize)]
struct FalImageOutput {
    #[serde(default)]
    images: Vec<FalImage>,
}

/// ESRGAN upscale output from fal.ai (single image, not array).
#[derive(Debug, Deserialize)]
struct FalUpscaleOutput {
    image: FalImage,
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for FalProvider {
    fn model_id(&self) -> &str {
        &self.endpoint
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model_id = request.model.as_deref().unwrap_or(&self.llm_model);
        let span = tracing::info_span!(
            "llm.complete",
            provider = "fal",
            model = %model_id,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let body = self.build_llm_body(&request);
        debug!(endpoint = %self.endpoint, "fal.ai completion request");

        let (result, timing) = match &self.execution_mode {
            FalExecutionMode::Sync => {
                let result = self.execute_sync(&body).await?;
                let elapsed = millis_u64(start.elapsed());
                let timing = RequestTiming {
                    queue_ms: None,
                    execution_ms: Some(elapsed),
                    total_ms: Some(elapsed),
                };
                (result, timing)
            }
            FalExecutionMode::Queue { poll_interval } => {
                self.execute_queue_llm(&body, *poll_interval).await?
            }
            FalExecutionMode::Webhook { url } => {
                let result = self.execute_webhook(&body, url).await?;
                let timing = RequestTiming {
                    queue_ms: None,
                    execution_ms: None,
                    total_ms: Some(millis_u64(start.elapsed())),
                };
                (result, timing)
            }
        };

        // Parse the fal.ai response.
        let fal_response: FalLlmResponse = serde_json::from_value(result)
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        if let Some(error) = fal_response.error {
            return Err(BlazenError::request(format!("fal.ai model error: {error}")));
        }

        span.record(
            "duration_ms",
            timing
                .total_ms
                .unwrap_or_else(|| millis_u64(start.elapsed())),
        );
        span.record("finish_reason", "stop");

        // fal.ai/any-llm does not return token usage, so cost will be None.
        let usage: Option<TokenUsage> = None;
        let cost = usage
            .as_ref()
            .and_then(|u| crate::pricing::compute_cost(&self.endpoint, u));

        Ok(CompletionResponse {
            content: fal_response.output,
            tool_calls: Vec::new(), // fal.ai/any-llm doesn't support tool calling.
            usage,
            model: self.endpoint.clone(),
            finish_reason: Some("stop".to_owned()),
            cost,
            timing: Some(timing),
            images: vec![],
            audio: vec![],
            videos: vec![],
            metadata: serde_json::Value::Null,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let model_id = request.model.as_deref().unwrap_or(&self.llm_model);
        let span = tracing::info_span!(
            "llm.stream",
            provider = "fal",
            model = %model_id,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        // fal.ai does not natively support SSE streaming for LLM.
        // We simulate streaming by executing the request and then emitting
        // the complete result as a single chunk.
        let response = self.complete(request).await?;

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        span.record("chunk_count", 1u64);

        let chunks: Vec<Result<StreamChunk, BlazenError>> = vec![Ok(StreamChunk {
            delta: response.content,
            tool_calls: Vec::new(),
            finish_reason: Some("stop".to_owned()),
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::ModelRegistry for FalProvider {
    async fn list_models(&self) -> Result<Vec<crate::traits::ModelInfo>, BlazenError> {
        // fal.ai is a compute platform without a model listing API.
        Ok(Vec::new())
    }

    async fn get_model(
        &self,
        _model_id: &str,
    ) -> Result<Option<crate::traits::ModelInfo>, BlazenError> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for FalProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        "fal"
    }

    async fn submit(&self, request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        let submit_response = self
            .queue_submit(&request.model, &request.input, request.webhook.as_deref())
            .await?;

        debug!(
            request_id = %submit_response.request_id,
            model = %request.model,
            "fal.ai compute job submitted"
        );

        Ok(JobHandle {
            id: submit_response.request_id,
            provider: "fal".to_owned(),
            model: request.model,
            submitted_at: Utc::now(),
        })
    }

    async fn status(&self, job: &JobHandle) -> Result<JobStatus, BlazenError> {
        let status_body = self.queue_poll_status(&job.model, &job.id).await?;

        match status_body.status.as_str() {
            "IN_QUEUE" => Ok(JobStatus::Queued),
            "IN_PROGRESS" => Ok(JobStatus::Running),
            "COMPLETED" => {
                if let Some(error) = status_body.error {
                    Ok(JobStatus::Failed { error })
                } else {
                    Ok(JobStatus::Completed)
                }
            }
            other => {
                // Defensive: treat unknown statuses as queued.
                warn!(status = %other, "unknown fal.ai queue status, treating as Queued");
                Ok(JobStatus::Queued)
            }
        }
    }

    async fn result(&self, job: JobHandle) -> Result<ComputeResult, BlazenError> {
        let poll_interval = match &self.execution_mode {
            FalExecutionMode::Queue { poll_interval } => *poll_interval,
            _ => DEFAULT_POLL_INTERVAL,
        };

        // External callers use submit() -> result() without access to the
        // server-provided URLs, so we fall back to manual URL construction.
        let (output, status_json, timing) = self
            .poll_until_complete(&job.model, &job.id, poll_interval, None, None)
            .await?;

        Ok(ComputeResult {
            job: Some(job),
            output,
            timing,
            cost: None, // fal.ai does not return per-request cost in API responses.
            metadata: status_json,
        })
    }

    /// Submit a job and wait for the result, using server-provided URLs for
    /// queue polling and result retrieval.
    ///
    /// This overrides the default `run()` to avoid the 405 errors that occur
    /// when manually constructing URLs for models with multi-segment IDs
    /// (e.g. `fal-ai/kling-video/v2.1/pro/image-to-video`).
    async fn run(&self, request: ComputeRequest) -> Result<ComputeResult, BlazenError> {
        let poll_interval = match &self.execution_mode {
            FalExecutionMode::Queue { poll_interval } => *poll_interval,
            _ => DEFAULT_POLL_INTERVAL,
        };

        let submit_response = self
            .queue_submit(&request.model, &request.input, request.webhook.as_deref())
            .await?;

        debug!(
            request_id = %submit_response.request_id,
            model = %request.model,
            "fal.ai compute job submitted (via run)"
        );

        let job = JobHandle {
            id: submit_response.request_id.clone(),
            provider: "fal".to_owned(),
            model: request.model,
            submitted_at: Utc::now(),
        };

        let (output, status_json, timing) = self
            .poll_until_complete(
                &job.model,
                &job.id,
                poll_interval,
                submit_response.status_url.as_deref(),
                submit_response.response_url.as_deref(),
            )
            .await?;

        Ok(ComputeResult {
            job: Some(job),
            output,
            timing,
            cost: None,
            metadata: status_json,
        })
    }

    async fn cancel(&self, job: &JobHandle) -> Result<(), BlazenError> {
        let cancel_url = format!(
            "{}/{}/requests/{}/cancel",
            self.base_queue_url, job.model, job.id
        );

        let request = self.apply_auth(HttpRequest::put(&cancel_url));
        let response = self.client.send(request).await?;

        let status = response.status;
        if (200..300).contains(&status) || status == 202 {
            debug!(request_id = %job.id, "fal.ai job cancellation requested");
            return Ok(());
        }

        // 400 = ALREADY_COMPLETED, which is fine.
        if status == 400 {
            debug!(request_id = %job.id, "fal.ai job already completed, cancel is a no-op");
            return Ok(());
        }

        let error_body = response.text();
        Err(BlazenError::request(format!(
            "cancel failed (HTTP {status}): {error_body}"
        )))
    }
}

// ---------------------------------------------------------------------------
// ImageGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ImageGeneration for FalProvider {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_MODEL)
            .to_owned();

        // Build the input JSON.
        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        // Image size: pass as object if both dimensions are set.
        if let (Some(w), Some(h)) = (request.width, request.height) {
            input["image_size"] = serde_json::json!({
                "width": w,
                "height": h,
            });
        }

        // Number of images.
        if let Some(n) = request.num_images {
            input["num_images"] = serde_json::json!(n);
        }

        // Negative prompt.
        if let Some(ref neg) = request.negative_prompt {
            input["negative_prompt"] = serde_json::json!(neg);
        }

        // Merge extra parameters from the request.
        if let serde_json::Value::Object(params) = &request.parameters {
            for (k, v) in params {
                input[k] = v.clone();
            }
        }

        // Submit as a compute job and wait for the result.
        let compute_request = ComputeRequest {
            model: model.clone(),
            input,
            webhook: None,
        };
        let result = self.run(compute_request).await?;

        // Parse the image output.
        let image_output: FalImageOutput =
            serde_json::from_value(result.output.clone()).map_err(|e| {
                BlazenError::Serialization(format!("failed to parse image output: {e}"))
            })?;

        let images = image_output
            .images
            .into_iter()
            .map(|img| {
                let content_type = img.content_type.unwrap_or_else(|| "image/jpeg".to_owned());
                GeneratedImage {
                    media: MediaOutput {
                        url: img.url,
                        base64: None,
                        raw_content: None,
                        media_type: MediaType::from_mime(&content_type),
                        file_size: None,
                        metadata: serde_json::Value::Null,
                    },
                    width: img.width,
                    height: img.height,
                }
            })
            .collect();

        Ok(ImageResult {
            images,
            timing: result.timing,
            cost: result.cost,
            metadata: serde_json::Value::Null,
        })
    }

    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_UPSCALE_MODEL)
            .to_owned();

        // Build the input JSON.
        let mut input = serde_json::json!({
            "image_url": request.image_url,
            "scale": request.scale,
        });

        // Merge extra parameters.
        if let serde_json::Value::Object(params) = &request.parameters {
            for (k, v) in params {
                input[k] = v.clone();
            }
        }

        // Submit as a compute job and wait for the result.
        let compute_request = ComputeRequest {
            model: model.clone(),
            input,
            webhook: None,
        };
        let result = self.run(compute_request).await?;

        // ESRGAN returns a single image object, not an array.
        let upscale_output: FalUpscaleOutput = serde_json::from_value(result.output.clone())
            .map_err(|e| {
                BlazenError::Serialization(format!("failed to parse upscale output: {e}"))
            })?;

        let content_type = upscale_output
            .image
            .content_type
            .unwrap_or_else(|| "image/png".to_owned());
        let image = GeneratedImage {
            media: MediaOutput {
                url: upscale_output.image.url,
                base64: None,
                raw_content: None,
                media_type: MediaType::from_mime(&content_type),
                file_size: None,
                metadata: serde_json::Value::Null,
            },
            width: upscale_output.image.width,
            height: upscale_output.image.height,
        };

        Ok(ImageResult {
            images: vec![image],
            timing: result.timing,
            cost: result.cost,
            metadata: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// Media parsing helpers
// ---------------------------------------------------------------------------

/// Parse a video from fal.ai response output.
///
/// fal.ai video models return `{ "video": { "url": "...", "content_type": "...", ... } }`.
fn parse_fal_video(output: &serde_json::Value) -> Result<GeneratedVideo, BlazenError> {
    let video_obj = output
        .get("video")
        .ok_or_else(|| BlazenError::Serialization("missing 'video' field in response".into()))?;

    let url = video_obj
        .get("url")
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let content_type = video_obj
        .get("content_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("video/mp4");
    let file_size = video_obj
        .get("file_size")
        .and_then(serde_json::Value::as_u64);

    Ok(GeneratedVideo {
        media: MediaOutput {
            url,
            base64: None,
            raw_content: None,
            media_type: MediaType::from_mime(content_type),
            file_size,
            metadata: video_obj.clone(),
        },
        width: None,
        height: None,
        duration_seconds: None,
        fps: None,
    })
}

/// Parse audio from fal.ai response output.
///
/// fal.ai audio models may return either:
/// - `{ "audio_url": { "url": "...", ... } }` (object with url field)
/// - `{ "audio_url": "https://..." }` (direct URL string)
/// - `{ "audio": { "url": "...", ... } }` (nested object)
#[allow(clippy::too_many_lines)]
fn parse_fal_audio(output: &serde_json::Value) -> Result<GeneratedAudio, BlazenError> {
    // Try `audio_url` as an object first (e.g. chatterbox returns this).
    if let Some(audio_obj) = output.get("audio_url") {
        if let Some(obj) = audio_obj.as_object() {
            let url = obj
                .get("url")
                .and_then(serde_json::Value::as_str)
                .map(String::from);
            let content_type = obj
                .get("content_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("audio/wav");
            let file_size = obj.get("file_size").and_then(serde_json::Value::as_u64);

            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url,
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::from_mime(content_type),
                    file_size,
                    metadata: audio_obj.clone(),
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }

        // `audio_url` as a plain string.
        if let Some(url_str) = audio_obj.as_str() {
            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url: Some(url_str.to_owned()),
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::Wav,
                    file_size: None,
                    metadata: serde_json::Value::Null,
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }
    }

    // Try `audio` as a nested object.
    if let Some(audio_obj) = output.get("audio") {
        let url = audio_obj
            .get("url")
            .and_then(serde_json::Value::as_str)
            .map(String::from);
        let content_type = audio_obj
            .get("content_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("audio/wav");
        let file_size = audio_obj
            .get("file_size")
            .and_then(serde_json::Value::as_u64);

        return Ok(GeneratedAudio {
            media: MediaOutput {
                url,
                base64: None,
                raw_content: None,
                media_type: MediaType::from_mime(content_type),
                file_size,
                metadata: audio_obj.clone(),
            },
            duration_seconds: None,
            sample_rate: None,
            channels: None,
        });
    }

    // Try `audio_file` as an object (e.g. stable-audio returns this).
    if let Some(audio_obj) = output.get("audio_file") {
        if let Some(obj) = audio_obj.as_object() {
            let url = obj
                .get("url")
                .and_then(serde_json::Value::as_str)
                .map(String::from);
            let content_type = obj
                .get("content_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("audio/wav");
            let file_size = obj.get("file_size").and_then(serde_json::Value::as_u64);

            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url,
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::from_mime(content_type),
                    file_size,
                    metadata: audio_obj.clone(),
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }

        if let Some(url_str) = audio_obj.as_str() {
            return Ok(GeneratedAudio {
                media: MediaOutput {
                    url: Some(url_str.to_owned()),
                    base64: None,
                    raw_content: None,
                    media_type: MediaType::Wav,
                    file_size: None,
                    metadata: serde_json::Value::Null,
                },
                duration_seconds: None,
                sample_rate: None,
                channels: None,
            });
        }
    }

    Err(BlazenError::Serialization(
        "missing 'audio_url', 'audio', or 'audio_file' field in response".into(),
    ))
}

/// Merge extra parameters from a `serde_json::Value` into an input object.
fn merge_parameters(input: &mut serde_json::Value, parameters: &serde_json::Value) {
    if let Some(params) = parameters.as_object() {
        for (k, v) in params {
            input[k] = v.clone();
        }
    }
}

// ---------------------------------------------------------------------------
// VideoGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl VideoGeneration for FalProvider {
    async fn text_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_TEXT_TO_VIDEO_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        if let Some(dur) = request.duration_seconds {
            // Kling/MiniMax expect duration as a string (e.g. "5").
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let secs = dur as u32;
            input["duration"] = serde_json::json!(secs.to_string());
        }
        if let Some(ref np) = request.negative_prompt {
            input["negative_prompt"] = serde_json::json!(np);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let video = parse_fal_video(&result.output)?;

        Ok(VideoResult {
            videos: vec![video],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }

    async fn image_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_TO_VIDEO_MODEL)
            .to_owned();

        let image_url = request
            .image_url
            .as_deref()
            .ok_or_else(|| BlazenError::Validation {
                field: Some("image_url".into()),
                message: "image_url is required for image-to-video".into(),
            })?;

        let mut input = serde_json::json!({
            "prompt": request.prompt,
            "image_url": image_url,
        });

        if let Some(dur) = request.duration_seconds {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let secs = dur as u32;
            input["duration"] = serde_json::json!(secs.to_string());
        }
        if let Some(ref np) = request.negative_prompt {
            input["negative_prompt"] = serde_json::json!(np);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let video = parse_fal_video(&result.output)?;

        Ok(VideoResult {
            videos: vec![video],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioGeneration for FalProvider {
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_TTS_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "text": request.text,
        });

        if let Some(ref voice) = request.voice {
            input["voice"] = serde_json::json!(voice);
        }
        if let Some(ref voice_url) = request.voice_url {
            input["voice_url"] = serde_json::json!(voice_url);
        }
        if let Some(ref language) = request.language {
            input["language"] = serde_json::json!(language);
        }
        if let Some(speed) = request.speed {
            input["speed"] = serde_json::json!(speed);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let audio = parse_fal_audio(&result.output)?;

        Ok(AudioResult {
            audio: vec![audio],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }

    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_MUSIC_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        if let Some(dur) = request.duration_seconds {
            input["duration"] = serde_json::json!(dur);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let audio = parse_fal_audio(&result.output)?;

        Ok(AudioResult {
            audio: vec![audio],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_SFX_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "prompt": request.prompt,
        });

        if let Some(dur) = request.duration_seconds {
            input["duration"] = serde_json::json!(dur);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;
        let audio = parse_fal_audio(&result.output)?;

        Ok(AudioResult {
            audio: vec![audio],
            timing: result.timing,
            cost: result.cost,
            metadata: result.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Transcription implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Transcription for FalProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        let model = request
            .model
            .as_deref()
            .unwrap_or(DEFAULT_TRANSCRIPTION_MODEL)
            .to_owned();

        let mut input = serde_json::json!({
            "audio_url": request.audio_url,
        });

        if let Some(ref lang) = request.language {
            input["language"] = serde_json::json!(lang);
        }
        if request.diarize {
            input["diarize"] = serde_json::json!(true);
        }
        merge_parameters(&mut input, &request.parameters);

        let compute_req = ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result = self.run(compute_req).await?;

        // Parse Whisper response:
        // { "text": "...", "chunks": [...], "inferred_languages": ["en"], ... }
        let text = result
            .output
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();

        let language = result
            .output
            .get("inferred_languages")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        let segments = result
            .output
            .get("chunks")
            .and_then(|v| v.as_array())
            .map(|chunks| {
                chunks
                    .iter()
                    .filter_map(|chunk| {
                        let seg_text = chunk.get("text")?.as_str()?.to_owned();
                        let timestamps = chunk.get("timestamp")?.as_array()?;
                        let start = timestamps.first()?.as_f64()?;
                        let end = timestamps.get(1)?.as_f64()?;
                        let speaker = chunk
                            .get("speaker")
                            .and_then(serde_json::Value::as_str)
                            .map(String::from);
                        Some(TranscriptionSegment {
                            text: seg_text,
                            start,
                            end,
                            speaker,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(TranscriptionResult {
            text,
            segments,
            language,
            timing: result.timing,
            cost: result.cost,
            metadata: result.output,
        })
    }
}

// ---------------------------------------------------------------------------
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl crate::traits::ProviderInfo for FalProvider {
    fn provider_name(&self) -> &'static str {
        "fal"
    }

    fn base_url(&self) -> &str {
        &self.base_queue_url
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities {
            streaming: false,
            tool_calling: false,
            structured_output: false,
            vision: false,
            model_listing: false,
            embeddings: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    // -----------------------------------------------------------------------
    // Config / builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_config() {
        let provider = FalProvider::new("fal-test");
        assert_eq!(provider.endpoint, "fal-ai/any-llm");
        assert_eq!(provider.llm_model, "anthropic/claude-sonnet-4.5");
        assert!(matches!(
            provider.execution_mode,
            FalExecutionMode::Queue { .. }
        ));
    }

    #[test]
    fn with_endpoint_override() {
        let provider = FalProvider::new("fal-test").with_endpoint("fal-ai/fast-sdxl");
        assert_eq!(provider.endpoint, "fal-ai/fast-sdxl");
    }

    #[test]
    fn with_base_url_override() {
        let provider = FalProvider::new("fal-test").with_base_url("https://custom.fal.run");
        assert_eq!(provider.base_queue_url, "https://custom.fal.run");
    }

    #[allow(deprecated)]
    #[test]
    fn with_model_deprecated_alias() {
        let provider = FalProvider::new("fal-test").with_model("fal-ai/fast-sdxl");
        assert_eq!(provider.endpoint, "fal-ai/fast-sdxl");
    }

    #[test]
    fn with_llm_model_override() {
        let provider = FalProvider::new("fal-test").with_llm_model("openai/gpt-4o");
        assert_eq!(provider.llm_model, "openai/gpt-4o");
    }

    #[test]
    fn with_sync_execution() {
        let provider = FalProvider::new("fal-test").with_execution_mode(FalExecutionMode::Sync);
        assert!(matches!(provider.execution_mode, FalExecutionMode::Sync));
    }

    // -----------------------------------------------------------------------
    // LLM body builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_body_basic() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello world")]);

        let body = provider.build_llm_body(&request);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.5");
        assert!(body["prompt"].as_str().unwrap().contains("Hello world"));
    }

    #[test]
    fn build_body_with_system() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![
            ChatMessage::system("Be helpful"),
            ChatMessage::user("Hello"),
        ]);

        let body = provider.build_llm_body(&request);
        assert_eq!(body["system_prompt"], "Be helpful");
        assert!(body["prompt"].as_str().unwrap().contains("Hello"));
    }

    #[test]
    fn build_body_model_override() {
        let provider = FalProvider::new("fal-test");
        let request =
            CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_model("openai/gpt-4o");

        let body = provider.build_llm_body(&request);
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    fn build_body_with_temperature() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_temperature(0.7);

        let body = provider.build_llm_body(&request);
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature was {temp}");
    }

    #[test]
    fn build_body_with_max_tokens() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_max_tokens(1024);

        let body = provider.build_llm_body(&request);
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn build_body_with_top_p() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_top_p(0.9);

        let body = provider.build_llm_body(&request);
        assert_eq!(body["top_p"], serde_json::json!(0.9_f32));
    }

    #[test]
    fn build_body_with_response_format() {
        let provider = FalProvider::new("fal-test");
        let schema = serde_json::json!({"type": "object"});
        let request = CompletionRequest::new(vec![ChatMessage::user("Hi")])
            .with_response_format(schema.clone());

        let body = provider.build_llm_body(&request);
        assert_eq!(body["response_format"], schema);
    }

    #[test]
    fn test_text_backward_compat() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_llm_body(&request);
        assert!(body["prompt"].as_str().unwrap().contains("Hello"));
    }

    #[test]
    fn test_build_body_image_url_drops_image() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "Describe this",
            "https://example.com/cat.jpg",
            None,
        )]);

        let body = provider.build_llm_body(&request);
        // Only the text part should be preserved.
        let prompt = body["prompt"].as_str().unwrap();
        assert!(prompt.contains("Describe this"));
        assert!(!prompt.contains("cat.jpg"));
    }

    #[test]
    fn test_build_body_base64_image_drops_image() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_base64(
            "What is this",
            "abc123",
            "image/png",
        )]);

        let body = provider.build_llm_body(&request);
        let prompt = body["prompt"].as_str().unwrap();
        assert!(prompt.contains("What is this"));
        assert!(!prompt.contains("abc123"));
    }

    #[test]
    fn test_build_body_multipart_text_only() {
        use crate::types::{ContentPart, ImageContent, ImageSource};

        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user_parts(vec![
            ContentPart::Text {
                text: "First".into(),
            },
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/a.png".into(),
                },
                media_type: None,
            }),
            ContentPart::Text {
                text: "Second".into(),
            },
        ])]);

        let body = provider.build_llm_body(&request);
        let prompt = body["prompt"].as_str().unwrap();
        // Both text parts should be concatenated.
        assert!(prompt.contains("First"));
        assert!(prompt.contains("Second"));
    }

    // -----------------------------------------------------------------------
    // Wire type parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_fal_llm_response() {
        let json = r#"{"output":"Hello! How can I help you?"}"#;
        let response: FalLlmResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            response.output.as_deref(),
            Some("Hello! How can I help you?")
        );
        assert!(response.error.is_none());
    }

    #[test]
    fn parse_fal_error_response() {
        let json = r#"{"output":null,"error":"Model not found"}"#;
        let response: FalLlmResponse = serde_json::from_str(json).unwrap();
        assert!(response.output.is_none());
        assert_eq!(response.error.as_deref(), Some("Model not found"));
    }

    #[test]
    fn parse_queue_submit_response() {
        let json = r#"{
            "request_id": "abc-123-def",
            "response_url": "https://queue.fal.run/model/requests/abc-123-def/response",
            "status_url": "https://queue.fal.run/model/requests/abc-123-def/status",
            "cancel_url": "https://queue.fal.run/model/requests/abc-123-def/cancel"
        }"#;
        let response: FalQueueSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.request_id, "abc-123-def");
    }

    #[test]
    fn parse_queue_submit_response_minimal() {
        // Backwards compat: only request_id is required.
        let json = r#"{"request_id":"abc-123-def"}"#;
        let response: FalQueueSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.request_id, "abc-123-def");
    }

    #[test]
    fn parse_status_completed() {
        let json = r#"{"status":"COMPLETED"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "COMPLETED");
        assert!(response.error.is_none());
    }

    #[test]
    fn parse_status_completed_with_metrics() {
        let json = r#"{"status":"COMPLETED","metrics":{"inference_time":3.42}}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "COMPLETED");
        let metrics = response.metrics.unwrap();
        assert!((metrics.inference_time.unwrap() - 3.42).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_status_completed_with_error() {
        let json = r#"{"status":"COMPLETED","error":"Out of memory"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "COMPLETED");
        assert_eq!(response.error.as_deref(), Some("Out of memory"));
    }

    #[test]
    fn parse_status_in_queue() {
        let json = r#"{"status":"IN_QUEUE","queue_position":2}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "IN_QUEUE");
        assert_eq!(response.queue_position, Some(2));
    }

    #[test]
    fn parse_status_in_progress() {
        let json = r#"{"status":"IN_PROGRESS"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "IN_PROGRESS");
    }

    // -----------------------------------------------------------------------
    // Image wire type tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_image_output() {
        let json = r#"{
            "images": [
                {
                    "url": "https://v3.fal.media/files/rabbit/abc123.png",
                    "width": 1024,
                    "height": 768,
                    "content_type": "image/jpeg"
                }
            ]
        }"#;
        let output: FalImageOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.images[0].width, Some(1024));
        assert_eq!(output.images[0].height, Some(768));
    }

    #[test]
    fn parse_upscale_output() {
        let json = r#"{
            "image": {
                "url": "https://v3.fal.media/files/out/upscaled.png",
                "width": 2048,
                "height": 2048,
                "content_type": "image/png"
            }
        }"#;
        let output: FalUpscaleOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.image.width, Some(2048));
        assert_eq!(output.image.height, Some(2048));
        assert_eq!(output.image.content_type.as_deref(), Some("image/png"));
    }

    // -----------------------------------------------------------------------
    // Timing helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_timing_with_all_data() {
        let start = Instant::now();
        // Simulate some elapsed time.
        let in_progress = start; // effectively 0ms queue time for testing
        let timing = build_timing(start, Some(in_progress), Some(1.5));
        assert!(timing.total_ms.is_some());
        assert!(timing.queue_ms.is_some());
        assert_eq!(timing.execution_ms, Some(1500)); // 1.5s = 1500ms
    }

    #[test]
    fn build_timing_without_in_progress() {
        let start = Instant::now();
        let timing = build_timing(start, None, Some(2.0));
        assert!(timing.total_ms.is_some());
        assert!(timing.queue_ms.is_none());
        assert_eq!(timing.execution_ms, Some(2000));
    }

    #[test]
    fn build_timing_without_inference_time() {
        let start = Instant::now();
        let timing = build_timing(start, None, None);
        assert!(timing.total_ms.is_some());
        assert!(timing.queue_ms.is_none());
        // No inference_time and no in_progress => no execution_ms.
        assert!(timing.execution_ms.is_none());
    }

    // -----------------------------------------------------------------------
    // ComputeProvider trait tests (unit, not integration)
    // -----------------------------------------------------------------------

    #[test]
    fn provider_id_is_fal() {
        let provider = FalProvider::new("fal-test");
        assert_eq!(ComputeProvider::provider_id(&provider), "fal");
    }

    // -----------------------------------------------------------------------
    // Video parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_video_output_standard() {
        let output = serde_json::json!({
            "video": {
                "url": "https://v3.fal.media/files/rabbit/abc123.mp4",
                "file_name": "output.mp4",
                "file_size": 1234567,
                "content_type": "video/mp4"
            }
        });
        let video = parse_fal_video(&output).unwrap();
        assert_eq!(
            video.media.url.as_deref(),
            Some("https://v3.fal.media/files/rabbit/abc123.mp4")
        );
        assert_eq!(video.media.media_type, MediaType::Mp4);
        assert_eq!(video.media.file_size, Some(1234567));
    }

    #[test]
    fn parse_video_output_minimal() {
        let output = serde_json::json!({
            "video": {
                "url": "https://example.com/video.mp4"
            }
        });
        let video = parse_fal_video(&output).unwrap();
        assert_eq!(
            video.media.url.as_deref(),
            Some("https://example.com/video.mp4")
        );
        // Defaults to video/mp4.
        assert_eq!(video.media.media_type, MediaType::Mp4);
    }

    #[test]
    fn parse_video_output_missing_field() {
        let output = serde_json::json!({"result": "done"});
        let err = parse_fal_video(&output).unwrap_err();
        assert!(matches!(err, BlazenError::Serialization(_)));
    }

    // -----------------------------------------------------------------------
    // Audio parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_audio_output_audio_url_object() {
        let output = serde_json::json!({
            "audio_url": {
                "url": "https://v3.fal.media/files/audio/speech.wav",
                "content_type": "audio/wav",
                "file_size": 98765
            }
        });
        let audio = parse_fal_audio(&output).unwrap();
        assert_eq!(
            audio.media.url.as_deref(),
            Some("https://v3.fal.media/files/audio/speech.wav")
        );
        assert_eq!(audio.media.media_type, MediaType::Wav);
        assert_eq!(audio.media.file_size, Some(98765));
    }

    #[test]
    fn parse_audio_output_audio_url_string() {
        let output = serde_json::json!({
            "audio_url": "https://example.com/audio.wav"
        });
        let audio = parse_fal_audio(&output).unwrap();
        assert_eq!(
            audio.media.url.as_deref(),
            Some("https://example.com/audio.wav")
        );
        assert_eq!(audio.media.media_type, MediaType::Wav);
    }

    #[test]
    fn parse_audio_output_nested_audio() {
        let output = serde_json::json!({
            "audio": {
                "url": "https://example.com/music.mp3",
                "content_type": "audio/mpeg"
            }
        });
        let audio = parse_fal_audio(&output).unwrap();
        assert_eq!(
            audio.media.url.as_deref(),
            Some("https://example.com/music.mp3")
        );
        assert_eq!(audio.media.media_type, MediaType::Mp3);
    }

    #[test]
    fn parse_audio_output_missing_field() {
        let output = serde_json::json!({"result": "done"});
        let err = parse_fal_audio(&output).unwrap_err();
        assert!(matches!(err, BlazenError::Serialization(_)));
    }

    // -----------------------------------------------------------------------
    // merge_parameters tests
    // -----------------------------------------------------------------------

    #[test]
    fn merge_params_into_input() {
        let mut input = serde_json::json!({"prompt": "hello"});
        let params = serde_json::json!({"seed": 42, "guidance_scale": 7.5});
        merge_parameters(&mut input, &params);
        assert_eq!(input["seed"], 42);
        assert_eq!(input["guidance_scale"], 7.5);
        assert_eq!(input["prompt"], "hello");
    }

    #[test]
    fn merge_params_null_is_noop() {
        let mut input = serde_json::json!({"prompt": "hello"});
        merge_parameters(&mut input, &serde_json::Value::Null);
        assert_eq!(input, serde_json::json!({"prompt": "hello"}));
    }
}
