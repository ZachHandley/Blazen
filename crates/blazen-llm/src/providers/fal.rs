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

use std::pin::Pin;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::Stream;
use futures_util::stream;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::LlmError;
use crate::types::{CompletionRequest, CompletionResponse, MessageContent, StreamChunk};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FAL_QUEUE_URL: &str = "https://queue.fal.run";
const FAL_SYNC_URL: &str = "https://fal.run";

/// Default poll interval for queue-based execution.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Maximum number of poll iterations before giving up.
const MAX_POLL_ITERATIONS: u32 = 600; // 10 minutes at 1s intervals

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
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::fal::FalProvider;
///
/// let provider = FalProvider::new("fal-key-...")
///     .with_model("fal-ai/any-llm");
/// ```
#[derive(Debug, Clone)]
pub struct FalProvider {
    client: Client,
    api_key: String,
    default_model: String,
    /// The underlying LLM model to use when proxying through `fal-ai/any-llm`.
    llm_model: String,
    execution_mode: FalExecutionMode,
}

impl FalProvider {
    /// Create a new fal.ai provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            default_model: "fal-ai/any-llm".to_owned(),
            llm_model: "anthropic/claude-sonnet-4".to_owned(),
            execution_mode: FalExecutionMode::Queue {
                poll_interval: DEFAULT_POLL_INTERVAL,
            },
        }
    }

    /// Override the fal.ai model endpoint (e.g. `fal-ai/any-llm`).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set the underlying LLM model used by `fal-ai/any-llm`.
    ///
    /// This is the model name passed in the request body (e.g.
    /// `"anthropic/claude-sonnet-4"`, `"openai/gpt-4o"`).
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

    /// Build the JSON request body for the `fal-ai/any-llm` endpoint.
    ///
    /// fal-ai/any-llm is text-only. Non-text content (images, files) is
    /// dropped with a warning.
    fn build_body(&self, request: &CompletionRequest) -> serde_json::Value {
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

        body
    }

    /// Resolve the fal.ai model endpoint to use.
    fn resolve_fal_model(&self) -> &str {
        &self.default_model
    }

    /// Apply fal.ai authentication (`Authorization: Key <key>`).
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.header("Authorization", format!("Key {}", self.api_key))
    }

    /// Execute synchronously: POST to fal.run and wait for the response.
    async fn execute_sync(&self, body: &serde_json::Value) -> Result<serde_json::Value, LlmError> {
        let model = self.resolve_fal_model();
        let url = format!("{FAL_SYNC_URL}/{model}");

        let response = self
            .apply_auth(self.client.post(&url))
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unable to read error body>"));
            return match status.as_u16() {
                401 => Err(LlmError::AuthFailed),
                429 => Err(LlmError::RateLimited { retry_after: None }),
                _ => Err(LlmError::RequestFailed(format!(
                    "HTTP {status}: {error_body}"
                ))),
            };
        }

        response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))
    }

    /// Execute via queue: submit, poll, return result.
    async fn execute_queue(
        &self,
        body: &serde_json::Value,
        poll_interval: Duration,
    ) -> Result<serde_json::Value, LlmError> {
        let model = self.resolve_fal_model();
        let submit_url = format!("{FAL_QUEUE_URL}/{model}");

        // Submit to queue.
        let submit_response = self
            .apply_auth(self.client.post(&submit_url))
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = submit_response.status();
        if !status.is_success() {
            let error_body = submit_response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unable to read error body>"));
            return match status.as_u16() {
                401 => Err(LlmError::AuthFailed),
                429 => Err(LlmError::RateLimited { retry_after: None }),
                _ => Err(LlmError::RequestFailed(format!(
                    "HTTP {status}: {error_body}"
                ))),
            };
        }

        let queue_response: FalQueueResponse = submit_response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let request_id = queue_response.request_id;
        debug!(request_id = %request_id, "fal.ai job submitted to queue");

        // Poll for completion.
        let status_url = format!("{FAL_QUEUE_URL}/{model}/requests/{request_id}/status");
        let result_url = format!("{FAL_QUEUE_URL}/{model}/requests/{request_id}");

        for _ in 0..MAX_POLL_ITERATIONS {
            tokio::time::sleep(poll_interval).await;

            let status_response = self
                .apply_auth(self.client.get(&status_url))
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            if !status_response.status().is_success() {
                continue;
            }

            let status_body: FalStatusResponse = status_response
                .json()
                .await
                .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

            match status_body.status.as_str() {
                "COMPLETED" => {
                    // Fetch the result.
                    let result_response = self
                        .apply_auth(self.client.get(&result_url))
                        .send()
                        .await
                        .map_err(|e| LlmError::Http(e.to_string()))?;

                    return result_response
                        .json()
                        .await
                        .map_err(|e| LlmError::InvalidResponse(e.to_string()));
                }
                "FAILED" => {
                    let error_msg = status_body
                        .error
                        .unwrap_or_else(|| "unknown error".to_owned());
                    return Err(LlmError::RequestFailed(format!(
                        "fal.ai job failed: {error_msg}"
                    )));
                }
                // IN_QUEUE, IN_PROGRESS -- keep polling.
                _ => {}
            }
        }

        Err(LlmError::RequestFailed(
            "fal.ai job timed out waiting for completion".into(),
        ))
    }

    /// Execute via webhook: submit with webhook URL.
    async fn execute_webhook(
        &self,
        body: &serde_json::Value,
        webhook_url: &str,
    ) -> Result<serde_json::Value, LlmError> {
        let model = self.resolve_fal_model();
        let submit_url = format!("{FAL_QUEUE_URL}/{model}");

        let submit_response = self
            .apply_auth(self.client.post(&submit_url))
            .header("fal_webhook", webhook_url)
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = submit_response.status();
        if !status.is_success() {
            let error_body = submit_response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unable to read error body>"));
            return Err(LlmError::RequestFailed(format!(
                "HTTP {status}: {error_body}"
            )));
        }

        // Webhook mode returns the queue submission response. The actual
        // result will be delivered to the webhook URL.
        submit_response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FalQueueResponse {
    request_id: String,
}

#[derive(Debug, Deserialize)]
struct FalStatusResponse {
    status: String,
    #[serde(default)]
    error: Option<String>,
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

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for FalProvider {
    fn model_id(&self) -> &str {
        &self.default_model
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
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

        let body = self.build_body(&request);
        debug!(model = %self.default_model, "fal.ai completion request");

        let result = match &self.execution_mode {
            FalExecutionMode::Sync => self.execute_sync(&body).await?,
            FalExecutionMode::Queue { poll_interval } => {
                self.execute_queue(&body, *poll_interval).await?
            }
            FalExecutionMode::Webhook { url } => self.execute_webhook(&body, url).await?,
        };

        // Parse the fal.ai response.
        let fal_response: FalLlmResponse =
            serde_json::from_value(result).map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        if let Some(error) = fal_response.error {
            return Err(LlmError::RequestFailed(format!(
                "fal.ai model error: {error}"
            )));
        }

        span.record("duration_ms", u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
        span.record("finish_reason", "stop");

        Ok(CompletionResponse {
            content: fal_response.output,
            tool_calls: Vec::new(), // fal.ai/any-llm doesn't support tool calling.
            usage: None,
            model: self.default_model.clone(),
            finish_reason: Some("stop".to_owned()),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError> {
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

        span.record("duration_ms", u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
        span.record("chunk_count", 1u64);

        let chunks: Vec<Result<StreamChunk, LlmError>> = vec![Ok(StreamChunk {
            delta: response.content,
            tool_calls: Vec::new(),
            finish_reason: Some("stop".to_owned()),
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[test]
    fn default_config() {
        let provider = FalProvider::new("fal-test");
        assert_eq!(provider.default_model, "fal-ai/any-llm");
        assert_eq!(provider.llm_model, "anthropic/claude-sonnet-4");
        assert!(matches!(
            provider.execution_mode,
            FalExecutionMode::Queue { .. }
        ));
    }

    #[test]
    fn with_model_override() {
        let provider = FalProvider::new("fal-test").with_model("fal-ai/fast-sdxl");
        assert_eq!(provider.default_model, "fal-ai/fast-sdxl");
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

    #[test]
    fn build_body_basic() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello world")]);

        let body = provider.build_body(&request);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4");
        assert!(body["prompt"].as_str().unwrap().contains("Hello world"));
    }

    #[test]
    fn build_body_with_system() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![
            ChatMessage::system("Be helpful"),
            ChatMessage::user("Hello"),
        ]);

        let body = provider.build_body(&request);
        assert_eq!(body["system_prompt"], "Be helpful");
        assert!(body["prompt"].as_str().unwrap().contains("Hello"));
    }

    #[test]
    fn build_body_model_override() {
        let provider = FalProvider::new("fal-test");
        let request =
            CompletionRequest::new(vec![ChatMessage::user("Hi")]).with_model("openai/gpt-4o");

        let body = provider.build_body(&request);
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    fn test_text_backward_compat() {
        let provider = FalProvider::new("fal-test");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request);
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

        let body = provider.build_body(&request);
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

        let body = provider.build_body(&request);
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

        let body = provider.build_body(&request);
        let prompt = body["prompt"].as_str().unwrap();
        // Both text parts should be concatenated.
        assert!(prompt.contains("First"));
        assert!(prompt.contains("Second"));
    }

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
    fn parse_queue_response() {
        let json = r#"{"request_id":"abc-123-def"}"#;
        let response: FalQueueResponse = serde_json::from_str(json).unwrap();
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
    fn parse_status_failed() {
        let json = r#"{"status":"FAILED","error":"Out of memory"}"#;
        let response: FalStatusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "FAILED");
        assert_eq!(response.error.as_deref(), Some("Out of memory"));
    }
}
