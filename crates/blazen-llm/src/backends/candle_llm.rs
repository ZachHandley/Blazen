//! Bridge between [`blazen_llm_candle::CandleLlmProvider`] and
//! [`crate::CompletionModel`].
//!
//! This module implements the `CompletionModel` trait for `CandleLlmProvider`,
//! mapping Blazen's provider-agnostic request/response types to and from the
//! candle engine's own types.
//!
//! When the upstream `engine` feature is enabled on `blazen-llm-candle`,
//! inference runs locally via the candle runtime using GGUF quantized models.
//! Without it, the `infer`/`infer_stream` methods on the provider return an
//! error, which is propagated as [`BlazenError::Unsupported`].
//!
//! # Limitations
//!
//! - Only text content is supported (no vision/audio).
//! - Tool calling is not implemented at the candle level -- the model
//!   receives tool definitions in the system prompt but structured tool
//!   call parsing is not performed.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_llm_candle::CandleLlmProvider;
use futures_util::Stream;

use crate::error::BlazenError;
use crate::types::{ChatMessage, CompletionRequest, CompletionResponse, StreamChunk};

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Convert a Blazen [`ChatMessage`] into a `(role, content)` pair for candle.
fn convert_message(msg: &ChatMessage) -> (String, String) {
    let role = match msg.role {
        crate::types::Role::System => "system",
        crate::types::Role::User => "user",
        crate::types::Role::Assistant => "assistant",
        crate::types::Role::Tool => "tool",
    };

    // Extract text content. For multipart messages, concatenate all text
    // parts. Image / audio / video parts are silently dropped since candle
    // only supports text generation.
    let content = if let Some(text) = msg.content.as_text() {
        text.to_string()
    } else {
        let parts = msg.content.as_parts();
        parts
            .iter()
            .filter_map(|p| {
                if let crate::types::ContentPart::Text { text } = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    (role.to_string(), content)
}

/// Convert Blazen messages into the `Vec<(String, String)>` expected by
/// [`CandleLlmProvider::infer`].
fn convert_messages(messages: &[ChatMessage]) -> Vec<(String, String)> {
    messages.iter().map(convert_message).collect()
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

/// Wrapper around `CandleLlmProvider` that caches a `model_id: String`
/// suitable for the `CompletionModel::model_id(&self) -> &str` signature.
///
/// `CandleLlmProvider::model_id` returns `Option<&str>`, which cannot
/// satisfy the trait's non-optional return type, so we snapshot it
/// (falling back to `"candle-local"` when no id was configured).
/// `CandleLlmProvider::infer` and `infer_stream` already take `&self`,
/// so no outer mutex is needed; the provider manages its own internal
/// `tokio::sync::Mutex<Option<CandleEngine>>` for interior mutability.
pub struct CandleLlmCompletionModel {
    provider: CandleLlmProvider,
    model_id: String,
}

impl CandleLlmCompletionModel {
    /// Create a new completion model wrapper from a `CandleLlmProvider`.
    #[must_use]
    pub fn new(provider: CandleLlmProvider) -> Self {
        let model_id = provider.model_id().unwrap_or("candle-local").to_string();
        Self { provider, model_id }
    }
}

#[async_trait]
impl crate::traits::CompletionModel for CandleLlmCompletionModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let messages = convert_messages(&request.messages);
        let max_tokens = request.max_tokens.map(|t| t as usize);
        let temperature = request.temperature.map(f64::from);
        let top_p = request.top_p.map(f64::from);

        let result = self
            .provider
            .infer(messages, max_tokens, temperature, top_p)
            .await
            .map_err(|e| BlazenError::provider("candle-llm", e.to_string()))?;

        #[allow(clippy::cast_possible_truncation)]
        let usage = Some(crate::types::TokenUsage {
            prompt_tokens: result.prompt_tokens as u32,
            completion_tokens: result.completion_tokens as u32,
            total_tokens: (result.prompt_tokens + result.completion_tokens) as u32,
            ..Default::default()
        });

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let total_ms = (result.total_time_secs * 1000.0) as u64;
        let timing = Some(crate::types::RequestTiming {
            queue_ms: None,
            execution_ms: Some(total_ms),
            total_ms: Some(total_ms),
        });

        Ok(CompletionResponse {
            content: Some(result.content),
            tool_calls: Vec::new(),
            reasoning: None,
            citations: Vec::new(),
            artifacts: Vec::new(),
            usage,
            model: self.model_id.clone(),
            finish_reason: Some("stop".to_string()),
            cost: None,
            timing,
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            metadata: serde_json::Value::Null,
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let messages = convert_messages(&request.messages);
        let max_tokens = request.max_tokens.map(|t| t as usize);
        let temperature = request.temperature.map(f64::from);
        let top_p = request.top_p.map(f64::from);

        let rx = self
            .provider
            .infer_stream(messages, max_tokens, temperature, top_p)
            .await
            .map_err(|e| BlazenError::provider("candle-llm", e.to_string()))?;

        // Convert the mpsc receiver into an async Stream using unfold.
        // State: (receiver, whether we've sent the final chunk).
        let stream = futures_util::stream::unfold((rx, false), |(mut rx, done)| async move {
            if done {
                return None;
            }
            match rx.recv().await {
                Some(Ok(text)) => {
                    let chunk = StreamChunk {
                        delta: Some(text),
                        tool_calls: Vec::new(),
                        finish_reason: None,
                        reasoning_delta: None,
                        citations: Vec::new(),
                        artifacts: Vec::new(),
                    };
                    Some((Ok(chunk), (rx, false)))
                }
                Some(Err(e)) => {
                    let err = BlazenError::stream_error(e.to_string());
                    Some((Err(err), (rx, true)))
                }
                None => {
                    // Channel closed -- emit final chunk with finish reason.
                    let chunk = StreamChunk {
                        delta: None,
                        tool_calls: Vec::new(),
                        finish_reason: Some("stop".to_string()),
                        reasoning_delta: None,
                        citations: Vec::new(),
                        artifacts: Vec::new(),
                    };
                    Some((Ok(chunk), (rx, true)))
                }
            }
        });

        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// LocalModel implementation
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: gives callers explicit `load`/`unload` control
/// over the underlying candle engine while preserving the existing lazy
/// auto-load-on-first-inference behavior provided by
/// [`CandleLlmProvider::infer`] and [`CandleLlmProvider::infer_stream`].
///
/// The impl forwards to the inherent methods on [`CandleLlmProvider`]
/// and wraps [`blazen_llm_candle::CandleLlmError`] into
/// [`BlazenError::Provider`] via [`BlazenError::provider`]. The upstream
/// crate does not define a `From<CandleLlmError> for BlazenError`
/// conversion (and cannot, because `blazen-llm-candle` does not depend
/// on `blazen-llm` -- the dependency edge runs the other way), so we do
/// the conversion inline here.
///
/// Without the upstream `engine` feature, the inherent `load`,
/// `unload`, and `is_loaded` methods on [`CandleLlmProvider`] are stubs
/// that return [`blazen_llm_candle::CandleLlmError::EngineNotAvailable`]
/// (for `load`), succeed as no-ops (for `unload`), or return `false`
/// (for `is_loaded`). This mirrors the behavior of `infer` /
/// `infer_stream` and lets downstream crates depend on `LocalModel`
/// without unconditionally pulling in the heavy candle runtime.
#[async_trait]
impl crate::traits::LocalModel for CandleLlmProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        CandleLlmProvider::load(self)
            .await
            .map_err(|e| BlazenError::provider("candle-llm", e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        CandleLlmProvider::unload(self)
            .await
            .map_err(|e| BlazenError::provider("candle-llm", e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        CandleLlmProvider::is_loaded(self).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, MessageContent, Role};

    #[test]
    fn convert_text_message() {
        let msg = ChatMessage::user("hello world");
        let (role, content) = convert_message(&msg);
        assert_eq!(role, "user");
        assert_eq!(content, "hello world");
    }

    #[test]
    fn convert_system_message() {
        let msg = ChatMessage::system("you are helpful");
        let (role, content) = convert_message(&msg);
        assert_eq!(role, "system");
        assert_eq!(content, "you are helpful");
    }

    #[test]
    fn convert_multipart_keeps_text_only() {
        use crate::types::{AudioContent, ContentPart};

        let msg = ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "hello".into(),
                },
                ContentPart::Audio(AudioContent::from_url("https://example.com/a.mp3")),
                ContentPart::Text {
                    text: "world".into(),
                },
            ]),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
            tool_result: None,
        };
        let (role, content) = convert_message(&msg);
        assert_eq!(role, "user");
        assert_eq!(content, "hello\nworld");
    }

    #[test]
    fn convert_messages_preserves_order() {
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("usr"),
            ChatMessage {
                role: Role::Assistant,
                content: MessageContent::Text("asst".into()),
                tool_call_id: None,
                name: None,
                tool_calls: Vec::new(),
                tool_result: None,
            },
        ];
        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].0, "system");
        assert_eq!(converted[1].0, "user");
        assert_eq!(converted[2].0, "assistant");
    }
}
