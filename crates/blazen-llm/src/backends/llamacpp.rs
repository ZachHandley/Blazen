//! Bridge between [`blazen_llm_llamacpp::LlamaCppProvider`] and
//! [`crate::CompletionModel`].
//!
//! This module implements the `CompletionModel` trait for `LlamaCppProvider`,
//! mapping Blazen's provider-agnostic request/response types to and from the
//! llama.cpp engine's own types.
//!
//! When the upstream `engine` feature is enabled on `blazen-llm-llamacpp`,
//! inference runs locally via llama.cpp. Without it, the `infer`/`infer_stream`
//! methods on the provider return an error, which is propagated as
//! [`BlazenError::Provider`].
//!
//! # Limitations
//!
//! - **No tool calling**: llama.cpp does not natively support structured
//!   tool-call extraction. `tool_calls` in the response will always be empty.
//! - **No vision**: Image content parts are silently dropped -- llama.cpp
//!   text pipelines do not consume them.
//! - **No reasoning trace**: The provider does not expose chain-of-thought
//!   content separately from the main response.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_llm_llamacpp::{ChatMessageInput, ChatRole, LlamaCppProvider};
use futures_util::Stream;

use crate::error::BlazenError;
use crate::types::{
    ChatMessage, CompletionRequest, CompletionResponse, ContentPart, Role, StreamChunk,
};

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Convert a Blazen [`Role`] into a llama.cpp [`ChatRole`].
const fn to_chat_role(role: &Role) -> ChatRole {
    match role {
        Role::System => ChatRole::System,
        Role::User => ChatRole::User,
        Role::Assistant => ChatRole::Assistant,
        Role::Tool => ChatRole::Tool,
    }
}

/// Build a single [`ChatMessageInput`] from a Blazen [`ChatMessage`],
/// concatenating all text parts. Non-text parts (images, audio, video, files)
/// are silently dropped -- llama.cpp text pipelines do not consume them.
fn convert_message(msg: &ChatMessage) -> ChatMessageInput {
    let role = to_chat_role(&msg.role);

    // Fast path: plain text content.
    if let Some(text) = msg.content.as_text() {
        return ChatMessageInput::new(role, text);
    }

    // Walk all content parts, collecting text only.
    let parts = msg.content.as_parts();
    let mut text_segments: Vec<String> = Vec::new();
    for part in &parts {
        if let ContentPart::Text { text } = part {
            text_segments.push(text.clone());
        }
        // Images, audio, video, and files are silently dropped.
    }

    ChatMessageInput::new(role, text_segments.join("\n"))
}

/// Convert Blazen messages into the [`ChatMessageInput`] list expected by
/// [`LlamaCppProvider::infer`].
fn convert_messages(messages: &[ChatMessage]) -> Vec<ChatMessageInput> {
    messages.iter().map(convert_message).collect()
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for LlamaCppProvider {
    fn model_id(&self) -> &str {
        LlamaCppProvider::model_id(self)
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let messages = convert_messages(&request.messages);

        let result = self
            .infer(messages)
            .await
            .map_err(|e| BlazenError::provider("llamacpp", e.to_string()))?;

        let usage = Some(crate::types::TokenUsage {
            prompt_tokens: result.usage.prompt_tokens,
            completion_tokens: result.usage.completion_tokens,
            total_tokens: result.usage.total_tokens,
            ..Default::default()
        });

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let total_ms = (f64::from(result.usage.total_time_sec) * 1000.0) as u64;
        let timing = Some(crate::types::RequestTiming {
            queue_ms: None,
            execution_ms: Some(total_ms),
            total_ms: Some(total_ms),
        });

        Ok(CompletionResponse {
            content: result.content,
            tool_calls: Vec::new(),
            reasoning: None,
            citations: Vec::new(),
            artifacts: Vec::new(),
            usage,
            model: result.model,
            finish_reason: Some(result.finish_reason),
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
        use futures_util::StreamExt;

        let messages = convert_messages(&request.messages);

        let engine_stream = self
            .infer_stream(messages)
            .await
            .map_err(|e| BlazenError::provider("llamacpp", e.to_string()))?;

        let mapped = engine_stream.map(|chunk_result| {
            chunk_result
                .map(|chunk| StreamChunk {
                    delta: chunk.delta,
                    tool_calls: Vec::new(),
                    finish_reason: chunk.finish_reason,
                    reasoning_delta: None,
                    citations: Vec::new(),
                    artifacts: Vec::new(),
                })
                .map_err(|e| BlazenError::stream_error(e.to_string()))
        });

        Ok(Box::pin(mapped))
    }
}

// ---------------------------------------------------------------------------
// LocalModel implementation
// ---------------------------------------------------------------------------

/// `LocalModel` bridge: gives callers explicit `load`/`unload` control over
/// the underlying llama.cpp engine while preserving the existing lazy
/// auto-load-on-first-inference behavior provided by
/// [`LlamaCppProvider::infer`] and [`LlamaCppProvider::infer_stream`].
///
/// The impl forwards to the inherent methods on [`LlamaCppProvider`] and
/// wraps [`blazen_llm_llamacpp::LlamaCppError`] into
/// [`BlazenError::Provider`] via [`BlazenError::provider`]. The upstream
/// crate does not define a `From<LlamaCppError> for BlazenError`
/// conversion (and cannot, because `blazen-llm-llamacpp` does not depend
/// on `blazen-llm` -- the dependency edge runs the other way), so we do
/// the conversion inline here.
///
/// Without the upstream `engine` feature, the inherent `load`,
/// `unload`, and `is_loaded` methods on [`LlamaCppProvider`] are stubs
/// that return [`blazen_llm_llamacpp::LlamaCppError::EngineNotAvailable`]
/// (for `load`), succeed as no-ops (for `unload`), or return `false`
/// (for `is_loaded`). This mirrors the behavior of `infer` /
/// `infer_stream` and lets downstream crates depend on `LocalModel`
/// without unconditionally pulling in the llama.cpp runtime.
#[async_trait]
impl crate::traits::LocalModel for LlamaCppProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        LlamaCppProvider::load(self)
            .await
            .map_err(|e| BlazenError::provider("llamacpp", e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        LlamaCppProvider::unload(self)
            .await
            .map_err(|e| BlazenError::provider("llamacpp", e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        LlamaCppProvider::is_loaded(self).await
    }

    fn device(&self) -> crate::device::Device {
        LlamaCppProvider::device_str(self)
            .and_then(|s| crate::device::Device::parse(s).ok())
            .unwrap_or(crate::device::Device::Cpu)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{ChatMessage, MessageContent, Role};

    /// Plain text messages pass through as text-only `ChatMessageInput`s.
    #[test]
    fn convert_text_only_message() {
        let msg = ChatMessage::user("hello world");
        let converted = convert_message(&msg);
        assert_eq!(converted.role, ChatRole::User);
        assert_eq!(converted.text, "hello world");
    }

    /// Role mappings are correct.
    #[test]
    fn role_mapping() {
        assert_eq!(to_chat_role(&Role::System), ChatRole::System);
        assert_eq!(to_chat_role(&Role::User), ChatRole::User);
        assert_eq!(to_chat_role(&Role::Assistant), ChatRole::Assistant);
        assert_eq!(to_chat_role(&Role::Tool), ChatRole::Tool);
    }

    /// Multi-part messages with text + non-text content keep only the text.
    #[test]
    fn convert_mixed_content_drops_non_text() {
        use crate::types::AudioContent;

        let msg = ChatMessage::user_parts(vec![
            ContentPart::Text {
                text: "hello".into(),
            },
            ContentPart::Audio(AudioContent::from_url("https://a.com/c.mp3")),
            ContentPart::Text {
                text: "world".into(),
            },
        ]);
        let converted = convert_message(&msg);
        assert_eq!(converted.text, "hello\nworld");
    }

    /// System messages are converted correctly.
    #[test]
    fn convert_system_message() {
        let msg = ChatMessage::system("you are helpful");
        let converted = convert_message(&msg);
        assert_eq!(converted.role, ChatRole::System);
        assert_eq!(converted.text, "you are helpful");
    }

    /// A conversation with multiple messages preserves order and roles.
    #[test]
    fn convert_conversation_preserves_order() {
        let messages = vec![
            ChatMessage::system("be concise"),
            ChatMessage::user("hello"),
            ChatMessage {
                role: Role::Assistant,
                content: MessageContent::Text("hi there".into()),
                tool_call_id: None,
                name: None,
                tool_calls: Vec::new(),
                tool_result: None,
            },
        ];
        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, ChatRole::System);
        assert_eq!(converted[1].role, ChatRole::User);
        assert_eq!(converted[2].role, ChatRole::Assistant);
        assert_eq!(converted[2].text, "hi there");
    }

    /// Empty message list produces empty output.
    #[test]
    fn convert_empty_messages() {
        let converted = convert_messages(&[]);
        assert!(converted.is_empty());
    }
}
