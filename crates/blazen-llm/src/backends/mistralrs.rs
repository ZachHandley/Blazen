//! Bridge between [`blazen_llm_mistralrs::MistralRsProvider`] and
//! [`crate::CompletionModel`].
//!
//! This module implements the `CompletionModel` trait for `MistralRsProvider`,
//! mapping Blazen's provider-agnostic request/response types to and from the
//! mistral.rs engine's own types.
//!
//! When the upstream `engine` feature is enabled on `blazen-llm-mistralrs`,
//! inference runs locally via the mistral.rs runtime. Without it, the
//! `infer`/`infer_stream` methods on the provider return an error, which is
//! propagated as [`BlazenError::Unsupported`].
//!
//! # Vision / multimodal
//!
//! Vision messages are supported when the provider is constructed with
//! [`MistralRsOptions::vision`](blazen_llm_mistralrs::MistralRsOptions::vision)
//! set to `true`. Image content parts in a [`CompletionRequest`] are
//! translated into [`InferenceImage`] values:
//!
//! - [`ImageSource::File`](crate::types::ImageSource::File) is passed
//!   through as a local path; the engine decodes it via `image::open`.
//! - [`ImageSource::Base64`](crate::types::ImageSource::Base64) is decoded
//!   inside this bridge and passed as raw bytes.
//! - [`ImageSource::Url`](crate::types::ImageSource::Url) is **not**
//!   supported -- the bridge never fetches network resources. Callers must
//!   materialise URL images into bytes or files before sending.

use std::pin::Pin;

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use blazen_llm_mistralrs::{
    ChatMessageInput, ChatRole, InferenceImage, InferenceImageSource, MistralRsProvider,
};
use futures_util::Stream;

use crate::error::BlazenError;
use crate::types::{
    ChatMessage, CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource,
    Role, StreamChunk,
};

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Convert a Blazen [`Role`] into a mistral.rs [`ChatRole`].
const fn to_chat_role(role: &Role) -> ChatRole {
    match role {
        Role::System => ChatRole::System,
        Role::User => ChatRole::User,
        Role::Assistant => ChatRole::Assistant,
        Role::Tool => ChatRole::Tool,
    }
}

/// Convert a Blazen [`ImageContent`] into an [`InferenceImage`] for the
/// mistral.rs backend.
///
/// URL sources are rejected with [`BlazenError::Unsupported`] because the
/// bridge does not perform network fetches -- callers should resolve URLs
/// into bytes or a file path before constructing the request.
fn convert_image(img: &ImageContent) -> Result<InferenceImage, BlazenError> {
    match &img.source {
        ImageSource::File { path } => Ok(InferenceImage {
            source: InferenceImageSource::Path(path.clone()),
        }),
        ImageSource::Base64 { data } => {
            let bytes = BASE64_STANDARD.decode(data.as_bytes()).map_err(|e| {
                BlazenError::unsupported(format!(
                    "mistralrs backend could not decode base64 image data: {e}"
                ))
            })?;
            Ok(InferenceImage {
                source: InferenceImageSource::Bytes(bytes),
            })
        }
        ImageSource::Url { .. } => Err(BlazenError::unsupported(
            "mistralrs backend does not fetch image URLs -- \
             pass an ImageSource::File or ImageSource::Base64 instead",
        )),
        ImageSource::ProviderFile { provider, id } => {
            tracing::warn!(
                target: "blazen::content",
                provider = ?provider,
                id = %id,
                "mistralrs: ProviderFile not supported by local inference backend; dropped. \
                 Provider file references describe remote, cloud-hosted assets, but mistralrs \
                 runs entirely on-device and cannot resolve them. Materialise the image into \
                 ImageSource::File or ImageSource::Base64 before sending."
            );
            Err(BlazenError::unsupported(
                "mistralrs backend does not support ImageSource::ProviderFile -- \
                 local inference cannot resolve remote provider file references; \
                 pass an ImageSource::File or ImageSource::Base64 instead",
            ))
        }
        ImageSource::Handle { handle } => {
            tracing::warn!(
                target: "blazen::content",
                handle = ?handle,
                "mistralrs: ContentHandle not supported by local inference backend; dropped. \
                 Content handles point at externally-stored bytes that the local engine \
                 cannot dereference. Resolve the handle to ImageSource::File or \
                 ImageSource::Base64 before sending."
            );
            Err(BlazenError::unsupported(
                "mistralrs backend does not support ImageSource::Handle -- \
                 local inference cannot dereference external content handles; \
                 pass an ImageSource::File or ImageSource::Base64 instead",
            ))
        }
    }
}

/// Build a single [`ChatMessageInput`] from a Blazen [`ChatMessage`],
/// concatenating all text parts and extracting image attachments.
///
/// Non-text, non-image parts (audio, video, files) are silently dropped --
/// mistral.rs text/vision pipelines do not consume them.
fn convert_message(msg: &ChatMessage) -> Result<ChatMessageInput, BlazenError> {
    let role = to_chat_role(&msg.role);

    // Fast path: plain text content.
    if let Some(text) = msg.content.as_text() {
        return Ok(ChatMessageInput::text(role, text));
    }

    // Walk all content parts, collecting text and images.
    let parts = msg.content.as_parts();
    let mut text_segments: Vec<String> = Vec::new();
    let mut images: Vec<InferenceImage> = Vec::new();
    for part in &parts {
        match part {
            ContentPart::Text { text } => text_segments.push(text.clone()),
            ContentPart::Image(img) => images.push(convert_image(img)?),
            // Audio, video, and files are not supported by mistral.rs
            // chat pipelines (audio is a separate pipeline category).
            ContentPart::Audio(_) | ContentPart::Video(_) | ContentPart::File(_) => {}
        }
    }

    let text = text_segments.join("\n");
    Ok(if images.is_empty() {
        ChatMessageInput::text(role, text)
    } else {
        ChatMessageInput::with_images(role, text, images)
    })
}

/// Convert Blazen messages into the [`ChatMessageInput`] list expected by
/// [`MistralRsProvider::infer`].
fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<ChatMessageInput>, BlazenError> {
    messages.iter().map(convert_message).collect()
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for MistralRsProvider {
    fn model_id(&self) -> &str {
        MistralRsProvider::model_id(self)
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let messages = convert_messages(&request.messages)?;

        let result = self
            .infer(messages)
            .await
            .map_err(|e| BlazenError::provider("mistralrs", e.to_string()))?;

        let tool_calls: Vec<crate::types::ToolCall> = result
            .tool_calls
            .into_iter()
            .map(|tc| {
                let arguments = serde_json::from_str(&tc.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(tc.arguments.clone()));
                crate::types::ToolCall {
                    id: tc.id,
                    name: tc.name,
                    arguments,
                }
            })
            .collect();

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

        let reasoning = result
            .reasoning_content
            .map(|text| crate::types::ReasoningTrace {
                text,
                signature: None,
                redacted: false,
                effort: None,
            });

        Ok(CompletionResponse {
            content: result.content,
            tool_calls,
            reasoning,
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

        let messages = convert_messages(&request.messages)?;

        let engine_stream = self
            .infer_stream(messages)
            .await
            .map_err(|e| BlazenError::provider("mistralrs", e.to_string()))?;

        let mapped = engine_stream.map(|chunk_result| {
            chunk_result
                .map(|chunk| {
                    let tool_calls: Vec<crate::types::ToolCall> = chunk
                        .tool_calls
                        .into_iter()
                        .map(|tc| {
                            let arguments =
                                serde_json::from_str(&tc.arguments).unwrap_or_else(|_| {
                                    serde_json::Value::String(tc.arguments.clone())
                                });
                            crate::types::ToolCall {
                                id: tc.id,
                                name: tc.name,
                                arguments,
                            }
                        })
                        .collect();

                    StreamChunk {
                        delta: chunk.delta,
                        tool_calls,
                        finish_reason: chunk.finish_reason,
                        reasoning_delta: chunk.reasoning_delta,
                        citations: Vec::new(),
                        artifacts: Vec::new(),
                    }
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
/// the underlying mistral.rs engine while preserving the existing lazy
/// auto-load-on-first-inference behavior provided by
/// [`MistralRsProvider::infer`] and [`MistralRsProvider::infer_stream`].
///
/// The impl forwards to the inherent methods on [`MistralRsProvider`] and
/// wraps [`blazen_llm_mistralrs::MistralRsError`] into
/// [`BlazenError::Provider`] via [`BlazenError::provider`]. The upstream
/// crate does not define a `From<MistralRsError> for BlazenError`
/// conversion (and cannot, because `blazen-llm-mistralrs` does not depend
/// on `blazen-llm` -- the dependency edge runs the other way), so we do
/// the conversion inline here.
///
/// Without the upstream `engine` feature, the inherent `load`,
/// `unload`, and `is_loaded` methods on [`MistralRsProvider`] are stubs
/// that return [`MistralRsError::EngineNotAvailable`] (for `load`),
/// succeed as no-ops (for `unload`), or return `false` (for
/// `is_loaded`). This mirrors the behavior of `infer` / `infer_stream`
/// and lets downstream crates depend on `LocalModel` without
/// unconditionally pulling in the heavy mistral.rs runtime.
#[async_trait]
impl crate::traits::LocalModel for MistralRsProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        MistralRsProvider::load(self)
            .await
            .map_err(|e| BlazenError::provider("mistralrs", e.to_string()))
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        MistralRsProvider::unload(self)
            .await
            .map_err(|e| BlazenError::provider("mistralrs", e.to_string()))
    }

    async fn is_loaded(&self) -> bool {
        MistralRsProvider::is_loaded(self).await
    }

    fn device(&self) -> crate::device::Device {
        self.options()
            .device
            .as_deref()
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

    use crate::types::{ChatMessage, ImageContent, ImageSource, MessageContent};

    /// Plain text messages pass through as text-only `ChatMessageInput`s.
    #[test]
    fn convert_text_only_message() {
        let msg = ChatMessage::user("hello world");
        let converted = convert_message(&msg).expect("text conversion should succeed");
        assert_eq!(converted.role, ChatRole::User);
        assert_eq!(converted.text, "hello world");
        assert!(converted.images.is_empty());
    }

    /// A single image content message (no surrounding text) translates
    /// into a `ChatMessageInput` with empty text and one image.
    #[test]
    fn convert_single_image_message_file_source() {
        let msg = ChatMessage {
            role: Role::User,
            content: MessageContent::Image(ImageContent {
                source: ImageSource::File {
                    path: std::path::PathBuf::from("/tmp/pic.png"),
                },
                media_type: None,
            }),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
            tool_result: None,
        };
        let converted = convert_message(&msg).expect("file image conversion should succeed");
        assert_eq!(converted.role, ChatRole::User);
        assert_eq!(converted.text, "");
        assert_eq!(converted.images.len(), 1);
        assert!(matches!(
            &converted.images[0].source,
            InferenceImageSource::Path(p) if p == std::path::Path::new("/tmp/pic.png")
        ));
    }

    /// A user message containing text + a file-sourced image produces a
    /// multimodal `ChatMessageInput` with both parts attached.
    #[test]
    fn convert_mixed_text_and_file_image() {
        use std::path::PathBuf;

        let msg = ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "describe this".into(),
                },
                ContentPart::Image(ImageContent {
                    source: ImageSource::File {
                        path: PathBuf::from("/tmp/cat.png"),
                    },
                    media_type: Some("image/png".into()),
                }),
            ]),
            tool_call_id: None,
            name: None,
            tool_calls: Vec::new(),
            tool_result: None,
        };
        let converted = convert_message(&msg).expect("mixed conversion should succeed");
        assert_eq!(converted.text, "describe this");
        assert_eq!(converted.images.len(), 1);
        assert!(matches!(
            &converted.images[0].source,
            InferenceImageSource::Path(p) if p == std::path::Path::new("/tmp/cat.png")
        ));
    }

    /// Base64 images are decoded on the bridge side and passed to the
    /// engine as raw bytes.
    #[test]
    fn convert_base64_image_decodes_to_bytes() {
        // "hello" as base64 (non-image payload is fine for this unit
        // test -- actual decoding happens inside the engine layer).
        let b64 = BASE64_STANDARD.encode(b"hello");
        let msg = ChatMessage::user_image_base64("what", b64, "image/png");
        let converted = convert_message(&msg).expect("b64 conversion should succeed");
        assert_eq!(converted.text, "what");
        assert_eq!(converted.images.len(), 1);
        match &converted.images[0].source {
            InferenceImageSource::Bytes(bytes) => assert_eq!(bytes, b"hello"),
            InferenceImageSource::Path(_) => panic!("expected bytes, got path"),
        }
    }

    /// URL image sources are rejected -- the mistralrs backend does not
    /// perform network fetches.
    #[test]
    fn convert_url_image_returns_unsupported() {
        let msg =
            ChatMessage::user_image_url("what", "https://example.com/img.png", Some("image/png"));
        let err = convert_message(&msg).expect_err("URL source should be unsupported");
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "expected Unsupported, got {err:?}"
        );
    }

    /// Malformed base64 produces a clean `Unsupported` error rather than
    /// a panic.
    #[test]
    fn convert_bad_base64_returns_unsupported() {
        let msg = ChatMessage::user_image_base64("what", "!!!not-base64!!!", "image/png");
        let err = convert_message(&msg).expect_err("bad b64 should fail");
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    /// Multi-message conversations round-trip: text system + mixed user
    /// become the right sequence of `ChatMessageInput`s.
    #[test]
    fn convert_conversation_preserves_order_and_roles() {
        let b64 = BASE64_STANDARD.encode(b"fake-png");
        let messages = vec![
            ChatMessage::system("you are a vision assistant"),
            ChatMessage::user_image_base64("what is this", b64, "image/png"),
        ];
        let converted = convert_messages(&messages).expect("conversation should convert");
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, ChatRole::System);
        assert_eq!(converted[0].text, "you are a vision assistant");
        assert!(converted[0].images.is_empty());
        assert_eq!(converted[1].role, ChatRole::User);
        assert_eq!(converted[1].text, "what is this");
        assert_eq!(converted[1].images.len(), 1);
    }

    /// Audio / video / file parts are silently dropped; only text and
    /// images reach the engine.
    #[test]
    fn convert_drops_unsupported_content_parts() {
        use crate::types::{AudioContent, VideoContent};

        let msg = ChatMessage::user_parts(vec![
            ContentPart::Text {
                text: "hello".into(),
            },
            ContentPart::Audio(AudioContent::from_url("https://a.com/c.mp3")),
            ContentPart::Video(VideoContent::from_url("https://v.com/c.mp4")),
        ]);
        let converted = convert_message(&msg).expect("drop should succeed");
        assert_eq!(converted.text, "hello");
        assert!(converted.images.is_empty());
    }
}
