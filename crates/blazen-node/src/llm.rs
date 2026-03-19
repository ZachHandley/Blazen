//! JavaScript wrappers for LLM completion models.
//!
//! Provides [`JsCompletionModel`] with factory constructors for each
//! supported provider (`OpenAI`, Anthropic, Gemini, etc.).

use std::sync::Arc;

use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::CompletionModel;
use blazen_llm::types::{
    ChatMessage, CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource,
    MessageContent, Role, ToolDefinition,
};

use crate::error::llm_error_to_napi;

/// Stream callback: takes a `serde_json::Value` chunk, returns nothing meaningful.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
type StreamChunkCallbackTsfn =
    ThreadsafeFunction<serde_json::Value, Unknown<'static>, serde_json::Value, Status, false, true>;

// ---------------------------------------------------------------------------
// Role string enum
// ---------------------------------------------------------------------------

/// The role of a participant in a chat conversation.
#[napi(string_enum)]
pub enum JsRole {
    #[napi(value = "system")]
    System,
    #[napi(value = "user")]
    User,
    #[napi(value = "assistant")]
    Assistant,
    #[napi(value = "tool")]
    Tool,
}

// ---------------------------------------------------------------------------
// Response / options object structs
// ---------------------------------------------------------------------------

/// A tool invocation requested by the model.
#[napi(object)]
pub struct JsToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Token usage statistics for a completion request.
#[napi(object)]
pub struct JsTokenUsage {
    #[napi(js_name = "promptTokens")]
    pub prompt_tokens: u32,
    #[napi(js_name = "completionTokens")]
    pub completion_tokens: u32,
    #[napi(js_name = "totalTokens")]
    pub total_tokens: u32,
}

/// Timing metadata for a completion request.
#[napi(object)]
pub struct JsRequestTiming {
    #[napi(js_name = "queueMs")]
    pub queue_ms: Option<i64>,
    #[napi(js_name = "executionMs")]
    pub execution_ms: Option<i64>,
    #[napi(js_name = "totalMs")]
    pub total_ms: Option<i64>,
}

/// The result of a chat completion.
#[napi(object)]
pub struct JsCompletionResponse {
    pub content: Option<String>,
    #[napi(js_name = "toolCalls")]
    pub tool_calls: Vec<JsToolCall>,
    pub usage: Option<JsTokenUsage>,
    pub model: String,
    #[napi(js_name = "finishReason")]
    pub finish_reason: Option<String>,
    pub cost: Option<f64>,
    pub timing: Option<JsRequestTiming>,
    pub images: Vec<serde_json::Value>,
    pub audio: Vec<serde_json::Value>,
    pub videos: Vec<serde_json::Value>,
    pub metadata: serde_json::Value,
}

/// Describes a tool that the model may invoke during a conversation.
#[napi(object)]
pub struct JsToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Options for a chat completion request.
#[napi(object)]
pub struct JsCompletionOptions {
    pub temperature: Option<f64>,
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<i64>,
    #[napi(js_name = "topP")]
    pub top_p: Option<f64>,
    pub model: Option<String>,
    pub tools: Option<Vec<JsToolDefinition>>,
}

// ---------------------------------------------------------------------------
// Content part types for multimodal messages
// ---------------------------------------------------------------------------

/// How an image is provided (URL or base64).
#[napi(object)]
pub struct JsImageSource {
    #[napi(js_name = "sourceType")]
    pub source_type: String,
    pub url: Option<String>,
    pub data: Option<String>,
}

/// Image content for multimodal messages.
#[napi(object)]
pub struct JsImageContent {
    pub source: JsImageSource,
    #[napi(js_name = "mediaType")]
    pub media_type: Option<String>,
}

/// A single part in a multi-part message.
#[napi(object)]
pub struct JsContentPart {
    #[napi(js_name = "partType")]
    pub part_type: String,
    pub text: Option<String>,
    pub image: Option<JsImageContent>,
}

// ---------------------------------------------------------------------------
// ChatMessage class
// ---------------------------------------------------------------------------

/// Options for creating a `ChatMessage`.
#[napi(object)]
pub struct ChatMessageOptions {
    /// Role: "system", "user", "assistant", or "tool". Defaults to "user".
    pub role: Option<String>,
    /// Text content.
    pub content: Option<String>,
    /// Multimodal content parts (alternative to content).
    pub parts: Option<Vec<JsContentPart>>,
}

/// A single message in a chat conversation.
#[napi(js_name = "ChatMessage")]
pub struct JsChatMessage {
    pub(crate) inner: ChatMessage,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsChatMessage {
    /// Create a new chat message from an options object.
    ///
    /// `role` defaults to `"user"` if not provided.
    /// Supply either `content` (text) or `parts` (multimodal).
    #[napi(constructor)]
    pub fn new(options: ChatMessageOptions) -> Result<Self> {
        let role_str = options.role.as_deref().unwrap_or("user");
        let role = parse_role(role_str)?;

        let content = if let Some(parts) = options.parts {
            let rust_parts = convert_js_parts(parts)?;
            MessageContent::Parts(rust_parts)
        } else {
            MessageContent::Text(options.content.unwrap_or_default())
        };

        Ok(Self {
            inner: ChatMessage { role, content },
        })
    }

    /// Create a system message.
    #[napi(factory)]
    pub fn system(content: String) -> Self {
        Self {
            inner: ChatMessage::system(content),
        }
    }

    /// Create a user message.
    #[napi(factory)]
    pub fn user(content: String) -> Self {
        Self {
            inner: ChatMessage::user(content),
        }
    }

    /// Create an assistant message.
    #[napi(factory)]
    pub fn assistant(content: String) -> Self {
        Self {
            inner: ChatMessage::assistant(content),
        }
    }

    /// Create a tool result message.
    #[napi(factory)]
    pub fn tool(content: String) -> Self {
        Self {
            inner: ChatMessage::tool(content),
        }
    }

    /// Create a user message containing text and an image from a URL.
    #[napi(factory, js_name = "userImageUrl")]
    pub fn user_image_url(text: String, url: String, media_type: Option<String>) -> Self {
        Self {
            inner: ChatMessage::user_image_url(text, url, media_type.as_deref()),
        }
    }

    /// Create a user message containing text and a base64-encoded image.
    #[napi(factory, js_name = "userImageBase64")]
    pub fn user_image_base64(text: String, data: String, media_type: String) -> Self {
        Self {
            inner: ChatMessage::user_image_base64(text, data, media_type),
        }
    }

    /// Create a user message from an explicit list of content parts.
    #[napi(factory, js_name = "userParts")]
    pub fn user_parts(parts: Vec<JsContentPart>) -> Result<Self> {
        let content_parts = convert_js_parts(parts)?;
        Ok(Self {
            inner: ChatMessage::user_parts(content_parts),
        })
    }

    /// The role of the message author.
    #[napi(getter)]
    pub fn role(&self) -> String {
        match self.inner.role {
            Role::System => "system".to_owned(),
            Role::User => "user".to_owned(),
            Role::Assistant => "assistant".to_owned(),
            Role::Tool => "tool".to_owned(),
        }
    }

    /// The text content of the message, if any.
    #[napi(getter)]
    pub fn content(&self) -> Option<String> {
        self.inner.content.text_content()
    }
}

// ---------------------------------------------------------------------------
// Helper: build a JsCompletionResponse from the internal type
// ---------------------------------------------------------------------------

pub(crate) fn build_response(response: CompletionResponse) -> JsCompletionResponse {
    JsCompletionResponse {
        content: response.content,
        tool_calls: response
            .tool_calls
            .into_iter()
            .map(|tc| JsToolCall {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
            })
            .collect(),
        usage: response.usage.map(|u| JsTokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }),
        model: response.model,
        finish_reason: response.finish_reason,
        cost: response.cost,
        #[allow(clippy::cast_possible_wrap)]
        timing: response.timing.map(|t| JsRequestTiming {
            queue_ms: t.queue_ms.map(|v| v as i64),
            execution_ms: t.execution_ms.map(|v| v as i64),
            total_ms: t.total_ms.map(|v| v as i64),
        }),
        images: response
            .images
            .iter()
            .map(|img| serde_json::to_value(img).unwrap_or_default())
            .collect(),
        audio: response
            .audio
            .iter()
            .map(|a| serde_json::to_value(a).unwrap_or_default())
            .collect(),
        videos: response
            .videos
            .iter()
            .map(|v| serde_json::to_value(v).unwrap_or_default())
            .collect(),
        metadata: response.metadata,
    }
}

// ---------------------------------------------------------------------------
// CompletionModel wrapper
// ---------------------------------------------------------------------------

/// A chat completion model.
///
/// Use the static factory methods to create an instance for your provider:
///
/// ```javascript
/// const model = CompletionModel.openai("sk-...");
/// const response = await model.complete([
///   ChatMessage.user("What is 2 + 2?")
/// ]);
/// ```
#[napi(js_name = "CompletionModel")]
pub struct JsCompletionModel {
    pub(crate) inner: Arc<dyn CompletionModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompletionModel {
    // -----------------------------------------------------------------
    // Provider factory methods
    // -----------------------------------------------------------------

    /// Create an `OpenAI` completion model.
    #[napi(factory)]
    pub fn openai(api_key: String, model: Option<String>) -> Self {
        let mut provider = blazen_llm::providers::openai::OpenAiProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an Anthropic completion model.
    #[napi(factory)]
    pub fn anthropic(api_key: String, model: Option<String>) -> Self {
        let mut provider = blazen_llm::providers::anthropic::AnthropicProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Google Gemini completion model.
    #[napi(factory)]
    pub fn gemini(api_key: String, model: Option<String>) -> Self {
        let mut provider = blazen_llm::providers::gemini::GeminiProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an Azure `OpenAI` completion model.
    #[napi(factory)]
    pub fn azure(api_key: String, resource_name: String, deployment_name: String) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::azure::AzureOpenAiProvider::new(
                api_key,
                resource_name,
                deployment_name,
            )),
        }
    }

    /// Create a fal.ai completion model.
    #[napi(factory)]
    pub fn fal(api_key: String, model: Option<String>) -> Self {
        let mut provider = blazen_llm::providers::fal::FalProvider::new(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an `OpenRouter` completion model.
    #[napi(factory)]
    pub fn openrouter(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::openrouter(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Groq completion model.
    #[napi(factory)]
    pub fn groq(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::groq(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Together AI completion model.
    #[napi(factory)]
    pub fn together(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::together(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Mistral AI completion model.
    #[napi(factory)]
    pub fn mistral(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::mistral(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a `DeepSeek` completion model.
    #[napi(factory)]
    pub fn deepseek(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::deepseek(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Fireworks AI completion model.
    #[napi(factory)]
    pub fn fireworks(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::fireworks(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Perplexity completion model.
    #[napi(factory)]
    pub fn perplexity(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::perplexity(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an xAI (Grok) completion model.
    #[napi(factory)]
    pub fn xai(api_key: String, model: Option<String>) -> Self {
        let mut provider = blazen_llm::providers::openai_compat::OpenAiCompatProvider::xai(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create a Cohere completion model.
    #[napi(factory)]
    pub fn cohere(api_key: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::cohere(api_key);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Create an AWS Bedrock completion model.
    #[napi(factory)]
    pub fn bedrock(api_key: String, region: String, model: Option<String>) -> Self {
        let mut provider =
            blazen_llm::providers::openai_compat::OpenAiCompatProvider::bedrock(api_key, region);
        if let Some(m) = model {
            provider = provider.with_model(m);
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    // -----------------------------------------------------------------
    // Model configuration
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Completion methods
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    ///
    /// Messages should be an array of `ChatMessage` instances.
    ///
    /// Returns a typed response with `content`, `toolCalls`, `usage`, `model`,
    /// and `finishReason` fields.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        Ok(build_response(response))
    }

    /// Perform a chat completion with additional options.
    ///
    /// Options object may include:
    /// - `temperature` (number): Sampling temperature (0.0 - 2.0)
    /// - `maxTokens` (number): Maximum tokens to generate
    /// - `topP` (number): Nucleus sampling parameter
    /// - `model` (string): Override the default model
    /// - `tools` (array): Tool definitions for function calling
    #[napi(js_name = "completeWithOptions")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);

        // Apply options.
        if let Some(temp) = options.temperature {
            request.temperature = Some(temp as f32);
        }
        if let Some(max) = options.max_tokens {
            request.max_tokens = Some(max as u32);
        }
        if let Some(top_p) = options.top_p {
            request.top_p = Some(top_p as f32);
        }
        if let Some(model) = options.model {
            request.model = Some(model);
        }
        if let Some(tools) = options.tools {
            request.tools = tools
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect();
        }

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        Ok(build_response(response))
    }

    /// Stream a chat completion.
    ///
    /// The `onChunk` callback receives each chunk as it arrives, with keys:
    /// `delta`, `finishReason`, `toolCalls`.
    ///
    /// ```javascript
    /// await model.stream(
    ///   [ChatMessage.user("Tell me a story")],
    ///   (chunk) => { if (chunk.delta) process.stdout.write(chunk.delta); }
    /// );
    /// ```
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);

        let stream = self
            .inner
            .stream(request)
            .await
            .map_err(llm_error_to_napi)?;

        let mut stream = std::pin::pin!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    let chunk_json = serde_json::json!({
                        "delta": chunk.delta,
                        "finishReason": chunk.finish_reason,
                        "toolCalls": chunk.tool_calls.iter().map(|tc| {
                            serde_json::json!({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
                        }).collect::<Vec<_>>(),
                    });
                    on_chunk.call(chunk_json, ThreadsafeFunctionCallMode::Blocking);
                }
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }

    /// Stream a chat completion with additional options.
    ///
    /// Options object may include:
    /// - `temperature` (number): Sampling temperature (0.0 - 2.0)
    /// - `maxTokens` (number): Maximum tokens to generate
    /// - `topP` (number): Nucleus sampling parameter
    /// - `model` (string): Override the default model
    /// - `tools` (array): Tool definitions for function calling
    #[napi(js_name = "streamWithOptions")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsCompletionOptions,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);

        if let Some(temp) = options.temperature {
            request.temperature = Some(temp as f32);
        }
        if let Some(max) = options.max_tokens {
            request.max_tokens = Some(max as u32);
        }
        if let Some(top_p) = options.top_p {
            request.top_p = Some(top_p as f32);
        }
        if let Some(model) = options.model {
            request.model = Some(model);
        }
        if let Some(tools) = options.tools {
            request.tools = tools
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect();
        }

        let stream = self
            .inner
            .stream(request)
            .await
            .map_err(llm_error_to_napi)?;

        let mut stream = std::pin::pin!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    let chunk_json = serde_json::json!({
                        "delta": chunk.delta,
                        "finishReason": chunk.finish_reason,
                        "toolCalls": chunk.tool_calls.iter().map(|tc| {
                            serde_json::json!({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
                        }).collect::<Vec<_>>(),
                    });
                    on_chunk.call(chunk_json, ThreadsafeFunctionCallMode::Blocking);
                }
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a role string into a [`Role`], returning a napi error on invalid input.
fn parse_role(role_str: &str) -> Result<Role> {
    match role_str {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Invalid role: \"{role_str}\". Must be one of: system, user, assistant, tool"),
        )),
    }
}

/// Convert a `Vec<JsContentPart>` into `Vec<ContentPart>`.
fn convert_js_parts(parts: Vec<JsContentPart>) -> Result<Vec<ContentPart>> {
    parts
        .into_iter()
        .map(|part| match part.part_type.as_str() {
            "text" => Ok(ContentPart::Text {
                text: part.text.unwrap_or_default(),
            }),
            "image" => {
                let img = part.image.ok_or_else(|| {
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        "Content part with partType \"image\" must include an `image` field",
                    )
                })?;
                let source = match img.source.source_type.as_str() {
                    "url" => ImageSource::Url {
                        url: img.source.url.unwrap_or_default(),
                    },
                    "base64" => ImageSource::Base64 {
                        data: img.source.data.unwrap_or_default(),
                    },
                    other => {
                        return Err(napi::Error::new(
                            napi::Status::InvalidArg,
                            format!(
                                "Invalid image source type: \"{other}\". Must be \"url\" or \"base64\""
                            ),
                        ))
                    }
                };
                Ok(ContentPart::Image(ImageContent {
                    source,
                    media_type: img.media_type,
                }))
            }
            other => Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!("Invalid content part type: \"{other}\". Must be \"text\" or \"image\""),
            )),
        })
        .collect()
}
