//! JavaScript wrapper for LLM completion models.
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
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::types::{JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response};

/// Stream callback: takes a `serde_json::Value` chunk, returns nothing meaningful.
/// `CalleeHandled = false` to avoid the error-first callback convention.
/// `Weak = true` so it does not prevent Node.js from exiting.
pub(crate) type StreamChunkCallbackTsfn =
    ThreadsafeFunction<serde_json::Value, Unknown<'static>, serde_json::Value, Status, false, true>;

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
        if let Some(fmt) = options.response_format {
            request = request.with_response_format(fmt);
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
        if let Some(fmt) = options.response_format {
            request = request.with_response_format(fmt);
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
