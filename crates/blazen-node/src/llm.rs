//! JavaScript wrappers for LLM completion models.
//!
//! Provides [`JsCompletionModel`] with factory constructors for each
//! supported provider (`OpenAI`, Anthropic, Gemini, etc.).

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::CompletionModel;
use blazen_llm::types::{ChatMessage, CompletionRequest, MessageContent, Role};

use crate::error::llm_error_to_napi;

/// A chat completion model.
///
/// Use the static factory methods to create an instance for your provider:
///
/// ```javascript
/// const model = CompletionModel.openai("sk-...");
/// const response = await model.complete([
///   { role: "user", content: "What is 2 + 2?" }
/// ]);
/// ```
#[napi(js_name = "CompletionModel")]
pub struct JsCompletionModel {
    inner: Arc<dyn CompletionModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsCompletionModel {
    // -----------------------------------------------------------------
    // Provider factory methods
    // -----------------------------------------------------------------

    /// Create an `OpenAI` completion model.
    #[napi(factory)]
    pub fn openai(api_key: String) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::openai::OpenAiProvider::new(api_key)),
        }
    }

    /// Create an Anthropic completion model.
    #[napi(factory)]
    pub fn anthropic(api_key: String) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::anthropic::AnthropicProvider::new(
                api_key,
            )),
        }
    }

    /// Create a Google Gemini completion model.
    #[napi(factory)]
    pub fn gemini(api_key: String) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::gemini::GeminiProvider::new(api_key)),
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
    pub fn fal(api_key: String) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::fal::FalProvider::new(api_key)),
        }
    }

    /// Create an `OpenRouter` completion model.
    #[napi(factory)]
    pub fn openrouter(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::openrouter(api_key),
            ),
        }
    }

    /// Create a Groq completion model.
    #[napi(factory)]
    pub fn groq(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::groq(api_key),
            ),
        }
    }

    /// Create a Together AI completion model.
    #[napi(factory)]
    pub fn together(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::together(api_key),
            ),
        }
    }

    /// Create a Mistral AI completion model.
    #[napi(factory)]
    pub fn mistral(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::mistral(api_key),
            ),
        }
    }

    /// Create a `DeepSeek` completion model.
    #[napi(factory)]
    pub fn deepseek(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::deepseek(api_key),
            ),
        }
    }

    /// Create a Fireworks AI completion model.
    #[napi(factory)]
    pub fn fireworks(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::fireworks(api_key),
            ),
        }
    }

    /// Create a Perplexity completion model.
    #[napi(factory)]
    pub fn perplexity(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::perplexity(api_key),
            ),
        }
    }

    /// Create an xAI (Grok) completion model.
    #[napi(factory)]
    pub fn xai(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::xai(api_key),
            ),
        }
    }

    /// Create a Cohere completion model.
    #[napi(factory)]
    pub fn cohere(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::cohere(api_key),
            ),
        }
    }

    /// Create an AWS Bedrock completion model.
    #[napi(factory)]
    pub fn bedrock(api_key: String, region: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatProvider::bedrock(
                    api_key, region,
                ),
            ),
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
    /// Messages should be an array of `{ role: string, content: string }` objects.
    ///
    /// Returns the response as a JSON object with `content`, `toolCalls`,
    /// `usage`, `model`, and `finishReason` fields.
    #[napi]
    pub async fn complete(&self, messages: Vec<serde_json::Value>) -> Result<serde_json::Value> {
        let chat_messages = parse_messages(&messages)?;
        let request = CompletionRequest::new(chat_messages);

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        // Convert response to JSON.
        let tool_calls: Vec<serde_json::Value> = response
            .tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            })
            .collect();

        let usage = response.usage.as_ref().map(|u| {
            serde_json::json!({
                "promptTokens": u.prompt_tokens,
                "completionTokens": u.completion_tokens,
                "totalTokens": u.total_tokens,
            })
        });

        Ok(serde_json::json!({
            "content": response.content,
            "toolCalls": tool_calls,
            "usage": usage,
            "model": response.model,
            "finishReason": response.finish_reason,
        }))
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
    #[allow(clippy::cast_possible_truncation)]
    pub async fn complete_with_options(
        &self,
        messages: Vec<serde_json::Value>,
        options: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let chat_messages = parse_messages(&messages)?;
        let mut request = CompletionRequest::new(chat_messages);

        // Apply options.
        if let Some(temp) = options
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
        {
            request.temperature = Some(temp as f32);
        }
        if let Some(max) = options.get("maxTokens").and_then(serde_json::Value::as_u64) {
            request.max_tokens = Some(max as u32);
        }
        if let Some(top_p) = options.get("topP").and_then(serde_json::Value::as_f64) {
            request.top_p = Some(top_p as f32);
        }
        if let Some(model) = options.get("model").and_then(|v| v.as_str()) {
            request.model = Some(model.to_owned());
        }

        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        let tool_calls: Vec<serde_json::Value> = response
            .tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            })
            .collect();

        let usage = response.usage.as_ref().map(|u| {
            serde_json::json!({
                "promptTokens": u.prompt_tokens,
                "completionTokens": u.completion_tokens,
                "totalTokens": u.total_tokens,
            })
        });

        Ok(serde_json::json!({
            "content": response.content,
            "toolCalls": tool_calls,
            "usage": usage,
            "model": response.model,
            "finishReason": response.finish_reason,
        }))
    }
}

/// Parse JS message objects into [`ChatMessage`] values.
fn parse_messages(messages: &[serde_json::Value]) -> Result<Vec<ChatMessage>> {
    messages
        .iter()
        .map(|msg| {
            let role_str = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

            let content = msg
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();

            let role = match role_str {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => Role::User,
            };

            Ok(ChatMessage {
                role,
                content: MessageContent::Text(content),
            })
        })
        .collect()
}
