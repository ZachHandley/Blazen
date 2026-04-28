//! JavaScript bindings for the generic OpenAI-compatible provider.
//!
//! Exposes [`JsOpenAiCompatProvider`] for talking to any
//! OpenAI-compatible chat-completions endpoint, plus the
//! [`JsOpenAiCompatConfig`] options object and the [`JsAuthMethod`] enum
//! that describes how the API key is sent.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::CompletionModel;
use blazen_llm::providers::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::providers::completion_model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// AuthMethod enum
// ---------------------------------------------------------------------------

/// How to send the API key to an OpenAI-compatible endpoint.
#[napi(string_enum)]
pub enum JsAuthMethod {
    /// `Authorization: Bearer <key>` (`OpenAI`, `OpenRouter`, Groq, etc.).
    #[napi(value = "bearer")]
    Bearer,
    /// `api-key: <key>` (Azure `OpenAI`).
    #[napi(value = "azure-api-key")]
    AzureApiKey,
    /// `Authorization: Key <key>` (fal.ai).
    #[napi(value = "key-prefix")]
    KeyPrefix,
}

// ---------------------------------------------------------------------------
// OpenAiCompatConfig
// ---------------------------------------------------------------------------

/// Configuration for an OpenAI-compatible provider.
///
/// `customHeaderName` overrides `authMethod` and sends the key as
/// `<customHeaderName>: <apiKey>` (e.g. `x-api-key: sk-...`). This maps
/// to the Rust `AuthMethod::ApiKeyHeader` variant which can't be
/// represented as a plain string-enum value.
#[napi(object)]
pub struct JsOpenAiCompatConfig {
    /// A human-readable name for this provider (used in logs and model info).
    #[napi(js_name = "providerName")]
    pub provider_name: String,
    /// The base URL for the API (e.g. `https://api.example.com/v1`).
    #[napi(js_name = "baseUrl")]
    pub base_url: String,
    /// The API key.
    #[napi(js_name = "apiKey")]
    pub api_key: String,
    /// The default model to use if the request does not override it.
    #[napi(js_name = "defaultModel")]
    pub default_model: String,
    /// How to send the API key. Defaults to `Bearer` when omitted.
    #[napi(js_name = "authMethod")]
    pub auth_method: Option<JsAuthMethod>,
    /// Send the API key as a custom header instead of using `authMethod`.
    /// Mutually exclusive with `authMethod`.
    #[napi(js_name = "customHeaderName")]
    pub custom_header_name: Option<String>,
    /// Extra headers to include in every request, as `[name, value]` tuples.
    #[napi(js_name = "extraHeaders")]
    pub extra_headers: Option<Vec<Vec<String>>>,
    /// Query parameters to include in every request, as `[name, value]` tuples.
    #[napi(js_name = "queryParams")]
    pub query_params: Option<Vec<Vec<String>>>,
    /// Whether this provider supports the `/models` listing endpoint.
    /// Defaults to `false`.
    #[napi(js_name = "supportsModelListing")]
    pub supports_model_listing: Option<bool>,
}

impl From<JsOpenAiCompatConfig> for OpenAiCompatConfig {
    fn from(val: JsOpenAiCompatConfig) -> Self {
        let auth_method = if let Some(name) = val.custom_header_name {
            AuthMethod::ApiKeyHeader(name)
        } else {
            match val.auth_method.unwrap_or(JsAuthMethod::Bearer) {
                JsAuthMethod::Bearer => AuthMethod::Bearer,
                JsAuthMethod::AzureApiKey => AuthMethod::AzureApiKey,
                JsAuthMethod::KeyPrefix => AuthMethod::KeyPrefix,
            }
        };
        let convert_pairs = |opt: Option<Vec<Vec<String>>>| -> Vec<(String, String)> {
            opt.unwrap_or_default()
                .into_iter()
                .filter_map(|pair| {
                    let mut iter = pair.into_iter();
                    Some((iter.next()?, iter.next()?))
                })
                .collect()
        };
        Self {
            provider_name: val.provider_name,
            base_url: val.base_url,
            api_key: val.api_key,
            default_model: val.default_model,
            auth_method,
            extra_headers: convert_pairs(val.extra_headers),
            query_params: convert_pairs(val.query_params),
            supports_model_listing: val.supports_model_listing.unwrap_or(false),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAiCompatProvider
// ---------------------------------------------------------------------------

/// A generic OpenAI-compatible chat completion provider.
///
/// ```typescript
/// const compat = OpenAiCompatProvider.create({
///   providerName: "my-host",
///   baseUrl: "https://api.example.com/v1",
///   apiKey: "sk-...",
///   defaultModel: "my-model",
///   authMethod: "bearer",
/// });
/// ```
#[napi(js_name = "OpenAiCompatProvider")]
pub struct JsOpenAiCompatProvider {
    inner: Arc<OpenAiCompatProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl JsOpenAiCompatProvider {
    /// Create a new OpenAI-compatible provider.
    #[napi(factory)]
    pub fn create(config: JsOpenAiCompatConfig) -> Result<Self> {
        let cfg: OpenAiCompatConfig = config.into();
        Ok(Self {
            inner: Arc::new(OpenAiCompatProvider::new(cfg)),
        })
    }

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion.
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
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
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
        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(build_response(response))
    }

    /// Stream a chat completion.
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
                    on_chunk.call(
                        build_stream_chunk(chunk),
                        ThreadsafeFunctionCallMode::Blocking,
                    );
                }
                Err(e) => return Err(napi::Error::from_reason(e.to_string())),
            }
        }
        Ok(())
    }

    /// Stream a chat completion with additional options.
    #[napi(js_name = "streamWithOptions")]
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
                    on_chunk.call(
                        build_stream_chunk(chunk),
                        ThreadsafeFunctionCallMode::Blocking,
                    );
                }
                Err(e) => return Err(napi::Error::from_reason(e.to_string())),
            }
        }
        Ok(())
    }
}
