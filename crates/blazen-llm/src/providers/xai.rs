//! xAI (Grok) provider -- advanced reasoning and multimodal capabilities.
//!
//! Uses the OpenAI-compatible API at `https://api.x.ai/v1`.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use super::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use crate::error::BlazenError;
use crate::http::HttpClient;
use crate::traits::{
    CompletionModel, ModelInfo, ModelRegistry, ProviderCapabilities, ProviderInfo,
};
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// An xAI (Grok) chat completion provider.
///
/// Delegates to [`OpenAiCompatProvider`] with the xAI base URL and
/// authentication pre-configured.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::xai::XaiProvider;
///
/// let provider = XaiProvider::new("xai-...")
///     .with_model("grok-3-mini");
/// ```
pub struct XaiProvider {
    inner: OpenAiCompatProvider,
}

impl std::fmt::Debug for XaiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XaiProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Clone for XaiProvider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl XaiProvider {
    /// Create a new xAI provider with the given API key.
    ///
    /// Uses the default HTTP client backend (reqwest on native, fetch on WASM).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatProvider::new(OpenAiCompatConfig {
                provider_name: "xai".into(),
                base_url: "https://api.x.ai/v1".into(),
                api_key: api_key.into(),
                default_model: "grok-3".into(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: true,
            }),
        }
    }

    /// Create a new xAI provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            inner: OpenAiCompatProvider::new_with_client(
                OpenAiCompatConfig {
                    provider_name: "xai".into(),
                    base_url: "https://api.x.ai/v1".into(),
                    api_key: api_key.into(),
                    default_model: "grok-3".into(),
                    auth_method: AuthMethod::Bearer,
                    extra_headers: Vec::new(),
                    query_params: Vec::new(),
                    supports_model_listing: true,
                },
                client,
            ),
        }
    }

    /// Override the default model for this provider instance.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.inner = self.inner.with_model(model);
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.inner = self.inner.with_http_client(client);
        self
    }
}

super::impl_simple_from_options!(XaiProvider, no_base_url);

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for XaiProvider {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.inner.stream(request).await
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ModelRegistry for XaiProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError> {
        self.inner.list_models().await
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, BlazenError> {
        self.inner.get_model(model_id).await
    }
}

// ---------------------------------------------------------------------------
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl ProviderInfo for XaiProvider {
    fn provider_name(&self) -> &'static str {
        "xai"
    }

    fn base_url(&self) -> &'static str {
        "https://api.x.ai/v1"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: true,
            model_listing: true,
            embeddings: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::providers::sse::OaiResponse;

    #[test]
    fn test_grok_reasoning_passes_through() {
        let json_body = r#"{
            "id": "x",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "grok-2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the answer is 42",
                    "reasoning": "step 1: consider...; step 2: calculate..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;
        let parsed: OaiResponse = serde_json::from_str(json_body).unwrap();
        let msg = &parsed.choices[0].message;
        assert_eq!(
            msg.reasoning.as_deref(),
            Some("step 1: consider...; step 2: calculate...")
        );
    }
}
