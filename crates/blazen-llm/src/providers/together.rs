//! Together AI provider -- fast open-source model inference.
//!
//! Uses the OpenAI-compatible API at `https://api.together.xyz/v1`.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use super::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use crate::error::BlazenError;
use crate::http::HttpClient;
use crate::retry::RetryConfig;
use crate::traits::{
    CompletionModel, ModelInfo, ModelRegistry, ProviderCapabilities, ProviderInfo,
};
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A Together AI chat completion provider.
///
/// Delegates to [`OpenAiCompatProvider`] with the Together AI base URL and
/// authentication pre-configured.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::together::TogetherProvider;
///
/// let provider = TogetherProvider::new("tok-...")
///     .with_model("meta-llama/Llama-3.3-70B-Instruct-Turbo");
/// ```
pub struct TogetherProvider {
    inner: OpenAiCompatProvider,
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
}

impl std::fmt::Debug for TogetherProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TogetherProvider")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl Clone for TogetherProvider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            retry_config: self.retry_config.clone(),
        }
    }
}

impl TogetherProvider {
    /// Create a new Together AI provider with the given API key.
    ///
    /// Uses the default HTTP client backend (reqwest on native, fetch on WASM).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatProvider::new(OpenAiCompatConfig {
                provider_name: "together".into(),
                base_url: "https://api.together.xyz/v1".into(),
                api_key: api_key.into(),
                default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo".into(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: true,
            }),
            retry_config: None,
        }
    }

    /// Create a new Together AI provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            inner: OpenAiCompatProvider::new_with_client(
                OpenAiCompatConfig {
                    provider_name: "together".into(),
                    base_url: "https://api.together.xyz/v1".into(),
                    api_key: api_key.into(),
                    default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo".into(),
                    auth_method: AuthMethod::Bearer,
                    extra_headers: Vec::new(),
                    query_params: Vec::new(),
                    supports_model_listing: true,
                },
                client,
            ),
            retry_config: None,
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

    /// Return a clone of the underlying HTTP client.
    ///
    /// Escape hatch delegating to the wrapped
    /// [`OpenAiCompatProvider`]. Useful for issuing raw HTTP requests
    /// (custom headers, debugging, endpoints not yet covered by Blazen)
    /// while reusing the provider's connection pool and TLS config.
    #[must_use]
    pub fn http_client(&self) -> Arc<dyn HttpClient> {
        self.inner.http_client()
    }

    /// Set the provider-level default retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }
}

super::impl_simple_from_options!(TogetherProvider, "together", no_base_url);

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for TogetherProvider {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config.as_ref()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        Some(Self::http_client(self))
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
impl ModelRegistry for TogetherProvider {
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

impl ProviderInfo for TogetherProvider {
    fn provider_name(&self) -> &'static str {
        "together"
    }

    fn base_url(&self) -> &'static str {
        "https://api.together.xyz/v1"
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
    fn test_together_reasoning_or_citations_pass_through_compat() {
        // Verify the shared OaiMessage parses reasoning_content + citations in a single
        // fixture, exercising the openai_compat -> Together delegation path's wire types.
        let json_body = r#"{
            "id": "x",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-ai/DeepSeek-R1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thinking...",
                    "citations": [{"url": "https://together.ai", "title": "Together"}]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3}
            }
        }"#;
        let parsed: OaiResponse = serde_json::from_str(json_body).unwrap();
        let msg = &parsed.choices[0].message;
        assert_eq!(msg.reasoning_content.as_deref(), Some("thinking..."));
        assert_eq!(msg.citations.len(), 1);
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.completion_tokens_details.unwrap().reasoning_tokens, 3);
    }
}
