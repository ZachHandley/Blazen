//! Perplexity AI provider -- search-augmented generation.
//!
//! Uses the OpenAI-compatible API at `https://api.perplexity.ai`.

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

/// A Perplexity AI chat completion provider.
///
/// Delegates to [`OpenAiCompatProvider`] with the Perplexity base URL and
/// authentication pre-configured.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::perplexity::PerplexityProvider;
///
/// let provider = PerplexityProvider::new("pplx-...")
///     .with_model("sonar");
/// ```
pub struct PerplexityProvider {
    inner: OpenAiCompatProvider,
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
}

impl std::fmt::Debug for PerplexityProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerplexityProvider")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl Clone for PerplexityProvider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            retry_config: self.retry_config.clone(),
        }
    }
}

impl PerplexityProvider {
    /// Create a new Perplexity provider with the given API key.
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
                provider_name: "perplexity".into(),
                base_url: "https://api.perplexity.ai".into(),
                api_key: api_key.into(),
                default_model: "sonar-pro".into(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            }),
            retry_config: None,
        }
    }

    /// Create a new Perplexity provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            inner: OpenAiCompatProvider::new_with_client(
                OpenAiCompatConfig {
                    provider_name: "perplexity".into(),
                    base_url: "https://api.perplexity.ai".into(),
                    api_key: api_key.into(),
                    default_model: "sonar-pro".into(),
                    auth_method: AuthMethod::Bearer,
                    extra_headers: Vec::new(),
                    query_params: Vec::new(),
                    supports_model_listing: false,
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

super::impl_simple_from_options!(PerplexityProvider, "perplexity", no_base_url);

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for PerplexityProvider {
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
impl ModelRegistry for PerplexityProvider {
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

impl ProviderInfo for PerplexityProvider {
    fn provider_name(&self) -> &'static str {
        "perplexity"
    }

    fn base_url(&self) -> &'static str {
        "https://api.perplexity.ai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: false,
            structured_output: false,
            vision: false,
            model_listing: false,
            embeddings: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::providers::sse::OaiResponse;

    #[test]
    fn test_perplexity_citations_pass_through() {
        let json_body = r#"{
            "id": "x",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "sonar",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the answer per source A",
                    "citations": [
                        {"url": "https://example.com/a", "title": "Source A"},
                        {"url": "https://example.com/b"}
                    ]
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
        assert_eq!(msg.citations.len(), 2);
        assert_eq!(msg.citations[0]["url"], "https://example.com/a");
        assert_eq!(msg.citations[0]["title"], "Source A");
    }
}
