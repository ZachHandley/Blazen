//! AWS Bedrock provider -- via the Mantle OpenAI-compatible endpoint.
//!
//! Uses the OpenAI-compatible API at `https://bedrock-mantle.{region}.api.aws/v1`.

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

/// An AWS Bedrock chat completion provider (via Mantle).
///
/// Delegates to [`OpenAiCompatProvider`] with the Bedrock Mantle base URL
/// and authentication pre-configured.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::bedrock::BedrockProvider;
///
/// let provider = BedrockProvider::new("aws-key", "us-east-1")
///     .with_model("anthropic.claude-sonnet-4-20250514-v1:0");
/// ```
pub struct BedrockProvider {
    inner: OpenAiCompatProvider,
    retry_config: Option<Arc<RetryConfig>>,
    base_url: String,
}

impl std::fmt::Debug for BedrockProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockProvider")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl Clone for BedrockProvider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            retry_config: self.retry_config.clone(),
            base_url: self.base_url.clone(),
        }
    }
}

impl BedrockProvider {
    /// Create a new Bedrock provider with the given API key and AWS region.
    ///
    /// Uses the default HTTP client backend (reqwest on native, fetch on WASM).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>, region: impl Into<String>) -> Self {
        let region = region.into();
        let base_url = format!("https://bedrock-mantle.{region}.api.aws/v1");
        Self {
            inner: OpenAiCompatProvider::new(OpenAiCompatConfig {
                provider_name: "bedrock".into(),
                base_url: base_url.clone(),
                api_key: api_key.into(),
                default_model: "anthropic.claude-sonnet-4-5-20250929-v1:0".into(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: true,
            }),
            retry_config: None,
            base_url,
        }
    }

    /// Create a new Bedrock provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(
        api_key: impl Into<String>,
        region: impl Into<String>,
        client: Arc<dyn HttpClient>,
    ) -> Self {
        let region = region.into();
        let base_url = format!("https://bedrock-mantle.{region}.api.aws/v1");
        Self {
            inner: OpenAiCompatProvider::new_with_client(
                OpenAiCompatConfig {
                    provider_name: "bedrock".into(),
                    base_url: base_url.clone(),
                    api_key: api_key.into(),
                    default_model: "anthropic.claude-sonnet-4-5-20250929-v1:0".into(),
                    auth_method: AuthMethod::Bearer,
                    extra_headers: Vec::new(),
                    query_params: Vec::new(),
                    supports_model_listing: true,
                },
                client,
            ),
            retry_config: None,
            base_url,
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

    /// Configure retry behavior for this provider.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Construct from typed [`BedrockOptions`](crate::types::provider_options::BedrockOptions).
    ///
    /// `opts.base.base_url` is ignored — the Bedrock URL is derived from
    /// `region`.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    /// # Errors
    ///
    /// Returns [`BlazenError::Auth`] if no API key is provided and the
    /// `AWS_ACCESS_KEY_ID` environment variable is not set.
    pub fn from_options(
        opts: crate::types::provider_options::BedrockOptions,
    ) -> Result<Self, crate::BlazenError> {
        let api_key = crate::keys::resolve_api_key("bedrock", opts.base.api_key)?;
        let mut provider = Self::new(api_key, &opts.region);
        if let Some(m) = opts.base.model {
            provider = provider.with_model(m);
        }
        Ok(provider)
    }
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for BedrockProvider {
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
impl ModelRegistry for BedrockProvider {
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

impl ProviderInfo for BedrockProvider {
    fn provider_name(&self) -> &'static str {
        "bedrock"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: true,
            model_listing: true,
            embeddings: false,
        }
    }
}
