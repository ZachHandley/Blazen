//! Cohere provider -- enterprise language models.
//!
//! Uses the OpenAI-compatible API at `https://api.cohere.ai/compatibility/v1`.

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

/// A Cohere chat completion provider.
///
/// Delegates to [`OpenAiCompatProvider`] with the Cohere compatibility
/// endpoint and authentication pre-configured.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::cohere::CohereProvider;
///
/// let provider = CohereProvider::new("co-...")
///     .with_model("command-r-plus-08-2024");
/// ```
pub struct CohereProvider {
    inner: OpenAiCompatProvider,
}

impl std::fmt::Debug for CohereProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Clone for CohereProvider {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl CohereProvider {
    /// Create a new Cohere provider with the given API key.
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
                provider_name: "cohere".into(),
                base_url: "https://api.cohere.ai/compatibility/v1".into(),
                api_key: api_key.into(),
                default_model: "command-a-08-2025".into(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            }),
        }
    }

    /// Create a new Cohere provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            inner: OpenAiCompatProvider::new_with_client(
                OpenAiCompatConfig {
                    provider_name: "cohere".into(),
                    base_url: "https://api.cohere.ai/compatibility/v1".into(),
                    api_key: api_key.into(),
                    default_model: "command-a-08-2025".into(),
                    auth_method: AuthMethod::Bearer,
                    extra_headers: Vec::new(),
                    query_params: Vec::new(),
                    supports_model_listing: false,
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

super::impl_simple_from_options!(CohereProvider, no_base_url);

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for CohereProvider {
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
impl ModelRegistry for CohereProvider {
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

impl ProviderInfo for CohereProvider {
    fn provider_name(&self) -> &'static str {
        "cohere"
    }

    fn base_url(&self) -> &'static str {
        "https://api.cohere.ai/compatibility/v1"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: false,
            model_listing: false,
            embeddings: true,
        }
    }
}
