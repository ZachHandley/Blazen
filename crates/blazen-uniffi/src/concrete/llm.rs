//! Per-engine LLM `#[uniffi::Object]` providers.
//!
//! Each `<Engine>Provider` here is a thin UniFFI-exported wrapper around
//! the matching canonical concrete provider in
//! [`blazen_llm::providers`]. Foreign bindgen (Go / Swift / Kotlin /
//! Ruby) emits a real class per engine — `OpenAiProvider`,
//! `AnthropicProvider`, `GeminiProvider`, `AzureOpenAiProvider`,
//! `BedrockProvider`, `MistralProvider`, `FireworksProvider`,
//! `DeepSeekProvider`, `PerplexityProvider`, `TogetherProvider`,
//! `GroqProvider`, `OpenRouterProvider`, `CohereProvider`,
//! `XaiProvider`, `FalLlmProvider` — rather than overloading the central
//! `Model` opaque factory in [`crate::llm`].
//!
//! Completion routes through
//! [`blazen_llm::providers::capabilities::LLMProvider::complete`] using
//! the wire-format [`crate::llm::ModelRequest`] / [`crate::llm::ModelResponse`]
//! conversions defined in [`crate::llm`].
//!
//! ## Naming
//!
//! `blazen_llm::providers::fal::FalProvider` is the LLM-capable
//! Fal client; the per-capability concretes (`FalTtsProvider`,
//! `FalSttProvider`, `FalMusicProvider`, `FalVcProvider`,
//! `FalImageGenProvider`, ...) wrap the same upstream type for their
//! respective capability surfaces. The uniffi LLM wrapper here is named
//! [`FalLlmProvider`] to disambiguate at the binding-surface layer.

#![allow(unused_imports)]

use std::sync::Arc;

use crate::errors::BlazenError;
use crate::llm::{ModelRequest, ModelResponse};

// ---------------------------------------------------------------------------
// OpenAiProvider — chat / completion against api.openai.com (or compat)
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::openai::OpenAiProvider`].
#[derive(uniffi::Object)]
pub struct OpenAiProvider {
    inner: Arc<blazen_llm::providers::openai::OpenAiProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl OpenAiProvider {
    /// Construct a new OpenAI provider.
    ///
    /// `api_key` — bearer token. `model` overrides the provider's
    /// default chat model when `Some`. `base_url` overrides the official
    /// `https://api.openai.com/v1` endpoint (for proxies / local
    /// OpenAI-compatible servers).
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>, base_url: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::openai::OpenAiProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        if let Some(url) = base_url {
            inner = inner.with_base_url(url);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl OpenAiProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// AnthropicProvider — Claude family via api.anthropic.com
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::anthropic::AnthropicProvider`].
#[derive(uniffi::Object)]
pub struct AnthropicProvider {
    inner: Arc<blazen_llm::providers::anthropic::AnthropicProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl AnthropicProvider {
    /// Construct a new Anthropic provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>, base_url: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::anthropic::AnthropicProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        if let Some(url) = base_url {
            inner = inner.with_base_url(url);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl AnthropicProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// GeminiProvider — Google Gemini via generativelanguage.googleapis.com
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::gemini::GeminiProvider`].
#[derive(uniffi::Object)]
pub struct GeminiProvider {
    inner: Arc<blazen_llm::providers::gemini::GeminiProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl GeminiProvider {
    /// Construct a new Gemini provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>, base_url: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::gemini::GeminiProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        if let Some(url) = base_url {
            inner = inner.with_base_url(url);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl GeminiProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// AzureOpenAiProvider — Azure OpenAI deployment (resource + deployment name)
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::azure::AzureOpenAiProvider`].
///
/// Azure determines the underlying model from the `deployment_name`
/// (and the URL from `resource_name` + `deployment_name`), so this
/// concrete intentionally does NOT expose a separate `model` argument.
#[derive(uniffi::Object)]
pub struct AzureOpenAiProvider {
    inner: Arc<blazen_llm::providers::azure::AzureOpenAiProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl AzureOpenAiProvider {
    /// Construct a new Azure OpenAI provider.
    ///
    /// `resource_name` is the Azure resource (e.g. `"my-resource"`) that
    /// forms part of the URL host (`<resource>.openai.azure.com`).
    /// `deployment_name` is the Azure deployment id — this doubles as
    /// the model selector (Azure routes by deployment, not model name).
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, resource_name: String, deployment_name: String) -> Arc<Self> {
        let inner = blazen_llm::providers::azure::AzureOpenAiProvider::new(
            api_key,
            resource_name,
            deployment_name,
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl AzureOpenAiProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// BedrockProvider — AWS Bedrock (region-scoped)
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::bedrock::BedrockProvider`].
#[derive(uniffi::Object)]
pub struct BedrockProvider {
    inner: Arc<blazen_llm::providers::bedrock::BedrockProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl BedrockProvider {
    /// Construct a new Bedrock provider.
    ///
    /// `region` is the AWS region (e.g. `"us-east-1"`).
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, region: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::bedrock::BedrockProvider::new(api_key, region);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl BedrockProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// MistralProvider — Mistral AI
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::mistral::MistralProvider`].
#[derive(uniffi::Object)]
pub struct MistralProvider {
    inner: Arc<blazen_llm::providers::mistral::MistralProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl MistralProvider {
    /// Construct a new Mistral provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::mistral::MistralProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl MistralProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// FireworksProvider — Fireworks AI
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::fireworks::FireworksProvider`].
#[derive(uniffi::Object)]
pub struct FireworksProvider {
    inner: Arc<blazen_llm::providers::fireworks::FireworksProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FireworksProvider {
    /// Construct a new Fireworks provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::fireworks::FireworksProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl FireworksProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// DeepSeekProvider — DeepSeek
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::deepseek::DeepSeekProvider`].
#[derive(uniffi::Object)]
pub struct DeepSeekProvider {
    inner: Arc<blazen_llm::providers::deepseek::DeepSeekProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl DeepSeekProvider {
    /// Construct a new DeepSeek provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::deepseek::DeepSeekProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl DeepSeekProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// PerplexityProvider — Perplexity
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::perplexity::PerplexityProvider`].
#[derive(uniffi::Object)]
pub struct PerplexityProvider {
    inner: Arc<blazen_llm::providers::perplexity::PerplexityProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl PerplexityProvider {
    /// Construct a new Perplexity provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::perplexity::PerplexityProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl PerplexityProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// TogetherProvider — Together AI
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::together::TogetherProvider`].
#[derive(uniffi::Object)]
pub struct TogetherProvider {
    inner: Arc<blazen_llm::providers::together::TogetherProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl TogetherProvider {
    /// Construct a new Together provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::together::TogetherProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl TogetherProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// GroqProvider — Groq
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::groq::GroqProvider`].
#[derive(uniffi::Object)]
pub struct GroqProvider {
    inner: Arc<blazen_llm::providers::groq::GroqProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl GroqProvider {
    /// Construct a new Groq provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::groq::GroqProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl GroqProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// OpenRouterProvider — OpenRouter
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::openrouter::OpenRouterProvider`].
#[derive(uniffi::Object)]
pub struct OpenRouterProvider {
    inner: Arc<blazen_llm::providers::openrouter::OpenRouterProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl OpenRouterProvider {
    /// Construct a new OpenRouter provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::openrouter::OpenRouterProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl OpenRouterProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// CohereProvider — Cohere
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::cohere::CohereProvider`].
#[derive(uniffi::Object)]
pub struct CohereProvider {
    inner: Arc<blazen_llm::providers::cohere::CohereProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl CohereProvider {
    /// Construct a new Cohere provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::cohere::CohereProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl CohereProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// XaiProvider — xAI (Grok)
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::xai::XaiProvider`].
#[derive(uniffi::Object)]
pub struct XaiProvider {
    inner: Arc<blazen_llm::providers::xai::XaiProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl XaiProvider {
    /// Construct a new xAI provider.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::xai::XaiProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_model(m);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl XaiProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// FalLlmProvider — fal.ai LLM endpoints
// ---------------------------------------------------------------------------

/// Concrete LLM provider wrapping
/// [`blazen_llm::providers::fal::FalProvider`].
///
/// Named `FalLlmProvider` (not `FalProvider`) at the binding-surface
/// layer to disambiguate from the per-capability `FalTtsProvider` /
/// `FalSttProvider` / `FalMusicProvider` / `FalVcProvider` /
/// `FalImageGenProvider` concretes. The Rust-side upstream type retains
/// its original name (`blazen_llm::providers::fal::FalProvider`).
///
/// `base_url` overrides the queue base URL
/// (default `https://queue.fal.run`) — used for proxies / staging
/// environments.
#[derive(uniffi::Object)]
pub struct FalLlmProvider {
    inner: Arc<blazen_llm::providers::fal::FalProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalLlmProvider {
    /// Construct a new fal.ai LLM provider.
    ///
    /// `model` is the underlying LLM model id sent in the request body
    /// (e.g. `"anthropic/claude-sonnet-4.5"`, `"openai/gpt-4o"`).
    /// `base_url` overrides the default queue URL.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String, model: Option<String>, base_url: Option<String>) -> Arc<Self> {
        let mut inner = blazen_llm::providers::fal::FalProvider::new(api_key);
        if let Some(m) = model {
            inner = inner.with_llm_model(m);
        }
        if let Some(url) = base_url {
            inner = inner.with_base_url(url);
        }
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Perform a non-streaming chat / completion request.
    pub async fn complete(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

#[uniffi::export]
impl FalLlmProvider {
    /// Synchronous variant of [`complete`](Self::complete).
    pub fn complete_blocking(
        self: Arc<Self>,
        request: ModelRequest,
    ) -> Result<ModelResponse, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime().block_on(async move { this.complete(request).await })
    }
}

// ---------------------------------------------------------------------------
// Polymorphic capability-base trait impls
// ---------------------------------------------------------------------------
//
// Each engine implements both [`crate::concrete::bases::BaseProvider`] and
// [`crate::concrete::bases::LlmProvider`] so foreign (Kotlin/Swift/Go)
// consumers can hold a polymorphic `LlmProvider` reference and Rust-side
// code can collect engines into capability-erased
// `Arc<dyn BaseProvider>` containers. The inherent `complete` methods
// on each engine (which use `self: Arc<Self>`) continue to take
// precedence at the call site `engine.complete(...)`; the trait
// methods (which use `&self`) are reachable via UFCS / `dyn LlmProvider`
// dispatch.

// OpenAiProvider -----------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for OpenAiProvider {
    fn provider_id(&self) -> String {
        "openai".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for OpenAiProvider {
    fn provider_id(&self) -> String {
        "openai".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// AnthropicProvider --------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for AnthropicProvider {
    fn provider_id(&self) -> String {
        "anthropic".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for AnthropicProvider {
    fn provider_id(&self) -> String {
        "anthropic".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// GeminiProvider -----------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for GeminiProvider {
    fn provider_id(&self) -> String {
        "gemini".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for GeminiProvider {
    fn provider_id(&self) -> String {
        "gemini".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// AzureOpenAiProvider ------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for AzureOpenAiProvider {
    fn provider_id(&self) -> String {
        "azure-openai".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for AzureOpenAiProvider {
    fn provider_id(&self) -> String {
        "azure-openai".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// BedrockProvider ----------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for BedrockProvider {
    fn provider_id(&self) -> String {
        "bedrock".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for BedrockProvider {
    fn provider_id(&self) -> String {
        "bedrock".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// MistralProvider ----------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for MistralProvider {
    fn provider_id(&self) -> String {
        "mistral".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for MistralProvider {
    fn provider_id(&self) -> String {
        "mistral".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// FireworksProvider --------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FireworksProvider {
    fn provider_id(&self) -> String {
        "fireworks".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for FireworksProvider {
    fn provider_id(&self) -> String {
        "fireworks".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// DeepSeekProvider ---------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for DeepSeekProvider {
    fn provider_id(&self) -> String {
        "deepseek".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for DeepSeekProvider {
    fn provider_id(&self) -> String {
        "deepseek".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// PerplexityProvider -------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for PerplexityProvider {
    fn provider_id(&self) -> String {
        "perplexity".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for PerplexityProvider {
    fn provider_id(&self) -> String {
        "perplexity".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// TogetherProvider ---------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for TogetherProvider {
    fn provider_id(&self) -> String {
        "together".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for TogetherProvider {
    fn provider_id(&self) -> String {
        "together".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// GroqProvider -------------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for GroqProvider {
    fn provider_id(&self) -> String {
        "groq".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for GroqProvider {
    fn provider_id(&self) -> String {
        "groq".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// OpenRouterProvider -------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for OpenRouterProvider {
    fn provider_id(&self) -> String {
        "openrouter".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for OpenRouterProvider {
    fn provider_id(&self) -> String {
        "openrouter".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// CohereProvider -----------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for CohereProvider {
    fn provider_id(&self) -> String {
        "cohere".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for CohereProvider {
    fn provider_id(&self) -> String {
        "cohere".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// XaiProvider --------------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for XaiProvider {
    fn provider_id(&self) -> String {
        "xai".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for XaiProvider {
    fn provider_id(&self) -> String {
        "xai".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// FalLlmProvider -----------------------------------------------------------

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FalLlmProvider {
    fn provider_id(&self) -> String {
        "fal".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::LlmProvider for FalLlmProvider {
    fn provider_id(&self) -> String {
        "fal".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Llm
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        use blazen_llm::providers::capabilities::LLMProvider as _;
        let core_req: blazen_llm::types::ModelRequest = request.try_into()?;
        let core_res = self.inner.complete(core_req).await?;
        Ok(core_res.into())
    }
}

// ---------------------------------------------------------------------------
// Deferred: OpenAiCompatProvider
// ---------------------------------------------------------------------------
//
// `blazen_llm::providers::openai_compat::OpenAiCompatProvider`'s
// constructor takes an `OpenAiCompatConfig` struct that carries an
// `AuthMethod` enum (Bearer / ApiKey / Custom-header-named) and a
// `Vec<(String, String)>` of custom HTTP headers. Neither maps cleanly
// to a flat uniffi `Record` (the `AuthMethod` enum has variant fields
// the foreign-bindgen surface can't express ergonomically, and the
// header tuple-vec inflates the FFI surface with marshalling overhead).
//
// Deferring this concrete until a follow-up sub-task lifts a typed
// `OpenAiCompatOptions` UDL record. Users that need OpenAI-compatible
// endpoints today can point `OpenAiProvider::new(..., base_url=Some(...))`
// at their proxy — that covers the bearer-auth case.

// ---------------------------------------------------------------------------
// Deferred: streaming surface
// ---------------------------------------------------------------------------
//
// The polymorphic [`crate::concrete::bases::LlmProvider`] trait only
// declares `complete` (non-streaming). Streaming completion continues
// to flow through the existing [`crate::provider_custom::CustomProvider`]
// foreign-callback surface; engine-specific streaming (when it lands as
// inherent methods on the concretes here) will be exposed in a later
// sub-task without changing this trait shape.
