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
// Per-engine streaming free functions
// ---------------------------------------------------------------------------
//
// Each engine gets a `<engine>_complete_streaming` (+ `_blocking`) free
// function mirroring the central
// [`crate::streaming::complete_streaming`]. The polymorphic
// [`crate::concrete::bases::LlmProvider`] trait still only declares
// `complete` (non-streaming) — UniFFI cannot export generic functions and
// cannot dispatch `dyn` over an async-streaming trait method ergonomically
// across all four foreign languages, so streaming is exposed as a thin
// per-engine free function over the shared
// [`crate::streaming::drive_completion_stream`] helper instead. Each
// function converts the wire-format [`ModelRequest`], starts the engine's
// `stream()` (surfacing a failed *start* both to the caller and the sink),
// then delegates the drive loop to the shared helper.
//
// These are hand-written (not a declarative macro) on purpose: the
// `#[uniffi::export]` proc-macro reads pre-expansion tokens, so
// macro-generated exports never register in the UniFFI metadata.

use crate::streaming::{CompletionStreamSink, clone_error, drive_completion_stream};

/// Macro body shared by every per-engine async streaming wrapper. Not a
/// `macro_rules!` over the `#[uniffi::export]` items (those must be
/// hand-written so the proc-macro sees them) — this is an ordinary helper
/// invoked from inside each hand-written export.
async fn drive_provider_stream<P>(
    inner: &P,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError>
where
    P: blazen_llm::providers::capabilities::LLMProvider + ?Sized,
{
    let core_request: blazen_llm::types::ModelRequest = match request.try_into() {
        Ok(r) => r,
        Err(err) => {
            let _ = sink.on_error(clone_error(&err)).await;
            return Err(err);
        }
    };
    let stream = match inner.stream(core_request).await {
        Ok(s) => s,
        Err(err) => {
            // Surface a failed start both to the caller (return Err) and to
            // the sink, matching the central `complete_streaming` contract.
            let wire_err = BlazenError::from(err);
            let _ = sink.on_error(clone_error(&wire_err)).await;
            return Err(wire_err);
        }
    };
    drive_completion_stream(stream, sink).await
}

// OpenAiProvider -----------------------------------------------------------

/// Stream a chat / completion from [`OpenAiProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn openai_provider_complete_streaming(
    provider: Arc<OpenAiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`openai_provider_complete_streaming`].
#[uniffi::export]
pub fn openai_provider_complete_streaming_blocking(
    provider: Arc<OpenAiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(openai_provider_complete_streaming(provider, request, sink))
}

// AnthropicProvider --------------------------------------------------------

/// Stream a chat / completion from [`AnthropicProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn anthropic_provider_complete_streaming(
    provider: Arc<AnthropicProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`anthropic_provider_complete_streaming`].
#[uniffi::export]
pub fn anthropic_provider_complete_streaming_blocking(
    provider: Arc<AnthropicProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(anthropic_provider_complete_streaming(
        provider, request, sink,
    ))
}

// GeminiProvider -----------------------------------------------------------

/// Stream a chat / completion from [`GeminiProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn gemini_provider_complete_streaming(
    provider: Arc<GeminiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`gemini_provider_complete_streaming`].
#[uniffi::export]
pub fn gemini_provider_complete_streaming_blocking(
    provider: Arc<GeminiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(gemini_provider_complete_streaming(provider, request, sink))
}

// AzureOpenAiProvider ------------------------------------------------------

/// Stream a chat / completion from [`AzureOpenAiProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn azure_openai_provider_complete_streaming(
    provider: Arc<AzureOpenAiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`azure_openai_provider_complete_streaming`].
#[uniffi::export]
pub fn azure_openai_provider_complete_streaming_blocking(
    provider: Arc<AzureOpenAiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(azure_openai_provider_complete_streaming(
        provider, request, sink,
    ))
}

// BedrockProvider ----------------------------------------------------------

/// Stream a chat / completion from [`BedrockProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn bedrock_provider_complete_streaming(
    provider: Arc<BedrockProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`bedrock_provider_complete_streaming`].
#[uniffi::export]
pub fn bedrock_provider_complete_streaming_blocking(
    provider: Arc<BedrockProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(bedrock_provider_complete_streaming(provider, request, sink))
}

// MistralProvider ----------------------------------------------------------

/// Stream a chat / completion from [`MistralProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn mistral_provider_complete_streaming(
    provider: Arc<MistralProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`mistral_provider_complete_streaming`].
#[uniffi::export]
pub fn mistral_provider_complete_streaming_blocking(
    provider: Arc<MistralProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(mistral_provider_complete_streaming(provider, request, sink))
}

// FireworksProvider --------------------------------------------------------

/// Stream a chat / completion from [`FireworksProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn fireworks_provider_complete_streaming(
    provider: Arc<FireworksProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`fireworks_provider_complete_streaming`].
#[uniffi::export]
pub fn fireworks_provider_complete_streaming_blocking(
    provider: Arc<FireworksProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(fireworks_provider_complete_streaming(
        provider, request, sink,
    ))
}

// DeepSeekProvider ---------------------------------------------------------

/// Stream a chat / completion from [`DeepSeekProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn deepseek_provider_complete_streaming(
    provider: Arc<DeepSeekProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`deepseek_provider_complete_streaming`].
#[uniffi::export]
pub fn deepseek_provider_complete_streaming_blocking(
    provider: Arc<DeepSeekProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(deepseek_provider_complete_streaming(
        provider, request, sink,
    ))
}

// PerplexityProvider -------------------------------------------------------

/// Stream a chat / completion from [`PerplexityProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn perplexity_provider_complete_streaming(
    provider: Arc<PerplexityProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`perplexity_provider_complete_streaming`].
#[uniffi::export]
pub fn perplexity_provider_complete_streaming_blocking(
    provider: Arc<PerplexityProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(perplexity_provider_complete_streaming(
        provider, request, sink,
    ))
}

// TogetherProvider ---------------------------------------------------------

/// Stream a chat / completion from [`TogetherProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn together_provider_complete_streaming(
    provider: Arc<TogetherProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`together_provider_complete_streaming`].
#[uniffi::export]
pub fn together_provider_complete_streaming_blocking(
    provider: Arc<TogetherProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(together_provider_complete_streaming(
        provider, request, sink,
    ))
}

// GroqProvider -------------------------------------------------------------

/// Stream a chat / completion from [`GroqProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn groq_provider_complete_streaming(
    provider: Arc<GroqProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`groq_provider_complete_streaming`].
#[uniffi::export]
pub fn groq_provider_complete_streaming_blocking(
    provider: Arc<GroqProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(groq_provider_complete_streaming(provider, request, sink))
}

// OpenRouterProvider -------------------------------------------------------

/// Stream a chat / completion from [`OpenRouterProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn openrouter_provider_complete_streaming(
    provider: Arc<OpenRouterProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`openrouter_provider_complete_streaming`].
#[uniffi::export]
pub fn openrouter_provider_complete_streaming_blocking(
    provider: Arc<OpenRouterProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(openrouter_provider_complete_streaming(
        provider, request, sink,
    ))
}

// CohereProvider -----------------------------------------------------------

/// Stream a chat / completion from [`CohereProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn cohere_provider_complete_streaming(
    provider: Arc<CohereProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`cohere_provider_complete_streaming`].
#[uniffi::export]
pub fn cohere_provider_complete_streaming_blocking(
    provider: Arc<CohereProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(cohere_provider_complete_streaming(provider, request, sink))
}

// XaiProvider --------------------------------------------------------------

/// Stream a chat / completion from [`XaiProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn xai_provider_complete_streaming(
    provider: Arc<XaiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`xai_provider_complete_streaming`].
#[uniffi::export]
pub fn xai_provider_complete_streaming_blocking(
    provider: Arc<XaiProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(xai_provider_complete_streaming(provider, request, sink))
}

// FalLlmProvider -----------------------------------------------------------

/// Stream a chat / completion from [`FalLlmProvider`] into `sink`.
#[uniffi::export(async_runtime = "tokio")]
pub async fn fal_llm_provider_complete_streaming(
    provider: Arc<FalLlmProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    drive_provider_stream(provider.inner.as_ref(), request, sink).await
}

/// Synchronous variant of [`fal_llm_provider_complete_streaming`].
#[uniffi::export]
pub fn fal_llm_provider_complete_streaming_blocking(
    provider: Arc<FalLlmProvider>,
    request: ModelRequest,
    sink: Arc<dyn CompletionStreamSink>,
) -> Result<(), BlazenError> {
    crate::runtime::runtime().block_on(fal_llm_provider_complete_streaming(provider, request, sink))
}
