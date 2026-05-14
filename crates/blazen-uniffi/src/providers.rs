//! Per-provider factories for [`CompletionModel`] and [`EmbeddingModel`].
//!
//! Each `#[uniffi::export]` function in this module constructs a concrete
//! upstream provider, wraps it as `Arc<dyn blazen_llm::CompletionModel>` (or
//! `Arc<dyn blazen_llm::EmbeddingModel>`), and hands it back to the foreign
//! caller through the opaque [`CompletionModel`] / [`EmbeddingModel`] handles
//! defined in [`crate::llm`].
//!
//! ## Argument shape
//!
//! UniFFI doesn't model Rust's builder pattern cleanly across all four target
//! languages, so every factory exposes flat positional/keyword arguments
//! instead of a builder struct. Cloud providers all take:
//!
//! - `api_key: String` — may be empty when the provider resolves it from an
//!   environment variable (see `blazen_llm::keys::resolve_api_key`).
//! - `model: Option<String>` — overrides the provider's default model id.
//! - `base_url: Option<String>` — overrides the provider's base URL (useful
//!   for local proxies / staging endpoints; ignored by providers whose URL
//!   shape is fixed, like Azure / Bedrock).
//!
//! Provider-specific knobs (Azure `resource_name`/`deployment_name`, Bedrock
//! `region`, fal endpoint kind, etc.) appear as extra positional arguments
//! before the common `model`/`base_url` tail.
//!
//! ## Local backends
//!
//! Local in-process providers (`mistralrs`, `llamacpp`, `candle-llm`,
//! `candle-embed`) are gated behind the matching cargo features. Their
//! factories accept a `model_path_or_id` plus a handful of frequently-used
//! knobs (device, quantization, context length); less-common knobs default
//! to the underlying provider's defaults. Callers who need full control
//! should drop down to the Rust crate directly (the UniFFI surface
//! intentionally trims the option matrix to keep the foreign API
//! ergonomic).
//!
//! ## Errors
//!
//! Every factory returns [`BlazenResult`] — provider construction can fail
//! when an API key is missing, an option struct is invalid, or a local
//! model fails to load. Errors flow through [`BlazenError`]'s `From` impls
//! and surface as the canonical UniFFI error variants
//! (`Auth` / `Validation` / `Provider` / ...).

use std::sync::Arc;

use blazen_llm::CompletionModel as CoreCompletionModel;
use blazen_llm::EmbeddingModel as CoreEmbeddingModel;
use blazen_llm::types::provider_options::{
    AzureOptions, BedrockOptions, FalLlmEndpointKind, FalOptions, ProviderOptions,
};

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::{CompletionModel, EmbeddingModel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [`ProviderOptions`] from the common cloud-provider argument tuple.
///
/// An empty `api_key` is normalised to `None` so the provider falls back to
/// the well-known environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
/// ...) via `blazen_llm::keys::resolve_api_key`.
fn provider_options(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> ProviderOptions {
    ProviderOptions {
        api_key: if api_key.is_empty() {
            None
        } else {
            Some(api_key)
        },
        model,
        base_url,
    }
}

// ---------------------------------------------------------------------------
// Cloud LLM providers — CompletionModel factories
// ---------------------------------------------------------------------------

/// Build an `OpenAI` chat-completion model.
///
/// `base_url` defaults to `https://api.openai.com/v1`; override it to target
/// any OpenAI-compatible proxy that uses the official-OpenAI request shape.
#[uniffi::export]
pub fn new_openai_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::openai::OpenAiProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build an Anthropic Messages-API chat-completion model.
#[uniffi::export]
pub fn new_anthropic_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::anthropic::AnthropicProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Google Gemini chat-completion model.
#[uniffi::export]
pub fn new_gemini_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::gemini::GeminiProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build an Azure `OpenAI` chat-completion model.
///
/// Azure derives its endpoint from `resource_name` + `deployment_name` and
/// its model id from `deployment_name`, so `base_url` is intentionally not
/// exposed here. `api_version` defaults to the provider's pinned API
/// version when `None`.
#[uniffi::export]
pub fn new_azure_completion_model(
    api_key: String,
    resource_name: String,
    deployment_name: String,
    api_version: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = AzureOptions {
        base: provider_options(api_key, None, None),
        resource_name,
        deployment_name,
        api_version,
    };
    let provider = blazen_llm::providers::azure::AzureOpenAiProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build an AWS Bedrock chat-completion model.
///
/// `region` selects the AWS region (e.g. `"us-east-1"`); `api_key` is the
/// Bedrock API key (which can be obtained via `aws bedrock` IAM keys or
/// passed as an empty string to resolve from `AWS_BEARER_TOKEN_BEDROCK`).
#[uniffi::export]
pub fn new_bedrock_completion_model(
    api_key: String,
    region: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = BedrockOptions {
        base: provider_options(api_key, model, base_url),
        region,
    };
    let provider = blazen_llm::providers::bedrock::BedrockProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build an `OpenRouter` chat-completion model.
#[uniffi::export]
pub fn new_openrouter_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::openrouter::OpenRouterProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Groq chat-completion model.
#[uniffi::export]
pub fn new_groq_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider =
        blazen_llm::providers::groq::GroqProvider::from_options(opts).map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Together AI chat-completion model.
#[uniffi::export]
pub fn new_together_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::together::TogetherProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Mistral chat-completion model.
#[uniffi::export]
pub fn new_mistral_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::mistral::MistralProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a `DeepSeek` chat-completion model.
#[uniffi::export]
pub fn new_deepseek_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::deepseek::DeepSeekProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Fireworks AI chat-completion model.
#[uniffi::export]
pub fn new_fireworks_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::fireworks::FireworksProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Perplexity chat-completion model.
#[uniffi::export]
pub fn new_perplexity_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::perplexity::PerplexityProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build an xAI (Grok) chat-completion model.
#[uniffi::export]
pub fn new_xai_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider =
        blazen_llm::providers::xai::XaiProvider::from_options(opts).map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a Cohere chat-completion model.
#[uniffi::export]
pub fn new_cohere_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = provider_options(api_key, model, base_url);
    let provider = blazen_llm::providers::cohere::CohereProvider::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a fal.ai chat-completion model.
///
/// `endpoint` selects the fal endpoint family — one of
/// `"openai_chat"` (default), `"openai_responses"`, `"openai_embeddings"`,
/// `"openrouter"`, `"any_llm"`. Unrecognised values fall back to
/// `OpenAiChat`. `enterprise` promotes the endpoint to its enterprise /
/// SOC2-eligible variant; `auto_route_modality` toggles automatic routing
/// to a vision/audio/video endpoint when the request carries media.
#[uniffi::export]
pub fn new_fal_completion_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
    endpoint: Option<String>,
    enterprise: bool,
    auto_route_modality: bool,
) -> BlazenResult<Arc<CompletionModel>> {
    let endpoint_kind = endpoint.as_deref().map(|s| match s {
        "openai_responses" => FalLlmEndpointKind::OpenAiResponses,
        "openai_embeddings" => FalLlmEndpointKind::OpenAiEmbeddings,
        "openrouter" => FalLlmEndpointKind::OpenRouter,
        "any_llm" => FalLlmEndpointKind::AnyLlm,
        _ => FalLlmEndpointKind::OpenAiChat,
    });
    let opts = FalOptions {
        base: provider_options(api_key, model, base_url),
        endpoint: endpoint_kind,
        enterprise,
        auto_route_modality,
    };
    let provider =
        blazen_llm::providers::fal::FalProvider::from_options(opts).map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a generic OpenAI-compatible chat-completion model.
///
/// Targets any service that speaks the official OpenAI Chat Completions
/// wire format (vLLM, llama-server, LM Studio, local proxies, ...). Uses
/// `Authorization: Bearer <api_key>` auth.
#[uniffi::export]
pub fn new_openai_compat_completion_model(
    provider_name: String,
    base_url: String,
    api_key: String,
    model: String,
) -> BlazenResult<Arc<CompletionModel>> {
    let config = blazen_llm::providers::openai_compat::OpenAiCompatConfig {
        provider_name,
        base_url,
        api_key,
        default_model: model,
        auth_method: blazen_llm::providers::openai_compat::AuthMethod::Bearer,
        extra_headers: Vec::new(),
        query_params: Vec::new(),
        supports_model_listing: false,
    };
    let provider = blazen_llm::providers::openai_compat::OpenAiCompatProvider::new(config);
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Construct a [`CompletionModel`] for an Ollama server.
///
/// Convenience for [`new_custom_completion_model_with_openai_protocol`] with
/// `base_url = format!("http://{host}:{port}/v1")` and no API key. Delegates
/// to [`blazen_llm::ollama`], which knows how to speak Ollama's flavour of
/// the `OpenAI` chat-completions protocol.
#[uniffi::export]
pub fn new_ollama_completion_model(
    host: String,
    port: u16,
    model: String,
) -> BlazenResult<Arc<CompletionModel>> {
    let provider = blazen_llm::ollama(host, port, model);
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Construct a [`CompletionModel`] for an LM Studio server.
///
/// Convenience wrapper around [`blazen_llm::lm_studio`] — targets LM Studio's
/// local `OpenAI`-compatible endpoint on `http://{host}:{port}/v1`.
#[uniffi::export]
pub fn new_lm_studio_completion_model(
    host: String,
    port: u16,
    model: String,
) -> BlazenResult<Arc<CompletionModel>> {
    let provider = blazen_llm::lm_studio(host, port, model);
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Construct a [`CompletionModel`] that speaks the `OpenAI` chat-completions
/// protocol against an arbitrary base URL.
///
/// This is the same wire format as
/// [`new_openai_compat_completion_model`], but wrapped in a
/// [`blazen_llm::CustomProviderHandle`] for consistent ergonomics with the
/// `new_ollama_completion_model` / `new_lm_studio_completion_model`
/// factories. `api_key` is optional: passing `None` (or an empty `Some`)
/// omits the `Authorization` header entirely.
#[uniffi::export]
pub fn new_custom_completion_model_with_openai_protocol(
    provider_id: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
) -> BlazenResult<Arc<CompletionModel>> {
    let cfg = blazen_llm::providers::openai_compat::OpenAiCompatConfig {
        provider_name: provider_id.clone(),
        base_url,
        api_key: api_key.unwrap_or_default(),
        default_model: model,
        auth_method: blazen_llm::providers::openai_compat::AuthMethod::Bearer,
        extra_headers: Vec::new(),
        query_params: Vec::new(),
        supports_model_listing: true,
    };
    let provider = blazen_llm::openai_compat(provider_id, cfg);
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

// ---------------------------------------------------------------------------
// Cloud LLM providers — EmbeddingModel factories
// ---------------------------------------------------------------------------

/// Build an `OpenAI` embedding model.
///
/// Defaults to `text-embedding-3-small` (1536 dimensions) when `model` is
/// `None`. Passing a custom `model` keeps the model's default
/// dimensionality; callers needing a non-default dimensionality should
/// drop down to the underlying Rust API.
#[uniffi::export]
pub fn new_openai_embedding_model(
    api_key: String,
    model: Option<String>,
    base_url: Option<String>,
) -> BlazenResult<Arc<EmbeddingModel>> {
    let opts = provider_options(api_key, model, base_url);
    let em = blazen_llm::providers::openai::OpenAiEmbeddingModel::from_options(opts)
        .map_err(BlazenError::from)?;
    let inner: Arc<dyn CoreEmbeddingModel> = Arc::new(em);
    Ok(EmbeddingModel::from_arc(inner))
}

/// Build a fal.ai embedding model.
///
/// Routes through fal's OpenAI-compatible embeddings endpoint.
/// `model` defaults to `"openai/text-embedding-3-small"` (1536 dims);
/// `dimensions` overrides the produced vector size (matching the upstream
/// model's supported dimensionality).
#[uniffi::export]
pub fn new_fal_embedding_model(
    api_key: String,
    model: Option<String>,
    dimensions: Option<u32>,
) -> BlazenResult<Arc<EmbeddingModel>> {
    let fal_opts = FalOptions {
        base: provider_options(api_key, None, None),
        endpoint: Some(FalLlmEndpointKind::OpenAiEmbeddings),
        enterprise: false,
        auto_route_modality: false,
    };
    let provider = blazen_llm::providers::fal::FalProvider::from_options(fal_opts)
        .map_err(BlazenError::from)?;
    let mut em = provider.embedding_model();
    if let Some(m) = model {
        em = em.with_model(m);
    }
    if let Some(d) = dimensions {
        em = em.with_dimensions(d as usize);
    }
    let inner: Arc<dyn CoreEmbeddingModel> = Arc::new(em);
    Ok(EmbeddingModel::from_arc(inner))
}

// ---------------------------------------------------------------------------
// Local LLM providers — CompletionModel factories
// ---------------------------------------------------------------------------

/// Build a local mistral.rs chat-completion model.
///
/// `model_id` is the `HuggingFace` repo id (e.g.
/// `"mistralai/Mistral-7B-Instruct-v0.3"`) or a local GGUF path. The
/// optional `device`/`quantization` strings follow Blazen's parser format
/// (`"cpu"`, `"cuda:0"`, `"metal"`, `"q4_k_m"`, ...). Set `vision = true`
/// for multimodal models like LLaVA / Qwen2-VL.
#[cfg(feature = "mistralrs")]
#[uniffi::export]
pub fn new_mistralrs_completion_model(
    model_id: String,
    device: Option<String>,
    quantization: Option<String>,
    context_length: Option<u32>,
    vision: bool,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = blazen_llm::MistralRsOptions {
        model_id,
        quantization,
        device,
        context_length: context_length.map(|c| c as usize),
        max_batch_size: None,
        chat_template: None,
        cache_dir: None,
        vision,
    };
    let provider =
        blazen_llm::MistralRsProvider::from_options(opts).map_err(|e| BlazenError::Provider {
            kind: "MistralRsInit".into(),
            message: e.to_string(),
            provider: Some("mistralrs".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a local llama.cpp chat-completion model.
///
/// `model_path` is either a local GGUF file path or a `HuggingFace` repo
/// id; `n_gpu_layers` offloads the given number of layers to the GPU when
/// the device supports it.
#[cfg(feature = "llamacpp")]
#[uniffi::export]
pub fn new_llamacpp_completion_model(
    model_path: String,
    device: Option<String>,
    quantization: Option<String>,
    context_length: Option<u32>,
    n_gpu_layers: Option<u32>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = blazen_llm::LlamaCppOptions {
        model_path: Some(model_path),
        device,
        quantization,
        context_length: context_length.map(|c| c as usize),
        n_gpu_layers,
        cache_dir: None,
    };
    let provider = crate::runtime::runtime()
        .block_on(async { blazen_llm::LlamaCppProvider::from_options(opts).await })
        .map_err(|e| BlazenError::Provider {
            kind: "LlamaCppInit".into(),
            message: e.to_string(),
            provider: Some("llamacpp".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(provider);
    Ok(CompletionModel::from_arc(inner))
}

/// Build a local candle chat-completion model.
///
/// Wraps [`CandleLlmProvider`](blazen_llm::CandleLlmProvider) through the
/// [`CandleLlmCompletionModel`](blazen_llm::CandleLlmCompletionModel) trait
/// bridge so it satisfies the same `CompletionModel` trait as remote
/// providers.
#[cfg(feature = "candle-llm")]
#[uniffi::export]
pub fn new_candle_completion_model(
    model_id: String,
    device: Option<String>,
    quantization: Option<String>,
    revision: Option<String>,
    context_length: Option<u32>,
) -> BlazenResult<Arc<CompletionModel>> {
    let opts = blazen_llm::CandleLlmOptions {
        model_id: Some(model_id),
        device,
        quantization,
        revision,
        context_length: context_length.map(|c| c as usize),
        cache_dir: None,
    };
    let provider =
        blazen_llm::CandleLlmProvider::from_options(opts).map_err(|e| BlazenError::Provider {
            kind: "CandleLlmInit".into(),
            message: e.to_string(),
            provider: Some("candle-llm".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let bridge = blazen_llm::CandleLlmCompletionModel::new(provider);
    let inner: Arc<dyn CoreCompletionModel> = Arc::new(bridge);
    Ok(CompletionModel::from_arc(inner))
}

// ---------------------------------------------------------------------------
// Local embedding providers — EmbeddingModel factories
// ---------------------------------------------------------------------------

/// Build a local fastembed (ONNX Runtime) embedding model.
///
/// `model_name` selects a variant from fastembed's catalog (case-insensitive
/// debug spelling: `"BGESmallENV15"`, `"AllMiniLML6V2"`, ...). When `None`,
/// defaults to `BGESmallENV15`.
#[cfg(feature = "embed")]
#[uniffi::export]
pub fn new_fastembed_embedding_model(
    model_name: Option<String>,
    max_batch_size: Option<u32>,
    show_download_progress: Option<bool>,
) -> BlazenResult<Arc<EmbeddingModel>> {
    let opts = blazen_llm::EmbedOptions {
        model_name,
        cache_dir: None,
        max_batch_size: max_batch_size.map(|n| n as usize),
        show_download_progress,
    };
    let model = blazen_llm::EmbedModel::from_options(opts).map_err(|e| BlazenError::Provider {
        kind: "FastEmbedInit".into(),
        message: e.to_string(),
        provider: Some("fastembed".into()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    })?;
    let inner: Arc<dyn CoreEmbeddingModel> = Arc::new(model);
    Ok(EmbeddingModel::from_arc(inner))
}

/// Build a local candle text-embedding model.
///
/// Loads weights from `HuggingFace` and runs inference on-device. Defaults
/// to `"sentence-transformers/all-MiniLM-L6-v2"` when `model_id` is `None`.
#[cfg(feature = "candle-embed")]
#[uniffi::export]
pub fn new_candle_embedding_model(
    model_id: Option<String>,
    device: Option<String>,
    revision: Option<String>,
) -> BlazenResult<Arc<EmbeddingModel>> {
    let opts = blazen_llm::CandleEmbedOptions {
        model_id,
        device,
        revision,
        cache_dir: None,
    };
    let model = crate::runtime::runtime()
        .block_on(async { blazen_llm::CandleEmbedModel::from_options(opts).await })
        .map_err(|e| BlazenError::Provider {
            kind: "CandleEmbedInit".into(),
            message: e.to_string(),
            provider: Some("candle-embed".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        })?;
    let inner: Arc<dyn CoreEmbeddingModel> = Arc::new(model);
    Ok(EmbeddingModel::from_arc(inner))
}

// ---------------------------------------------------------------------------
// Tract (pure-Rust ONNX) embedding adapter
// ---------------------------------------------------------------------------
//
// Upstream `TractEmbedModel` does not implement `blazen_llm::EmbeddingModel`
// directly — its `embed()` returns a backend-specific `TractResponse`. We
// adapt it here so it can flow through the same `EmbeddingModel` opaque
// handle the rest of the bindings use.

/// Adapter implementing [`blazen_llm::EmbeddingModel`] over
/// [`blazen_embed_tract::TractEmbedModel`].
#[cfg(feature = "tract")]
struct TractEmbedAdapter {
    inner: blazen_embed_tract::TractEmbedModel,
}

#[cfg(feature = "tract")]
#[async_trait::async_trait]
impl CoreEmbeddingModel for TractEmbedAdapter {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed(
        &self,
        texts: &[String],
    ) -> Result<blazen_llm::types::EmbeddingResponse, blazen_llm::BlazenError> {
        let resp = self
            .inner
            .embed(texts)
            .await
            .map_err(|e| blazen_llm::BlazenError::provider("tract", e.to_string()))?;
        Ok(blazen_llm::types::EmbeddingResponse {
            embeddings: resp.embeddings,
            model: resp.model,
            usage: None,
            cost: None,
            timing: None,
            metadata: serde_json::Value::Null,
        })
    }
}

/// Build a local tract (pure-Rust ONNX) embedding model.
///
/// Drop-in replacement for [`new_fastembed_embedding_model`] for targets
/// where the prebuilt ONNX Runtime binaries can't link (musl-libc, some
/// sandboxed environments). Loads the same fastembed model catalog via
/// `tract_onnx`.
#[cfg(feature = "tract")]
#[uniffi::export]
pub fn new_tract_embedding_model(
    model_name: Option<String>,
    max_batch_size: Option<u32>,
    show_download_progress: Option<bool>,
) -> BlazenResult<Arc<EmbeddingModel>> {
    let opts = blazen_embed_tract::TractOptions {
        model_name,
        cache_dir: None,
        max_batch_size: max_batch_size.map(|n| n as usize),
        show_download_progress,
    };
    let model = blazen_embed_tract::TractEmbedModel::from_options(opts).map_err(|e| {
        BlazenError::Provider {
            kind: "TractInit".into(),
            message: e.to_string(),
            provider: Some("tract".into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        }
    })?;
    let inner: Arc<dyn CoreEmbeddingModel> = Arc::new(TractEmbedAdapter { inner: model });
    Ok(EmbeddingModel::from_arc(inner))
}
