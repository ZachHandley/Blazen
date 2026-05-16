//! Typed mirrors of the provider/model descriptor types from
//! [`blazen_llm::traits`] and [`blazen_llm::pricing`].
//!
//! These are plain `#[napi(object)]` shapes so JS callers can construct,
//! inspect, and pass them through to provider registry functions without
//! going through `serde_json::Value`.

#![allow(clippy::needless_pass_by_value)]

use napi_derive::napi;

use blazen_llm::pricing::PricingEntry as RustPricingEntry;
use blazen_llm::traits::{
    ModelCapabilities as RustModelCapabilities, ModelInfo as RustModelInfo,
    ModelPricing as RustModelPricing, ProviderCapabilities as RustProviderCapabilities,
    ProviderConfig as RustProviderConfig,
};
use blazen_llm::types::ProviderId as RustProviderId;

use super::pricing::JsModelPricing;

// ---------------------------------------------------------------------------
// JsProviderId
// ---------------------------------------------------------------------------

/// Discriminant identifying a provider in the
/// [`crate::types::tool_output::LlmPayload::provider_raw`] escape hatch.
///
/// Mirrors [`blazen_llm::types::ProviderId`].
#[napi(string_enum, js_name = "ProviderId")]
pub enum JsProviderId {
    #[napi(value = "openai")]
    OpenAi,
    #[napi(value = "openai_compat")]
    OpenAiCompat,
    #[napi(value = "azure")]
    Azure,
    #[napi(value = "anthropic")]
    Anthropic,
    #[napi(value = "gemini")]
    Gemini,
    #[napi(value = "responses")]
    Responses,
    #[napi(value = "fal")]
    Fal,
}

impl From<JsProviderId> for RustProviderId {
    fn from(p: JsProviderId) -> Self {
        match p {
            JsProviderId::OpenAi => Self::OpenAi,
            JsProviderId::OpenAiCompat => Self::OpenAiCompat,
            JsProviderId::Azure => Self::Azure,
            JsProviderId::Anthropic => Self::Anthropic,
            JsProviderId::Gemini => Self::Gemini,
            JsProviderId::Responses => Self::Responses,
            JsProviderId::Fal => Self::Fal,
        }
    }
}

impl From<RustProviderId> for JsProviderId {
    fn from(p: RustProviderId) -> Self {
        match p {
            RustProviderId::OpenAi => Self::OpenAi,
            RustProviderId::OpenAiCompat => Self::OpenAiCompat,
            RustProviderId::Azure => Self::Azure,
            RustProviderId::Anthropic => Self::Anthropic,
            RustProviderId::Gemini => Self::Gemini,
            RustProviderId::Responses => Self::Responses,
            RustProviderId::Fal => Self::Fal,
        }
    }
}

// ---------------------------------------------------------------------------
// JsPricingEntry
// ---------------------------------------------------------------------------

/// A canonical pricing entry stored in the global pricing registry.
///
/// Mirrors [`blazen_llm::PricingEntry`]. The existing
/// [`crate::types::pricing::JsModelPricing`] is a richer "input" type that
/// also carries `perImage` / `perSecond`; this shape covers the
/// per-million token rates the registry actually stores, plus optional
/// per-image and per-second rates for multimodal/audio/video models.
#[napi(object, js_name = "PricingEntry")]
pub struct JsPricingEntry {
    /// USD per million input (prompt) tokens.
    #[napi(js_name = "inputPerMillion")]
    pub input_per_million: f64,
    /// USD per million output (completion) tokens.
    #[napi(js_name = "outputPerMillion")]
    pub output_per_million: f64,
    /// USD per image (for multimodal models). `null` if not applicable.
    #[napi(js_name = "perImage")]
    pub per_image: Option<f64>,
    /// USD per second (for audio/video models). `null` if not applicable.
    #[napi(js_name = "perSecond")]
    pub per_second: Option<f64>,
}

impl From<RustPricingEntry> for JsPricingEntry {
    fn from(p: RustPricingEntry) -> Self {
        Self {
            input_per_million: p.input_per_million,
            output_per_million: p.output_per_million,
            per_image: p.per_image,
            per_second: p.per_second,
        }
    }
}

impl From<JsPricingEntry> for RustPricingEntry {
    fn from(p: JsPricingEntry) -> Self {
        Self {
            input_per_million: p.input_per_million,
            output_per_million: p.output_per_million,
            per_image: p.per_image,
            per_second: p.per_second,
        }
    }
}

// ---------------------------------------------------------------------------
// JsProviderCapabilities
// ---------------------------------------------------------------------------

/// Capability flags advertised by a provider.
///
/// Mirrors [`blazen_llm::traits::ProviderCapabilities`].
#[napi(object, js_name = "ProviderCapabilities")]
#[allow(clippy::struct_excessive_bools)]
pub struct JsProviderCapabilities {
    /// Whether the provider supports streaming responses.
    pub streaming: bool,
    /// Whether the provider supports tool/function calling.
    #[napi(js_name = "toolCalling")]
    pub tool_calling: bool,
    /// Whether the provider supports structured output (JSON mode).
    #[napi(js_name = "structuredOutput")]
    pub structured_output: bool,
    /// Whether the provider supports vision/image inputs.
    pub vision: bool,
    /// Whether the provider supports the /models listing endpoint.
    #[napi(js_name = "modelListing")]
    pub model_listing: bool,
    /// Whether the provider supports embeddings.
    pub embeddings: bool,
}

impl From<RustProviderCapabilities> for JsProviderCapabilities {
    fn from(c: RustProviderCapabilities) -> Self {
        Self {
            streaming: c.streaming,
            tool_calling: c.tool_calling,
            structured_output: c.structured_output,
            vision: c.vision,
            model_listing: c.model_listing,
            embeddings: c.embeddings,
        }
    }
}

impl From<JsProviderCapabilities> for RustProviderCapabilities {
    fn from(c: JsProviderCapabilities) -> Self {
        Self {
            streaming: c.streaming,
            tool_calling: c.tool_calling,
            structured_output: c.structured_output,
            vision: c.vision,
            model_listing: c.model_listing,
            embeddings: c.embeddings,
        }
    }
}

// ---------------------------------------------------------------------------
// JsModelCapabilities
// ---------------------------------------------------------------------------

/// Capability flags advertised by a model.
///
/// Mirrors [`blazen_llm::traits::ModelCapabilities`].
#[napi(object, js_name = "ModelCapabilities")]
#[allow(clippy::struct_excessive_bools)]
pub struct JsModelCapabilities {
    /// Supports chat completions.
    pub chat: bool,
    /// Supports streaming responses.
    pub streaming: bool,
    /// Supports tool/function calling.
    #[napi(js_name = "toolUse")]
    pub tool_use: bool,
    /// Supports structured output (JSON schema constraints).
    #[napi(js_name = "structuredOutput")]
    pub structured_output: bool,
    /// Supports vision / image inputs.
    pub vision: bool,
    /// Supports image generation.
    #[napi(js_name = "imageGeneration")]
    pub image_generation: bool,
    /// Supports text embeddings.
    pub embeddings: bool,
    /// Video generation support.
    #[napi(js_name = "videoGeneration")]
    pub video_generation: bool,
    /// Text-to-speech synthesis.
    #[napi(js_name = "textToSpeech")]
    pub text_to_speech: bool,
    /// Speech-to-text transcription.
    #[napi(js_name = "speechToText")]
    pub speech_to_text: bool,
    /// Audio generation (music, sound effects).
    #[napi(js_name = "audioGeneration")]
    pub audio_generation: bool,
    /// 3D model generation.
    #[napi(js_name = "threeDGeneration")]
    pub three_d_generation: bool,
}

impl From<RustModelCapabilities> for JsModelCapabilities {
    fn from(c: RustModelCapabilities) -> Self {
        Self {
            chat: c.chat,
            streaming: c.streaming,
            tool_use: c.tool_use,
            structured_output: c.structured_output,
            vision: c.vision,
            image_generation: c.image_generation,
            embeddings: c.embeddings,
            video_generation: c.video_generation,
            text_to_speech: c.text_to_speech,
            speech_to_text: c.speech_to_text,
            audio_generation: c.audio_generation,
            three_d_generation: c.three_d_generation,
        }
    }
}

impl From<JsModelCapabilities> for RustModelCapabilities {
    fn from(c: JsModelCapabilities) -> Self {
        Self {
            chat: c.chat,
            streaming: c.streaming,
            tool_use: c.tool_use,
            structured_output: c.structured_output,
            vision: c.vision,
            image_generation: c.image_generation,
            embeddings: c.embeddings,
            video_generation: c.video_generation,
            text_to_speech: c.text_to_speech,
            speech_to_text: c.speech_to_text,
            audio_generation: c.audio_generation,
            three_d_generation: c.three_d_generation,
        }
    }
}

// ---------------------------------------------------------------------------
// JsProviderConfig
// ---------------------------------------------------------------------------

/// Configuration metadata describing a provider instance.
///
/// Mirrors [`blazen_llm::ProviderConfig`].
#[napi(object, js_name = "ProviderConfig")]
pub struct JsProviderConfig {
    /// A human-readable name for this provider instance.
    pub name: Option<String>,
    /// The model identifier (e.g. `"my-org/llama-3-8b"`).
    #[napi(js_name = "modelId")]
    pub model_id: Option<String>,
    /// A provider identifier (e.g. `"elevenlabs"`, `"fal"`).
    #[napi(js_name = "providerId")]
    pub provider_id: Option<String>,
    /// Base URL for HTTP-based providers.
    #[napi(js_name = "baseUrl")]
    pub base_url: Option<String>,
    /// Context window size in tokens.
    #[napi(js_name = "contextLength")]
    pub context_length: Option<u32>,
    /// Maximum output tokens the model supports.
    #[napi(js_name = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
    /// Estimated memory footprint in bytes when loaded (host RAM if on CPU,
    /// GPU VRAM otherwise).
    #[napi(js_name = "memoryEstimateBytes")]
    pub memory_estimate_bytes: Option<f64>,
    /// Pricing information for automatic cost tracking.
    pub pricing: Option<JsModelPricing>,
    /// Capability flags.
    pub capabilities: Option<JsModelCapabilities>,
}

/// Convert [`blazen_llm::traits::ModelPricing`] into the existing
/// [`JsModelPricing`] napi mirror.
fn rust_to_js_pricing(p: RustModelPricing) -> JsModelPricing {
    JsModelPricing {
        input_per_million: p.input_per_million,
        output_per_million: p.output_per_million,
        per_image: p.per_image,
        per_second: p.per_second,
    }
}

/// Convert a [`JsModelPricing`] napi mirror back into
/// [`blazen_llm::traits::ModelPricing`].
fn js_to_rust_pricing(p: JsModelPricing) -> RustModelPricing {
    RustModelPricing {
        input_per_million: p.input_per_million,
        output_per_million: p.output_per_million,
        per_image: p.per_image,
        per_second: p.per_second,
    }
}

#[allow(clippy::cast_possible_truncation)]
impl From<RustProviderConfig> for JsProviderConfig {
    fn from(c: RustProviderConfig) -> Self {
        Self {
            name: c.name,
            model_id: c.model_id,
            provider_id: c.provider_id,
            base_url: c.base_url,
            context_length: c.context_length.map(|v| v as u32),
            max_output_tokens: c.max_output_tokens.map(|v| v as u32),
            #[allow(clippy::cast_precision_loss)]
            memory_estimate_bytes: c.memory_estimate_bytes.map(|v| v as f64),
            pricing: c.pricing.map(rust_to_js_pricing),
            capabilities: c.capabilities.map(Into::into),
        }
    }
}

impl From<JsProviderConfig> for RustProviderConfig {
    fn from(c: JsProviderConfig) -> Self {
        Self {
            name: c.name,
            model_id: c.model_id,
            provider_id: c.provider_id,
            base_url: c.base_url,
            context_length: c.context_length.map(u64::from),
            max_output_tokens: c.max_output_tokens.map(u64::from),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            memory_estimate_bytes: c.memory_estimate_bytes.map(|v| v as u64),
            pricing: c.pricing.map(js_to_rust_pricing),
            capabilities: c.capabilities.map(Into::into),
        }
    }
}

// ---------------------------------------------------------------------------
// JsModelInfo
// ---------------------------------------------------------------------------

/// Information about a model offered by a provider.
///
/// Mirrors [`blazen_llm::traits::ModelInfo`].
#[napi(object, js_name = "ModelInfo")]
pub struct JsModelInfo {
    /// The model identifier used in API requests (e.g. `"gpt-4o"`).
    pub id: String,
    /// A human-readable display name, if different from the id.
    pub name: Option<String>,
    /// The provider that serves this model.
    pub provider: String,
    /// Maximum context window length in tokens.
    #[napi(js_name = "contextLength")]
    pub context_length: Option<u32>,
    /// Pricing information, if available.
    pub pricing: Option<JsModelPricing>,
    /// What this model can do.
    pub capabilities: JsModelCapabilities,
}

#[allow(clippy::cast_possible_truncation)]
impl From<RustModelInfo> for JsModelInfo {
    fn from(m: RustModelInfo) -> Self {
        Self {
            id: m.id,
            name: m.name,
            provider: m.provider,
            context_length: m.context_length.map(|v| v as u32),
            pricing: m.pricing.map(rust_to_js_pricing),
            capabilities: m.capabilities.into(),
        }
    }
}

impl From<JsModelInfo> for RustModelInfo {
    fn from(m: JsModelInfo) -> Self {
        Self {
            id: m.id,
            name: m.name,
            provider: m.provider,
            context_length: m.context_length.map(u64::from),
            pricing: m.pricing.map(js_to_rust_pricing),
            capabilities: m.capabilities.into(),
        }
    }
}
