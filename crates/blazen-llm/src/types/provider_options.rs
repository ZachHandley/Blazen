//! Unified provider options types.
//!
//! Each provider factory accepts an options struct that extends [`ProviderOptions`]
//! via `#[serde(flatten)]`. This gives every provider `model` and `base_url`
//! overrides, while provider-specific types add their own fields.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Base
// ---------------------------------------------------------------------------

/// Options shared by every provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ProviderOptions {
    /// Override the default model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Override the provider's base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

// ---------------------------------------------------------------------------
// fal.ai
// ---------------------------------------------------------------------------

/// Options specific to the fal.ai provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct FalOptions {
    /// Common provider options (`model`, `base_url`).
    #[serde(flatten)]
    pub base: ProviderOptions,
    /// The fal endpoint family. Defaults to `OpenAiChat`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<FalLlmEndpointKind>,
    /// Promote the endpoint to its enterprise / SOC2-eligible variant.
    #[serde(default)]
    pub enterprise: bool,
    /// Auto-route to vision/audio/video variant when content has media.
    #[serde(default = "default_true")]
    pub auto_route_modality: bool,
}

impl Default for FalOptions {
    fn default() -> Self {
        Self {
            base: ProviderOptions::default(),
            endpoint: None,
            enterprise: false,
            auto_route_modality: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Simplified fal.ai endpoint family.
///
/// Maps to the core [`FalLlmEndpoint`](crate::providers::fal::FalLlmEndpoint)
/// variants. The `enterprise` flag on [`FalOptions`] controls whether
/// the enterprise/SOC2 path is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[serde(rename_all = "snake_case")]
pub enum FalLlmEndpointKind {
    /// OpenAI-compatible chat-completions endpoint (default).
    OpenAiChat,
    /// OpenAI-compatible Responses endpoint.
    OpenAiResponses,
    /// OpenAI-compatible embeddings endpoint.
    OpenAiEmbeddings,
    /// `OpenRouter` proxy via fal.
    OpenRouter,
    /// fal-ai/any-llm proxy.
    AnyLlm,
}

// ---------------------------------------------------------------------------
// Azure
// ---------------------------------------------------------------------------

/// Options specific to Azure `OpenAI`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct AzureOptions {
    /// Common provider options (`model`, `base_url`).
    #[serde(flatten)]
    pub base: ProviderOptions,
    /// Azure resource name.
    pub resource_name: String,
    /// Azure deployment name.
    pub deployment_name: String,
    /// API version override (e.g. `"2024-02-15-preview"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

// ---------------------------------------------------------------------------
// Bedrock
// ---------------------------------------------------------------------------

/// Options specific to AWS Bedrock.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct BedrockOptions {
    /// Common provider options (`model`, `base_url`).
    #[serde(flatten)]
    pub base: ProviderOptions,
    /// AWS region (e.g. `"us-east-1"`).
    pub region: String,
}
