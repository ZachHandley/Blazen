//! [`ApiProtocol`] — selects how a `CustomProvider` talks to its backend
//! for completion calls.
//!
//! Mirrors [`blazen_llm::providers::custom::ApiProtocol`], which has two
//! variants: `OpenAi(OpenAiCompatConfig)` and `Custom`. Phase A surfaces
//! both variants and the wrapped [`OpenAiCompatConfig`] as UniFFI types so
//! Phase B's `CustomProvider` factories can take an `ApiProtocol` argument
//! directly.
//!
//! ## Wire shape
//!
//! - [`AuthMethod`] is a UniFFI Enum mirroring upstream
//!   [`blazen_llm::providers::openai_compat::AuthMethod`] exactly: 4
//!   variants, 1 with an associated `String`.
//! - [`OpenAiCompatConfig`] is a UniFFI Record. The upstream
//!   `Vec<(String, String)>` for headers / query params becomes
//!   `Vec<KeyValue>` (a 2-tuple Record) so the foreign side gets a named
//!   shape rather than an opaque pair.
//! - [`ApiProtocol`] is a UniFFI Enum: `OpenAi { config: OpenAiCompatConfig }`
//!   or `Custom`.

use blazen_llm::providers::custom::ApiProtocol as CoreApiProtocol;
use blazen_llm::providers::openai_compat::{
    AuthMethod as CoreAuthMethod, OpenAiCompatConfig as CoreOpenAiCompatConfig,
};

// ---------------------------------------------------------------------------
// KeyValue
// ---------------------------------------------------------------------------

/// Simple key/value pair for extra HTTP headers and query parameters.
///
/// Upstream uses `Vec<(String, String)>`; UniFFI Records can't represent
/// raw tuples, so we lift them into a named record.
#[derive(Debug, Clone, uniffi::Record)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

impl From<&(String, String)> for KeyValue {
    fn from(t: &(String, String)) -> Self {
        Self {
            key: t.0.clone(),
            value: t.1.clone(),
        }
    }
}

impl From<KeyValue> for (String, String) {
    fn from(kv: KeyValue) -> Self {
        (kv.key, kv.value)
    }
}

// ---------------------------------------------------------------------------
// AuthMethod
// ---------------------------------------------------------------------------

/// How a [`CustomProvider`] authenticates with an OpenAI-compatible backend.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum AuthMethod {
    /// `Authorization: Bearer <key>` (OpenAI, OpenRouter, Groq, etc.).
    Bearer,
    /// A custom header name for the API key (e.g. `x-api-key`).
    ApiKeyHeader { header_name: String },
    /// `api-key: <key>` (Azure OpenAI).
    AzureApiKey,
    /// `Authorization: Key <key>` (fal.ai).
    KeyPrefix,
}

impl From<&CoreAuthMethod> for AuthMethod {
    fn from(core: &CoreAuthMethod) -> Self {
        match core {
            CoreAuthMethod::Bearer => Self::Bearer,
            CoreAuthMethod::ApiKeyHeader(h) => Self::ApiKeyHeader {
                header_name: h.clone(),
            },
            CoreAuthMethod::AzureApiKey => Self::AzureApiKey,
            CoreAuthMethod::KeyPrefix => Self::KeyPrefix,
        }
    }
}

impl From<AuthMethod> for CoreAuthMethod {
    fn from(ffi: AuthMethod) -> Self {
        match ffi {
            AuthMethod::Bearer => Self::Bearer,
            AuthMethod::ApiKeyHeader { header_name } => Self::ApiKeyHeader(header_name),
            AuthMethod::AzureApiKey => Self::AzureApiKey,
            AuthMethod::KeyPrefix => Self::KeyPrefix,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAiCompatConfig
// ---------------------------------------------------------------------------

/// Configuration for an OpenAI-compatible provider backend.
///
/// Used by the [`ApiProtocol::OpenAi`] variant.
#[derive(Debug, Clone, uniffi::Record)]
pub struct OpenAiCompatConfig {
    /// Human-readable name for this provider (used in logs and model info).
    pub provider_name: String,
    /// Base URL for the API (e.g. `https://api.openai.com/v1`).
    pub base_url: String,
    /// API key. May be empty if the provider doesn't require auth.
    pub api_key: String,
    /// Default model to use if a request doesn't override it.
    pub default_model: String,
    /// How to send the API key.
    pub auth_method: AuthMethod,
    /// Extra HTTP headers to include in every request.
    pub extra_headers: Vec<KeyValue>,
    /// Query parameters to include in every request (e.g. Azure's
    /// `api-version`).
    pub query_params: Vec<KeyValue>,
    /// Whether this provider supports the `/models` listing endpoint.
    pub supports_model_listing: bool,
}

impl From<&CoreOpenAiCompatConfig> for OpenAiCompatConfig {
    fn from(core: &CoreOpenAiCompatConfig) -> Self {
        Self {
            provider_name: core.provider_name.clone(),
            base_url: core.base_url.clone(),
            api_key: core.api_key.clone(),
            default_model: core.default_model.clone(),
            auth_method: AuthMethod::from(&core.auth_method),
            extra_headers: core.extra_headers.iter().map(KeyValue::from).collect(),
            query_params: core.query_params.iter().map(KeyValue::from).collect(),
            supports_model_listing: core.supports_model_listing,
        }
    }
}

impl From<OpenAiCompatConfig> for CoreOpenAiCompatConfig {
    fn from(ffi: OpenAiCompatConfig) -> Self {
        Self {
            provider_name: ffi.provider_name,
            base_url: ffi.base_url,
            api_key: ffi.api_key,
            default_model: ffi.default_model,
            auth_method: ffi.auth_method.into(),
            extra_headers: ffi.extra_headers.into_iter().map(Into::into).collect(),
            query_params: ffi.query_params.into_iter().map(Into::into).collect(),
            supports_model_listing: ffi.supports_model_listing,
        }
    }
}

// ---------------------------------------------------------------------------
// ApiProtocol
// ---------------------------------------------------------------------------

/// Selects how a [`CustomProvider`] talks to its backend for completion
/// calls.
///
/// - [`ApiProtocol::OpenAi`]: framework handles HTTP, SSE parsing, tool
///   calls, retries. The wrapped `OpenAiCompatConfig` supplies the base
///   URL, model, optional API key, headers, and query parameters.
/// - [`ApiProtocol::Custom`]: framework dispatches every completion
///   method to a foreign-implemented [`crate::provider_custom::CustomProvider`]
///   trait object. No additional configuration here — the foreign impl owns
///   the wire format.
///
/// Media-generation calls always go through the foreign-implemented
/// `CustomProvider` regardless of which protocol is selected here.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum ApiProtocol {
    /// OpenAI Chat Completions wire format.
    OpenAi { config: OpenAiCompatConfig },
    /// User-defined protocol — handled by a foreign-implemented
    /// [`crate::provider_custom::CustomProvider`] trait object.
    Custom,
}

impl From<&CoreApiProtocol> for ApiProtocol {
    fn from(core: &CoreApiProtocol) -> Self {
        match core {
            CoreApiProtocol::OpenAi(cfg) => Self::OpenAi {
                config: OpenAiCompatConfig::from(cfg),
            },
            CoreApiProtocol::Custom => Self::Custom,
        }
    }
}

impl From<ApiProtocol> for CoreApiProtocol {
    fn from(ffi: ApiProtocol) -> Self {
        match ffi {
            ApiProtocol::OpenAi { config } => Self::OpenAi(config.into()),
            ApiProtocol::Custom => Self::Custom,
        }
    }
}
