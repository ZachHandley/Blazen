//! LLM provider implementations.
//!
//! All providers are always available. The only opt-in features are:
//! - `reqwest` — enables the native HTTP client (for non-WASM targets)
//! - `tiktoken` — enables exact BPE token counting
//!
//! ## Native providers (custom API formats)
//!
//! - [`openai`] — `OpenAI` Chat Completions API
//! - [`anthropic`] — Anthropic Messages API
//! - [`gemini`] — Google Gemini API
//! - [`azure`] — Azure `OpenAI` Service
//! - [`fal`] — fal.ai compute platform (LLM + media generation)
//!
//! ## OpenAI-compatible providers
//!
//! - [`openai_compat`] — Generic OpenAI-compatible base
//! - [`groq`] — Groq (ultra-fast LPU inference)
//! - [`openrouter`] — `OpenRouter` (400+ models)
//! - [`together`] — Together AI
//! - [`mistral`] — Mistral AI
//! - [`deepseek`] — `DeepSeek`
//! - [`fireworks`] — Fireworks AI
//! - [`perplexity`] — Perplexity
//! - [`xai`] — xAI (Grok)
//! - [`cohere`] — Cohere
//! - [`bedrock`] — AWS Bedrock (via Mantle)

// Shared SSE parser used by OpenAI-compatible and Azure providers.
pub(crate) mod sse;

// Shared multimodal content helpers and HTTP utilities.
pub(crate) mod openai_format;

// OpenAI Responses API body conversion (used by fal "openai-responses" route).
pub(crate) mod responses_format;

// Native providers
pub mod anthropic;
pub mod azure;
pub mod fal;
pub mod gemini;
pub mod openai;
pub mod openai_compat;

// OpenAI-compatible dedicated providers
pub mod bedrock;
pub mod cohere;
pub mod deepseek;
pub mod fireworks;
pub mod groq;
pub mod mistral;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod xai;

// ---------------------------------------------------------------------------
// Shared `from_options` macro
// ---------------------------------------------------------------------------

/// Generate a `from_options(api_key, ProviderOptions) -> Self` constructor for
/// a "simple" provider that has `Self::new(api_key)`, `with_model(m)`, and
/// (optionally) `with_base_url(url)` builders.
///
/// Bindings should deserialize their native options dict into
/// [`ProviderOptions`](crate::types::provider_options::ProviderOptions) and
/// call this method instead of manually destructuring fields.
///
/// Use the `, no_base_url` variant for providers that don't expose
/// `with_base_url` (the OpenAI-compatible wrappers).
macro_rules! impl_simple_from_options {
    ($provider:ty) => {
        impl $provider {
            /// Construct from typed [`ProviderOptions`](crate::types::provider_options::ProviderOptions).
            #[must_use]
            pub fn from_options(
                api_key: impl Into<String>,
                opts: $crate::types::provider_options::ProviderOptions,
            ) -> Self {
                let mut p = Self::new(api_key);
                if let Some(m) = opts.model {
                    p = p.with_model(m);
                }
                if let Some(url) = opts.base_url {
                    p = p.with_base_url(url);
                }
                p
            }
        }
    };
    ($provider:ty, no_base_url) => {
        impl $provider {
            /// Construct from typed [`ProviderOptions`](crate::types::provider_options::ProviderOptions).
            ///
            /// `base_url` is ignored — this provider's endpoint is fixed.
            #[must_use]
            pub fn from_options(
                api_key: impl Into<String>,
                opts: $crate::types::provider_options::ProviderOptions,
            ) -> Self {
                let mut p = Self::new(api_key);
                if let Some(m) = opts.model {
                    p = p.with_model(m);
                }
                p
            }
        }
    };
}

pub(crate) use impl_simple_from_options;
