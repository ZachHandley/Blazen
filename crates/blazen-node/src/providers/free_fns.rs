//! Free functions exposed at the top level of the Node SDK.
//!
//! These mirror the hand-rolled helpers in `blazen-llm` so JS callers can
//! query env-var mappings, resolve API keys, look up context windows,
//! extract inline artifacts from a string, and feed `ModelInfo` data into
//! the global pricing registry.

#![allow(clippy::needless_pass_by_value)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::artifacts::extract_inline_artifacts as rust_extract_inline_artifacts;
use blazen_llm::keys::{
    PROVIDER_ENV_VARS, env_var_for_provider as rust_env_var_for_provider,
    resolve_api_key as rust_resolve_api_key,
};
use blazen_llm::pricing::register_from_model_info as rust_register_from_model_info;
use blazen_llm::providers::format_provider_http_tail as rust_format_provider_http_tail;
use blazen_llm::tokens::get_context_window as rust_get_context_window;
use blazen_llm::traits::ModelInfo;

use crate::error::blazen_error_to_napi;
use crate::types::JsArtifact;

// ---------------------------------------------------------------------------
// Inline artifact extraction
// ---------------------------------------------------------------------------

/// Extract inline artifacts (SVG, code blocks, mermaid, etc.) from a string of
/// LLM-generated text. Returns the artifacts in source order.
#[napi(js_name = "extractInlineArtifacts")]
#[must_use]
pub fn extract_inline_artifacts(content: String) -> Vec<JsArtifact> {
    rust_extract_inline_artifacts(&content)
        .iter()
        .map(JsArtifact::from)
        .collect()
}

// ---------------------------------------------------------------------------
// Provider env-var helpers
// ---------------------------------------------------------------------------

/// Return the environment variable name for `provider`, or `null` if the
/// provider has no well-known env var.
#[napi(js_name = "envVarForProvider")]
#[must_use]
pub fn env_var_for_provider(provider: String) -> Option<String> {
    rust_env_var_for_provider(&provider).map(str::to_owned)
}

/// Return the full mapping of provider name to env var name.
///
/// JS callers receive an array of `{ provider, envVar }` records.
#[napi(js_name = "providerEnvVars")]
#[must_use]
pub fn provider_env_vars() -> Vec<JsProviderEnvVar> {
    PROVIDER_ENV_VARS
        .iter()
        .map(|(provider, env_var)| JsProviderEnvVar {
            provider: (*provider).to_owned(),
            env_var: (*env_var).to_owned(),
        })
        .collect()
}

/// A single `(provider, envVar)` pair returned by [`provider_env_vars`].
#[napi(object)]
pub struct JsProviderEnvVar {
    /// Canonical provider name (e.g. `"openai"`, `"anthropic"`).
    pub provider: String,
    /// Environment variable name (e.g. `"OPENAI_API_KEY"`).
    #[napi(js_name = "envVar")]
    pub env_var: String,
}

/// Resolve an API key for `provider`.
///
/// Order: explicit value → environment variable → throws.
#[napi(js_name = "resolveApiKey")]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
pub fn resolve_api_key(provider: String, explicit: Option<String>) -> Result<String> {
    rust_resolve_api_key(&provider, explicit).map_err(blazen_error_to_napi)
}

// ---------------------------------------------------------------------------
// Provider HTTP tail formatter
// ---------------------------------------------------------------------------

/// Format the tail of a `ProviderHttp` error message, the same way
/// `BlazenError::ProviderHttp` does internally.
#[napi(js_name = "formatProviderHttpTail")]
#[must_use]
pub fn format_provider_http_tail(
    detail: Option<String>,
    raw_body: String,
    request_id: Option<String>,
) -> String {
    rust_format_provider_http_tail(detail.as_deref(), &raw_body, request_id.as_deref())
}

// ---------------------------------------------------------------------------
// Context window lookup
// ---------------------------------------------------------------------------

/// Best-effort context window size (in tokens) for `model`.
///
/// Falls back to 128 000 when the model string does not match a known pattern.
#[napi(js_name = "getContextWindow")]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn get_context_window(model: String) -> u32 {
    rust_get_context_window(&model) as u32
}

// ---------------------------------------------------------------------------
// Pricing registration
// ---------------------------------------------------------------------------

/// Register pricing for a model from a `ModelInfo`-shaped object.
///
/// The argument must be JSON-deserializable into [`blazen_llm::traits::ModelInfo`]
/// (i.e. carry `id`, `provider`, `capabilities`, optional `pricing`, …).
#[napi(js_name = "registerFromModelInfo")]
#[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
pub fn register_from_model_info(info: serde_json::Value) -> Result<()> {
    let parsed: ModelInfo = serde_json::from_value(info)
        .map_err(|e| napi::Error::from_reason(format!("invalid ModelInfo: {e}")))?;
    rust_register_from_model_info(&parsed);
    Ok(())
}
