//! JavaScript binding for the [`ApiProtocol`] enum used by
//! `CustomProvider` to select between the OpenAI-compatible wire format
//! and a fully user-defined protocol.
//!
//! Exposes [`JsApiProtocol`] as a NAPI class with two static factory
//! methods (`openai` and `custom`), a `kind` getter that returns
//! `"openai"` or `"custom"`, and a `config` getter that returns the
//! wrapped [`JsOpenAiCompatConfig`] when applicable.

use napi_derive::napi;

use blazen_llm::providers::custom::ApiProtocol;
use blazen_llm::providers::openai_compat::OpenAiCompatConfig;

use crate::providers::openai_compat::{JsAuthMethod, JsOpenAiCompatConfig};

// ---------------------------------------------------------------------------
// JsApiProtocol
// ---------------------------------------------------------------------------

/// Selects how a custom provider talks to its backend for completion
/// calls.
///
/// ```javascript
/// import { ApiProtocol } from "blazen";
///
/// const p1 = ApiProtocol.openai({
///   providerName: "my-host",
///   baseUrl: "https://api.example.com/v1",
///   apiKey: "sk-...",
///   defaultModel: "my-model",
/// });
/// console.log(p1.kind); // "openai"
///
/// const p2 = ApiProtocol.custom();
/// console.log(p2.kind); // "custom"
/// ```
#[napi(js_name = "ApiProtocol")]
pub struct JsApiProtocol {
    inner: ApiProtocol,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsApiProtocol {
    // -----------------------------------------------------------------
    // Static factories
    // -----------------------------------------------------------------

    /// Build an OpenAI-compatible API protocol wrapping the supplied
    /// configuration.
    #[napi(factory)]
    pub fn openai(config: JsOpenAiCompatConfig) -> Self {
        let cfg: OpenAiCompatConfig = config.into();
        Self {
            inner: ApiProtocol::OpenAi(cfg),
        }
    }

    /// Build a custom (user-defined) API protocol. The completion
    /// dispatch path is handled by the host-language object passed to
    /// `CustomProvider`.
    #[napi(factory)]
    pub fn custom() -> Self {
        Self {
            inner: ApiProtocol::Custom,
        }
    }

    // -----------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------

    /// Discriminator string: `"openai"` or `"custom"`.
    #[napi(getter)]
    pub fn kind(&self) -> &'static str {
        match &self.inner {
            ApiProtocol::OpenAi(_) => "openai",
            ApiProtocol::Custom => "custom",
        }
    }

    /// The wrapped [`JsOpenAiCompatConfig`] when `kind === "openai"`,
    /// otherwise `null`.
    #[napi(getter)]
    pub fn config(&self) -> Option<JsOpenAiCompatConfig> {
        match &self.inner {
            ApiProtocol::OpenAi(cfg) => Some(openai_compat_to_js(cfg)),
            ApiProtocol::Custom => None,
        }
    }
}

impl JsApiProtocol {
    /// Internal: borrow the underlying Rust enum. Used by Phase B
    /// wiring (`CustomProvider::with_protocol(...)`) once the JS-side
    /// `CustomProvider` constructor learns to accept a `JsApiProtocol`.
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &ApiProtocol {
        &self.inner
    }
}

/// Convert the Rust [`OpenAiCompatConfig`] back into the napi mirror
/// struct. Needed because `JsOpenAiCompatConfig` is a plain `#[napi(object)]`
/// (no `From<OpenAiCompatConfig>` impl exists on the JS side; only the
/// inverse direction is wired in [`crate::providers::openai_compat`]).
fn openai_compat_to_js(cfg: &OpenAiCompatConfig) -> JsOpenAiCompatConfig {
    use blazen_llm::providers::openai_compat::AuthMethod;

    let (auth_method, custom_header_name) = match &cfg.auth_method {
        AuthMethod::Bearer => (Some(JsAuthMethod::Bearer), None),
        AuthMethod::AzureApiKey => (Some(JsAuthMethod::AzureApiKey), None),
        AuthMethod::KeyPrefix => (Some(JsAuthMethod::KeyPrefix), None),
        AuthMethod::ApiKeyHeader(name) => (None, Some(name.clone())),
    };

    let to_pairs = |pairs: &[(String, String)]| -> Option<Vec<Vec<String>>> {
        if pairs.is_empty() {
            None
        } else {
            Some(
                pairs
                    .iter()
                    .map(|(k, v)| vec![k.clone(), v.clone()])
                    .collect(),
            )
        }
    };

    JsOpenAiCompatConfig {
        provider_name: cfg.provider_name.clone(),
        base_url: cfg.base_url.clone(),
        api_key: cfg.api_key.clone(),
        default_model: cfg.default_model.clone(),
        auth_method,
        custom_header_name,
        extra_headers: to_pairs(&cfg.extra_headers),
        query_params: to_pairs(&cfg.query_params),
        supports_model_listing: Some(cfg.supports_model_listing),
    }
}
