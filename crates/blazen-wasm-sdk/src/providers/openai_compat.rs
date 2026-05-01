//! `wasm-bindgen` wrapper for the generic OpenAI-compatible provider.
//!
//! Exposes [`WasmOpenAiCompatProvider`], [`WasmOpenAiCompatConfig`], and
//! [`WasmAuthMethod`] so JS callers can target arbitrary OpenAI-compatible
//! endpoints (including private deployments and proxies).

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::providers::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use blazen_llm::traits::CompletionModel;

use super::{
    apply_request_options, as_dyn_completion, complete_promise, fetch_client, stream_promise,
};
use crate::completion_model::WasmCompletionModel;

// ---------------------------------------------------------------------------
// Auth method (JS-friendly enum)
// ---------------------------------------------------------------------------

/// How to authenticate with an OpenAI-compatible provider.
///
/// Re-exposes [`blazen_llm::providers::openai_compat::AuthMethod`] as a
/// `wasm-bindgen`-friendly tagged class.
///
/// ```js
/// const auth = WasmAuthMethod.bearer();
/// const auth = WasmAuthMethod.apiKeyHeader('x-api-key');
/// const auth = WasmAuthMethod.azureApiKey();
/// const auth = WasmAuthMethod.keyPrefix();
/// ```
#[wasm_bindgen(js_name = "WasmAuthMethod")]
pub struct WasmAuthMethod {
    inner: AuthMethod,
}

#[wasm_bindgen(js_class = "WasmAuthMethod")]
impl WasmAuthMethod {
    /// `Authorization: Bearer <key>`.
    #[wasm_bindgen]
    #[must_use]
    pub fn bearer() -> WasmAuthMethod {
        Self {
            inner: AuthMethod::Bearer,
        }
    }

    /// Send the key in a custom header.
    #[wasm_bindgen(js_name = "apiKeyHeader")]
    #[must_use]
    pub fn api_key_header(header_name: &str) -> WasmAuthMethod {
        Self {
            inner: AuthMethod::ApiKeyHeader(header_name.to_owned()),
        }
    }

    /// `api-key: <key>` (Azure `OpenAI`).
    #[wasm_bindgen(js_name = "azureApiKey")]
    #[must_use]
    pub fn azure_api_key() -> WasmAuthMethod {
        Self {
            inner: AuthMethod::AzureApiKey,
        }
    }

    /// `Authorization: Key <key>` (fal.ai-style).
    #[wasm_bindgen(js_name = "keyPrefix")]
    #[must_use]
    pub fn key_prefix() -> WasmAuthMethod {
        Self {
            inner: AuthMethod::KeyPrefix,
        }
    }
}

impl Clone for WasmAuthMethod {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration class
// ---------------------------------------------------------------------------

/// Configuration for an OpenAI-compatible provider.
///
/// Mirrors [`blazen_llm::providers::openai_compat::OpenAiCompatConfig`] but
/// exposed as a builder-style class for JS callers.
///
/// ```js
/// const cfg = new WasmOpenAiCompatConfig('my-llm', 'https://api.example.com/v1', 'sk-...', 'gpt-4o');
/// cfg.authMethod = WasmAuthMethod.bearer();
/// cfg.addHeader('x-custom', 'value');
/// const provider = new WasmOpenAiCompatProvider(cfg);
/// ```
#[wasm_bindgen(js_name = "WasmOpenAiCompatConfig")]
pub struct WasmOpenAiCompatConfig {
    inner: OpenAiCompatConfig,
}

#[wasm_bindgen(js_class = "WasmOpenAiCompatConfig")]
impl WasmOpenAiCompatConfig {
    /// Create a new configuration.
    ///
    /// `authMethod` defaults to `WasmAuthMethod.bearer()` and
    /// `supportsModelListing` defaults to `true`.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(
        provider_name: &str,
        base_url: &str,
        api_key: &str,
        default_model: &str,
    ) -> WasmOpenAiCompatConfig {
        Self {
            inner: OpenAiCompatConfig {
                provider_name: provider_name.to_owned(),
                base_url: base_url.to_owned(),
                api_key: api_key.to_owned(),
                default_model: default_model.to_owned(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: true,
            },
        }
    }

    /// Return a new config with the given authentication method.
    #[wasm_bindgen(js_name = "withAuthMethod")]
    #[must_use]
    pub fn with_auth_method(&self, auth_method: &WasmAuthMethod) -> WasmOpenAiCompatConfig {
        let mut next = self.inner.clone();
        next.auth_method = auth_method.inner.clone();
        Self { inner: next }
    }

    /// Return a new config with an additional request header.
    #[wasm_bindgen(js_name = "addHeader")]
    #[must_use]
    pub fn add_header(&self, key: &str, value: &str) -> WasmOpenAiCompatConfig {
        let mut next = self.inner.clone();
        next.extra_headers.push((key.to_owned(), value.to_owned()));
        Self { inner: next }
    }

    /// Return a new config with an additional request query parameter.
    #[wasm_bindgen(js_name = "addQueryParam")]
    #[must_use]
    pub fn add_query_param(&self, key: &str, value: &str) -> WasmOpenAiCompatConfig {
        let mut next = self.inner.clone();
        next.query_params.push((key.to_owned(), value.to_owned()));
        Self { inner: next }
    }

    /// Return a new config with the `/models` listing flag set as given.
    #[wasm_bindgen(js_name = "withSupportsModelListing")]
    #[must_use]
    pub fn with_supports_model_listing(&self, supports: bool) -> WasmOpenAiCompatConfig {
        let mut next = self.inner.clone();
        next.supports_model_listing = supports;
        Self { inner: next }
    }

    /// The configured provider name.
    #[wasm_bindgen(getter, js_name = "providerName")]
    pub fn provider_name(&self) -> String {
        self.inner.provider_name.clone()
    }

    /// The configured base URL.
    #[wasm_bindgen(getter, js_name = "baseUrl")]
    pub fn base_url(&self) -> String {
        self.inner.base_url.clone()
    }

    /// The configured default model identifier.
    #[wasm_bindgen(getter, js_name = "defaultModel")]
    pub fn default_model(&self) -> String {
        self.inner.default_model.clone()
    }
}

impl Clone for WasmOpenAiCompatConfig {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider class
// ---------------------------------------------------------------------------

/// A generic OpenAI-compatible chat-completion provider.
///
/// Use this when targeting an endpoint that is not covered by one of the
/// dedicated provider classes (e.g. a private deployment or proxy).
///
/// ```js
/// const cfg = new WasmOpenAiCompatConfig(
///     'my-llm',
///     'https://my-llm.internal/v1',
///     'sk-...',
///     'my-default-model',
/// );
/// const provider = new WasmOpenAiCompatProvider(cfg);
/// const res = await provider.complete([ChatMessage.user('Hi!')]);
/// ```
#[wasm_bindgen(js_name = "WasmOpenAiCompatProvider")]
pub struct WasmOpenAiCompatProvider {
    inner: Arc<OpenAiCompatProvider>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmOpenAiCompatProvider {}
unsafe impl Sync for WasmOpenAiCompatProvider {}

#[wasm_bindgen(js_class = "WasmOpenAiCompatProvider")]
impl WasmOpenAiCompatProvider {
    /// Create a new provider from a [`WasmOpenAiCompatConfig`].
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(config: &WasmOpenAiCompatConfig) -> WasmOpenAiCompatProvider {
        let provider = OpenAiCompatProvider::new_with_client(config.inner.clone(), fetch_client());
        Self {
            inner: Arc::new(provider),
        }
    }

    /// The default model identifier for this provider instance.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Convert into a generic [`CompletionModel`].
    #[wasm_bindgen(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> WasmCompletionModel {
        WasmCompletionModel::from_arc(as_dyn_completion(Arc::clone(&self.inner)))
    }

    /// Perform a non-streaming chat completion.
    #[wasm_bindgen]
    pub fn complete(&self, messages: JsValue) -> js_sys::Promise {
        complete_promise(as_dyn_completion(Arc::clone(&self.inner)), messages)
    }

    /// Perform a non-streaming completion with additional options.
    #[wasm_bindgen(js_name = "completeWithOptions")]
    pub fn complete_with_options(&self, messages: JsValue, options: JsValue) -> js_sys::Promise {
        let model = as_dyn_completion(Arc::clone(&self.inner));
        future_to_promise(async move {
            let msgs = crate::chat_message::js_messages_to_vec(&messages)?;
            let request = blazen_llm::types::CompletionRequest::new(msgs);
            let request = apply_request_options(request, options)?;
            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&response).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Perform a streaming chat completion.
    #[wasm_bindgen]
    pub fn stream(&self, messages: JsValue, callback: js_sys::Function) -> js_sys::Promise {
        stream_promise(
            as_dyn_completion(Arc::clone(&self.inner)),
            messages,
            callback,
        )
    }
}
