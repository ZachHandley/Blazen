//! Standalone `wasm-bindgen` wrappers for each cloud LLM provider.
//!
//! These classes complement the [`crate::completion_model::WasmCompletionModel`]
//! factory methods. They give JavaScript callers a typed, named class per
//! provider, plus access to provider-specific capability methods that the
//! generic `CompletionModel` does not surface (e.g. `OpenAiProvider.textToSpeech`,
//! `FalProvider.generateImage`).
//!
//! Each class is itself a `CompletionModel` -- call `.toCompletionModel()` to
//! obtain a generic [`crate::completion_model::WasmCompletionModel`] that can
//! be passed to `runAgent`, `batchComplete`, decorators, etc.

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::FetchHttpClient;
use blazen_llm::http::HttpClient;
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::{CompletionRequest, ToolDefinition};

use crate::chat_message::js_messages_to_vec;
use crate::completion_model::WasmCompletionModel;

pub mod anthropic;
pub mod azure;
pub mod bedrock;
pub mod compat_providers;
pub mod fal;
pub mod gemini;
pub mod openai;
pub mod openai_compat;
pub mod typed_tool;

pub use anthropic::WasmAnthropicProvider;
pub use azure::WasmAzureOpenAiProvider;
pub use bedrock::WasmBedrockProvider;
pub use compat_providers::{
    WasmCohereProvider, WasmDeepSeekProvider, WasmFireworksProvider, WasmGroqProvider,
    WasmMistralProvider, WasmOpenRouterProvider, WasmPerplexityProvider, WasmTogetherProvider,
    WasmXaiProvider,
};
pub use fal::WasmFalProvider;
pub use gemini::WasmGeminiProvider;
pub use openai::WasmOpenAiProvider;
pub use openai_compat::{WasmAuthMethod, WasmOpenAiCompatConfig, WasmOpenAiCompatProvider};
pub use typed_tool::{WasmTypedTool, typed_tool_simple};

// ---------------------------------------------------------------------------
// Shared helpers (used by per-provider wrappers in this module)
// ---------------------------------------------------------------------------

/// Build a `FetchHttpClient`-backed [`Arc<dyn HttpClient>`] for use in WASM.
pub(crate) fn fetch_client() -> Arc<dyn HttpClient> {
    FetchHttpClient::new().into_arc()
}

/// Convert a concrete provider `Arc` into the trait-object form expected by
/// [`crate::completion_model::WasmCompletionModel`] and the shared
/// `complete_promise` / `stream_promise` helpers.
///
/// This exists because `Arc<T> -> Arc<dyn Trait>` requires a coercion site
/// (`let` with explicit type, or function-call boundary), which is not
/// available at every call site in this module.
pub(crate) fn as_dyn_completion<T>(arc: Arc<T>) -> Arc<dyn CompletionModel>
where
    T: CompletionModel + 'static,
{
    arc
}

/// Resolve the API key for `provider` (explicit > env var) and surface
/// errors as `JsValue` strings.
pub(crate) fn resolve_key(
    provider: &str,
    explicit: Option<String>,
) -> Result<String, JsValue> {
    blazen_llm::keys::resolve_api_key(provider, explicit)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Apply request-level options (temperature, maxTokens, topP, model, tools,
/// responseFormat) from a JS plain object to a [`CompletionRequest`].
///
/// Mirrors the option handling of
/// [`crate::completion_model::WasmCompletionModel::complete_with_options`].
pub(crate) fn apply_request_options(
    mut request: CompletionRequest,
    options: JsValue,
) -> Result<CompletionRequest, JsValue> {
    if !options.is_object() {
        return Ok(request);
    }
    let opts: serde_json::Value = serde_wasm_bindgen::from_value(options)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    if let Some(temp) = opts.get("temperature").and_then(|v| v.as_f64()) {
        #[allow(clippy::cast_possible_truncation)]
        {
            request = request.with_temperature(temp as f32);
        }
    }
    if let Some(max) = opts.get("maxTokens").and_then(|v| v.as_u64()) {
        #[allow(clippy::cast_possible_truncation)]
        {
            request = request.with_max_tokens(max as u32);
        }
    }
    if let Some(top_p) = opts.get("topP").and_then(|v| v.as_f64()) {
        #[allow(clippy::cast_possible_truncation)]
        {
            request = request.with_top_p(top_p as f32);
        }
    }
    if let Some(m) = opts.get("model").and_then(|v| v.as_str()) {
        request = request.with_model(m);
    }
    if let Some(fmt) = opts.get("responseFormat") {
        request = request.with_response_format(fmt.clone());
    }
    if let Some(tools) = opts.get("tools").and_then(|v| v.as_array()) {
        let mut defs = Vec::with_capacity(tools.len());
        for (i, t) in tools.iter().enumerate() {
            let def: ToolDefinition = serde_json::from_value(t.clone()).map_err(|e| {
                JsValue::from_str(&format!("invalid tool definition at index {i}: {e}"))
            })?;
            defs.push(def);
        }
        if !defs.is_empty() {
            request = request.with_tools(defs);
        }
    }
    Ok(request)
}

/// Run a non-streaming completion through `model`, converting messages from
/// a JS array and the response back to a JS object.
pub(crate) fn complete_promise(
    model: Arc<dyn CompletionModel>,
    messages: JsValue,
) -> js_sys::Promise {
    future_to_promise(async move {
        let msgs = js_messages_to_vec(&messages)?;
        let request = CompletionRequest::new(msgs);
        let response = model
            .complete(request)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&response).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

/// Streaming sibling of [`complete_promise`].
pub(crate) fn stream_promise(
    model: Arc<dyn CompletionModel>,
    messages: JsValue,
    callback: js_sys::Function,
) -> js_sys::Promise {
    future_to_promise(async move {
        let msgs = js_messages_to_vec(&messages)?;
        let request = CompletionRequest::new(msgs);
        let mut stream = model
            .stream(request)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        use futures_util::StreamExt;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| JsValue::from_str(&e.to_string()))?;
            let js_chunk = serde_wasm_bindgen::to_value(&chunk)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let _ = callback.call1(&JsValue::NULL, &js_chunk);
        }
        Ok(JsValue::UNDEFINED)
    })
}

// ---------------------------------------------------------------------------
// Free functions exposed to JS
// ---------------------------------------------------------------------------

/// Scan an LLM-generated text string for inline artifacts (fenced code
/// blocks, `<svg>` runs, mermaid diagrams, etc.) and return them as a typed
/// JS array.
///
/// Mirrors [`blazen_llm::extract_inline_artifacts`].
#[wasm_bindgen(js_name = "extractInlineArtifacts")]
pub fn extract_inline_artifacts(content: &str) -> Result<JsValue, JsValue> {
    let artifacts = blazen_llm::extract_inline_artifacts(content);
    serde_wasm_bindgen::to_value(&artifacts).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Return the well-known environment variable that holds the API key for
/// `provider`, or `undefined` if the provider is not registered.
///
/// Examples: `envVarForProvider("openai") === "OPENAI_API_KEY"`.
#[wasm_bindgen(js_name = "envVarForProvider")]
#[must_use]
pub fn env_var_for_provider(provider: &str) -> Option<String> {
    blazen_llm::keys::env_var_for_provider(provider).map(str::to_owned)
}

/// Resolve an API key for `provider`. Resolution order:
/// 1. `explicit` -- if non-empty, used directly.
/// 2. The provider's well-known environment variable (see
///    [`env_var_for_provider`]).
///
/// Throws when no key can be found.
#[wasm_bindgen(js_name = "resolveApiKey")]
pub fn resolve_api_key(provider: &str, explicit: Option<String>) -> Result<String, JsValue> {
    blazen_llm::keys::resolve_api_key(provider, explicit)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Format the trailing portion of a `BlazenError::ProviderHttp` Display
/// string. Useful when surfacing structured provider errors to JS callers.
///
/// `detail` is preferred when present; otherwise the first 200 characters
/// of `rawBody` are used. When `requestId` is provided, ` (request-id=...)`
/// is appended.
#[wasm_bindgen(js_name = "formatProviderHttpTail")]
#[must_use]
pub fn format_provider_http_tail(
    detail: Option<String>,
    raw_body: &str,
    request_id: Option<String>,
) -> String {
    blazen_llm::providers::format_provider_http_tail(
        detail.as_deref(),
        raw_body,
        request_id.as_deref(),
    )
}

/// Best-effort context window size in tokens for a model identifier.
///
/// Falls back to 128 000 when the model string does not match any known
/// pattern.
#[wasm_bindgen(js_name = "getContextWindow")]
#[must_use]
pub fn get_context_window(model: &str) -> u32 {
    let value = blazen_llm::tokens::get_context_window(model);
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Register pricing for a model from a JS object shaped like
/// [`blazen_llm::ModelInfo`]. Does nothing when the info object lacks
/// pricing data.
#[wasm_bindgen(js_name = "registerFromModelInfo")]
pub fn register_from_model_info(info: JsValue) -> Result<(), JsValue> {
    let model_info: blazen_llm::ModelInfo = serde_wasm_bindgen::from_value(info)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    blazen_llm::pricing::register_from_model_info(&model_info);
    Ok(())
}
