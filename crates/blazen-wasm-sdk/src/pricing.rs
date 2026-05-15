//! Pricing registry bindings for WASM.
//!
//! Exposes [`blazen_llm::pricing`] functions to JavaScript so that users
//! can register custom model pricing and look up per-token costs from the
//! browser.

use wasm_bindgen::prelude::*;

/// Register (or overwrite) pricing for a model.
///
/// Both `inputPerMillion` and `outputPerMillion` are in USD per million
/// tokens. The `modelId` is normalized before storage, so
/// `"openai/gpt-4.1"` and `"gpt-4.1"` resolve to the same entry.
///
/// ```js
/// registerPricing('my-model', 2.5, 10.0);
/// ```
#[wasm_bindgen(js_name = "registerPricing")]
pub fn register_pricing(model_id: &str, input_per_million: f64, output_per_million: f64) {
    blazen_llm::register_pricing(
        model_id,
        blazen_llm::PricingEntry {
            input_per_million,
            output_per_million,
            per_image: None,
            per_second: None,
        },
    );
}

/// Look up pricing for a model by its ID.
///
/// Returns a plain object `{ inputPerMillion, outputPerMillion }` if the
/// model is known, or `null` otherwise.
///
/// ```js
/// const p = lookupPricing('gpt-4.1');
/// if (p) console.log(`Input: $${p.inputPerMillion}/M tokens`);
/// ```
#[wasm_bindgen(js_name = "lookupPricing")]
pub fn lookup_pricing(model_id: &str) -> JsValue {
    match blazen_llm::lookup_pricing(model_id) {
        Some(entry) => {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("inputPerMillion"),
                &JsValue::from_f64(entry.input_per_million),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("outputPerMillion"),
                &JsValue::from_f64(entry.output_per_million),
            );
            obj.into()
        }
        None => JsValue::NULL,
    }
}

/// Compute the estimated USD cost of a completion given a model ID and
/// token usage counts.
///
/// Returns the cost as a number, or `null` if the model has no pricing
/// entry registered.
///
/// ```js
/// const cost = computeCost('gpt-4.1', 1000, 500);
/// if (cost !== null) console.log(`Cost: $${cost.toFixed(6)}`);
/// ```
#[wasm_bindgen(js_name = "computeCost")]
pub fn compute_cost(model_id: &str, prompt_tokens: u32, completion_tokens: u32) -> JsValue {
    let usage = blazen_llm::TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        ..Default::default()
    };
    match blazen_llm::compute_cost(model_id, &usage) {
        Some(cost) => JsValue::from_f64(cost),
        None => JsValue::NULL,
    }
}

/// Refresh the pricing registry from the blazen.dev Cloudflare Worker
/// (or any compatible mirror). Goes out via `fetch()`, parses the bulk
/// catalog, and registers every entry. Resolves to the count.
///
/// Call once at app startup to populate pricing for models the built-in
/// baseline doesn't carry. Misses still return `null` from `computeCost`;
/// no auto-refresh or retry beyond the global registry.
///
/// ```js
/// const count = await refreshPricing();
/// // or:
/// await refreshPricing("https://my-mirror.example/pricing.json");
/// ```
#[wasm_bindgen(js_name = "refreshPricing")]
pub async fn refresh_pricing(url: Option<String>) -> Result<u32, JsValue> {
    let target = url.unwrap_or_else(|| blazen_llm::DEFAULT_PRICING_URL.to_owned());
    blazen_llm::refresh_default_with_url(&target)
        .await
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Rust-name parity exports for the pricing_fetcher convenience functions.
// ---------------------------------------------------------------------------

// wasm-bindgen doesn't support `#[wasm_bindgen]` directly on `pub const`
// (error: "will not work on constants unless you are defining a
// typescript_custom_section"), so we expose the two URLs as zero-arg getter
// functions. The JS-visible name is preserved via `js_name`, e.g.
// `import { DEFAULT_PRICING_URL } from "blazen-wasm-sdk"; await DEFAULT_PRICING_URL()`.

/// Bulk endpoint URL for the default pricing catalog (mirrors
/// [`blazen_llm::DEFAULT_PRICING_URL`]).
#[wasm_bindgen(js_name = "DEFAULT_PRICING_URL")]
#[must_use]
pub fn default_pricing_url() -> String {
    blazen_llm::DEFAULT_PRICING_URL.to_owned()
}

/// Per-model endpoint base URL (mirrors
/// [`blazen_llm::DEFAULT_MODEL_PRICING_URL_BASE`]). Append a normalized model
/// ID to fetch a single entry.
#[wasm_bindgen(js_name = "DEFAULT_MODEL_PRICING_URL_BASE")]
#[must_use]
pub fn default_model_pricing_url_base() -> String {
    blazen_llm::DEFAULT_MODEL_PRICING_URL_BASE.to_owned()
}

fn pricing_entry_to_js(entry: blazen_llm::PricingEntry) -> JsValue {
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("inputPerMillion"),
        &JsValue::from_f64(entry.input_per_million),
    );
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("outputPerMillion"),
        &JsValue::from_f64(entry.output_per_million),
    );
    if let Some(per_image) = entry.per_image {
        let _ = js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("perImage"),
            &JsValue::from_f64(per_image),
        );
    }
    if let Some(per_second) = entry.per_second {
        let _ = js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("perSecond"),
            &JsValue::from_f64(per_second),
        );
    }
    obj.into()
}

/// Bulk refresh the pricing registry from `DEFAULT_PRICING_URL` via the
/// browser `fetch()` API. Resolves to the number of entries registered.
/// Direct parity with [`blazen_llm::refresh_default`].
#[wasm_bindgen(js_name = "refreshDefault")]
pub async fn refresh_default() -> Result<u32, JsValue> {
    blazen_llm::refresh_default()
        .await
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Bulk refresh the pricing registry from `url` via the browser `fetch()`
/// API. Resolves to the number of entries registered. Direct parity with
/// [`blazen_llm::refresh_default_with_url`].
#[wasm_bindgen(js_name = "refreshDefaultWithUrl")]
pub async fn refresh_default_with_url(url: String) -> Result<u32, JsValue> {
    blazen_llm::refresh_default_with_url(&url)
        .await
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Fetch a single model's pricing from `DEFAULT_MODEL_PRICING_URL_BASE` via
/// the browser `fetch()` API and register it. Resolves to a plain object
/// `{ inputPerMillion, outputPerMillion, perImage?, perSecond? }` or `null`
/// on a 404. Direct parity with [`blazen_llm::fetch_one_default`].
#[wasm_bindgen(js_name = "fetchOneDefault")]
pub async fn fetch_one_default(model_id: String) -> Result<JsValue, JsValue> {
    match blazen_llm::fetch_one_default(&model_id).await {
        Ok(Some(entry)) => Ok(pricing_entry_to_js(entry)),
        Ok(None) => Ok(JsValue::NULL),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

/// Fetch a single model's pricing from `{urlBase}{modelId}` via the browser
/// `fetch()` API and register it. Resolves to `null` on a 404. Direct parity
/// with [`blazen_llm::fetch_one_default_with_url_base`].
#[wasm_bindgen(js_name = "fetchOneDefaultWithUrlBase")]
pub async fn fetch_one_default_with_url_base(
    url_base: String,
    model_id: String,
) -> Result<JsValue, JsValue> {
    match blazen_llm::fetch_one_default_with_url_base(&url_base, &model_id).await {
        Ok(Some(entry)) => Ok(pricing_entry_to_js(entry)),
        Ok(None) => Ok(JsValue::NULL),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}
