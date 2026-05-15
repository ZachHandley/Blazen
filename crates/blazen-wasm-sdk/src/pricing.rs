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
