//! Model pricing registration and lookup for the Node.js bindings.

use napi_derive::napi;

/// Pricing information for a model in USD per million tokens.
///
/// At minimum, `inputPerMillion` and `outputPerMillion` must be provided
/// for registration to take effect.
#[napi(object)]
pub struct JsModelPricing {
    /// USD per million input (prompt) tokens.
    pub input_per_million: Option<f64>,
    /// USD per million output (completion) tokens.
    pub output_per_million: Option<f64>,
    /// USD per image (for multimodal models).
    pub per_image: Option<f64>,
    /// USD per second (for audio/video models).
    pub per_second: Option<f64>,
}

/// Register (or overwrite) pricing for a model.
///
/// Both `inputPerMillion` and `outputPerMillion` must be set in the
/// `pricing` object; if either is `null`/`undefined` the call is silently
/// ignored.
///
/// ```javascript
/// registerPricing("my-custom-model", {
///   inputPerMillion: 2.0,
///   outputPerMillion: 8.0,
/// });
/// ```
#[napi]
#[allow(clippy::needless_pass_by_value)]
pub fn register_pricing(model_id: String, pricing: JsModelPricing) {
    if let (Some(input), Some(output)) = (pricing.input_per_million, pricing.output_per_million) {
        blazen_llm::register_pricing(
            &model_id,
            blazen_llm::PricingEntry {
                input_per_million: input,
                output_per_million: output,
            },
        );
    }
}

/// Look up pricing for a model by its ID.
///
/// Returns `null` if the model is unknown. Model IDs are normalized before
/// lookup (date suffixes, provider prefixes, and casing are stripped).
///
/// ```javascript
/// const pricing = lookupPricing("gpt-4.1");
/// if (pricing) {
///   console.log(`Input: $${pricing.inputPerMillion}/M tokens`);
/// }
/// ```
#[napi]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn lookup_pricing(model_id: String) -> Option<JsModelPricing> {
    blazen_llm::lookup_pricing(&model_id).map(|e| JsModelPricing {
        input_per_million: Some(e.input_per_million),
        output_per_million: Some(e.output_per_million),
        per_image: None,
        per_second: None,
    })
}
