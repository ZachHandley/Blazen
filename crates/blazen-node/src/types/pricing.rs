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
                per_image: pricing.per_image,
                per_second: pricing.per_second,
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
        per_image: e.per_image,
        per_second: e.per_second,
    })
}

// ---------------------------------------------------------------------------
// Modality-aware cost helpers (Wave 3)
// ---------------------------------------------------------------------------

/// Compute the cost in USD for an image-generation call given the model id
/// and the number of images returned. Returns `null` when the model has no
/// `perImage` pricing entry registered. Mirrors
/// [`blazen_llm::compute_image_cost`].
///
/// ```javascript
/// const cost = computeImageCost("dall-e-3", 4);
/// ```
#[napi(js_name = "computeImageCost")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn compute_image_cost(model_id: String, image_count: u32) -> Option<f64> {
    blazen_llm::pricing::compute_image_cost(&model_id, image_count)
}

/// Compute the cost in USD for an audio call (TTS / STT) given the model id
/// and the duration in seconds. Returns `null` when the model has no
/// `perSecond` pricing entry registered. Mirrors
/// [`blazen_llm::compute_audio_cost`].
#[napi(js_name = "computeAudioCost")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn compute_audio_cost(model_id: String, seconds: f64) -> Option<f64> {
    blazen_llm::pricing::compute_audio_cost(&model_id, seconds)
}

/// Compute the cost in USD for a video-generation call given the model id
/// and the output duration in seconds. Returns `null` when the model has no
/// `perSecond` pricing entry registered. Mirrors
/// [`blazen_llm::compute_video_cost`].
#[napi(js_name = "computeVideoCost")]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn compute_video_cost(model_id: String, seconds: f64) -> Option<f64> {
    blazen_llm::pricing::compute_video_cost(&model_id, seconds)
}
