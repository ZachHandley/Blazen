//! Error mapping helpers for the pipeline binding.
//!
//! `wasm-bindgen` doesn't have a custom-exception story comparable to PyO3's
//! `create_exception!` or napi-rs's `napi::Error`, so pipeline errors are
//! converted to a `JsValue` carrying a stringified message. The shape is
//! deliberately the same as every other wasm-sdk module so JS callers can
//! pattern-match on the leading prefix (`"pipeline: …"`).

use wasm_bindgen::JsValue;

/// Convert a [`blazen_pipeline::PipelineError`] into a [`JsValue`] suitable
/// for returning from a `Result<_, JsValue>` exposed to JS.
///
/// The output is a simple string-shaped error so JS callers see
/// `error.message === "pipeline: …"`.
#[must_use]
pub fn pipeline_err(err: blazen_pipeline::PipelineError) -> JsValue {
    JsValue::from_str(&format!("pipeline: {err}"))
}
