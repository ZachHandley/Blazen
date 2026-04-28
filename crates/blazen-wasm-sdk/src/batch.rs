//! `wasm-bindgen` wrapper for [`blazen_llm::batch::complete_batch`].
//!
//! Exposes `completeBatch()` as an async function that runs multiple
//! completion requests in parallel with bounded concurrency.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::batch::{BatchConfig as InnerBatchConfig, complete_batch};
use blazen_llm::types::CompletionRequest;

use crate::chat_message::js_messages_to_vec;
use crate::completion_model::WasmCompletionModel;

// ---------------------------------------------------------------------------
// BatchConfig (tsify-derived plain struct)
// ---------------------------------------------------------------------------

/// Configuration for a batch completion run.
///
/// Mirrors [`blazen_llm::batch::BatchConfig`] as a TypeScript-friendly plain
/// object. Use `concurrency: 0` for unlimited concurrency.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct BatchConfig {
    /// Maximum number of concurrent requests. `0` means unlimited.
    pub concurrency: usize,
}

impl BatchConfig {
    /// Convert this typed config into the underlying `blazen-llm` config.
    fn into_inner(self) -> InnerBatchConfig {
        InnerBatchConfig::new(self.concurrency)
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self { concurrency: 0 }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run multiple completion requests in parallel with bounded concurrency.
///
/// `messageSets` is a JS array of message arrays — each inner array is a
/// set of `ChatMessage` or plain JSON objects that form one completion
/// request.
///
/// The optional `options` object supports:
/// - `concurrency` (number) — max concurrent requests (default: 0 = unlimited)
///
/// Returns a `Promise` that resolves to a JS object with:
/// - `responses` (array) — one `CompletionResponse` or `null` per request
/// - `errors` (array) — one error string or `null` per request
/// - `totalUsage` (object | undefined) — aggregated token usage
/// - `totalCost` (number | undefined) — aggregated cost in USD
///
/// ```js
/// const result = await completeBatch(model, [
///   [ChatMessage.user('Question 1')],
///   [ChatMessage.user('Question 2')],
///   [ChatMessage.user('Question 3')],
/// ], { concurrency: 2 });
///
/// for (let i = 0; i < result.responses.length; i++) {
///   if (result.errors[i]) {
///     console.error(`Request ${i} failed:`, result.errors[i]);
///   } else {
///     console.log(`Response ${i}:`, result.responses[i].content);
///   }
/// }
/// ```
#[wasm_bindgen(js_name = "completeBatch")]
pub fn complete_batch_js(
    model: &WasmCompletionModel,
    message_sets: JsValue,
    options: JsValue,
) -> js_sys::Promise {
    let model_arc = model.inner_arc();

    future_to_promise(async move {
        // Parse the outer array of message sets.
        let outer_array = js_sys::Array::from(&message_sets);
        let len = outer_array.length();
        let mut requests = Vec::with_capacity(len as usize);

        for i in 0..len {
            let inner_messages = outer_array.get(i);
            let msgs = js_messages_to_vec(&inner_messages).map_err(|e| {
                JsValue::from_str(&format!(
                    "Failed to parse message set at index {i}: {e:?}"
                ))
            })?;
            requests.push(CompletionRequest::new(msgs));
        }

        // Parse optional configuration. Accepts either a tsify-typed
        // `BatchConfig` or a loose `{ concurrency }` object.
        let mut concurrency: usize = 0;
        if options.is_object() {
            if let Ok(c) = js_sys::Reflect::get(&options, &JsValue::from_str("concurrency")) {
                if let Some(n) = c.as_f64() {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        concurrency = n as usize;
                    }
                }
            }
        }

        let config = BatchConfig { concurrency }.into_inner();
        let result = complete_batch(model_arc.as_ref(), requests, config).await;

        // Build the result JS object.
        let obj = js_sys::Object::new();

        // Build `responses` and `errors` arrays in parallel.
        let responses_arr = js_sys::Array::new_with_length(result.responses.len() as u32);
        let errors_arr = js_sys::Array::new_with_length(result.responses.len() as u32);

        for (i, res) in result.responses.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as u32;
            match res {
                Ok(response) => {
                    let js_resp = serde_wasm_bindgen::to_value(response)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?;
                    responses_arr.set(idx, js_resp);
                    errors_arr.set(idx, JsValue::NULL);
                }
                Err(err) => {
                    responses_arr.set(idx, JsValue::NULL);
                    errors_arr.set(idx, JsValue::from_str(&err.to_string()));
                }
            }
        }

        js_sys::Reflect::set(&obj, &JsValue::from_str("responses"), &responses_arr)?;
        js_sys::Reflect::set(&obj, &JsValue::from_str("errors"), &errors_arr)?;

        // Attach aggregated usage if present.
        if let Some(ref usage) = result.total_usage {
            let usage_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &usage_obj,
                &JsValue::from_str("promptTokens"),
                &JsValue::from_f64(f64::from(usage.prompt_tokens)),
            )?;
            js_sys::Reflect::set(
                &usage_obj,
                &JsValue::from_str("completionTokens"),
                &JsValue::from_f64(f64::from(usage.completion_tokens)),
            )?;
            js_sys::Reflect::set(
                &usage_obj,
                &JsValue::from_str("totalTokens"),
                &JsValue::from_f64(f64::from(usage.total_tokens)),
            )?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("totalUsage"), &usage_obj)?;
        }

        // Attach aggregated cost if present.
        if let Some(cost) = result.total_cost {
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("totalCost"),
                &JsValue::from_f64(cost),
            )?;
        }

        Ok(obj.into())
    })
}
