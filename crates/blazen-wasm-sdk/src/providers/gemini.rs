//! `wasm-bindgen` wrapper for [`blazen_llm::providers::gemini::GeminiProvider`].

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::providers::gemini::GeminiProvider;
use blazen_llm::traits::CompletionModel;

use super::{
    apply_request_options, as_dyn_completion, complete_promise, fetch_client, resolve_key,
    stream_promise,
};
use crate::completion_model::WasmCompletionModel;

/// A Google Gemini chat-completion provider.
///
/// ```js
/// const provider = new GeminiProvider({ apiKey: 'AIza...', model: 'gemini-2.5-flash' });
/// const res = await provider.complete([ChatMessage.user('Hi!')]);
/// ```
#[wasm_bindgen(js_name = "GeminiProvider")]
pub struct WasmGeminiProvider {
    inner: Arc<GeminiProvider>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmGeminiProvider {}
unsafe impl Sync for WasmGeminiProvider {}

#[wasm_bindgen(js_class = "GeminiProvider")]
impl WasmGeminiProvider {
    /// Create a new Gemini provider.
    ///
    /// `options` is an optional plain JS object with:
    /// - `apiKey` (string) -- defaults to `GEMINI_API_KEY` env var
    /// - `model` (string) -- defaults to `gemini-2.5-flash`
    /// - `baseUrl` (string) -- override the API base URL
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<WasmGeminiProvider, JsValue> {
        let api_key_opt = read_string(&options, "apiKey");
        let model_opt = read_string(&options, "model");
        let base_url_opt = read_string(&options, "baseUrl");

        let api_key = resolve_key("gemini", api_key_opt)?;
        let mut provider = GeminiProvider::new_with_client(api_key, fetch_client());
        if let Some(m) = model_opt {
            provider = provider.with_model(m);
        }
        if let Some(url) = base_url_opt {
            provider = provider.with_base_url(url);
        }
        Ok(Self {
            inner: Arc::new(provider),
        })
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
        complete_promise(
            as_dyn_completion(Arc::clone(&self.inner)),
            messages,
        )
    }

    /// Perform a non-streaming completion with additional options.
    #[wasm_bindgen(js_name = "completeWithOptions")]
    pub fn complete_with_options(
        &self,
        messages: JsValue,
        options: JsValue,
    ) -> js_sys::Promise {
        let model = as_dyn_completion(Arc::clone(&self.inner));
        future_to_promise(async move {
            let msgs = crate::chat_message::js_messages_to_vec(&messages)?;
            let request = blazen_llm::types::CompletionRequest::new(msgs);
            let request = apply_request_options(request, options)?;
            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))
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

fn read_string(obj: &JsValue, key: &str) -> Option<String> {
    if !obj.is_object() {
        return None;
    }
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}
