//! `wasm-bindgen` wrapper for [`blazen_llm::providers::openai::OpenAiProvider`].

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::compute::AudioGeneration;
use blazen_llm::providers::openai::OpenAiProvider;
use blazen_llm::traits::CompletionModel;

use super::{
    apply_request_options, as_dyn_completion, complete_promise, fetch_client, resolve_key,
    stream_promise,
};
use crate::completion_model::WasmCompletionModel;

/// An `OpenAI` chat-completion provider with native text-to-speech support.
///
/// ```js
/// const provider = new OpenAiProvider({ apiKey: 'sk-...', model: 'gpt-4o' });
/// const res = await provider.complete([ChatMessage.user('Hi!')]);
/// const audio = await provider.textToSpeech({ text: 'Hello!', voice: 'alloy' });
/// ```
#[wasm_bindgen(js_name = "OpenAiProvider")]
pub struct WasmOpenAiProvider {
    inner: Arc<OpenAiProvider>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmOpenAiProvider {}
unsafe impl Sync for WasmOpenAiProvider {}

#[wasm_bindgen(js_class = "OpenAiProvider")]
impl WasmOpenAiProvider {
    /// Create a new `OpenAI` provider.
    ///
    /// `options` is an optional plain JS object with:
    /// - `apiKey` (string) -- defaults to `OPENAI_API_KEY` env var
    /// - `model` (string) -- defaults to `gpt-4.1`
    /// - `baseUrl` (string) -- override the API base URL
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<WasmOpenAiProvider, JsValue> {
        let api_key_opt = read_string(&options, "apiKey");
        let model_opt = read_string(&options, "model");
        let base_url_opt = read_string(&options, "baseUrl");

        let api_key = resolve_key("openai", api_key_opt)?;
        let mut provider = OpenAiProvider::new_with_client(api_key, fetch_client());
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

    /// Convert into a generic [`CompletionModel`] for use with `runAgent`,
    /// `batchComplete`, decorators, etc.
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

    /// Synthesize speech from text via `OpenAI`'s `/v1/audio/speech` endpoint.
    ///
    /// `request` is a plain JS object shaped like
    /// [`blazen_llm::compute::SpeechRequest`].
    #[wasm_bindgen(js_name = "textToSpeech")]
    pub fn text_to_speech(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::SpeechRequest =
                serde_wasm_bindgen::from_value(request)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = AudioGeneration::text_to_speech(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }
}

/// Read an optional string property from a JS object.
fn read_string(obj: &JsValue, key: &str) -> Option<String> {
    if !obj.is_object() {
        return None;
    }
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}
