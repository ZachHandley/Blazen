//! `wasm-bindgen` wrappers for the OpenAI-compatible cloud LLM providers
//! (Groq, `OpenRouter`, Together, Mistral, `DeepSeek`, Fireworks, Perplexity,
//! xAI, Cohere).

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::traits::CompletionModel;

use super::{
    apply_request_options, as_dyn_completion, complete_promise, fetch_client, resolve_key,
    stream_promise,
};
use crate::completion_model::WasmCompletionModel;

fn read_string(obj: &JsValue, key: &str) -> Option<String> {
    if !obj.is_object() {
        return None;
    }
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}

/// Generate a `wasm-bindgen` provider class wrapping a dedicated
/// `blazen_llm::providers::*::*Provider` type.
///
/// Each generated class exposes:
/// - constructor `new(options?)` accepting `{ apiKey?, model? }`
/// - `modelId` getter
/// - `toCompletionModel()`
/// - `complete(messages)`, `completeWithOptions(messages, options)`,
///   `stream(messages, callback)`
macro_rules! impl_compat_provider {
    (
        struct $wrapper:ident,
        wraps $rust_provider:path,
        js_name $js_name:literal,
        env_provider $env_provider:literal,
        doc $doc:literal
    ) => {
        #[doc = $doc]
        #[wasm_bindgen(js_name = $js_name)]
        pub struct $wrapper {
            inner: Arc<$rust_provider>,
        }

        // SAFETY: WASM is single-threaded.
        unsafe impl Send for $wrapper {}
        unsafe impl Sync for $wrapper {}

        #[wasm_bindgen(js_class = $js_name)]
        impl $wrapper {
            /// Create a new provider instance.
            ///
            /// `options` is an optional plain JS object with:
            /// - `apiKey` (string) -- explicit API key (overrides env var)
            /// - `model` (string) -- override the default model
            #[wasm_bindgen(constructor)]
            pub fn new(options: JsValue) -> Result<$wrapper, JsValue> {
                let api_key_opt = read_string(&options, "apiKey");
                let model_opt = read_string(&options, "model");
                let api_key = resolve_key($env_provider, api_key_opt)?;
                let mut provider = <$rust_provider>::new_with_client(api_key, fetch_client());
                if let Some(m) = model_opt {
                    provider = provider.with_model(m);
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
                complete_promise(as_dyn_completion(Arc::clone(&self.inner)), messages)
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
    };
}

impl_compat_provider! {
    struct WasmGroqProvider,
    wraps blazen_llm::providers::groq::GroqProvider,
    js_name "GroqProvider",
    env_provider "groq",
    doc " A Groq chat-completion provider (ultra-fast LPU inference)."
}

impl_compat_provider! {
    struct WasmOpenRouterProvider,
    wraps blazen_llm::providers::openrouter::OpenRouterProvider,
    js_name "OpenRouterProvider",
    env_provider "openrouter",
    doc " An OpenRouter chat-completion provider (400+ models)."
}

impl_compat_provider! {
    struct WasmTogetherProvider,
    wraps blazen_llm::providers::together::TogetherProvider,
    js_name "TogetherProvider",
    env_provider "together",
    doc " A Together AI chat-completion provider."
}

impl_compat_provider! {
    struct WasmMistralProvider,
    wraps blazen_llm::providers::mistral::MistralProvider,
    js_name "MistralProvider",
    env_provider "mistral",
    doc " A Mistral AI chat-completion provider."
}

impl_compat_provider! {
    struct WasmDeepSeekProvider,
    wraps blazen_llm::providers::deepseek::DeepSeekProvider,
    js_name "DeepSeekProvider",
    env_provider "deepseek",
    doc " A DeepSeek chat-completion provider."
}

impl_compat_provider! {
    struct WasmFireworksProvider,
    wraps blazen_llm::providers::fireworks::FireworksProvider,
    js_name "FireworksProvider",
    env_provider "fireworks",
    doc " A Fireworks AI chat-completion provider."
}

impl_compat_provider! {
    struct WasmPerplexityProvider,
    wraps blazen_llm::providers::perplexity::PerplexityProvider,
    js_name "PerplexityProvider",
    env_provider "perplexity",
    doc " A Perplexity chat-completion provider."
}

impl_compat_provider! {
    struct WasmXaiProvider,
    wraps blazen_llm::providers::xai::XaiProvider,
    js_name "XaiProvider",
    env_provider "xai",
    doc " An xAI (Grok) chat-completion provider."
}

impl_compat_provider! {
    struct WasmCohereProvider,
    wraps blazen_llm::providers::cohere::CohereProvider,
    js_name "CohereProvider",
    env_provider "cohere",
    doc " A Cohere chat-completion provider."
}
