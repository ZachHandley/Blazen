//! `wasm-bindgen` wrapper for [`blazen_llm::providers::azure::AzureOpenAiProvider`].

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::providers::azure::AzureOpenAiProvider;
use blazen_llm::traits::CompletionModel;

use super::{
    apply_request_options, as_dyn_completion, complete_promise, fetch_client, resolve_key,
    stream_promise,
};
use crate::completion_model::WasmCompletionModel;

/// An Azure `OpenAI` chat-completion provider.
///
/// Azure does not expose a single global endpoint -- each deployment lives
/// at `https://{resourceName}.openai.azure.com/openai/deployments/{deploymentName}`.
///
/// ```js
/// const provider = new AzureOpenAiProvider({
///     resourceName: 'my-azure-resource',
///     deploymentName: 'gpt-4o-deployment',
///     apiKey: '...',
/// });
/// ```
#[wasm_bindgen(js_name = "AzureOpenAiProvider")]
pub struct WasmAzureOpenAiProvider {
    inner: Arc<AzureOpenAiProvider>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmAzureOpenAiProvider {}
unsafe impl Sync for WasmAzureOpenAiProvider {}

#[wasm_bindgen(js_class = "AzureOpenAiProvider")]
impl WasmAzureOpenAiProvider {
    /// Create a new Azure `OpenAI` provider.
    ///
    /// `options` is a plain JS object with:
    /// - `resourceName` (string, required) -- the Azure resource name (subdomain)
    /// - `deploymentName` (string, required) -- the model deployment name
    /// - `apiKey` (string) -- defaults to `AZURE_OPENAI_API_KEY` env var
    /// - `apiVersion` (string) -- override the Azure API version
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<WasmAzureOpenAiProvider, JsValue> {
        let resource_name = read_string(&options, "resourceName")
            .ok_or_else(|| JsValue::from_str("AzureOpenAiProvider: 'resourceName' is required"))?;
        let deployment_name = read_string(&options, "deploymentName").ok_or_else(|| {
            JsValue::from_str("AzureOpenAiProvider: 'deploymentName' is required")
        })?;
        let api_key_opt = read_string(&options, "apiKey");
        let api_version_opt = read_string(&options, "apiVersion");

        let api_key = resolve_key("azure", api_key_opt)?;
        let mut provider = AzureOpenAiProvider::new_with_client(
            api_key,
            resource_name,
            deployment_name,
            fetch_client(),
        );
        if let Some(v) = api_version_opt {
            provider = provider.with_api_version(v);
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
