//! `wasm-bindgen` wrapper for [`blazen_llm::providers::base::BaseProvider`].
//!
//! V1 surface: a chainable builder that JS callers obtain indirectly (Phase B
//! adds the `CustomProvider.withDispatch` factory that produces one). The
//! wrapper has no `#[wasm_bindgen(constructor)]` because WASM does not
//! support `class extends BaseProvider` — JS users compose behavior through
//! the dispatch/factory surface, not subclassing.
//!
//! Hook fields are stored as [`js_sys::Function`] references; actual
//! invocation is wired in Phase B alongside `CustomProvider` dispatch.
//!
//! Phase B adds [`WasmBaseProvider::extract`] for typed structured-output
//! extraction: takes a JSON Schema object and a list of chat messages,
//! returns the parsed JSON payload as a JS value. Only works when the
//! wrapper carries an inner completion model (i.e. when constructed via
//! one of the Phase B factories on `WasmCustomProvider`); otherwise
//! returns a clear "no inner completion model" error.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use js_sys::{Array, Function};
use wasm_bindgen::prelude::*;

use blazen_llm::traits::CompletionModel;
use blazen_llm::types::CompletionRequest;

use super::defaults::WasmCompletionProviderDefaults;

/// Provider wrapper that carries instance-level
/// [`WasmCompletionProviderDefaults`] applied to every completion call.
///
/// JS does not construct this directly. Phase B exposes factories on
/// `CustomProvider` (and other providers) that return a `BaseProvider`
/// handle which the user then configures via the builder methods below.
#[wasm_bindgen(js_name = "BaseProvider")]
pub struct WasmBaseProvider {
    defaults: Rc<RefCell<WasmCompletionProviderDefaults>>,
    model_id: Rc<RefCell<String>>,
    provider_id: Rc<RefCell<String>>,
    /// Inner completion model. `None` when the wrapper is built bare (no
    /// associated provider), `Some` when a Phase B factory on
    /// `WasmCustomProvider` (or another provider) constructs the wrapper
    /// from a concrete provider. Drives [`WasmBaseProvider::extract`].
    inner: Rc<RefCell<Option<Arc<dyn CompletionModel>>>>,
}

impl Default for WasmBaseProvider {
    fn default() -> Self {
        Self {
            defaults: Rc::new(RefCell::new(WasmCompletionProviderDefaults::default())),
            model_id: Rc::new(RefCell::new(String::new())),
            provider_id: Rc::new(RefCell::new(String::new())),
            inner: Rc::new(RefCell::new(None)),
        }
    }
}

impl Clone for WasmBaseProvider {
    fn clone(&self) -> Self {
        Self {
            defaults: Rc::clone(&self.defaults),
            model_id: Rc::clone(&self.model_id),
            provider_id: Rc::clone(&self.provider_id),
            inner: Rc::clone(&self.inner),
        }
    }
}

#[wasm_bindgen(js_class = "BaseProvider")]
impl WasmBaseProvider {
    /// Set the default system prompt. Chainable.
    #[wasm_bindgen(js_name = "withSystemPrompt")]
    #[must_use]
    pub fn with_system_prompt(self, s: String) -> WasmBaseProvider {
        self.defaults.borrow().set_system_prompt(Some(s));
        self
    }

    /// Set the default tools (JS array of tool definitions). Chainable.
    #[wasm_bindgen(js_name = "withTools")]
    #[must_use]
    pub fn with_tools(self, tools: Array) -> WasmBaseProvider {
        self.defaults.borrow().set_tools(tools);
        self
    }

    /// Set the default `responseFormat` (any JS object). Chainable.
    #[wasm_bindgen(js_name = "withResponseFormat")]
    #[must_use]
    pub fn with_response_format(self, fmt: JsValue) -> WasmBaseProvider {
        self.defaults.borrow().set_response_format(fmt);
        self
    }

    /// Set the universal `before_request` hook on the embedded base
    /// defaults. Chainable.
    #[wasm_bindgen(js_name = "withBeforeRequest")]
    #[must_use]
    pub fn with_before_request(self, hook: Function) -> WasmBaseProvider {
        self.defaults.borrow().base().set_before_request(Some(hook));
        self
    }

    /// Set the typed `before_completion` hook. Chainable.
    #[wasm_bindgen(js_name = "withBeforeCompletion")]
    #[must_use]
    pub fn with_before_completion(self, hook: Function) -> WasmBaseProvider {
        self.defaults.borrow().set_before_completion(Some(hook));
        self
    }

    /// Replace the entire [`WasmCompletionProviderDefaults`]. Chainable.
    #[wasm_bindgen(js_name = "withDefaults")]
    #[must_use]
    pub fn with_defaults(self, d: WasmCompletionProviderDefaults) -> WasmBaseProvider {
        *self.defaults.borrow_mut() = d;
        self
    }

    /// Inspect the configured defaults.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn defaults(&self) -> WasmCompletionProviderDefaults {
        self.defaults.borrow().clone()
    }

    /// The default model identifier for this provider.
    #[wasm_bindgen(getter, js_name = "modelId")]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.borrow().clone()
    }

    /// The provider identifier (e.g. `"openai"`, `"custom"`).
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.borrow().clone()
    }

    /// Typed structured extraction.
    ///
    /// Accepts a JSON Schema object describing the desired output shape and
    /// the conversation messages, sends a chat completion with the schema
    /// wired into `response_format` (using the same `{"type":
    /// "json_schema", ...}` envelope as Phase 0's
    /// [`blazen_llm::traits::StructuredOutput`]), and returns the parsed
    /// JSON response as a plain JS value.
    ///
    /// Errors when no inner completion model is configured (the wrapper
    /// was built bare), when the schema or messages payload is malformed,
    /// when the completion fails, or when the returned content is not
    /// valid JSON.
    pub async fn extract(&self, schema: JsValue, messages: JsValue) -> Result<JsValue, JsValue> {
        let inner = self
            .inner
            .borrow()
            .clone()
            .ok_or_else(|| JsValue::from_str("extract() requires an inner completion model; build the BaseProvider via a CustomProvider factory"))?;

        let schema_json: serde_json::Value = serde_wasm_bindgen::from_value(schema)
            .map_err(|e| JsValue::from_str(&format!("invalid JSON Schema object: {e}")))?;
        let msgs = crate::chat_message::js_messages_to_vec(&messages)?;

        // Mirror Phase 0's blanket StructuredOutput::extract: temperature
        // pinned to 0, schema fed into response_format directly. This is
        // the same envelope blazen_llm uses internally.
        let mut request = CompletionRequest::new(msgs);
        request.temperature = Some(0.0);
        request = request.with_response_format(schema_json);

        let response = inner
            .complete(request)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let content = response
            .content
            .ok_or_else(|| JsValue::from_str("extract: completion returned no content"))?;
        let parsed: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            JsValue::from_str(&format!("extract: response was not valid JSON: {e}"))
        })?;

        serde_wasm_bindgen::to_value(&parsed).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl WasmBaseProvider {
    /// Internal constructor used by Phase B factories on `CustomProvider`
    /// and other providers. Not exposed to JS — JS callers obtain a
    /// `BaseProvider` handle by calling those factories.
    #[must_use]
    #[allow(dead_code)] // Used by Phase B factories (CustomProvider.withDispatch, etc).
    pub(crate) fn new_internal(
        defaults: WasmCompletionProviderDefaults,
        model_id: String,
        provider_id: String,
    ) -> Self {
        Self {
            defaults: Rc::new(RefCell::new(defaults)),
            model_id: Rc::new(RefCell::new(model_id)),
            provider_id: Rc::new(RefCell::new(provider_id)),
            inner: Rc::new(RefCell::new(None)),
        }
    }

    /// Internal constructor that attaches an inner completion model. Used
    /// by Phase B `WasmCustomProvider` factories so the resulting
    /// `WasmBaseProvider` handle can power [`Self::extract`].
    #[must_use]
    pub(crate) fn new_with_inner(
        defaults: WasmCompletionProviderDefaults,
        model_id: String,
        provider_id: String,
        inner: Arc<dyn CompletionModel>,
    ) -> Self {
        Self {
            defaults: Rc::new(RefCell::new(defaults)),
            model_id: Rc::new(RefCell::new(model_id)),
            provider_id: Rc::new(RefCell::new(provider_id)),
            inner: Rc::new(RefCell::new(Some(inner))),
        }
    }

    /// Crate-internal accessor returning the wrapped completion model,
    /// when one is configured.
    #[allow(dead_code)] // Used by Phase B WasmCustomProvider.
    pub(crate) fn inner_completion_model(&self) -> Option<Arc<dyn CompletionModel>> {
        self.inner.borrow().clone()
    }
}
