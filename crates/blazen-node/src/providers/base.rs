//! JavaScript binding for [`blazen_llm::providers::base::BaseProvider`].
//!
//! Exposes [`JsBaseProvider`] as a NAPI class with a constructor (so JS
//! users can subclass it via `class MyLLM extends BaseProvider`) and
//! builder methods that mirror the Rust [`BaseProvider`] surface:
//! `withSystemPrompt`, `withTools`, `withResponseFormat`,
//! `withBeforeRequest`, `withBeforeCompletion`, `withDefaults`.
//!
//! ## V1 status
//!
//! For V1, the constructor accepts an `inner` JS object (typically a
//! [`JsCompletionModel`] handle) and an optional
//! [`JsCompletionProviderDefaults`]. The `inner` object is stashed for
//! Phase D's real subclass-detection wiring — today the builder methods
//! mutate the defaults snapshot, and the `inner` slot lets the
//! framework recover the underlying Rust [`CompletionModel`] when one
//! is present.

use std::sync::Arc;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::CompletionModel;
use blazen_llm::providers::base::BaseProvider;
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::llm_error_to_napi;
use crate::generated::JsToolDefinition;
use crate::providers::completion_model::JsCompletionModel;
use crate::providers::defaults::{
    BeforeCompletionTsfn, BeforeRequestTsfn, JsCompletionProviderDefaults,
};
use crate::types::JsChatMessage;

// ---------------------------------------------------------------------------
// JsBaseProvider
// ---------------------------------------------------------------------------

/// A completion provider wrapper that applies a
/// [`JsCompletionProviderDefaults`] to every completion request before
/// delegating to the inner model.
///
/// `BaseProvider` is intended to be subclassed from JavaScript:
///
/// ```javascript
/// import { BaseProvider, CompletionModel } from "blazen";
///
/// class TerseLlm extends BaseProvider {
///   constructor() {
///     const inner = CompletionModel.openai({ apiKey: "sk-..." });
///     super(inner);
///     this.withSystemPrompt("Be terse.");
///   }
/// }
/// ```
///
/// Today (V1) the constructor stores an opaque reference to the inner
/// object — Phase D will wire `class extends` to fire the JS `complete`
/// override before falling back to the inner Rust model.
#[napi(js_name = "BaseProvider")]
pub struct JsBaseProvider {
    /// The configured defaults. Stored behind a Mutex so the builder
    /// methods can mutate in place without taking `self` by value (napi
    /// methods take `&self`).
    pub(crate) defaults: Arc<Mutex<JsCompletionProviderDefaults>>,
    /// Optional underlying Rust completion model. Populated when the
    /// constructor receives a [`JsCompletionModel`] with a Rust-side
    /// `inner` provider. `None` for fully-subclassed JS-only providers
    /// (Phase D will wire that path).
    pub(crate) inner: Option<Arc<dyn CompletionModel>>,
    /// The provider ID for logging and introspection. Defaults to the
    /// inner model's ID when present, otherwise `"base"`.
    pub(crate) provider_id: Arc<Mutex<String>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::return_self_not_must_use
)]
impl JsBaseProvider {
    /// Construct a new [`BaseProvider`].
    ///
    /// `inner` is the underlying completion model — pass a
    /// [`JsCompletionModel`] instance. JS subclasses that fully
    /// override `complete` may pass `null` here (Phase D will wire
    /// subclass dispatch end-to-end; today calls to `complete` on a
    /// subclass-only provider report unsupported).
    ///
    /// `defaults` optionally seeds the
    /// [`JsCompletionProviderDefaults`]; when omitted, an empty
    /// defaults bag is created.
    #[napi(constructor)]
    pub fn new(
        inner: Option<&JsCompletionModel>,
        defaults: Option<&JsCompletionProviderDefaults>,
    ) -> Result<Self> {
        let rust_inner: Option<Arc<dyn CompletionModel>> = match inner {
            Some(model) => model.inner.as_ref().map(Arc::clone),
            None => None,
        };

        let provider_id = rust_inner
            .as_ref()
            .map_or_else(|| "base".to_owned(), |m| m.model_id().to_owned());

        let defaults_owned = defaults.map_or_else(
            || JsCompletionProviderDefaults::new(None, None, None, None, None),
            JsCompletionProviderDefaults::clone_shared,
        );

        Ok(Self {
            defaults: Arc::new(Mutex::new(defaults_owned)),
            inner: rust_inner,
            provider_id: Arc::new(Mutex::new(provider_id)),
        })
    }

    // -----------------------------------------------------------------
    // Builder methods (each returns `self` reference-equivalent)
    // -----------------------------------------------------------------
    //
    // napi-rs builder methods conventionally take `&self` and return a
    // freshly-wrapped `JsBaseProvider` (sharing the same Arc<Mutex>
    // pointers) so chained JS calls behave like the Rust builder API.

    /// Set the default system prompt prepended to requests when no
    /// system message is already present.
    #[napi(js_name = "withSystemPrompt")]
    pub fn with_system_prompt(&self, prompt: String) -> Self {
        if let Ok(g) = self.defaults.lock()
            && let Ok(mut sp) = g.system_prompt.lock()
        {
            *sp = Some(prompt);
        }
        self.clone_shared()
    }

    /// Replace the default tools appended to every completion request.
    #[napi(js_name = "withTools")]
    pub fn with_tools(&self, tools: Vec<JsToolDefinition>) -> Self {
        if let Ok(g) = self.defaults.lock()
            && let Ok(mut t) = g.tools.lock()
        {
            *t = tools;
        }
        self.clone_shared()
    }

    /// Set the default `responseFormat` (JSON Schema object).
    #[napi(js_name = "withResponseFormat")]
    pub fn with_response_format(&self, format: serde_json::Value) -> Self {
        if let Ok(g) = self.defaults.lock()
            && let Ok(mut rf) = g.response_format.lock()
        {
            *rf = Some(format);
        }
        self.clone_shared()
    }

    /// Set the universal `beforeRequest` hook (fires for any request
    /// type). V1: stored only — Phase B wires dispatch.
    #[napi(js_name = "withBeforeRequest")]
    pub fn with_before_request(&self, hook: BeforeRequestTsfn) -> Self {
        if let Ok(g) = self.defaults.lock()
            && let Ok(base_g) = g.base.lock()
            && let Ok(mut br) = base_g.before_request.lock()
        {
            *br = Some(hook);
        }
        self.clone_shared()
    }

    /// Set the typed `beforeCompletion` hook (fires after the universal
    /// hook, with a typed completion request). V1: stored only — Phase
    /// B wires dispatch.
    #[napi(js_name = "withBeforeCompletion")]
    pub fn with_before_completion(&self, hook: BeforeCompletionTsfn) -> Self {
        if let Ok(g) = self.defaults.lock()
            && let Ok(mut bc) = g.before_completion.lock()
        {
            *bc = Some(hook);
        }
        self.clone_shared()
    }

    /// Replace the entire defaults bag.
    #[napi(js_name = "withDefaults")]
    pub fn with_defaults(&self, defaults: &JsCompletionProviderDefaults) -> Self {
        if let Ok(mut g) = self.defaults.lock() {
            *g = defaults.clone_shared();
        }
        self.clone_shared()
    }

    // -----------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------

    /// The currently-configured defaults.
    #[napi(getter)]
    pub fn defaults(&self) -> JsCompletionProviderDefaults {
        self.defaults.lock().map_or_else(
            |_| JsCompletionProviderDefaults::new(None, None, None, None, None),
            |g| g.clone_shared(),
        )
    }

    /// The inner model's `modelId`. Returns the empty string when the
    /// provider was constructed without a Rust-side `inner` (JS subclass
    /// path).
    #[napi(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner
            .as_ref()
            .map(|m| m.model_id().to_owned())
            .unwrap_or_default()
    }

    /// The provider identifier used for logging. Defaults to the inner
    /// model's `modelId` when present, otherwise `"base"`. Subclasses
    /// may override.
    #[napi(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id
            .lock()
            .map_or_else(|_| "base".to_owned(), |g| g.clone())
    }

    // -----------------------------------------------------------------
    // Typed structured extraction
    // -----------------------------------------------------------------

    /// Typed structured extraction.
    ///
    /// Sends a completion request with a JSON Schema `response_format`
    /// envelope and parses the model's response as JSON. The schema
    /// argument is a plain JSON Schema object (callers using zod can
    /// convert with `zodToJsonSchema(zSchema)` from the `zod-to-json-schema`
    /// package).
    ///
    /// The `response_format` is wired up as the `OpenAI`-style
    /// `{"type":"json_schema","json_schema":{"name":"Extract","schema":...,"strict":true}}`
    /// envelope; provider implementations that don't natively support
    /// structured outputs fall back to a system-instruction shim (see
    /// `crates/blazen-llm/src/providers/anthropic.rs::build_json_schema_system_instruction`).
    ///
    /// Returns the parsed JSON value. The TypeScript surface declares
    /// the return as `any` because the schema shape is only known at
    /// runtime; callers can narrow via TS generics on their wrapper.
    ///
    /// ```typescript
    /// const schema = {
    ///   type: "object",
    ///   properties: {
    ///     name: { type: "string" },
    ///     age:  { type: "integer" },
    ///   },
    ///   required: ["name", "age"],
    /// };
    /// const result = await provider.extract(schema, [
    ///   ChatMessage.user("My name is Alice and I am 30."),
    /// ]);
    /// // -> { name: "Alice", age: 30 }
    /// ```
    #[napi(js_name = "extract")]
    pub async fn extract(
        &self,
        schema: serde_json::Value,
        messages: Vec<&JsChatMessage>,
    ) -> Result<serde_json::Value> {
        // BaseProvider.extract requires a concrete underlying Rust model.
        // Subclass-only providers (no `inner`) hit the Phase D dispatch path
        // for `complete`; until that lands `extract` is unsupported for them.
        let inner = self.inner.clone().ok_or_else(|| {
            napi::Error::from_reason(
                "BaseProvider.extract requires a concrete inner CompletionModel; subclass-only providers should override `complete` and call `extract` from there",
            )
        })?;

        // Snapshot the defaults bag so we get the same system_prompt /
        // tools / response_format treatment as `complete()` would. We
        // route through `BaseProvider::with_defaults` to share the apply
        // logic that lives next to the production code path.
        let defaults_rust = self
            .defaults
            .lock()
            .ok()
            .map(|g| g.to_rust())
            .unwrap_or_default();
        let provider = BaseProvider::with_defaults(inner, defaults_rust);

        // Build the response_format envelope expected by every provider's
        // schema adapter. Using a fixed name keeps the wire shape stable;
        // callers who need a custom name can pass a fully-formed envelope
        // by stuffing it into the request manually via withResponseFormat.
        let response_format = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "Extract",
                "schema": schema,
                "strict": true,
            },
        });

        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages).with_response_format(response_format);

        let response = provider
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;

        // Pull the assistant content and parse as JSON. An empty/None
        // body is reported as a clear error instead of silently returning
        // null — callers expect structured data here.
        let content = response.content.unwrap_or_default();
        if content.trim().is_empty() {
            return Err(napi::Error::from_reason(
                "BaseProvider.extract: model returned empty content; unable to parse JSON",
            ));
        }
        serde_json::from_str::<serde_json::Value>(&content).map_err(|e| {
            napi::Error::from_reason(format!(
                "BaseProvider.extract: failed to parse model response as JSON ({e}); raw content: {content}"
            ))
        })
    }
}

impl JsBaseProvider {
    /// Internal: snapshot into a Rust [`BaseProvider`] for downstream
    /// dispatch. Returns `None` if no inner Rust model is configured
    /// (subclass-only path; Phase D will route those through the JS
    /// `complete` override).
    #[allow(dead_code)]
    pub(crate) fn to_rust(&self) -> Option<BaseProvider> {
        let inner = self.inner.as_ref().map(Arc::clone)?;
        let defaults = self
            .defaults
            .lock()
            .ok()
            .map(|g| g.to_rust())
            .unwrap_or_default();
        Some(BaseProvider::with_defaults(inner, defaults))
    }

    /// Clone the JS wrapper preserving the shared `Arc<Mutex>` slots so
    /// builder chains share state with the original instance.
    fn clone_shared(&self) -> Self {
        Self {
            defaults: Arc::clone(&self.defaults),
            inner: self.inner.as_ref().map(Arc::clone),
            provider_id: Arc::clone(&self.provider_id),
        }
    }
}

// Suppress unused-import warning: `ToolDefinition` is referenced indirectly
// through `JsToolDefinition`'s field shape but the bridge code never names
// the Rust type. Re-export to keep the symbol path discoverable.
#[allow(dead_code)]
type _ToolDefinitionAlias = ToolDefinition;
