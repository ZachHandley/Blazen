//! [`BaseProvider`] — wraps any [`crate::llm::CompletionModel`] with a
//! [`CompletionProviderDefaults`] that is applied to every completion
//! request before delegating to the inner model.
//!
//! Mirrors [`blazen_llm::providers::base::BaseProvider`]. Used as the
//! foundation for Phase B's `CustomProvider`, and accessible directly so
//! foreign-language callers can wrap any built-in provider with a system
//! prompt + default tools + `response_format`.
//!
//! ## Builder shape
//!
//! UniFFI `Object`s are reference-counted (`Arc<Self>`), so builder methods
//! follow the clone-with-mutation pattern: each `with_*` method takes
//! `self: Arc<Self>`, clones the inner state, mutates it, and returns a new
//! `Arc<Self>`. Foreign callers see this as a fluent chain, e.g.:
//!
//! ```kotlin
//! val provider = newOpenaiCompletionModel("sk-...")
//!     .let { BaseProvider.fromCompletionModel(it) }
//!     .withSystemPrompt("be terse")
//!     .withToolsJson(toolsJson)
//! ```

use std::sync::Arc;

use blazen_llm::CompletionModel as CoreCompletionModel;
use blazen_llm::providers::base::BaseProvider as CoreBaseProvider;
use blazen_llm::types::{
    ChatMessage as CoreChatMessage, CompletionRequest as CoreCompletionRequest,
};
use parking_lot::RwLock;

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::{ChatMessage, CompletionModel};
use crate::provider_defaults::CompletionProviderDefaults;

/// A [`crate::llm::CompletionModel`] wrapped with applied
/// [`CompletionProviderDefaults`].
///
/// Construct via [`BaseProvider::from_completion_model`] (wraps an existing
/// model with no defaults) or [`BaseProvider::with_completion_defaults`]
/// (wraps with explicit defaults). Mutate via the `with_*` builder methods.
///
/// Phase B's `CustomProvider` factories will return `Arc<BaseProvider>`
/// directly; for Phase A this class is reachable by lifting any existing
/// `CompletionModel` factory result.
#[derive(uniffi::Object)]
pub struct BaseProvider {
    /// Mutable handle so the builder methods (`with_system_prompt`, etc.)
    /// can produce a new `Arc<Self>` without re-implementing every upstream
    /// builder. The lock is uncontended in practice — builders run on a
    /// single thread before the provider is shared.
    inner: RwLock<CoreBaseProvider>,
}

impl BaseProvider {
    /// Internal: wrap an upstream `BaseProvider` in the FFI handle.
    pub(crate) fn from_core(core: CoreBaseProvider) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(core),
        })
    }

    /// Internal: snapshot the current upstream `BaseProvider` (clone).
    fn snapshot(&self) -> CoreBaseProvider {
        self.inner.read().clone()
    }
}

#[uniffi::export]
impl BaseProvider {
    /// Wrap an existing [`CompletionModel`] with empty defaults.
    ///
    /// Equivalent to using the wrapped model directly, but lets callers
    /// attach defaults later via the `with_*` methods.
    #[uniffi::constructor]
    #[must_use]
    pub fn from_completion_model(model: Arc<CompletionModel>) -> Arc<Self> {
        let inner: Arc<dyn CoreCompletionModel> = Arc::clone(&model.inner);
        Self::from_core(CoreBaseProvider::new(inner))
    }

    /// Wrap a [`CompletionModel`] with explicit
    /// [`CompletionProviderDefaults`].
    #[uniffi::constructor]
    #[must_use]
    pub fn with_completion_defaults(
        model: Arc<CompletionModel>,
        defaults: CompletionProviderDefaults,
    ) -> Arc<Self> {
        let inner: Arc<dyn CoreCompletionModel> = Arc::clone(&model.inner);
        let core = CoreBaseProvider::with_defaults(inner, defaults.into());
        Self::from_core(core)
    }

    /// Replace the entire [`CompletionProviderDefaults`] on this provider,
    /// returning a new `Arc<BaseProvider>` (clone-with-mutation).
    #[must_use]
    pub fn with_defaults(self: Arc<Self>, defaults: CompletionProviderDefaults) -> Arc<Self> {
        let next = self.snapshot().set_defaults(defaults.into());
        Self::from_core(next)
    }

    /// Set the default system prompt.
    #[must_use]
    pub fn with_system_prompt(self: Arc<Self>, prompt: String) -> Arc<Self> {
        let next = self.snapshot().with_system_prompt(prompt);
        Self::from_core(next)
    }

    /// Set the default tools (JSON-encoded `Vec<ToolDefinition>`).
    ///
    /// Malformed JSON is treated as an empty tool list — matching the
    /// upstream `#[derive(Default)]` semantics. Foreign callers should
    /// validate the JSON before sending it across the FFI.
    #[must_use]
    pub fn with_tools_json(self: Arc<Self>, tools_json: String) -> Arc<Self> {
        let tools = if tools_json.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&tools_json).unwrap_or_default()
        };
        let next = self.snapshot().with_tools(tools);
        Self::from_core(next)
    }

    /// Set the default `response_format` (JSON-encoded `serde_json::Value`).
    ///
    /// Malformed JSON or an empty string is treated as JSON null.
    #[must_use]
    pub fn with_response_format_json(self: Arc<Self>, fmt_json: String) -> Arc<Self> {
        let value: serde_json::Value = if fmt_json.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&fmt_json).unwrap_or(serde_json::Value::Null)
        };
        let next = self.snapshot().with_response_format(value);
        Self::from_core(next)
    }

    /// Inspect the currently-configured defaults (data only — hooks are
    /// not surfaced in Phase A).
    #[must_use]
    pub fn defaults(self: Arc<Self>) -> CompletionProviderDefaults {
        CompletionProviderDefaults::from(self.inner.read().defaults())
    }

    /// The model id of the wrapped inner `CompletionModel`.
    #[must_use]
    pub fn model_id(self: Arc<Self>) -> String {
        self.inner.read().model_id().to_owned()
    }

    /// Unwrap to a plain [`CompletionModel`] handle that applies the
    /// configured defaults on every call.
    ///
    /// Use this when you want to pass the wrapped provider to an API that
    /// takes a generic `CompletionModel` (the agent runner, workflow
    /// steps, etc.).
    #[must_use]
    pub fn as_completion_model(self: Arc<Self>) -> Arc<CompletionModel> {
        // BaseProvider implements CompletionModel — wrap it in the FFI
        // CompletionModel handle so the wrapping defaults are applied on
        // every call.
        let core = self.snapshot();
        let inner: Arc<dyn CoreCompletionModel> = Arc::new(core);
        CompletionModel::from_arc(inner)
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl BaseProvider {
    /// Extract structured output from the model by constraining its
    /// response to a JSON Schema.
    ///
    /// Mirrors the upstream
    /// [`blazen_llm::traits::StructuredOutput::extract`] blanket impl: the
    /// `schema_json` is injected as the request's `response_format` and
    /// the completion is dispatched as usual. Returns the model's raw
    /// content (which the foreign caller deserializes into its own typed
    /// shape — UniFFI cannot return a generic typed value across the FFI).
    ///
    /// `schema_json` must be a valid JSON Schema string; an empty string or
    /// malformed JSON falls back to `null` (the request is sent without a
    /// `response_format`).
    pub async fn extract(
        self: Arc<Self>,
        schema_json: String,
        messages: Vec<ChatMessage>,
    ) -> BlazenResult<String> {
        // Parse the schema. An empty schema string degrades gracefully into
        // an unconstrained request rather than an error — matches the
        // permissive shape of the other `*_json` helpers on this type.
        let response_format = if schema_json.trim().is_empty() {
            None
        } else {
            Some(serde_json::from_str::<serde_json::Value>(&schema_json)?)
        };

        let core_messages = messages
            .into_iter()
            .map(CoreChatMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        // The defaults-aware `BaseProvider::complete` upstream will apply
        // any configured `response_format` ONLY when the request itself
        // doesn't already specify one — `extract` always supplies one, so
        // the caller's schema wins.
        let request = CoreCompletionRequest {
            messages: core_messages,
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: None,
            top_p: None,
            response_format,
            model: None,
            modalities: None,
            image_config: None,
            audio_config: None,
        };

        let core = self.snapshot();
        let response = core.complete(request).await.map_err(BlazenError::from)?;
        Ok(response.content.unwrap_or_default())
    }
}
