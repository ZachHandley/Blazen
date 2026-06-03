//! [`LlmProviderDefaults`] — wraps any LLM provider with a
//! [`ProviderDefaults`] that is applied to every completion
//! request before delegating to the inner provider.
//!
//! Mirrors [`blazen_llm::providers::base::LlmProviderDefaults`]. Used as the
//! foundation for Phase B's `CustomProvider`, and accessible internally so
//! the `CustomProvider` builder chain can carry through the system prompt /
//! default tools / `response_format` slots without re-implementing them.
//!
//! ## Builder shape
//!
//! UniFFI `Object`s are reference-counted (`Arc<Self>`), so builder methods
//! follow the clone-with-mutation pattern: each `with_*` method takes
//! `self: Arc<Self>`, clones the inner state, mutates it, and returns a new
//! `Arc<Self>`.

use std::sync::Arc;

use blazen_llm::Model as CoreModel;
use blazen_llm::providers::base::LlmProviderDefaults as CoreBaseProvider;
use blazen_llm::types::{ChatMessage as CoreChatMessage, ModelRequest as CoreModelRequest};
use parking_lot::RwLock;

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::ChatMessage;
use crate::provider_defaults::ProviderDefaults;

/// An LLM provider wrapped with applied [`ProviderDefaults`].
///
/// Constructed internally by the `CustomProvider` machinery to carry
/// builder-style defaults (`with_system_prompt`, etc.) alongside the inner
/// provider handle. Mutate via the `with_*` builder methods.
#[derive(uniffi::Object)]
pub struct LlmProviderDefaults {
    /// Mutable handle so the builder methods (`with_system_prompt`, etc.)
    /// can produce a new `Arc<Self>` without re-implementing every upstream
    /// builder. The lock is uncontended in practice — builders run on a
    /// single thread before the provider is shared.
    inner: RwLock<CoreBaseProvider>,
}

impl LlmProviderDefaults {
    /// Internal: wrap an upstream `LlmProviderDefaults` in the FFI handle.
    pub(crate) fn from_core(core: CoreBaseProvider) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(core),
        })
    }

    /// Internal: snapshot the current upstream `LlmProviderDefaults` (clone).
    fn snapshot(&self) -> CoreBaseProvider {
        self.inner.read().clone()
    }
}

#[uniffi::export]
impl LlmProviderDefaults {
    /// Replace the entire [`ProviderDefaults`] on this provider,
    /// returning a new `Arc<LlmProviderDefaults>` (clone-with-mutation).
    #[must_use]
    pub fn with_defaults(self: Arc<Self>, defaults: ProviderDefaults) -> Arc<Self> {
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
    pub fn defaults(self: Arc<Self>) -> ProviderDefaults {
        ProviderDefaults::from(self.inner.read().defaults())
    }

    /// The model id of the wrapped inner provider.
    #[must_use]
    pub fn model_id(self: Arc<Self>) -> String {
        self.inner.read().model_id().to_owned()
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl LlmProviderDefaults {
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

        // The defaults-aware `LlmProviderDefaults::complete` upstream will apply
        // any configured `response_format` ONLY when the request itself
        // doesn't already specify one — `extract` always supplies one, so
        // the caller's schema wins.
        let request = CoreModelRequest {
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
            tool_choice: None,
        };

        let core = self.snapshot();
        let response = core.complete(request).await.map_err(BlazenError::from)?;
        Ok(response.content.unwrap_or_default())
    }
}
