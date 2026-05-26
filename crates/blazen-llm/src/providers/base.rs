//! [`LlmProviderDefaults`] — wraps any [`Model`] with a
//! [`ProviderDefaults`] that is applied to every completion request
//! before delegating to the inner model.
//!
//! Used as the foundation for [`crate::providers::custom::CustomProvider`]
//! and accessible directly so users can wrap any other built-in provider
//! (`OpenAI`, Anthropic, Gemini, etc.) with a system prompt + default tools +
//! `response_format` + before-request hooks.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use crate::error::BlazenError;
use crate::http::HttpClient;
use crate::providers::defaults::{
    BaseProviderDefaults, BeforeModelRequestHook, BeforeRequestHook, ProviderDefaults,
};
use crate::retry::RetryConfig;
use crate::traits::Model;
use crate::types::{ModelRequest, ModelResponse, StreamChunk, ToolDefinition};

/// Wraps any `Model` with instance-level defaults that are
/// applied to every `complete()` / `stream()` call before delegation.
pub struct LlmProviderDefaults {
    inner: Arc<dyn Model>,
    defaults: ProviderDefaults,
}

impl std::fmt::Debug for LlmProviderDefaults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmProviderDefaults")
            .field("model_id", &self.inner.model_id())
            .field("defaults", &self.defaults)
            .finish()
    }
}

impl Clone for LlmProviderDefaults {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            defaults: self.defaults.clone(),
        }
    }
}

impl LlmProviderDefaults {
    /// Construct with no defaults — equivalent to using `inner` directly,
    /// but lets the user attach defaults via builder methods.
    #[must_use]
    pub fn new(inner: Arc<dyn Model>) -> Self {
        Self {
            inner,
            defaults: ProviderDefaults::default(),
        }
    }

    /// Construct with explicit defaults.
    #[must_use]
    pub fn with_defaults(inner: Arc<dyn Model>, defaults: ProviderDefaults) -> Self {
        Self { inner, defaults }
    }

    /// Replace the entire `ProviderDefaults`.
    #[must_use]
    pub fn set_defaults(mut self, defaults: ProviderDefaults) -> Self {
        self.defaults = defaults;
        self
    }

    /// Replace the universal `BaseProviderDefaults` (preserving role-specific fields).
    #[must_use]
    pub fn set_base_defaults(mut self, base: BaseProviderDefaults) -> Self {
        self.defaults.base = base;
        self
    }

    #[must_use]
    pub fn with_system_prompt(mut self, s: impl Into<String>) -> Self {
        self.defaults.system_prompt = Some(s.into());
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.defaults.tools = tools;
        self
    }

    #[must_use]
    pub fn with_response_format(mut self, fmt: serde_json::Value) -> Self {
        self.defaults.response_format = Some(fmt);
        self
    }

    #[must_use]
    pub fn with_before_request(mut self, hook: BeforeRequestHook) -> Self {
        self.defaults.base.before_request = Some(hook);
        self
    }

    #[must_use]
    pub fn with_before_model(mut self, hook: BeforeModelRequestHook) -> Self {
        self.defaults.before_model = Some(hook);
        self
    }

    /// Inspect the configured defaults.
    #[must_use]
    pub fn defaults(&self) -> &ProviderDefaults {
        &self.defaults
    }

    /// Inspect the inner model.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn Model> {
        &self.inner
    }
}

#[async_trait]
impl Model for LlmProviderDefaults {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.inner.retry_config()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.inner.http_client()
    }

    async fn complete(&self, mut request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        self.defaults.apply(&mut request).await?;
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        mut request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        self.defaults.apply(&mut request).await?;
        self.inner.stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, MessageContent, Role};

    /// Build a minimal `ModelResponse` for tests (the type has no `Default` impl).
    fn empty_response(model: &str) -> ModelResponse {
        ModelResponse {
            content: None,
            tool_calls: Vec::new(),
            reasoning: None,
            citations: Vec::new(),
            artifacts: Vec::new(),
            usage: None,
            model: model.to_owned(),
            finish_reason: None,
            cost: None,
            timing: None,
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Minimal echo model — returns the last user message text as content.
    struct EchoModel;

    #[async_trait]
    impl Model for EchoModel {
        fn model_id(&self) -> &'static str {
            "echo"
        }

        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
            let last = request
                .messages
                .last()
                .map(|m| match &m.content {
                    MessageContent::Text(t) => t.clone(),
                    _ => String::new(),
                })
                .unwrap_or_default();
            let mut resp = empty_response("echo");
            resp.content = Some(last);
            Ok(resp)
        }

        async fn stream(
            &self,
            _request: ModelRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            Err(BlazenError::unsupported("EchoModel does not stream"))
        }
    }

    #[tokio::test]
    async fn base_provider_prepends_system_prompt() {
        let bp = LlmProviderDefaults::new(Arc::new(EchoModel)).with_system_prompt("be terse");
        let req = ModelRequest::new(vec![ChatMessage::user("hi")]);
        // EchoModel returns the LAST message; since defaults insert the system
        // message at the front, the last message remains "hi".
        let resp = bp.complete(req).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn base_provider_applies_defaults_before_delegation() {
        use std::sync::Mutex;

        struct RecordingModel(Arc<Mutex<Vec<ModelRequest>>>);

        #[async_trait]
        impl Model for RecordingModel {
            fn model_id(&self) -> &'static str {
                "rec"
            }
            async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
                self.0.lock().unwrap().push(request);
                Ok(empty_response("rec"))
            }
            async fn stream(
                &self,
                _r: ModelRequest,
            ) -> Result<
                Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>,
                BlazenError,
            > {
                Err(BlazenError::unsupported(""))
            }
        }

        let recorded: Arc<Mutex<Vec<ModelRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let bp = LlmProviderDefaults::new(Arc::new(RecordingModel(Arc::clone(&recorded))))
            .with_system_prompt("be helpful")
            .with_response_format(serde_json::json!({"type": "json_object"}));

        let req = ModelRequest::new(vec![ChatMessage::user("hi")]);
        bp.complete(req).await.unwrap();

        let seen = recorded.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let r = &seen[0];
        assert_eq!(
            r.messages.len(),
            2,
            "system message should have been prepended"
        );
        assert!(matches!(r.messages[0].role, Role::System));
        assert!(r.response_format.is_some());
    }
}
