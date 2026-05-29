//! Tracing instrumentation for [`Model`] implementations.
//!
//! Wraps any `Model` in a [`TracingModel`] that emits structured `tracing`
//! spans for every LLM call. Spans carry both Blazen-flavored attribute names
//! (`provider`, `model`, `prompt_tokens`, …) and `OpenInference` / `gen_ai.*`
//! aliases (`openinference.span.kind`, `gen_ai.system`, `gen_ai.request.model`,
//! `gen_ai.usage.input_tokens`, …) so the same span is eval-grade in Phoenix
//! Arize and ingestable by `PostHog` AI without any post-processing.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_llm::{BlazenError, Model, ModelRequest, ModelResponse, StreamChunk};
use futures_util::Stream;
use tracing::Instrument;

/// Runtime configuration for a [`TracingModel`].
///
/// Defaults are privacy-safe: token counts, model id, provider, and finish
/// reason are always recorded; raw prompt + completion message text are NOT
/// recorded unless [`TracingConfig::with_message_capture`] is opted into.
#[derive(Debug, Clone, Copy, Default)]
pub struct TracingConfig {
    capture_messages: bool,
}

impl TracingConfig {
    /// Enable or disable raw prompt + completion message capture as span
    /// attributes (`llm.input_messages` / `llm.output_messages`).
    ///
    /// When enabled, the wrapper serializes the incoming
    /// [`ModelRequest::messages`] and the outgoing [`ModelResponse::content`]
    /// to JSON and attaches them to the span. This is what Phoenix's
    /// eval-grade surfaces (faithfulness, schema-compliance, RAG hit rate)
    /// need. It is also what tells you what your model actually said — so
    /// privacy-sensitive deployments should leave this off.
    ///
    /// Stream calls do NOT capture messages even with this flag — accumulating
    /// stream chunks for span attribution is intentionally out of scope.
    #[must_use]
    pub fn with_message_capture(mut self, capture: bool) -> Self {
        self.capture_messages = capture;
        self
    }

    /// Returns whether raw messages are captured.
    #[must_use]
    pub fn capture_messages(&self) -> bool {
        self.capture_messages
    }
}

/// A wrapper around any [`Model`] that adds tracing spans.
///
/// Each call to [`complete`](Model::complete) or [`stream`](Model::stream) is
/// wrapped in an `info_span!` that records the provider name, model id, token
/// usage, duration, finish reason, plus `OpenInference` + `gen_ai.*` aliases for
/// eval-grade ingest by Phoenix Arize and `PostHog` AI.
pub struct TracingModel<M> {
    inner: M,
    provider_name: &'static str,
    config: TracingConfig,
}

impl<M> TracingModel<M> {
    /// Create a new tracing wrapper with an explicit [`TracingConfig`].
    pub fn new(inner: M, provider_name: &'static str, config: TracingConfig) -> Self {
        Self {
            inner,
            provider_name,
            config,
        }
    }

    /// Create a new tracing wrapper with default config (no message capture).
    ///
    /// Equivalent to `TracingModel::new(inner, provider_name,
    /// TracingConfig::default())`.
    pub fn new_default(inner: M, provider_name: &'static str) -> Self {
        Self::new(inner, provider_name, TracingConfig::default())
    }

    /// Get a reference to the inner model.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Consume the wrapper and return the inner model.
    pub fn into_inner(self) -> M {
        self.inner
    }
}

#[async_trait]
impl<M: Model> Model for TracingModel<M> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        let model_id = self.inner.model_id().to_string();
        let span = tracing::info_span!(
            "llm.complete",
            // Blazen-flavored attribute names (back-compat).
            provider = self.provider_name,
            model = %model_id,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
            // `OpenInference` + gen_ai.* aliases (eval-grade ingest).
            openinference.span.kind = "LLM",
            gen_ai.system = self.provider_name,
            gen_ai.request.model = %model_id,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.response.finish_reasons = tracing::field::Empty,
            llm.token_count.prompt = tracing::field::Empty,
            llm.token_count.completion = tracing::field::Empty,
            llm.model_name = %model_id,
            llm.input_messages = tracing::field::Empty,
            llm.output_messages = tracing::field::Empty,
        );

        if self.config.capture_messages
            && let Ok(messages_json) = serde_json::to_string(&request.messages)
        {
            span.record("llm.input_messages", messages_json.as_str());
        }

        let start = std::time::Instant::now();
        let result = self.inner.complete(request).instrument(span.clone()).await;
        let elapsed = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        span.record("duration_ms", elapsed);

        if let Ok(ref response) = result {
            if let Some(ref usage) = response.usage {
                span.record("prompt_tokens", usage.prompt_tokens);
                span.record("completion_tokens", usage.completion_tokens);
                span.record("total_tokens", usage.total_tokens);
                span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                span.record("gen_ai.usage.output_tokens", usage.completion_tokens);
                span.record("llm.token_count.prompt", usage.prompt_tokens);
                span.record("llm.token_count.completion", usage.completion_tokens);
            }
            if let Some(ref reason) = response.finish_reason {
                span.record("finish_reason", reason.as_str());
                span.record("gen_ai.response.finish_reasons", reason.as_str());
            }
            if self.config.capture_messages
                && let Some(ref content) = response.content
                && let Ok(out_json) = serde_json::to_string(content)
            {
                span.record("llm.output_messages", out_json.as_str());
            }
        }

        result
    }

    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let model_id = self.inner.model_id().to_string();
        let span = tracing::info_span!(
            "llm.stream",
            provider = self.provider_name,
            model = %model_id,
            duration_ms = tracing::field::Empty,
            openinference.span.kind = "LLM",
            gen_ai.system = self.provider_name,
            gen_ai.request.model = %model_id,
            llm.model_name = %model_id,
            llm.input_messages = tracing::field::Empty,
        );

        if self.config.capture_messages
            && let Ok(messages_json) = serde_json::to_string(&request.messages)
        {
            span.record("llm.input_messages", messages_json.as_str());
        }

        let start = std::time::Instant::now();
        let result = self.inner.stream(request).instrument(span.clone()).await;
        let elapsed = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        span.record("duration_ms", elapsed);

        result
    }
}
