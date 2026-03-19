//! Tracing instrumentation for [`CompletionModel`] implementations.
//!
//! Wraps any `CompletionModel` in a [`TracingCompletionModel`] that emits
//! structured `tracing` spans for every LLM call.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_llm::{
    BlazenError, CompletionModel, CompletionRequest, CompletionResponse, StreamChunk,
};
use futures_util::Stream;
use tracing::Instrument;

/// A wrapper around any [`CompletionModel`] that adds tracing spans.
///
/// Each call to [`complete`](CompletionModel::complete) or
/// [`stream`](CompletionModel::stream) is wrapped in an `info_span!` that
/// records the provider name, model id, token usage, duration, and finish
/// reason.
pub struct TracingCompletionModel<M> {
    inner: M,
    provider_name: &'static str,
}

impl<M> TracingCompletionModel<M> {
    /// Create a new tracing wrapper around an existing model.
    pub fn new(inner: M, provider_name: &'static str) -> Self {
        Self {
            inner,
            provider_name,
        }
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
impl<M: CompletionModel> CompletionModel for TracingCompletionModel<M> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let span = tracing::info_span!(
            "llm.complete",
            provider = self.provider_name,
            model = %self.inner.model_id(),
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        );

        let start = std::time::Instant::now();
        let result = self.inner.complete(request).instrument(span.clone()).await;
        let elapsed = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        span.record("duration_ms", elapsed);

        if let Ok(ref response) = result {
            if let Some(ref usage) = response.usage {
                span.record("prompt_tokens", usage.prompt_tokens);
                span.record("completion_tokens", usage.completion_tokens);
                span.record("total_tokens", usage.total_tokens);
            }
            if let Some(ref reason) = response.finish_reason {
                span.record("finish_reason", reason.as_str());
            }
        }

        result
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let span = tracing::info_span!(
            "llm.stream",
            provider = self.provider_name,
            model = %self.inner.model_id(),
            duration_ms = tracing::field::Empty,
        );

        let start = std::time::Instant::now();
        let result = self.inner.stream(request).instrument(span.clone()).await;
        let elapsed = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        span.record("duration_ms", elapsed);

        result
    }
}
