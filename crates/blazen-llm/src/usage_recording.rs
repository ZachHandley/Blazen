//! Auto-emit `UsageEvent`s on every successful provider call.
//!
//! `UsageRecordingCompletionModel` and `UsageRecordingEmbeddingModel` wrap any
//! `CompletionModel` / `EmbeddingModel`, capturing the response's `usage` and
//! `cost` fields and emitting a typed `blazen_events::UsageEvent` via a
//! pluggable `UsageEmitter` after each successful call.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use uuid::Uuid;

use blazen_events::{Modality, UsageEvent};

use crate::error::BlazenError;
use crate::traits::{CompletionModel, EmbeddingModel};
use crate::types::{
    CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk, TokenUsage,
};

// ---------------------------------------------------------------------------
// Emitter trait
// ---------------------------------------------------------------------------

/// A sink for emitted `UsageEvent`s.
///
/// Implementations send the event somewhere observable: a workflow's
/// broadcast stream, a tokio channel, a tracing span field, etc.
pub trait UsageEmitter: Send + Sync + std::fmt::Debug {
    /// Emit a single usage event.
    fn emit(&self, event: UsageEvent);
}

/// A no-op emitter that drops every event. Useful as a default when no
/// downstream observer is wired up.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopUsageEmitter;

impl UsageEmitter for NoopUsageEmitter {
    fn emit(&self, _event: UsageEvent) {}
}

// ---------------------------------------------------------------------------
// CompletionModel decorator
// ---------------------------------------------------------------------------

/// Decorator that wraps a `CompletionModel` and emits `UsageEvent` after each
/// successful `complete` / `stream` call.
pub struct UsageRecordingCompletionModel {
    inner: Arc<dyn CompletionModel>,
    emitter: Arc<dyn UsageEmitter>,
    provider_label: String,
    run_id: Uuid,
}

impl UsageRecordingCompletionModel {
    /// Wrap `inner` with a usage-recording layer.
    pub fn new(
        inner: impl CompletionModel + 'static,
        emitter: Arc<dyn UsageEmitter>,
        provider_label: impl Into<String>,
        run_id: Uuid,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
            emitter,
            provider_label: provider_label.into(),
            run_id,
        }
    }

    /// Wrap an already-`Arc`'d completion model.
    #[must_use]
    pub fn from_arc(
        inner: Arc<dyn CompletionModel>,
        emitter: Arc<dyn UsageEmitter>,
        provider_label: impl Into<String>,
        run_id: Uuid,
    ) -> Self {
        Self {
            inner,
            emitter,
            provider_label: provider_label.into(),
            run_id,
        }
    }
}

#[async_trait]
impl CompletionModel for UsageRecordingCompletionModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let start = Instant::now();
        let resp = self.inner.complete(request).await?;
        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        emit_completion_usage(
            &self.emitter,
            &self.provider_label,
            self.inner.model_id(),
            resp.usage.as_ref(),
            resp.cost,
            elapsed_ms,
            self.run_id,
        );
        Ok(resp)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // Streaming path: only emit the initial connect timing as latency_ms;
        // chunks may carry usage in their stream_complete event but that
        // routing is out of scope for this decorator.
        let start = Instant::now();
        let stream = self.inner.stream(request).await?;
        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        // Emit a "connect" usage with zero token counts but latency populated.
        self.emitter.emit(UsageEvent {
            provider: self.provider_label.clone(),
            model: self.inner.model_id().to_owned(),
            modality: Modality::Llm,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            reasoning_tokens: 0,
            cached_input_tokens: 0,
            audio_input_tokens: 0,
            audio_output_tokens: 0,
            image_count: 0,
            audio_seconds: 0.0,
            video_seconds: 0.0,
            cost_usd: None,
            latency_ms: elapsed_ms,
            run_id: self.run_id,
        });
        Ok(stream)
    }
}

// ---------------------------------------------------------------------------
// EmbeddingModel decorator
// ---------------------------------------------------------------------------

/// Decorator that wraps an `EmbeddingModel` and emits `UsageEvent` after each
/// successful `embed` call.
pub struct UsageRecordingEmbeddingModel {
    inner: Arc<dyn EmbeddingModel>,
    emitter: Arc<dyn UsageEmitter>,
    provider_label: String,
    run_id: Uuid,
}

impl UsageRecordingEmbeddingModel {
    /// Wrap `inner` with a usage-recording layer.
    pub fn new(
        inner: impl EmbeddingModel + 'static,
        emitter: Arc<dyn UsageEmitter>,
        provider_label: impl Into<String>,
        run_id: Uuid,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
            emitter,
            provider_label: provider_label.into(),
            run_id,
        }
    }

    /// Wrap an already-`Arc`'d embedding model.
    #[must_use]
    pub fn from_arc(
        inner: Arc<dyn EmbeddingModel>,
        emitter: Arc<dyn UsageEmitter>,
        provider_label: impl Into<String>,
        run_id: Uuid,
    ) -> Self {
        Self {
            inner,
            emitter,
            provider_label: provider_label.into(),
            run_id,
        }
    }
}

#[async_trait]
impl EmbeddingModel for UsageRecordingEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResponse, BlazenError> {
        let start = Instant::now();
        let resp = self.inner.embed(texts).await?;
        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let tu = resp.usage.as_ref();
        self.emitter.emit(UsageEvent {
            provider: self.provider_label.clone(),
            model: self.inner.model_id().to_owned(),
            modality: Modality::Embedding,
            prompt_tokens: tu.map_or(0, |t| t.prompt_tokens),
            completion_tokens: 0,
            total_tokens: tu.map_or(0, |t| t.total_tokens),
            reasoning_tokens: 0,
            cached_input_tokens: tu.map_or(0, |t| t.cached_input_tokens),
            audio_input_tokens: 0,
            audio_output_tokens: 0,
            image_count: 0,
            audio_seconds: 0.0,
            video_seconds: 0.0,
            cost_usd: resp.cost,
            latency_ms: elapsed_ms,
            run_id: self.run_id,
        });
        Ok(resp)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn emit_completion_usage(
    emitter: &Arc<dyn UsageEmitter>,
    provider: &str,
    model: &str,
    usage: Option<&TokenUsage>,
    cost: Option<f64>,
    latency_ms: u64,
    run_id: Uuid,
) {
    let event = UsageEvent {
        provider: provider.to_owned(),
        model: model.to_owned(),
        modality: Modality::Llm,
        prompt_tokens: usage.map_or(0, |t| t.prompt_tokens),
        completion_tokens: usage.map_or(0, |t| t.completion_tokens),
        total_tokens: usage.map_or(0, |t| t.total_tokens),
        reasoning_tokens: usage.map_or(0, |t| t.reasoning_tokens),
        cached_input_tokens: usage.map_or(0, |t| t.cached_input_tokens),
        audio_input_tokens: usage.map_or(0, |t| t.audio_input_tokens),
        audio_output_tokens: usage.map_or(0, |t| t.audio_output_tokens),
        image_count: 0,
        audio_seconds: 0.0,
        video_seconds: 0.0,
        cost_usd: cost,
        latency_ms,
        run_id,
    };
    emitter.emit(event);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct CapturingEmitter {
        events: Mutex<Vec<UsageEvent>>,
    }

    impl UsageEmitter for CapturingEmitter {
        fn emit(&self, event: UsageEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    struct MockCompletionModel;
    #[async_trait]
    impl CompletionModel for MockCompletionModel {
        #[allow(clippy::unnecessary_literal_bound)]
        fn model_id(&self) -> &str {
            "mock-model"
        }
        async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse, BlazenError> {
            Ok(CompletionResponse {
                content: Some("hi".into()),
                tool_calls: vec![],
                reasoning: None,
                citations: vec![],
                artifacts: vec![],
                usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    ..Default::default()
                }),
                model: "mock-model".into(),
                finish_reason: Some("stop".into()),
                cost: Some(0.001),
                timing: None,
                images: vec![],
                audio: vec![],
                videos: vec![],
                metadata: serde_json::Value::Null,
            })
        }
        async fn stream(
            &self,
            _: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
        {
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }

    #[tokio::test]
    async fn completion_decorator_emits_usage_event_with_cost() {
        let emitter = Arc::new(CapturingEmitter::default());
        let dyn_emitter: Arc<dyn UsageEmitter> = emitter.clone();
        let run_id = Uuid::new_v4();
        let model = UsageRecordingCompletionModel::new(
            MockCompletionModel,
            dyn_emitter,
            "test-provider",
            run_id,
        );
        let req = CompletionRequest::new(vec![ChatMessage::user("hi")]);
        let _ = model.complete(req).await.unwrap();

        let events = emitter.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].provider, "test-provider");
        assert_eq!(events[0].model, "mock-model");
        assert_eq!(events[0].prompt_tokens, 10);
        assert_eq!(events[0].completion_tokens, 5);
        assert_eq!(events[0].total_tokens, 15);
        assert_eq!(events[0].cost_usd, Some(0.001));
        assert_eq!(events[0].run_id, run_id);
    }
}
