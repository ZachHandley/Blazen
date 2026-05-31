//! [`PipelineHandler`] -- the live handle returned by
//! [`Pipeline::start`](crate::pipeline::Pipeline::start), mirroring the uniffi
//! [`WorkflowHandler`](crate::workflow::WorkflowHandler) control surface.

use std::sync::Arc;

use tokio::sync::Mutex as AsyncMutex;
use tokio_stream::StreamExt;

use crate::errors::{BlazenError, BlazenResult};
use crate::llm::TokenUsage;
use crate::pipeline::pipeline::pipeline_result_to_wire;
use crate::workflow::{Event, InputResponse, WorkflowResult, any_event_to_wire};

use blazen_pipeline::PipelineHandler as CorePipelineHandler;

// ---------------------------------------------------------------------------
// Streaming sink + wire event
// ---------------------------------------------------------------------------

/// An intermediate event emitted by a pipeline stage, tagged with provenance.
///
/// Wraps a workflow-level [`Event`] with the stage name, optional branch name
/// (for parallel stages), and the workflow run ID that produced it, so foreign
/// consumers can tell which part of the pipeline emitted each event.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PipelineEvent {
    /// Name of the stage that produced this event.
    pub stage_name: String,
    /// For parallel stages, the name of the specific branch; `None` for
    /// sequential stages.
    pub branch_name: Option<String>,
    /// The workflow run ID (UUID, as a string) that produced this event.
    pub workflow_run_id: String,
    /// The underlying workflow event.
    pub event: Event,
}

/// Foreign-implementable sink for intermediate pipeline events.
///
/// Mirrors [`WorkflowEventSink`](crate::workflow::WorkflowEventSink): UniFFI's
/// async-iterator support across Go, Swift, Kotlin, and Ruby is uneven, so
/// streaming uses a foreign-callable sink trait. Each idiomatic wrapper adapts
/// the callbacks into its host streaming type (Go channel, Swift
/// `AsyncStream`, Kotlin `Flow`, Ruby `Enumerator::Lazy`).
///
/// The pump invokes [`on_event`](Self::on_event) for each [`PipelineEvent`] in
/// order, then exactly one [`on_close`](Self::on_close) when the pipeline
/// completes.
#[uniffi::export(with_foreign)]
pub trait PipelineEventSink: Send + Sync {
    /// One intermediate event arrived from a stage.
    fn on_event(&self, event: PipelineEvent);
    /// The stream ended — the pipeline reached a terminal state (or the
    /// subscription was cancelled). Fires exactly once.
    fn on_close(&self);
}

// ---------------------------------------------------------------------------
// PipelineHandler
// ---------------------------------------------------------------------------

/// A live handle to a running pipeline.
///
/// Returned by [`Pipeline::start`](crate::pipeline::Pipeline::start). Provides
/// the same surface as the workflow handler:
///
/// **Consumption (consumes the handler):**
/// - [`result`](Self::result) — await the final [`WorkflowResult`].
/// - [`pause`](Self::pause) — park the pipeline and capture a snapshot.
///
/// **Streaming (borrows the handler):**
/// - [`stream_events`](Self::stream_events) — pump stage events to a foreign
///   [`PipelineEventSink`].
///
/// **Control (borrows the handler, may be called repeatedly):**
/// - [`resume_in_place`](Self::resume_in_place) / [`snapshot`](Self::snapshot)
/// - [`respond_to_input`](Self::respond_to_input) — human-in-the-loop
/// - [`abort`](Self::abort) / [`progress`](Self::progress)
/// - [`usage_total`](Self::usage_total) / [`cost_total_usd`](Self::cost_total_usd)
#[derive(uniffi::Object)]
pub struct PipelineHandler {
    /// `Option` because [`result`](Self::result) / [`pause`](Self::pause)
    /// consume the inner handler. `AsyncMutex` so the control methods can
    /// borrow it across `.await`.
    inner: Arc<AsyncMutex<Option<CorePipelineHandler>>>,
}

impl PipelineHandler {
    /// Wrap a fresh core [`CorePipelineHandler`].
    pub(crate) fn new(handler: CorePipelineHandler) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(AsyncMutex::new(Some(handler))),
        })
    }

    /// Borrow the inner handler for a control operation, erroring if it was
    /// already consumed by [`result`](Self::result) / [`pause`](Self::pause).
    async fn with_handler<T>(
        &self,
        f: impl FnOnce(&CorePipelineHandler) -> Result<T, BlazenError>,
    ) -> BlazenResult<T> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "PipelineHandler already consumed".into(),
        })?;
        f(handler)
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl PipelineHandler {
    /// Await the final pipeline result, consuming the handler.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the run failed.
    pub async fn result(self: Arc<Self>) -> BlazenResult<WorkflowResult> {
        let handler = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or(BlazenError::Validation {
                message: "PipelineHandler already consumed".into(),
            })?
        };
        let result = handler.result().await.map_err(BlazenError::from)?;
        Ok(pipeline_result_to_wire(&result))
    }

    /// Pump intermediate stage events to `sink` until the pipeline completes.
    ///
    /// Returns immediately; the pump runs on the shared Tokio runtime. Each
    /// call subscribes from the current point in time. `sink.on_close()` fires
    /// exactly once when the stream ends.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed.
    pub async fn stream_events(
        self: Arc<Self>,
        sink: Arc<dyn PipelineEventSink>,
    ) -> BlazenResult<()> {
        let mut stream = {
            let guard = self.inner.lock().await;
            let handler = guard.as_ref().ok_or(BlazenError::Validation {
                message: "PipelineHandler already consumed".into(),
            })?;
            handler.stream_events()
        };
        tokio::spawn(async move {
            while let Some(event) = stream.next().await {
                sink.on_event(PipelineEvent {
                    stage_name: event.stage_name,
                    branch_name: event.branch_name,
                    workflow_run_id: event.workflow_run_id.to_string(),
                    event: any_event_to_wire(&*event.event),
                });
            }
            sink.on_close();
        });
        Ok(())
    }

    /// Park the pipeline after the current stage and return a snapshot of its
    /// state, consuming the handler.
    ///
    /// The returned snapshot is encoded as a JSON string and can later be used
    /// with `Pipeline.resume` (foreign-side, once exposed) to continue.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the pipeline has already terminated.
    pub async fn pause(self: Arc<Self>) -> BlazenResult<String> {
        let handler = {
            let mut guard = self.inner.lock().await;
            guard.take().ok_or(BlazenError::Validation {
                message: "PipelineHandler already consumed".into(),
            })?
        };
        let snapshot = handler.pause().await.map_err(BlazenError::from)?;
        snapshot.to_json().map_err(BlazenError::from)
    }

    /// Capture a resumable snapshot of the pipeline's current state without
    /// stopping it, encoded as a JSON string. Mirrors
    /// `WorkflowHandler::snapshot`.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the pipeline has already terminated.
    pub async fn snapshot(self: Arc<Self>) -> BlazenResult<String> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "PipelineHandler already consumed".into(),
        })?;
        let snap = handler.snapshot().await.map_err(BlazenError::from)?;
        snap.to_json().map_err(BlazenError::from)
    }

    /// Snapshot the running aggregate [`TokenUsage`] for this run. Safe to call
    /// at any point; matches `WorkflowResult` totals once `result()` completes.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed.
    pub async fn usage_total(self: Arc<Self>) -> BlazenResult<TokenUsage> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "PipelineHandler already consumed".into(),
        })?;
        Ok(TokenUsage::from(handler.usage_total().await))
    }

    /// Snapshot the running aggregate cost in USD for this run.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed.
    pub async fn cost_total_usd(self: Arc<Self>) -> BlazenResult<f64> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref().ok_or(BlazenError::Validation {
            message: "PipelineHandler already consumed".into(),
        })?;
        Ok(handler.cost_total_usd().await)
    }
}

#[uniffi::export]
impl PipelineHandler {
    /// Resume the pipeline in place. Forwarded to the active stage's inner
    /// workflow(s) so a workflow parked on an `InputRequestEvent` (or paused)
    /// unparks. A no-op between stages.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the pipeline has already terminated.
    pub async fn resume_in_place(self: Arc<Self>) -> BlazenResult<()> {
        self.with_handler(|h| h.resume_in_place().map_err(BlazenError::from))
            .await
    }

    /// Deliver a human-in-the-loop response to the active stage's inner
    /// workflow. For a sequential stage this targets the one in-flight
    /// workflow; for a parallel stage the response is broadcast to every live
    /// branch (the workflow that requested input consumes it, others ignore a
    /// response they did not request).
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed or
    /// `response.response_json` is not valid JSON; [`BlazenError::Workflow`]
    /// if the pipeline has already terminated.
    pub async fn respond_to_input(self: Arc<Self>, response: InputResponse) -> BlazenResult<()> {
        let parsed: serde_json::Value = serde_json::from_str(&response.response_json)?;
        let core_response = blazen_events::InputResponseEvent {
            request_id: response.request_id,
            response: parsed,
        };
        self.with_handler(move |h| h.respond_to_input(core_response).map_err(BlazenError::from))
            .await
    }

    /// Abort the pipeline. Any pending `result()` resolves with a workflow
    /// error.
    ///
    /// # Errors
    /// [`BlazenError::Validation`] if the handler was already consumed;
    /// [`BlazenError::Workflow`] if the pipeline has already terminated.
    pub async fn abort(self: Arc<Self>) -> BlazenResult<()> {
        self.with_handler(|h| h.abort().map_err(BlazenError::from))
            .await
    }

    /// Best-effort polled view of the pipeline's stage cursor: the 1-based
    /// index of the stage currently executing and the total stage count.
    /// Returns `(current_stage_index, total_stages)`. Returns `None` after the
    /// handler has been consumed.
    pub async fn progress(self: Arc<Self>) -> Option<PipelineProgress> {
        let guard = self.inner.lock().await;
        let handler = guard.as_ref()?;
        let p = handler.progress();
        Some(PipelineProgress {
            current_stage_index: p.current_stage_index,
            total_stages: p.total_stages,
            percent: p.percent,
        })
    }
}

/// Best-effort progress snapshot of a running pipeline.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PipelineProgress {
    /// 1-based index of the stage currently executing.
    pub current_stage_index: u32,
    /// Total number of stages in the pipeline.
    pub total_stages: u32,
    /// Completion percent in `[0, 100]`.
    pub percent: f32,
}
