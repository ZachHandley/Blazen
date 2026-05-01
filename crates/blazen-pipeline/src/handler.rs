//! [`PipelineHandler`] -- the handle returned after starting a pipeline.
//!
//! Provides three consumption modes (mirroring [`WorkflowHandler`]):
//!
//! 1. **Await the final result** via [`PipelineHandler::result`].
//! 2. **Stream intermediate events** via [`PipelineHandler::stream_events`],
//!    which emits [`PipelineEvent`] wrappers tagging each event with its
//!    stage and branch name.
//! 3. **Control the pipeline** via [`PipelineHandler::pause`],
//!    [`PipelineHandler::resume_in_place`], [`PipelineHandler::snapshot`],
//!    and [`PipelineHandler::abort`].

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use blazen_core::SessionRefRegistry;
use blazen_events::AnyEvent;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::pipeline::ProgressSnapshot;
use crate::snapshot::{PipelineResult, PipelineSnapshot};

/// Commands sent from the handler to the execution loop via the control channel.
pub(crate) enum PipelineControl {
    /// Pause the pipeline. Inner workflow handlers are aborted (via Drop) when
    /// the stage future is cancelled, and a snapshot is sent back.
    Pause,
    /// Resume a paused pipeline in place. Currently a no-op at the pipeline
    /// level -- true in-place resume requires keeping stage futures alive,
    /// which is deferred to a future task.
    Resume,
    /// Abort the pipeline. Inner workflow handlers are aborted (via Drop) when
    /// the stage future is cancelled.
    Abort,
}

/// An event from a pipeline stage, tagged with provenance metadata.
///
/// Wraps a workflow-level event with the stage name and optional branch
/// name so consumers can tell which part of the pipeline emitted it.
pub struct PipelineEvent {
    /// The name of the stage that produced this event.
    pub stage_name: String,
    /// For parallel stages, the name of the specific branch. `None` for
    /// sequential stages.
    pub branch_name: Option<String>,
    /// The workflow run ID that produced this event.
    pub workflow_run_id: Uuid,
    /// The underlying event from the workflow.
    pub event: Box<dyn AnyEvent>,
}

impl fmt::Debug for PipelineEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineEvent")
            .field("stage_name", &self.stage_name)
            .field("branch_name", &self.branch_name)
            .field("workflow_run_id", &self.workflow_run_id)
            .field("event_type", &self.event.event_type_id())
            .finish_non_exhaustive()
    }
}

impl Clone for PipelineEvent {
    fn clone(&self) -> Self {
        Self {
            stage_name: self.stage_name.clone(),
            branch_name: self.branch_name.clone(),
            workflow_run_id: self.workflow_run_id,
            event: self.event.clone_boxed(),
        }
    }
}

/// Handle to a running pipeline.
///
/// Created by [`Pipeline::run`](crate::Pipeline::run). Allows awaiting
/// the final result, streaming events, pausing, resuming, or aborting
/// the pipeline.
pub struct PipelineHandler {
    /// Receives the final result when the pipeline completes.
    result_rx: Option<oneshot::Receiver<Result<PipelineResult, PipelineError>>>,
    /// Sender side of the broadcast channel for streaming events.
    stream_tx: broadcast::Sender<PipelineEvent>,
    /// Control channel to the execution loop (pause/resume/abort).
    control_tx: mpsc::UnboundedSender<PipelineControl>,
    /// Receives the snapshot from the pipeline after a pause.
    snapshot_rx: Option<oneshot::Receiver<PipelineSnapshot>>,
    /// Shared session-ref registry for this pipeline run. Cloned from the
    /// pipeline at construction so the handler outlives the `execute_pipeline`
    /// task and consumers can resolve session-ref markers in stage outputs.
    session_refs: Arc<SessionRefRegistry>,
    /// Shared 1-based index of the stage currently executing. Written by
    /// the executor task before each stage runs and read by
    /// [`PipelineHandler::progress`].
    current_stage: Arc<AtomicUsize>,
    /// Total number of stages on the pipeline. Captured at construction.
    total_stages: u32,
}

impl PipelineHandler {
    /// Create a new handler (crate-internal).
    pub(crate) fn new(
        result_rx: oneshot::Receiver<Result<PipelineResult, PipelineError>>,
        stream_tx: broadcast::Sender<PipelineEvent>,
        control_tx: mpsc::UnboundedSender<PipelineControl>,
        snapshot_rx: oneshot::Receiver<PipelineSnapshot>,
        session_refs: Arc<SessionRefRegistry>,
        current_stage: Arc<AtomicUsize>,
        total_stages: u32,
    ) -> Self {
        Self {
            result_rx: Some(result_rx),
            stream_tx,
            control_tx,
            snapshot_rx: Some(snapshot_rx),
            session_refs,
            current_stage,
            total_stages,
        }
    }

    /// Snapshot the pipeline's current progress without affecting execution.
    ///
    /// Reads the 1-based stage index that the executor publishes before
    /// each stage runs, plus the total stage count captured when the
    /// handler was constructed. The result is best-effort and may be
    /// stale by a few microseconds — there is no synchronisation between
    /// the executor task and `progress()` callers.
    #[must_use]
    pub fn progress(&self) -> ProgressSnapshot {
        let cur = self.current_stage.load(Ordering::Relaxed);
        let total = self.total_stages;
        let current_stage_index = u32::try_from(cur).unwrap_or(u32::MAX);
        let percent = if total == 0 {
            0.0_f32
        } else {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let p = (f64::from(current_stage_index) / f64::from(total) * 100.0) as f32;
            p.clamp(0.0, 100.0)
        };
        ProgressSnapshot {
            current_stage_index,
            total_stages: total,
            percent,
            current_stage_name: None,
        }
    }

    /// Returns a clone of the shared session-ref registry handle.
    #[must_use]
    pub fn session_refs(&self) -> Arc<SessionRefRegistry> {
        Arc::clone(&self.session_refs)
    }

    /// Await the final pipeline result.
    ///
    /// Consumes the handler. Returns the [`PipelineResult`] containing the
    /// final output and all stage results, or a [`PipelineError`].
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ChannelClosed`] if the execution task was
    /// dropped before sending a result.
    ///
    /// # Panics
    ///
    /// Panics if `result()` was already called on this handler.
    pub async fn result(mut self) -> Result<PipelineResult, PipelineError> {
        let rx = self
            .result_rx
            .take()
            .expect("result() called after result was already consumed");
        rx.await.unwrap_or(Err(PipelineError::ChannelClosed))
    }

    /// Subscribe to intermediate events from pipeline stages.
    ///
    /// Each call returns a fresh stream starting from the current point in
    /// time. Events are wrapped in [`PipelineEvent`] with stage/branch
    /// provenance.
    pub fn stream_events(
        &self,
    ) -> impl tokio_stream::Stream<Item = PipelineEvent> + Send + Unpin + use<> {
        let rx = self.stream_tx.subscribe();
        BroadcastStream::new(rx).filter_map(std::result::Result::ok)
    }

    /// Pause the running pipeline and return a snapshot of its state.
    ///
    /// Consumes the handler since the pipeline is no longer running after
    /// a pause. The returned [`PipelineSnapshot`] can be serialized and
    /// later used with [`Pipeline::resume`](crate::Pipeline::resume).
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ChannelClosed`] if the pipeline has
    /// already terminated.
    pub async fn pause(mut self) -> Result<PipelineSnapshot, PipelineError> {
        self.control_tx
            .send(PipelineControl::Pause)
            .map_err(|_| PipelineError::ChannelClosed)?;

        let snapshot_rx = self
            .snapshot_rx
            .take()
            .ok_or(PipelineError::ChannelClosed)?;

        // Await the snapshot from the execution loop.
        snapshot_rx.await.map_err(|_| PipelineError::ChannelClosed)
    }

    /// Resume a paused pipeline in place.
    ///
    /// Currently a no-op at the pipeline level. True in-place resume
    /// (parking inner workflows without dropping them) requires significant
    /// restructuring of the stage execution model and is deferred to a
    /// future task.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ChannelClosed`] if the pipeline has
    /// already terminated.
    pub fn resume_in_place(&self) -> Result<(), PipelineError> {
        self.control_tx
            .send(PipelineControl::Resume)
            .map_err(|_| PipelineError::ChannelClosed)
    }

    /// Capture a [`PipelineSnapshot`] without stopping the pipeline.
    ///
    /// Not yet implemented -- returns [`PipelineError::ChannelClosed`].
    /// A full implementation requires a request/reply oneshot pattern
    /// similar to `WorkflowHandler::snapshot`.
    ///
    /// # Errors
    ///
    /// Always returns [`PipelineError::ChannelClosed`] (stub).
    #[allow(clippy::unused_async)]
    pub async fn snapshot(&self) -> Result<PipelineSnapshot, PipelineError> {
        Err(PipelineError::ChannelClosed)
    }

    /// Abort the running pipeline.
    ///
    /// Sends an abort signal to the execution loop. The loop tears down the
    /// current stage (inner workflow handlers are aborted via `Drop`) and
    /// exits.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ChannelClosed`] if the pipeline has
    /// already terminated.
    pub fn abort(&self) -> Result<(), PipelineError> {
        self.control_tx
            .send(PipelineControl::Abort)
            .map_err(|_| PipelineError::ChannelClosed)
    }

    /// Returns a reference to the broadcast sender for forwarding events
    /// from within the execution loop.
    #[allow(dead_code)]
    pub(crate) fn stream_sender(&self) -> &broadcast::Sender<PipelineEvent> {
        &self.stream_tx
    }
}

impl Drop for PipelineHandler {
    fn drop(&mut self) {
        // Best-effort abort so the spawned execution task doesn't leak.
        // Ignore errors -- the loop may have already exited.
        let _ = self.control_tx.send(PipelineControl::Abort);
    }
}
