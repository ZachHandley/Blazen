//! [`PipelineHandler`] -- the handle returned after starting a pipeline.
//!
//! Provides three consumption modes (mirroring [`WorkflowHandler`]):
//!
//! 1. **Await the final result** via [`PipelineHandler::result`].
//! 2. **Stream intermediate events** via [`PipelineHandler::stream_events`],
//!    which emits [`PipelineEvent`] wrappers tagging each event with its
//!    stage and branch name.
//! 3. **Pause the pipeline** via [`PipelineHandler::pause`], which returns
//!    a serializable [`PipelineSnapshot`].

use std::fmt;

use blazen_events::AnyEvent;
use tokio::sync::{broadcast, oneshot};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

use crate::error::PipelineError;
use crate::snapshot::{PipelineResult, PipelineSnapshot};

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
/// the final result, streaming events, or pausing the pipeline.
pub struct PipelineHandler {
    /// Receives the final result when the pipeline completes.
    result_rx: Option<oneshot::Receiver<Result<PipelineResult, PipelineError>>>,
    /// Sender side of the broadcast channel for streaming events.
    stream_tx: broadcast::Sender<PipelineEvent>,
    /// Sends the pause signal to the pipeline execution loop.
    pause_tx: Option<oneshot::Sender<()>>,
    /// Receives the snapshot from the pipeline after a pause.
    snapshot_rx: Option<oneshot::Receiver<PipelineSnapshot>>,
}

impl PipelineHandler {
    /// Create a new handler (crate-internal).
    pub(crate) fn new(
        result_rx: oneshot::Receiver<Result<PipelineResult, PipelineError>>,
        stream_tx: broadcast::Sender<PipelineEvent>,
        pause_tx: oneshot::Sender<()>,
        snapshot_rx: oneshot::Receiver<PipelineSnapshot>,
    ) -> Self {
        Self {
            result_rx: Some(result_rx),
            stream_tx,
            pause_tx: Some(pause_tx),
            snapshot_rx: Some(snapshot_rx),
        }
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
        let pause_tx = self.pause_tx.take().ok_or(PipelineError::ChannelClosed)?;

        let snapshot_rx = self
            .snapshot_rx
            .take()
            .ok_or(PipelineError::ChannelClosed)?;

        // Send the pause signal.
        pause_tx
            .send(())
            .map_err(|()| PipelineError::ChannelClosed)?;

        // Await the snapshot from the execution loop.
        snapshot_rx.await.map_err(|_| PipelineError::ChannelClosed)
    }

    /// Returns a reference to the broadcast sender for forwarding events
    /// from within the execution loop.
    #[allow(dead_code)]
    pub(crate) fn stream_sender(&self) -> &broadcast::Sender<PipelineEvent> {
        &self.stream_tx
    }
}
