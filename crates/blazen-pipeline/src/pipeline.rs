//! Pipeline execution engine.
//!
//! A [`Pipeline`] orchestrates multiple [`Workflow`]s as sequential or
//! parallel stages. It handles input mapping, conditional execution,
//! event streaming, pause/resume, and persistence callbacks.

use std::sync::Arc;
use std::time::{Duration, Instant};

use blazen_core::SessionRefRegistry;
use blazen_events::{AnyEvent, StopEvent};
use chrono::Utc;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tokio_stream::StreamExt;
use uuid::Uuid;

use tracing::Instrument;

use crate::builder::{PersistFn, PersistJsonFn};
use crate::error::PipelineError;
use crate::handler::{PipelineControl, PipelineEvent, PipelineHandler};
use crate::snapshot::{PipelineResult, PipelineSnapshot, StageResult};
use crate::stage::{JoinStrategy, ParallelStage, Stage, StageKind};
use crate::state::PipelineState;

/// A validated, ready-to-run pipeline.
///
/// Constructed via [`PipelineBuilder`](crate::PipelineBuilder). Execute with
/// [`Pipeline::start`] which consumes the pipeline and returns a
/// [`PipelineHandler`] for awaiting results, streaming events, or pausing.
pub struct Pipeline {
    pub(crate) name: String,
    pub(crate) stages: Vec<StageKind>,
    pub(crate) persist_fn: Option<PersistFn>,
    pub(crate) persist_json_fn: Option<PersistJsonFn>,
    pub(crate) timeout_per_stage: Option<Duration>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("stage_count", &self.stages.len())
            .field("timeout_per_stage", &self.timeout_per_stage)
            .finish_non_exhaustive()
    }
}

impl Pipeline {
    /// Execute the pipeline, consuming it.
    ///
    /// Spawns a background task that iterates through stages sequentially,
    /// running each stage's workflow and collecting results.
    ///
    /// Returns a [`PipelineHandler`] for awaiting the result, streaming
    /// events, or pausing.
    #[must_use]
    pub fn start(self, input: serde_json::Value) -> PipelineHandler {
        let run_id = Uuid::new_v4();
        let session_refs = Arc::new(SessionRefRegistry::new());
        self.start_with_id(
            input,
            run_id,
            0,
            Vec::new(),
            std::collections::HashMap::new(),
            session_refs,
        )
    }

    /// Internal: start execution from a specific stage index (used for resume).
    #[must_use]
    fn start_with_id(
        self,
        input: serde_json::Value,
        run_id: Uuid,
        start_index: usize,
        completed: Vec<StageResult>,
        shared_state: std::collections::HashMap<String, serde_json::Value>,
        session_refs: Arc<SessionRefRegistry>,
    ) -> PipelineHandler {
        // Result channel.
        let (result_tx, result_rx) = oneshot::channel();

        // Broadcast channel for streaming events.
        let (stream_tx, _) = broadcast::channel::<PipelineEvent>(256);

        // Control channel (replaces the old oneshot pause channel).
        let (control_tx, control_rx) = mpsc::unbounded_channel::<PipelineControl>();

        // Snapshot channel (still a oneshot -- sent once on pause).
        let (snapshot_tx, snapshot_rx) = oneshot::channel::<PipelineSnapshot>();

        let handler = PipelineHandler::new(
            result_rx,
            stream_tx.clone(),
            control_tx,
            snapshot_rx,
            Arc::clone(&session_refs),
        );

        tokio::spawn(execute_pipeline(
            self.name,
            self.stages,
            input,
            run_id,
            start_index,
            completed,
            shared_state,
            self.timeout_per_stage,
            self.persist_fn,
            self.persist_json_fn,
            result_tx,
            stream_tx,
            control_rx,
            snapshot_tx,
            session_refs,
        ));

        handler
    }

    /// Resume a pipeline from a previously captured snapshot.
    ///
    /// The pipeline must have the same stages as when it was originally
    /// paused (the stage list is provided via the `Pipeline` itself).
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationFailed`] if the snapshot's
    /// stage index is out of bounds.
    pub fn resume(self, snapshot: PipelineSnapshot) -> Result<PipelineHandler, PipelineError> {
        if snapshot.current_stage_index > self.stages.len() {
            return Err(PipelineError::ValidationFailed(format!(
                "snapshot stage index {} exceeds pipeline stage count {}",
                snapshot.current_stage_index,
                self.stages.len(),
            )));
        }

        // Verify the snapshot came from the same pipeline.
        if snapshot.pipeline_name != self.name {
            return Err(PipelineError::ValidationFailed(format!(
                "snapshot pipeline name '{}' does not match this pipeline '{}'",
                snapshot.pipeline_name, self.name,
            )));
        }

        // Cross-reference completed stage names against the current pipeline
        // definition to detect configuration drift.
        for (i, sr) in snapshot.completed_stages.iter().enumerate() {
            let expected = self.stages[i].name();
            if sr.name != expected {
                return Err(PipelineError::ValidationFailed(format!(
                    "completed stage {} name '{}' does not match pipeline stage '{}'",
                    i, sr.name, expected,
                )));
            }
        }

        let session_refs = Arc::new(SessionRefRegistry::new());
        Ok(self.start_with_id(
            snapshot.input,
            snapshot.run_id,
            snapshot.current_stage_index,
            snapshot.completed_stages,
            snapshot.shared_state,
            session_refs,
        ))
    }
}

// ---------------------------------------------------------------------------
// Pipeline execution loop
// ---------------------------------------------------------------------------

/// Core execution loop for the pipeline.
///
/// Runs in a spawned task. Iterates through stages sequentially, running
/// each stage's workflow and collecting results. Checks for control signals
/// (pause/resume/abort) between and during stages.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
async fn execute_pipeline(
    pipeline_name: String,
    stages: Vec<StageKind>,
    input: serde_json::Value,
    run_id: Uuid,
    start_index: usize,
    completed: Vec<StageResult>,
    shared_state: std::collections::HashMap<String, serde_json::Value>,
    timeout_per_stage: Option<Duration>,
    persist_fn: Option<PersistFn>,
    persist_json_fn: Option<PersistJsonFn>,
    result_tx: oneshot::Sender<Result<PipelineResult, PipelineError>>,
    stream_tx: broadcast::Sender<PipelineEvent>,
    mut control_rx: mpsc::UnboundedReceiver<PipelineControl>,
    snapshot_tx: oneshot::Sender<PipelineSnapshot>,
    session_refs: Arc<SessionRefRegistry>,
) {
    let span = tracing::info_span!(
        "pipeline.run",
        pipeline_name = %pipeline_name,
        stage_count = stages.len(),
    );
    let _enter = span.enter();

    let mut stage_results: Vec<StageResult> = completed;

    // Build the stage_results IndexMap from completed stages.
    let stage_results_map: indexmap::IndexMap<String, serde_json::Value> = stage_results
        .iter()
        .map(|sr| (sr.name.clone(), sr.output.clone()))
        .collect();

    // When resuming, restore both shared_state and stage_results from the
    // snapshot. For a fresh start both maps are empty and we fall through
    // to `PipelineState::new`.
    let mut state = if shared_state.is_empty() && stage_results_map.is_empty() {
        PipelineState::new(input.clone())
    } else {
        PipelineState::restore(input.clone(), shared_state, stage_results_map)
    };

    for (stage_idx, stage) in stages.iter().enumerate().skip(start_index) {
        // Check for control signals between stages (non-blocking).
        if let Ok(control) = control_rx.try_recv() {
            match control {
                PipelineControl::Pause => {
                    let snapshot = build_snapshot(
                        &pipeline_name,
                        run_id,
                        stage_idx,
                        &stage_results,
                        &state,
                        &input,
                    );
                    let _ = snapshot_tx.send(snapshot);
                    let _ = result_tx.send(Err(PipelineError::Paused));
                    return;
                }
                PipelineControl::Resume => {
                    // No-op between stages -- we're already running.
                }
                PipelineControl::Abort => {
                    let _ = result_tx.send(Err(PipelineError::Aborted));
                    return;
                }
            }
        }

        let stage_span = tracing::info_span!(
            "pipeline.stage",
            stage_name = %stage.name(),
            stage_index = stage_idx,
            duration_ms = tracing::field::Empty,
            skipped = tracing::field::Empty,
        );
        let _stage_enter = stage_span.enter();

        tracing::info!(
            pipeline = %pipeline_name,
            stage = %stage.name(),
            index = stage_idx,
            "executing pipeline stage"
        );

        let stage_start = Instant::now();

        // Race stage execution against control signals so pause/abort can
        // interrupt a running stage.
        let stage_future = async {
            match stage {
                StageKind::Sequential(s) => {
                    run_sequential_stage(s, &state, &stream_tx, timeout_per_stage, &session_refs)
                        .instrument(
                            tracing::info_span!("pipeline.stage.sequential", stage_name = %s.name),
                        )
                        .await
                }
                StageKind::Parallel(p) => {
                    run_parallel_stage(p, &state, &stream_tx, timeout_per_stage, &session_refs)
                        .instrument(tracing::info_span!(
                            "pipeline.stage.parallel",
                            branch_count = p.branches.len()
                        ))
                        .await
                }
            }
        };

        let result = tokio::select! {
            biased;

            // Control signal -- takes priority when both are ready.
            Some(control) = control_rx.recv() => {
                match control {
                    PipelineControl::Pause => {
                        // The stage future is dropped here, which drops any
                        // inner WorkflowHandlers. Their Drop impls send Abort
                        // to the inner workflow event loops, giving clean shutdown.
                        let snapshot = build_snapshot(
                            &pipeline_name,
                            run_id,
                            stage_idx,
                            &stage_results,
                            &state,
                            &input,
                        );
                        let _ = snapshot_tx.send(snapshot);
                        let _ = result_tx.send(Err(PipelineError::Paused));
                        return;
                    }
                    PipelineControl::Resume => {
                        // No-op during stage execution. True in-place resume
                        // requires keeping the stage future alive, which is
                        // deferred to a future task.
                        continue;
                    }
                    PipelineControl::Abort => {
                        // The stage future is dropped here, which drops any
                        // inner WorkflowHandlers. Their Drop impls send Abort.
                        let _ = result_tx.send(Err(PipelineError::Aborted));
                        return;
                    }
                }
            }

            result = stage_future => result,
        };

        #[allow(clippy::cast_possible_truncation)]
        let duration_ms = stage_start.elapsed().as_millis() as u64;

        match result {
            Ok(stage_output) => {
                stage_span.record("duration_ms", duration_ms);
                stage_span.record("skipped", false);
                let sr = StageResult {
                    name: stage.name().to_owned(),
                    output: stage_output.clone(),
                    skipped: false,
                    duration_ms,
                };
                state.record_stage_result(stage.name(), stage_output);
                stage_results.push(sr);
            }
            Err(StageOutcome::Skipped) => {
                stage_span.record("duration_ms", duration_ms);
                stage_span.record("skipped", true);
                let sr = StageResult {
                    name: stage.name().to_owned(),
                    output: serde_json::Value::Null,
                    skipped: true,
                    duration_ms,
                };
                stage_results.push(sr);
            }
            Err(StageOutcome::Failed(e)) => {
                stage_span.record("duration_ms", duration_ms);
                let _ = result_tx.send(Err(e));
                return;
            }
        }

        // Call persist callbacks after each stage.
        if let Err(e) = call_persist_callbacks(
            persist_fn.as_ref(),
            persist_json_fn.as_ref(),
            &pipeline_name,
            run_id,
            stage_idx + 1,
            &stage_results,
            &state,
            &input,
        )
        .await
        {
            tracing::warn!(
                pipeline = %pipeline_name,
                error = %e,
                "persist callback failed (continuing)"
            );
        }
    }

    // All stages completed.
    let final_output = state.last_result().clone();
    let pipeline_result = PipelineResult {
        pipeline_name,
        run_id,
        final_output,
        stage_results,
        shared_state: state.shared_clone(),
        session_refs: Arc::clone(&session_refs),
    };
    let _ = result_tx.send(Ok(pipeline_result));
}

// ---------------------------------------------------------------------------
// Stage execution helpers
// ---------------------------------------------------------------------------

/// Outcome for stage execution that distinguishes between skipped and failed.
enum StageOutcome {
    Skipped,
    Failed(PipelineError),
}

/// Run a single sequential stage.
#[allow(clippy::similar_names)]
async fn run_sequential_stage(
    stage: &Stage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
) -> Result<serde_json::Value, StageOutcome> {
    // Check condition.
    if let Some(condition) = &stage.condition
        && !condition(state)
    {
        tracing::info!(stage = %stage.name, "stage skipped (condition false)");
        return Err(StageOutcome::Skipped);
    }

    // Determine input.
    let workflow_input = if let Some(mapper) = &stage.input_mapper {
        mapper(state)
    } else {
        state.last_result().clone()
    };

    // Run the workflow with the shared session-ref registry.
    let handler = stage
        .workflow
        .run_with_registry(workflow_input, Arc::clone(session_refs))
        .await
        .map_err(|e| StageOutcome::Failed(PipelineError::Workflow(e)))?;

    // Subscribe to the workflow's event stream and forward events.
    let stage_name = stage.name.clone();
    let stream_tx_clone = stream_tx.clone();
    let mut wf_stream = handler.stream_events();

    // Forward events in a separate task while awaiting the result.
    let forward_handle = tokio::spawn({
        let stage_name = stage_name.clone();
        async move {
            while let Some(event) = wf_stream.next().await {
                let pipeline_event = PipelineEvent {
                    stage_name: stage_name.clone(),
                    branch_name: None,
                    workflow_run_id: Uuid::nil(),
                    event,
                };
                let _ = stream_tx_clone.send(pipeline_event);
            }
        }
    });

    // Await the workflow result, optionally with a timeout.
    #[allow(clippy::single_match_else)]
    let wf_result = if let Some(timeout_dur) = timeout {
        match tokio::time::timeout(timeout_dur, handler.result()).await {
            Ok(r) => r,
            Err(_) => {
                forward_handle.abort();
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: stage_name.clone(),
                    source: Box::new(blazen_core::WorkflowError::Timeout {
                        elapsed: timeout_dur,
                    }),
                }));
            }
        }
    } else {
        handler.result().await
    };

    match wf_result {
        Ok(wf_res) => {
            // Wait for the forwarding task to finish cleanly.
            let _ = forward_handle.await;
            let output = extract_stop_result(&*wf_res.event);
            Ok(output)
        }
        Err(e) => {
            // Abort the forward handle on error -- it may still be running
            // if the workflow errored before the stream closed.
            forward_handle.abort();
            Err(StageOutcome::Failed(PipelineError::StageFailed {
                stage_name,
                source: Box::new(e),
            }))
        }
    }
}

/// Run a parallel stage with multiple branches.
async fn run_parallel_stage(
    parallel: &ParallelStage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
) -> Result<serde_json::Value, StageOutcome> {
    match parallel.join_strategy {
        JoinStrategy::WaitAll => {
            run_parallel_wait_all(parallel, state, stream_tx, timeout, session_refs).await
        }
        JoinStrategy::FirstCompletes => {
            run_parallel_first_completes(parallel, state, stream_tx, timeout, session_refs).await
        }
    }
}

/// Run all branches and wait for all to complete.
async fn run_parallel_wait_all(
    parallel: &ParallelStage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
) -> Result<serde_json::Value, StageOutcome> {
    let mut set = JoinSet::new();
    let mut forward_handles: Vec<JoinHandle<()>> = Vec::new();

    for branch in &parallel.branches {
        if let Some(condition) = &branch.condition
            && !condition(state)
        {
            continue;
        }

        let workflow_input = if let Some(mapper) = &branch.input_mapper {
            mapper(state)
        } else {
            state.last_result().clone()
        };

        let branch_name = branch.name.clone();
        let stage_name = parallel.name.clone();

        let handler = match branch
            .workflow
            .run_with_registry(workflow_input, Arc::clone(session_refs))
            .await
        {
            Ok(h) => h,
            Err(e) => {
                // Abort all forward handles before returning.
                for fh in &forward_handles {
                    fh.abort();
                }
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: format!("{}::{}", parallel.name, branch_name),
                    source: Box::new(e),
                }));
            }
        };

        // Forward events from this branch.
        let mut wf_stream = handler.stream_events();
        let fwd_stage = stage_name;
        let fwd_branch = branch_name.clone();
        let fwd_tx = stream_tx.clone();
        let fh = tokio::spawn(async move {
            while let Some(event) = wf_stream.next().await {
                let pipeline_event = PipelineEvent {
                    stage_name: fwd_stage.clone(),
                    branch_name: Some(fwd_branch.clone()),
                    workflow_run_id: Uuid::nil(),
                    event,
                };
                let _ = fwd_tx.send(pipeline_event);
            }
        });
        forward_handles.push(fh);

        set.spawn(async move {
            let result = if let Some(t) = timeout {
                match tokio::time::timeout(t, handler.result()).await {
                    Ok(r) => r,
                    Err(_) => Err(blazen_core::WorkflowError::Timeout { elapsed: t }),
                }
            } else {
                handler.result().await
            };
            (branch_name, result)
        });
    }

    // Collect all results.
    let mut results = serde_json::Map::new();
    while let Some(join_result) = set.join_next().await {
        match join_result {
            Ok((branch_name, Ok(wf_res))) => {
                let output = extract_stop_result(&*wf_res.event);
                results.insert(branch_name, output);
            }
            Ok((branch_name, Err(e))) => {
                // Abort all forward handles on error.
                for fh in &forward_handles {
                    fh.abort();
                }
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: format!("{}::{}", parallel.name, branch_name),
                    source: Box::new(e),
                }));
            }
            Err(e) => {
                // Abort all forward handles on error.
                for fh in &forward_handles {
                    fh.abort();
                }
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: parallel.name.clone(),
                    source: Box::new(e),
                }));
            }
        }
    }

    // Abort all forward handles now that results are collected.
    // (They should have already finished, but abort to be safe.)
    for fh in &forward_handles {
        fh.abort();
    }

    Ok(serde_json::Value::Object(results))
}

/// Run all branches and return the first to complete.
async fn run_parallel_first_completes(
    parallel: &ParallelStage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
) -> Result<serde_json::Value, StageOutcome> {
    let mut set = JoinSet::new();
    let mut forward_handles: Vec<JoinHandle<()>> = Vec::new();

    for branch in &parallel.branches {
        if let Some(condition) = &branch.condition
            && !condition(state)
        {
            continue;
        }

        let workflow_input = if let Some(mapper) = &branch.input_mapper {
            mapper(state)
        } else {
            state.last_result().clone()
        };

        let branch_name = branch.name.clone();
        let stage_name = parallel.name.clone();

        let handler = match branch
            .workflow
            .run_with_registry(workflow_input, Arc::clone(session_refs))
            .await
        {
            Ok(h) => h,
            Err(e) => {
                // Abort all forward handles before returning.
                for fh in &forward_handles {
                    fh.abort();
                }
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: format!("{}::{}", parallel.name, branch_name),
                    source: Box::new(e),
                }));
            }
        };

        let mut wf_stream = handler.stream_events();
        let fwd_stage = stage_name;
        let fwd_branch = branch_name.clone();
        let fwd_tx = stream_tx.clone();
        let fh = tokio::spawn(async move {
            while let Some(event) = wf_stream.next().await {
                let pipeline_event = PipelineEvent {
                    stage_name: fwd_stage.clone(),
                    branch_name: Some(fwd_branch.clone()),
                    workflow_run_id: Uuid::nil(),
                    event,
                };
                let _ = fwd_tx.send(pipeline_event);
            }
        });
        forward_handles.push(fh);

        set.spawn(async move {
            let result = if let Some(t) = timeout {
                match tokio::time::timeout(t, handler.result()).await {
                    Ok(r) => r,
                    Err(_) => Err(blazen_core::WorkflowError::Timeout { elapsed: t }),
                }
            } else {
                handler.result().await
            };
            (branch_name, result)
        });
    }

    // Return the first successful result.
    let outcome = if let Some(join_result) = set.join_next().await {
        // Abort remaining branches.
        set.abort_all();

        match join_result {
            Ok((branch_name, Ok(wf_res))) => {
                let output = extract_stop_result(&*wf_res.event);
                let mut result_map = serde_json::Map::new();
                result_map.insert(branch_name, output);
                Ok(serde_json::Value::Object(result_map))
            }
            Ok((branch_name, Err(e))) => Err(StageOutcome::Failed(PipelineError::StageFailed {
                stage_name: format!("{}::{}", parallel.name, branch_name),
                source: Box::new(e),
            })),
            Err(e) => Err(StageOutcome::Failed(PipelineError::StageFailed {
                stage_name: parallel.name.clone(),
                source: Box::new(e),
            })),
        }
    } else {
        // No branches to run (all skipped by conditions).
        Ok(serde_json::Value::Null)
    };

    // Abort all forward handles -- remaining branches are cancelled.
    for fh in &forward_handles {
        fh.abort();
    }

    outcome
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the result value from a `StopEvent` or fall back to the event's
/// JSON representation.
fn extract_stop_result(event: &dyn AnyEvent) -> serde_json::Value {
    if let Some(stop) = event.as_any().downcast_ref::<StopEvent>() {
        stop.result.clone()
    } else {
        event.to_json()
    }
}

/// Build a pipeline snapshot from the current state.
fn build_snapshot(
    pipeline_name: &str,
    run_id: Uuid,
    current_stage_index: usize,
    stage_results: &[StageResult],
    state: &PipelineState,
    input: &serde_json::Value,
) -> PipelineSnapshot {
    PipelineSnapshot {
        pipeline_name: pipeline_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        current_stage_index,
        completed_stages: stage_results.to_vec(),
        active_snapshots: Vec::new(),
        shared_state: state.shared_clone(),
        input: input.clone(),
    }
}

/// Call persist callbacks if configured.
#[allow(clippy::too_many_arguments)]
async fn call_persist_callbacks(
    persist_fn: Option<&PersistFn>,
    persist_json_fn: Option<&PersistJsonFn>,
    pipeline_name: &str,
    run_id: Uuid,
    next_stage_index: usize,
    stage_results: &[StageResult],
    state: &PipelineState,
    input: &serde_json::Value,
) -> Result<(), PipelineError> {
    let snapshot = build_snapshot(
        pipeline_name,
        run_id,
        next_stage_index,
        stage_results,
        state,
        input,
    );

    if let Some(f) = persist_fn {
        f(snapshot.clone()).await?;
    }

    if let Some(f) = persist_json_fn {
        let json = serde_json::to_string(&snapshot).map_err(PipelineError::Serialization)?;
        f(json).await?;
    }

    Ok(())
}
