//! Pipeline execution engine.
//!
//! A [`Pipeline`] orchestrates multiple [`Workflow`]s as sequential or
//! parallel stages. It handles input mapping, conditional execution,
//! event streaming, pause/resume, and persistence callbacks.

use std::time::{Duration, Instant};

use blazen_events::{AnyEvent, StopEvent};
use chrono::Utc;
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinSet;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::builder::{PersistFn, PersistJsonFn};
use crate::error::PipelineError;
use crate::handler::{PipelineEvent, PipelineHandler};
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
        self.start_with_id(input, run_id, 0, Vec::new())
    }

    /// Internal: start execution from a specific stage index (used for resume).
    #[must_use]
    fn start_with_id(
        self,
        input: serde_json::Value,
        run_id: Uuid,
        start_index: usize,
        completed: Vec<StageResult>,
    ) -> PipelineHandler {
        // Result channel.
        let (result_tx, result_rx) = oneshot::channel();

        // Broadcast channel for streaming events.
        let (stream_tx, _) = broadcast::channel::<PipelineEvent>(256);

        // Pause/snapshot channels.
        let (pause_tx, pause_rx) = oneshot::channel::<()>();
        let (snapshot_tx, snapshot_rx) = oneshot::channel::<PipelineSnapshot>();

        let handler = PipelineHandler::new(result_rx, stream_tx.clone(), pause_tx, snapshot_rx);

        tokio::spawn(execute_pipeline(
            self.name,
            self.stages,
            input,
            run_id,
            start_index,
            completed,
            self.timeout_per_stage,
            self.persist_fn,
            self.persist_json_fn,
            result_tx,
            stream_tx,
            pause_rx,
            snapshot_tx,
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

        Ok(self.start_with_id(
            snapshot.input,
            snapshot.run_id,
            snapshot.current_stage_index,
            snapshot.completed_stages,
        ))
    }
}

// ---------------------------------------------------------------------------
// Pipeline execution loop
// ---------------------------------------------------------------------------

/// Core execution loop for the pipeline.
///
/// Runs in a spawned task. Iterates through stages sequentially, running
/// each stage's workflow and collecting results. Checks for pause signals
/// between stages.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn execute_pipeline(
    pipeline_name: String,
    stages: Vec<StageKind>,
    input: serde_json::Value,
    run_id: Uuid,
    start_index: usize,
    completed: Vec<StageResult>,
    timeout_per_stage: Option<Duration>,
    persist_fn: Option<PersistFn>,
    persist_json_fn: Option<PersistJsonFn>,
    result_tx: oneshot::Sender<Result<PipelineResult, PipelineError>>,
    stream_tx: broadcast::Sender<PipelineEvent>,
    mut pause_rx: oneshot::Receiver<()>,
    snapshot_tx: oneshot::Sender<PipelineSnapshot>,
) {
    let mut state = PipelineState::new(input.clone());
    let mut stage_results: Vec<StageResult> = completed;

    // Restore state from completed stages.
    for sr in &stage_results {
        state.record_stage_result(&sr.name, sr.output.clone());
    }

    for stage_idx in start_index..stages.len() {
        // Check for pause signal between stages (non-blocking).
        if let Ok(()) | Err(oneshot::error::TryRecvError::Closed) = pause_rx.try_recv() {
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

        let stage = &stages[stage_idx];

        tracing::info!(
            pipeline = %pipeline_name,
            stage = %stage.name(),
            index = stage_idx,
            "executing pipeline stage"
        );

        let stage_start = Instant::now();

        // Race stage execution against the pause signal so pause can
        // interrupt a running stage.
        let stage_future = async {
            match stage {
                StageKind::Sequential(s) => {
                    run_sequential_stage(s, &state, &stream_tx, timeout_per_stage).await
                }
                StageKind::Parallel(p) => {
                    run_parallel_stage(p, &state, &stream_tx, timeout_per_stage).await
                }
            }
        };

        let result = tokio::select! {
            biased;

            // Pause signal -- takes priority when both are ready.
            _ = &mut pause_rx => {
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

            result = stage_future => result,
        };

        let duration_ms = stage_start.elapsed().as_millis() as u64;

        match result {
            Ok(stage_output) => {
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
                let sr = StageResult {
                    name: stage.name().to_owned(),
                    output: serde_json::Value::Null,
                    skipped: true,
                    duration_ms,
                };
                stage_results.push(sr);
            }
            Err(StageOutcome::Failed(e)) => {
                let _ = result_tx.send(Err(e));
                return;
            }
        }

        // Call persist callbacks after each stage.
        if let Err(e) = call_persist_callbacks(
            &persist_fn,
            &persist_json_fn,
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
async fn run_sequential_stage(
    stage: &Stage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
) -> Result<serde_json::Value, StageOutcome> {
    // Check condition.
    if let Some(condition) = &stage.condition {
        if !condition(state) {
            tracing::info!(stage = %stage.name, "stage skipped (condition false)");
            return Err(StageOutcome::Skipped);
        }
    }

    // Determine input.
    let workflow_input = if let Some(mapper) = &stage.input_mapper {
        mapper(state)
    } else {
        state.last_result().clone()
    };

    // Run the workflow.
    let handler = stage
        .workflow
        .run(workflow_input)
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

    // Wait for the forwarding task to finish.
    let _ = forward_handle.await;

    match wf_result {
        Ok(event) => {
            let output = extract_stop_result(&*event);
            Ok(output)
        }
        Err(e) => Err(StageOutcome::Failed(PipelineError::StageFailed {
            stage_name,
            source: Box::new(e),
        })),
    }
}

/// Run a parallel stage with multiple branches.
async fn run_parallel_stage(
    parallel: &ParallelStage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
) -> Result<serde_json::Value, StageOutcome> {
    match parallel.join_strategy {
        JoinStrategy::WaitAll => run_parallel_wait_all(parallel, state, stream_tx, timeout).await,
        JoinStrategy::FirstCompletes => {
            run_parallel_first_completes(parallel, state, stream_tx, timeout).await
        }
    }
}

/// Run all branches and wait for all to complete.
async fn run_parallel_wait_all(
    parallel: &ParallelStage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
) -> Result<serde_json::Value, StageOutcome> {
    let mut set = JoinSet::new();

    for branch in &parallel.branches {
        if let Some(condition) = &branch.condition {
            if !condition(state) {
                continue;
            }
        }

        let workflow_input = if let Some(mapper) = &branch.input_mapper {
            mapper(state)
        } else {
            state.last_result().clone()
        };

        let branch_name = branch.name.clone();
        let stage_name = parallel.name.clone();

        let handler = match branch.workflow.run(workflow_input).await {
            Ok(h) => h,
            Err(e) => {
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
        tokio::spawn(async move {
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
            Ok((branch_name, Ok(event))) => {
                let output = extract_stop_result(&*event);
                results.insert(branch_name, output);
            }
            Ok((branch_name, Err(e))) => {
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: format!("{}::{}", parallel.name, branch_name),
                    source: Box::new(e),
                }));
            }
            Err(e) => {
                return Err(StageOutcome::Failed(PipelineError::StageFailed {
                    stage_name: parallel.name.clone(),
                    source: Box::new(e),
                }));
            }
        }
    }

    Ok(serde_json::Value::Object(results))
}

/// Run all branches and return the first to complete.
async fn run_parallel_first_completes(
    parallel: &ParallelStage,
    state: &PipelineState,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    timeout: Option<Duration>,
) -> Result<serde_json::Value, StageOutcome> {
    let mut set = JoinSet::new();

    for branch in &parallel.branches {
        if let Some(condition) = &branch.condition {
            if !condition(state) {
                continue;
            }
        }

        let workflow_input = if let Some(mapper) = &branch.input_mapper {
            mapper(state)
        } else {
            state.last_result().clone()
        };

        let branch_name = branch.name.clone();
        let stage_name = parallel.name.clone();

        let handler = match branch.workflow.run(workflow_input).await {
            Ok(h) => h,
            Err(e) => {
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
        tokio::spawn(async move {
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
    if let Some(join_result) = set.join_next().await {
        // Abort remaining branches.
        set.abort_all();

        match join_result {
            Ok((branch_name, Ok(event))) => {
                let output = extract_stop_result(&*event);
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
    }
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
    persist_fn: &Option<PersistFn>,
    persist_json_fn: &Option<PersistJsonFn>,
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
