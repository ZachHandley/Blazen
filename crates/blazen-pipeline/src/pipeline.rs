//! Pipeline execution engine.
//!
//! A [`Pipeline`] orchestrates multiple [`Workflow`]s as sequential or
//! parallel stages. It handles input mapping, conditional execution,
//! event streaming, pause/resume, and persistence callbacks.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use blazen_core::runtime;
use blazen_core::runtime::{JoinHandle, JoinSet};
use blazen_core::{SessionRefRegistry, WorkflowHandler};
use blazen_events::{
    AnyEvent, EventEnvelope, InputResponseEvent, ProgressEvent, ProgressKind, StopEvent,
};
use blazen_llm::retry::RetryConfig;
use chrono::Utc;
use tokio::sync::{Mutex, broadcast, mpsc, oneshot};
use tokio_stream::StreamExt;
use uuid::Uuid;

use tracing::Instrument;

use crate::builder::{PersistFn, PersistJsonFn};
use crate::error::PipelineError;
use crate::handler::{PipelineControl, PipelineEvent, PipelineHandler, UsageTotals};
use crate::snapshot::{PipelineResult, PipelineSnapshot, StageResult};
use crate::stage::{JoinStrategy, LoopDecision, LoopStage, ParallelStage, Stage, StageKind};
use crate::state::PipelineState;

/// Live handles into the currently-executing stage's inner workflow(s), so
/// pipeline-level control (resume / respond-to-input) can be forwarded to the
/// active [`WorkflowHandler`](blazen_core::WorkflowHandler)(s) without
/// consuming them.
///
/// A sequential stage registers exactly one handler; a parallel stage
/// registers one per live branch. The slot is cleared when the stage ends,
/// dropping the supervisor's `Arc`s so each inner workflow's `Drop` tears the
/// event loop down if it is still live.
#[derive(Default)]
struct ActiveStage {
    handlers: Mutex<Vec<Arc<WorkflowHandler>>>,
}

impl ActiveStage {
    /// Register the (single) inner workflow handler for a starting stage,
    /// replacing any previously-registered handlers.
    async fn set_current(&self, handler: Arc<WorkflowHandler>) {
        let mut guard = self.handlers.lock().await;
        guard.clear();
        guard.push(handler);
    }

    /// Register an additional inner workflow handler (one per parallel branch).
    async fn push_branch(&self, handler: Arc<WorkflowHandler>) {
        self.handlers.lock().await.push(handler);
    }

    /// Drop all registered handlers — called when a stage ends so the inner
    /// workflow event loops are torn down via `WorkflowHandler::Drop`.
    async fn clear(&self) {
        self.handlers.lock().await.clear();
    }

    /// Forward an in-place resume to every live inner workflow handler.
    async fn forward_resume(&self) {
        for h in self.handlers.lock().await.iter() {
            let _ = h.resume_in_place();
        }
    }

    /// Forward a human-in-the-loop response to every live inner workflow
    /// handler. The workflow that requested input consumes it; the others
    /// ignore a response they did not request.
    async fn forward_input(&self, response: &InputResponseEvent) {
        for h in self.handlers.lock().await.iter() {
            let _ = h.respond_to_input(response.clone());
        }
    }
}

/// Lightweight, polled view of a running pipeline's progress.
///
/// Returned by [`PipelineHandler::progress`]. Reads do not synchronize
/// with the executor task — the values are best-effort and may briefly
/// be one stage behind the actual position.
#[derive(Debug, Clone)]
pub struct ProgressSnapshot {
    /// 1-based index of the stage currently executing (or just completed).
    /// `0` before the first stage starts.
    pub current_stage_index: u32,
    /// Total number of stages declared on the pipeline.
    pub total_stages: u32,
    /// Progress as a percentage in `0.0..=100.0`.
    pub percent: f32,
    /// Name of the current stage, when available. Always `None` from the
    /// simple atomic-index implementation; reserved for future use if we
    /// ever stash a `Mutex<String>` alongside the index.
    pub current_stage_name: Option<String>,
}

/// A validated, ready-to-run pipeline.
///
/// Constructed via [`PipelineBuilder`](crate::PipelineBuilder). Execute with
/// [`Pipeline::start`] which consumes the pipeline and returns a
/// [`PipelineHandler`] for awaiting results, streaming events, or pausing.
///
/// The type parameter `S` is the typed shared-state slot exposed via
/// [`PipelineState::shared`]/[`PipelineState::shared_mut`]. It defaults to
/// [`serde_json::Value`] for ergonomic JSON-only use; supply a concrete `S`
/// (must be `Default + Clone + Send + Sync + 'static` and additionally
/// `Serialize + DeserializeOwned` for snapshot/resume) when you want a
/// strongly-typed bag of state to thread through stages.
pub struct Pipeline<S = serde_json::Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    pub(crate) name: String,
    pub(crate) stages: Vec<StageKind<S>>,
    pub(crate) persist_fn: Option<PersistFn>,
    pub(crate) persist_json_fn: Option<PersistJsonFn>,
    pub(crate) event_envelope_tx: Option<mpsc::UnboundedSender<EventEnvelope>>,
    pub(crate) timeout_per_stage: Option<Duration>,
    pub(crate) total_timeout: Option<Duration>,
    pub(crate) retry_config: Option<Arc<RetryConfig>>,
}

impl<S> Clone for Pipeline<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            stages: self.stages.clone(),
            persist_fn: self.persist_fn.clone(),
            persist_json_fn: self.persist_json_fn.clone(),
            event_envelope_tx: self.event_envelope_tx.clone(),
            timeout_per_stage: self.timeout_per_stage,
            total_timeout: self.total_timeout,
            retry_config: self.retry_config.clone(),
        }
    }
}

impl<S> std::fmt::Debug for Pipeline<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("stage_count", &self.stages.len())
            .field("timeout_per_stage", &self.timeout_per_stage)
            .field("total_timeout", &self.total_timeout)
            .field("retry_config", &self.retry_config.is_some())
            .finish_non_exhaustive()
    }
}

impl<S> Pipeline<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Pipeline-level retry default, if set.
    ///
    /// When `Some`, this configuration acts as the default for every LLM
    /// call inside the pipeline. Workflow / step / per-call overrides take
    /// precedence over this value.
    #[must_use]
    pub fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config.as_ref()
    }
}

impl<S> Pipeline<S>
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize + serde::de::DeserializeOwned,
{
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

        // Shared progress counter — incremented by the executor before each
        // stage runs and read by `PipelineHandler::progress()`. Initialised
        // to `start_index` so a resumed pipeline reports its correct
        // 0-based offset from the get-go.
        let total_stages =
            u32::try_from(self.stages.len()).expect("pipeline stage count fits in u32");
        let current_stage = Arc::new(AtomicUsize::new(start_index));

        // Shared running usage/cost totals, seeded from any completed stages
        // carried over on resume so a live `usage_total()` reflects prior work.
        let mut seeded = UsageTotals::default();
        for sr in &completed {
            if let Some(u) = &sr.usage {
                seeded.usage.add(u);
            }
            seeded.cost_usd += sr.cost_usd.unwrap_or(0.0);
        }
        let usage_totals = Arc::new(Mutex::new(seeded));

        let handler = PipelineHandler::new(
            result_rx,
            stream_tx.clone(),
            control_tx,
            snapshot_rx,
            Arc::clone(&session_refs),
            Arc::clone(&current_stage),
            total_stages,
            Arc::clone(&usage_totals),
        );

        runtime::spawn(execute_pipeline(
            self.name,
            self.stages,
            input,
            run_id,
            start_index,
            completed,
            shared_state,
            self.timeout_per_stage,
            self.total_timeout,
            self.retry_config.clone(),
            self.persist_fn,
            self.persist_json_fn,
            self.event_envelope_tx,
            result_tx,
            stream_tx,
            control_rx,
            snapshot_tx,
            session_refs,
            current_stage,
            usage_totals,
        ));

        handler
    }

    /// Execute the pipeline and await the final result in one call.
    ///
    /// Equivalent to `self.start(input).result().await`. For streaming,
    /// pausing, snapshotting, or human-in-the-loop input, use
    /// [`Pipeline::start`] and drive the returned [`PipelineHandler`].
    ///
    /// # Errors
    ///
    /// Returns any [`PipelineError`] the run produces (stage failure,
    /// timeout, abort, serialization).
    pub async fn run(self, input: serde_json::Value) -> Result<PipelineResult, PipelineError> {
        self.start(input).result().await
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
async fn execute_pipeline<S>(
    pipeline_name: String,
    stages: Vec<StageKind<S>>,
    input: serde_json::Value,
    run_id: Uuid,
    start_index: usize,
    completed: Vec<StageResult>,
    shared_state: std::collections::HashMap<String, serde_json::Value>,
    timeout_per_stage: Option<Duration>,
    total_timeout: Option<Duration>,
    retry_config: Option<Arc<RetryConfig>>,
    persist_fn: Option<PersistFn>,
    persist_json_fn: Option<PersistJsonFn>,
    event_envelope_tx: Option<mpsc::UnboundedSender<EventEnvelope>>,
    result_tx: oneshot::Sender<Result<PipelineResult, PipelineError>>,
    stream_tx: broadcast::Sender<PipelineEvent>,
    mut control_rx: mpsc::UnboundedReceiver<PipelineControl>,
    snapshot_tx: oneshot::Sender<PipelineSnapshot>,
    session_refs: Arc<SessionRefRegistry>,
    current_stage: Arc<AtomicUsize>,
    usage_totals: Arc<Mutex<UsageTotals>>,
) where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize + serde::de::DeserializeOwned,
{
    let span = tracing::info_span!(
        "pipeline.run",
        pipeline_name = %pipeline_name,
        stage_count = stages.len(),
    );
    let _enter = span.enter();

    // Pipeline-level retry config is stored but not yet propagated to per-step
    // retry resolution. That wiring lands in wave 2A.B; for now we just
    // acknowledge that the config travelled through `start_with_id` and into
    // the executor so it survives builds/clippy.
    if let Some(rc) = retry_config.as_ref() {
        tracing::trace!(
            pipeline = %pipeline_name,
            max_retries = rc.max_retries,
            initial_delay_ms = rc.initial_delay_ms,
            max_delay_ms = rc.max_delay_ms,
            "pipeline-level retry_config stored; per-step propagation pending wave 2A.B",
        );
    }

    let stage_results: Vec<StageResult> = completed;

    // Build the stage_results IndexMap from completed stages.
    let stage_results_map: indexmap::IndexMap<String, serde_json::Value> = stage_results
        .iter()
        .map(|sr| (sr.name.clone(), sr.output.clone()))
        .collect();

    // When resuming, restore both shared_state and stage_results from the
    // snapshot. For a fresh start both maps are empty and we fall through
    // to `PipelineState::new`. Restore is fallible (typed `S` is decoded
    // from the persisted JSON) so a decode failure short-circuits the run.
    let state: PipelineState<S> = if shared_state.is_empty() && stage_results_map.is_empty() {
        PipelineState::<S>::new(input.clone())
    } else {
        let shared_json = serde_json::Value::Object(shared_state.into_iter().collect());
        match PipelineState::<S>::restore(shared_json, stage_results_map, input.clone()) {
            Ok(s) => s,
            Err(e) => {
                let _ = result_tx.send(Err(PipelineError::Serialization(e)));
                return;
            }
        }
    };

    // The run-loop body is wrapped in an async block so it can be cancelled
    // wholesale by `tokio::time::timeout` when a `total_timeout` is set.
    // Dropping the future cancels any in-flight stage future, which in turn
    // drops any active `WorkflowHandler`s; their `Drop` impls send `Abort`
    // to the inner workflow event loops for clean shutdown -- the same path
    // used by Pause/Abort below.
    let session_refs_for_run = Arc::clone(&session_refs);
    let current_stage_for_run = Arc::clone(&current_stage);
    let active_stage = Arc::new(ActiveStage::default());
    let run_fut = run_pipeline_loop(
        &pipeline_name,
        &stages,
        run_id,
        start_index,
        &input,
        state,
        stage_results,
        timeout_per_stage,
        persist_fn.as_ref(),
        persist_json_fn.as_ref(),
        &stream_tx,
        event_envelope_tx.as_ref(),
        &mut control_rx,
        &session_refs_for_run,
        &current_stage_for_run,
        &active_stage,
        &usage_totals,
    );

    let outcome = match total_timeout {
        Some(d) => match runtime::timeout(d, run_fut).await {
            Ok(inner) => inner,
            Err(_) => RunOutcome::Failed(PipelineError::Timeout {
                elapsed_ms: u64::try_from(d.as_millis()).unwrap_or(u64::MAX),
            }),
        },
        None => run_fut.await,
    };

    match outcome {
        RunOutcome::Completed {
            final_output,
            stage_results,
            shared_state,
            usage_total,
            cost_total_usd,
        } => {
            let pipeline_result = PipelineResult {
                pipeline_name,
                run_id,
                final_output,
                stage_results,
                shared_state,
                usage_total,
                cost_total_usd,
                session_refs: Arc::clone(&session_refs),
            };
            let _ = result_tx.send(Ok(pipeline_result));
        }
        RunOutcome::Paused(snapshot) => {
            let _ = snapshot_tx.send(snapshot);
            let _ = result_tx.send(Err(PipelineError::Paused));
        }
        RunOutcome::Aborted => {
            let _ = result_tx.send(Err(PipelineError::Aborted));
        }
        RunOutcome::Failed(err) => {
            let _ = result_tx.send(Err(err));
        }
    }
}

/// What the pipeline run-loop produced, to be turned into messages on the
/// outer channels by [`execute_pipeline`].
enum RunOutcome {
    Completed {
        final_output: serde_json::Value,
        stage_results: Vec<StageResult>,
        shared_state: std::collections::HashMap<String, serde_json::Value>,
        usage_total: blazen_llm::types::TokenUsage,
        cost_total_usd: f64,
    },
    Paused(PipelineSnapshot),
    Aborted,
    Failed(PipelineError),
}

/// The actual stage-driving loop. Returns a [`RunOutcome`] describing what
/// happened so the caller can send the appropriate messages on the
/// pipeline's result/snapshot channels. Wrapping the loop in a single
/// future also lets the caller cancel the entire run via
/// [`tokio::time::timeout`] for the total-pipeline-timeout feature.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_pipeline_loop<S>(
    pipeline_name: &str,
    stages: &[StageKind<S>],
    run_id: Uuid,
    start_index: usize,
    input: &serde_json::Value,
    mut state: PipelineState<S>,
    mut stage_results: Vec<StageResult>,
    timeout_per_stage: Option<Duration>,
    persist_fn: Option<&PersistFn>,
    persist_json_fn: Option<&PersistJsonFn>,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    event_envelope_tx: Option<&mpsc::UnboundedSender<EventEnvelope>>,
    control_rx: &mut mpsc::UnboundedReceiver<PipelineControl>,
    session_refs: &Arc<SessionRefRegistry>,
    current_stage: &Arc<AtomicUsize>,
    active_stage: &Arc<ActiveStage>,
    usage_totals: &Arc<Mutex<UsageTotals>>,
) -> RunOutcome
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize,
{
    let total_stages = u32::try_from(stages.len()).unwrap_or(u32::MAX);
    for (stage_idx, stage) in stages.iter().enumerate().skip(start_index) {
        // Publish per-stage progress to anyone subscribed to the pipeline
        // event stream BEFORE the stage actually starts. This is a typed
        // `ProgressEvent` (not the older `blazen::lifecycle` `DynamicEvent`)
        // wrapped in a `PipelineEvent` so consumers can downcast it.
        let current_1based = u32::try_from(stage_idx.saturating_add(1)).unwrap_or(u32::MAX);
        // Snapshot the index for `PipelineHandler::progress()` readers.
        // Store the 1-based index so an idle handler reports a sensible
        // "current_stage_index" without needing to know whether a stage
        // has started yet.
        current_stage.store(stage_idx.saturating_add(1), Ordering::Relaxed);
        let percent = if total_stages == 0 {
            0.0_f32
        } else {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let p = (f64::from(current_1based) / f64::from(total_stages) * 100.0) as f32;
            p.clamp(0.0, 100.0)
        };
        let progress = ProgressEvent {
            kind: ProgressKind::Pipeline,
            current: current_1based,
            total: Some(total_stages),
            percent: Some(percent),
            label: stage.name().to_owned(),
            run_id,
        };
        let progress_event: Box<dyn AnyEvent> = Box::new(progress);
        if let Some(env_tx) = event_envelope_tx {
            let env =
                EventEnvelope::new(progress_event.clone_boxed(), Some(stage.name().to_owned()));
            let _ = env_tx.send(env);
        }
        let _ = stream_tx.send(PipelineEvent {
            stage_name: stage.name().to_owned(),
            branch_name: None,
            workflow_run_id: Uuid::nil(),
            event: progress_event,
        });
        // Drain any control signals that arrived between stages (non-blocking).
        while let Ok(control) = control_rx.try_recv() {
            match control {
                PipelineControl::Pause => {
                    let snapshot = build_snapshot(
                        pipeline_name,
                        run_id,
                        stage_idx,
                        0,
                        &stage_results,
                        &state,
                        input,
                    );
                    return RunOutcome::Paused(snapshot);
                }
                PipelineControl::Resume | PipelineControl::RespondToInput { .. } => {
                    // No-op between stages -- nothing is parked / awaiting input.
                }
                PipelineControl::Snapshot { reply } => {
                    let snap = build_snapshot(
                        pipeline_name,
                        run_id,
                        stage_idx,
                        0,
                        &stage_results,
                        &state,
                        input,
                    );
                    let _ = reply.send(Ok(snap));
                }
                PipelineControl::Abort => {
                    return RunOutcome::Aborted;
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

        // Loop stages need mutable access to the shared state across rounds
        // (so each round's output is visible to the `until` predicate) and
        // handle pause/abort at round boundaries internally. They are driven
        // outside the `tokio::select!` race used for the leaf stage kinds.
        let result = if let StageKind::Loop(loop_stage) = stage {
            match run_loop_stage(
                loop_stage,
                &mut state,
                stream_tx,
                event_envelope_tx,
                timeout_per_stage,
                session_refs,
                control_rx,
                pipeline_name,
                run_id,
                stage_idx,
                &stage_results,
                input,
                active_stage,
            )
            .instrument(tracing::info_span!(
                "pipeline.stage.loop",
                stage_name = %loop_stage.name,
                max_iterations = loop_stage.max_iterations,
            ))
            .await
            {
                LoopOutcome::Finished(result) => result,
                LoopOutcome::Paused(snapshot) => return RunOutcome::Paused(snapshot),
                LoopOutcome::Aborted => return RunOutcome::Aborted,
            }
        } else {
            // Race stage execution against control signals so pause/abort can
            // interrupt a running stage, and snapshot/resume/respond-to-input
            // can be serviced WITHOUT dropping the in-flight stage. The inner
            // workflow handler(s) live in `active_stage`, so the stage future
            // can be polled across control messages without being cancelled.
            let stage_future =
                async {
                    match stage {
                        StageKind::Sequential(s) => run_sequential_stage(
                            s,
                            &state,
                            stream_tx,
                            event_envelope_tx,
                            timeout_per_stage,
                            session_refs,
                            active_stage,
                        )
                        .instrument(
                            tracing::info_span!("pipeline.stage.sequential", stage_name = %s.name),
                        )
                        .await,
                        StageKind::Parallel(p) => {
                            run_parallel_stage(
                                p,
                                &state,
                                stream_tx,
                                event_envelope_tx,
                                timeout_per_stage,
                                session_refs,
                                active_stage,
                            )
                            .instrument(tracing::info_span!(
                                "pipeline.stage.parallel",
                                branch_count = p.branches.len()
                            ))
                            .await
                        }
                        // Loop is handled above with mutable state; unreachable here.
                        StageKind::Loop(_) => unreachable!("loop stage handled above"),
                    }
                };
            let mut stage_future = std::pin::pin!(stage_future);

            loop {
                tokio::select! {
                    biased;

                    // Control signal -- takes priority when both are ready.
                    Some(control) = control_rx.recv() => {
                        match control {
                            PipelineControl::Pause => {
                                // Drop `active_stage`'s handlers; their Drop impls
                                // send Abort to the inner workflow event loops.
                                active_stage.clear().await;
                                let snapshot = build_snapshot(
                                    pipeline_name,
                                    run_id,
                                    stage_idx,
                                    0,
                                    &stage_results,
                                    &state,
                                    input,
                                );
                                return RunOutcome::Paused(snapshot);
                            }
                            PipelineControl::Resume => {
                                // Forward to the active inner workflow(s) so one
                                // parked on an InputRequestEvent (or paused) unparks.
                                active_stage.forward_resume().await;
                            }
                            PipelineControl::RespondToInput { response } => {
                                active_stage.forward_input(&response).await;
                            }
                            PipelineControl::Snapshot { reply } => {
                                let snap = build_snapshot(
                                    pipeline_name,
                                    run_id,
                                    stage_idx,
                                    0,
                                    &stage_results,
                                    &state,
                                    input,
                                );
                                let _ = reply.send(Ok(snap));
                            }
                            PipelineControl::Abort => {
                                active_stage.clear().await;
                                return RunOutcome::Aborted;
                            }
                        }
                    }

                    res = &mut stage_future => {
                        active_stage.clear().await;
                        break res;
                    }
                }
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let duration_ms = stage_start.elapsed().as_millis() as u64;

        match result {
            Ok(StageSuccess {
                output: stage_output,
                usage: stage_usage,
                cost_usd: stage_cost,
            }) => {
                stage_span.record("duration_ms", duration_ms);
                stage_span.record("skipped", false);

                // Roll per-stage totals into the pipeline-wide totals.
                state.usage_total.add(&stage_usage);
                state.cost_total_usd += stage_cost;

                // Mirror the same delta into the shared running totals so a
                // live `PipelineHandler::usage_total()` reflects completed
                // stages mid-run.
                {
                    let mut totals = usage_totals.lock().await;
                    totals.usage.add(&stage_usage);
                    totals.cost_usd += stage_cost;
                }

                let usage_for_result = if stage_usage == blazen_llm::types::TokenUsage::default() {
                    None
                } else {
                    Some(stage_usage)
                };
                let cost_for_result = if stage_cost == 0.0 {
                    None
                } else {
                    Some(stage_cost)
                };

                let sr = StageResult {
                    name: stage.name().to_owned(),
                    output: stage_output.clone(),
                    skipped: false,
                    duration_ms,
                    usage: usage_for_result,
                    cost_usd: cost_for_result,
                };
                // Capture the output mapper (if any) for Sequential stages
                // before we move `stage_output` into the shared-state
                // result map. Parallel and Loop stages don't carry an
                // `output_mapper` field — only `Stage<S>` does — so they
                // simply skip this step.
                let output_mapper = match stage {
                    StageKind::Sequential(s) => s.output_mapper.clone(),
                    StageKind::Parallel(_) | StageKind::Loop(_) => None,
                };
                let mapper_input = output_mapper.as_ref().map(|_| stage_output.clone());
                state.record_stage_result(stage.name().to_owned(), stage_output);
                if let (Some(om), Some(value)) = (output_mapper, mapper_input)
                    && let Err(e) = om(&mut state, value)
                {
                    return RunOutcome::Failed(e);
                }
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
                    usage: None,
                    cost_usd: None,
                };
                stage_results.push(sr);
            }
            Err(StageOutcome::Failed(e)) => {
                stage_span.record("duration_ms", duration_ms);
                return RunOutcome::Failed(e);
            }
        }

        // Call persist callbacks after each stage.
        if let Err(e) = call_persist_callbacks(
            persist_fn,
            persist_json_fn,
            pipeline_name,
            run_id,
            stage_idx + 1,
            &stage_results,
            &state,
            input,
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
    let shared_state = shared_state_to_map(&state);
    let usage_total = state.usage_total().clone();
    let cost_total_usd = state.cost_total_usd();
    RunOutcome::Completed {
        final_output,
        stage_results,
        shared_state,
        usage_total,
        cost_total_usd,
    }
}

/// Serialize the typed `PipelineState::<S>` shared slot into a flat
/// `HashMap<String, serde_json::Value>` for the JSON-only snapshot/result
/// surface. If the typed state isn't a JSON object (e.g. `S` is a tuple or
/// scalar) or serialization fails, we fall back to an empty map -- the
/// snapshot/result containers carry the JSON-shape contract regardless of
/// the typed-state shape.
fn shared_state_to_map<S>(
    state: &PipelineState<S>,
) -> std::collections::HashMap<String, serde_json::Value>
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize,
{
    state
        .shared_to_json()
        .ok()
        .and_then(|v| match v {
            serde_json::Value::Object(map) => {
                Some(map.into_iter().collect::<std::collections::HashMap<_, _>>())
            }
            _ => None,
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Stage execution helpers
// ---------------------------------------------------------------------------

/// Outcome for stage execution that distinguishes between skipped and failed.
enum StageOutcome {
    Skipped,
    Failed(PipelineError),
}

/// Output of a successfully-executed stage: the workflow's final value plus
/// per-stage usage / cost totals aggregated from `UsageEvent`s emitted
/// while the stage was running.
struct StageSuccess {
    output: serde_json::Value,
    usage: blazen_llm::types::TokenUsage,
    cost_usd: f64,
}

/// Run a single sequential stage.
#[allow(clippy::similar_names, clippy::too_many_lines)]
async fn run_sequential_stage<S>(
    stage: &Stage<S>,
    state: &PipelineState<S>,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    event_envelope_tx: Option<&mpsc::UnboundedSender<EventEnvelope>>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
    active_stage: &Arc<ActiveStage>,
) -> Result<StageSuccess, StageOutcome>
where
    S: Default + Clone + Send + Sync + 'static,
{
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

    // Run the workflow with the shared session-ref registry. Detach the
    // result receiver so the handler can stay alive in `active_stage` for
    // pipeline-level control forwarding (snapshot / resume / respond-to-input)
    // while we await the terminal event on the receiver.
    let mut handler = stage
        .workflow
        .run_with_registry(workflow_input, Arc::clone(session_refs))
        .await
        .map_err(|e| StageOutcome::Failed(PipelineError::Workflow(e)))?;
    let result_rx = handler
        .take_result_rx()
        .expect("fresh workflow handler always has a result receiver");
    let handler = Arc::new(handler);

    // Subscribe to the workflow's event stream BEFORE the `set_current` await
    // point below: `stream_events()` is a `broadcast::Receiver` that only sees
    // events sent AFTER it subscribes, and `set_current` yields. Subscribing
    // first shrinks the window in which a very fast workflow could emit (and we
    // miss) its `StreamEnd` sentinel. The bounded await on `forward_handle`
    // further down handles the residual race so we can never hang.
    let stage_name = stage.name.clone();
    let stream_tx_clone = stream_tx.clone();
    let mut wf_stream = handler.stream_events();

    active_stage.set_current(Arc::clone(&handler)).await;

    // Forward events in a separate task while awaiting the result.
    // Returns the per-stage usage / cost totals it observed.
    let env_tx_clone = event_envelope_tx.cloned();
    let mut forward_handle: JoinHandle<(blazen_llm::types::TokenUsage, f64)> = runtime::spawn({
        let stage_name = stage_name.clone();
        async move {
            let mut usage = blazen_llm::types::TokenUsage::default();
            let mut cost: f64 = 0.0;
            while let Some(event) = wf_stream.next().await {
                // Break on the workflow's stream-end sentinel. Without
                // this, the loop relies on the broadcast Sender count
                // dropping to zero — which happens reliably on native
                // (WorkflowHandler::result drops its sender after the
                // event loop joins) but NOT on wasi/Workers, where
                // step-handler `JsContext` clones can outlive the event
                // loop until JS GC reclaims them. Mirrors the
                // `Workflow::run_streaming` forwarder in `blazen-node`.
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                if let Some(ue) = event.as_any().downcast_ref::<blazen_events::UsageEvent>() {
                    usage.add(&blazen_llm::types::TokenUsage {
                        prompt_tokens: ue.prompt_tokens,
                        completion_tokens: ue.completion_tokens,
                        total_tokens: ue.total_tokens,
                        reasoning_tokens: ue.reasoning_tokens,
                        cached_input_tokens: ue.cached_input_tokens,
                        audio_input_tokens: ue.audio_input_tokens,
                        audio_output_tokens: ue.audio_output_tokens,
                    });
                    if let Some(c) = ue.cost_usd {
                        cost += c;
                    }
                }
                if let Some(env_tx) = &env_tx_clone {
                    let env = EventEnvelope::new(event.clone_boxed(), Some(stage_name.clone()));
                    let _ = env_tx.send(env);
                }
                let pipeline_event = PipelineEvent {
                    stage_name: stage_name.clone(),
                    branch_name: None,
                    workflow_run_id: Uuid::nil(),
                    event,
                };
                let _ = stream_tx_clone.send(pipeline_event);
            }
            (usage, cost)
        }
    });

    // Await the workflow's terminal event via the detached receiver,
    // optionally with a timeout. The handler stays alive in `active_stage`
    // so pipeline-level control can be forwarded to it concurrently; the
    // supervisor clears it once the stage ends.
    #[allow(clippy::single_match_else)]
    let wf_result = if let Some(timeout_dur) = timeout {
        match runtime::timeout(timeout_dur, result_rx).await {
            Ok(r) => r.unwrap_or(Err(blazen_core::WorkflowError::ChannelClosed)),
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
        result_rx
            .await
            .unwrap_or(Err(blazen_core::WorkflowError::ChannelClosed))
    };

    match wf_result {
        Ok(event) => {
            // Wait for the forwarding task to finish cleanly so we capture
            // every UsageEvent emitted before the workflow shut down — but
            // BOUNDED. The workflow's terminal result already arrived, so its
            // `StreamEnd` sentinel is imminent. If the forward task never sees
            // it (a `broadcast` subscribe-after-emit race for trivial
            // workflows, made unrecoverable because the live `WorkflowHandler`
            // in `active_stage` keeps a stream `Sender` clone alive so the
            // receiver never closes), abort rather than hang forever. In that
            // race the task observed no events anyway, so the usage it would
            // have reported was already lost — bounding costs nothing extra.
            let (stage_usage, stage_cost) = if let Ok(joined) =
                runtime::timeout(Duration::from_secs(3), &mut forward_handle).await
            {
                joined.unwrap_or_default()
            } else {
                forward_handle.abort();
                tracing::warn!(
                    stage = %stage_name,
                    "event-forward task did not observe StreamEnd; usage totals may be incomplete"
                );
                <(blazen_llm::types::TokenUsage, f64)>::default()
            };
            let output = extract_stop_result(&*event);
            Ok(StageSuccess {
                output,
                usage: stage_usage,
                cost_usd: stage_cost,
            })
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

/// What a [`run_loop_stage`] invocation produced.
enum LoopOutcome {
    /// The loop ran to completion (or failed); carries the stage result so
    /// the caller folds it into the pipeline like any other stage.
    Finished(Result<StageSuccess, StageOutcome>),
    /// A pause control signal arrived at a round boundary; the snapshot
    /// carries the current iteration so the loop can be resumed later.
    Paused(PipelineSnapshot),
    /// An abort control signal arrived at a round boundary.
    Aborted,
}

/// Run a loop stage: re-run the inner stage until the `until` predicate
/// returns [`LoopDecision::Done`] / [`LoopDecision::Abort`], or until
/// `max_iterations` rounds have run.
///
/// Each round's inner-stage output is recorded into `state` under the inner
/// stage's name so the `until` predicate (and `on_round_complete` hook) can
/// observe per-round progress. The final round's output is returned as the
/// loop stage's own [`StageSuccess`].
///
/// Pause / resume / abort control signals are honored at round boundaries,
/// mirroring the between-stages handling in [`run_pipeline_loop`]. A paused
/// snapshot carries the current iteration via
/// [`PipelineSnapshot::current_iteration`].
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_loop_stage<S>(
    loop_stage: &LoopStage<S>,
    state: &mut PipelineState<S>,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    event_envelope_tx: Option<&mpsc::UnboundedSender<EventEnvelope>>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
    control_rx: &mut mpsc::UnboundedReceiver<PipelineControl>,
    pipeline_name: &str,
    run_id: Uuid,
    stage_idx: usize,
    stage_results: &[StageResult],
    input: &serde_json::Value,
    active_stage: &Arc<ActiveStage>,
) -> LoopOutcome
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize,
{
    // Reject a nested loop up front with a clear error.
    if let StageKind::Loop(_) = loop_stage.inner.as_ref() {
        return LoopOutcome::Finished(Err(StageOutcome::Failed(PipelineError::ValidationFailed(
            format!(
                "nested loop not supported (loop stage '{}')",
                loop_stage.name
            ),
        ))));
    }

    let mut last_success: Option<StageSuccess> = None;
    let mut iteration: u32 = 0;
    let mut total_usage = blazen_llm::types::TokenUsage::default();
    let mut total_cost: f64 = 0.0;

    loop {
        // Cap on iterations: stop cleanly once we've run `max_iterations`.
        if iteration >= loop_stage.max_iterations {
            break;
        }

        // Honor control signals between rounds (non-blocking).
        while let Ok(control) = control_rx.try_recv() {
            match control {
                PipelineControl::Pause => {
                    let snapshot = build_snapshot(
                        pipeline_name,
                        run_id,
                        stage_idx,
                        iteration,
                        stage_results,
                        state,
                        input,
                    );
                    return LoopOutcome::Paused(snapshot);
                }
                PipelineControl::Resume | PipelineControl::RespondToInput { .. } => {}
                PipelineControl::Snapshot { reply } => {
                    let snap = build_snapshot(
                        pipeline_name,
                        run_id,
                        stage_idx,
                        iteration,
                        stage_results,
                        state,
                        input,
                    );
                    let _ = reply.send(Ok(snap));
                }
                PipelineControl::Abort => return LoopOutcome::Aborted,
            }
        }

        // Run the inner stage for this round, racing against control signals
        // so pause/abort can interrupt a long-running round while
        // snapshot/resume/respond-to-input are serviced without dropping it.
        let inner_future = async {
            match loop_stage.inner.as_ref() {
                StageKind::Sequential(s) => {
                    run_sequential_stage(
                        s,
                        state,
                        stream_tx,
                        event_envelope_tx,
                        timeout,
                        session_refs,
                        active_stage,
                    )
                    .await
                }
                StageKind::Parallel(p) => {
                    run_parallel_stage(
                        p,
                        state,
                        stream_tx,
                        event_envelope_tx,
                        timeout,
                        session_refs,
                        active_stage,
                    )
                    .await
                }
                // Rejected above before the loop began.
                StageKind::Loop(_) => unreachable!("nested loop rejected above"),
            }
        };
        let round_result = {
            let mut inner_future = std::pin::pin!(inner_future);
            loop {
                tokio::select! {
                    biased;

                    Some(control) = control_rx.recv() => {
                        match control {
                            PipelineControl::Pause => {
                                active_stage.clear().await;
                                let snapshot = build_snapshot(
                                    pipeline_name,
                                    run_id,
                                    stage_idx,
                                    iteration,
                                    stage_results,
                                    state,
                                    input,
                                );
                                return LoopOutcome::Paused(snapshot);
                            }
                            PipelineControl::Resume => active_stage.forward_resume().await,
                            PipelineControl::RespondToInput { response } => {
                                active_stage.forward_input(&response).await;
                            }
                            PipelineControl::Snapshot { reply } => {
                                let snap = build_snapshot(
                                    pipeline_name,
                                    run_id,
                                    stage_idx,
                                    iteration,
                                    stage_results,
                                    state,
                                    input,
                                );
                                let _ = reply.send(Ok(snap));
                            }
                            PipelineControl::Abort => {
                                active_stage.clear().await;
                                return LoopOutcome::Aborted;
                            }
                        }
                    }

                    res = &mut inner_future => {
                        active_stage.clear().await;
                        break res;
                    }
                }
            }
        };

        match round_result {
            Ok(success) => {
                total_usage.add(&success.usage);
                total_cost += success.cost_usd;
                // Record this round's output under the inner stage name so
                // the `until` predicate can observe it via `stage_result`.
                state.record_stage_result(
                    loop_stage.inner.name().to_owned(),
                    success.output.clone(),
                );
                last_success = Some(success);
            }
            // A skipped inner stage still counts as a completed round but
            // contributes no output/usage.
            Err(StageOutcome::Skipped) => {}
            Err(StageOutcome::Failed(e)) => {
                return LoopOutcome::Finished(Err(StageOutcome::Failed(e)));
            }
        }

        // Fire the per-round hook (after the round, before the decision).
        if let Some(hook) = &loop_stage.on_round_complete {
            hook(iteration, state).await;
        }

        iteration += 1;

        match (loop_stage.until)(state, iteration) {
            LoopDecision::Done => break,
            // Fall through to the next round.
            LoopDecision::Continue => {}
            LoopDecision::Abort(msg) => {
                return LoopOutcome::Finished(Err(StageOutcome::Failed(PipelineError::Workflow(
                    blazen_core::WorkflowError::Other(anyhow::anyhow!(msg)),
                ))));
            }
        }
    }

    let output = last_success
        .as_ref()
        .map_or(serde_json::Value::Null, |s| s.output.clone());
    LoopOutcome::Finished(Ok(StageSuccess {
        output,
        usage: total_usage,
        cost_usd: total_cost,
    }))
}

/// Run a parallel stage with multiple branches.
async fn run_parallel_stage<S>(
    parallel: &ParallelStage<S>,
    state: &PipelineState<S>,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    event_envelope_tx: Option<&mpsc::UnboundedSender<EventEnvelope>>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
    active_stage: &Arc<ActiveStage>,
) -> Result<StageSuccess, StageOutcome>
where
    S: Default + Clone + Send + Sync + 'static,
{
    match parallel.join_strategy {
        JoinStrategy::WaitAll => {
            run_parallel_wait_all(
                parallel,
                state,
                stream_tx,
                event_envelope_tx,
                timeout,
                session_refs,
                active_stage,
            )
            .await
        }
        JoinStrategy::FirstCompletes => {
            run_parallel_first_completes(
                parallel,
                state,
                stream_tx,
                event_envelope_tx,
                timeout,
                session_refs,
                active_stage,
            )
            .await
        }
    }
}

/// Run all branches and wait for all to complete.
#[allow(clippy::too_many_lines)]
async fn run_parallel_wait_all<S>(
    parallel: &ParallelStage<S>,
    state: &PipelineState<S>,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    event_envelope_tx: Option<&mpsc::UnboundedSender<EventEnvelope>>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
    active_stage: &Arc<ActiveStage>,
) -> Result<StageSuccess, StageOutcome>
where
    S: Default + Clone + Send + Sync + 'static,
{
    let mut set = JoinSet::new();
    let mut forward_handles: Vec<JoinHandle<(blazen_llm::types::TokenUsage, f64)>> = Vec::new();

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

        let mut handler = match branch
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
        // Detach the result receiver and register the live handler so
        // pipeline-level control can fan out to every parallel branch.
        let result_rx = handler
            .take_result_rx()
            .expect("fresh workflow handler always has a result receiver");
        let handler = Arc::new(handler);
        active_stage.push_branch(Arc::clone(&handler)).await;

        // Forward events from this branch and accumulate UsageEvent totals.
        let mut wf_stream = handler.stream_events();
        let fwd_stage = stage_name;
        let fwd_branch = branch_name.clone();
        let fwd_tx = stream_tx.clone();
        let fwd_env_tx = event_envelope_tx.cloned();
        let fh: JoinHandle<(blazen_llm::types::TokenUsage, f64)> = runtime::spawn(async move {
            let mut usage = blazen_llm::types::TokenUsage::default();
            let mut cost: f64 = 0.0;
            while let Some(event) = wf_stream.next().await {
                // Break on the workflow's stream-end sentinel so the forwarder
                // terminates deterministically even when a step-handler
                // Context clone keeps a broadcast sender alive past the event
                // loop's exit. Mirrors the sequential-stage forwarder.
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                if let Some(ue) = event.as_any().downcast_ref::<blazen_events::UsageEvent>() {
                    usage.add(&blazen_llm::types::TokenUsage {
                        prompt_tokens: ue.prompt_tokens,
                        completion_tokens: ue.completion_tokens,
                        total_tokens: ue.total_tokens,
                        reasoning_tokens: ue.reasoning_tokens,
                        cached_input_tokens: ue.cached_input_tokens,
                        audio_input_tokens: ue.audio_input_tokens,
                        audio_output_tokens: ue.audio_output_tokens,
                    });
                    if let Some(c) = ue.cost_usd {
                        cost += c;
                    }
                }
                if let Some(env_tx) = &fwd_env_tx {
                    let env = EventEnvelope::new(event.clone_boxed(), Some(fwd_stage.clone()));
                    let _ = env_tx.send(env);
                }
                let pipeline_event = PipelineEvent {
                    stage_name: fwd_stage.clone(),
                    branch_name: Some(fwd_branch.clone()),
                    workflow_run_id: Uuid::nil(),
                    event,
                };
                let _ = fwd_tx.send(pipeline_event);
            }
            (usage, cost)
        });
        forward_handles.push(fh);

        set.spawn(async move {
            // Keep the Arc alive for the duration of the await so control
            // forwarding stays valid until the branch resolves.
            let _handler = handler;
            let result = if let Some(t) = timeout {
                match runtime::timeout(t, result_rx).await {
                    Ok(r) => r.unwrap_or(Err(blazen_core::WorkflowError::ChannelClosed)),
                    Err(_) => Err(blazen_core::WorkflowError::Timeout { elapsed: t }),
                }
            } else {
                result_rx
                    .await
                    .unwrap_or(Err(blazen_core::WorkflowError::ChannelClosed))
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

    // Drain forward handles to harvest per-branch usage / cost totals, then
    // sum into the parent stage's totals. Bounded: every branch already
    // produced its terminal result above, so its `StreamEnd` is imminent; if a
    // forwarder never observes it (the broadcast subscribe-after-emit race —
    // see `run_sequential_stage`), drop it rather than hang the whole pipeline.
    let mut stage_usage = blazen_llm::types::TokenUsage::default();
    let mut stage_cost: f64 = 0.0;
    for mut fh in forward_handles {
        let (u, c) = if let Ok(joined) = runtime::timeout(Duration::from_secs(3), &mut fh).await {
            joined.unwrap_or_default()
        } else {
            fh.abort();
            <(blazen_llm::types::TokenUsage, f64)>::default()
        };
        stage_usage.add(&u);
        stage_cost += c;
    }

    Ok(StageSuccess {
        output: serde_json::Value::Object(results),
        usage: stage_usage,
        cost_usd: stage_cost,
    })
}

/// Run all branches and return the first to complete.
#[allow(clippy::too_many_lines)]
async fn run_parallel_first_completes<S>(
    parallel: &ParallelStage<S>,
    state: &PipelineState<S>,
    stream_tx: &broadcast::Sender<PipelineEvent>,
    event_envelope_tx: Option<&mpsc::UnboundedSender<EventEnvelope>>,
    timeout: Option<Duration>,
    session_refs: &Arc<SessionRefRegistry>,
    active_stage: &Arc<ActiveStage>,
) -> Result<StageSuccess, StageOutcome>
where
    S: Default + Clone + Send + Sync + 'static,
{
    let mut set = JoinSet::new();
    let mut forward_handles: Vec<JoinHandle<(blazen_llm::types::TokenUsage, f64)>> = Vec::new();

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

        let mut handler = match branch
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
        let result_rx = handler
            .take_result_rx()
            .expect("fresh workflow handler always has a result receiver");
        let handler = Arc::new(handler);
        active_stage.push_branch(Arc::clone(&handler)).await;

        let mut wf_stream = handler.stream_events();
        let fwd_stage = stage_name;
        let fwd_branch = branch_name.clone();
        let fwd_tx = stream_tx.clone();
        let fwd_env_tx = event_envelope_tx.cloned();
        let fh: JoinHandle<(blazen_llm::types::TokenUsage, f64)> = runtime::spawn(async move {
            let mut usage = blazen_llm::types::TokenUsage::default();
            let mut cost: f64 = 0.0;
            while let Some(event) = wf_stream.next().await {
                // Break on the workflow's stream-end sentinel so the forwarder
                // terminates deterministically even when a step-handler
                // Context clone keeps a broadcast sender alive past the event
                // loop's exit. Mirrors the sequential-stage forwarder.
                if event.event_type_id() == "blazen::StreamEnd" {
                    break;
                }
                if let Some(ue) = event.as_any().downcast_ref::<blazen_events::UsageEvent>() {
                    usage.add(&blazen_llm::types::TokenUsage {
                        prompt_tokens: ue.prompt_tokens,
                        completion_tokens: ue.completion_tokens,
                        total_tokens: ue.total_tokens,
                        reasoning_tokens: ue.reasoning_tokens,
                        cached_input_tokens: ue.cached_input_tokens,
                        audio_input_tokens: ue.audio_input_tokens,
                        audio_output_tokens: ue.audio_output_tokens,
                    });
                    if let Some(c) = ue.cost_usd {
                        cost += c;
                    }
                }
                if let Some(env_tx) = &fwd_env_tx {
                    let env = EventEnvelope::new(event.clone_boxed(), Some(fwd_stage.clone()));
                    let _ = env_tx.send(env);
                }
                let pipeline_event = PipelineEvent {
                    stage_name: fwd_stage.clone(),
                    branch_name: Some(fwd_branch.clone()),
                    workflow_run_id: Uuid::nil(),
                    event,
                };
                let _ = fwd_tx.send(pipeline_event);
            }
            (usage, cost)
        });
        forward_handles.push(fh);

        set.spawn(async move {
            // Keep the Arc alive across the await so control forwarding to
            // this branch remains valid until it resolves.
            let _handler = handler;
            let result = if let Some(t) = timeout {
                match runtime::timeout(t, result_rx).await {
                    Ok(r) => r.unwrap_or(Err(blazen_core::WorkflowError::ChannelClosed)),
                    Err(_) => Err(blazen_core::WorkflowError::Timeout { elapsed: t }),
                }
            } else {
                result_rx
                    .await
                    .unwrap_or(Err(blazen_core::WorkflowError::ChannelClosed))
            };
            (branch_name, result)
        });
    }

    // Return the first successful result.
    let outcome: Result<serde_json::Value, StageOutcome> = if let Some(join_result) =
        set.join_next().await
    {
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
    };

    // Drain forward handles to harvest per-branch usage / cost totals.
    // Losing branches are aborted via `set.abort_all()` above; their event
    // streams close shortly after, so awaiting these handles will not hang
    // on a stuck consumer. Use a short timeout per handle to be safe.
    let mut stage_usage = blazen_llm::types::TokenUsage::default();
    let mut stage_cost: f64 = 0.0;
    for fh in forward_handles {
        let (u, c) = match runtime::timeout(Duration::from_millis(50), fh).await {
            Ok(joined) => joined.unwrap_or_default(),
            Err(_) => (blazen_llm::types::TokenUsage::default(), 0.0),
        };
        stage_usage.add(&u);
        stage_cost += c;
    }

    match outcome {
        Ok(output) => Ok(StageSuccess {
            output,
            usage: stage_usage,
            cost_usd: stage_cost,
        }),
        Err(e) => Err(e),
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
#[allow(clippy::too_many_arguments)]
fn build_snapshot<S>(
    pipeline_name: &str,
    run_id: Uuid,
    current_stage_index: usize,
    current_iteration: u32,
    stage_results: &[StageResult],
    state: &PipelineState<S>,
    input: &serde_json::Value,
) -> PipelineSnapshot
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize,
{
    PipelineSnapshot {
        pipeline_name: pipeline_name.to_owned(),
        run_id,
        timestamp: Utc::now(),
        current_stage_index,
        completed_stages: stage_results.to_vec(),
        active_snapshots: Vec::new(),
        shared_state: shared_state_to_map(state),
        input: input.clone(),
        current_iteration,
    }
}

/// Call persist callbacks if configured.
#[allow(clippy::too_many_arguments)]
async fn call_persist_callbacks<S>(
    persist_fn: Option<&PersistFn>,
    persist_json_fn: Option<&PersistJsonFn>,
    pipeline_name: &str,
    run_id: Uuid,
    next_stage_index: usize,
    stage_results: &[StageResult],
    state: &PipelineState<S>,
    input: &serde_json::Value,
) -> Result<(), PipelineError>
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize,
{
    let snapshot = build_snapshot(
        pipeline_name,
        run_id,
        next_stage_index,
        0,
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

// ---------------------------------------------------------------------------
// SubExecutable impl — embed a Pipeline inside a parent Workflow
// ---------------------------------------------------------------------------

/// Lets a [`Pipeline`] be wrapped in a
/// [`SubPipelineStep`](blazen_core::SubPipelineStep) and embedded as a
/// child runner inside a parent [`blazen_core::Workflow`].
///
/// The trait takes `&self`, so each invocation clones the pipeline
/// blueprint (cheap — every field is `Arc`-backed or `Copy`) and runs
/// it to completion. The returned JSON is the pipeline's
/// [`PipelineResult::final_output`](crate::snapshot::PipelineResult).
#[async_trait::async_trait]
impl<S> blazen_core::SubExecutable for Pipeline<S>
where
    S: Default + Clone + Send + Sync + 'static + serde::Serialize + serde::de::DeserializeOwned,
{
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: blazen_core::Context,
    ) -> Result<serde_json::Value, blazen_core::WorkflowError> {
        let handler = self.clone().start(input);
        let result = handler
            .result()
            .await
            .map_err(|e| blazen_core::WorkflowError::Other(anyhow::Error::new(e)))?;
        serde_json::to_value(&result.final_output).map_err(blazen_core::WorkflowError::from)
    }
}
