//! Pipeline surface for the UniFFI bindings.
//!
//! A pipeline composes one or more [`Workflow`](crate::workflow::Workflow)s
//! into an ordered run. The first stage's input is the JSON payload passed
//! to [`Pipeline::run`]; each subsequent stage receives the previous
//! stage's `StopEvent` result as its `StartEvent` payload.
//!
//! ## Why a thin re-implementation here?
//!
//! The full [`blazen_pipeline`] orchestrator wants an **owned**
//! `blazen_core::Workflow` per stage, but UniFFI workflows are surfaced as
//! long-lived `Arc<crate::workflow::Workflow>` handles (the foreign caller
//! keeps a reference). Rather than fight the borrow checker to reclaim the
//! inner workflow at build time, this module implements a minimal,
//! FFI-friendly sequential / parallel orchestrator on top of the
//! [`Workflow::run`](crate::workflow::Workflow) async surface. Conditions,
//! input mappers, persistence callbacks, and pause/resume snapshots are
//! intentionally deferred to a later phase if foreign demand materialises.
//!
//! ## Example (Go)
//!
//! ```go,ignore
//! ingest, _ := blazen.NewWorkflowBuilder("ingest").Step(...).Build()
//! enrich, _ := blazen.NewWorkflowBuilder("enrich").Step(...).Build()
//!
//! pipe, _ := blazen.NewPipelineBuilder("etl").
//!     AddWorkflow(ingest).
//!     AddWorkflow(enrich).
//!     TotalTimeoutMs(60_000).
//!     Build()
//!
//! result, err := pipe.Run(`{"source":"s3://..."}`)
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinSet;

use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;
use crate::workflow::{Event, Workflow, WorkflowResult};

// ---------------------------------------------------------------------------
// Stage representation
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum StageDef {
    Sequential {
        name: String,
        workflow: Arc<Workflow>,
    },
    Parallel {
        name: String,
        branches: Vec<(String, Arc<Workflow>)>,
        wait_all: bool,
    },
}

impl StageDef {
    fn name(&self) -> &str {
        match self {
            StageDef::Sequential { name, .. } | StageDef::Parallel { name, .. } => name,
        }
    }
}

struct BuilderState {
    name: String,
    stages: Vec<StageDef>,
    timeout_per_stage: Option<Duration>,
    total_timeout: Option<Duration>,
    /// Auto-incremented index used to mint stage names for
    /// [`PipelineBuilder::add_workflow`] calls that don't supply one.
    next_auto_id: u32,
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Builder for a [`Pipeline`].
///
/// Use [`PipelineBuilder::new`] to start, attach workflows via
/// [`add_workflow`](Self::add_workflow) / [`stage`](Self::stage) /
/// [`parallel`](Self::parallel), then call [`build`](Self::build) to
/// validate and produce a runnable [`Pipeline`].
#[derive(uniffi::Object)]
pub struct PipelineBuilder {
    state: parking_lot::Mutex<Option<BuilderState>>,
}

impl PipelineBuilder {
    fn take_state(&self) -> BlazenResult<BuilderState> {
        self.state.lock().take().ok_or(BlazenError::Validation {
            message: "PipelineBuilder already consumed".into(),
        })
    }

    fn replace_state(&self, state: BuilderState) {
        *self.state.lock() = Some(state);
    }
}

#[uniffi::export]
impl PipelineBuilder {
    /// Create a new builder with the given pipeline name.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(name: String) -> Arc<Self> {
        Arc::new(Self {
            state: parking_lot::Mutex::new(Some(BuilderState {
                name,
                stages: Vec::new(),
                timeout_per_stage: None,
                total_timeout: None,
                next_auto_id: 0,
            })),
        })
    }

    /// Append a sequential workflow stage with an auto-generated stage name
    /// of the form `"stage-{N}"` (zero-based).
    ///
    /// Use [`stage`](Self::stage) when the stage name matters for
    /// downstream tooling that filters by it.
    pub fn add_workflow(
        self: Arc<Self>,
        workflow: Arc<Workflow>,
    ) -> BlazenResult<Arc<PipelineBuilder>> {
        let mut state = self.take_state()?;
        let name = format!("stage-{}", state.next_auto_id);
        state.next_auto_id += 1;
        state.stages.push(StageDef::Sequential { name, workflow });
        self.replace_state(state);
        Ok(self)
    }

    /// Append a sequential stage with an explicit name. The stage name must
    /// be unique within the pipeline (enforced at [`build`](Self::build)).
    pub fn stage(
        self: Arc<Self>,
        name: String,
        workflow: Arc<Workflow>,
    ) -> BlazenResult<Arc<PipelineBuilder>> {
        let mut state = self.take_state()?;
        state.stages.push(StageDef::Sequential { name, workflow });
        self.replace_state(state);
        Ok(self)
    }

    /// Append a parallel stage running multiple workflows concurrently.
    ///
    /// `branch_names` and `workflows` are positionally paired; a length
    /// mismatch yields [`BlazenError::Validation`]. When `wait_all` is
    /// `true` every branch must complete and outputs are collected into a
    /// JSON object keyed by branch name. When `wait_all` is `false` the
    /// pipeline proceeds as soon as the first branch finishes and the
    /// remaining branches are dropped (which aborts their inner workflows
    /// via `WorkflowHandler`'s `Drop` impl).
    pub fn parallel(
        self: Arc<Self>,
        name: String,
        branch_names: Vec<String>,
        workflows: Vec<Arc<Workflow>>,
        wait_all: bool,
    ) -> BlazenResult<Arc<PipelineBuilder>> {
        if branch_names.len() != workflows.len() {
            return Err(BlazenError::Validation {
                message: format!(
                    "parallel stage '{name}': branch_names ({}) and workflows ({}) length mismatch",
                    branch_names.len(),
                    workflows.len(),
                ),
            });
        }
        let mut state = self.take_state()?;
        let branches: Vec<(String, Arc<Workflow>)> =
            branch_names.into_iter().zip(workflows).collect();
        state.stages.push(StageDef::Parallel {
            name,
            branches,
            wait_all,
        });
        self.replace_state(state);
        Ok(self)
    }

    /// Per-stage timeout in milliseconds. Each stage's workflow gets at
    /// most this long to produce its `StopEvent` before the pipeline
    /// aborts with [`BlazenError::Timeout`].
    pub fn timeout_per_stage_ms(
        self: Arc<Self>,
        millis: u64,
    ) -> BlazenResult<Arc<PipelineBuilder>> {
        let mut state = self.take_state()?;
        state.timeout_per_stage = Some(Duration::from_millis(millis));
        self.replace_state(state);
        Ok(self)
    }

    /// Total wall-clock timeout for the entire pipeline run, in
    /// milliseconds. The pipeline aborts with [`BlazenError::Timeout`] if
    /// it does not finish within this duration.
    pub fn total_timeout_ms(self: Arc<Self>, millis: u64) -> BlazenResult<Arc<PipelineBuilder>> {
        let mut state = self.take_state()?;
        state.total_timeout = Some(Duration::from_millis(millis));
        self.replace_state(state);
        Ok(self)
    }

    /// Validate the pipeline definition and produce a runnable
    /// [`Pipeline`].
    ///
    /// Fails with [`BlazenError::Validation`] if the pipeline has zero
    /// stages or if any stage names are duplicated.
    pub fn build(self: Arc<Self>) -> BlazenResult<Arc<Pipeline>> {
        let state = self.take_state()?;
        if state.stages.is_empty() {
            return Err(BlazenError::Validation {
                message: "pipeline must have at least one stage".into(),
            });
        }
        let mut seen = HashSet::new();
        for s in &state.stages {
            if !seen.insert(s.name().to_owned()) {
                return Err(BlazenError::Validation {
                    message: format!("duplicate stage name: '{}'", s.name()),
                });
            }
        }

        Ok(Arc::new(Pipeline {
            name: state.name,
            stages: state.stages,
            timeout_per_stage: state.timeout_per_stage,
            total_timeout: state.total_timeout,
            running: parking_lot::Mutex::new(false),
        }))
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A validated, runnable pipeline.
///
/// Multiple runs are allowed — invoking [`run`](Self::run) twice in a row
/// is safe and produces independent runs — but the implementation rejects
/// **overlapping** runs on the same handle to avoid surprising aliasing of
/// inner workflow state across concurrent foreign callers.
#[derive(uniffi::Object)]
pub struct Pipeline {
    name: String,
    stages: Vec<StageDef>,
    timeout_per_stage: Option<Duration>,
    total_timeout: Option<Duration>,
    running: parking_lot::Mutex<bool>,
}

impl Pipeline {
    fn lock_running(self: &Arc<Self>) -> BlazenResult<RunGuard> {
        let mut guard = self.running.lock();
        if *guard {
            return Err(BlazenError::Validation {
                message: "Pipeline.run is already in flight on this handle; await it before starting another".into(),
            });
        }
        *guard = true;
        Ok(RunGuard {
            pipeline: Arc::clone(self),
        })
    }
}

/// Drop-guard that clears the `running` flag when a pipeline run ends
/// (including via panic or cancellation), so the same `Pipeline` handle
/// can be re-used for a fresh run afterwards.
struct RunGuard {
    pipeline: Arc<Pipeline>,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        *self.pipeline.running.lock() = false;
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl Pipeline {
    /// Execute the pipeline to completion. `input_json` is parsed as JSON
    /// and passed as the first stage's `StartEvent` payload; each
    /// subsequent stage receives the previous stage's `StopEvent` result.
    ///
    /// Returns a [`WorkflowResult`] whose `event` field is a synthetic
    /// `StopEvent` carrying the final stage output, and whose
    /// `total_*_tokens` / `total_cost_usd` fields are the sum across every
    /// stage's `WorkflowResult`.
    pub async fn run(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let _guard = self.lock_running()?;
        let initial: serde_json::Value = serde_json::from_str(&input_json)?;
        let stages = self.stages.clone();
        let per_stage = self.timeout_per_stage;
        let total = self.total_timeout;
        let pipeline_name = self.name.clone();
        let fut = run_stages(pipeline_name, stages, initial, per_stage);
        match total {
            Some(d) => match tokio::time::timeout(d, fut).await {
                Ok(inner) => inner,
                Err(_) => Err(BlazenError::Timeout {
                    message: "pipeline total timeout exceeded".into(),
                    elapsed_ms: u64::try_from(d.as_millis()).unwrap_or(u64::MAX),
                }),
            },
            None => fut.await,
        }
    }
}

#[uniffi::export]
impl Pipeline {
    /// Synchronous variant of [`run`](Self::run) — blocks the current
    /// thread on the shared Tokio runtime. Provided for callers that want
    /// fire-and-forget usage without engaging their host language's async
    /// machinery (Ruby scripts, simple Go `main` functions).
    pub fn run_blocking(self: Arc<Self>, input_json: String) -> BlazenResult<WorkflowResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.run(input_json).await })
    }

    /// Stage names in registration order — useful for foreign-side
    /// introspection / debug logging without re-running the pipeline.
    #[must_use]
    pub fn stage_names(self: Arc<Self>) -> Vec<String> {
        self.stages.iter().map(|s| s.name().to_owned()).collect()
    }
}

// ---------------------------------------------------------------------------
// Execution engine
// ---------------------------------------------------------------------------

async fn run_stages(
    pipeline_name: String,
    stages: Vec<StageDef>,
    initial_input: serde_json::Value,
    per_stage_timeout: Option<Duration>,
) -> BlazenResult<WorkflowResult> {
    let mut cur_input = initial_input;
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;
    let mut total_cost: f64 = 0.0;
    let mut last_event = Event {
        event_type: "StopEvent".into(),
        data_json: "null".into(),
    };

    for stage in stages {
        let stage_name = stage.name().to_owned();
        let next_input = match stage {
            StageDef::Sequential { workflow, name } => {
                let result = run_one_workflow(workflow, &cur_input, per_stage_timeout, &name)
                    .await
                    .map_err(|e| annotate(&pipeline_name, &stage_name, e))?;
                total_in = total_in.saturating_add(result.total_input_tokens);
                total_out = total_out.saturating_add(result.total_output_tokens);
                total_cost += result.total_cost_usd;
                let next = parse_event_payload(&result.event);
                last_event = result.event;
                next
            }
            StageDef::Parallel {
                branches,
                wait_all,
                name,
            } => {
                let (results_json, in_t, out_t, cost) = run_parallel_branches(
                    branches,
                    cur_input.clone(),
                    per_stage_timeout,
                    wait_all,
                    &name,
                )
                .await
                .map_err(|e| annotate(&pipeline_name, &stage_name, e))?;
                total_in = total_in.saturating_add(in_t);
                total_out = total_out.saturating_add(out_t);
                total_cost += cost;
                last_event = Event {
                    event_type: "StopEvent".into(),
                    data_json: results_json.to_string(),
                };
                results_json
            }
        };
        cur_input = next_input;
    }

    Ok(WorkflowResult {
        event: last_event,
        total_input_tokens: total_in,
        total_output_tokens: total_out,
        total_cost_usd: total_cost,
    })
}

async fn run_one_workflow(
    workflow: Arc<Workflow>,
    input: &serde_json::Value,
    timeout: Option<Duration>,
    stage_label: &str,
) -> BlazenResult<WorkflowResult> {
    let payload = serde_json::to_string(input)?;
    let fut = workflow.run(payload);
    match timeout {
        Some(d) => match tokio::time::timeout(d, fut).await {
            Ok(inner) => inner,
            Err(_) => Err(BlazenError::Timeout {
                message: format!("stage '{stage_label}' exceeded per-stage timeout"),
                elapsed_ms: u64::try_from(d.as_millis()).unwrap_or(u64::MAX),
            }),
        },
        None => fut.await,
    }
}

async fn run_parallel_branches(
    branches: Vec<(String, Arc<Workflow>)>,
    input: serde_json::Value,
    timeout: Option<Duration>,
    wait_all: bool,
    parent_name: &str,
) -> BlazenResult<(serde_json::Value, u64, u64, f64)> {
    let payload = serde_json::to_string(&input)?;
    let mut set: JoinSet<(String, BlazenResult<WorkflowResult>)> = JoinSet::new();
    for (branch_name, workflow) in branches {
        let payload_clone = payload.clone();
        let branch_label = format!("{parent_name}::{branch_name}");
        set.spawn(async move {
            let fut = workflow.run(payload_clone);
            let res = match timeout {
                Some(d) => match tokio::time::timeout(d, fut).await {
                    Ok(inner) => inner,
                    Err(_) => Err(BlazenError::Timeout {
                        message: format!("branch '{branch_label}' exceeded per-stage timeout"),
                        elapsed_ms: u64::try_from(d.as_millis()).unwrap_or(u64::MAX),
                    }),
                },
                None => fut.await,
            };
            (branch_name, res)
        });
    }

    let mut acc = serde_json::Map::new();
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;
    let mut total_cost: f64 = 0.0;

    if wait_all {
        while let Some(joined) = set.join_next().await {
            let (branch_name, res) = joined.map_err(|e| BlazenError::Internal {
                message: format!("branch join failure: {e}"),
            })?;
            let result = res?;
            total_in = total_in.saturating_add(result.total_input_tokens);
            total_out = total_out.saturating_add(result.total_output_tokens);
            total_cost += result.total_cost_usd;
            acc.insert(branch_name, parse_event_payload(&result.event));
        }
    } else if let Some(joined) = set.join_next().await {
        let (branch_name, res) = joined.map_err(|e| BlazenError::Internal {
            message: format!("branch join failure: {e}"),
        })?;
        set.abort_all();
        let result = res?;
        total_in = total_in.saturating_add(result.total_input_tokens);
        total_out = total_out.saturating_add(result.total_output_tokens);
        total_cost += result.total_cost_usd;
        acc.insert(branch_name, parse_event_payload(&result.event));
    }

    Ok((
        serde_json::Value::Object(acc),
        total_in,
        total_out,
        total_cost,
    ))
}

/// Parse a `StopEvent`'s `data_json` into a JSON `Value` so it can flow as
/// the next stage's input. Non-JSON / malformed payloads fall back to
/// `null` rather than failing the whole pipeline — the downstream
/// workflow can decide how to handle that.
fn parse_event_payload(event: &Event) -> serde_json::Value {
    serde_json::from_str(&event.data_json).unwrap_or(serde_json::Value::Null)
}

/// Decorate a stage failure with the pipeline + stage labels so errors
/// surfacing on the foreign side carry enough context to be actionable
/// without forcing callers to add their own wrapping.
fn annotate(pipeline_name: &str, stage_name: &str, err: BlazenError) -> BlazenError {
    match err {
        BlazenError::Timeout {
            message,
            elapsed_ms,
        } => BlazenError::Timeout {
            message: format!("pipeline '{pipeline_name}' stage '{stage_name}': {message}"),
            elapsed_ms,
        },
        BlazenError::Workflow { message } => BlazenError::Workflow {
            message: format!("pipeline '{pipeline_name}' stage '{stage_name}': {message}"),
        },
        BlazenError::Validation { message } => BlazenError::Validation {
            message: format!("pipeline '{pipeline_name}' stage '{stage_name}': {message}"),
        },
        other => other,
    }
}
