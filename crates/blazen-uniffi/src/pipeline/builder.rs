//! [`PipelineBuilder`] -- fluent builder over the real
//! [`blazen_pipeline::PipelineBuilder`].

use std::sync::Arc;
use std::time::Duration;

use crate::errors::{BlazenError, BlazenResult};
use crate::pipeline::pipeline::Pipeline;
use crate::workflow::Workflow;

use blazen_pipeline::{JoinStrategy, ParallelStage, PipelineBuilder as CorePipelineBuilder, Stage};

/// Mutable builder state, swapped out of the `Mutex<Option<...>>` on each
/// fluent call so the real (move-consuming) `blazen_pipeline::PipelineBuilder`
/// can be threaded through.
struct BuilderState {
    inner: CorePipelineBuilder,
    /// Stage names in registration order, mirrored so the built [`Pipeline`]
    /// can answer `stage_names()` without the core engine exposing them.
    stage_names: Vec<String>,
    /// Auto-incremented index used to mint stage names for
    /// [`PipelineBuilder::add_workflow`] calls that don't supply one.
    next_auto_id: u32,
}

/// Builder for a [`Pipeline`].
///
/// Use [`PipelineBuilder::new`] to start, attach workflows via
/// [`add_workflow`](Self::add_workflow) / [`stage`](Self::stage) /
/// [`parallel`](Self::parallel), then call [`build`](Self::build) to validate
/// and produce a runnable [`Pipeline`].
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

    /// Build an owned core [`Stage`] from a uniffi [`Workflow`]. The inner
    /// `Arc<blazen_core::Workflow>` is cloned (cheap, Arc-backed) into an owned
    /// `blazen_core::Workflow`, with all stage mappers left as `None`.
    fn make_stage(name: String, workflow: &Arc<Workflow>) -> Stage {
        Stage {
            name,
            workflow: workflow.core_workflow(),
            input_mapper: None,
            condition: None,
            output_mapper: None,
        }
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
                inner: CorePipelineBuilder::new(name),
                stage_names: Vec::new(),
                next_auto_id: 0,
            })),
        })
    }

    /// Append a sequential workflow stage with an auto-generated stage name
    /// of the form `"stage-{N}"` (zero-based).
    ///
    /// Use [`stage`](Self::stage) when the stage name matters for downstream
    /// tooling that filters by it.
    pub fn add_workflow(self: Arc<Self>, workflow: Arc<Workflow>) -> BlazenResult<Arc<Self>> {
        let mut state = self.take_state()?;
        let name = format!("stage-{}", state.next_auto_id);
        state.next_auto_id += 1;
        let stage = Self::make_stage(name.clone(), &workflow);
        state.inner = state.inner.stage(stage);
        state.stage_names.push(name);
        self.replace_state(state);
        Ok(self)
    }

    /// Append a sequential stage with an explicit name. The stage name must be
    /// unique within the pipeline (enforced at [`build`](Self::build)).
    pub fn stage(
        self: Arc<Self>,
        name: String,
        workflow: Arc<Workflow>,
    ) -> BlazenResult<Arc<Self>> {
        let mut state = self.take_state()?;
        let stage = Self::make_stage(name.clone(), &workflow);
        state.inner = state.inner.stage(stage);
        state.stage_names.push(name);
        self.replace_state(state);
        Ok(self)
    }

    /// Append a parallel stage running multiple workflows concurrently.
    ///
    /// `branch_names` and `workflows` are positionally paired; a length
    /// mismatch yields [`BlazenError::Validation`]. When `wait_all` is `true`
    /// every branch must complete and outputs are collected into a JSON object
    /// keyed by branch name. When `wait_all` is `false` the pipeline proceeds
    /// as soon as the first branch finishes and the remaining branches are
    /// cancelled.
    pub fn parallel(
        self: Arc<Self>,
        name: String,
        branch_names: Vec<String>,
        workflows: Vec<Arc<Workflow>>,
        wait_all: bool,
    ) -> BlazenResult<Arc<Self>> {
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
        let branches: Vec<Stage> = branch_names
            .into_iter()
            .zip(workflows.iter())
            .map(|(branch_name, wf)| Self::make_stage(branch_name, wf))
            .collect();
        let parallel = ParallelStage {
            name: name.clone(),
            branches,
            join_strategy: if wait_all {
                JoinStrategy::WaitAll
            } else {
                JoinStrategy::FirstCompletes
            },
        };
        state.inner = state.inner.parallel(parallel);
        state.stage_names.push(name);
        self.replace_state(state);
        Ok(self)
    }

    /// Per-stage timeout in milliseconds. Each stage's workflow gets at most
    /// this long to produce its `StopEvent` before the pipeline aborts with
    /// [`BlazenError::Timeout`].
    pub fn timeout_per_stage_ms(self: Arc<Self>, millis: u64) -> BlazenResult<Arc<Self>> {
        let mut state = self.take_state()?;
        state.inner = state.inner.timeout_per_stage(Duration::from_millis(millis));
        self.replace_state(state);
        Ok(self)
    }

    /// Total wall-clock timeout for the entire pipeline run, in milliseconds.
    /// The pipeline aborts with [`BlazenError::Timeout`] if it does not finish
    /// within this duration.
    pub fn total_timeout_ms(self: Arc<Self>, millis: u64) -> BlazenResult<Arc<Self>> {
        let mut state = self.take_state()?;
        state.inner = state.inner.total_timeout(Duration::from_millis(millis));
        self.replace_state(state);
        Ok(self)
    }

    /// Validate the pipeline definition and produce a runnable [`Pipeline`].
    ///
    /// Fails with [`BlazenError::Validation`] if the pipeline has zero stages
    /// or if any stage names are duplicated.
    pub fn build(self: Arc<Self>) -> BlazenResult<Arc<Pipeline>> {
        let state = self.take_state()?;
        let core = state.inner.build().map_err(BlazenError::from)?;
        Ok(Pipeline::new(core, state.stage_names))
    }
}
