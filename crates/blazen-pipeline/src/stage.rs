//! Stage definitions for the pipeline orchestrator.
//!
//! A pipeline is composed of stages, each wrapping a [`Workflow`]. Stages
//! can be sequential (one workflow) or parallel (multiple branches running
//! concurrently with a configurable join strategy).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use blazen_core::Workflow;
use serde_json::Value;

use crate::error::PipelineError;
use crate::state::PipelineState;

/// A function that maps pipeline state into the JSON input for a stage's
/// workflow.
pub type InputMapperFn<S = Value> = Arc<dyn Fn(&PipelineState<S>) -> Value + Send + Sync>;

/// A function that decides whether a stage should execute.
pub type ConditionFn<S = Value> = Arc<dyn Fn(&PipelineState<S>) -> bool + Send + Sync>;

/// A function that maps a stage's workflow output back into the typed
/// pipeline state. Invoked after the stage's output has been recorded in the
/// shared-state stage-result map, giving downstream stages a typed view.
pub type OutputMapperFn<S = Value> =
    Arc<dyn Fn(&mut PipelineState<S>, Value) -> Result<(), PipelineError> + Send + Sync>;

/// A single sequential stage in the pipeline.
///
/// Wraps a [`Workflow`] with optional input mapping and conditional
/// execution logic.
pub struct Stage<S = Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Human-readable name for this stage (used in results and logging).
    pub name: String,
    /// The workflow to execute for this stage.
    pub workflow: Workflow,
    /// Optional function that transforms pipeline state into the workflow
    /// input. When `None`, the previous stage's output (or the pipeline
    /// input for the first stage) is passed through directly.
    pub input_mapper: Option<InputMapperFn<S>>,
    /// Optional predicate that determines whether this stage should execute.
    /// When `None` the stage always runs. When the predicate returns `false`,
    /// the stage is skipped and its result is marked accordingly.
    pub condition: Option<ConditionFn<S>>,
    /// Optional function invoked after the stage's workflow has produced its
    /// output and that output has been recorded into the shared-state stage
    /// result map. Receives the mutable [`PipelineState`] and the stage's
    /// output JSON, and may project the output into the typed `S` state.
    pub output_mapper: Option<OutputMapperFn<S>>,
}

impl<S> Stage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Wire this stage into the typed shared state `S` via a pair of
    /// projection closures.
    ///
    /// `extract` reads from `PipelineState<S>` and produces the typed input
    /// `I` that will be serialized into the workflow's JSON input
    /// (replacing any pre-existing [`input_mapper`](Self::input_mapper)). If
    /// serialization fails the input falls back to [`Value::Null`].
    ///
    /// `store` receives the deserialized output `O` (decoded from the
    /// workflow's stop value) along with a mutable reference to the pipeline
    /// state, so it can mutate `S` to make the result visible to downstream
    /// stages (replacing any pre-existing
    /// [`output_mapper`](Self::output_mapper)).
    ///
    /// Deserialization failures surface as
    /// [`PipelineError::Serialization`] and abort the pipeline.
    #[must_use]
    pub fn with_typed_state<I, O>(
        mut self,
        extract: impl Fn(&PipelineState<S>) -> I + Send + Sync + 'static,
        store: impl Fn(&mut PipelineState<S>, O) + Send + Sync + 'static,
    ) -> Self
    where
        I: serde::Serialize + 'static,
        O: serde::de::DeserializeOwned + 'static,
    {
        self.input_mapper = Some(Arc::new(move |state| {
            serde_json::to_value(extract(state)).unwrap_or(Value::Null)
        }));
        self.output_mapper = Some(Arc::new(move |state, stop_value| {
            let typed: O = serde_json::from_value(stop_value)?;
            store(state, typed);
            Ok(())
        }));
        self
    }
}

impl<S> Clone for Stage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            workflow: self.workflow.clone(),
            input_mapper: self.input_mapper.clone(),
            condition: self.condition.clone(),
            output_mapper: self.output_mapper.clone(),
        }
    }
}

impl<S> std::fmt::Debug for Stage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage")
            .field("name", &self.name)
            .field("has_input_mapper", &self.input_mapper.is_some())
            .field("has_condition", &self.condition.is_some())
            .field("has_output_mapper", &self.output_mapper.is_some())
            .finish_non_exhaustive()
    }
}

/// How to join the results of parallel branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinStrategy {
    /// Wait for all branches to complete and collect all results.
    WaitAll,
    /// Return as soon as the first branch completes; cancel the rest.
    FirstCompletes,
}

/// A parallel stage that runs multiple branches concurrently.
///
/// Each branch is a [`Stage`]; all branches are spawned simultaneously.
/// The [`JoinStrategy`] determines how results are collected.
pub struct ParallelStage<S = Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Human-readable name for this parallel group.
    pub name: String,
    /// The branches to run concurrently.
    pub branches: Vec<Stage<S>>,
    /// How to join the branch results.
    pub join_strategy: JoinStrategy,
}

impl<S> Clone for ParallelStage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            branches: self.branches.clone(),
            join_strategy: self.join_strategy,
        }
    }
}

impl<S> std::fmt::Debug for ParallelStage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelStage")
            .field("name", &self.name)
            .field("branch_count", &self.branches.len())
            .field("join_strategy", &self.join_strategy)
            .finish_non_exhaustive()
    }
}

/// A function invoked after each round of a [`LoopStage`] completes.
///
/// Receives the just-completed iteration count (0-based) and a reference to
/// the current pipeline state, returning a future that runs to completion
/// before the next round (or loop exit) is decided.
pub type RoundCompleteFn<S = Value> =
    Arc<dyn Fn(u32, &PipelineState<S>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// A function that decides whether a [`LoopStage`] should keep iterating.
///
/// Receives the current pipeline state and the number of iterations completed
/// so far (1-based after the just-finished round), and returns a
/// [`LoopDecision`].
pub type LoopUntilFn<S = Value> = Arc<dyn Fn(&PipelineState<S>, u32) -> LoopDecision + Send + Sync>;

/// The decision returned by a [`LoopStage`]'s `until` predicate after each
/// round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopDecision {
    /// Run the inner stage again (subject to the `max_iterations` cap).
    Continue,
    /// Stop looping cleanly; the loop stage succeeds.
    Done,
    /// Stop looping with an error carrying the given reason.
    Abort(String),
}

/// A stage that repeatedly runs an inner stage until a predicate signals
/// completion (or a maximum iteration count is reached).
///
/// The inner stage may be [`StageKind::Sequential`] or
/// [`StageKind::Parallel`]; nesting another [`StageKind::Loop`] is rejected at
/// execution time.
pub struct LoopStage<S = Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Human-readable name for this loop stage (used in results and logging).
    pub name: String,
    /// Hard cap on the number of rounds. The loop stops once this many rounds
    /// have run even if `until` never returned [`LoopDecision::Done`].
    pub max_iterations: u32,
    /// The inner stage to run each round.
    pub inner: Box<StageKind<S>>,
    /// Predicate evaluated after each round to decide whether to continue,
    /// finish, or abort.
    pub until: LoopUntilFn<S>,
    /// Optional async hook invoked after each round (before `until`).
    pub on_round_complete: Option<RoundCompleteFn<S>>,
}

impl<S> Clone for LoopStage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            max_iterations: self.max_iterations,
            inner: self.inner.clone(),
            until: self.until.clone(),
            on_round_complete: self.on_round_complete.clone(),
        }
    }
}

impl<S> std::fmt::Debug for LoopStage<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopStage")
            .field("name", &self.name)
            .field("max_iterations", &self.max_iterations)
            .field("inner_kind", &self.inner.name())
            .field("has_on_round_complete", &self.on_round_complete.is_some())
            .finish_non_exhaustive()
    }
}

/// A stage in the pipeline -- sequential, parallel, or a loop.
#[derive(Debug)]
pub enum StageKind<S = Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// A single sequential stage.
    Sequential(Stage<S>),
    /// A parallel stage with multiple branches.
    Parallel(ParallelStage<S>),
    /// A loop stage that re-runs an inner stage until a predicate stops it.
    Loop(LoopStage<S>),
}

impl<S> Clone for StageKind<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        match self {
            StageKind::Sequential(s) => StageKind::Sequential(s.clone()),
            StageKind::Parallel(p) => StageKind::Parallel(p.clone()),
            StageKind::Loop(l) => StageKind::Loop(l.clone()),
        }
    }
}

impl<S> StageKind<S>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// Returns the name of this stage.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            StageKind::Sequential(s) => &s.name,
            StageKind::Parallel(p) => &p.name,
            StageKind::Loop(l) => &l.name,
        }
    }
}
