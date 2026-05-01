//! Stage definitions for the pipeline orchestrator.
//!
//! A pipeline is composed of stages, each wrapping a [`Workflow`]. Stages
//! can be sequential (one workflow) or parallel (multiple branches running
//! concurrently with a configurable join strategy).

use std::sync::Arc;

use blazen_core::Workflow;
use serde_json::Value;

use crate::state::PipelineState;

/// A function that maps pipeline state into the JSON input for a stage's
/// workflow.
pub type InputMapperFn<S = Value> = Arc<dyn Fn(&PipelineState<S>) -> Value + Send + Sync>;

/// A function that decides whether a stage should execute.
pub type ConditionFn<S = Value> = Arc<dyn Fn(&PipelineState<S>) -> bool + Send + Sync>;

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

/// A stage in the pipeline -- either sequential or parallel.
#[derive(Debug)]
pub enum StageKind<S = Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    /// A single sequential stage.
    Sequential(Stage<S>),
    /// A parallel stage with multiple branches.
    Parallel(ParallelStage<S>),
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
        }
    }
}
