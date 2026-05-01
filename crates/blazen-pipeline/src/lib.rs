//! # `Blazen` Pipeline
//!
//! A multi-workflow pipeline orchestrator for the `Blazen` framework.
//!
//! This crate adds a higher-level [`Pipeline`] abstraction that orchestrates
//! multiple [`Workflow`](blazen_core::Workflow)s as sequential or parallel
//! stages. Each stage wraps a workflow and can optionally transform its
//! input, conditionally execute, and contribute to shared pipeline state.
//!
//! # Architecture
//!
//! A pipeline is an ordered list of stages. Each stage is either:
//!
//! - **Sequential** -- a single workflow runs, receives the previous stage's
//!   output (or a custom-mapped input), and produces a result.
//! - **Parallel** -- multiple branches run concurrently with a configurable
//!   join strategy ([`WaitAll`](JoinStrategy::WaitAll) or
//!   [`FirstCompletes`](JoinStrategy::FirstCompletes)).
//!
//! Between stages, the pipeline checks for pause signals and calls
//! optional persistence callbacks with a serializable [`PipelineSnapshot`].

pub mod builder;
pub mod error;
pub mod handler;
pub mod pipeline;
pub mod snapshot;
pub mod stage;
pub mod state;

pub use builder::{PersistFn, PersistJsonFn, PipelineBuilder};
pub use error::PipelineError;
pub use handler::{PipelineEvent, PipelineHandler};
pub use pipeline::{Pipeline, ProgressSnapshot};
pub use snapshot::{ActiveWorkflowSnapshot, PipelineResult, PipelineSnapshot, StageResult};
pub use stage::{ConditionFn, InputMapperFn, JoinStrategy, ParallelStage, Stage, StageKind};
pub use state::PipelineState;
