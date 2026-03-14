//! Fluent builder for constructing a [`Pipeline`].
//!
//! Use [`PipelineBuilder::new`] to start, add stages with [`stage`] and
//! [`parallel`], optionally configure persistence callbacks, then call
//! [`build`] to get a validated [`Pipeline`].
//!
//! [`stage`]: PipelineBuilder::stage
//! [`parallel`]: PipelineBuilder::parallel
//! [`build`]: PipelineBuilder::build

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::error::PipelineError;
use crate::pipeline::Pipeline;
use crate::snapshot::PipelineSnapshot;
use crate::stage::{ParallelStage, Stage, StageKind};

/// Async callback that receives a typed [`PipelineSnapshot`] after each
/// stage completes.
pub type PersistFn = Arc<
    dyn Fn(PipelineSnapshot) -> Pin<Box<dyn Future<Output = Result<(), PipelineError>> + Send>>
        + Send
        + Sync,
>;

/// Async callback that receives the snapshot serialized as a JSON string
/// after each stage completes.
pub type PersistJsonFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<(), PipelineError>> + Send>> + Send + Sync,
>;

/// Fluent builder for constructing a [`Pipeline`].
pub struct PipelineBuilder {
    name: String,
    stages: Vec<StageKind>,
    persist_fn: Option<PersistFn>,
    persist_json_fn: Option<PersistJsonFn>,
    timeout_per_stage: Option<Duration>,
}

impl PipelineBuilder {
    /// Create a new builder with the given pipeline name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            persist_fn: None,
            persist_json_fn: None,
            timeout_per_stage: None,
        }
    }

    /// Add a sequential stage to the pipeline.
    #[must_use]
    pub fn stage(mut self, stage: Stage) -> Self {
        self.stages.push(StageKind::Sequential(stage));
        self
    }

    /// Add a parallel stage (multiple branches) to the pipeline.
    #[must_use]
    pub fn parallel(mut self, parallel: ParallelStage) -> Self {
        self.stages.push(StageKind::Parallel(parallel));
        self
    }

    /// Set a persist callback that receives a typed [`PipelineSnapshot`]
    /// after each stage completes.
    #[must_use]
    pub fn on_persist(mut self, f: PersistFn) -> Self {
        self.persist_fn = Some(f);
        self
    }

    /// Set a persist callback that receives the snapshot as a JSON string
    /// after each stage completes.
    #[must_use]
    pub fn on_persist_json(mut self, f: PersistJsonFn) -> Self {
        self.persist_json_fn = Some(f);
        self
    }

    /// Set a per-stage timeout. Each stage's workflow will be given this
    /// duration before being considered timed out.
    #[must_use]
    pub fn timeout_per_stage(mut self, timeout: Duration) -> Self {
        self.timeout_per_stage = Some(timeout);
        self
    }

    /// Validate and build the pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationFailed`] if the pipeline has no
    /// stages or if any stage names are duplicated.
    pub fn build(self) -> Result<Pipeline, PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::ValidationFailed(
                "pipeline must have at least one stage".into(),
            ));
        }

        // Check for duplicate stage names.
        let mut seen = std::collections::HashSet::new();
        for stage in &self.stages {
            let name = stage.name();
            if !seen.insert(name.to_owned()) {
                return Err(PipelineError::ValidationFailed(format!(
                    "duplicate stage name: '{name}'"
                )));
            }
        }

        Ok(Pipeline {
            name: self.name,
            stages: self.stages,
            persist_fn: self.persist_fn,
            persist_json_fn: self.persist_json_fn,
            timeout_per_stage: self.timeout_per_stage,
        })
    }
}
