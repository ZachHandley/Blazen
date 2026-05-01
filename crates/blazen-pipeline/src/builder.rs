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

use blazen_llm::retry::RetryConfig;

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
pub struct PipelineBuilder<S = serde_json::Value>
where
    S: Default + Clone + Send + Sync + 'static,
{
    name: String,
    stages: Vec<StageKind<S>>,
    persist_fn: Option<PersistFn>,
    persist_json_fn: Option<PersistJsonFn>,
    timeout_per_stage: Option<Duration>,
    total_timeout: Option<Duration>,
    retry_config: Option<Arc<RetryConfig>>,
}

impl PipelineBuilder<serde_json::Value> {
    /// Create a new builder with the JSON-backed default state.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self::new_typed(name)
    }
}

impl<S: Default + Clone + Send + Sync + 'static> PipelineBuilder<S> {
    /// Create a new builder for a typed shared state `S`.
    ///
    /// Use this when you want `PipelineState<S>` instead of the default
    /// `PipelineState<serde_json::Value>`.
    #[must_use]
    pub fn new_typed(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            persist_fn: None,
            persist_json_fn: None,
            timeout_per_stage: None,
            total_timeout: None,
            retry_config: None,
        }
    }

    /// Add a sequential stage to the pipeline.
    #[must_use]
    pub fn stage(mut self, stage: Stage<S>) -> Self {
        self.stages.push(StageKind::Sequential(stage));
        self
    }

    /// Add a parallel stage (multiple branches) to the pipeline.
    #[must_use]
    pub fn parallel(mut self, parallel: ParallelStage<S>) -> Self {
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

    /// Set a total wall-clock timeout for the entire pipeline run.
    ///
    /// When set, the pipeline run-loop is cancelled and a
    /// [`PipelineError::Timeout`] is emitted if the pipeline does not finish
    /// within this duration. Default is `None` (unlimited).
    #[must_use]
    pub fn total_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Disable the total-pipeline timeout (default).
    #[must_use]
    pub fn no_total_timeout(mut self) -> Self {
        self.total_timeout = None;
        self
    }

    /// Set a default retry configuration for every LLM call inside this
    /// pipeline. Workflow / step / per-call overrides take precedence.
    /// Default is `None` (use the provider's own retry config or the
    /// blazen-llm default).
    #[must_use]
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Disable pipeline-level retries for all LLM calls (`max_retries = 0`).
    /// Workflow / step / per-call overrides still take precedence.
    #[must_use]
    pub fn no_retry(mut self) -> Self {
        self.retry_config = Some(Arc::new(RetryConfig {
            max_retries: 0,
            ..RetryConfig::default()
        }));
        self
    }

    /// Clear any pipeline-level retry config; LLM calls fall through to
    /// workflow / step / provider defaults.
    #[must_use]
    pub fn clear_retry_config(mut self) -> Self {
        self.retry_config = None;
        self
    }

    /// Validate and build the pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationFailed`] if the pipeline has no
    /// stages or if any stage names are duplicated.
    pub fn build(self) -> Result<Pipeline<S>, PipelineError> {
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
            total_timeout: self.total_timeout,
            retry_config: self.retry_config,
        })
    }
}

impl PipelineBuilder<serde_json::Value> {
    /// Pivot the builder to use a typed shared state `S2`.
    ///
    /// Carries over name, persist callbacks, timeouts, and `retry_config`.
    /// Stages registered before `with_state` are dropped (they were typed
    /// against the previous state shape) — register stages after this call.
    #[must_use]
    pub fn with_state<S2: Default + Clone + Send + Sync + 'static>(self) -> PipelineBuilder<S2> {
        PipelineBuilder {
            name: self.name,
            stages: Vec::new(),
            persist_fn: self.persist_fn,
            persist_json_fn: self.persist_json_fn,
            timeout_per_stage: self.timeout_per_stage,
            total_timeout: self.total_timeout,
            retry_config: self.retry_config,
        }
    }
}
