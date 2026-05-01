//! Fluent builder for constructing a [`Workflow`].
//!
//! Use [`WorkflowBuilder::new`] to start, chain configuration methods, and
//! call [`WorkflowBuilder::build`] to produce a validated [`Workflow`].

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use blazen_events::{InputRequestEvent, InputResponseEvent};
use blazen_llm::retry::RetryConfig;

use crate::error::WorkflowError;
use crate::step::{ParallelSubWorkflowsStep, StepKind, StepRegistration, SubWorkflowStep};
use crate::workflow::Workflow;

/// Async callback for handling input requests inline (without pausing).
///
/// When registered on a [`WorkflowBuilder`], the event loop will invoke this
/// callback instead of auto-pausing when an [`InputRequestEvent`] arrives.
/// The callback should return an [`InputResponseEvent`] which will be
/// injected back into the event queue.
pub type InputHandlerFn = Arc<
    dyn Fn(
            InputRequestEvent,
        )
            -> Pin<Box<dyn Future<Output = Result<InputResponseEvent, WorkflowError>> + Send>>
        + Send
        + Sync,
>;

/// Fluent builder for constructing a [`Workflow`].
pub struct WorkflowBuilder {
    name: String,
    steps: Vec<StepKind>,
    timeout: Option<Duration>,
    /// Default retry configuration applied to LLM calls inside this
    /// workflow. Step / per-call overrides take precedence; pipeline /
    /// provider defaults take lower precedence.
    retry_config: Option<Arc<RetryConfig>>,
    /// Optional inline handler for input requests (HITL without pausing).
    input_handler: Option<InputHandlerFn>,
    /// Whether to automatically publish lifecycle events to the broadcast stream.
    auto_publish_events: bool,
    /// Policy applied to live session references when the workflow is
    /// paused or snapshotted.
    session_pause_policy: crate::session_ref::SessionPausePolicy,
    /// Checkpoint store for durable persistence (requires `persist` feature).
    #[cfg(feature = "persist")]
    checkpoint_store: Option<Arc<dyn blazen_persist::CheckpointStore>>,
    /// Whether to automatically checkpoint after each step completes.
    #[cfg(feature = "persist")]
    checkpoint_after_step: bool,
    /// Whether to collect an append-only history of workflow events (requires `telemetry` feature).
    #[cfg(feature = "telemetry")]
    collect_history: bool,
}

impl WorkflowBuilder {
    /// Create a new builder with the given workflow name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            timeout: Some(Duration::from_mins(5)),
            retry_config: None,
            input_handler: None,
            auto_publish_events: true,
            session_pause_policy: crate::session_ref::SessionPausePolicy::default(),
            #[cfg(feature = "persist")]
            checkpoint_store: None,
            #[cfg(feature = "persist")]
            checkpoint_after_step: false,
            #[cfg(feature = "telemetry")]
            collect_history: false,
        }
    }

    /// Register a regular handler-backed step.
    #[must_use]
    pub fn step(mut self, registration: StepRegistration) -> Self {
        self.steps.push(StepKind::Regular(registration));
        self
    }

    /// Register a sub-workflow step that runs another [`Workflow`] as
    /// its handler.
    #[must_use]
    pub fn add_subworkflow_step(mut self, step: SubWorkflowStep) -> Self {
        self.steps.push(StepKind::SubWorkflow(step));
        self
    }

    /// Register a parallel sub-workflow fan-out step.
    #[must_use]
    pub fn add_parallel_subworkflows(mut self, step: ParallelSubWorkflowsStep) -> Self {
        self.steps.push(StepKind::ParallelSubWorkflows(step));
        self
    }

    /// Borrow the most recently registered regular step's timeout slot
    /// for mutation. Panics with `caller`'s name if no step is
    /// registered yet, or if the most recent step is not a regular
    /// (handler-backed) step.
    fn last_regular_timeout_mut(&mut self, caller: &str) -> &mut Option<Duration> {
        match self.steps.last_mut() {
            Some(StepKind::Regular(reg)) => &mut reg.timeout,
            Some(StepKind::SubWorkflow(step)) => &mut step.timeout,
            Some(StepKind::ParallelSubWorkflows(_)) => panic!(
                "{caller}() is not supported on a ParallelSubWorkflows step; \
                 set per-branch timeouts on each SubWorkflowStep instead"
            ),
            None => panic!(
                "{caller}() called before any step was registered; \
                 call .step(...) or .add_subworkflow_step(...) first"
            ),
        }
    }

    /// Borrow the most recently registered step's retry-config slot for
    /// mutation. Panics with `caller`'s name if no step is registered
    /// yet, or if the most recent step does not carry a retry slot.
    fn last_retry_slot_mut(&mut self, caller: &str) -> &mut Option<Arc<RetryConfig>> {
        match self.steps.last_mut() {
            Some(StepKind::Regular(reg)) => &mut reg.retry_config,
            Some(StepKind::SubWorkflow(step)) => &mut step.retry_config,
            Some(StepKind::ParallelSubWorkflows(_)) => panic!(
                "{caller}() is not supported on a ParallelSubWorkflows step; \
                 set per-branch retries on each SubWorkflowStep instead"
            ),
            None => panic!(
                "{caller}() called before any step was registered; \
                 call .step(...) or .add_subworkflow_step(...) first"
            ),
        }
    }

    /// Set the timeout on the most recently registered step.
    ///
    /// # Panics
    ///
    /// Panics if no step has been registered yet, or if the most
    /// recently registered step is a `ParallelSubWorkflows` (set
    /// per-branch timeouts instead).
    #[must_use]
    pub fn step_timeout(mut self, timeout: Duration) -> Self {
        *self.last_regular_timeout_mut("step_timeout") = Some(timeout);
        self
    }

    /// Clear the timeout on the most recently registered step (default).
    ///
    /// # Panics
    ///
    /// Panics if no step has been registered yet, or if the most
    /// recently registered step is a `ParallelSubWorkflows`.
    #[must_use]
    pub fn no_step_timeout(mut self) -> Self {
        *self.last_regular_timeout_mut("no_step_timeout") = None;
        self
    }

    /// Set the retry config on the most recently registered step.
    ///
    /// # Panics
    ///
    /// Panics if no step has been registered yet, or if the most
    /// recently registered step is a `ParallelSubWorkflows`.
    #[must_use]
    pub fn step_retry(mut self, config: RetryConfig) -> Self {
        *self.last_retry_slot_mut("step_retry") = Some(Arc::new(config));
        self
    }

    /// Disable retries on the most recently registered step.
    ///
    /// # Panics
    ///
    /// Panics if no step has been registered yet, or if the most
    /// recently registered step is a `ParallelSubWorkflows`.
    #[must_use]
    pub fn no_step_retry(mut self) -> Self {
        *self.last_retry_slot_mut("no_step_retry") = Some(Arc::new(RetryConfig {
            max_retries: 0,
            ..RetryConfig::default()
        }));
        self
    }

    /// Set the maximum execution time for the workflow.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Disable the execution timeout (workflow runs until `StopEvent`).
    #[must_use]
    pub fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Set a default retry configuration for every LLM call inside this
    /// workflow. Step / per-call overrides take precedence; pipeline /
    /// provider defaults take lower precedence.
    #[must_use]
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Disable workflow-level retries (`max_retries = 0`). Step / per-call
    /// overrides still take precedence.
    #[must_use]
    pub fn no_retry(mut self) -> Self {
        self.retry_config = Some(Arc::new(RetryConfig {
            max_retries: 0,
            ..RetryConfig::default()
        }));
        self
    }

    /// Clear any workflow-level retry config.
    #[must_use]
    pub fn clear_retry_config(mut self) -> Self {
        self.retry_config = None;
        self
    }

    /// Register an inline handler for [`InputRequestEvent`]s.
    ///
    /// When set, the event loop will call this handler instead of
    /// auto-pausing when an input request arrives. The handler should
    /// return an [`InputResponseEvent`] which is injected back into the
    /// event queue, allowing the workflow to continue without interruption.
    #[must_use]
    pub fn input_handler(mut self, handler: InputHandlerFn) -> Self {
        self.input_handler = Some(handler);
        self
    }

    /// Enable or disable automatic publishing of lifecycle events to the
    /// broadcast stream.
    ///
    /// When enabled, the event loop will publish `DynamicEvent`s with type
    /// `"blazen::lifecycle"` at key decision points (event routed, step
    /// started, step completed, step failed) and a typed
    /// [`ProgressEvent`](blazen_events::ProgressEvent) with
    /// [`ProgressKind::Workflow`](blazen_events::ProgressKind::Workflow)
    /// after each step completes. Consumers that subscribe via
    /// [`WorkflowHandler::stream_events`](crate::WorkflowHandler::stream_events)
    /// will receive these alongside any events published by steps.
    ///
    /// Defaults to `true` — call this method with `false` to opt out
    /// (e.g. for benchmarks or extremely event-noisy workflows).
    #[must_use]
    pub fn auto_publish_events(mut self, enabled: bool) -> Self {
        self.auto_publish_events = enabled;
        self
    }

    /// Configure the policy applied to live session references when the
    /// workflow is paused or snapshotted. Defaults to `PickleOrError`.
    #[must_use]
    pub fn session_pause_policy(mut self, policy: crate::session_ref::SessionPausePolicy) -> Self {
        self.session_pause_policy = policy;
        self
    }

    /// Enable collection of an append-only history of workflow events.
    ///
    /// When enabled, the event loop records a chronological log of
    /// everything that happens during the workflow run: events received,
    /// steps dispatched, steps completed/failed, pauses, and completion.
    /// The history can be retrieved via
    /// [`WorkflowHandler::collect_history`](crate::WorkflowHandler::collect_history)
    /// after the workflow completes.
    ///
    /// Requires the `telemetry` feature.
    #[cfg(feature = "telemetry")]
    #[must_use]
    pub fn with_history(mut self) -> Self {
        self.collect_history = true;
        self
    }

    /// Set the checkpoint store for durable persistence.
    ///
    /// When a checkpoint store is configured, the workflow can persist its
    /// state to durable storage for crash recovery or migration.
    ///
    /// Requires the `persist` feature.
    #[cfg(feature = "persist")]
    #[must_use]
    pub fn checkpoint_store(mut self, store: Arc<dyn blazen_persist::CheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }

    /// Enable or disable automatic checkpointing after each step completes.
    ///
    /// When enabled (and a checkpoint store is configured), the workflow will
    /// save a checkpoint after each event is dispatched to step handlers.
    /// Checkpointing is best-effort: errors are logged but do not fail the
    /// workflow.
    ///
    /// Defaults to `false`.
    ///
    /// Requires the `persist` feature.
    #[cfg(feature = "persist")]
    #[must_use]
    pub fn checkpoint_after_step(mut self, enabled: bool) -> Self {
        self.checkpoint_after_step = enabled;
        self
    }

    /// Validate and build the workflow.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::ValidationFailed`] if no steps are registered.
    pub fn build(self) -> crate::error::Result<Workflow> {
        if self.steps.is_empty() {
            return Err(WorkflowError::ValidationFailed(
                "workflow must have at least one step".into(),
            ));
        }

        // Build the event-type -> handlers registry.
        let mut registry: HashMap<String, Vec<StepKind>> = HashMap::new();
        for step in self.steps {
            for &event_type in step.accepts() {
                registry
                    .entry(event_type.to_owned())
                    .or_default()
                    .push(step.clone());
            }
        }

        Ok(Workflow {
            name: self.name,
            step_registry: registry,
            timeout: self.timeout,
            retry_config: self.retry_config,
            input_handler: self.input_handler,
            auto_publish_events: self.auto_publish_events,
            session_pause_policy: self.session_pause_policy,
            #[cfg(feature = "persist")]
            checkpoint_store: self.checkpoint_store,
            #[cfg(feature = "persist")]
            checkpoint_after_step: self.checkpoint_after_step,
            #[cfg(feature = "telemetry")]
            collect_history: self.collect_history,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::{StepFn, StepOutput, StepRegistration};
    use blazen_events::{Event, StartEvent, StopEvent};

    fn make_step(name: &str) -> StepRegistration {
        let handler: StepFn = Arc::new(|_event, _ctx| {
            Box::pin(async move {
                Ok(StepOutput::Single(Box::new(StopEvent {
                    result: serde_json::json!(null),
                })))
            })
        });
        StepRegistration::new(
            name.to_owned(),
            vec![StartEvent::event_type()],
            vec![StopEvent::event_type()],
            handler,
            0,
        )
    }

    /// Helper: borrow the `StepRegistration` underlying a builder step,
    /// asserting it is a `Regular` variant (the only one the legacy
    /// tests construct).
    fn as_reg(kind: &crate::step::StepKind) -> &StepRegistration {
        match kind {
            crate::step::StepKind::Regular(reg) => reg,
            other => panic!("expected StepKind::Regular, got {other:?}"),
        }
    }

    #[test]
    fn step_timeout_sets_timeout_on_last_step() {
        let builder = WorkflowBuilder::new("test")
            .step(make_step("a"))
            .step_timeout(Duration::from_millis(100));

        let last = as_reg(builder.steps.last().expect("step registered"));
        assert_eq!(last.timeout, Some(Duration::from_millis(100)));
    }

    #[test]
    fn step_timeout_only_affects_most_recent_step() {
        let builder = WorkflowBuilder::new("test")
            .step(make_step("a"))
            .step_timeout(Duration::from_millis(100))
            .step(make_step("b"))
            .step_timeout(Duration::from_millis(250));

        assert_eq!(builder.steps.len(), 2);
        assert_eq!(
            as_reg(&builder.steps[0]).timeout,
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            as_reg(&builder.steps[1]).timeout,
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn no_step_timeout_clears_timeout_on_last_step() {
        let builder = WorkflowBuilder::new("test")
            .step(make_step("a"))
            .step_timeout(Duration::from_secs(1))
            .no_step_timeout();

        let last = as_reg(builder.steps.last().expect("step registered"));
        assert!(last.timeout.is_none());
    }

    #[test]
    #[should_panic(expected = "step_timeout() called before any step was registered")]
    fn step_timeout_panics_without_step() {
        let _ = WorkflowBuilder::new("test").step_timeout(Duration::from_millis(100));
    }

    #[test]
    #[should_panic(expected = "no_step_timeout() called before any step was registered")]
    fn no_step_timeout_panics_without_step() {
        let _ = WorkflowBuilder::new("test").no_step_timeout();
    }

    #[test]
    fn step_retry_sets_retry_config_on_last_step() {
        let builder = WorkflowBuilder::new("test")
            .step(make_step("a"))
            .step_retry(RetryConfig {
                max_retries: 9,
                ..RetryConfig::default()
            });

        let last = as_reg(builder.steps.last().expect("step registered"));
        assert_eq!(last.retry_config.as_ref().unwrap().max_retries, 9);
    }

    #[test]
    fn step_retry_only_affects_most_recent_step() {
        let builder = WorkflowBuilder::new("test")
            .step(make_step("a"))
            .step_retry(RetryConfig {
                max_retries: 3,
                ..RetryConfig::default()
            })
            .step(make_step("b"))
            .step_retry(RetryConfig {
                max_retries: 5,
                ..RetryConfig::default()
            });

        assert_eq!(builder.steps.len(), 2);
        assert_eq!(
            as_reg(&builder.steps[0])
                .retry_config
                .as_ref()
                .unwrap()
                .max_retries,
            3
        );
        assert_eq!(
            as_reg(&builder.steps[1])
                .retry_config
                .as_ref()
                .unwrap()
                .max_retries,
            5
        );
    }

    #[test]
    fn no_step_retry_sets_zero_retries_on_last_step() {
        let builder = WorkflowBuilder::new("test")
            .step(make_step("a"))
            .no_step_retry();

        let last = as_reg(builder.steps.last().expect("step registered"));
        assert_eq!(last.retry_config.as_ref().unwrap().max_retries, 0);
    }

    #[test]
    #[should_panic(expected = "step_retry() called before any step was registered")]
    fn step_retry_panics_without_step() {
        let _ = WorkflowBuilder::new("test").step_retry(RetryConfig::default());
    }

    #[test]
    #[should_panic(expected = "no_step_retry() called before any step was registered")]
    fn no_step_retry_panics_without_step() {
        let _ = WorkflowBuilder::new("test").no_step_retry();
    }
}
