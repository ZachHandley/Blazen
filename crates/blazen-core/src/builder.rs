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

use crate::error::WorkflowError;
use crate::step::StepRegistration;
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
    steps: Vec<StepRegistration>,
    timeout: Option<Duration>,
    /// Optional inline handler for input requests (HITL without pausing).
    input_handler: Option<InputHandlerFn>,
    /// Whether to automatically publish lifecycle events to the broadcast stream.
    auto_publish_events: bool,
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
            timeout: Some(Duration::from_secs(300)), // 5 min default
            input_handler: None,
            auto_publish_events: false,
            #[cfg(feature = "persist")]
            checkpoint_store: None,
            #[cfg(feature = "persist")]
            checkpoint_after_step: false,
            #[cfg(feature = "telemetry")]
            collect_history: false,
        }
    }

    /// Register a step.
    #[must_use]
    pub fn step(mut self, registration: StepRegistration) -> Self {
        self.steps.push(registration);
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
    /// started, step completed, step failed). Consumers that subscribe via
    /// [`WorkflowHandler::stream_events`](crate::WorkflowHandler::stream_events)
    /// will receive these alongside any events published by steps.
    ///
    /// Defaults to `false`.
    #[must_use]
    pub fn auto_publish_events(mut self, enabled: bool) -> Self {
        self.auto_publish_events = enabled;
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
        let mut registry: HashMap<String, Vec<StepRegistration>> = HashMap::new();
        for step in self.steps {
            for &event_type in &step.accepts {
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
            input_handler: self.input_handler,
            auto_publish_events: self.auto_publish_events,
            #[cfg(feature = "persist")]
            checkpoint_store: self.checkpoint_store,
            #[cfg(feature = "persist")]
            checkpoint_after_step: self.checkpoint_after_step,
            #[cfg(feature = "telemetry")]
            collect_history: self.collect_history,
        })
    }
}
