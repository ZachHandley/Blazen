//! Step definitions for the workflow engine.
//!
//! A *step* is a named async function that accepts a type-erased event and
//! a [`Context`], returning a [`StepOutput`] that the event loop routes to
//! downstream steps.
//!
//! Steps are registered via [`StepRegistration`] which carries metadata about
//! which event types the step accepts and emits, plus an optional concurrency
//! limit.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use blazen_events::AnyEvent;

use crate::context::Context;
use crate::error::WorkflowError;

/// The result of a step execution.
#[derive(Debug)]
pub enum StepOutput {
    /// A single event to route to downstream steps.
    Single(Box<dyn AnyEvent>),
    /// Multiple events to route (fan-out).
    Multiple(Vec<Box<dyn AnyEvent>>),
    /// No output -- the step performed a side-effect only.
    None,
}

/// Type-erased async step function.
///
/// Wrapped in [`Arc`] so it can be cloned across concurrent invocations
/// within the event loop.
pub type StepFn = Arc<
    dyn Fn(
            Box<dyn AnyEvent>,
            Context,
        ) -> Pin<Box<dyn Future<Output = Result<StepOutput, WorkflowError>> + Send>>
        + Send
        + Sync,
>;

/// Metadata about a registered step, including its handler function.
#[derive(Clone)]
pub struct StepRegistration {
    /// Human-readable name for this step (used in logging and error messages).
    pub name: String,
    /// Event type identifiers this step accepts (matches
    /// [`Event::event_type()`](blazen_events::Event::event_type)).
    pub accepts: Vec<&'static str>,
    /// Event type identifiers this step may emit (informational).
    pub emits: Vec<&'static str>,
    /// The async handler function.
    pub handler: StepFn,
    /// Maximum number of concurrent invocations of this step (0 = unlimited).
    pub max_concurrency: usize,
    /// Semaphore that enforces [`max_concurrency`](Self::max_concurrency).
    /// `None` when `max_concurrency` is `0` (unlimited).
    pub semaphore: Option<Arc<Semaphore>>,
    /// Per-step wall-clock timeout. When `Some(d)`, the workflow event-loop
    /// wraps each invocation of this step's handler in
    /// [`tokio::time::timeout`]. `None` means unlimited (default).
    pub timeout: Option<Duration>,
    /// Per-step retry configuration applied at LLM-call time. When
    /// `Some(cfg)`, this overrides workflow / pipeline / provider defaults
    /// for retries within this step's handler. `None` means inherit from
    /// the next-outer scope (default).
    pub retry_config: Option<Arc<blazen_llm::retry::RetryConfig>>,
}

impl std::fmt::Debug for StepRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepRegistration")
            .field("name", &self.name)
            .field("accepts", &self.accepts)
            .field("emits", &self.emits)
            .field("max_concurrency", &self.max_concurrency)
            .field("timeout", &self.timeout)
            .field("retry_config", &self.retry_config.is_some())
            .finish_non_exhaustive()
    }
}

impl StepRegistration {
    /// Create a new step registration with an optional concurrency semaphore.
    ///
    /// When `max_concurrency` is `0`, the semaphore is `None` (unlimited).
    /// When positive, a [`Semaphore`] with that many permits is created.
    #[must_use]
    pub fn new(
        name: String,
        accepts: Vec<&'static str>,
        emits: Vec<&'static str>,
        handler: StepFn,
        max_concurrency: usize,
    ) -> Self {
        let semaphore = if max_concurrency > 0 {
            Some(Arc::new(Semaphore::new(max_concurrency)))
        } else {
            None
        };
        Self {
            name,
            accepts,
            emits,
            handler,
            max_concurrency,
            semaphore,
            timeout: None,
            retry_config: None,
        }
    }

    /// Set a per-step wall-clock timeout. When `Some(d)`, the workflow
    /// event-loop wraps each invocation of this step's handler in
    /// `tokio::time::timeout(d, ...)`. `None` means unlimited (default).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Clear any per-step timeout (default).
    #[must_use]
    pub fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Set a per-step retry configuration. Overrides workflow / pipeline /
    /// provider defaults at LLM-call time.
    #[must_use]
    pub fn with_retry_config(mut self, config: blazen_llm::retry::RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Disable retries for this step (`max_retries = 0`).
    #[must_use]
    pub fn no_retry(mut self) -> Self {
        self.retry_config = Some(Arc::new(blazen_llm::retry::RetryConfig {
            max_retries: 0,
            ..blazen_llm::retry::RetryConfig::default()
        }));
        self
    }

    /// Clear any per-step retry config.
    #[must_use]
    pub fn clear_retry_config(mut self) -> Self {
        self.retry_config = None;
        self
    }
}

// ---------------------------------------------------------------------------
// Sub-workflow step kinds
// ---------------------------------------------------------------------------

/// How to join the results of multiple parallel sub-workflow branches.
///
/// Mirrors `blazen_pipeline::stage::JoinStrategy` but lives in `blazen-core`
/// so workflow code can fan out child workflows without taking a dependency
/// on the pipeline crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinStrategy {
    /// Wait for every branch to complete and collect all outputs.
    WaitAll,
    /// Resolve as soon as the first branch completes; abort the rest.
    FirstCompletes,
}

/// Maps a parent event into the JSON input passed to the child workflow.
pub type SubWorkflowInputMapper = Arc<dyn Fn(&dyn AnyEvent) -> serde_json::Value + Send + Sync>;

/// Maps the child workflow's terminal `StopEvent.result` JSON back into a
/// concrete event that the parent step emits.
pub type SubWorkflowOutputMapper =
    Arc<dyn Fn(serde_json::Value) -> Box<dyn AnyEvent> + Send + Sync>;

/// A workflow step that delegates to another [`Workflow`](crate::Workflow).
///
/// The parent workflow's event loop spawns the child via
/// [`Workflow::run`](crate::Workflow::run), forwards the result through the
/// step's [`output_mapper`](Self::output_mapper), and treats child failures
/// as [`WorkflowError::SubWorkflowFailed`]. Per-step `timeout` and
/// `retry_config` apply to the child run as a whole.
#[derive(Clone)]
pub struct SubWorkflowStep {
    /// Human-readable name for this step (used in logging and errors).
    pub name: String,
    /// Event type identifiers this step accepts.
    pub accepts: Vec<&'static str>,
    /// Event type identifiers this step may emit (informational).
    pub emits: Vec<&'static str>,
    /// The child workflow to run. Wrapped in [`Arc`] so the registration
    /// (and the parent registry's `Vec<StepKind>` clones) stays cheap to
    /// clone across dispatch attempts and retries.
    pub workflow: Arc<crate::workflow::Workflow>,
    /// Maps the parent event into the child workflow's input JSON.
    pub input_mapper: SubWorkflowInputMapper,
    /// Maps the child's terminal result JSON into an event for the parent.
    pub output_mapper: SubWorkflowOutputMapper,
    /// Per-step wall-clock timeout for the entire child run. `None` means
    /// inherit the child workflow's own timeout (default).
    pub timeout: Option<Duration>,
    /// Per-step retry configuration. When `Some(cfg)`, the child run is
    /// retried up to `cfg.max_retries` times on failure with the
    /// configured backoff.
    pub retry_config: Option<Arc<blazen_llm::retry::RetryConfig>>,
}

impl std::fmt::Debug for SubWorkflowStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubWorkflowStep")
            .field("name", &self.name)
            .field("accepts", &self.accepts)
            .field("emits", &self.emits)
            .field("workflow", &self.workflow)
            .field("timeout", &self.timeout)
            .field("retry_config", &self.retry_config.is_some())
            .finish_non_exhaustive()
    }
}

impl SubWorkflowStep {
    /// Create a sub-workflow step with default JSON-passthrough mappers.
    ///
    /// The input mapper passes the parent event's JSON form
    /// (`event.to_json()`) into the child workflow as input.
    /// The output mapper wraps the child's `WorkflowResult.final_output`
    /// (a `serde_json::Value`) in a `DynamicEvent` whose `event_type` is
    /// `"<step_name>::output"`.
    ///
    /// Use this when you don't need custom event-routing logic -- most
    /// language-binding callers (Python/Node/WASM) will use this entry.
    #[must_use]
    pub fn with_json_mappers(
        name: impl Into<String>,
        accepts: Vec<&'static str>,
        emits: Vec<&'static str>,
        workflow: std::sync::Arc<crate::workflow::Workflow>,
    ) -> Self {
        let name_str = name.into();
        let output_event_type: &'static str =
            blazen_events::intern_event_type(&format!("{name_str}::output"));
        let output_event_type_owned = output_event_type;
        Self {
            name: name_str,
            accepts,
            emits,
            workflow,
            input_mapper: std::sync::Arc::new(|event| event.to_json()),
            output_mapper: std::sync::Arc::new(move |value| {
                Box::new(blazen_events::DynamicEvent {
                    event_type: output_event_type_owned.to_string(),
                    data: value,
                })
            }),
            timeout: None,
            retry_config: None,
        }
    }

    /// Set a per-step wall-clock timeout for the entire child run.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set a per-step retry configuration applied to the child run as a whole.
    #[must_use]
    pub fn with_retry_config(mut self, cfg: blazen_llm::retry::RetryConfig) -> Self {
        self.retry_config = Some(std::sync::Arc::new(cfg));
        self
    }
}

/// Fan out into multiple parallel sub-workflow branches.
///
/// Each branch is a [`SubWorkflowStep`] that runs concurrently. The
/// [`JoinStrategy`] controls whether the parent step waits for all branches
/// or only the first to complete.
#[derive(Clone)]
pub struct ParallelSubWorkflowsStep {
    /// Human-readable name for this fan-out step.
    pub name: String,
    /// Event type identifiers this step accepts.
    pub accepts: Vec<&'static str>,
    /// Event type identifiers this step may emit (informational).
    pub emits: Vec<&'static str>,
    /// The branches to run concurrently. Per-branch `timeout` and
    /// `retry_config` are honored individually.
    pub branches: Vec<SubWorkflowStep>,
    /// How to join the branch results.
    pub join_strategy: JoinStrategy,
}

impl std::fmt::Debug for ParallelSubWorkflowsStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelSubWorkflowsStep")
            .field("name", &self.name)
            .field("accepts", &self.accepts)
            .field("emits", &self.emits)
            .field("branch_count", &self.branches.len())
            .field("join_strategy", &self.join_strategy)
            .finish_non_exhaustive()
    }
}

/// One entry in a workflow's step registry.
///
/// Most steps are [`StepKind::Regular`] — a named async handler. The other
/// variants spawn child workflows ([`StepKind::SubWorkflow`]) or fan out
/// into parallel branches ([`StepKind::ParallelSubWorkflows`]).
#[derive(Clone, Debug)]
pub enum StepKind {
    /// A regular handler-backed step.
    Regular(StepRegistration),
    /// A step that runs another workflow as its handler.
    SubWorkflow(SubWorkflowStep),
    /// A step that fans out into multiple sub-workflows in parallel.
    ParallelSubWorkflows(ParallelSubWorkflowsStep),
}

impl StepKind {
    /// Human-readable name for this step.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            StepKind::Regular(r) => &r.name,
            StepKind::SubWorkflow(s) => &s.name,
            StepKind::ParallelSubWorkflows(p) => &p.name,
        }
    }

    /// Event type identifiers this step accepts.
    #[must_use]
    pub fn accepts(&self) -> &[&'static str] {
        match self {
            StepKind::Regular(r) => &r.accepts,
            StepKind::SubWorkflow(s) => &s.accepts,
            StepKind::ParallelSubWorkflows(p) => &p.accepts,
        }
    }

    /// Event type identifiers this step may emit (informational).
    #[must_use]
    pub fn emits(&self) -> &[&'static str] {
        match self {
            StepKind::Regular(r) => &r.emits,
            StepKind::SubWorkflow(s) => &s.emits,
            StepKind::ParallelSubWorkflows(p) => &p.emits,
        }
    }
}

impl From<StepRegistration> for StepKind {
    fn from(reg: StepRegistration) -> Self {
        StepKind::Regular(reg)
    }
}

impl From<SubWorkflowStep> for StepKind {
    fn from(step: SubWorkflowStep) -> Self {
        StepKind::SubWorkflow(step)
    }
}

impl From<ParallelSubWorkflowsStep> for StepKind {
    fn from(step: ParallelSubWorkflowsStep) -> Self {
        StepKind::ParallelSubWorkflows(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handler() -> StepFn {
        Arc::new(|_event, _ctx| Box::pin(async { Ok(StepOutput::None) }))
    }

    fn make_registration() -> StepRegistration {
        StepRegistration::new(
            "test_step".to_owned(),
            vec!["ev::a"],
            vec!["ev::b"],
            make_handler(),
            0,
        )
    }

    #[test]
    fn step_registration_default_has_no_timeout() {
        let reg = make_registration();
        assert!(reg.timeout.is_none());
    }

    #[test]
    fn step_registration_with_timeout_sets_field() {
        let reg = make_registration().with_timeout(Duration::from_millis(250));
        assert_eq!(reg.timeout, Some(Duration::from_millis(250)));
    }

    #[test]
    fn step_registration_no_timeout_clears_field() {
        let reg = make_registration()
            .with_timeout(Duration::from_secs(1))
            .no_timeout();
        assert!(reg.timeout.is_none());
    }

    #[test]
    fn step_registration_default_has_no_retry_config() {
        let reg = StepRegistration::new(
            "n".into(),
            vec![],
            vec![],
            std::sync::Arc::new(|_, _| Box::pin(async { Ok(StepOutput::None) })),
            0,
        );
        assert!(reg.retry_config.is_none());
    }

    #[test]
    fn step_registration_with_retry_config_sets_field() {
        let reg = StepRegistration::new(
            "n".into(),
            vec![],
            vec![],
            std::sync::Arc::new(|_, _| Box::pin(async { Ok(StepOutput::None) })),
            0,
        )
        .with_retry_config(blazen_llm::retry::RetryConfig {
            max_retries: 11,
            ..blazen_llm::retry::RetryConfig::default()
        });
        assert_eq!(reg.retry_config.unwrap().max_retries, 11);
    }
}
