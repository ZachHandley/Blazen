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

use zagents_events::AnyEvent;

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
    /// [`Event::event_type()`](zagents_events::Event::event_type)).
    pub accepts: Vec<&'static str>,
    /// Event type identifiers this step may emit (informational).
    pub emits: Vec<&'static str>,
    /// The async handler function.
    pub handler: StepFn,
    /// Maximum number of concurrent invocations of this step (0 = unlimited).
    pub max_concurrency: usize,
}

impl std::fmt::Debug for StepRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepRegistration")
            .field("name", &self.name)
            .field("accepts", &self.accepts)
            .field("emits", &self.emits)
            .field("max_concurrency", &self.max_concurrency)
            .finish_non_exhaustive()
    }
}
