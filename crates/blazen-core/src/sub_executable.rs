//! [`SubExecutable`] — a uniform trait for embeddable child runners.
//!
//! Both [`Workflow`](crate::Workflow) and
//! [`Pipeline<S>`](https://docs.rs/blazen-pipeline) (in the
//! `blazen-pipeline` crate) implement this trait, which lets the workflow
//! event-loop dispatch into either kind of child without taking a
//! pipeline-crate dependency. Bindings and user code can construct a
//! [`SubPipelineStep`](crate::step::SubPipelineStep) holding any
//! `Arc<dyn SubExecutable>` and embed it inside a parent
//! [`Workflow`](crate::Workflow) just like a
//! [`SubWorkflowStep`](crate::step::SubWorkflowStep).
//!
//! The pivot type is [`serde_json::Value`] in both directions so that
//! every binding language can flow data through trait objects without
//! associated-type churn.

use async_trait::async_trait;

use crate::Context;
use crate::WorkflowError;

/// Anything that can be embedded as a child runner inside a parent
/// [`Workflow`](crate::Workflow).
///
/// Implementations execute an opaque JSON payload to completion and
/// return the terminal JSON value (typically a `StopEvent.result` or a
/// `PipelineResult.final_output`). Errors surface as
/// [`WorkflowError`].
#[async_trait]
pub trait SubExecutable: Send + Sync + std::fmt::Debug {
    /// Run this child executable with the given JSON input and parent
    /// [`Context`]. Returns the terminal JSON result on success.
    ///
    /// # Errors
    ///
    /// Returns whatever error the underlying runner produces, surfaced
    /// as a [`WorkflowError`].
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: Context,
    ) -> Result<serde_json::Value, WorkflowError>;
}
