//! Core error types for the `Blazen` workflow engine.

use std::time::Duration;

use thiserror::Error;

/// Errors produced by the workflow engine.
#[derive(Debug, Error)]
pub enum WorkflowError {
    /// A type-erased event could not be downcast to the expected concrete type.
    #[error("event downcast failed: expected {expected}, got {got}")]
    EventDowncastFailed {
        /// The type name that was expected.
        expected: &'static str,
        /// The actual event type identifier.
        got: String,
    },

    /// No step handler is registered for the given event type.
    #[error("no handler for event type: {event_type}")]
    NoHandler {
        /// The event type identifier that had no matching handler.
        event_type: String,
    },

    /// The workflow exceeded its configured timeout.
    #[error("workflow timed out after {elapsed:?}")]
    Timeout {
        /// How long the workflow ran before being terminated.
        elapsed: Duration,
    },

    /// A named step returned an error during execution.
    #[error("step '{step_name}' failed: {source}")]
    StepFailed {
        /// The name of the step that failed.
        step_name: String,
        /// The underlying error from the step.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The internal event channel was closed unexpectedly.
    #[error("internal channel closed")]
    ChannelClosed,

    /// The workflow definition failed validation before execution.
    #[error("workflow validation failed: {0}")]
    ValidationFailed(String),

    /// An error related to the shared workflow context.
    #[error("context error: {0}")]
    Context(String),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The workflow was paused.
    ///
    /// This is not truly an error -- it signals that the workflow event loop
    /// exited cleanly because a pause was requested. The accompanying
    /// [`WorkflowSnapshot`](crate::snapshot::WorkflowSnapshot) is delivered
    /// through a separate channel.
    #[error("workflow paused")]
    Paused,

    /// A catch-all for other errors.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Convenience alias for `Result<T, WorkflowError>`.
pub type Result<T, E = WorkflowError> = std::result::Result<T, E>;
