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

    /// A binary serialization or deserialization error (e.g. `MessagePack`).
    #[error("binary serialization error: {0}")]
    BinarySerialization(String),

    /// Returned when a snapshot was written by a newer version of
    /// `blazen-core` than the reader supports.
    #[error("snapshot version {snapshot} is newer than supported version {supported}")]
    SnapshotVersionMismatch {
        /// The version recorded in the snapshot being read.
        snapshot: u32,
        /// The maximum snapshot version this reader can handle.
        supported: u32,
    },

    /// The workflow was paused.
    ///
    /// This is not truly an error -- it signals that the workflow event loop
    /// exited cleanly because a pause was requested. The accompanying
    /// [`WorkflowSnapshot`](crate::snapshot::WorkflowSnapshot) is delivered
    /// through a separate channel.
    #[error("workflow paused")]
    Paused,

    /// The workflow paused because a step requested human input.
    #[error("workflow paused for input: request_id={request_id}")]
    InputRequired {
        request_id: String,
        prompt: String,
        metadata: serde_json::Value,
    },

    /// One or more live session references could not be serialized for a
    /// snapshot. The `keys` vector contains the string-formatted UUIDs of
    /// the offending entries.
    #[error("session refs cannot be serialized for snapshot: {keys:?}")]
    SessionRefsNotSerializable {
        /// String-formatted UUIDs of the live session refs that could not
        /// be persisted.
        keys: Vec<String>,
    },

    /// Returned when a step ID cannot be looked up in the
    /// [`StepDeserializerRegistry`](crate::step_registry::StepDeserializerRegistry)
    /// — typically because the peer node that received a distributed
    /// workflow request doesn't have the same step code compiled in.
    #[error("unknown step id `{step_id}` — not found in step deserializer registry")]
    UnknownStep {
        /// The step ID that could not be resolved.
        step_id: String,
    },

    /// A catch-all for other errors.
    #[error("{0}")]
    Other(#[from] anyhow::Error),

    /// A per-step timeout fired before the step's handler completed.
    #[error("step '{step_name}' timed out after {elapsed_ms}ms")]
    StepTimeout {
        /// The name of the step whose handler exceeded its configured
        /// timeout.
        step_name: String,
        /// How long the handler ran (in milliseconds) before being
        /// terminated by `tokio::time::timeout`.
        elapsed_ms: u64,
    },

    /// A sub-workflow step failed, either because the inner workflow
    /// errored, its per-step timeout elapsed, or all of its retries were
    /// exhausted. The `message` carries a string-formatted version of
    /// the underlying [`WorkflowError`] so the error type can stay
    /// `Send + Sync + 'static` without requiring a recursive `Box<Self>`.
    #[error("sub-workflow '{step_name}' failed: {message}")]
    SubWorkflowFailed {
        /// The name of the sub-workflow step that failed.
        step_name: String,
        /// String-formatted underlying error (typically a
        /// [`WorkflowError`] from the inner workflow run).
        message: String,
    },
}

/// Convenience alias for `Result<T, WorkflowError>`.
pub type Result<T, E = WorkflowError> = std::result::Result<T, E>;
