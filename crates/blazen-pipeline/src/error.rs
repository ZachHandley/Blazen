//! Pipeline error types.
//!
//! [`PipelineError`] covers all failure modes specific to the pipeline
//! orchestrator, from validation failures to stage execution errors.

/// Errors produced by the pipeline orchestrator.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The pipeline definition failed validation before execution.
    #[error("pipeline validation failed: {0}")]
    ValidationFailed(String),

    /// A named stage returned an error during execution.
    #[error("stage '{stage_name}' failed: {source}")]
    StageFailed {
        /// The name of the stage that failed.
        stage_name: String,
        /// The underlying error from the stage's workflow.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// An error propagated from the underlying workflow engine.
    #[error("workflow error: {0}")]
    Workflow(#[from] blazen_core::WorkflowError),

    /// The pipeline was paused between stages.
    #[error("pipeline was paused")]
    Paused,

    /// The pipeline was aborted.
    #[error("pipeline was aborted")]
    Aborted,

    /// An internal channel was closed unexpectedly.
    #[error("channel closed")]
    ChannelClosed,

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The user-provided persist callback returned an error.
    #[error("persist callback failed: {0}")]
    PersistFailed(String),
}

/// Convenience alias for `Result<T, PipelineError>`.
pub type Result<T, E = PipelineError> = std::result::Result<T, E>;
