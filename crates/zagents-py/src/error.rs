//! Error types for the ZAgents Python bindings.
//!
//! Maps Rust-side errors to appropriate Python exception types.

use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Error type for `ZAgents` Python bindings.
#[derive(Error, Debug)]
pub enum ZAgentsPyError {
    /// A workflow engine error.
    #[error("Workflow error: {0}")]
    Workflow(String),

    /// An LLM provider error.
    #[error("LLM error: {0}")]
    Llm(String),

    /// The operation timed out.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// An invalid argument was provided.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// An event-related error.
    #[error("Event error: {0}")]
    Event(String),

    /// A serialization / deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<ZAgentsPyError> for PyErr {
    fn from(err: ZAgentsPyError) -> PyErr {
        match err {
            ZAgentsPyError::InvalidArgument(msg) => PyValueError::new_err(msg),
            ZAgentsPyError::Timeout(msg) => PyTimeoutError::new_err(msg),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<zagents_core::WorkflowError> for ZAgentsPyError {
    fn from(err: zagents_core::WorkflowError) -> Self {
        match &err {
            zagents_core::WorkflowError::Timeout { .. } => ZAgentsPyError::Timeout(err.to_string()),
            zagents_core::WorkflowError::ValidationFailed(msg) => {
                ZAgentsPyError::InvalidArgument(msg.clone())
            }
            _ => ZAgentsPyError::Workflow(err.to_string()),
        }
    }
}

impl From<zagents_llm::LlmError> for ZAgentsPyError {
    fn from(err: zagents_llm::LlmError) -> Self {
        ZAgentsPyError::Llm(err.to_string())
    }
}

impl From<serde_json::Error> for ZAgentsPyError {
    fn from(err: serde_json::Error) -> Self {
        ZAgentsPyError::Serialization(err.to_string())
    }
}

/// Convenience alias for `Result<T, ZAgentsPyError>`.
pub type Result<T> = std::result::Result<T, ZAgentsPyError>;

/// Convert a [`Result`] to a [`PyResult`].
pub fn to_py_result<T>(result: Result<T>) -> PyResult<T> {
    result.map_err(std::convert::Into::into)
}
