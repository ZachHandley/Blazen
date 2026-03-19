//! Error types for the Blazen Python bindings.
//!
//! Maps Rust-side errors to appropriate Python exception types.

use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Error type for `Blazen` Python bindings.
#[derive(Error, Debug)]
pub enum BlazenPyError {
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

impl From<BlazenPyError> for PyErr {
    fn from(err: BlazenPyError) -> PyErr {
        match err {
            BlazenPyError::InvalidArgument(msg) => PyValueError::new_err(msg),
            BlazenPyError::Timeout(msg) => PyTimeoutError::new_err(msg),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<blazen_core::WorkflowError> for BlazenPyError {
    fn from(err: blazen_core::WorkflowError) -> Self {
        match &err {
            blazen_core::WorkflowError::Timeout { .. } => BlazenPyError::Timeout(err.to_string()),
            blazen_core::WorkflowError::ValidationFailed(msg) => {
                BlazenPyError::InvalidArgument(msg.clone())
            }
            _ => BlazenPyError::Workflow(err.to_string()),
        }
    }
}

impl From<blazen_llm::BlazenError> for BlazenPyError {
    fn from(err: blazen_llm::BlazenError) -> Self {
        match err {
            blazen_llm::BlazenError::Timeout { .. } => Self::Timeout(err.to_string()),
            blazen_llm::BlazenError::Auth { .. } | blazen_llm::BlazenError::Validation { .. } => {
                Self::InvalidArgument(err.to_string())
            }
            _ => Self::Llm(err.to_string()),
        }
    }
}

impl From<serde_json::Error> for BlazenPyError {
    fn from(err: serde_json::Error) -> Self {
        BlazenPyError::Serialization(err.to_string())
    }
}

/// Convenience alias for `Result<T, BlazenPyError>`.
pub type Result<T> = std::result::Result<T, BlazenPyError>;

/// Convert a [`Result`] to a [`PyResult`].
pub fn to_py_result<T>(result: Result<T>) -> PyResult<T> {
    result.map_err(std::convert::Into::into)
}
