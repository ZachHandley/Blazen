//! Error types for the Blazen Python bindings.
//!
//! Maps Rust-side errors to appropriate Python exception types.
//! Provides rich exception subclasses that mirror [`BlazenError`] variants.

use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Custom Python exception hierarchy
// ---------------------------------------------------------------------------

pyo3::create_exception!(blazen, BlazenException, pyo3::exceptions::PyException);
pyo3::create_exception!(blazen, AuthError, BlazenException);
pyo3::create_exception!(blazen, RateLimitError, BlazenException);
pyo3::create_exception!(blazen, BlazenTimeoutError, BlazenException);
pyo3::create_exception!(blazen, ValidationError, BlazenException);
pyo3::create_exception!(blazen, ContentPolicyError, BlazenException);
pyo3::create_exception!(blazen, ProviderError, BlazenException);
pyo3::create_exception!(blazen, UnsupportedError, BlazenException);
pyo3::create_exception!(blazen, ComputeError, BlazenException);
pyo3::create_exception!(blazen, MediaError, BlazenException);

/// Register all custom exception types on the Python module.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BlazenError", m.py().get_type::<BlazenException>())?;
    m.add("AuthError", m.py().get_type::<AuthError>())?;
    m.add("RateLimitError", m.py().get_type::<RateLimitError>())?;
    m.add("TimeoutError", m.py().get_type::<BlazenTimeoutError>())?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add(
        "ContentPolicyError",
        m.py().get_type::<ContentPolicyError>(),
    )?;
    m.add("ProviderError", m.py().get_type::<ProviderError>())?;
    m.add("UnsupportedError", m.py().get_type::<UnsupportedError>())?;
    m.add("ComputeError", m.py().get_type::<ComputeError>())?;
    m.add("MediaError", m.py().get_type::<MediaError>())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// BlazenPyError -- internal error type
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BlazenError -> PyErr conversion (rich exception mapping)
// ---------------------------------------------------------------------------

/// Convert a [`BlazenError`] to a rich [`PyErr`] with the appropriate
/// exception subclass.
///
/// Cannot implement `From<BlazenError> for PyErr` due to the orphan rule,
/// so this is a standalone function used throughout the crate.
pub fn blazen_error_to_pyerr(err: blazen_llm::BlazenError) -> PyErr {
    match &err {
        blazen_llm::BlazenError::Auth { message } => AuthError::new_err(message.clone()),
        blazen_llm::BlazenError::RateLimit { .. } => RateLimitError::new_err(err.to_string()),
        blazen_llm::BlazenError::Timeout { .. } => BlazenTimeoutError::new_err(err.to_string()),
        blazen_llm::BlazenError::Validation { .. } => ValidationError::new_err(err.to_string()),
        blazen_llm::BlazenError::ContentPolicy { message } => {
            ContentPolicyError::new_err(message.clone())
        }
        blazen_llm::BlazenError::Provider { .. } => ProviderError::new_err(err.to_string()),
        blazen_llm::BlazenError::Unsupported { message } => {
            UnsupportedError::new_err(message.clone())
        }
        blazen_llm::BlazenError::Compute(_) => ComputeError::new_err(err.to_string()),
        blazen_llm::BlazenError::Media(_) => MediaError::new_err(err.to_string()),
        _ => BlazenException::new_err(err.to_string()),
    }
}

/// Convenience alias for `Result<T, BlazenPyError>`.
pub type Result<T> = std::result::Result<T, BlazenPyError>;

/// Convert a [`Result`] to a [`PyResult`].
pub fn to_py_result<T>(result: Result<T>) -> PyResult<T> {
    result.map_err(std::convert::Into::into)
}
