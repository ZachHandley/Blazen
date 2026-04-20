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

// NOTE: `BlazenPyError` is used only for the model-manager code path
// (`manager.rs::load`/`unload`/`ensure_loaded`). Provider HTTP errors
// flow through `blazen_error_to_pyerr` directly from `providers/*.rs`
// and do NOT pass through this conversion. If that ever changes,
// add `BlazenError::ProviderHttp(_)` handling below.
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

/// Construct a `ProviderError` exception with structured attributes attached
/// via `setattr`. Keeps the existing `create_exception!` class hierarchy
/// (`BlazenError -> ProviderError`) while giving Python callers typed
/// attribute access: `e.provider`, `e.status`, `e.endpoint`, `e.request_id`,
/// `e.detail`, `e.raw_body`, `e.retry_after_ms`.
///
/// Attributes that don't apply (e.g. `status` on a non-HTTP `Provider`)
/// are set to `None`.
#[allow(clippy::too_many_arguments)]
fn build_provider_error(
    message: String,
    provider: String,
    status: Option<u16>,
    endpoint: Option<String>,
    request_id: Option<String>,
    detail: Option<String>,
    raw_body: Option<String>,
    retry_after_ms: Option<u64>,
) -> PyErr {
    Python::attach(|py| {
        let err = ProviderError::new_err(message);
        // setattr on the exception's value object. If the GIL call or
        // setattr fails (should not in practice), we still return the
        // basic exception so callers get *something* rather than a panic.
        let value = err.value(py);
        let _ = value.setattr("provider", provider);
        let _ = value.setattr("status", status);
        let _ = value.setattr("endpoint", endpoint);
        let _ = value.setattr("request_id", request_id);
        let _ = value.setattr("detail", detail);
        let _ = value.setattr("raw_body", raw_body);
        let _ = value.setattr("retry_after_ms", retry_after_ms);
        err
    })
}

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
        blazen_llm::BlazenError::Provider {
            provider,
            message: _,
            status_code,
        } => build_provider_error(
            err.to_string(),
            provider.clone(),
            *status_code,
            None,
            None,
            None,
            None,
            None,
        ),
        blazen_llm::BlazenError::ProviderHttp(d) => build_provider_error(
            err.to_string(),
            d.provider.to_string(),
            Some(d.status),
            Some(d.endpoint.clone()),
            d.request_id.clone(),
            d.detail.clone(),
            Some(d.raw_body.clone()),
            d.retry_after_ms,
        ),
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
