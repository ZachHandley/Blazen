//! Error conversion utilities for napi-rs.
//!
//! Converts internal `ZAgents` errors into [`napi::Error`] for the Node.js side.

use napi::Status;

/// Convert any `Display`-able error into a [`napi::Error`].
pub fn to_napi_error(err: impl std::fmt::Display) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`WorkflowError`](zagents_core::WorkflowError) into a [`napi::Error`].
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn workflow_error_to_napi(err: zagents_core::WorkflowError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert an [`LlmError`](zagents_llm::LlmError) into a [`napi::Error`].
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llm_error_to_napi(err: zagents_llm::LlmError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}
