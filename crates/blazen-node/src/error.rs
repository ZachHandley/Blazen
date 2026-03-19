//! Error conversion utilities for napi-rs.
//!
//! Converts internal `Blazen` errors into [`napi::Error`] for the Node.js side.

use napi::Status;

/// Convert any `Display`-able error into a [`napi::Error`].
pub fn to_napi_error(err: impl std::fmt::Display) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`WorkflowError`](blazen_core::WorkflowError) into a [`napi::Error`].
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn workflow_error_to_napi(err: blazen_core::WorkflowError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Convert a [`BlazenError`](blazen_llm::BlazenError) into a [`napi::Error`].
///
/// Intentionally takes by value for use with `map_err`.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn blazen_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    napi::Error::new(Status::GenericFailure, err.to_string())
}

/// Backwards-compatible alias for [`blazen_error_to_napi`].
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn llm_error_to_napi(err: blazen_llm::BlazenError) -> napi::Error {
    blazen_error_to_napi(err)
}
