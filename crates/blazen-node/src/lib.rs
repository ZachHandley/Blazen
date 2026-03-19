//! `Blazen` Node.js bindings.
//!
//! Exposes the `Blazen` framework to Node.js / TypeScript via napi-rs.
//!
//! # Modules
//!
//! - [`types`] -- Shared type definitions (messages, completions, tools, media).
//! - [`compute`] -- Compute request, result, and job types.
//! - [`providers`] -- LLM completion model wrappers and provider factories.
//! - [`error`] -- Error conversion utilities.
//! - [`agent`] -- Agentic tool execution loop bindings.
//! - [`workflow`] -- Workflow builder, runner, context, handler, and events.

pub mod agent;
pub mod compute;
pub mod error;
pub mod providers;
pub mod types;
pub mod workflow;

use napi_derive::napi;

/// Returns the version of the blazen library.
#[napi]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
