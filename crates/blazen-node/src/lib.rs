//! `Blazen` Node.js bindings.
//!
//! Exposes the `Blazen` framework to Node.js / TypeScript via napi-rs.
//!
//! # Modules
//!
//! - [`context`] -- JavaScript wrapper for the workflow `Context`.
//! - [`error`] -- Error conversion utilities.
//! - [`event`] -- Event conversion between JS objects and Rust events.
//! - [`handler`] -- Workflow handler for pause/resume and streaming control.
//! - [`llm`] -- LLM completion model wrappers with provider factories.
//! - [`agent`] -- Agentic tool execution loop bindings.
//! - [`workflow`] -- Workflow builder and runner.

pub mod agent;
pub mod compute;
pub mod context;
pub mod error;
pub mod event;
pub mod fal;
pub mod handler;
pub mod llm;
pub mod workflow;

use napi_derive::napi;

/// Returns the version of the blazen library.
#[napi]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
