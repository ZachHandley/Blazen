//! ZAgents Node.js bindings.
//!
//! Exposes the ZAgents framework to Node.js / TypeScript via napi-rs.
//!
//! # Modules
//!
//! - [`context`] -- JavaScript wrapper for the workflow `Context`.
//! - [`error`] -- Error conversion utilities.
//! - [`event`] -- Event conversion between JS objects and Rust events.
//! - [`handler`] -- Internal workflow handler utilities.
//! - [`llm`] -- LLM completion model wrappers with provider factories.
//! - [`workflow`] -- Workflow builder and runner.

pub mod context;
pub mod error;
pub mod event;
mod handler;
pub mod llm;
pub mod workflow;

use napi_derive::napi;

/// Returns the version of the zagents library.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
