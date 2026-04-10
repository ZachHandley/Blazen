//! Session ref infrastructure wrapper for `blazen-node` bindings.
//!
//! Unlike Python bindings, Node does NOT need a separate `ContextVar`
//! shim on top of the Tokio `task_local!` defined in
//! [`blazen_core::session_ref`]. napi-rs step handlers are scheduled
//! directly on the Tokio runtime that drives `JsWorkflow::run()`, so the
//! Tokio `task_local` installed via [`with_session_registry`] is
//! visible from inside a step's `async` block without any additional
//! plumbing.
//!
//! This module re-exports the relevant items from `blazen-core` so local
//! `use crate::workflow::session_ref::...` imports inside `blazen-node`
//! stay consistent with the Python crate layout and so a future
//! `session_ref_convert.rs` file can live alongside this one.

#[doc(inline)]
pub use blazen_core::session_ref::{
    CURRENT_SESSION_REGISTRY, SESSION_REF_TAG, current_session_registry, with_session_registry,
};
