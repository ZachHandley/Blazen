//! JavaScript wrapper for [`WorkflowHandler`](zagents_core::WorkflowHandler).
//!
//! The handler is returned from `Workflow.run()` internally and provides
//! access to the final result and streaming. For the public API, streaming
//! is exposed via `Workflow.runStreaming()` which uses a callback pattern
//! that is more natural in JavaScript.
//!
//! This module is kept internal -- the workflow module handles the public
//! surface area.

// Currently, the WorkflowHandler is consumed directly in workflow.rs
// rather than being exposed as a separate JS class. This avoids the
// complexity of exposing the oneshot receiver across the FFI boundary.
//
// If we need to expose the handler directly in the future (e.g., for
// more fine-grained control over streaming), we can add a JsWorkflowHandler
// class here.
