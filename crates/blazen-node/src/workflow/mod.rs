//! Workflow builder, runner, event conversion, context, and handler.

pub mod context;
pub mod event;
pub mod events_typed;
pub mod handler;
pub mod session_ref;
pub mod session_ref_convert;
pub mod session_ref_serializable;
#[allow(clippy::module_inception)]
pub mod workflow;

// Re-export the main types.
pub use context::{JsContext, JsSessionNamespace, JsStateNamespace, SerializableRefPayload};
pub use events_typed::{
    JsDynamicEvent, JsEventEnvelope, JsInputRequestEvent, JsInputResponseEvent,
};
pub use handler::JsWorkflowHandler;
pub use workflow::{JsSessionPausePolicy, JsWorkflow, JsWorkflowBuilder, JsWorkflowResult};
