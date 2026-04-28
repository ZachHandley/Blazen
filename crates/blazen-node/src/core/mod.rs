//! Typed Node bindings for `blazen-core` types that have no other home.
//!
//! These wrappers expose primitives like the session-ref registry,
//! state value enum, step deserializer registry, and workflow snapshot
//! as napi classes so JavaScript code can interact with them directly
//! instead of going through ad-hoc JSON.

#[cfg(feature = "distributed")]
pub mod distributed;
pub mod session_ref;
pub mod snapshot;
pub mod step;
pub mod step_registry;
pub mod value;

#[cfg(feature = "distributed")]
pub use distributed::{JsPeerClient, JsRemoteWorkflowRequest, JsRemoteWorkflowResponse};
pub use session_ref::{JsRefLifetime, JsRegistryKey, JsRemoteRefDescriptor, JsSessionRefRegistry};
pub use snapshot::JsWorkflowSnapshot;
pub use step::{JsStepOutput, JsStepOutputKind, JsStepRegistration};
pub use step_registry::{
    JsStepDeserializerRegistry, lookup_step_builder_ids, register_step_builder_id,
    registered_step_builder_ids,
};
pub use value::{JsBytesWrapper, JsStateValue, JsStateValueKind};
