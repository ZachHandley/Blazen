//! Python wrappers for the remaining `blazen_core` public types that are
//! not already exposed via [`crate::workflow`], [`crate::peer`], or other
//! domain-specific modules.
//!
//! The module is deliberately named `core_types` rather than `core` so it
//! never shadows the standard `::core` crate inside this binding crate.
//!
//! Sub-modules:
//!
//! - [`session_ref`] -- `SessionRefRegistry`, `RegistryKey`,
//!   `RemoteRefDescriptor` (core variant), `RefLifetime`.
//! - [`value`] -- `BytesWrapper`, `StateValue` typed enum class.
//! - [`step`] -- `StepOutput` (typed enum) and `StepRegistration`
//!   (read-only metadata for a registered step).
//! - [`step_registry`] -- `StepDeserializerRegistry` and the free-function
//!   API (`register_step_builder`, `lookup_step_builder`,
//!   `registered_step_ids`).
//! - [`snapshot`] -- typed `WorkflowSnapshot` class with
//!   `to_json`/`from_json`/`to_msgpack`/`from_msgpack`.
//! - [`workflow_result`] -- `WorkflowResult` typed wrapper bundling the
//!   terminal event with its session-ref registry.
//! - [`distributed`] (feature-gated) -- `RemoteWorkflowRequest`,
//!   `RemoteWorkflowResponse`, `PeerClient` ABC.

pub mod session_ref;
pub mod snapshot;
pub mod step;
pub mod step_registry;
pub mod value;
pub mod workflow_result;

#[cfg(feature = "distributed")]
pub mod distributed;

pub use session_ref::{PyRefLifetime, PyRegistryKey, PyRemoteRefDescriptor, PySessionRefRegistry};
pub use snapshot::PyWorkflowSnapshot;
pub use step::{PyStepOutput, PyStepRegistration};
pub use step_registry::{
    PyStepDeserializerRegistry, lookup_step_builder, register_step_builder, registered_step_ids,
};
pub use value::{PyBytesWrapper, PyStateValue};
pub use workflow_result::PyWorkflowResult;

#[cfg(feature = "distributed")]
pub use distributed::{PyPeerClient, PyRemoteWorkflowRequest, PyRemoteWorkflowResponse};
