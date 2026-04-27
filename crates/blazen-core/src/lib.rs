//! # `Blazen` Core
//!
//! The workflow engine at the heart of the `Blazen` framework.
//!
//! This crate provides:
//!
//! - [`Context`] -- shared, typed key/value state accessible by all steps.
//! - [`StepFn`] / [`StepOutput`] / [`StepRegistration`] -- step definition
//!   primitives.
//! - [`WorkflowBuilder`] / [`Workflow`] -- fluent construction and execution
//!   of event-driven workflows.
//! - [`WorkflowHandler`] -- handle for awaiting results and streaming
//!   intermediate events.
//! - [`WorkflowError`] -- comprehensive error type for everything that can
//!   go wrong during workflow execution.
//!
//! # Architecture
//!
//! A workflow is a directed graph of *steps*. Each step declares which event
//! types it accepts and what it may emit. The runtime maintains an internal
//! event queue; when an event arrives, the engine looks up matching step
//! handlers and spawns them concurrently. Step outputs are fed back into the
//! queue until a [`StopEvent`](blazen_events::StopEvent) terminates the
//! loop.

pub mod builder;
pub mod context;
#[cfg(feature = "distributed")]
pub mod distributed;
pub mod error;
pub(crate) mod event_loop;
pub mod handler;
pub(crate) mod runtime;
pub mod session_ref;
pub mod snapshot;
pub mod step;
pub mod step_registry;
pub mod value;
pub mod workflow;

pub use builder::{InputHandlerFn, WorkflowBuilder};
pub use context::Context;
#[cfg(feature = "distributed")]
pub use distributed::{PeerClient, RemoteWorkflowRequest, RemoteWorkflowResponse};
pub use error::{Result, WorkflowError};
pub use handler::{WorkflowHandler, WorkflowResult};
pub use session_ref::{
    CURRENT_SESSION_REGISTRY, RefLifetime, RegistryKey, RemoteRefDescriptor,
    SERIALIZED_SESSION_REFS_META_KEY, SESSION_REF_TAG, SessionPausePolicy, SessionRefError,
    SessionRefRegistry, SessionRefSerializable, current_session_registry, with_session_registry,
};
pub use snapshot::{SNAPSHOT_VERSION, SerializedEvent, WorkflowSnapshot};
pub use step::{StepFn, StepOutput, StepRegistration};
pub use step_registry::{
    StepBuilderFn, StepDeserializerRegistry, lookup_step_builder, register_step_builder,
    registered_step_ids,
};
pub use value::{BytesWrapper, StateValue};
pub use workflow::{SessionRefDeserializerFn, Workflow};
