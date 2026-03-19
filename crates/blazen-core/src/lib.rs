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
pub mod error;
pub(crate) mod event_loop;
pub mod handler;
pub mod snapshot;
pub mod step;
pub mod value;
pub mod workflow;

pub use builder::{InputHandlerFn, WorkflowBuilder};
pub use context::Context;
pub use error::{Result, WorkflowError};
pub use handler::WorkflowHandler;
pub use snapshot::{SerializedEvent, WorkflowSnapshot};
pub use step::{StepFn, StepOutput, StepRegistration};
pub use value::{BytesWrapper, StateValue};
pub use workflow::Workflow;
