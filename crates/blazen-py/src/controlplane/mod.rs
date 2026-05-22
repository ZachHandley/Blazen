//! Python bindings for the `blazen-controlplane` crate.
//!
//! Exposes the orchestrator-side `Client` (submit / cancel / describe /
//! list workers / subscribe to events) and the worker-side `Worker`
//! (connect to a control plane, advertise capabilities, and dispatch
//! incoming assignments to a user-supplied `AssignmentHandler`).
//!
//! All classes here are feature-gated behind `distributed` on this
//! crate, which in turn enables `blazen-controlplane/client` and
//! `blazen-controlplane/server`.

pub mod client;
pub mod model_client;
pub mod types;
pub mod worker;

pub use client::{PyControlPlaneClient, PyRunEventStream};
pub use model_client::PyModelClient;
pub use types::{
    PyAdmissionMode, PyControlPlaneResourceHint, PyControlPlaneRunStatus,
    PyControlPlaneWorkerCapability,
};
pub use worker::{
    PyAssignmentContext, PyAssignmentHandler, PyControlPlaneWorker, PyControlPlaneWorkerConfig,
};
