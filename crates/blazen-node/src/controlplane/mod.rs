//! Node bindings for the [`blazen_controlplane`] crate.
//!
//! Exposes the orchestrator-side [`ControlPlaneClient`] and the
//! worker-side [`ControlPlaneWorker`] / [`ControlPlaneWorkerConfig`] to
//! Node.js. Native-only — the underlying gRPC transport (tonic) is not
//! available on the wasm32-wasi* target.
//!
//! - `client` — `ControlPlaneClient`, `ControlPlaneRunEventStream`.
//! - `worker` — `ControlPlaneWorker`, `ControlPlaneWorkerConfig`,
//!   `AssignmentContext`.
//! - `types` — shared `#[napi(object)]` wrappers and tagged-union
//!   helpers (`AdmissionMode`, `Assignment`, `RunStateSnapshot`, …).
//!
//! [`ControlPlaneClient`]: client::JsControlPlaneClient
//! [`ControlPlaneWorker`]: worker::JsControlPlaneWorker
//! [`ControlPlaneWorkerConfig`]: worker::JsControlPlaneWorkerConfig

pub mod client;
pub mod types;
pub mod worker;

pub use client::JsControlPlaneClient;
pub use types::{
    JsAdmissionMode, JsAdmissionModeTag, JsAssignment, JsClientConnectOptions, JsMtlsOptions,
    JsRunEvent, JsRunStateSnapshot, JsRunStatus, JsSubmitWorkflowOptions, JsSubscribeAllOptions,
    JsWorkerCapability, JsWorkerInfo,
};
pub use worker::{JsAssignmentContext, JsControlPlaneWorker, JsControlPlaneWorkerConfig};
