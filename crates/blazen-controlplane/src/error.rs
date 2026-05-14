//! Error type for the [`blazen-controlplane`](crate) crate.

use thiserror::Error;

/// All errors produced by the control-plane transport, protocol,
/// admission, and TLS layers.
#[derive(Debug, Error)]
pub enum ControlPlaneError {
    /// A postcard encode or decode call returned an error.
    #[error("postcard encoding failed: {0}")]
    Encode(#[from] postcard::Error),

    /// A JSON encode or decode call returned an error (used for payloads
    /// carried as `Vec<u8>` JSON bytes inside postcard envelopes).
    #[error("json encoding failed: {0}")]
    Json(#[from] serde_json::Error),

    /// Wraps any tonic transport-layer failure. Held as a String to keep
    /// the public API stable across tonic minor versions.
    #[error("tonic transport error: {0}")]
    Transport(String),

    /// Payload's `envelope_version` is newer than this build understands.
    #[error("envelope version {got} is newer than supported {supported}")]
    EnvelopeVersion {
        /// The version reported on the wire.
        got: u32,
        /// The highest version this build supports.
        supported: u32,
    },

    /// TLS configuration (PEM read, identity assembly, etc.) failed.
    #[error("TLS configuration error: {0}")]
    Tls(String),

    /// Auth interceptor rejected the request: bearer token missing or
    /// wrong.
    #[error("unauthenticated: {0}")]
    Unauthenticated(String),

    /// The control plane refused to schedule a submission because no
    /// connected worker advertised a matching capability and the
    /// submitter did not opt into `wait_for_worker`.
    #[error("no worker matches capability `{workflow_name}` (required tags: {required_tags:?})")]
    NoMatchingWorker {
        /// The workflow name the submitter requested.
        workflow_name: String,
        /// The tags that the submission required.
        required_tags: Vec<String>,
    },

    /// A submission targeted a `VramBudget` worker but omitted the
    /// required `resource_hint.vram_mb` estimate.
    #[error("VramBudget worker requires a `resource_hint.vram_mb` estimate on every assignment")]
    MissingVramHint,

    /// Lookup of a run by id failed.
    #[error("unknown run id `{0}`")]
    UnknownRun(uuid::Uuid),

    /// Lookup of a worker by node id failed.
    #[error("unknown worker `{0}`")]
    UnknownWorker(String),

    /// Wraps a `blazen_core::WorkflowError` so callers can use `?` on
    /// core operations without re-mapping.
    #[error("workflow error: {0}")]
    Workflow(#[from] blazen_core::error::WorkflowError),
}
