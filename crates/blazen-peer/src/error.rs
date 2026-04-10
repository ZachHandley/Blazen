//! Error type for the [`blazen-peer`](crate) crate.

use thiserror::Error;

/// All errors produced by the peer transport, protocol, and TLS layers.
#[derive(Debug, Error)]
pub enum PeerError {
    /// A postcard encode or decode call returned an error. This is
    /// produced both by the client (when it cannot serialize a request
    /// payload) and by the server (when it cannot decode an incoming
    /// payload), so the variant is intentionally bidirectional.
    #[error("postcard encoding failed: {0}")]
    Encode(#[from] postcard::Error),

    /// Wraps any tonic transport-layer failure. We hold the message as
    /// a `String` rather than `tonic::transport::Error` to keep the
    /// public API stable across tonic minor version bumps.
    #[error("tonic transport error: {0}")]
    Transport(String),

    /// The peer received a payload whose `envelope_version` is newer
    /// than the highest version this build understands. Callers should
    /// treat this as an unrecoverable mismatch and refuse the request.
    #[error("envelope version {got} is newer than supported {supported}")]
    EnvelopeVersion {
        /// The version reported on the wire.
        got: u32,
        /// The highest version this build supports.
        supported: u32,
    },

    /// The remote workflow ran but produced an error result. The
    /// payload string is whatever
    /// [`crate::protocol::SubWorkflowResponse::error`] held.
    #[error("blazen workflow error: {0}")]
    Workflow(String),

    /// Producing or consuming a TLS configuration failed (bad PEM,
    /// missing key, refusing client cert, etc.).
    #[error("TLS configuration error: {0}")]
    Tls(String),

    /// The remote peer asked us to invoke a step that is not registered
    /// on this node. Returned by the server-side dispatcher when the
    /// requested `step_id` cannot be resolved against the local
    /// [`blazen_core::step_registry`].
    #[error("unknown step id `{step_id}`")]
    UnknownStep {
        /// The unrecognized step identifier.
        step_id: String,
    },
}
