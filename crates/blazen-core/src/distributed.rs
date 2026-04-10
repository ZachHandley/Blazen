//! Distributed workflow execution types (requires `distributed` feature).
//!
//! This module defines the [`PeerClient`] trait and the lightweight
//! request/response types that [`Workflow::run_remote`] uses to invoke a
//! workflow on a remote node. The types are transport-agnostic —
//! `blazen-peer` provides the canonical gRPC implementation of
//! [`PeerClient`] via `BlazenPeerClient`, but any transport (HTTP, NATS,
//! in-process mock, etc.) can implement the trait.
//!
//! ## Why a trait instead of a concrete client?
//!
//! `blazen-peer` depends on `blazen-core` (it needs [`crate::Context`],
//! [`crate::StepRegistration`], etc.). If `blazen-core` depended back on
//! `blazen-peer` there would be a cyclic crate dependency. Defining the
//! trait here lets `blazen-core` stay at the bottom of the dependency
//! graph while `blazen-peer` implements the trait from above.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use uuid::Uuid;

use crate::error::WorkflowError;
use crate::session_ref::RemoteRefDescriptor;

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// Transport-agnostic request for invoking a sub-workflow on a remote
/// peer.
///
/// Mirrors the fields of `blazen_peer::protocol::SubWorkflowRequest`
/// but without the postcard/serde wire encoding — the [`PeerClient`]
/// implementor is responsible for serializing this into whatever format
/// the transport requires.
#[derive(Debug, Clone)]
pub struct RemoteWorkflowRequest {
    /// Symbolic name of the workflow to invoke on the remote peer.
    pub workflow_name: String,
    /// Ordered list of step IDs to execute as part of this sub-workflow.
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step.
    pub input: serde_json::Value,
    /// Optional timeout in seconds. `None` means "use the server's
    /// default deadline".
    pub timeout_secs: Option<u64>,
}

/// Transport-agnostic response from a remote sub-workflow invocation.
///
/// Mirrors the useful parts of
/// `blazen_peer::protocol::SubWorkflowResponse` in a form that
/// [`Workflow::run_remote`] can consume without depending on
/// `blazen-peer`.
#[derive(Debug, Clone)]
pub struct RemoteWorkflowResponse {
    /// Optional terminal result. `None` when the workflow exited
    /// without producing one.
    pub result: Option<serde_json::Value>,
    /// Descriptors for any session refs the sub-workflow registered
    /// that the parent should be able to dereference remotely.
    pub remote_refs: HashMap<Uuid, RemoteRefDescriptor>,
    /// Error message if the sub-workflow failed. When `Some`, callers
    /// should ignore `result`.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Abstraction over a transport that can invoke a sub-workflow on a
/// remote node.
///
/// The canonical implementation lives in `blazen-peer` as
/// `BlazenPeerClient`. Tests can supply a mock implementation that
/// returns canned responses without standing up a gRPC server.
pub trait PeerClient: Send + Sync {
    /// Invoke a sub-workflow described by `request` on the remote peer.
    ///
    /// Implementations should serialize the request, send it over their
    /// transport, await the response, and deserialize it into a
    /// [`RemoteWorkflowResponse`].
    ///
    /// # Errors
    ///
    /// Returns a [`WorkflowError`] if the transport fails, the remote
    /// peer is unreachable, or the response cannot be decoded.
    fn invoke_sub_workflow<'a>(
        &'a self,
        request: RemoteWorkflowRequest,
    ) -> Pin<Box<dyn Future<Output = Result<RemoteWorkflowResponse, WorkflowError>> + Send + 'a>>;
}
