//! Subclassable bindings for the distributed peer-client surface.
//!
//! Mirrors [`blazen_core::distributed::PeerClient`] and the
//! `RemoteWorkflowRequest` / `RemoteWorkflowResponse` request/response
//! pair. The trait itself is gated on the `distributed` feature on
//! `blazen-core`; the Node binding mirrors that gate via the
//! `distributed` feature on this crate.
//!
//! [`JsPeerClient`] is exposed as an abstract base. JS code may
//! `extend` it and override `invokeSubWorkflow`, `derefSessionRef`, and
//! `releaseSessionRef` to plug a custom transport (HTTP, NATS,
//! in-process mock, etc.) into `Workflow.runRemote`. The canonical
//! gRPC implementation continues to live in `BlazenPeerClient`
//! ([`crate::peer::JsBlazenPeerClient`]).
//!
//! [`JsRemoteWorkflowRequest`] / [`JsRemoteWorkflowResponse`] are
//! `#[napi(object)]` plain-object types that mirror the
//! transport-agnostic shapes in `blazen_core::distributed`.

#![cfg(feature = "distributed")]

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// JsRemoteWorkflowRequest
// ---------------------------------------------------------------------------

/// Transport-agnostic request for invoking a sub-workflow on a remote
/// peer.
///
/// Mirrors [`blazen_core::distributed::RemoteWorkflowRequest`]. Concrete
/// transports (gRPC, HTTP, NATS, etc.) serialize this into whatever
/// wire format they require.
#[napi(object)]
pub struct JsRemoteWorkflowRequest {
    /// Symbolic name of the workflow to invoke on the remote peer.
    #[napi(js_name = "workflowName")]
    pub workflow_name: String,
    /// Ordered list of step IDs to execute as part of this sub-workflow.
    #[napi(js_name = "stepIds")]
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step.
    pub input: serde_json::Value,
    /// Optional timeout in seconds. `None` means "use the server's
    /// default deadline".
    #[napi(js_name = "timeoutSecs")]
    pub timeout_secs: Option<u32>,
}

// ---------------------------------------------------------------------------
// JsRemoteWorkflowResponse
// ---------------------------------------------------------------------------

/// Transport-agnostic response from a remote sub-workflow invocation.
///
/// Mirrors [`blazen_core::distributed::RemoteWorkflowResponse`].
#[napi(object)]
pub struct JsRemoteWorkflowResponse {
    /// Optional terminal result. `None` when the workflow exited
    /// without producing one.
    pub result: Option<serde_json::Value>,
    /// Descriptors for any session refs the sub-workflow registered
    /// that the parent should be able to dereference remotely. Keyed
    /// by the registry UUID rendered as a string.
    #[napi(js_name = "remoteRefs")]
    pub remote_refs: HashMap<String, serde_json::Value>,
    /// Error message if the sub-workflow failed. When `Some`, callers
    /// should ignore `result`.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// JsPeerClient (subclassable abstract base)
// ---------------------------------------------------------------------------

/// Abstract base class for custom peer-client transports.
///
/// Mirrors [`blazen_core::distributed::PeerClient`]. Subclass and
/// override `invokeSubWorkflow`, `derefSessionRef`, and
/// `releaseSessionRef` to plug a JS-side transport into Blazen's
/// distributed-execution surface. The canonical gRPC implementation
/// lives in `BlazenPeerClient`
/// ([`crate::peer::JsBlazenPeerClient`]); this base exists for callers
/// who want to swap in a different transport (HTTP, NATS, an in-process
/// mock for tests, etc.) without touching the Rust core.
///
/// ```javascript
/// class MyTransport extends PeerClient {
///   async invokeSubWorkflow(request) { /* ... */ }
///   async derefSessionRef(refUuid) { /* ... */ }
///   async releaseSessionRef(refUuid) { /* ... */ }
/// }
/// ```
#[napi(js_name = "PeerClient")]
pub struct JsPeerClient {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsPeerClient {
    /// Create a new peer-client base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Invoke a sub-workflow on the remote peer. Subclasses **must**
    /// override this method.
    #[napi(js_name = "invokeSubWorkflow")]
    pub async fn invoke_sub_workflow(
        &self,
        _request: JsRemoteWorkflowRequest,
    ) -> Result<JsRemoteWorkflowResponse> {
        Err(napi::Error::from_reason(
            "subclass must override invokeSubWorkflow()",
        ))
    }

    /// Dereference a remote session ref by UUID. Returns the raw
    /// payload bytes. Subclasses **must** override this method.
    #[napi(js_name = "derefSessionRef")]
    pub async fn deref_session_ref(&self, _ref_uuid: String) -> Result<Buffer> {
        Err(napi::Error::from_reason(
            "subclass must override derefSessionRef()",
        ))
    }

    /// Release (drop) a remote session ref on the origin node.
    /// Returns `true` when the ref was found and dropped, `false` when
    /// it was already gone. Subclasses **must** override this method.
    #[napi(js_name = "releaseSessionRef")]
    pub async fn release_session_ref(&self, _ref_uuid: String) -> Result<bool> {
        Err(napi::Error::from_reason(
            "subclass must override releaseSessionRef()",
        ))
    }
}
