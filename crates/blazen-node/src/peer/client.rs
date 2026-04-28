//! Node bindings for [`blazen_peer::BlazenPeerClient`].
//!
//! The native client requires `&mut self` for every RPC, but napi-rs
//! only exposes `&self` to JS-callable methods. We wrap the client in
//! a `tokio::sync::Mutex` and acquire it for the duration of each
//! call, which also serializes concurrent JS callers cleanly.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;
use uuid::Uuid;

use blazen_core::session_ref::RegistryKey;
use blazen_peer::BlazenPeerClient;

use crate::error::peer_error_to_napi;
use crate::peer::types::{
    JsDerefRequest, JsDerefResponse, JsReleaseRequest, JsReleaseResponse, JsSubWorkflowRequest,
    JsSubWorkflowResponse,
};

/// Client handle for talking to a remote `BlazenPeerServer`.
///
/// ```typescript
/// const client = await BlazenPeerClient.connect("http://node-b:7443", "node-a");
/// const resp = await client.invokeSubWorkflow({
///     workflowName: "summarize",
///     stepIds: ["fetch", "summarize"],
///     input: { url: "https://example.com" },
///     timeoutSecs: 60,
/// });
/// ```
#[napi(js_name = "BlazenPeerClient")]
pub struct JsBlazenPeerClient {
    inner: Arc<Mutex<BlazenPeerClient>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsBlazenPeerClient {
    /// Open a connection to a peer at `endpoint`.
    ///
    /// `endpoint` must be a valid gRPC URI, e.g.
    /// `http://node-b.local:7443`. `nodeId` identifies this client in
    /// trace logs.
    #[napi(factory)]
    pub async fn connect(endpoint: String, node_id: String) -> Result<Self> {
        let client = BlazenPeerClient::connect(endpoint, node_id)
            .await
            .map_err(peer_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(client)),
        })
    }

    /// Invoke a sub-workflow on the connected peer.
    #[napi(js_name = "invokeSubWorkflow")]
    pub async fn invoke_sub_workflow(
        &self,
        request: JsSubWorkflowRequest,
    ) -> Result<JsSubWorkflowResponse> {
        let native_req = request
            .into_native()
            .map_err(|e| napi::Error::from_reason(format!("invalid input JSON: {e}")))?;
        let mut guard = self.inner.lock().await;
        let resp = guard
            .invoke_sub_workflow(native_req)
            .await
            .map_err(peer_error_to_napi)?;
        JsSubWorkflowResponse::from_native(resp)
            .map_err(|e| napi::Error::from_reason(format!("response decode failed: {e}")))
    }

    /// Dereference a remote session ref. Returns the raw bytes of the
    /// underlying value and the envelope version of the response.
    #[napi(js_name = "derefSessionRef")]
    pub async fn deref_session_ref(&self, request: JsDerefRequest) -> Result<JsDerefResponse> {
        let envelope_version = request.envelope_version;
        let uuid = Uuid::parse_str(&request.ref_uuid)
            .map_err(|e| napi::Error::from_reason(format!("invalid ref UUID: {e}")))?;
        let mut guard = self.inner.lock().await;
        let payload = guard
            .deref_session_ref(RegistryKey(uuid))
            .await
            .map_err(peer_error_to_napi)?;
        Ok(JsDerefResponse {
            envelope_version,
            payload: Buffer::from(payload),
        })
    }

    /// Release (drop) a remote session ref. Returns whether the ref was
    /// found and released on the origin node.
    #[napi(js_name = "releaseSessionRef")]
    pub async fn release_session_ref(
        &self,
        request: JsReleaseRequest,
    ) -> Result<JsReleaseResponse> {
        let envelope_version = request.envelope_version;
        let uuid = Uuid::parse_str(&request.ref_uuid)
            .map_err(|e| napi::Error::from_reason(format!("invalid ref UUID: {e}")))?;
        let mut guard = self.inner.lock().await;
        let released = guard
            .release_session_ref(RegistryKey(uuid))
            .await
            .map_err(peer_error_to_napi)?;
        Ok(JsReleaseResponse {
            envelope_version,
            released,
        })
    }
}
