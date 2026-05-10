//! Wasi-compatible (and native-fallback) HTTP/JSON peer client.
//!
//! Mirrors [`crate::peer::client::JsBlazenPeerClient`]'s method surface but
//! uses [`blazen_peer::HttpPeerClient`] under the hood — pure HTTP/JSON over
//! `Arc<dyn blazen_llm::http::HttpClient>`. The resolved HTTP client comes
//! from a [`blazen_llm::http_napi_wasi::LazyHttpClient`] proxy on wasi (which
//! defers to whatever the host registered via `setDefaultHttpClient`) and
//! from [`blazen_llm::ReqwestHttpClient`] on native targets.
//!
//! Use this on Cloudflare Workers / Deno where the gRPC transport doesn't
//! compile, or on native when you want pure-HTTP/JSON peer transport.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;
use uuid::Uuid;

use blazen_core::session_ref::RegistryKey;
use blazen_peer::HttpPeerClient;

use crate::error::peer_error_to_napi;
use crate::peer::types::{
    JsDerefRequest, JsDerefResponse, JsReleaseRequest, JsReleaseResponse, JsSubWorkflowRequest,
    JsSubWorkflowResponse,
};

/// HTTP/JSON peer client. Mirrors
/// [`crate::peer::client::JsBlazenPeerClient`]'s method surface but speaks
/// pure HTTP/JSON to a peer (or peer-shim) at `baseUrl`. Available on every
/// target — including `wasm32-wasip1*` where the gRPC client doesn't
/// compile.
///
/// ```typescript
/// const client = HttpPeerClient.newHttp("https://peer.example.com", "node-a");
/// const resp = await client.invokeSubWorkflow({
///     workflowName: "summarize",
///     stepIds: ["fetch", "summarize"],
///     input: { url: "https://example.com" },
///     timeoutSecs: 60,
/// });
/// ```
#[napi(js_name = "HttpPeerClient")]
pub struct JsHttpPeerClient {
    inner: Arc<Mutex<HttpPeerClient>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsHttpPeerClient {
    /// Build a new HTTP/JSON peer client.
    ///
    /// `baseUrl` is the peer's HTTP root (e.g.
    /// `https://peer.example.com`); a trailing slash is tolerated and
    /// trimmed before each request. `nodeId` identifies this caller in
    /// trace logs and is sent to the peer as the
    /// `X-Blazen-Peer-Node-Id` header.
    ///
    /// On wasi the underlying HTTP client comes from a
    /// [`blazen_llm::http_napi_wasi::LazyHttpClient`] proxy that defers
    /// to whatever the host registered via `setDefaultHttpClient`. On
    /// native a stock [`blazen_llm::ReqwestHttpClient`] is used.
    #[napi(factory, js_name = "newHttp")]
    pub fn new_http(base_url: String, node_id: String) -> Self {
        let http = default_http_client();
        let client = HttpPeerClient::new(base_url, http, node_id);
        Self {
            inner: Arc::new(Mutex::new(client)),
        }
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

/// Resolve the platform-appropriate default HTTP client. Mirrors
/// `blazen_llm::default_http_client` (which is `pub(crate)`): on wasi we
/// build a [`blazen_llm::http_napi_wasi::LazyHttpClient`] (which defers to
/// `setDefaultHttpClient`), on native we build a stock
/// [`blazen_llm::ReqwestHttpClient`].
#[cfg(target_os = "wasi")]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::http_napi_wasi::LazyHttpClient::new().into_arc()
}

#[cfg(not(target_os = "wasi"))]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::ReqwestHttpClient::new().into_arc()
}
