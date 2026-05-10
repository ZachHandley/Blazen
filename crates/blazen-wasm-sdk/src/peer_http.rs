//! HTTP/JSON peer client for the WASM SDK.
//!
//! Wraps [`blazen_peer::HttpPeerClient`] over the SDK's host
//! [`blazen_llm::FetchHttpClient`]. Mirrors the Node `JsHttpPeerClient`
//! method surface (`invokeSubWorkflow`, `derefSessionRef`,
//! `releaseSessionRef`). Use this from browser apps that need to talk
//! to a remote Blazen peer over plain HTTPS — the gRPC transport in
//! [`blazen_peer::client::BlazenPeerClient`] does not compile to
//! `wasm32-unknown-unknown`, so this binding is the wasm-compatible
//! alternative.
//!
//! ```typescript
//! import { HttpPeerClient } from '@blazen-dev/sdk';
//!
//! const client = HttpPeerClient.newHttp("https://peer.example.com", "node-a");
//! const resp = await client.invokeSubWorkflow({
//!     workflowName: "summarize",
//!     stepIds: ["fetch", "summarize"],
//!     input: { url: "https://example.com" },
//!     timeoutSecs: 60,
//! });
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tsify_next::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_core::session_ref::RegistryKey;
use blazen_llm::FetchHttpClient;
use blazen_llm::http::HttpClient;
use blazen_peer::{HttpPeerClient, RemoteRefDescriptor};

// ---------------------------------------------------------------------------
// Wire envelopes (tsify pure-data)
// ---------------------------------------------------------------------------

/// Request to invoke a sub-workflow on a remote peer over HTTP/JSON.
///
/// Mirrors the Node binding's `JsSubWorkflowRequest`. JS callers
/// supply this as a plain object; the binding converts it into a
/// native [`blazen_peer::SubWorkflowRequest`] before posting it.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmSubWorkflowRequest {
    /// Symbolic name of the workflow to invoke on the remote peer.
    pub workflow_name: String,
    /// Ordered list of step IDs to execute. Empty means "use the
    /// remote workflow's default step set".
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step. Any
    /// JSON-serializable value is accepted.
    pub input: serde_json::Value,
    /// Optional wall-clock timeout in seconds.
    pub timeout_secs: Option<u32>,
}

/// Result of a remote sub-workflow invocation.
///
/// Mirrors the Node binding's `JsSubWorkflowResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmSubWorkflowResponse {
    /// Envelope version of the wire payload.
    pub envelope_version: u32,
    /// Public state values exported by the sub-workflow, decoded from
    /// JSON.
    pub state_json: HashMap<String, serde_json::Value>,
    /// Optional terminal result, decoded from JSON. `null` when the
    /// workflow exited without producing one.
    pub result: Option<serde_json::Value>,
    /// Descriptors for any session refs the sub-workflow registered
    /// that the parent should be able to dereference remotely. Keyed
    /// by the registry UUID rendered as a string.
    pub remote_refs: HashMap<String, WasmPeerRemoteRefDescriptor>,
    /// Error message if the sub-workflow failed. When present,
    /// callers should ignore `result` and `state_json`.
    pub error: Option<String>,
}

/// Metadata describing a remote session ref handed back by an
/// `invokeSubWorkflow` call. Mirrors the Node binding's
/// `JsPeerRemoteRefDescriptor`.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmPeerRemoteRefDescriptor {
    /// Stable identifier of the node that owns the underlying value.
    pub origin_node_id: String,
    /// Type tag mirroring the Rust `SessionRefSerializable::blazen_type_tag`.
    pub type_tag: String,
    /// Wall-clock creation time on the origin node, in milliseconds
    /// since the Unix epoch.
    pub created_at_epoch_ms: f64,
}

impl From<RemoteRefDescriptor> for WasmPeerRemoteRefDescriptor {
    #[allow(clippy::cast_precision_loss)]
    fn from(value: RemoteRefDescriptor) -> Self {
        Self {
            origin_node_id: value.origin_node_id,
            type_tag: value.type_tag,
            // f64 carries 53 bits of integer precision — comfortably
            // beyond any real-world `Duration::since(UNIX_EPOCH).as_millis()`
            // for the next ~285,000 years.
            created_at_epoch_ms: value.created_at_epoch_ms as f64,
        }
    }
}

/// Request to dereference a remote session ref.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmDerefRequest {
    /// Envelope version of the wire payload.
    pub envelope_version: u32,
    /// UUID of the registry entry on the origin node, as a string.
    pub ref_uuid: String,
}

/// Response containing the dereferenced bytes for a session ref.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmDerefResponse {
    /// Envelope version of the wire payload.
    pub envelope_version: u32,
    /// Raw payload returned by the origin node.
    pub payload: Vec<u8>,
}

/// Request to release (drop) a remote session ref.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmReleaseRequest {
    /// Envelope version of the wire payload.
    pub envelope_version: u32,
    /// UUID of the registry entry to drop on the origin node, as a
    /// string.
    pub ref_uuid: String,
}

/// Acknowledgement for a [`WasmReleaseRequest`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmReleaseResponse {
    /// Envelope version of the wire payload.
    pub envelope_version: u32,
    /// `true` if the registry entry was found and dropped, `false` if
    /// it was already gone.
    pub released: bool,
}

// ---------------------------------------------------------------------------
// WasmHttpPeerClient
// ---------------------------------------------------------------------------

/// HTTP/JSON peer client. Mirrors the Node `HttpPeerClient` method
/// surface but speaks pure HTTP/JSON to a peer (or peer-shim) at
/// `baseUrl`. Available on every wasm target — the gRPC transport in
/// [`blazen_peer::client::BlazenPeerClient`] does not compile to
/// `wasm32-unknown-unknown`.
///
/// All async methods return a JS `Promise` resolving to the
/// corresponding wire envelope. Errors are surfaced as rejected
/// promises with a `string` message.
#[wasm_bindgen(js_name = "HttpPeerClient")]
pub struct WasmHttpPeerClient {
    inner: Arc<Mutex<HttpPeerClient>>,
}

// SAFETY: WASM is single-threaded; there is no other thread to race
// with. The `tokio::sync::Mutex` only buys us the ability to hold a
// borrow across `.await` points (a hard requirement of
// `HttpPeerClient`'s `&mut self` async methods).
unsafe impl Send for WasmHttpPeerClient {}
unsafe impl Sync for WasmHttpPeerClient {}

#[wasm_bindgen(js_class = "HttpPeerClient")]
impl WasmHttpPeerClient {
    /// Build a new HTTP/JSON peer client.
    ///
    /// `baseUrl` is the peer's HTTP root (e.g.
    /// `https://peer.example.com`); a trailing slash is tolerated and
    /// trimmed before each request. `nodeId` identifies this caller
    /// in trace logs and is sent to the peer as the
    /// `X-Blazen-Peer-Node-Id` header.
    ///
    /// The underlying HTTP client is the SDK's stock
    /// [`blazen_llm::FetchHttpClient`], which routes every request
    /// through the browser `fetch()` API.
    #[wasm_bindgen(js_name = "newHttp")]
    #[must_use]
    pub fn new_http(base_url: String, node_id: String) -> WasmHttpPeerClient {
        let http: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
        let client = HttpPeerClient::new(base_url, http, node_id);
        Self {
            inner: Arc::new(Mutex::new(client)),
        }
    }

    /// Invoke a sub-workflow on the connected peer (returns a
    /// Promise resolving to a [`WasmSubWorkflowResponse`]).
    #[wasm_bindgen(js_name = "invokeSubWorkflow")]
    pub fn invoke_sub_workflow(&self, request: WasmSubWorkflowRequest) -> js_sys::Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            let native_req = blazen_peer::SubWorkflowRequest::new(
                request.workflow_name,
                request.step_ids,
                &request.input,
                request.timeout_secs.map(u64::from),
            )
            .map_err(|e| JsValue::from_str(&format!("invalid input JSON: {e}")))?;
            let mut guard = inner.lock().await;
            let resp = guard
                .invoke_sub_workflow(native_req)
                .await
                .map_err(|e| JsValue::from_str(&format!("peer error: {e}")))?;
            let state_json = resp
                .state_values()
                .map_err(|e| JsValue::from_str(&format!("response decode failed: {e}")))?;
            let result = resp
                .result_value()
                .map_err(|e| JsValue::from_str(&format!("response decode failed: {e}")))?;
            let remote_refs = resp
                .remote_refs
                .into_iter()
                .map(|(k, v)| (k.to_string(), WasmPeerRemoteRefDescriptor::from(v)))
                .collect();
            let wire = WasmSubWorkflowResponse {
                envelope_version: resp.envelope_version,
                state_json,
                result,
                remote_refs,
                error: resp.error,
            };
            serde_wasm_bindgen::to_value(&wire)
                .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
        }))
    }

    /// Dereference a remote session ref. Returns the raw bytes of the
    /// underlying value and the envelope version of the response.
    #[wasm_bindgen(js_name = "derefSessionRef")]
    pub fn deref_session_ref(&self, request: WasmDerefRequest) -> js_sys::Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            let envelope_version = request.envelope_version;
            let uuid = Uuid::parse_str(&request.ref_uuid)
                .map_err(|e| JsValue::from_str(&format!("invalid ref UUID: {e}")))?;
            let mut guard = inner.lock().await;
            let payload = guard
                .deref_session_ref(RegistryKey(uuid))
                .await
                .map_err(|e| JsValue::from_str(&format!("peer error: {e}")))?;
            let wire = WasmDerefResponse {
                envelope_version,
                payload,
            };
            serde_wasm_bindgen::to_value(&wire)
                .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
        }))
    }

    /// Release (drop) a remote session ref. Resolves with whether
    /// the ref was found and released on the origin node.
    #[wasm_bindgen(js_name = "releaseSessionRef")]
    pub fn release_session_ref(&self, request: WasmReleaseRequest) -> js_sys::Promise {
        let inner = Arc::clone(&self.inner);
        future_to_promise(SendFuture(async move {
            let envelope_version = request.envelope_version;
            let uuid = Uuid::parse_str(&request.ref_uuid)
                .map_err(|e| JsValue::from_str(&format!("invalid ref UUID: {e}")))?;
            let mut guard = inner.lock().await;
            let released = guard
                .release_session_ref(RegistryKey(uuid))
                .await
                .map_err(|e| JsValue::from_str(&format!("peer error: {e}")))?;
            let wire = WasmReleaseResponse {
                envelope_version,
                released,
            };
            serde_wasm_bindgen::to_value(&wire)
                .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
        }))
    }
}

// ---------------------------------------------------------------------------
// SendFuture wrapper (mirrors the helper in `crate::http_client`)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded; there is no other thread to race
// with.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}
