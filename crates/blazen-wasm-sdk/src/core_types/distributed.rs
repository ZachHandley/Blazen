//! `wasm-bindgen` wrappers for [`blazen_core::distributed`].
//!
//! Exposes the transport-agnostic remote-workflow envelopes
//! ([`WasmRemoteWorkflowRequest`] / [`WasmRemoteWorkflowResponse`]) and an
//! abstract [`WasmPeerClient`] backed by a JS callback. The native gRPC
//! transport (`BlazenPeerClient` in `blazen-peer`) is NOT bound here --
//! `tonic` and HTTP/2 sockets do not compile to `wasm32`. Instead, JS
//! consumers wire up their own transport (HTTP, WebSocket, `postMessage`,
//! an in-memory mock for tests, etc.) by passing an `invokeSubWorkflow`
//! callback to the [`WasmPeerClient`] constructor.
//!
//! ```js
//! import { PeerClient, RemoteWorkflowRequest } from '@blazen-dev/sdk';
//!
//! const client = new PeerClient(async (req) => {
//!   const res = await fetch('/blazen/peer/invoke', {
//!     method: 'POST',
//!     body: JSON.stringify(req),
//!   });
//!   return await res.json();
//! });
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

use blazen_core::distributed::{PeerClient, RemoteWorkflowRequest, RemoteWorkflowResponse};
use blazen_core::error::WorkflowError;
use blazen_core::session_ref::RemoteRefDescriptor;

// ---------------------------------------------------------------------------
// SendFuture wrapper
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
///
/// Mirrors the helper in `crate::http_client`. SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded; there is no other thread to race with.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: Future> Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// Wire envelopes (tsify pure-data)
// ---------------------------------------------------------------------------

/// JS-facing copy of [`RemoteWorkflowRequest`].
///
/// Round-trips through `serde-wasm-bindgen` so the JS callback receives a
/// plain object and returns a plain object. The `timeout_secs` field uses
/// `bigint` on the JS side (tsify maps `u64` -> `bigint`).
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmRemoteWorkflowRequest {
    /// Symbolic name of the workflow to invoke on the remote peer.
    pub workflow_name: String,
    /// Ordered list of step IDs to execute as part of this sub-workflow.
    pub step_ids: Vec<String>,
    /// Initial input value passed to the workflow's first step.
    pub input: serde_json::Value,
    /// Optional timeout in seconds. `None` means "use the server's default".
    pub timeout_secs: Option<u64>,
}

impl From<RemoteWorkflowRequest> for WasmRemoteWorkflowRequest {
    fn from(r: RemoteWorkflowRequest) -> Self {
        Self {
            workflow_name: r.workflow_name,
            step_ids: r.step_ids,
            input: r.input,
            timeout_secs: r.timeout_secs,
        }
    }
}

impl From<WasmRemoteWorkflowRequest> for RemoteWorkflowRequest {
    fn from(r: WasmRemoteWorkflowRequest) -> Self {
        Self {
            workflow_name: r.workflow_name,
            step_ids: r.step_ids,
            input: r.input,
            timeout_secs: r.timeout_secs,
        }
    }
}

/// JS-facing copy of [`RemoteWorkflowResponse`].
///
/// `remote_refs` maps the string-formatted UUID of each session ref the
/// sub-workflow registered to its [`RemoteRefDescriptor`]. The Rust-side
/// `RemoteWorkflowResponse` keys this map by [`Uuid`] directly; the wasm
/// envelope stringifies the keys so JS callers see plain object keys.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmRemoteWorkflowResponse {
    /// Optional terminal result. `null` when the workflow exited without
    /// producing one.
    pub result: Option<serde_json::Value>,
    /// String-keyed copy of the `remote_refs` map (UUIDs stringified).
    pub remote_refs: HashMap<String, RemoteRefDescriptor>,
    /// Error message if the sub-workflow failed.
    pub error: Option<String>,
}

impl From<RemoteWorkflowResponse> for WasmRemoteWorkflowResponse {
    fn from(r: RemoteWorkflowResponse) -> Self {
        Self {
            result: r.result,
            remote_refs: r
                .remote_refs
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            error: r.error,
        }
    }
}

impl TryFrom<WasmRemoteWorkflowResponse> for RemoteWorkflowResponse {
    type Error = String;

    fn try_from(r: WasmRemoteWorkflowResponse) -> Result<Self, Self::Error> {
        let remote_refs = r
            .remote_refs
            .into_iter()
            .map(|(k, v)| {
                Uuid::parse_str(&k)
                    .map(|uuid| (uuid, v))
                    .map_err(|e| format!("invalid UUID `{k}` in remote_refs: {e}"))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        Ok(Self {
            result: r.result,
            remote_refs,
            error: r.error,
        })
    }
}

// ---------------------------------------------------------------------------
// Promise-await helper
// ---------------------------------------------------------------------------

async fn await_promise(value: JsValue) -> Result<JsValue, WorkflowError> {
    if value.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = value.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| WorkflowError::Context(format!("PeerClient handler rejected: {e:?}")))
    } else {
        Ok(value)
    }
}

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_PEER_CLIENT_HANDLER: &str = r#"
/** Handler signature for `PeerClient(invokeSubWorkflow)`. The handler
 *  receives a `RemoteWorkflowRequest` envelope and must resolve to a
 *  `RemoteWorkflowResponse`-shaped object (synchronously or via Promise). */
export type InvokeSubWorkflowHandler = (
    request: RemoteWorkflowRequest,
) => Promise<RemoteWorkflowResponse> | RemoteWorkflowResponse;
"#;

// ---------------------------------------------------------------------------
// WasmPeerClient
// ---------------------------------------------------------------------------

/// JS-callback-backed [`PeerClient`] for invoking sub-workflows on a remote
/// peer over a transport of the caller's choosing.
///
/// Mirrors the Python `PeerClient` ABC and the Node `PeerClient` class but
/// uses a JS-callback constructor argument (the wasm-bindgen ABI doesn't
/// support runtime subclassing across the JS<->Rust boundary the way `PyO3`
/// and napi-rs do).
#[wasm_bindgen(js_name = "PeerClient")]
pub struct WasmPeerClient {
    inner: Arc<JsPeerClient>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmPeerClient {}
unsafe impl Sync for WasmPeerClient {}

#[wasm_bindgen(js_class = "PeerClient")]
impl WasmPeerClient {
    /// Construct a new client from a JS handler that knows how to send a
    /// [`WasmRemoteWorkflowRequest`] to a remote peer and return its
    /// [`WasmRemoteWorkflowResponse`].
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(invoke_sub_workflow_handler: js_sys::Function) -> Self {
        Self {
            inner: Arc::new(JsPeerClient {
                invoke_sub_workflow_handler,
            }),
        }
    }
}

impl WasmPeerClient {
    /// Borrow the inner `Arc<dyn PeerClient>` for use by other crate
    /// modules that need to plug a JS-backed peer client into a workflow.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn inner_arc(&self) -> Arc<dyn PeerClient> {
        Arc::clone(&self.inner) as Arc<dyn PeerClient>
    }
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

struct JsPeerClient {
    invoke_sub_workflow_handler: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsPeerClient {}
unsafe impl Sync for JsPeerClient {}

impl JsPeerClient {
    async fn invoke_impl(
        &self,
        request: RemoteWorkflowRequest,
    ) -> Result<RemoteWorkflowResponse, WorkflowError> {
        let wire_request: WasmRemoteWorkflowRequest = request.into();
        let js_request = serde_wasm_bindgen::to_value(&wire_request).map_err(|e| {
            WorkflowError::Context(format!(
                "PeerClient: failed to serialise request to JS: {e}"
            ))
        })?;
        let raw = self
            .invoke_sub_workflow_handler
            .call1(&JsValue::NULL, &js_request)
            .map_err(|e| WorkflowError::Context(format!("PeerClient handler threw: {e:?}")))?;
        let resolved = await_promise(raw).await?;
        let wire_response: WasmRemoteWorkflowResponse = serde_wasm_bindgen::from_value(resolved)
            .map_err(|e| {
                WorkflowError::Context(format!("PeerClient: invalid response shape: {e}"))
            })?;
        wire_response
            .try_into()
            .map_err(|e: String| WorkflowError::Context(format!("PeerClient: {e}")))
    }
}

impl PeerClient for JsPeerClient {
    fn invoke_sub_workflow<'a>(
        &'a self,
        request: RemoteWorkflowRequest,
    ) -> Pin<Box<dyn Future<Output = Result<RemoteWorkflowResponse, WorkflowError>> + Send + 'a>>
    {
        // SAFETY: WASM is single-threaded.
        Box::pin(SendFuture(self.invoke_impl(request)))
    }
}
