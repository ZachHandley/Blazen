//! WASM control-plane client + worker for browsers / wasm32-unknown-unknown.
//!
//! The standard control-plane transport in [`blazen_controlplane`] is gRPC
//! bidi, which depends on `tonic` / `rustls` / `tokio["full"]` and does not
//! compile to `wasm32-unknown-unknown`. Browser environments therefore use
//! the alternative HTTP/SSE tier exposed by
//! [`blazen_controlplane::http`]:
//!
//! - Unary requests (register, submit, cancel, describe, result, heartbeat,
//!   event, release) go over plain HTTP via the existing
//!   [`blazen_llm::FetchHttpClient`].
//! - Server-streaming responses (worker frames, run events) come back as
//!   Server-Sent Events read via the browser-native
//!   [`web_sys::EventSource`].
//!
//! Every request body and SSE event payload is a
//! [`blazen_controlplane::http::PostcardEnvelope`] — JSON `{ postcard_b64 }`
//! wrapping the postcard-encoded payload type from
//! [`blazen_controlplane::protocol`]. Auth, when configured server-side via
//! `BLAZEN_PEER_TOKEN`, is the standard `Authorization: Bearer <token>`
//! header.
//!
//! ## TypeScript surface
//!
//! ```typescript
//! import init, { ControlPlaneClient, ControlPlaneWorker } from '@blazen-dev/sdk';
//!
//! await init();
//!
//! const client = await ControlPlaneClient.connect('https://cp.example.com');
//! const snap = await client.submitWorkflow({
//!     workflowName: 'summarize',
//!     input: { url: 'https://example.com' },
//!     waitForWorker: true,
//! });
//!
//! const sub = client.subscribeRunEvents(
//!     snap.runId,
//!     (ev) => console.log(ev.eventType, ev.data),
//!     (err) => console.error(err),
//! );
//! // later: sub.close();
//! ```

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};
use web_sys::EventSource;

use blazen_controlplane::protocol::{
    self, AdmissionModeWire, AssignmentEvent, AssignmentResult, AssignmentStatus, CancelRequest,
    CapabilityWire, ENVELOPE_VERSION, RespondToInputRequest, ResourceHintWire, RunEventWire,
    RunStateSnapshotWire, RunStatusWire, ServerToWorker, SubmitRequest, Welcome, WorkerHeartbeat,
    WorkerHello,
};
use blazen_llm::FetchHttpClient;
use blazen_llm::http::{HttpClient, HttpRequest, HttpResponse};

// ---------------------------------------------------------------------------
// SendFuture wrapper (mirrors `crate::http_client::SendFuture`)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded; there is no other thread to race with.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded; there is no other thread to race with.
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

// ---------------------------------------------------------------------------
// Wire envelopes (tsify pure-data) exposed to JS
// ---------------------------------------------------------------------------

/// Capability advertised by a worker. Mirrors
/// [`blazen_controlplane::protocol::CapabilityWire`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmCapability {
    /// Capability kind tag (e.g. `"workflow:summarize"`).
    pub kind: String,
    /// Schema version of the capability.
    pub version: u32,
}

impl From<&WasmCapability> for CapabilityWire {
    fn from(c: &WasmCapability) -> Self {
        Self {
            kind: c.kind.clone(),
            version: c.version,
        }
    }
}

impl From<&CapabilityWire> for WasmCapability {
    fn from(c: &CapabilityWire) -> Self {
        Self {
            kind: c.kind.clone(),
            version: c.version,
        }
    }
}

/// Admission strategy advertised by a worker. Tagged-union mirror of
/// [`AdmissionModeWire`] usable from plain JS object literals.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum WasmAdmissionMode {
    /// Hard count cap.
    Fixed {
        /// Maximum number of concurrent in-flight assignments.
        #[serde(rename = "maxInFlight")]
        max_in_flight: u32,
    },
    /// VRAM-sum cap.
    VramBudget {
        /// Maximum sum of `resource_hint.vram_mb` across in-flight assignments.
        #[serde(rename = "totalMb")]
        total_mb: u64,
    },
    /// Worker self-decides via offer/claim/decline negotiation.
    Reactive,
}

impl From<&WasmAdmissionMode> for AdmissionModeWire {
    fn from(m: &WasmAdmissionMode) -> Self {
        match m {
            WasmAdmissionMode::Fixed { max_in_flight } => Self::Fixed {
                max_in_flight: *max_in_flight,
            },
            WasmAdmissionMode::VramBudget { total_mb } => Self::VramBudget {
                max_vram_mb: *total_mb,
            },
            WasmAdmissionMode::Reactive => Self::Reactive,
        }
    }
}

/// Optional resource estimate attached to a submitted workflow run.
/// Mirrors [`ResourceHintWire`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmResourceHint {
    /// VRAM estimate in MB. Required when targeting a `VramBudget` worker.
    pub vram_mb: Option<u64>,
    /// CPU-core estimate. Advisory.
    pub cpu_cores: Option<f32>,
    /// Expected runtime in seconds. Advisory.
    pub expected_seconds: Option<u32>,
}

impl From<&WasmResourceHint> for ResourceHintWire {
    fn from(h: &WasmResourceHint) -> Self {
        Self {
            vram_mb: h.vram_mb,
            cpu_cores: h.cpu_cores,
            expected_seconds: h.expected_seconds,
        }
    }
}

/// Worker configuration accepted by [`WasmControlPlaneWorker::connect`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmWorkerConfig {
    /// Stable identifier of the worker node.
    pub node_id: String,
    /// Capabilities advertised by this worker.
    pub capabilities: Vec<WasmCapability>,
    /// Free-form key/value attributes used by tag predicates.
    #[serde(default)]
    pub tags: Option<BTreeMap<String, String>>,
    /// Admission strategy. `None` defaults to
    /// [`WasmAdmissionMode::Reactive`].
    #[serde(default)]
    pub admission: Option<WasmAdmissionMode>,
    /// Optional bearer token to send as `Authorization: Bearer ...`.
    #[serde(default)]
    pub bearer_token: Option<String>,
}

/// Submission request accepted by
/// [`WasmControlPlaneClient::submit_workflow`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmSubmitRequest {
    /// Symbolic name of the workflow to run.
    pub workflow_name: String,
    /// JSON-serialisable input forwarded to the workflow's first step.
    pub input: serde_json::Value,
    /// Workflow version. `None` = latest available.
    #[serde(default)]
    pub workflow_version: Option<u32>,
    /// Required tags a matching worker must advertise. AND-combined.
    #[serde(default)]
    pub required_tags: Option<Vec<String>>,
    /// Optional dedupe key.
    #[serde(default)]
    pub idempotency_key: Option<String>,
    /// Optional deadline in milliseconds from submission.
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    /// If `true` and no worker matches at submit time, queue the request
    /// until one appears. Defaults to `false`.
    #[serde(default)]
    pub wait_for_worker: Option<bool>,
    /// Optional resource estimate. Required when targeting a `VramBudget`
    /// worker.
    #[serde(default)]
    pub resource_hint: Option<WasmResourceHint>,
}

/// Snapshot of a workflow run's state. JS-friendly mirror of
/// [`RunStateSnapshotWire`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmRunStateSnapshot {
    /// Identifier of the run.
    pub run_id: String,
    /// Current lifecycle status.
    pub status: WasmRunStatus,
    /// Wall-clock submission timestamp, milliseconds since the Unix epoch.
    pub started_at_ms: f64,
    /// Wall-clock completion timestamp, milliseconds since the Unix epoch.
    pub completed_at_ms: Option<f64>,
    /// Node id of the worker the run was assigned to, once assigned.
    pub assigned_to: Option<String>,
    /// Timestamp of the most recent event for this run, in ms since epoch.
    pub last_event_at_ms: Option<f64>,
    /// Terminal output if `status == Completed`. `null` otherwise.
    pub output: Option<serde_json::Value>,
    /// Error description when `status == Failed`. `null` otherwise.
    pub error: Option<String>,
}

/// JS-friendly mirror of [`RunStatusWire`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Tsify, PartialEq, Eq)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum WasmRunStatus {
    /// Submitted but not yet assigned to a worker.
    Pending,
    /// Assigned to a worker and executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cancelled.
    Cancelled,
}

impl From<RunStatusWire> for WasmRunStatus {
    fn from(s: RunStatusWire) -> Self {
        match s {
            RunStatusWire::Pending => Self::Pending,
            RunStatusWire::Running => Self::Running,
            RunStatusWire::Completed => Self::Completed,
            RunStatusWire::Failed => Self::Failed,
            RunStatusWire::Cancelled => Self::Cancelled,
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn snapshot_from_wire(wire: &RunStateSnapshotWire) -> Result<WasmRunStateSnapshot, JsValue> {
    let output = if wire.output_json.is_empty() {
        None
    } else {
        Some(
            serde_json::from_slice::<serde_json::Value>(&wire.output_json)
                .map_err(|e| JsValue::from_str(&format!("decode output_json: {e}")))?,
        )
    };
    Ok(WasmRunStateSnapshot {
        run_id: wire.run_id.to_string(),
        status: wire.status.into(),
        // u64 -> f64: control-plane timestamps fit comfortably within
        // the 53-bit f64 mantissa for the next ~285,000 years.
        started_at_ms: wire.started_at_ms as f64,
        completed_at_ms: wire.completed_at_ms.map(|v| v as f64),
        assigned_to: wire.assigned_to.clone(),
        last_event_at_ms: wire.last_event_at_ms.map(|v| v as f64),
        output,
        error: wire.error.clone(),
    })
}

/// JS-friendly mirror of [`RunEventWire`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmRunEvent {
    /// Identifier of the run this event belongs to.
    pub run_id: String,
    /// Free-form event kind.
    pub event_type: String,
    /// Decoded event payload (JSON).
    pub data: serde_json::Value,
    /// Wall-clock timestamp, milliseconds since the Unix epoch.
    pub timestamp_ms: f64,
}

#[allow(clippy::cast_precision_loss)]
fn event_from_wire(wire: &RunEventWire) -> Result<WasmRunEvent, JsValue> {
    let data: serde_json::Value = if wire.data_json.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&wire.data_json)
            .map_err(|e| JsValue::from_str(&format!("decode event data_json: {e}")))?
    };
    Ok(WasmRunEvent {
        run_id: wire.run_id.to_string(),
        event_type: wire.event_type.clone(),
        data,
        timestamp_ms: wire.timestamp_ms as f64,
    })
}

/// JS-shape passed to the worker handler.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmAssignment {
    /// Identifier of this assignment / run.
    pub run_id: String,
    /// Symbolic name of the workflow to execute.
    pub workflow_name: String,
    /// Raw JSON-encoded input bytes; callers may `JSON.parse(new
    /// TextDecoder().decode(inputJson))` if they prefer a Value.
    pub input_json: Vec<u8>,
    /// Workflow version, if pinned.
    pub workflow_version: Option<u32>,
    /// Optional deadline in ms from worker receipt.
    pub deadline_ms: Option<f64>,
    /// Attempt counter.
    pub attempt: u32,
}

// ---------------------------------------------------------------------------
// Postcard envelope helpers (mirror of `http::PostcardEnvelope`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostcardEnvelopeJson {
    postcard_b64: String,
}

/// JSON response shape for `POST /v1/cp/worker/register`. The HTTP
/// layer in [`blazen_controlplane::http`] returns this; the worker
/// connect handler decodes it before opening the SSE stream.
#[derive(Debug, Deserialize)]
struct RegisterResponse {
    session_id: String,
    welcome: PostcardEnvelopeJson,
}

fn encode_envelope<T: Serialize>(value: &T) -> Result<PostcardEnvelopeJson, JsValue> {
    let bytes = postcard::to_allocvec(value)
        .map_err(|e| JsValue::from_str(&format!("postcard encode: {e}")))?;
    Ok(PostcardEnvelopeJson {
        postcard_b64: BASE64.encode(&bytes),
    })
}

fn decode_envelope<T: serde::de::DeserializeOwned>(
    env: &PostcardEnvelopeJson,
) -> Result<T, JsValue> {
    let bytes = BASE64
        .decode(env.postcard_b64.as_bytes())
        .map_err(|e| JsValue::from_str(&format!("base64 decode: {e}")))?;
    postcard::from_bytes(&bytes).map_err(|e| JsValue::from_str(&format!("postcard decode: {e}")))
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

/// Trim a trailing slash from `base` so concatenation always produces
/// `<base>/v1/cp/...`. Mirrors [`blazen_peer::HttpPeerClient`]'s base
/// handling.
fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    format!("{base}{path}")
}

fn auth_header(token: Option<&str>) -> Vec<(String, String)> {
    let mut headers = vec![("Content-Type".to_owned(), "application/json".to_owned())];
    if let Some(t) = token {
        headers.push(("Authorization".to_owned(), format!("Bearer {t}")));
    }
    headers
}

async fn http_send(
    http: &Arc<dyn HttpClient>,
    request: HttpRequest,
) -> Result<HttpResponse, JsValue> {
    http.send(request)
        .await
        .map_err(|e| JsValue::from_str(&format!("control-plane HTTP error: {e}")))
}

fn http_ok(resp: &HttpResponse, label: &str) -> Result<(), JsValue> {
    if (200..300).contains(&resp.status) {
        Ok(())
    } else {
        let body = String::from_utf8_lossy(&resp.body);
        Err(JsValue::from_str(&format!(
            "control-plane {label} returned HTTP {}: {body}",
            resp.status
        )))
    }
}

async fn post_envelope<T: Serialize>(
    http: &Arc<dyn HttpClient>,
    url: String,
    token: Option<&str>,
    payload: &T,
    label: &str,
) -> Result<HttpResponse, JsValue> {
    let envelope = encode_envelope(payload)?;
    let body = serde_json::to_vec(&envelope)
        .map_err(|e| JsValue::from_str(&format!("encode envelope JSON: {e}")))?;
    let mut request = HttpRequest::post(url).body(body);
    request.headers = auth_header(token);
    let resp = http_send(http, request).await?;
    http_ok(&resp, label)?;
    Ok(resp)
}

async fn get_envelope_response(
    http: &Arc<dyn HttpClient>,
    url: String,
    token: Option<&str>,
    label: &str,
) -> Result<HttpResponse, JsValue> {
    let mut request = HttpRequest::get(url);
    request.headers = auth_header(token);
    let resp = http_send(http, request).await?;
    http_ok(&resp, label)?;
    Ok(resp)
}

fn decode_response_envelope<T: serde::de::DeserializeOwned>(
    resp: &HttpResponse,
    label: &str,
) -> Result<T, JsValue> {
    let env: PostcardEnvelopeJson = serde_json::from_slice(&resp.body)
        .map_err(|e| JsValue::from_str(&format!("{label}: parse envelope JSON: {e}")))?;
    decode_envelope(&env)
}

// ---------------------------------------------------------------------------
// SSE subscription handle
// ---------------------------------------------------------------------------

/// Handle returned by [`WasmControlPlaneClient::subscribe_run_events`] /
/// [`WasmControlPlaneClient::subscribe_all`]. Calling [`close`] closes
/// the underlying [`web_sys::EventSource`] and drops the registered
/// callbacks.
#[wasm_bindgen(js_name = "ControlPlaneSubscription")]
pub struct WasmControlPlaneSubscription {
    source: Option<EventSource>,
    // Closures must outlive the EventSource — we own them here and drop
    // them in `close` (or in the `Drop` impl). Storage-only fields.
    on_message: Option<Closure<dyn FnMut(web_sys::MessageEvent)>>,
    on_error: Option<Closure<dyn FnMut(web_sys::Event)>>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmControlPlaneSubscription {}
// SAFETY: WASM is single-threaded.
unsafe impl Sync for WasmControlPlaneSubscription {}

#[wasm_bindgen(js_class = "ControlPlaneSubscription")]
impl WasmControlPlaneSubscription {
    /// Stop the SSE subscription. Idempotent.
    pub fn close(&mut self) {
        if let Some(source) = self.source.take() {
            source.close();
        }
        self.on_message = None;
        self.on_error = None;
    }
}

impl Drop for WasmControlPlaneSubscription {
    fn drop(&mut self) {
        if let Some(ref source) = self.source {
            source.close();
        }
    }
}

/// Helper: open an `EventSource` against `url` and wire the supplied
/// JS callbacks. The returned subscription owns the `EventSource` plus
/// the closure storage so neither is dropped while events are flowing.
fn open_event_source(
    url: &str,
    on_event: &js_sys::Function,
    on_error: &js_sys::Function,
    decode: fn(&PostcardEnvelopeJson) -> Result<JsValue, JsValue>,
) -> Result<WasmControlPlaneSubscription, JsValue> {
    let source =
        EventSource::new(url).map_err(|e| JsValue::from_str(&format!("EventSource: {e:?}")))?;

    let on_event_cb = on_event.clone();
    let on_error_cb = on_error.clone();
    let on_error_for_msg = on_error.clone();

    let on_message = Closure::wrap(Box::new(move |evt: web_sys::MessageEvent| {
        let Some(raw) = evt.data().as_string() else {
            let _ = on_error_for_msg.call1(
                &JsValue::NULL,
                &JsValue::from_str("SSE frame was not a string"),
            );
            return;
        };
        let env: PostcardEnvelopeJson = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                let _ = on_error_for_msg.call1(
                    &JsValue::NULL,
                    &JsValue::from_str(&format!("parse SSE envelope JSON: {e}")),
                );
                return;
            }
        };
        match decode(&env) {
            Ok(js_val) => {
                let _ = on_event_cb.call1(&JsValue::NULL, &js_val);
            }
            Err(e) => {
                let msg = e
                    .as_string()
                    .unwrap_or_else(|| "decode SSE envelope".to_string());
                let _ = on_error_for_msg.call1(&JsValue::NULL, &JsValue::from_str(&msg));
            }
        }
    }) as Box<dyn FnMut(web_sys::MessageEvent)>);

    let on_err_closure = Closure::wrap(Box::new(move |_evt: web_sys::Event| {
        let _ = on_error_cb.call1(
            &JsValue::NULL,
            &JsValue::from_str("SSE connection error (see browser network panel)"),
        );
    }) as Box<dyn FnMut(web_sys::Event)>);

    source.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    source.set_onerror(Some(on_err_closure.as_ref().unchecked_ref()));

    Ok(WasmControlPlaneSubscription {
        source: Some(source),
        on_message: Some(on_message),
        on_error: Some(on_err_closure),
    })
}

fn decode_run_event_js(env: &PostcardEnvelopeJson) -> Result<JsValue, JsValue> {
    let wire: RunEventWire = decode_envelope(env)?;
    let wasm = event_from_wire(&wire)?;
    serde_wasm_bindgen::to_value(&wasm).map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
}

// ---------------------------------------------------------------------------
// WasmControlPlaneClient
// ---------------------------------------------------------------------------

/// HTTP/SSE control-plane orchestrator client.
///
/// Equivalent to [`blazen_controlplane::Client`] but for the browser.
/// Construct with [`WasmControlPlaneClient::connect`] and use the
/// `submit*` / `cancel*` / `describe*` / `subscribe*` methods. The
/// constructor performs no network I/O — it only validates the
/// endpoint shape; the first request happens lazily.
#[wasm_bindgen(js_name = "ControlPlaneClient")]
pub struct WasmControlPlaneClient {
    endpoint: String,
    bearer_token: Option<String>,
    http: Arc<dyn HttpClient>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmControlPlaneClient {}
// SAFETY: WASM is single-threaded.
unsafe impl Sync for WasmControlPlaneClient {}

#[wasm_bindgen(js_class = "ControlPlaneClient")]
impl WasmControlPlaneClient {
    /// Connect to a control plane at `endpoint`. The endpoint must be
    /// the HTTP root of the server (e.g. `https://cp.example.com`); the
    /// `/v1/cp/...` path is appended automatically.
    ///
    /// `bearerToken`, when provided, is sent as
    /// `Authorization: Bearer <token>` on every request. Pass `null`
    /// (or omit) when the server has no `BLAZEN_PEER_TOKEN` configured.
    ///
    /// Returns a `Promise<ControlPlaneClient>`. The promise resolves
    /// without performing any network I/O — call any of the unary
    /// methods to verify the endpoint is reachable.
    #[wasm_bindgen]
    pub fn connect(endpoint: String, bearer_token: Option<String>) -> js_sys::Promise {
        future_to_promise(SendFuture(async move {
            if endpoint.is_empty() {
                return Err(JsValue::from_str("endpoint must not be empty"));
            }
            if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                return Err(JsValue::from_str(
                    "endpoint must start with http:// or https://",
                ));
            }
            let client = WasmControlPlaneClient {
                endpoint,
                bearer_token,
                http: FetchHttpClient::new().into_arc(),
            };
            Ok(JsValue::from(client))
        }))
    }

    /// Submit a new workflow run. Returns a `Promise<RunStateSnapshot>`.
    #[wasm_bindgen(js_name = "submitWorkflow")]
    pub fn submit_workflow(&self, req: WasmSubmitRequest) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let payload = SubmitRequest {
                envelope_version: ENVELOPE_VERSION,
                workflow_name: req.workflow_name,
                workflow_version: req.workflow_version,
                input_json: serde_json::to_vec(&req.input)
                    .map_err(|e| JsValue::from_str(&format!("encode input: {e}")))?,
                required_tags: req.required_tags.unwrap_or_default(),
                idempotency_key: req.idempotency_key,
                deadline_ms: req.deadline_ms,
                wait_for_worker: req.wait_for_worker.unwrap_or(false),
                resource_hint: req.resource_hint.as_ref().map(Into::into),
                place: None,
            };
            let url = join_url(&endpoint, "/v1/cp/submit");
            let resp = post_envelope(&http, url, token.as_deref(), &payload, "submit").await?;
            let wire: RunStateSnapshotWire = decode_response_envelope(&resp, "submit")?;
            let snap = snapshot_from_wire(&wire)?;
            serde_wasm_bindgen::to_value(&snap)
                .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
        }))
    }

    /// Cancel an in-flight workflow run. Resolves with the run's
    /// post-cancel [`WasmRunStateSnapshot`].
    #[wasm_bindgen(js_name = "cancelWorkflow")]
    pub fn cancel_workflow(&self, run_id: String) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let uuid = Uuid::parse_str(&run_id)
                .map_err(|e| JsValue::from_str(&format!("invalid run_id UUID: {e}")))?;
            let payload = CancelRequest {
                envelope_version: ENVELOPE_VERSION,
                run_id: uuid,
            };
            let url = join_url(&endpoint, "/v1/cp/cancel");
            let resp = post_envelope(&http, url, token.as_deref(), &payload, "cancel").await?;
            let wire: RunStateSnapshotWire = decode_response_envelope(&resp, "cancel")?;
            let snap = snapshot_from_wire(&wire)?;
            serde_wasm_bindgen::to_value(&snap)
                .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
        }))
    }

    /// Answer an outstanding `input.request` raised by an in-flight
    /// assignment (the B2 input-request round-trip). `runId` identifies
    /// the run awaiting input, `requestId` is the correlation id echoed
    /// from the `input.request` event payload, and `response` is any
    /// JSON-serializable value forwarded to the worker's pending request.
    ///
    /// Resolves with `undefined` once the control plane accepts the
    /// response (the gateway pushes it to the worker over SSE as a
    /// `ServerToWorker::InputResponse` frame).
    #[wasm_bindgen(js_name = "respondToInput")]
    pub fn respond_to_input(
        &self,
        run_id: String,
        request_id: String,
        response: JsValue,
    ) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let uuid = Uuid::parse_str(&run_id)
                .map_err(|e| JsValue::from_str(&format!("invalid run_id UUID: {e}")))?;
            let value: serde_json::Value = serde_wasm_bindgen::from_value(response)
                .map_err(|e| JsValue::from_str(&format!("decode response value: {e}")))?;
            let response_json = serde_json::to_vec(&value)
                .map_err(|e| JsValue::from_str(&format!("encode response JSON: {e}")))?;
            let payload = RespondToInputRequest {
                envelope_version: ENVELOPE_VERSION,
                run_id: uuid,
                request_id,
                response_json,
            };
            let url = join_url(&endpoint, "/v1/cp/respond-to-input");
            post_envelope(&http, url, token.as_deref(), &payload, "respond-to-input").await?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Look up the current state of a run. Returns a
    /// `Promise<RunStateSnapshot>`.
    #[wasm_bindgen(js_name = "describeWorkflow")]
    pub fn describe_workflow(&self, run_id: String) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let uuid = Uuid::parse_str(&run_id)
                .map_err(|e| JsValue::from_str(&format!("invalid run_id UUID: {e}")))?;
            let url = join_url(&endpoint, &format!("/v1/cp/describe/{uuid}"));
            let resp = get_envelope_response(&http, url, token.as_deref(), "describe").await?;
            let wire: RunStateSnapshotWire = decode_response_envelope(&resp, "describe")?;
            let snap = snapshot_from_wire(&wire)?;
            serde_wasm_bindgen::to_value(&snap)
                .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
        }))
    }

    /// Subscribe to the SSE stream of events for a single run. Returns
    /// a [`WasmControlPlaneSubscription`] handle whose `close()` method
    /// closes the underlying `EventSource`.
    ///
    /// **Auth note**: browsers' `EventSource` API does not let callers
    /// set custom headers, so `bearerToken` is appended to the URL as
    /// `?access_token=<token>` for SSE connections; the server-side
    /// auth middleware accepts either form. If no token is configured
    /// the URL is left untouched.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `run_id` is not a valid UUID or if
    /// the browser refuses to construct the underlying `EventSource`.
    #[wasm_bindgen(js_name = "subscribeRunEvents")]
    pub fn subscribe_run_events(
        &self,
        run_id: &str,
        on_event: &js_sys::Function,
        on_error: &js_sys::Function,
    ) -> Result<WasmControlPlaneSubscription, JsValue> {
        let uuid = Uuid::parse_str(run_id)
            .map_err(|e| JsValue::from_str(&format!("invalid run_id UUID: {e}")))?;
        let mut url = join_url(&self.endpoint, &format!("/v1/cp/events/{uuid}"));
        if let Some(ref tok) = self.bearer_token {
            use std::fmt::Write as _;
            let _ = write!(url, "?access_token={tok}");
        }
        open_event_source(&url, on_event, on_error, decode_run_event_js)
    }

    /// Subscribe to the SSE stream of all run events. See
    /// [`WasmControlPlaneClient::subscribe_run_events`] for the auth
    /// caveat.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the browser refuses to construct
    /// the underlying `EventSource`.
    #[wasm_bindgen(js_name = "subscribeAll")]
    pub fn subscribe_all(
        &self,
        on_event: &js_sys::Function,
        on_error: &js_sys::Function,
    ) -> Result<WasmControlPlaneSubscription, JsValue> {
        let mut url = join_url(&self.endpoint, "/v1/cp/events");
        if let Some(ref tok) = self.bearer_token {
            use std::fmt::Write as _;
            let _ = write!(url, "?access_token={tok}");
        }
        open_event_source(&url, on_event, on_error, decode_run_event_js)
    }

    /// Read-only accessor for the configured endpoint.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.endpoint.clone()
    }
}

// ---------------------------------------------------------------------------
// WasmControlPlaneWorker
// ---------------------------------------------------------------------------

/// HTTP/SSE control-plane worker. Equivalent to
/// [`blazen_controlplane::Worker`] but for the browser; assignments are
/// pushed via SSE, results / events / heartbeats are `POST`ed back.
///
/// Lifecycle:
/// 1. [`WasmControlPlaneWorker::connect`] performs the `register`
///    handshake and returns the wrapper.
/// 2. [`WasmControlPlaneWorker::run`] opens the SSE stream and drives
///    assignment dispatch; each incoming `ServerToWorker::Assignment`
///    invokes the user-supplied handler. The promise returned by `run`
///    resolves when [`WasmControlPlaneWorker::shutdown`] is called.
/// 3. [`WasmControlPlaneWorker::shutdown`] POSTs `/release` and closes
///    the `EventSource`.
#[wasm_bindgen(js_name = "ControlPlaneWorker")]
pub struct WasmControlPlaneWorker {
    endpoint: String,
    session_id: String,
    bearer_token: Option<String>,
    http: Arc<dyn HttpClient>,
    state: Rc<RefCell<WorkerState>>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmControlPlaneWorker {}
// SAFETY: WASM is single-threaded.
unsafe impl Sync for WasmControlPlaneWorker {}

struct WorkerState {
    subscription: Option<WasmControlPlaneSubscription>,
    closed: bool,
    /// Outstanding `request_input` calls awaiting a server
    /// `InputResponse` frame, keyed by `request_id`. The B2
    /// input-request round-trip resolves/rejects these from the worker
    /// SSE `onmessage` closure.
    pending_inputs: HashMap<String, PendingInput>,
}

/// State for a single in-flight `request_input` call. Holds the JS
/// `Promise` resolve/reject callbacks plus the `setTimeout` id (if a
/// timeout was requested) so the SSE arrival path can clear the timer.
struct PendingInput {
    /// `resolve(value)` of the Promise returned by `request_input`.
    resolve: js_sys::Function,
    /// `setTimeout` handle to cancel when the response arrives first.
    timeout_id: Option<f64>,
    /// Storage for the timeout closure so it outlives the timer; dropped
    /// when the entry is removed.
    _timeout_closure: Option<Closure<dyn FnMut()>>,
}

#[wasm_bindgen(js_class = "ControlPlaneWorker")]
impl WasmControlPlaneWorker {
    /// Register with the control plane at `endpoint` and return the
    /// connected worker handle. The returned promise resolves once the
    /// `register` POST succeeds; it does not yet open the SSE stream —
    /// call [`WasmControlPlaneWorker::run`] for that.
    #[wasm_bindgen]
    pub fn connect(endpoint: String, config: WasmWorkerConfig) -> js_sys::Promise {
        future_to_promise(SendFuture(async move {
            if endpoint.is_empty() {
                return Err(JsValue::from_str("endpoint must not be empty"));
            }
            if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                return Err(JsValue::from_str(
                    "endpoint must start with http:// or https://",
                ));
            }
            let http: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
            let hello = WorkerHello {
                envelope_version: ENVELOPE_VERSION,
                node_id: config.node_id.clone(),
                capabilities: config.capabilities.iter().map(Into::into).collect(),
                tags: config.tags.clone().unwrap_or_default(),
                admission: config
                    .admission
                    .as_ref()
                    .map_or(AdmissionModeWire::Reactive, Into::into),
                supported_envelope_versions: vec![ENVELOPE_VERSION],
                labels: std::collections::BTreeMap::new(),
                taints: Vec::new(),
                descriptors: Vec::new(),
                place: None,
            };

            let url = join_url(&endpoint, "/v1/cp/worker/register");
            let envelope = encode_envelope(&hello)?;
            let body = serde_json::to_vec(&envelope)
                .map_err(|e| JsValue::from_str(&format!("encode register envelope: {e}")))?;
            let mut request = HttpRequest::post(url).body(body);
            request.headers = auth_header(config.bearer_token.as_deref());
            let resp = http_send(&http, request).await?;
            http_ok(&resp, "register")?;

            let reg: RegisterResponse = serde_json::from_slice(&resp.body)
                .map_err(|e| JsValue::from_str(&format!("parse register response: {e}")))?;
            let welcome: Welcome = decode_envelope(&reg.welcome)?;
            if welcome.envelope_version > ENVELOPE_VERSION {
                return Err(JsValue::from_str(&format!(
                    "server envelope version {} > client {}",
                    welcome.envelope_version, ENVELOPE_VERSION
                )));
            }

            let worker = WasmControlPlaneWorker {
                endpoint,
                session_id: reg.session_id,
                bearer_token: config.bearer_token,
                http,
                state: Rc::new(RefCell::new(WorkerState {
                    subscription: None,
                    closed: false,
                    pending_inputs: HashMap::new(),
                })),
            };
            Ok(JsValue::from(worker))
        }))
    }

    /// Drive the assignment loop: open the SSE stream, decode incoming
    /// [`ServerToWorker`] frames, dispatch [`ServerToWorker::Assignment`]
    /// to `handler`, and POST the handler's result back. The returned
    /// promise resolves when [`WasmControlPlaneWorker::shutdown`] is
    /// called or the SSE stream errors out.
    ///
    /// The handler receives a [`WasmAssignment`] and is expected to
    /// return a Promise resolving to the workflow output (any
    /// JSON-serializable value) or reject with an error.
    #[wasm_bindgen]
    pub fn run(&self, handler: js_sys::Function) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let session_id = self.session_id.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        let state = Rc::clone(&self.state);

        future_to_promise(SendFuture(async move {
            let stream_url = build_stream_url(&endpoint, &session_id, token.as_deref());
            let source = EventSource::new(&stream_url)
                .map_err(|e| JsValue::from_str(&format!("worker EventSource: {e:?}")))?;
            let done_flag: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));
            let done_flag_err = Rc::clone(&done_flag);

            let on_message = build_worker_message_closure(
                handler,
                endpoint.clone(),
                session_id.clone(),
                token.clone(),
                Arc::clone(&http),
                Rc::clone(&state),
            );
            let on_error = Closure::wrap(Box::new(move |_evt: web_sys::Event| {
                *done_flag_err.borrow_mut() = true;
            }) as Box<dyn FnMut(web_sys::Event)>);

            source.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
            source.set_onerror(Some(on_error.as_ref().unchecked_ref()));

            state.borrow_mut().subscription = Some(WasmControlPlaneSubscription {
                source: Some(source),
                on_message: Some(on_message),
                on_error: Some(on_error),
            });

            // Spin until shutdown is called or the SSE errors. We
            // can't block a real OS thread in WASM, so we
            // co-operatively yield via a 250 ms setTimeout.
            loop {
                if state.borrow().closed || *done_flag.borrow() {
                    break;
                }
                sleep_ms(250).await;
            }
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Send a non-terminal event up to the control plane while a run
    /// is in-flight. `dataJson` is opaque bytes — the canonical encoding
    /// is `JSON.stringify(...).encode()`.
    #[wasm_bindgen(js_name = "emitEvent")]
    pub fn emit_event(
        &self,
        run_id: String,
        event_type: String,
        data_json: Vec<u8>,
        timestamp_ms: f64,
    ) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let session_id = self.session_id.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let uuid = Uuid::parse_str(&run_id)
                .map_err(|e| JsValue::from_str(&format!("invalid run_id: {e}")))?;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let ts = timestamp_ms.max(0.0) as u64;
            post_worker_event_at(
                &http,
                &endpoint,
                &session_id,
                token.as_deref(),
                uuid,
                event_type,
                data_json,
                ts,
            )
            .await?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Raise an `input.request` from a running assignment and block on
    /// the orchestrator's answer (the B2 input-request round-trip).
    ///
    /// This:
    /// 1. generates a `request_id` (UUID v4),
    /// 2. emits an `input.request` [`AssignmentEvent`] up to the control
    ///    plane carrying `{ requestId, prompt, metadata }`,
    /// 3. returns a `Promise` that resolves with the decoded JSON answer
    ///    once the matching [`ServerToWorker::InputResponse`] frame
    ///    arrives over the worker SSE stream.
    ///
    /// If `timeoutMs` is provided the Promise rejects after that many
    /// milliseconds if no response has arrived; the pending entry is
    /// dropped so a late response is ignored.
    #[wasm_bindgen(js_name = "requestInput")]
    pub fn request_input(
        &self,
        run_id: String,
        prompt: String,
        metadata: JsValue,
        timeout_ms: Option<f64>,
    ) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let session_id = self.session_id.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        let state = Rc::clone(&self.state);

        // Parse run_id and the metadata value up front so construction
        // errors reject synchronously (mirrors the early-return style of
        // the unary methods).
        let uuid = match Uuid::parse_str(&run_id) {
            Ok(u) => u,
            Err(e) => {
                return js_sys::Promise::reject(&JsValue::from_str(&format!(
                    "invalid run_id UUID: {e}"
                )));
            }
        };
        let metadata_value: serde_json::Value = if metadata.is_undefined() || metadata.is_null() {
            serde_json::Value::Null
        } else {
            match serde_wasm_bindgen::from_value(metadata) {
                Ok(v) => v,
                Err(e) => {
                    return js_sys::Promise::reject(&JsValue::from_str(&format!(
                        "decode metadata: {e}"
                    )));
                }
            }
        };

        let request_id = Uuid::new_v4().to_string();

        // Build the Promise the caller awaits. We capture its
        // resolve/reject callbacks into the pending map keyed by
        // request_id; the SSE InputResponse arm (or the timeout) fires
        // them later.
        let request_id_for_exec = request_id.clone();
        js_sys::Promise::new(&mut move |resolve, reject| {
            // Register the pending entry, wiring an optional timeout that
            // rejects + evicts the entry if the answer never arrives.
            let timeout_entry = timeout_ms.and_then(|ms| {
                install_input_timeout(&state, &request_id_for_exec, &reject, ms)
            });
            let (timeout_id, timeout_closure) = match timeout_entry {
                Some((id, cb)) => (Some(id), Some(cb)),
                None => (None, None),
            };
            state.borrow_mut().pending_inputs.insert(
                request_id_for_exec.clone(),
                PendingInput {
                    resolve: resolve.clone(),
                    timeout_id,
                    _timeout_closure: timeout_closure,
                },
            );

            // Emit the input.request event. If the POST fails, reject the
            // Promise and drop the pending entry so the caller isn't
            // wedged.
            let endpoint = endpoint.clone();
            let session_id = session_id.clone();
            let token = token.clone();
            let http = Arc::clone(&http);
            let state = Rc::clone(&state);
            let request_id = request_id_for_exec.clone();
            let prompt = prompt.clone();
            let metadata_value = metadata_value.clone();
            let reject = reject.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let data = serde_json::json!({
                    "request_id": request_id,
                    "prompt": prompt,
                    "metadata": metadata_value,
                });
                let data_json = match serde_json::to_vec(&data) {
                    Ok(b) => b,
                    Err(e) => {
                        evict_and_reject(
                            &state,
                            &request_id,
                            &reject,
                            &JsValue::from_str(&format!("encode input.request payload: {e}")),
                        );
                        return;
                    }
                };
                if let Err(e) = post_worker_event(
                    &http,
                    &endpoint,
                    &session_id,
                    token.as_deref(),
                    uuid,
                    "input.request",
                    data_json,
                )
                .await
                {
                    evict_and_reject(&state, &request_id, &reject, &e);
                }
            });
        })
    }

    /// POST a heartbeat. `inFlight` reflects how many assignments are
    /// currently running locally. The control plane uses this for
    /// liveness and admission accounting.
    #[wasm_bindgen]
    pub fn heartbeat(&self, in_flight: u32) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let session_id = self.session_id.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let payload = WorkerHeartbeat {
                envelope_version: ENVELOPE_VERSION,
                in_flight,
                queue_depth: 0,
                mem_mb: 0,
                cpu_pct: 0.0,
                admission_snapshot: None,
            };
            let url = join_url(&endpoint, &format!("/v1/cp/worker/{session_id}/heartbeat"));
            post_envelope(&http, url, token.as_deref(), &payload, "heartbeat").await?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Gracefully disconnect from the control plane. POSTs
    /// `/v1/cp/worker/{session}/release` and closes the SSE stream.
    #[wasm_bindgen]
    pub fn shutdown(&self) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let session_id = self.session_id.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        let state = Rc::clone(&self.state);
        future_to_promise(SendFuture(async move {
            let url = join_url(&endpoint, &format!("/v1/cp/worker/{session_id}/release"));
            let mut request = HttpRequest::post(url);
            request.headers = auth_header(token.as_deref());
            // We don't surface release errors — the worker is going
            // away regardless. If the server is gone, the connection
            // would already be torn down anyway.
            let _ = http.send(request).await;
            {
                let mut guard = state.borrow_mut();
                guard.closed = true;
                if let Some(mut sub) = guard.subscription.take() {
                    sub.close();
                }
            }
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Read-only accessor for the server-assigned session id.
    #[wasm_bindgen(getter, js_name = "sessionId")]
    #[must_use]
    pub fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Yield to the browser event loop for `ms` milliseconds. Used by the
/// SSE-driven worker loop in [`WasmControlPlaneWorker::run`] which
/// can't block a real thread.
async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        let Some(set_timeout) = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Function>().ok())
        else {
            let _ = resolve.call0(&JsValue::NULL);
            return;
        };
        let _ = set_timeout.call2(&JsValue::NULL, &resolve, &JsValue::from(ms));
    });
    let _ = JsFuture::from(promise).await;
}

/// Build the worker SSE stream URL, appending `?access_token=<token>`
/// when bearer auth is configured. The `EventSource` API doesn't let
/// callers set custom headers, so the server accepts both the
/// `Authorization` header (for unary requests) and an `access_token`
/// query parameter (for SSE).
fn build_stream_url(endpoint: &str, session_id: &str, token: Option<&str>) -> String {
    let mut url = join_url(endpoint, &format!("/v1/cp/worker/{session_id}/stream"));
    if let Some(tok) = token {
        use std::fmt::Write as _;
        let _ = write!(url, "?access_token={tok}");
    }
    url
}

/// Build the SSE `onmessage` closure for the worker loop. Extracted out
/// of [`WasmControlPlaneWorker::run`] purely to keep that method's line
/// count under clippy's `too_many_lines` threshold.
fn build_worker_message_closure(
    handler: js_sys::Function,
    endpoint: String,
    session_id: String,
    token: Option<String>,
    http: Arc<dyn HttpClient>,
    state: Rc<RefCell<WorkerState>>,
) -> Closure<dyn FnMut(web_sys::MessageEvent)> {
    Closure::wrap(Box::new(move |evt: web_sys::MessageEvent| {
        let Some(raw) = evt.data().as_string() else {
            return;
        };
        let Ok(env) = serde_json::from_str::<PostcardEnvelopeJson>(&raw) else {
            return;
        };
        let frame: ServerToWorker = match decode_envelope(&env) {
            Ok(v) => v,
            Err(_) => return,
        };
        match frame {
            ServerToWorker::Assignment(assignment) => {
                dispatch_assignment(
                    &assignment,
                    handler.clone(),
                    endpoint.clone(),
                    session_id.clone(),
                    token.clone(),
                    Arc::clone(&http),
                );
            }
            ServerToWorker::InputResponse(resp) => {
                resolve_input_response(&state, &resp);
            }
            ServerToWorker::Cancel(_)
            | ServerToWorker::Drain(_)
            | ServerToWorker::Welcome(_)
            | ServerToWorker::Offer(_)
            | ServerToWorker::Reject { .. }
            | ServerToWorker::KeyResponse(_) => {
                // Non-Assignment frames don't drive the handler;
                // Welcome was already consumed by the server-side
                // register flow. Cancel / Drain are advisory in v1
                // of the WASM binding (browser workers run a single
                // handler call at a time today). Future revisions
                // can wire an `AbortController` here. KeyResponse is
                // unreachable in v1 (this binding never issues
                // `RequestKey`) but is matched explicitly so envelope
                // v3 compiles cleanly.
            }
        }
    }) as Box<dyn FnMut(web_sys::MessageEvent)>)
}

/// Resolve the pending [`WasmControlPlaneWorker::request_input`] Promise
/// that matches an incoming [`ServerToWorker::InputResponse`] frame.
/// Decodes the JSON answer bytes to a `JsValue`, clears any pending
/// timeout, and invokes the stored `resolve` callback. Unknown
/// `request_id`s (e.g. a late response after a timeout already evicted
/// the entry) are ignored.
fn resolve_input_response(state: &Rc<RefCell<WorkerState>>, resp: &protocol::InputResponse) {
    let Some(pending) = state.borrow_mut().pending_inputs.remove(&resp.request_id) else {
        return;
    };
    if let Some(id) = pending.timeout_id {
        clear_timeout(id);
    }
    let value: serde_json::Value = if resp.response_json.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&resp.response_json).unwrap_or(serde_json::Value::Null)
    };
    let js_val = serde_wasm_bindgen::to_value(&value).unwrap_or(JsValue::NULL);
    let _ = pending.resolve.call1(&JsValue::NULL, &js_val);
    // `pending` (and its `_timeout_closure`) drops here, freeing the
    // timeout closure now that the timer has been cleared.
}

/// POST an [`AssignmentEvent`] to the worker event endpoint. Shared by
/// [`WasmControlPlaneWorker::emit_event`] and the B2 `request_input`
/// path. `timestamp_ms` is the caller-supplied event time.
#[allow(clippy::too_many_arguments)]
async fn post_worker_event_at(
    http: &Arc<dyn HttpClient>,
    endpoint: &str,
    session_id: &str,
    token: Option<&str>,
    run_id: Uuid,
    event_type: String,
    data_json: Vec<u8>,
    timestamp_ms: u64,
) -> Result<(), JsValue> {
    let payload = AssignmentEvent {
        envelope_version: ENVELOPE_VERSION,
        run_id,
        event_type,
        data_json,
        timestamp_ms,
    };
    let url = join_url(endpoint, &format!("/v1/cp/worker/{session_id}/event"));
    post_envelope(http, url, token, &payload, "event").await?;
    Ok(())
}

/// Like [`post_worker_event_at`] but stamps the event with the current
/// wall-clock time. Used by the B2 `request_input` emit path.
#[allow(clippy::too_many_arguments)]
async fn post_worker_event(
    http: &Arc<dyn HttpClient>,
    endpoint: &str,
    session_id: &str,
    token: Option<&str>,
    run_id: Uuid,
    event_type: &str,
    data_json: Vec<u8>,
) -> Result<(), JsValue> {
    post_worker_event_at(
        http,
        endpoint,
        session_id,
        token,
        run_id,
        event_type.to_owned(),
        data_json,
        now_ms(),
    )
    .await
}

/// Current wall-clock time in milliseconds since the Unix epoch, read
/// from JS `Date.now()`.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn now_ms() -> u64 {
    js_sys::Date::now().max(0.0) as u64
}

/// Remove a pending input entry and reject its Promise. Used when the
/// `input.request` emit fails or its payload can't be encoded.
fn evict_and_reject(
    state: &Rc<RefCell<WorkerState>>,
    request_id: &str,
    reject: &js_sys::Function,
    err: &JsValue,
) {
    if let Some(pending) = state.borrow_mut().pending_inputs.remove(request_id) {
        if let Some(id) = pending.timeout_id {
            clear_timeout(id);
        }
    }
    let _ = reject.call1(&JsValue::NULL, err);
}

/// Install a `setTimeout` that rejects the pending `request_input`
/// Promise and evicts its entry after `ms` milliseconds. Returns the
/// timeout id (so the resolve path can cancel it) plus the closure to
/// keep alive. Returns `None` if `setTimeout` is unavailable.
fn install_input_timeout(
    state: &Rc<RefCell<WorkerState>>,
    request_id: &str,
    reject: &js_sys::Function,
    ms: f64,
) -> Option<(f64, Closure<dyn FnMut()>)> {
    let global = js_sys::global();
    let set_timeout = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
        .ok()
        .and_then(|v| v.dyn_into::<js_sys::Function>().ok())?;

    let state = Rc::clone(state);
    let reject = reject.clone();
    let request_id_owned = request_id.to_owned();
    let cb = Closure::wrap(Box::new(move || {
        // Drop the pending entry (without re-clearing this fired timer)
        // and reject the awaiting Promise.
        let _ = state.borrow_mut().pending_inputs.remove(&request_id_owned);
        let _ = reject.call1(
            &JsValue::NULL,
            &JsValue::from_str("request_input timed out waiting for InputResponse"),
        );
    }) as Box<dyn FnMut()>);

    let id = set_timeout
        .call2(
            &JsValue::NULL,
            cb.as_ref().unchecked_ref(),
            &JsValue::from_f64(ms),
        )
        .ok()
        .and_then(|v| v.as_f64())?;
    Some((id, cb))
}

/// Cancel a `setTimeout` previously created by [`install_input_timeout`].
fn clear_timeout(id: f64) {
    let global = js_sys::global();
    if let Some(clear) = js_sys::Reflect::get(&global, &JsValue::from_str("clearTimeout"))
        .ok()
        .and_then(|v| v.dyn_into::<js_sys::Function>().ok())
    {
        let _ = clear.call1(&JsValue::NULL, &JsValue::from_f64(id));
    }
}

/// Spawn the assignment handler for one incoming `Assignment` frame.
/// Decoupled from [`WasmControlPlaneWorker::run`] so the SSE-message
/// closure stays small and the `run` method body stays under clippy's
/// `too_many_lines` threshold.
fn dispatch_assignment(
    assignment: &protocol::Assignment,
    handler: js_sys::Function,
    endpoint: String,
    session_id: String,
    token: Option<String>,
    http: Arc<dyn HttpClient>,
) {
    let wasm_assignment = WasmAssignment {
        run_id: assignment.run_id.to_string(),
        workflow_name: assignment.workflow_name.clone(),
        input_json: assignment.input_json.clone(),
        workflow_version: assignment.workflow_version,
        #[allow(clippy::cast_precision_loss)]
        deadline_ms: assignment.deadline_ms.map(|v| v as f64),
        attempt: assignment.attempt,
    };
    let run_id = assignment.run_id;
    let Ok(js_arg) = serde_wasm_bindgen::to_value(&wasm_assignment) else {
        return;
    };
    wasm_bindgen_futures::spawn_local(async move {
        let (status, output_json, error) = await_handler_result(&handler, &js_arg).await;
        let result = AssignmentResult {
            envelope_version: ENVELOPE_VERSION,
            run_id,
            output_json,
            status,
            error,
        };
        let url = join_url(&endpoint, &format!("/v1/cp/worker/{session_id}/result"));
        let _ = post_envelope(&http, url, token.as_deref(), &result, "result").await;
    });
}

/// Invoke the JS handler and reduce its result (sync, Promise, or
/// rejection) to an [`AssignmentStatus`] tuple ready to ship up as an
/// [`AssignmentResult`]. Helper for [`dispatch_assignment`].
async fn await_handler_result(
    handler: &js_sys::Function,
    arg: &JsValue,
) -> (AssignmentStatus, Vec<u8>, Option<String>) {
    let raw_result = handler.call1(&JsValue::NULL, arg);
    let resolved = match raw_result {
        Ok(val) => {
            if val.has_type::<js_sys::Promise>() {
                let p: js_sys::Promise = val.unchecked_into();
                JsFuture::from(p).await
            } else {
                Ok(val)
            }
        }
        Err(e) => Err(e),
    };
    match resolved {
        Ok(v) => {
            let value: serde_json::Value =
                serde_wasm_bindgen::from_value(v).unwrap_or(serde_json::Value::Null);
            let bytes = serde_json::to_vec(&value).unwrap_or_else(|_| b"null".to_vec());
            (AssignmentStatus::Completed, bytes, None)
        }
        Err(e) => {
            let msg = e.as_string().unwrap_or_else(|| "handler rejected".into());
            (AssignmentStatus::Failed, Vec::new(), Some(msg))
        }
    }
}

// Unit-test coverage for the pure-data helpers in this module lives
// in the `tests/controlplane.rs` integration suite — wasm-bindgen-test
// only permits one `wasm_bindgen_test_configure!` per built binary, so
// keeping every browser test in a single crate avoids the
// "Cannot define export over existing namespace" error reported by
// the test runner.
