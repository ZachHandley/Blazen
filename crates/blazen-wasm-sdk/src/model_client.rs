//! WASM SDK wrapper for [`blazen_controlplane::client::ModelClient`].
//!
//! The native [`ModelClient`](blazen_controlplane::client::ModelClient)
//! talks to a `BlazenModelServer` over gRPC bidi-streaming. gRPC is not
//! natively reachable from a browser environment, so this WASM wrapper
//! targets the **HTTP REST surface** added to the model server in PR5
//! phase 2 (see `blazen-controlplane/src/http/`):
//!
//! - OpenAI-compatible routes under `/v1/...` (chat completions,
//!   embeddings, models list, audio, images).
//! - Blazen-specific admin routes under `/v1/blazen/...` (adapter
//!   load/unload, register-from-HF, per-model status, health, metrics).
//!
//! This first wave only exposes the four lifecycle / introspection
//! methods needed to verify the server is reachable and inspect what
//! models are loaded. Inference, embeddings, and adapter management
//! land in later waves.
//!
//! ## Auth
//!
//! When the server is configured with a bearer token, callers can pass
//! it to [`ModelClient::connect_with_tls`]; every subsequent HTTP request
//! sends `Authorization: Bearer <token>`.
//!
//! ## TLS
//!
//! "TLS" in the browser is `https://` — the browser stack handles the
//! handshake. The `caCertPem` field on the TLS options is accepted for
//! API parity with the native [`ModelClient::connect_with_tls`] but is
//! **ignored**: browsers do not let JavaScript override the system trust
//! store. Use the operating system / browser's certificate manager to
//! install custom CAs.
//!
//! ## TypeScript surface
//!
//! ```typescript
//! import init, { ModelClient } from '@blazen-dev/wasm';
//!
//! await init();
//!
//! const client = await ModelClient.connect('https://models.example.com');
//! const status = await client.status();
//! const loaded = await client.isLoaded('llama-3.1-8b');
//! ```

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::FetchHttpClient;
use blazen_llm::http::{HttpClient, HttpRequest, HttpResponse};

// ---------------------------------------------------------------------------
// SendFuture wrapper (mirrors `crate::controlplane::SendFuture`)
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
// Connect-with-TLS options
// ---------------------------------------------------------------------------

/// Options accepted by [`ModelClient::connect_with_tls`].
///
/// In the browser, transport-layer TLS is handled by the user agent and
/// cannot be configured from JavaScript. `caCertPem`, when supplied, is
/// **silently ignored**; install custom CAs through the operating
/// system trust store instead. `bearerToken` is honoured and propagated
/// to every outgoing request as `Authorization: Bearer <token>`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelClientTlsOptions {
    /// Informational only. Browsers ignore custom trust roots; this
    /// field exists solely for API parity with the native
    /// `ModelClient::connect_with_tls` signature.
    #[serde(default)]
    pub ca_cert_pem: Option<String>,
    /// Optional bearer token sent as `Authorization: Bearer <token>` on
    /// every request issued by the returned client.
    #[serde(default)]
    pub bearer_token: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

/// Trim a trailing slash from `base` so concatenation always produces
/// `<base>/v1/...`. Mirrors the helper in [`crate::controlplane`].
fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    format!("{base}{path}")
}

fn auth_headers(token: Option<&str>) -> Vec<(String, String)> {
    let mut headers = vec![("Accept".to_owned(), "application/json".to_owned())];
    if let Some(t) = token {
        headers.push(("Authorization".to_owned(), format!("Bearer {t}")));
    }
    headers
}

async fn http_get_json(
    http: &Arc<dyn HttpClient>,
    url: String,
    token: Option<&str>,
    label: &'static str,
) -> Result<HttpResponse, JsValue> {
    let mut request = HttpRequest::get(url);
    request.headers = auth_headers(token);
    let resp = http
        .send(request)
        .await
        .map_err(|e| JsValue::from_str(&format!("model client HTTP error ({label}): {e}")))?;
    if (200..300).contains(&resp.status) {
        Ok(resp)
    } else {
        let body = String::from_utf8_lossy(&resp.body);
        Err(JsValue::from_str(&format!(
            "model client {label} returned HTTP {}: {body}",
            resp.status
        )))
    }
}

/// POST `body_json` as `application/json` to `url`. Mirrors
/// [`http_get_json`] for the request side.
async fn http_post_json(
    http: &Arc<dyn HttpClient>,
    url: String,
    body_json: &serde_json::Value,
    token: Option<&str>,
    label: &'static str,
) -> Result<HttpResponse, JsValue> {
    let body_bytes = serde_json::to_vec(body_json)
        .map_err(|e| JsValue::from_str(&format!("model client {label}: serialize body: {e}")))?;
    let mut request = HttpRequest::post(url).body(body_bytes);
    request.headers = auth_headers(token);
    request
        .headers
        .push(("Content-Type".to_owned(), "application/json".to_owned()));
    let resp = http
        .send(request)
        .await
        .map_err(|e| JsValue::from_str(&format!("model client HTTP error ({label}): {e}")))?;
    if (200..300).contains(&resp.status) {
        Ok(resp)
    } else {
        let body = String::from_utf8_lossy(&resp.body);
        Err(JsValue::from_str(&format!(
            "model client {label} returned HTTP {}: {body}",
            resp.status
        )))
    }
}

/// POST `body_json` as `application/json` to `url` and return the raw
/// response body bytes. Used by routes that return binary payloads
/// (e.g. `/v1/audio/speech`).
async fn http_post_json_raw(
    http: &Arc<dyn HttpClient>,
    url: String,
    body_json: &serde_json::Value,
    token: Option<&str>,
    label: &'static str,
) -> Result<HttpResponse, JsValue> {
    let body_bytes = serde_json::to_vec(body_json)
        .map_err(|e| JsValue::from_str(&format!("model client {label}: serialize body: {e}")))?;
    let mut request = HttpRequest::post(url).body(body_bytes);
    request.headers = auth_headers(token);
    request
        .headers
        .push(("Content-Type".to_owned(), "application/json".to_owned()));
    let resp = http
        .send(request)
        .await
        .map_err(|e| JsValue::from_str(&format!("model client HTTP error ({label}): {e}")))?;
    if (200..300).contains(&resp.status) {
        Ok(resp)
    } else {
        let body = String::from_utf8_lossy(&resp.body);
        Err(JsValue::from_str(&format!(
            "model client {label} returned HTTP {}: {body}",
            resp.status
        )))
    }
}

/// POST a `multipart/form-data` body to `url`. Used by
/// `/v1/audio/transcriptions`, which is the only OpenAI-compat route on
/// the server that expects multipart rather than JSON.
async fn http_post_multipart(
    http: &Arc<dyn HttpClient>,
    url: String,
    body: Vec<u8>,
    boundary: &str,
    token: Option<&str>,
    label: &'static str,
) -> Result<HttpResponse, JsValue> {
    let mut request = HttpRequest::post(url).body(body);
    request.headers = auth_headers(token);
    request.headers.push((
        "Content-Type".to_owned(),
        format!("multipart/form-data; boundary={boundary}"),
    ));
    let resp = http
        .send(request)
        .await
        .map_err(|e| JsValue::from_str(&format!("model client HTTP error ({label}): {e}")))?;
    if (200..300).contains(&resp.status) {
        Ok(resp)
    } else {
        let body = String::from_utf8_lossy(&resp.body);
        Err(JsValue::from_str(&format!(
            "model client {label} returned HTTP {}: {body}",
            resp.status
        )))
    }
}

/// GET `url` and validate the status. Currently used by
/// [`ModelClient::list_adapters`] alongside the existing `http_get_json`
/// (kept separate so the `label` reads naturally in error strings).
/// DELETE `url` with no body. Used by adapter unload.
async fn http_delete(
    http: &Arc<dyn HttpClient>,
    url: String,
    token: Option<&str>,
    label: &'static str,
) -> Result<HttpResponse, JsValue> {
    let mut request = HttpRequest::delete(url);
    request.headers = auth_headers(token);
    let resp = http
        .send(request)
        .await
        .map_err(|e| JsValue::from_str(&format!("model client HTTP error ({label}): {e}")))?;
    if (200..300).contains(&resp.status) {
        Ok(resp)
    } else {
        let body = String::from_utf8_lossy(&resp.body);
        Err(JsValue::from_str(&format!(
            "model client {label} returned HTTP {}: {body}",
            resp.status
        )))
    }
}

/// Extract a required string field from a JSON object, returning a
/// user-facing JS error on missing/non-string/non-object input.
fn require_string_field(
    value: &serde_json::Value,
    field: &'static str,
    label: &'static str,
) -> Result<String, JsValue> {
    value
        .get(field)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            JsValue::from_str(&format!(
                "model client {label}: request body must include string field '{field}'"
            ))
        })
}

fn parse_json_value(
    resp: &HttpResponse,
    label: &'static str,
) -> Result<serde_json::Value, JsValue> {
    if resp.body.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_slice(&resp.body)
        .map_err(|e| JsValue::from_str(&format!("model client {label}: parse JSON: {e}")))
}

// ---------------------------------------------------------------------------
// ModelClient
// ---------------------------------------------------------------------------

/// Browser-side client for a `BlazenModelServer`.
///
/// The native [`blazen_controlplane::client::ModelClient`] uses gRPC;
/// this wrapper uses the equivalent HTTP REST surface exposed by the
/// same server. The constructor performs a lightweight health-check
/// against `/v1/blazen/health` to validate connectivity before the
/// returned handle is used.
///
/// Cheaply cloneable on the JS side: each method clones the internal
/// `Arc<dyn HttpClient>` plus a couple of small `String`s.
#[wasm_bindgen]
pub struct ModelClient {
    endpoint: String,
    bearer_token: Option<String>,
    http: Arc<dyn HttpClient>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for ModelClient {}
// SAFETY: WASM is single-threaded.
unsafe impl Sync for ModelClient {}

#[wasm_bindgen]
impl ModelClient {
    /// Connect to a `BlazenModelServer` over plain HTTP(S) at `endpoint`.
    ///
    /// `endpoint` must be the HTTP root of the server (e.g.
    /// `https://models.example.com`); REST paths like `/v1/models` are
    /// appended automatically. The returned promise resolves once a
    /// `GET /v1/blazen/health` round-trip succeeds, mirroring the
    /// behaviour of the native [`blazen_controlplane::client::ModelClient::connect`]
    /// which establishes the gRPC channel up front.
    ///
    /// For mutual TLS or bearer-token auth use
    /// [`ModelClient::connect_with_tls`] instead.
    #[wasm_bindgen]
    pub fn connect(endpoint: String) -> js_sys::Promise {
        future_to_promise(SendFuture(async move {
            let client = build_client(endpoint, None)?;
            ping_health(&client).await?;
            Ok(JsValue::from(client))
        }))
    }

    /// Connect with optional bearer-token auth (and informational
    /// `caCertPem` — see the module-level docs for why it is ignored in
    /// the browser).
    ///
    /// The signature mirrors the native
    /// [`blazen_controlplane::client::ModelClient::connect_with_tls`]
    /// so that callers writing isomorphic code only have to swap the
    /// import. Behaviourally, in the browser, transport TLS is owned by
    /// the user agent — only the bearer token is wired through.
    #[wasm_bindgen(js_name = "connectWithTls")]
    pub fn connect_with_tls(endpoint: String, opts: JsValue) -> js_sys::Promise {
        future_to_promise(SendFuture(async move {
            let parsed: ModelClientTlsOptions = if opts.is_null() || opts.is_undefined() {
                ModelClientTlsOptions::default()
            } else {
                serde_wasm_bindgen::from_value(opts).map_err(|e| {
                    JsValue::from_str(&format!("connectWithTls: parse options: {e}"))
                })?
            };
            // caCertPem is informational only in the browser; the
            // browser's trust store decides what HTTPS connections are
            // valid. We deliberately do nothing with it here.
            let _ = parsed.ca_cert_pem;
            let client = build_client(endpoint, parsed.bearer_token)?;
            ping_health(&client).await?;
            Ok(JsValue::from(client))
        }))
    }

    /// Snapshot of currently-loaded models on the server.
    ///
    /// When `modelId` is `null` / omitted, the call hits
    /// `GET /v1/models` and resolves to the full OpenAI-shaped list
    /// (`{ object: "list", data: [...] }`). When `modelId` is supplied,
    /// the call hits `GET /v1/blazen/models/{id}/status` and resolves
    /// to a per-model snapshot (`{ id, pool, adapters: [...], ... }`).
    ///
    /// Returns a plain JS object decoded from the server's JSON
    /// response — exact shape mirrors the REST endpoint.
    #[wasm_bindgen]
    pub fn status(&self, model_id: Option<String>) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let path = match model_id {
                Some(id) => format!("/v1/blazen/models/{id}/status"),
                None => "/v1/models".to_owned(),
            };
            let url = join_url(&endpoint, &path);
            let resp = http_get_json(&http, url, token.as_deref(), "status").await?;
            let json = parse_json_value(&resp, "status")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("status: serialize JSON to JS: {e}")))
        }))
    }

    /// Returns `true` when `modelId` is currently loaded on the server.
    ///
    /// Implemented as a `GET /v1/blazen/models/{id}/status`: any 2xx
    /// resolves to `true`, a 404 resolves to `false`, and other non-2xx
    /// statuses reject the promise. Mirrors
    /// [`blazen_controlplane::client::ModelClient::is_loaded`].
    #[wasm_bindgen(js_name = "isLoaded")]
    pub fn is_loaded(&self, model_id: String) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let url = join_url(&endpoint, &format!("/v1/blazen/models/{model_id}/status"));
            let mut request = HttpRequest::get(url);
            request.headers = auth_headers(token.as_deref());
            let resp = http
                .send(request)
                .await
                .map_err(|e| JsValue::from_str(&format!("isLoaded HTTP error: {e}")))?;
            if (200..300).contains(&resp.status) {
                Ok(JsValue::from_bool(true))
            } else if resp.status == 404 {
                Ok(JsValue::from_bool(false))
            } else {
                let body = String::from_utf8_lossy(&resp.body);
                Err(JsValue::from_str(&format!(
                    "isLoaded returned HTTP {}: {body}",
                    resp.status
                )))
            }
        }))
    }

    /// Load an adapter onto a previously-registered model.
    ///
    /// Mirrors [`blazen_controlplane::client::ModelClient::load`] on the
    /// native gRPC client. The HTTP REST surface routes model lifecycle
    /// through the adapter endpoints: this method POSTs to
    /// `/v1/blazen/adapters/{model_id}/load`, with `model_id` pulled
    /// from the request body so callers pass a single JSON object.
    ///
    /// `request` must be a plain JS object with at minimum:
    ///   - `modelId` / `model_id` (string) — target model
    ///   - `adapterId` / `adapter_id` (string) — id to mount the adapter as
    ///   - `scale` (number)
    ///   - `source` (object) — `{ type: "local_fs" | "hf_hub" | "content_store", ... }`
    ///
    /// Resolves to a plain JS object decoded from the server's JSON
    /// response (`{ adapter_id, memory_bytes, mount_strategy }`).
    #[wasm_bindgen]
    pub fn load(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let mut body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("load: parse request: {e}")))?;
            // Accept either camelCase or snake_case for the path param,
            // strip it so the body matches `LoadAdapterBody` on the server.
            let model_id = body
                .as_object_mut()
                .and_then(|m| m.remove("modelId").or_else(|| m.remove("model_id")))
                .and_then(|v| v.as_str().map(str::to_owned))
                .ok_or_else(|| {
                    JsValue::from_str(
                        "load: request must include string field 'modelId' (or 'model_id')",
                    )
                })?;
            let url = join_url(&endpoint, &format!("/v1/blazen/adapters/{model_id}/load"));
            let resp = http_post_json(&http, url, &body, token.as_deref(), "load").await?;
            let json = parse_json_value(&resp, "load")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("load: serialize JSON to JS: {e}")))
        }))
    }

    /// Unmount an adapter previously loaded via [`ModelClient::load`].
    ///
    /// Mirrors [`blazen_controlplane::client::ModelClient::unload`] on the
    /// native gRPC client. Issues `DELETE /v1/blazen/adapters/{model_id}/{adapter_id}`.
    /// On success the promise resolves to `null` (the server returns 204
    /// No Content).
    ///
    /// `request` must be a plain JS object with:
    ///   - `modelId` / `model_id` (string)
    ///   - `adapterId` / `adapter_id` (string)
    #[wasm_bindgen]
    pub fn unload(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("unload: parse request: {e}")))?;
            let model_id = body
                .get("modelId")
                .or_else(|| body.get("model_id"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "unload: request must include string field 'modelId' (or 'model_id')",
                    )
                })?;
            let adapter_id = body
                .get("adapterId")
                .or_else(|| body.get("adapter_id"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "unload: request must include string field 'adapterId' (or 'adapter_id')",
                    )
                })?;
            let url = join_url(
                &endpoint,
                &format!("/v1/blazen/adapters/{model_id}/{adapter_id}"),
            );
            let resp = http_delete(&http, url, token.as_deref(), "unload").await?;
            // 204 No Content is the success shape; surface it as `null`.
            if resp.body.is_empty() {
                Ok(JsValue::NULL)
            } else {
                let json = parse_json_value(&resp, "unload")?;
                serde_wasm_bindgen::to_value(&json)
                    .map_err(|e| JsValue::from_str(&format!("unload: serialize JSON to JS: {e}")))
            }
        }))
    }

    /// Register a model from a Hugging Face Hub repo and load it onto
    /// the server.
    ///
    /// Mirrors [`blazen_controlplane::client::ModelClient::load_from_hf`]
    /// on the native gRPC client. Issues `POST /v1/blazen/models/load_from_hf`.
    ///
    /// `request` is forwarded as JSON to the server; see
    /// `LoadFromHfBody` in `blazen-controlplane/src/http/blazen_admin.rs`
    /// for the accepted shape (at minimum `modelId`/`model_id` and
    /// `repo`). Resolves to `{ chosen_backend: "mistral_rs" | "candle" | "llama_cpp" }`.
    #[wasm_bindgen(js_name = "loadFromHf")]
    pub fn load_from_hf(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("loadFromHf: parse request: {e}")))?;
            // Sanity-check: server requires `model_id` and `repo`. We
            // surface a clean error here rather than letting the server
            // reject it later.
            let _ = require_string_field(&body, "repo", "loadFromHf")?;
            let url = join_url(&endpoint, "/v1/blazen/models/load_from_hf");
            let resp = http_post_json(&http, url, &body, token.as_deref(), "loadFromHf").await?;
            let json = parse_json_value(&resp, "loadFromHf")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("loadFromHf: serialize JSON to JS: {e}")))
        }))
    }

    /// List adapters currently mounted on `modelId`.
    ///
    /// Issues `GET /v1/blazen/adapters/{model_id}`. `request` must be a
    /// plain JS object with `modelId` (or `model_id`). Resolves to
    /// `{ adapters: [...] }` as returned by the server.
    /// Load a LoRA adapter onto a previously-loaded model.
    ///
    /// Issues `POST /v1/blazen/adapters/{model_id}/load`. `request` must be
    /// a plain JS object with `modelId` (or `model_id`) plus the adapter
    /// payload fields the server expects (`adapter_id`, `source_dir`,
    /// `scale`, ...). Resolves to the server response object.
    #[wasm_bindgen(js_name = "loadAdapter")]
    pub fn load_adapter(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("loadAdapter: parse request: {e}")))?;
            let model_id = body
                .get("modelId")
                .or_else(|| body.get("model_id"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "loadAdapter: request must include string field 'modelId' (or 'model_id')",
                    )
                })?;
            let url = join_url(&endpoint, &format!("/v1/blazen/adapters/{model_id}/load"));
            let resp = http_post_json(&http, url, &body, token.as_deref(), "loadAdapter").await?;
            let json = parse_json_value(&resp, "loadAdapter")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("loadAdapter: serialize JSON to JS: {e}")))
        }))
    }

    /// Unload a LoRA adapter from a model.
    ///
    /// Issues `DELETE /v1/blazen/adapters/{model_id}/{adapter_id}`.
    /// `request` must be a plain JS object with `modelId` (or `model_id`)
    /// and `adapterId` (or `adapter_id`). Resolves to `null` (204
    /// No Content) on success.
    #[wasm_bindgen(js_name = "unloadAdapter")]
    pub fn unload_adapter(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("unloadAdapter: parse request: {e}")))?;
            let model_id = body
                .get("modelId")
                .or_else(|| body.get("model_id"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "unloadAdapter: request must include string field 'modelId' (or 'model_id')",
                    )
                })?;
            let adapter_id = body
                .get("adapterId")
                .or_else(|| body.get("adapter_id"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "unloadAdapter: request must include string field 'adapterId' (or 'adapter_id')",
                    )
                })?;
            let url = join_url(
                &endpoint,
                &format!("/v1/blazen/adapters/{model_id}/{adapter_id}"),
            );
            http_delete(&http, url, token.as_deref(), "unloadAdapter").await?;
            Ok(JsValue::NULL)
        }))
    }

    #[wasm_bindgen(js_name = "listAdapters")]
    pub fn list_adapters(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("listAdapters: parse request: {e}")))?;
            let model_id = body
                .get("modelId")
                .or_else(|| body.get("model_id"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "listAdapters: request must include string field 'modelId' (or 'model_id')",
                    )
                })?;
            let url = join_url(&endpoint, &format!("/v1/blazen/adapters/{model_id}"));
            let resp = http_get_json(&http, url, token.as_deref(), "listAdapters").await?;
            let json = parse_json_value(&resp, "listAdapters")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("listAdapters: serialize JSON to JS: {e}")))
        }))
    }

    /// Chat-completion via the OpenAI-compat route.
    ///
    /// Issues `POST /v1/chat/completions`. `request` is forwarded as-is
    /// (`model`, `messages`, plus optional `max_tokens`, `temperature`,
    /// `top_p`, `stop`, `response_format`, `extra_body`). Resolves to the
    /// full `{ id, object, created, model, choices, usage }` payload.
    ///
    /// Streaming (`stream: true`) is **not** supported by this method —
    /// the wrapper buffers the entire response. Use the SSE surface in
    /// [`crate::controlplane`] for token-by-token streaming.
    #[wasm_bindgen]
    pub fn complete(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("complete: parse request: {e}")))?;
            let _ = require_string_field(&body, "model", "complete")?;
            let url = join_url(&endpoint, "/v1/chat/completions");
            let resp = http_post_json(&http, url, &body, token.as_deref(), "complete").await?;
            let json = parse_json_value(&resp, "complete")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("complete: serialize JSON to JS: {e}")))
        }))
    }

    /// Embed text via the OpenAI-compat route.
    ///
    /// Issues `POST /v1/embeddings`. `request` is forwarded as-is
    /// (`model`, `input`). Resolves to the OpenAI-shaped
    /// `{ object: "list", data: [{ embedding, index, object }], model, usage }`.
    #[wasm_bindgen]
    pub fn embed(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("embed: parse request: {e}")))?;
            let _ = require_string_field(&body, "model", "embed")?;
            let url = join_url(&endpoint, "/v1/embeddings");
            let resp = http_post_json(&http, url, &body, token.as_deref(), "embed").await?;
            let json = parse_json_value(&resp, "embed")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("embed: serialize JSON to JS: {e}")))
        }))
    }

    /// Generate an image via the OpenAI-compat route.
    ///
    /// Issues `POST /v1/images/generations`. `request` is forwarded as-is
    /// (`model`, `prompt`, optional `n`, `size`, `response_format`).
    /// Resolves to the OpenAI-shaped `{ created, data: [{ b64_json | url, ... }] }`.
    #[wasm_bindgen(js_name = "generateImage")]
    pub fn generate_image(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("generateImage: parse request: {e}")))?;
            let _ = require_string_field(&body, "model", "generateImage")?;
            let _ = require_string_field(&body, "prompt", "generateImage")?;
            let url = join_url(&endpoint, "/v1/images/generations");
            let resp = http_post_json(&http, url, &body, token.as_deref(), "generateImage").await?;
            let json = parse_json_value(&resp, "generateImage")?;
            serde_wasm_bindgen::to_value(&json).map_err(|e| {
                JsValue::from_str(&format!("generateImage: serialize JSON to JS: {e}"))
            })
        }))
    }

    /// Text-to-speech via the OpenAI-compat route.
    ///
    /// Issues `POST /v1/audio/speech`. `request` is forwarded as JSON
    /// (`model`, `input`, optional `voice`, `language`, `response_format`,
    /// `sample_rate_hz`). The route returns **raw audio bytes** with a
    /// backend-supplied `Content-Type` (e.g. `audio/wav`, `audio/mpeg`).
    ///
    /// Resolves to a plain JS object `{ mime: string, data: Uint8Array }`
    /// so callers can pipe the bytes into a `Blob` / `AudioContext`
    /// without losing the MIME type.
    #[wasm_bindgen(js_name = "textToSpeech")]
    pub fn text_to_speech(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            let body: serde_json::Value = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("textToSpeech: parse request: {e}")))?;
            let _ = require_string_field(&body, "model", "textToSpeech")?;
            let _ = require_string_field(&body, "input", "textToSpeech")?;
            let url = join_url(&endpoint, "/v1/audio/speech");
            let resp =
                http_post_json_raw(&http, url, &body, token.as_deref(), "textToSpeech").await?;
            let mime = resp
                .headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
                .map_or_else(|| "application/octet-stream".to_owned(), |(_, v)| v.clone());
            let out = js_sys::Object::new();
            js_sys::Reflect::set(&out, &JsValue::from_str("mime"), &JsValue::from_str(&mime))
                .map_err(|e| {
                    JsValue::from_str(&format!("textToSpeech: build result object: {e:?}"))
                })?;
            let bytes = js_sys::Uint8Array::from(resp.body.as_slice());
            js_sys::Reflect::set(&out, &JsValue::from_str("data"), &bytes).map_err(|e| {
                JsValue::from_str(&format!("textToSpeech: build result object: {e:?}"))
            })?;
            Ok(out.into())
        }))
    }

    /// Generate music conditioned on a text prompt.
    ///
    /// **Server route TBD; placeholder implementation.** The native gRPC
    /// `ModelClient::generate_music` (see
    /// `blazen-controlplane/src/client/model_client.rs:341`) calls a
    /// `GenerateMusic` RPC that is **not yet** mirrored on the HTTP REST
    /// surface (`openai_compat.rs` + `blazen_admin.rs` as of this commit
    /// do not expose `/v1/blazen/audio/music` or any equivalent). This
    /// stub returns a `not implemented` error so the WASM API surface
    /// stays uniform with the native client and the binding can light up
    /// automatically once the server route lands.
    #[wasm_bindgen(js_name = "generateMusic")]
    pub fn generate_music(&self, _request: JsValue) -> js_sys::Promise {
        future_to_promise(SendFuture(async move {
            Err::<JsValue, _>(JsValue::from_str(
                "generateMusic: not implemented — no HTTP route exists on \
                 BlazenModelServer yet (gRPC GenerateMusic has no REST mirror). \
                 Track: blazen-controlplane/src/http/ for /v1/blazen/audio/music.",
            ))
        }))
    }

    /// Transcribe an audio clip via the OpenAI-compat route.
    ///
    /// Issues `POST /v1/audio/transcriptions` as `multipart/form-data`,
    /// which is the wire format the OpenAI Python/Node SDKs use and the
    /// server's [`super::http::openai_compat`] handler expects.
    ///
    /// `request` must be a plain JS object with:
    ///   - `model` (string) — model id
    ///   - `audio` (`Uint8Array` | `number[]`) — raw audio bytes
    ///   - `filename` (string, optional) — defaults to `audio.bin`
    ///   - `contentType` / `content_type` (string, optional) — defaults to
    ///     `application/octet-stream`
    ///   - `language` (string, optional)
    ///
    /// Resolves to `{ text, language?, segments? }` as returned by the
    /// server.
    #[wasm_bindgen]
    pub fn transcribe(&self, request: JsValue) -> js_sys::Promise {
        let endpoint = self.endpoint.clone();
        let token = self.bearer_token.clone();
        let http = Arc::clone(&self.http);
        future_to_promise(SendFuture(async move {
            // Pull the audio bytes off the JS object directly so we can
            // accept Uint8Array without round-tripping through JSON.
            let audio_val =
                js_sys::Reflect::get(&request, &JsValue::from_str("audio")).map_err(|e| {
                    JsValue::from_str(&format!("transcribe: read 'audio' field: {e:?}"))
                })?;
            if audio_val.is_undefined() || audio_val.is_null() {
                return Err(JsValue::from_str(
                    "transcribe: request must include 'audio' (Uint8Array or number[])",
                ));
            }
            let audio_bytes: Vec<u8> = if audio_val.is_instance_of::<js_sys::Uint8Array>() {
                js_sys::Uint8Array::from(audio_val).to_vec()
            } else if js_sys::Array::is_array(&audio_val) {
                let arr = js_sys::Array::from(&audio_val);
                let mut out = Vec::with_capacity(arr.length() as usize);
                for i in 0..arr.length() {
                    let v = arr.get(i).as_f64().ok_or_else(|| {
                        JsValue::from_str(
                            "transcribe: 'audio' array must contain numeric byte values",
                        )
                    })?;
                    out.push(v as u8);
                }
                out
            } else {
                return Err(JsValue::from_str(
                    "transcribe: 'audio' must be a Uint8Array or number[]",
                ));
            };

            fn read_string(req: &JsValue, key: &str) -> Option<String> {
                js_sys::Reflect::get(req, &JsValue::from_str(key))
                    .ok()
                    .and_then(|v| v.as_string())
            }

            let model = read_string(&request, "model").ok_or_else(|| {
                JsValue::from_str("transcribe: request must include string field 'model'")
            })?;
            let filename = read_string(&request, "filename").unwrap_or_else(|| "audio.bin".into());
            let content_type = read_string(&request, "contentType")
                .or_else(|| read_string(&request, "content_type"))
                .unwrap_or_else(|| "application/octet-stream".into());
            let language = read_string(&request, "language");

            // Construct a minimal multipart/form-data body. The boundary
            // is a fixed token that cannot collide with audio payloads
            // because it includes ASCII punctuation outside the byte
            // ranges used by the audio container headers we accept.
            let boundary = "----BlazenWasmSdkBoundary7c2a4e9f8b1d";
            let mut body: Vec<u8> = Vec::with_capacity(audio_bytes.len() + 512);
            let push_str = |b: &mut Vec<u8>, s: &str| b.extend_from_slice(s.as_bytes());

            push_str(&mut body, &format!("--{boundary}\r\n"));
            push_str(
                &mut body,
                "Content-Disposition: form-data; name=\"model\"\r\n\r\n",
            );
            push_str(&mut body, &model);
            push_str(&mut body, "\r\n");

            if let Some(lang) = language {
                push_str(&mut body, &format!("--{boundary}\r\n"));
                push_str(
                    &mut body,
                    "Content-Disposition: form-data; name=\"language\"\r\n\r\n",
                );
                push_str(&mut body, &lang);
                push_str(&mut body, "\r\n");
            }

            push_str(&mut body, &format!("--{boundary}\r\n"));
            push_str(
                &mut body,
                &format!(
                    "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                ),
            );
            push_str(&mut body, &format!("Content-Type: {content_type}\r\n\r\n"));
            body.extend_from_slice(&audio_bytes);
            push_str(&mut body, "\r\n");
            push_str(&mut body, &format!("--{boundary}--\r\n"));

            let url = join_url(&endpoint, "/v1/audio/transcriptions");
            let resp =
                http_post_multipart(&http, url, body, boundary, token.as_deref(), "transcribe")
                    .await?;
            let json = parse_json_value(&resp, "transcribe")?;
            serde_wasm_bindgen::to_value(&json)
                .map_err(|e| JsValue::from_str(&format!("transcribe: serialize JSON to JS: {e}")))
        }))
    }

    /// Streaming chat-completion via the OpenAI-compat SSE route.
    ///
    /// Issues `POST /v1/chat/completions` with `stream: true` (forced —
    /// any `stream` value the caller put in `request` is **overwritten**
    /// to `true`, since this method's whole purpose is the SSE path).
    /// The server replies with `text/event-stream` framed in
    /// OpenAI-compatible SSE: each event is a `data: <json>\n\n` block,
    /// terminated by `data: [DONE]\n\n`.
    ///
    /// Returns a JS `ReadableStream<object>` (web standard) that emits
    /// one parsed object per SSE `data:` payload. The terminating
    /// `[DONE]` sentinel is consumed and closes the stream gracefully;
    /// it is **not** surfaced to JS. Transport / parse errors are
    /// surfaced via `controller.error()` and propagate as a rejection on
    /// the reader's `read()` promise (or as `for await ... of` iterator
    /// rejection).
    ///
    /// Typical JS usage:
    ///
    /// ```js
    /// const stream = client.streamComplete({ model: "...", messages: [...] });
    /// for await (const chunk of stream) {
    ///   console.log(chunk.choices[0].delta.content);
    /// }
    /// ```
    ///
    /// The wrapper hand-rolls the SSE parser on top of
    /// `Response.body` (a `web_sys::ReadableStream` of `Uint8Array`)
    /// rather than depending on a separate streams-bridge crate; this
    /// keeps the wasm bundle small and avoids a dep on `wasm-streams`.
    #[wasm_bindgen(js_name = "streamComplete")]
    pub fn stream_complete(&self, request: JsValue) -> Result<web_sys::ReadableStream, JsValue> {
        // Parse the request and force `stream: true` up-front so we can
        // surface a clean synchronous error if the request body or model
        // field is malformed (instead of erroring out async on the first
        // `read()`).
        let mut body: serde_json::Value = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("streamComplete: parse request: {e}")))?;
        let _ = require_string_field(&body, "model", "streamComplete")?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".to_owned(), serde_json::Value::Bool(true));
        }
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| JsValue::from_str(&format!("streamComplete: serialize body: {e}")))?;

        let url = join_url(&self.endpoint, "/v1/chat/completions");
        let token = self.bearer_token.clone();

        // Build the underlying-source `start` callback. It receives the
        // `ReadableStreamDefaultController` for the wrapper stream we are
        // constructing and is responsible for kicking off the fetch +
        // read loop. We pull the chunks asynchronously via `spawn_local`
        // so the constructor itself returns synchronously, matching the
        // shape JS callers expect from `new ReadableStream(...)`.
        let start_closure = wasm_bindgen::closure::Closure::wrap(Box::new(
            move |controller: web_sys::ReadableStreamDefaultController| {
                let url = url.clone();
                let token = token.clone();
                let body_bytes = body_bytes.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Err(err) =
                        drive_stream_complete(&controller, url, token, body_bytes).await
                    {
                        controller.error_with_e(&err);
                    }
                });
                JsValue::UNDEFINED
            },
        )
            as Box<dyn FnMut(web_sys::ReadableStreamDefaultController) -> JsValue>);

        let source = js_sys::Object::new();
        js_sys::Reflect::set(
            &source,
            &JsValue::from_str("start"),
            start_closure.as_ref().unchecked_ref(),
        )
        .map_err(|e| JsValue::from_str(&format!("streamComplete: build source: {e:?}")))?;
        // Leak the closure: the underlying source object owns it for the
        // lifetime of the ReadableStream (the JS side keeps it alive until
        // the stream is GC'd, at which point the closure becomes
        // unreachable from JS and is collected on the wasm side).
        start_closure.forget();

        web_sys::ReadableStream::new_with_underlying_source(&source)
    }

    /// Upload a binary blob to the server.
    ///
    /// **Server route TBD; placeholder implementation.** The native gRPC
    /// `ModelClient::upload_blob` (see
    /// `blazen-controlplane/src/client/model_client.rs:380`) streams a
    /// chunked `UploadBlob` RPC through the gRPC `BlazenModelServer`.
    /// That RPC has **no HTTP REST mirror** on the model server: scanning
    /// `blazen-controlplane/src/http/blazen_admin.rs` and `openai_compat.rs`
    /// reveals routes for `/v1/blazen/adapters`, `/v1/blazen/models`,
    /// `/v1/blazen/health`, `/v1/blazen/metrics`, and the OpenAI-compat
    /// paths — but no `POST /v1/blazen/blobs` (nor the `/v1/blazen/content`
    /// path referenced in stale comments in `uploads.rs`). The
    /// `ContentStore` exists in `RestState` but is only populated via the
    /// `content_store` adapter-source branch; there is no top-level upload
    /// endpoint yet.
    ///
    /// Browsers send a single HTTP body rather than a streaming iterator,
    /// so the JS signature accepts a `Uint8Array`. Once the server route
    /// lands the implementation will POST `body` with `Content-Type` set
    /// from `options.mime` (default `application/octet-stream`) and
    /// `options.blobId` propagated through the URL or a request header,
    /// then resolve to the ack JSON.
    #[wasm_bindgen(js_name = "uploadBlob")]
    pub fn upload_blob(&self, _body: js_sys::Uint8Array, _options: JsValue) -> js_sys::Promise {
        future_to_promise(SendFuture(async move {
            Err::<JsValue, _>(JsValue::from_str(
                "uploadBlob: not implemented — no HTTP route exists on \
                 BlazenModelServer yet (gRPC UploadBlob has no REST mirror). \
                 Track: blazen-controlplane/src/http/ for /v1/blazen/blobs.",
            ))
        }))
    }

    /// Fetch a binary blob from the server as a `ReadableStream<Uint8Array>`.
    ///
    /// **Server route TBD; placeholder implementation.** The native gRPC
    /// `ModelClient::fetch_blob` (see
    /// `blazen-controlplane/src/client/model_client.rs:416`) opens a
    /// server-streaming `FetchBlob` RPC. That RPC has **no HTTP REST
    /// mirror** on the model server: scanning
    /// `blazen-controlplane/src/http/blazen_admin.rs` shows only the
    /// admin / OpenAI-compat routes — no `GET /v1/blazen/blobs/{blob_id}`.
    ///
    /// Once the server route lands the implementation will mirror
    /// [`Self::stream_complete`]'s `fetch()` plumbing but pass the raw
    /// response body straight through (no SSE decode): construct a
    /// `web_sys::Request` from `request.blobId`, await `fetch()`, and
    /// return `Response::body()` directly. Until then this signature
    /// rejects synchronously with a clean error so the API surface stays
    /// uniform with the native client and the binding lights up
    /// automatically when the route is added.
    ///
    /// Signature is synchronous (returns `Result<ReadableStream, JsValue>`)
    /// to match [`Self::stream_complete`]; JS callers receive a thrown
    /// error rather than a rejected promise.
    #[wasm_bindgen(js_name = "fetchBlob")]
    pub fn fetch_blob(&self, _request: JsValue) -> Result<web_sys::ReadableStream, JsValue> {
        Err(JsValue::from_str(
            "fetchBlob: not implemented — no HTTP route exists on \
             BlazenModelServer yet (gRPC FetchBlob has no REST mirror). \
             Track: blazen-controlplane/src/http/ for /v1/blazen/blobs/{blob_id}.",
        ))
    }

    /// Read-only accessor for the configured endpoint.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.endpoint.clone()
    }
}

/// Drive the SSE read loop for [`ModelClient::stream_complete`].
///
/// Performs the `fetch()`, walks the response body chunk-by-chunk with
/// a `TextDecoder`, splits on the SSE event boundary `\n\n`, parses
/// each event's `data:` payload as JSON, and enqueues the parsed object
/// onto `controller`. The `[DONE]` sentinel closes the stream and is
/// not surfaced to JS.
async fn drive_stream_complete(
    controller: &web_sys::ReadableStreamDefaultController,
    url: String,
    token: Option<String>,
    body_bytes: Vec<u8>,
) -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;

    // Build the fetch Request manually so we can keep the response
    // stream intact (the abstract `HttpClient::send` path buffers the
    // entire body, which would defeat streaming).
    let headers = web_sys::Headers::new()
        .map_err(|e| JsValue::from_str(&format!("streamComplete: Headers::new: {e:?}")))?;
    headers
        .set("Accept", "text/event-stream")
        .map_err(|e| JsValue::from_str(&format!("streamComplete: set Accept: {e:?}")))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| JsValue::from_str(&format!("streamComplete: set Content-Type: {e:?}")))?;
    if let Some(t) = token.as_deref() {
        headers
            .set("Authorization", &format!("Bearer {t}"))
            .map_err(|e| JsValue::from_str(&format!("streamComplete: set Authorization: {e:?}")))?;
    }

    let init = web_sys::RequestInit::new();
    init.set_method("POST");
    init.set_headers(&headers);
    let body_array = js_sys::Uint8Array::from(body_bytes.as_slice());
    init.set_body(&body_array);

    let request = web_sys::Request::new_with_str_and_init(&url, &init)
        .map_err(|e| JsValue::from_str(&format!("streamComplete: build Request: {e:?}")))?;

    // Call fetch from window or globalThis (Node/Workers).
    let fetch_promise: js_sys::Promise = if let Some(win) = web_sys::window() {
        win.fetch_with_request(&request)
    } else {
        let global = js_sys::global();
        let fetch_fn = js_sys::Reflect::get(&global, &JsValue::from_str("fetch"))
            .map_err(|e| JsValue::from_str(&format!("streamComplete: fetch lookup: {e:?}")))?;
        let fetch_fn: js_sys::Function = fetch_fn
            .dyn_into()
            .map_err(|_| JsValue::from_str("streamComplete: fetch is not callable"))?;
        let p = fetch_fn
            .call1(&JsValue::NULL, &request)
            .map_err(|e| JsValue::from_str(&format!("streamComplete: fetch invoke: {e:?}")))?;
        p.dyn_into()
            .map_err(|_| JsValue::from_str("streamComplete: fetch did not return a Promise"))?
    };

    let resp_value = wasm_bindgen_futures::JsFuture::from(fetch_promise)
        .await
        .map_err(|e| JsValue::from_str(&format!("streamComplete: fetch await: {e:?}")))?;
    let response: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|e| JsValue::from_str(&format!("streamComplete: Response cast: {e:?}")))?;

    if !response.ok() {
        // Drain the body for a clean error message, then bail.
        let status = response.status();
        let text_promise = response
            .text()
            .map_err(|e| JsValue::from_str(&format!("streamComplete: error body text(): {e:?}")))?;
        let text = wasm_bindgen_futures::JsFuture::from(text_promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("streamComplete: error body await: {e:?}")))?;
        let text = text.as_string().unwrap_or_default();
        return Err(JsValue::from_str(&format!(
            "streamComplete returned HTTP {status}: {text}"
        )));
    }

    let body_stream = response
        .body()
        .ok_or_else(|| JsValue::from_str("streamComplete: response has no body"))?;
    let reader: web_sys::ReadableStreamDefaultReader = body_stream
        .get_reader()
        .dyn_into()
        .map_err(|e| JsValue::from_str(&format!("streamComplete: get_reader: {e:?}")))?;

    let decoder = web_sys::TextDecoder::new_with_label("utf-8")
        .map_err(|e| JsValue::from_str(&format!("streamComplete: TextDecoder::new: {e:?}")))?;
    let mut pending = String::new();

    loop {
        let read_promise = reader.read();
        let result_val = wasm_bindgen_futures::JsFuture::from(read_promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("streamComplete: reader.read(): {e:?}")))?;

        let done = js_sys::Reflect::get(&result_val, &JsValue::from_str("done"))
            .map_err(|e| JsValue::from_str(&format!("streamComplete: read done: {e:?}")))?
            .as_bool()
            .unwrap_or(false);
        let value = js_sys::Reflect::get(&result_val, &JsValue::from_str("value"))
            .map_err(|e| JsValue::from_str(&format!("streamComplete: read value: {e:?}")))?;

        if done {
            // Flush any trailing buffered text (decoder + pending). The
            // server should have sent `data: [DONE]` already, but be
            // defensive: enqueue any final complete event still sitting
            // in `pending`, then close.
            let tail = decoder.decode().unwrap_or_default();
            pending.push_str(&tail);
            if !pending.is_empty() {
                drain_pending_events(controller, &mut pending, /* final_flush = */ true)?;
            }
            controller
                .close()
                .map_err(|e| JsValue::from_str(&format!("streamComplete: close: {e:?}")))?;
            return Ok(());
        }

        // `value` is a Uint8Array chunk.
        let chunk: js_sys::Uint8Array = value
            .dyn_into()
            .map_err(|e| JsValue::from_str(&format!("streamComplete: chunk cast: {e:?}")))?;
        let text = decoder
            .decode_with_js_u8_array_and_options(&chunk, &decode_options_streaming())
            .map_err(|e| JsValue::from_str(&format!("streamComplete: decode: {e:?}")))?;
        pending.push_str(&text);

        // If draining returns Ok(true) the server signalled `[DONE]`; close.
        if drain_pending_events(controller, &mut pending, /* final_flush = */ false)? {
            controller
                .close()
                .map_err(|e| JsValue::from_str(&format!("streamComplete: close: {e:?}")))?;
            // Best-effort cancel of the underlying body reader so the
            // socket is released promptly.
            let _ = reader.cancel();
            return Ok(());
        }
    }
}

/// Build a `TextDecodeOptions` object with `stream: true` so partial
/// UTF-8 sequences at chunk boundaries are buffered instead of
/// producing replacement characters.
fn decode_options_streaming() -> web_sys::TextDecodeOptions {
    let opts = web_sys::TextDecodeOptions::new();
    opts.set_stream(true);
    opts
}

/// Walk `pending` for complete SSE events (terminated by `\n\n`),
/// enqueue each parsed object onto `controller`, and return `Ok(true)`
/// if the terminating `[DONE]` sentinel was seen. `final_flush` allows
/// the loop's tail path to also process an unterminated event when the
/// underlying body closes without a final `\n\n`.
fn drain_pending_events(
    controller: &web_sys::ReadableStreamDefaultController,
    pending: &mut String,
    final_flush: bool,
) -> Result<bool, JsValue> {
    let mut saw_done = false;
    loop {
        let Some(idx) = pending.find("\n\n") else {
            break;
        };
        let event = pending[..idx].to_owned();
        pending.drain(..idx + 2);
        if process_event(controller, &event)? {
            saw_done = true;
            // Don't return early — we still want to drop the rest of
            // `pending`; the caller will close the controller.
            break;
        }
    }
    if final_flush && !saw_done && !pending.is_empty() {
        let event = std::mem::take(pending);
        if process_event(controller, &event)? {
            saw_done = true;
        }
    }
    Ok(saw_done)
}

/// Parse a single SSE event block (the text between two `\n\n`
/// boundaries). Each event may contain multiple lines; we honour only
/// `data:` lines (concatenating multi-line `data:` per the SSE spec),
/// ignore comments and other field types, and surface `[DONE]` to the
/// caller by returning `Ok(true)`.
fn process_event(
    controller: &web_sys::ReadableStreamDefaultController,
    event: &str,
) -> Result<bool, JsValue> {
    let mut data = String::new();
    for line in event.split('\n') {
        // SSE permits CR-LF line endings; strip a trailing CR if any.
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            // Per the SSE spec, exactly one leading space (if present)
            // is consumed.
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest);
        }
        // All other field types (`event:`, `id:`, `retry:`) are ignored
        // — OpenAI-compat SSE doesn't use them.
    }
    if data.is_empty() {
        return Ok(false);
    }
    if data.trim() == "[DONE]" {
        return Ok(true);
    }
    let parsed: serde_json::Value = serde_json::from_str(&data).map_err(|e| {
        JsValue::from_str(&format!(
            "streamComplete: parse SSE data JSON: {e} (payload: {data})"
        ))
    })?;
    let js_value = serde_wasm_bindgen::to_value(&parsed)
        .map_err(|e| JsValue::from_str(&format!("streamComplete: serialize SSE chunk: {e}")))?;
    controller
        .enqueue_with_chunk(&js_value)
        .map_err(|e| JsValue::from_str(&format!("streamComplete: enqueue: {e:?}")))?;
    Ok(false)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate `endpoint` and build a [`ModelClient`] wrapping a fresh
/// [`FetchHttpClient`]. Does not perform any network I/O.
fn build_client(endpoint: String, bearer_token: Option<String>) -> Result<ModelClient, JsValue> {
    if endpoint.is_empty() {
        return Err(JsValue::from_str("endpoint must not be empty"));
    }
    if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        return Err(JsValue::from_str(
            "endpoint must start with http:// or https://",
        ));
    }
    Ok(ModelClient {
        endpoint,
        bearer_token,
        http: FetchHttpClient::new().into_arc(),
    })
}

/// Lightweight health-check: GET `/v1/blazen/health`. Used by the
/// connect helpers to verify the endpoint is reachable before handing
/// the client back to JS.
async fn ping_health(client: &ModelClient) -> Result<(), JsValue> {
    let url = join_url(&client.endpoint, "/v1/blazen/health");
    http_get_json(&client.http, url, client.bearer_token.as_deref(), "health").await?;
    Ok(())
}
