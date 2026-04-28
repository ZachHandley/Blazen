//! `opentelemetry_http::HttpClient` implementation backed by the browser
//! `fetch()` API.
//!
//! The `opentelemetry-http` 0.31 trait requires `Send + Sync` on every
//! `HttpClient`. On wasm32, `reqwest::Client` (and `reqwest::blocking::Client`)
//! produce futures backed by `Rc<RefCell<JsFuture>>`, which is `!Send`, so the
//! reqwest impls in `opentelemetry-http` fail to compile for the
//! `wasm32-unknown-unknown` target.
//!
//! This client mirrors the pattern from `blazen-llm`'s `FetchHttpClient`: build
//! a `web_sys::Request`, dispatch via `globalThis.fetch`, and read the response
//! body as bytes. It then `unsafe impl Send + Sync` is used to satisfy the
//! trait bound — this is vacuously safe on wasm32 because the target is
//! single-threaded and there is no other thread that could observe the
//! `!Send` internals.

use std::pin::Pin;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Request, Response};
use js_sys::{ArrayBuffer, Uint8Array};
use opentelemetry_http::{HttpClient, HttpError};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, RequestInit};

// ---------------------------------------------------------------------------
// SendFuture wrapper
// ---------------------------------------------------------------------------

/// Wraps a `!Send` future and asserts `Send` for it.
///
/// SAFETY: WebAssembly is single-threaded; no other thread exists that could
/// race with the wrapped future. The `Send` bound on the wrapper is
/// vacuously satisfied.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded; nothing can observe the `!Send` interior
// from another thread.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: Pin projection through a `#[repr(transparent)]`-style newtype.
        // We never move `F`; we only re-pin to it.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// WasmFetchHttpClient
// ---------------------------------------------------------------------------

/// `opentelemetry_http::HttpClient` backed by `web_sys::fetch`.
///
/// SAFETY of `Send + Sync`: WASM is single-threaded.
#[derive(Debug, Clone, Default)]
pub struct WasmFetchHttpClient;

// SAFETY: WASM is single-threaded; vacuously safe.
unsafe impl Send for WasmFetchHttpClient {}
// SAFETY: WASM is single-threaded; vacuously safe.
unsafe impl Sync for WasmFetchHttpClient {}

impl WasmFetchHttpClient {
    /// Construct a new `WasmFetchHttpClient`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an `http::Request<Bytes>` into a `web_sys::Request`.
fn build_web_request(request: &Request<Bytes>) -> Result<web_sys::Request, HttpError> {
    let method_str = request.method().as_str();
    let url = request.uri().to_string();

    let headers = Headers::new().map_err(js_to_error)?;
    for (name, value) in request.headers() {
        let value_str = value.to_str().map_err(|e| -> HttpError { Box::new(e) })?;
        headers
            .append(name.as_str(), value_str)
            .map_err(js_to_error)?;
    }

    let opts = RequestInit::new();
    opts.set_method(method_str);
    opts.set_headers(&headers);

    // Always provide the body as a Uint8Array view over the request bytes.
    // OTLP HTTP/protobuf requests are POSTs with a binary protobuf payload.
    let body = request.body();
    if !body.is_empty() {
        let uint8 = Uint8Array::from(body.as_ref());
        opts.set_body(&uint8);
    }

    web_sys::Request::new_with_str_and_init(&url, &opts).map_err(js_to_error)
}

/// Dispatch via `window.fetch` if available, otherwise `globalThis.fetch`
/// (Workers, Node `--experimental-fetch`, Deno).
fn call_fetch(web_request: &web_sys::Request) -> Result<js_sys::Promise, HttpError> {
    if let Some(window) = web_sys::window() {
        return Ok(window.fetch_with_request(web_request));
    }

    let global = js_sys::global();
    let fetch_fn = js_sys::Reflect::get(&global, &JsValue::from_str("fetch"))
        .map_err(js_to_error)?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| -> HttpError { "fetch is not a function".into() })?;
    let promise_val = fetch_fn
        .call1(&JsValue::NULL, web_request)
        .map_err(js_to_error)?;
    Ok(promise_val.into())
}

/// Extract response headers into an `http::HeaderMap`.
fn extract_headers(raw: &Headers) -> HeaderMap {
    let mut map = HeaderMap::new();
    if let Ok(Some(iter)) = js_sys::try_iter(raw) {
        for entry in iter.flatten() {
            let pair = js_sys::Array::from(&entry);
            let key: String = pair.get(0).as_string().unwrap_or_default();
            let value: String = pair.get(1).as_string().unwrap_or_default();
            if let (Ok(name), Ok(val)) = (
                HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(&value),
            ) {
                map.append(name, val);
            }
        }
    }
    map
}

/// Read a `web_sys::Response` body fully into `Bytes`.
async fn read_body(resp: &web_sys::Response) -> Result<Bytes, HttpError> {
    let array_buf_promise = resp.array_buffer().map_err(js_to_error)?;
    let array_buf: ArrayBuffer = JsFuture::from(array_buf_promise)
        .await
        .map_err(js_to_error)?
        .dyn_into()
        .map_err(|_| -> HttpError { "response body is not an ArrayBuffer".into() })?;
    let uint8 = Uint8Array::new(&array_buf);
    Ok(Bytes::from(uint8.to_vec()))
}

/// Convert a `JsValue` error into an `HttpError`.
fn js_to_error(value: JsValue) -> HttpError {
    let msg = value
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&value, &JsValue::from_str("message"))
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| format!("{value:?}"));
    msg.into()
}

// ---------------------------------------------------------------------------
// Trait impl
// ---------------------------------------------------------------------------

impl WasmFetchHttpClient {
    async fn send_bytes_impl(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        let web_request = build_web_request(&request)?;
        let resp_promise = call_fetch(&web_request)?;

        let resp_value = JsFuture::from(resp_promise).await.map_err(js_to_error)?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| -> HttpError { "fetch result is not a Response".into() })?;

        let status = resp.status();
        let headers = extract_headers(&resp.headers());
        let body = read_body(&resp).await?;

        let mut http_response = Response::builder()
            .status(status)
            .body(body)
            .map_err(|e| -> HttpError { Box::new(e) })?;
        *http_response.headers_mut() = headers;
        Ok(http_response)
    }
}

#[async_trait::async_trait]
impl HttpClient for WasmFetchHttpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        // SAFETY: WASM is single-threaded; the wrapped future cannot be
        // observed by another thread, so the `Send` bound the trait imposes
        // is vacuously satisfied.
        SendFuture(self.send_bytes_impl(request)).await
    }
}
