//! [`HttpClient`] implementation backed by the browser `fetch()` API.
//!
//! [`FetchHttpClient`] bridges the abstract [`HttpClient`] trait from
//! `blazen-llm` to the `web-sys` `fetch` primitives, allowing all Blazen
//! providers to work unmodified in the browser.

use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::Stream;
use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, Request, RequestInit, Response};

use crate::error::BlazenError;
use crate::http::{ByteStream, HttpClient, HttpMethod, HttpRequest, HttpResponse};

// ---------------------------------------------------------------------------
// SendFuture wrapper
// ---------------------------------------------------------------------------

/// A wrapper that unsafely implements `Send` for a non-Send future.
///
/// SAFETY: WebAssembly is single-threaded. There is no possibility of
/// concurrent access, so the `Send` bound is vacuously satisfied.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
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
// FetchHttpClient
// ---------------------------------------------------------------------------

/// An [`HttpClient`] that uses the browser `fetch()` API via `web-sys`.
///
/// Since WebAssembly is single-threaded, the `Send + Sync` bounds on
/// [`HttpClient`] are vacuously satisfied. We mark the struct accordingly
/// so it can be stored in an `Arc<dyn HttpClient>`.
#[derive(Debug, Clone)]
pub struct FetchHttpClient;

// SAFETY: WASM is single-threaded; there is no thread to race with.
unsafe impl Send for FetchHttpClient {}
unsafe impl Sync for FetchHttpClient {}

impl FetchHttpClient {
    /// Create a new `FetchHttpClient`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Wrap `Self` in an `Arc` for use as `Arc<dyn HttpClient>`.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn HttpClient> {
        Arc::new(self)
    }
}

impl Default for FetchHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `web_sys::Request` from a `blazen_llm::http::HttpRequest`.
fn build_web_request(req: &HttpRequest) -> Result<Request, BlazenError> {
    let method_str = match req.method {
        HttpMethod::Get => "GET",
        HttpMethod::Post => "POST",
        HttpMethod::Put => "PUT",
    };

    // Build the full URL with query parameters.
    let url = if req.query_params.is_empty() {
        req.url.clone()
    } else {
        let mut url = req.url.clone();
        url.push('?');
        for (i, (k, v)) in req.query_params.iter().enumerate() {
            if i > 0 {
                url.push('&');
            }
            let encoded_k: String = js_sys::encode_uri_component(k).into();
            let encoded_v: String = js_sys::encode_uri_component(v).into();
            url.push_str(&encoded_k);
            url.push('=');
            url.push_str(&encoded_v);
        }
        url
    };

    let headers = Headers::new().map_err(|e| BlazenError::request(format!("{e:?}")))?;
    for (k, v) in &req.headers {
        headers
            .append(k, v)
            .map_err(|e| BlazenError::request(format!("{e:?}")))?;
    }

    let opts = RequestInit::new();
    opts.set_method(method_str);
    opts.set_headers(&headers);

    if let Some(ref body) = req.body {
        let uint8 = Uint8Array::from(body.as_slice());
        opts.set_body(&uint8);
    }

    Request::new_with_str_and_init(&url, &opts).map_err(|e| BlazenError::request(format!("{e:?}")))
}

/// Call global `fetch()` -- works in both browser (window.fetch) and
/// Node.js / Deno / Workers (globalThis.fetch).
fn call_fetch(web_request: &Request) -> Result<js_sys::Promise, BlazenError> {
    let global = js_sys::global();
    web_sys::window()
        .map(|w| w.fetch_with_request(web_request))
        .or_else(|| {
            let fetch_fn = js_sys::Reflect::get(&global, &JsValue::from_str("fetch")).ok()?;
            let fetch_fn: js_sys::Function = fetch_fn.dyn_into().ok()?;
            Some(fetch_fn.call1(&JsValue::NULL, web_request).ok()?.into())
        })
        .ok_or_else(|| BlazenError::request("fetch() is not available in this environment"))
}

/// Extract response headers as a `Vec<(String, String)>`.
fn extract_headers(raw: &Headers) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    if let Ok(Some(iter)) = js_sys::try_iter(raw) {
        for entry in iter {
            if let Ok(entry) = entry {
                let pair = js_sys::Array::from(&entry);
                let key: String = pair.get(0).as_string().unwrap_or_default();
                let value: String = pair.get(1).as_string().unwrap_or_default();
                headers.push((key, value));
            }
        }
    }
    headers
}

/// Read the full body of a `web_sys::Response` into bytes.
async fn read_body(resp: &Response) -> Result<Vec<u8>, BlazenError> {
    let array_buf_promise = resp
        .array_buffer()
        .map_err(|e| BlazenError::request(format!("{e:?}")))?;
    let array_buf: ArrayBuffer = JsFuture::from(array_buf_promise)
        .await
        .map_err(|e| BlazenError::request(format!("{e:?}")))?
        .dyn_into()
        .map_err(|e| BlazenError::request(format!("{e:?}")))?;
    let uint8 = Uint8Array::new(&array_buf);
    Ok(uint8.to_vec())
}

// ---------------------------------------------------------------------------
// Streaming support
// ---------------------------------------------------------------------------

/// A wrapper around a `ReadableStreamDefaultReader` that implements
/// `Stream<Item = Result<Bytes, ...>>`.
///
/// SAFETY: WASM is single-threaded, so `Send` is vacuously safe.
struct ReadableStreamWrapper {
    reader: web_sys::ReadableStreamDefaultReader,
}

unsafe impl Send for ReadableStreamWrapper {}
unsafe impl Sync for ReadableStreamWrapper {}

impl Stream for ReadableStreamWrapper {
    type Item = Result<Bytes, Box<dyn std::error::Error + Send + Sync>>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // TODO: Implement proper streaming via ReadableStreamDefaultReader.read()
        //
        // The challenge is that reader.read() returns a JS Promise, which cannot
        // be polled synchronously in a Rust Stream::poll_next. A production
        // implementation would use an intermediate channel:
        //
        // 1. On first poll, spawn_local a task that loops reader.read() and sends
        //    chunks over a wasm-compatible channel (e.g., futures::channel::mpsc).
        // 2. poll_next then polls the receiver side of the channel.
        //
        // For now, the non-streaming `send()` path is fully functional. Streaming
        // will be implemented in a follow-up.
        let _ = &self.reader;
        std::task::Poll::Ready(None)
    }
}

// ---------------------------------------------------------------------------
// HttpClient implementation
// ---------------------------------------------------------------------------

/// The actual async implementation without Send constraints.
///
/// We wrap these in `SendFuture` to satisfy the `async_trait` `Send` bound,
/// which is safe because WASM is single-threaded.
impl FetchHttpClient {
    async fn send_impl(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        let web_request = build_web_request(&request)?;
        let resp_promise = call_fetch(&web_request)?;

        let resp_value = JsFuture::from(resp_promise)
            .await
            .map_err(|e| BlazenError::request(format!("{e:?}")))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|e| BlazenError::request(format!("{e:?}")))?;

        let status = resp.status();
        let headers = extract_headers(&resp.headers());
        let body = read_body(&resp).await?;

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn send_streaming_impl(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        let web_request = build_web_request(&request)?;
        let resp_promise = call_fetch(&web_request)?;

        let resp_value = JsFuture::from(resp_promise)
            .await
            .map_err(|e| BlazenError::request(format!("{e:?}")))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|e| BlazenError::request(format!("{e:?}")))?;

        let status = resp.status();
        let headers = extract_headers(&resp.headers());

        let body = resp
            .body()
            .ok_or_else(|| BlazenError::request("response has no body for streaming"))?;
        let reader: web_sys::ReadableStreamDefaultReader = body
            .get_reader()
            .dyn_into()
            .map_err(|e| BlazenError::request(format!("{e:?}")))?;

        let stream = ReadableStreamWrapper { reader };
        let boxed: ByteStream = Box::pin(stream);

        Ok((status, headers, boxed))
    }
}

#[async_trait::async_trait]
impl HttpClient for FetchHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        // SAFETY: WASM is single-threaded, Send is vacuously satisfied.
        SendFuture(self.send_impl(request)).await
    }

    async fn send_streaming(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        // SAFETY: WASM is single-threaded, Send is vacuously satisfied.
        SendFuture(self.send_streaming_impl(request)).await
    }
}
