//! `wasm-bindgen` wrapper for [`blazen_llm::http::HttpClient`].
//!
//! Exposes a `WasmHttpClient` class that lets TypeScript code plug a custom
//! HTTP backend (e.g. a request-signing proxy, an in-memory mock for tests,
//! or an alternative `fetch` polyfill) into the Blazen runtime.
//!
//! The class wraps a JS `send` callback that receives a serialised
//! [`HttpRequest`]-shaped object and returns either an
//! [`HttpResponse`]-shaped object or a Promise that resolves to one. An
//! optional `sendStreaming` callback may be provided for SSE responses; if
//! omitted, streaming requests fall back to the buffered handler and emit
//! the full body as a single chunk.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use blazen_llm::error::BlazenError;
use blazen_llm::http::{ByteStream, HttpClient, HttpMethod, HttpRequest, HttpResponse};

// ---------------------------------------------------------------------------
// SendFuture wrapper
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

// SAFETY: WASM is single-threaded; there is no other thread to race with.
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
// Wire types (plain serde structs that round-trip with the JS callback)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct WireRequest<'a> {
    method: &'static str,
    url: &'a str,
    headers: Vec<(&'a str, &'a str)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<&'a [u8]>,
    #[serde(rename = "queryParams")]
    query_params: Vec<(&'a str, &'a str)>,
}

#[derive(Deserialize)]
struct WireResponse {
    status: u16,
    #[serde(default)]
    headers: Vec<(String, String)>,
    #[serde(default)]
    body: Vec<u8>,
}

fn method_str(m: HttpMethod) -> &'static str {
    match m {
        HttpMethod::Get => "GET",
        HttpMethod::Post => "POST",
        HttpMethod::Put => "PUT",
        HttpMethod::Delete => "DELETE",
        HttpMethod::Patch => "PATCH",
    }
}

fn request_to_js(request: &HttpRequest) -> Result<JsValue, BlazenError> {
    let wire = WireRequest {
        method: method_str(request.method),
        url: &request.url,
        headers: request
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect(),
        body: request.body.as_deref(),
        query_params: request
            .query_params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect(),
    };
    serde_wasm_bindgen::to_value(&wire).map_err(|e| BlazenError::request(e.to_string()))
}

async fn await_promise(value: JsValue) -> Result<JsValue, BlazenError> {
    if value.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = value.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| BlazenError::request(format!("HTTP handler rejected: {e:?}")))
    } else {
        Ok(value)
    }
}

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_HTTP_CLIENT_HANDLERS: &str = r#"
/** Wire shape passed to a `WasmHttpClient` send handler. */
export interface HttpClientRequest {
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
    url: string;
    headers: [string, string][];
    body?: Uint8Array | number[] | null;
    queryParams: [string, string][];
}

/** Wire shape returned from a `WasmHttpClient` send handler. */
export interface HttpClientResponse {
    status: number;
    headers?: [string, string][];
    body?: Uint8Array | number[];
}

/** Buffered HTTP send handler. */
export type HttpSendHandler =
    (request: HttpClientRequest) => Promise<HttpClientResponse> | HttpClientResponse;

/** Streaming SSE handler. Each yielded chunk is delivered to `onChunk`. */
export type HttpStreamHandler = (
    request: HttpClientRequest,
    onChunk: (bytes: Uint8Array) => void,
) => Promise<{ status: number; headers?: [string, string][] }>;
"#;

// ---------------------------------------------------------------------------
// WasmHttpClient
// ---------------------------------------------------------------------------

/// A JavaScript-backed [`HttpClient`] implementation.
///
/// ```js
/// import { HttpClient } from '@blazen/sdk';
///
/// const client = new HttpClient(async (req) => {
///   const res = await fetch(req.url, {
///     method: req.method,
///     headers: Object.fromEntries(req.headers),
///     body: req.body ? new Uint8Array(req.body) : undefined,
///   });
///   return {
///     status: res.status,
///     headers: [...res.headers.entries()],
///     body: new Uint8Array(await res.arrayBuffer()),
///   };
/// });
/// ```
#[wasm_bindgen(js_name = "HttpClient")]
pub struct WasmHttpClient {
    inner: Arc<JsHttpClient>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmHttpClient {}
unsafe impl Sync for WasmHttpClient {}

#[wasm_bindgen(js_class = "HttpClient")]
impl WasmHttpClient {
    /// Construct a new client from a JS send handler.
    ///
    /// @param sendHandler        - Buffered request/response handler.
    /// @param sendStreamHandler  - Optional streaming handler for SSE.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(
        send_handler: js_sys::Function,
        send_stream_handler: Option<js_sys::Function>,
    ) -> Self {
        Self {
            inner: Arc::new(JsHttpClient {
                send_handler,
                send_stream_handler,
            }),
        }
    }
}

impl WasmHttpClient {
    /// Borrow the inner `Arc<dyn HttpClient>` for use by other crate
    /// modules that wire a custom client into `OpenAiCompatProvider`,
    /// `AnthropicProvider`, etc.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn inner_arc(&self) -> Arc<dyn HttpClient> {
        Arc::clone(&self.inner) as Arc<dyn HttpClient>
    }
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

struct JsHttpClient {
    send_handler: js_sys::Function,
    send_stream_handler: Option<js_sys::Function>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for JsHttpClient {}
unsafe impl Sync for JsHttpClient {}

impl std::fmt::Debug for JsHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsHttpClient")
            .field("has_stream_handler", &self.send_stream_handler.is_some())
            .finish_non_exhaustive()
    }
}

impl JsHttpClient {
    async fn send_impl(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        let js_request = request_to_js(&request)?;
        let raw = self
            .send_handler
            .call1(&JsValue::NULL, &js_request)
            .map_err(|e| BlazenError::request(format!("HTTP handler threw: {e:?}")))?;
        let resolved = await_promise(raw).await?;
        let wire: WireResponse = serde_wasm_bindgen::from_value(resolved)
            .map_err(|e| BlazenError::request(format!("invalid HTTP response: {e}")))?;
        Ok(HttpResponse {
            status: wire.status,
            headers: wire.headers,
            body: wire.body,
        })
    }

    async fn send_streaming_impl(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        use std::cell::RefCell;
        use std::rc::Rc;

        let Some(handler) = self.send_stream_handler.as_ref() else {
            let response = self.send_impl(request).await?;
            let status = response.status;
            let headers = response.headers.clone();
            let body = Bytes::from(response.body);
            let stream = futures_util::stream::once(async move { Ok(body) });
            return Ok((status, headers, Box::pin(stream)));
        };

        let js_request = request_to_js(&request)?;
        let chunks: Rc<RefCell<Vec<Bytes>>> = Rc::new(RefCell::new(Vec::new()));
        let chunks_ref = Rc::clone(&chunks);

        let on_chunk = Closure::wrap(Box::new(move |js_chunk: JsValue| {
            if let Ok(bytes) = serde_wasm_bindgen::from_value::<Vec<u8>>(js_chunk) {
                chunks_ref.borrow_mut().push(Bytes::from(bytes));
            }
        }) as Box<dyn FnMut(JsValue)>);

        let raw = handler
            .call2(&JsValue::NULL, &js_request, on_chunk.as_ref().unchecked_ref())
            .map_err(|e| BlazenError::request(format!("HTTP stream handler threw: {e:?}")))?;
        let resolved = await_promise(raw).await?;
        drop(on_chunk);

        #[derive(Deserialize, Default)]
        struct StreamMeta {
            status: u16,
            #[serde(default)]
            headers: Vec<(String, String)>,
        }

        let meta: StreamMeta = serde_wasm_bindgen::from_value(resolved)
            .map_err(|e| BlazenError::request(format!("invalid stream metadata: {e}")))?;

        let collected: Vec<Bytes> = chunks.borrow().clone();
        let stream = futures_util::stream::iter(collected.into_iter().map(Ok));
        Ok((meta.status, meta.headers, Box::pin(stream) as ByteStream))
    }
}

#[async_trait]
impl HttpClient for JsHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        // SAFETY: WASM is single-threaded.
        SendFuture(self.send_impl(request)).await
    }

    async fn send_streaming(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        // SAFETY: WASM is single-threaded.
        SendFuture(self.send_streaming_impl(request)).await
    }
}

