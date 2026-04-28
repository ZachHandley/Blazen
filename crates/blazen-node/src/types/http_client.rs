//! Custom HTTP transport bindings for the Node.js SDK.
//!
//! Exposes [`JsHttpClient`] as an abstract base class that JavaScript code
//! can subclass to provide a custom HTTP backend (e.g. Cloudflare `fetch`,
//! `undici`, mocked transports for tests). Subclasses override `send()` and
//! return an [`JsHttpResponse`] from the request described by [`JsHttpRequest`].
//!
//! The Rust adapter pieces ([`JsHttpClient::as_dyn_http_client`] and the
//! private adapter struct) are scaffolding for provider integration that
//! accepts a custom transport — they are exercised by future provider
//! constructor wiring rather than by the bindings module itself.

#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_llm::error::BlazenError;
use blazen_llm::http::{ByteStream, HttpClient, HttpMethod, HttpRequest, HttpResponse};

// ---------------------------------------------------------------------------
// Plain-object request and response types
// ---------------------------------------------------------------------------

/// An outgoing HTTP request, as seen by a JavaScript [`HttpClient`] subclass.
#[napi(object)]
pub struct JsHttpRequest {
    /// HTTP method (`"GET"`, `"POST"`, `"PUT"`, `"DELETE"`, `"PATCH"`).
    pub method: String,
    /// Full request URL (query parameters already encoded).
    pub url: String,
    /// Request headers as `[name, value]` tuples.
    pub headers: Vec<Vec<String>>,
    /// Request body bytes, if any.
    pub body: Option<Buffer>,
}

/// A complete HTTP response (body fully buffered).
#[napi(object)]
pub struct JsHttpResponse {
    /// Status code (e.g. `200`, `404`).
    pub status: u32,
    /// Response headers as `[name, value]` tuples.
    pub headers: Vec<Vec<String>>,
    /// Response body bytes.
    pub body: Buffer,
}

fn method_to_str(m: HttpMethod) -> &'static str {
    match m {
        HttpMethod::Get => "GET",
        HttpMethod::Post => "POST",
        HttpMethod::Put => "PUT",
        HttpMethod::Delete => "DELETE",
        HttpMethod::Patch => "PATCH",
    }
}

/// Percent-encode a string per RFC 3986 (unreserved set kept verbatim).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                use std::fmt::Write;
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    out
}

fn build_full_url(req: &HttpRequest) -> String {
    if req.query_params.is_empty() {
        return req.url.clone();
    }
    let mut url = req.url.clone();
    let sep = if url.contains('?') { '&' } else { '?' };
    url.push(sep);
    let mut first = true;
    for (k, v) in &req.query_params {
        if !first {
            url.push('&');
        }
        first = false;
        url.push_str(&percent_encode(k));
        url.push('=');
        url.push_str(&percent_encode(v));
    }
    url
}

// ---------------------------------------------------------------------------
// JsHttpClient (subclassable abstract base)
// ---------------------------------------------------------------------------

/// Threadsafe handler for the JS-side `send()` method: takes an
/// [`JsHttpRequest`] dict and returns an [`JsHttpResponse`] (or a Promise
/// thereof).
type SendHandlerTsfn =
    ThreadsafeFunction<JsHttpRequest, Promise<JsHttpResponse>, JsHttpRequest, Status, false, true>;

/// Abstract base class for custom HTTP transports.
///
/// Subclass this to plug in a custom HTTP backend (Cloudflare `fetch`,
/// `undici`, a mock for tests, etc.):
///
/// ```javascript
/// class FetchHttpClient extends HttpClient {
///   async send(request) {
///     const init = {
///       method: request.method,
///       headers: Object.fromEntries(request.headers),
///       body: request.body ?? undefined,
///     };
///     const res = await fetch(request.url, init);
///     const body = Buffer.from(await res.arrayBuffer());
///     const headers = [...res.headers.entries()].map(([k, v]) => [k, v]);
///     return { status: res.status, headers, body };
///   }
/// }
/// ```
///
/// Streaming requests (`send_streaming` in the Rust API) are not currently
/// dispatched to JS subclasses — providers that need streaming will fall
/// back to non-streaming mode when a custom client is used.
#[napi(js_name = "HttpClient")]
pub struct JsHttpClient {
    /// Threadsafe send-handler bound at construction time. `None` for the
    /// default subclass-overridable instance (which throws on use).
    pub(crate) send_handler: Option<Arc<SendHandlerTsfn>>,
}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::unused_async
)]
impl JsHttpClient {
    /// Construct a base `HttpClient`.
    ///
    /// JavaScript subclasses must override `send()`.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self { send_handler: None }
    }

    /// Send a request and return the buffered response.
    ///
    /// Subclasses **must** override this method.
    #[napi]
    pub async fn send(&self, _request: JsHttpRequest) -> Result<JsHttpResponse> {
        Err(napi::Error::from_reason("subclass must override send()"))
    }

    /// Bind a JavaScript callback as the send-handler.
    ///
    /// This is an alternative to subclassing: instead of `class Foo extends
    /// HttpClient`, callers can do:
    ///
    /// ```javascript
    /// const client = HttpClient.fromCallback(async (req) => {
    ///   /* ... fetch and return { status, headers, body } ... */
    /// });
    /// ```
    #[napi(factory, js_name = "fromCallback")]
    pub fn from_callback(handler: SendHandlerTsfn) -> Self {
        Self {
            send_handler: Some(Arc::new(handler)),
        }
    }
}

impl JsHttpClient {
    /// Build an `Arc<dyn HttpClient>` adapter that bridges the JS-side
    /// `send()` callback into the Rust `HttpClient` trait. Returns `None`
    /// when the `JsHttpClient` has no callback-based handler bound (e.g. a
    /// pure-subclass instance whose `send` lives in JS as a method).
    pub(crate) fn as_dyn_http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.send_handler.as_ref().map(|h| {
            Arc::new(JsHttpClientAdapter {
                handler: Arc::clone(h),
            }) as Arc<dyn HttpClient>
        })
    }
}

// ---------------------------------------------------------------------------
// Rust-side adapter implementing `HttpClient` for the JS callback
// ---------------------------------------------------------------------------

struct JsHttpClientAdapter {
    handler: Arc<SendHandlerTsfn>,
}

impl std::fmt::Debug for JsHttpClientAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsHttpClientAdapter")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl HttpClient for JsHttpClientAdapter {
    async fn send(&self, request: HttpRequest) -> std::result::Result<HttpResponse, BlazenError> {
        let js_req = JsHttpRequest {
            method: method_to_str(request.method).to_owned(),
            url: build_full_url(&request),
            headers: request
                .headers
                .iter()
                .map(|(k, v)| vec![k.clone(), v.clone()])
                .collect(),
            body: request.body.as_ref().map(|b| b.clone().into()),
        };

        let promise = self
            .handler
            .call_async(js_req)
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

        let resp = promise
            .await
            .map_err(|e| BlazenError::request(e.to_string()))?;

        let headers: Vec<(String, String)> = resp
            .headers
            .into_iter()
            .filter_map(|pair| {
                let mut iter = pair.into_iter();
                Some((iter.next()?, iter.next()?))
            })
            .collect();

        Ok(HttpResponse {
            status: u16::try_from(resp.status).unwrap_or(0),
            headers,
            body: resp.body.to_vec(),
        })
    }

    async fn send_streaming(
        &self,
        _request: HttpRequest,
    ) -> std::result::Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        Err(BlazenError::unsupported(
            "Custom JS HttpClient does not implement streaming; \
             use the default ReqwestHttpClient for SSE/streaming endpoints.",
        ))
    }
}
