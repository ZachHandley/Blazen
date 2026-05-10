//! Wasi OTLP HTTP transport. Implements [`opentelemetry_http::HttpClient`]
//! using `Arc<dyn blazen_llm::http::HttpClient>` so the wasi build (Cloudflare
//! Workers / Deno via napi-rs's wasi runtime) can export traces without
//! `reqwest` (no socket access on wasi) or `web_sys` (no JS DOM bindings on
//! wasi).
//!
//! The host registers a fetch-backed `HttpClient` via `setDefaultHttpClient(...)`
//! at module load (see the `blazen-node` napi binding); this exporter pulls it
//! through [`blazen_llm::http_napi_wasi::LazyHttpClient`].

use std::sync::Arc;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Request, Response};
use opentelemetry_http::{HttpClient as OtelHttpClient, HttpError};

use blazen_llm::http::{HttpClient as BlazenHttpClient, HttpMethod, HttpRequest};
use blazen_llm::http_napi_wasi::LazyHttpClient;

// ---------------------------------------------------------------------------
// WasiFetchHttpClient
// ---------------------------------------------------------------------------

/// `opentelemetry_http::HttpClient` backed by an `Arc<dyn blazen_llm::HttpClient>`.
///
/// Construct with [`WasiFetchHttpClient::new`] (uses the lazy proxy that
/// resolves the host-registered fetch client) or [`WasiFetchHttpClient::with_inner`]
/// (inject any concrete client, useful for tests).
#[derive(Debug, Clone)]
pub struct WasiFetchHttpClient {
    inner: Arc<dyn BlazenHttpClient>,
}

impl WasiFetchHttpClient {
    /// Construct a client that pulls the registered default
    /// `blazen_llm::HttpClient` lazily via [`LazyHttpClient`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(LazyHttpClient::new()),
        }
    }

    /// Construct a client wrapping an already-built `HttpClient`.
    #[must_use]
    pub fn with_inner(inner: Arc<dyn BlazenHttpClient>) -> Self {
        Self { inner }
    }
}

impl Default for WasiFetchHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn translate_method(method: &http::Method) -> Result<HttpMethod, HttpError> {
    Ok(match *method {
        http::Method::POST => HttpMethod::Post,
        http::Method::GET => HttpMethod::Get,
        http::Method::PUT => HttpMethod::Put,
        http::Method::DELETE => HttpMethod::Delete,
        http::Method::PATCH => HttpMethod::Patch,
        ref other => {
            return Err(format!("wasi OTLP: unsupported HTTP method {other}").into());
        }
    })
}

fn extract_headers(pairs: &[(String, String)]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for (k, v) in pairs {
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            map.append(name, val);
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Trait impl
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl OtelHttpClient for WasiFetchHttpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        let (parts, body) = request.into_parts();

        let method = translate_method(&parts.method)?;

        let mut headers: Vec<(String, String)> = Vec::with_capacity(parts.headers.len());
        for (name, value) in &parts.headers {
            let value_str = value.to_str().map_err(|e| -> HttpError { Box::new(e) })?;
            headers.push((name.as_str().to_owned(), value_str.to_owned()));
        }

        let blazen_req = HttpRequest {
            method,
            url: parts.uri.to_string(),
            headers,
            body: if body.is_empty() {
                None
            } else {
                Some(body.to_vec())
            },
            query_params: Vec::new(),
        };

        let blazen_resp = self
            .inner
            .send(blazen_req)
            .await
            .map_err(|e| -> HttpError { format!("wasi OTLP send: {e}").into() })?;

        let header_map = extract_headers(&blazen_resp.headers);

        let mut http_response = Response::builder()
            .status(blazen_resp.status)
            .body(Bytes::from(blazen_resp.body))
            .map_err(|e| -> HttpError { Box::new(e) })?;
        *http_response.headers_mut() = header_map;
        Ok(http_response)
    }
}
