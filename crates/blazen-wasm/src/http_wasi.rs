//! [`HttpClient`] implementation backed by WASI HTTP (`wasi:http/outgoing-handler`).
//!
//! This module bridges Blazen's abstract [`HttpClient`] trait to the WASI HTTP
//! outgoing handler, allowing LLM providers to make HTTP requests when running
//! inside a wasmtime-based runtime. TLS is handled transparently by the host.

use blazen_llm::error::BlazenError;
use blazen_llm::http::{ByteStream, HttpClient, HttpMethod, HttpRequest, HttpResponse};

use crate::wasi::http::outgoing_handler;
use crate::wasi::http::types::{
    Fields, IncomingBody, Method, OutgoingBody, OutgoingRequest, Scheme,
};
use crate::wasi::io::streams::StreamError;

use std::sync::Arc;

/// An [`HttpClient`] that uses `wasi:http/outgoing-handler` for outbound requests.
///
/// This is the HTTP backend used when Blazen runs as a WASM component on `ZLayer`.
/// The host runtime (wasmtime) provides TLS, DNS resolution, and connection pooling.
#[derive(Debug, Clone)]
pub struct WasiHttpClient;

impl WasiHttpClient {
    /// Create a new WASI HTTP client.
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

impl Default for WasiHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a URL string into (scheme, authority, `path_with_query`).
fn parse_url(url: &str) -> Result<(Scheme, String, String), BlazenError> {
    // Minimal URL parsing -- we avoid pulling in a full URL crate for WASM size.
    let (scheme, rest) = if let Some(rest) = url.strip_prefix("https://") {
        (Scheme::Https, rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        (Scheme::Http, rest)
    } else {
        return Err(BlazenError::request(format!(
            "unsupported URL scheme: {url}"
        )));
    };

    let (authority, path_with_query) = match rest.find('/') {
        Some(idx) => (rest[..idx].to_owned(), rest[idx..].to_owned()),
        None => (rest.to_owned(), "/".to_owned()),
    };

    Ok((scheme, authority, path_with_query))
}

/// Convert our [`HttpMethod`] to the WASI HTTP [`Method`].
fn to_wasi_method(method: HttpMethod) -> Method {
    match method {
        HttpMethod::Get => Method::Get,
        HttpMethod::Post => Method::Post,
        HttpMethod::Put => Method::Put,
        HttpMethod::Delete => Method::Delete,
        HttpMethod::Patch => Method::Patch,
    }
}

/// Build a WASI `OutgoingRequest` from our abstract `HttpRequest`.
fn build_outgoing_request(request: &HttpRequest) -> Result<OutgoingRequest, BlazenError> {
    let (scheme, authority, mut path_with_query) = parse_url(&request.url)?;

    // Append query parameters
    if !request.query_params.is_empty() {
        let separator = if path_with_query.contains('?') {
            '&'
        } else {
            '?'
        };
        path_with_query.push(separator);
        let pairs: Vec<String> = request
            .query_params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        path_with_query.push_str(&pairs.join("&"));
    }

    // Build headers
    let headers = Fields::new();
    for (name, value) in &request.headers {
        headers
            .append(&name.to_lowercase(), &value.as_bytes().to_vec())
            .map_err(|e| BlazenError::request(format!("failed to set header {name}: {e:?}")))?;
    }

    let outgoing = OutgoingRequest::new(headers);
    outgoing
        .set_method(&to_wasi_method(request.method))
        .map_err(|()| BlazenError::request("failed to set HTTP method"))?;
    outgoing
        .set_scheme(Some(&scheme))
        .map_err(|()| BlazenError::request("failed to set scheme"))?;
    outgoing
        .set_authority(Some(&authority))
        .map_err(|()| BlazenError::request("failed to set authority"))?;
    outgoing
        .set_path_with_query(Some(&path_with_query))
        .map_err(|()| BlazenError::request("failed to set path"))?;

    Ok(outgoing)
}

/// Write the request body to the outgoing request's body stream.
fn write_body(outgoing: &OutgoingRequest, body: &[u8]) -> Result<(), BlazenError> {
    let outgoing_body = outgoing
        .body()
        .map_err(|()| BlazenError::request("failed to get outgoing body"))?;

    let write_stream = outgoing_body
        .write()
        .map_err(|()| BlazenError::request("failed to get body write stream"))?;

    // Write in chunks to avoid blocking
    let mut offset = 0;
    while offset < body.len() {
        let chunk_size = (body.len() - offset).min(16384);
        write_stream
            .write(&body[offset..offset + chunk_size])
            .map_err(|e| BlazenError::request(format!("failed to write body: {e:?}")))?;
        offset += chunk_size;
    }

    // Must drop the write stream before finishing the body
    drop(write_stream);

    OutgoingBody::finish(outgoing_body, None)
        .map_err(|e| BlazenError::request(format!("failed to finish body: {e:?}")))?;

    Ok(())
}

/// Read the full response body from an `IncomingBody`.
fn read_incoming_body(incoming_body: IncomingBody) -> Result<Vec<u8>, BlazenError> {
    let stream = incoming_body
        .stream()
        .map_err(|()| BlazenError::request("failed to get incoming body stream"))?;

    let mut body = Vec::new();
    loop {
        match stream.read(65536) {
            Ok(chunk) => {
                if chunk.is_empty() {
                    // Poll the stream's readiness
                    let pollable = stream.subscribe();
                    pollable.block();
                    continue;
                }
                body.extend_from_slice(&chunk);
            }
            Err(StreamError::Closed) => break,
            Err(StreamError::LastOperationFailed(e)) => {
                return Err(BlazenError::request(format!(
                    "failed to read response body: {e:?}"
                )));
            }
        }
    }

    // Must drop stream before consuming the body's trailers
    drop(stream);

    // Consume trailers (required to complete the incoming body)
    IncomingBody::finish(incoming_body);

    Ok(body)
}

/// Extract headers from a WASI HTTP incoming response.
fn extract_headers(fields: &Fields) -> Vec<(String, String)> {
    fields
        .entries()
        .into_iter()
        .map(|(name, value)| (name, String::from_utf8_lossy(&value).into_owned()))
        .collect()
}

#[async_trait::async_trait]
impl HttpClient for WasiHttpClient {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError> {
        let outgoing = build_outgoing_request(&request)?;

        // Write body if present
        if let Some(ref body) = request.body {
            write_body(&outgoing, body)?;
        } else {
            // Must still finish the body even with no content
            let outgoing_body = outgoing
                .body()
                .map_err(|()| BlazenError::request("failed to get outgoing body"))?;
            OutgoingBody::finish(outgoing_body, None)
                .map_err(|e| BlazenError::request(format!("failed to finish body: {e:?}")))?;
        }

        // Send the request (no timeout -- the host runtime manages that)
        let future_response = outgoing_handler::handle(outgoing, None)
            .map_err(|e| BlazenError::request(format!("outgoing handler error: {e:?}")))?;

        // Block on the response future (WASI preview2 polling)
        let pollable = future_response.subscribe();
        pollable.block();

        let response = future_response
            .get()
            .ok_or_else(|| BlazenError::request("response future not ready"))?
            .map_err(|()| BlazenError::request("response future error"))?
            .map_err(|e| BlazenError::request(format!("HTTP error: {e:?}")))?;

        let status = response.status();
        let headers = extract_headers(&response.headers());

        let incoming_body = response
            .consume()
            .map_err(|()| BlazenError::request("failed to consume response body"))?;

        let body = read_incoming_body(incoming_body)?;

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn send_streaming(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError> {
        // For the MVP, we buffer the entire response and wrap it as a stream.
        // True streaming would require a background task reading from the WASI
        // input stream and feeding chunks through a channel, which is complex
        // in the single-threaded WASM environment. We can optimize this later
        // once wasi:io/streams async integration matures.
        let response = self.send(request).await?;
        let status = response.status;
        let headers = response.headers;

        // Wrap the buffered body as a single-item byte stream
        let body_bytes = bytes::Bytes::from(response.body);
        let stream = futures_util::stream::once(async move {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(body_bytes)
        });

        Ok((status, headers, Box::pin(stream)))
    }
}
