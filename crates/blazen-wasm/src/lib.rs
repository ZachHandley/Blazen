//! # Blazen WASM Component
//!
//! A WASIp2 WASM component that implements `wasi:http/incoming-handler` to
//! serve an OpenAI-compatible API backed by Blazen's LLM providers. Outbound
//! LLM API calls are made via `wasi:http/outgoing-handler`.
//!
//! ## Architecture
//!
//! ```text
//! [Client] --(HTTP)--> [wasmtime host]
//!                           |
//!                    wasi:http/incoming-handler
//!                           |
//!                      [blazen-wasm]
//!                           |
//!                    wasi:http/outgoing-handler
//!                           |
//!                   [LLM Provider APIs]
//! ```
//!
//! ## Endpoints
//!
//! | Method | Path                       | Description           |
//! |--------|----------------------------|-----------------------|
//! | GET    | /health                    | Health check          |
//! | POST   | /v1/chat/completions       | Chat completion       |
//! | POST   | /v1/images/generations     | Image generation      |
//! | POST   | /v1/audio/speech           | Text-to-speech        |
//! | POST   | /v1/agent/run              | Agent execution       |

pub mod http_wasi;
pub mod keys;
pub mod router;

// Generate WASI bindings for the `blazen-handler` world which includes
// `wasi:http/proxy` (both incoming-handler and outgoing-handler).
wit_bindgen::generate!({
    world: "blazen-handler",
    path: "wit",
});

use std::sync::Arc;

use crate::bindings::exports::wasi::http::incoming_handler::Guest;
use crate::bindings::wasi::http::types::{
    Fields, IncomingBody, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam,
};
use crate::bindings::wasi::io::streams::StreamError;

use http_wasi::WasiHttpClient;
use keys::KeyProvider;
use router::{RouteResponse, route};

// ---------------------------------------------------------------------------
// Component implementation
// ---------------------------------------------------------------------------

/// The Blazen WASM component that handles incoming HTTP requests.
struct BlazenComponent;

export!(BlazenComponent);

impl Guest for BlazenComponent {
    /// Handle an incoming HTTP request.
    ///
    /// This is the entry point called by the wasmtime runtime for each
    /// incoming HTTP request. It reads the request, routes it to the
    /// appropriate handler, and writes the response.
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        // Extract method and path from the incoming request
        let method = format!("{:?}", request.method());
        let method_str = match method.as_str() {
            "Get" => "GET",
            "Post" => "POST",
            "Put" => "PUT",
            "Delete" => "DELETE",
            "Patch" => "PATCH",
            "Head" => "HEAD",
            "Options" => "OPTIONS",
            _ => "UNKNOWN",
        };

        let path = request.path_with_query().unwrap_or_else(|| "/".to_owned());

        // Strip query string for routing
        let route_path = path.split('?').next().unwrap_or("/");

        // Read the request body
        let body = read_request_body(&request);

        // Initialize the key provider and HTTP client
        let key_provider = KeyProvider::from_env();
        let http_client: Arc<dyn blazen_llm::http::HttpClient> = WasiHttpClient::new().into_arc();

        // Route the request
        let result = route(method_str, route_path, &body, &key_provider, &http_client);

        // Write the response
        write_response(response_out, result);
    }
}

// ---------------------------------------------------------------------------
// Request body reading
// ---------------------------------------------------------------------------

/// Read the entire body from an incoming request.
fn read_request_body(request: &IncomingRequest) -> Vec<u8> {
    let incoming_body = match request.consume() {
        Ok(body) => body,
        Err(()) => return Vec::new(),
    };

    let stream = match incoming_body.stream() {
        Ok(s) => s,
        Err(()) => return Vec::new(),
    };

    let mut body = Vec::new();
    loop {
        match stream.read(65536) {
            Ok(chunk) => {
                if chunk.is_empty() {
                    let pollable = stream.subscribe();
                    pollable.block();
                    continue;
                }
                body.extend_from_slice(&chunk);
            }
            Err(StreamError::Closed) => break,
            Err(StreamError::LastOperationFailed(_)) => break,
        }
    }

    drop(stream);
    IncomingBody::finish(incoming_body);

    body
}

// ---------------------------------------------------------------------------
// Response writing
// ---------------------------------------------------------------------------

/// Write a `RouteResponse` to the WASI HTTP response outparam.
fn write_response(response_out: ResponseOutparam, result: RouteResponse) {
    let headers = Fields::new();

    // Set Content-Type
    let _ = headers.append("content-type", result.content_type.as_bytes());

    // Set Content-Length
    let len_str = result.body.len().to_string();
    let _ = headers.append("content-length", len_str.as_bytes());

    let response = OutgoingResponse::new(headers);
    response.set_status_code(result.status).ok();

    let outgoing_body = response.body().expect("failed to get outgoing body");

    // Set the response before writing the body
    ResponseOutparam::set(response_out, Ok(response));

    // Write the body
    if !result.body.is_empty() {
        let write_stream = outgoing_body
            .write()
            .expect("failed to get body write stream");

        let mut offset = 0;
        while offset < result.body.len() {
            let chunk_end = (offset + 16384).min(result.body.len());
            match write_stream.write(&result.body[offset..chunk_end]) {
                Ok(written) => offset += written as usize,
                Err(_) => break,
            }
        }

        drop(write_stream);
    }

    OutgoingBody::finish(outgoing_body, None).ok();
}
