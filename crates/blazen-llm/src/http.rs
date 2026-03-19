//! Abstract HTTP client layer.
//!
//! Provides an [`HttpClient`] trait that decouples provider logic from a
//! specific HTTP implementation. The default implementation wraps
//! [`reqwest::Client`] (see [`super::http_reqwest::ReqwestHttpClient`]),
//! but consumers can provide their own backend (e.g. WASI HTTP for WASM
//! targets) by implementing this trait.

use bytes::Bytes;
use futures_util::Stream;
use std::pin::Pin;

use crate::error::BlazenError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A boxed, `Send`-able byte stream used for SSE / streaming responses.
pub type ByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, Box<dyn std::error::Error + Send + Sync>>> + Send>>;

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
}

/// An outgoing HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub query_params: Vec<(String, String)>,
}

impl HttpRequest {
    /// Convenience: create a POST request with JSON body.
    pub fn post(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            query_params: Vec::new(),
        }
    }

    /// Convenience: create a GET request.
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Get,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            query_params: Vec::new(),
        }
    }

    /// Convenience: create a PUT request.
    pub fn put(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Put,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            query_params: Vec::new(),
        }
    }

    /// Add a header.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set a `Bearer` auth header.
    #[must_use]
    pub fn bearer_auth(self, token: &str) -> Self {
        self.header("Authorization", format!("Bearer {token}"))
    }

    /// Set a JSON body (serialises the value and sets `Content-Type`).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Serialization`] if the value cannot be serialized.
    pub fn json_body(mut self, value: &impl serde::Serialize) -> Result<Self, BlazenError> {
        let bytes = serde_json::to_vec(value)?;
        self.body = Some(bytes);
        self.headers
            .push(("Content-Type".to_owned(), "application/json".to_owned()));
        Ok(self)
    }

    /// Set a raw body.
    #[must_use]
    pub fn body(mut self, bytes: Vec<u8>) -> Self {
        self.body = Some(bytes);
        self
    }

    /// Add a query parameter.
    #[must_use]
    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.push((key.into(), value.into()));
        self
    }
}

/// A complete HTTP response (body fully buffered).
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Whether the status code indicates success (2xx).
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Deserialise the body as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Serialization`] if the body is not valid JSON
    /// or does not match the target type.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, BlazenError> {
        serde_json::from_slice(&self.body).map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Interpret the body as a UTF-8 string (lossy).
    #[must_use]
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    /// Look up a header by name (case-insensitive).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// An abstract HTTP client.
///
/// Providers hold an `Arc<dyn HttpClient>` and use it for all network I/O.
/// The default [`ReqwestHttpClient`](super::http_reqwest::ReqwestHttpClient)
/// wraps [`reqwest::Client`]; a WASI HTTP implementation can be plugged in
/// for WASM targets.
#[async_trait::async_trait]
pub trait HttpClient: Send + Sync + std::fmt::Debug {
    /// Send a request and return a fully-buffered response.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, BlazenError>;

    /// Send a request and return a streaming byte response (for SSE).
    ///
    /// Returns `(status_code, headers, byte_stream)`.
    async fn send_streaming(
        &self,
        request: HttpRequest,
    ) -> Result<(u16, Vec<(String, String)>, ByteStream), BlazenError>;
}
