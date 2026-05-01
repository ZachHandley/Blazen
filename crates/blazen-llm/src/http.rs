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
    Delete,
    Patch,
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

    /// Convenience: create a DELETE request.
    pub fn delete(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Delete,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            query_params: Vec::new(),
        }
    }

    /// Convenience: create a PATCH request.
    pub fn patch(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Patch,
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
// Client config
// ---------------------------------------------------------------------------

/// Configuration applied when constructing an [`HttpClient`].
///
/// `request_timeout` caps the wall-clock duration of a single HTTP request
/// (including reading the response body). `connect_timeout` caps the TCP /
/// TLS connection-establishment phase. `None` means *no timeout* — the
/// underlying client will wait indefinitely.
///
/// `user_agent` is sent as the `User-Agent` header on every request when
/// `Some`. When `None`, the underlying client's default User-Agent is used.
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Maximum wall-clock duration for a single request. `None` = unlimited.
    pub request_timeout: Option<std::time::Duration>,
    /// Maximum duration for the connection-establishment phase. `None` = unlimited.
    pub connect_timeout: Option<std::time::Duration>,
    /// User-Agent header string. `None` uses the underlying client's default.
    pub user_agent: Option<String>,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            request_timeout: Some(std::time::Duration::from_secs(60)),
            connect_timeout: Some(std::time::Duration::from_secs(10)),
            user_agent: None,
        }
    }
}

impl HttpClientConfig {
    /// Construct a config with no request or connect timeout.
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            request_timeout: None,
            connect_timeout: None,
            user_agent: None,
        }
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

    /// Return the configuration (timeouts, user-agent) this client was built
    /// with. Default impl returns a process-wide default config; concrete
    /// implementations should override.
    fn config(&self) -> &HttpClientConfig {
        static DEFAULT: std::sync::LazyLock<HttpClientConfig> =
            std::sync::LazyLock::new(HttpClientConfig::default);
        &DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn config_default_has_request_and_connect_timeouts() {
        let cfg = HttpClientConfig::default();
        assert_eq!(cfg.request_timeout, Some(Duration::from_secs(60)));
        assert_eq!(cfg.connect_timeout, Some(Duration::from_secs(10)));
        assert!(cfg.user_agent.is_none());
    }

    #[test]
    fn config_unlimited_has_no_timeouts() {
        let cfg = HttpClientConfig::unlimited();
        assert!(cfg.request_timeout.is_none());
        assert!(cfg.connect_timeout.is_none());
    }
}
