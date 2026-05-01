//! Plain-object napi mirror of [`blazen_llm::http::HttpClientConfig`].
//!
//! This sits next to [`super::http_client`] (which exposes the
//! subclassable [`crate::types::JsHttpClient`] base for fully-custom JS
//! transports) and represents the *configuration* used when constructing
//! the default reqwest-backed transport: timeouts and an optional
//! `User-Agent` header.

use napi_derive::napi;

use blazen_llm::http::HttpClientConfig;

// ---------------------------------------------------------------------------
// JsHttpClientConfig
// ---------------------------------------------------------------------------

/// Configuration applied when constructing the default HTTP client.
///
/// Mirrors [`HttpClientConfig`]. All fields are optional in JS — pass
/// `null` / `undefined` for any field to mean "no timeout / no UA
/// override".
///
/// ```typescript
/// const cfg: HttpClientConfig = {
///   requestTimeoutMs: 30_000,
///   connectTimeoutMs: 5_000,
///   userAgent: "my-app/1.0",
/// };
/// const unlimited = HttpClientConfig.unlimited();
/// ```
#[napi(object, js_name = "HttpClientConfig")]
pub struct JsHttpClientConfig {
    /// Maximum wall-clock duration for a single request, in milliseconds.
    /// `null` / `undefined` means unlimited.
    #[napi(js_name = "requestTimeoutMs")]
    pub request_timeout_ms: Option<f64>,
    /// Maximum duration for the connection-establishment phase, in
    /// milliseconds. `null` / `undefined` means unlimited.
    #[napi(js_name = "connectTimeoutMs")]
    pub connect_timeout_ms: Option<f64>,
    /// User-Agent header string. `null` / `undefined` uses the underlying
    /// client's default.
    #[napi(js_name = "userAgent")]
    pub user_agent: Option<String>,
}

impl Default for JsHttpClientConfig {
    fn default() -> Self {
        Self::from(HttpClientConfig::default())
    }
}

#[allow(clippy::cast_precision_loss)]
impl From<HttpClientConfig> for JsHttpClientConfig {
    fn from(c: HttpClientConfig) -> Self {
        Self {
            request_timeout_ms: c.request_timeout.map(|d| d.as_millis() as f64),
            connect_timeout_ms: c.connect_timeout.map(|d| d.as_millis() as f64),
            user_agent: c.user_agent,
        }
    }
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
impl From<JsHttpClientConfig> for HttpClientConfig {
    fn from(c: JsHttpClientConfig) -> Self {
        Self {
            request_timeout: c
                .request_timeout_ms
                .filter(|v| *v >= 0.0)
                .map(|v| std::time::Duration::from_millis(v as u64)),
            connect_timeout: c
                .connect_timeout_ms
                .filter(|v| *v >= 0.0)
                .map(|v| std::time::Duration::from_millis(v as u64)),
            user_agent: c.user_agent,
        }
    }
}

// ---------------------------------------------------------------------------
// Free function constructors — mirror Rust's `Default` and `unlimited()`
// associated functions.
// ---------------------------------------------------------------------------

/// Build a default [`JsHttpClientConfig`] (60s request, 10s connect, no UA).
#[napi(js_name = "defaultHttpClientConfig")]
#[must_use]
pub fn default_http_client_config() -> JsHttpClientConfig {
    JsHttpClientConfig::default()
}

/// Build a [`JsHttpClientConfig`] with no request or connect timeout.
/// Mirrors [`HttpClientConfig::unlimited`].
#[napi(js_name = "unlimitedHttpClientConfig")]
#[must_use]
pub fn unlimited_http_client_config() -> JsHttpClientConfig {
    HttpClientConfig::unlimited().into()
}
