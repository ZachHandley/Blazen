//! Thin wrapper over [`reqwest::Client`] tailored to the TGI HTTP API.
//!
//! Covers the seven endpoints the proxy provider needs:
//!
//! - `POST /generate`              — native TGI completion (single)
//! - `POST /generate_stream`       — native TGI SSE token stream
//! - `POST /v1/chat/completions`   — OpenAI-compatible chat (single + SSE)
//! - `POST /v1/completions`        — OpenAI-compatible legacy completion
//! - `GET  /info`                  — model id + runtime config
//! - `GET  /v1/models`             — base + loaded adapters listing
//! - `GET  /metrics`               — Prometheus text format (optional)
//!
//! Bodies are `serde_json::Value` rather than typed structs because
//! Blazen's provider-agnostic [`ModelRequest`] / [`ModelResponse`]
//! types live in `blazen-llm` (which this crate cannot depend on without
//! creating a cycle). The `backends/tgi.rs` bridge does the typed
//! conversion.

use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::TgiError;
use crate::options::TgiOptions;

/// Reusable HTTP client wrapping `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct TgiClient {
    inner: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    request_timeout: Duration,
    meta_timeout: Duration,
}

/// One row from `GET /v1/models`. TGI emits the OpenAI-shaped
/// `{data: [{id, object, ...}]}` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct TgiModelEntry {
    /// Model id — either the base model or a preloaded adapter id.
    pub id: String,
    /// `"model"` for the base, sometimes `"lora"` for an adapter row
    /// (TGI ≥ 2.0). Preserved verbatim for callers that want to filter.
    #[serde(default)]
    pub object: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<TgiModelEntry>,
}

/// Response from `GET /info`. Only the fields the bridge actually
/// reads are typed; TGI emits ~20 keys (`version`, `sha`, `model_dtype`,
/// `max_*_tokens`, ...) which we forward via [`Self::raw`].
#[derive(Debug, Clone, Deserialize)]
pub struct TgiInfo {
    /// Base model id the server was launched with.
    #[serde(default)]
    pub model_id: Option<String>,
    /// TGI version string (e.g. `"2.3.1"`).
    #[serde(default)]
    pub version: Option<String>,
    /// Max input tokens the server will accept (`--max-input-tokens`).
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    /// Max total tokens (`--max-total-tokens`).
    #[serde(default)]
    pub max_total_tokens: Option<u32>,
    /// Full upstream payload for callers that need fields we don't type.
    #[serde(flatten)]
    pub raw: Value,
}

impl TgiClient {
    /// Build a client from validated options.
    ///
    /// # Errors
    /// Returns [`TgiError::Init`] if the underlying `reqwest::Client`
    /// cannot be built (e.g. TLS init failed).
    pub fn new(opts: &TgiOptions) -> Result<Self, TgiError> {
        opts.validate()?;
        let inner = reqwest::Client::builder()
            .build()
            .map_err(|e| TgiError::Init(e.to_string()))?;
        Ok(Self {
            inner,
            endpoint: opts.endpoint_trimmed().to_string(),
            api_key: opts.api_key.clone(),
            request_timeout: opts.request_timeout,
            meta_timeout: opts.meta_timeout,
        })
    }

    /// Endpoint URL with any trailing slash already stripped.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Apply bearer-auth to a request builder when an API key is set.
    fn auth(&self, mut rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            rb = rb.bearer_auth(key);
        }
        rb
    }

    /// POST `/generate` (native TGI shape, single response).
    ///
    /// The supplied body must include `inputs`; sampling parameters go
    /// inside the nested `parameters` object per the TGI schema. To
    /// route through a preloaded adapter, set `body["adapter_id"]`.
    ///
    /// # Errors
    /// - [`TgiError::Request`] on transport failure.
    /// - [`TgiError::NotFound`] / [`TgiError::Validation`] /
    ///   [`TgiError::Overloaded`] / [`TgiError::Http`] on non-2xx.
    /// - [`TgiError::Decode`] when the response body isn't valid JSON.
    pub async fn generate(&self, body: &Value) -> Result<Value, TgiError> {
        let url = format!("{}/generate", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// POST `/generate_stream` (native TGI shape, SSE-framed token
    /// stream). Returns the raw `reqwest::Response` so the caller can
    /// drive the SSE parser.
    ///
    /// # Errors
    /// As [`Self::generate`], minus [`TgiError::Decode`] (the caller
    /// drains the stream).
    pub async fn generate_stream(&self, body: &Value) -> Result<reqwest::Response, TgiError> {
        let url = format!("{}/generate_stream", self.endpoint);
        self.post_stream(&url, body, self.request_timeout).await
    }

    /// POST `/v1/chat/completions` with `stream: false`. Forces the
    /// stream flag off so a single JSON object comes back.
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn chat_completions(&self, body: &Value) -> Result<Value, TgiError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(false);
        self.post_json_single(&url, &body, self.request_timeout)
            .await
    }

    /// POST `/v1/chat/completions` with `stream: true`. Returns the
    /// raw `reqwest::Response` for the caller to drive the SSE parser
    /// (the streaming wire format is OpenAI-shaped SSE).
    ///
    /// # Errors
    /// As [`Self::generate_stream`].
    pub async fn chat_completions_stream(
        &self,
        body: &Value,
    ) -> Result<reqwest::Response, TgiError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(true);
        self.post_stream(&url, &body, self.request_timeout).await
    }

    /// POST `/v1/completions` (OpenAI-compatible legacy completion).
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn completions(&self, body: &Value) -> Result<Value, TgiError> {
        let url = format!("{}/v1/completions", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(false);
        self.post_json_single(&url, &body, self.request_timeout)
            .await
    }

    /// GET `/info`. Returns the runtime configuration & model id.
    ///
    /// # Errors
    /// - [`TgiError::Http`] on non-2xx.
    /// - [`TgiError::Decode`] when the payload doesn't match the typed
    ///   subset.
    pub async fn info(&self) -> Result<TgiInfo, TgiError> {
        let url = format!("{}/info", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.meta_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_error(status.as_u16(), text));
        }
        let parsed: TgiInfo = serde_json::from_str(&text)?;
        Ok(parsed)
    }

    /// GET `/v1/models`. Returns the base model + any preloaded
    /// adapters as `TgiModelEntry` rows.
    ///
    /// # Errors
    /// As [`Self::info`].
    pub async fn list_models(&self) -> Result<Vec<TgiModelEntry>, TgiError> {
        let url = format!("{}/v1/models", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.meta_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_error(status.as_u16(), text));
        }
        let parsed: ModelsListResponse = serde_json::from_str(&text)?;
        Ok(parsed.data)
    }

    /// GET `/metrics`. Returns the Prometheus text-format body
    /// verbatim. Optional — the server only exposes this when started
    /// without `--no-metrics`.
    ///
    /// # Errors
    /// As [`Self::info`], minus [`TgiError::Decode`] (the body is
    /// plaintext, not JSON).
    pub async fn metrics(&self) -> Result<String, TgiError> {
        let url = format!("{}/metrics", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.meta_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_error(status.as_u16(), text));
        }
        Ok(text)
    }

    // -----------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------

    /// POST a JSON body, expect a single JSON object in the response.
    async fn post_json_single(
        &self,
        url: &str,
        body: &Value,
        timeout: Duration,
    ) -> Result<Value, TgiError> {
        let resp = self
            .auth(self.inner.post(url).timeout(timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_error(status.as_u16(), text));
        }
        Ok(serde_json::from_str(&text)?)
    }

    /// POST a JSON body, returning the raw response for the caller to
    /// drive a streaming parser.
    async fn post_stream(
        &self,
        url: &str,
        body: &Value,
        timeout: Duration,
    ) -> Result<reqwest::Response, TgiError> {
        let resp = self
            .auth(self.inner.post(url).timeout(timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(http_error(status.as_u16(), text));
        }
        Ok(resp)
    }
}

/// Map a non-2xx status to the most specific [`TgiError`] variant.
fn http_error(status: u16, body: String) -> TgiError {
    let body = cap(body, 4 * 1024);
    match status {
        404 => TgiError::NotFound(body),
        422 => TgiError::Validation(body),
        429 => TgiError::Overloaded(body),
        _ => TgiError::Http { status, body },
    }
}

/// Cap a string to `n` bytes for inclusion in error messages.
fn cap(mut s: String, n: usize) -> String {
    if s.len() > n {
        s.truncate(n);
        s.push_str("...[truncated]");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_passthrough_short_string() {
        assert_eq!(cap("hi".into(), 10), "hi");
    }

    #[test]
    fn cap_truncates_long_string() {
        let out = cap("a".repeat(20), 5);
        assert!(out.starts_with("aaaaa"));
        assert!(out.contains("truncated"));
    }

    #[test]
    fn http_error_routes_404_to_not_found() {
        let e = http_error(404, "no such model".into());
        assert!(matches!(e, TgiError::NotFound(_)));
    }

    #[test]
    fn http_error_routes_422_to_validation() {
        let e = http_error(422, "invalid adapter".into());
        assert!(matches!(e, TgiError::Validation(_)));
    }

    #[test]
    fn http_error_routes_429_to_overloaded() {
        let e = http_error(429, "queue full".into());
        assert!(matches!(e, TgiError::Overloaded(_)));
    }

    #[test]
    fn http_error_routes_other_to_http() {
        let e = http_error(500, "boom".into());
        assert!(matches!(e, TgiError::Http { status: 500, .. }));
    }

    #[test]
    fn client_build_validates_options() {
        let bad = TgiOptions::required("", "model");
        assert!(TgiClient::new(&bad).is_err());
        let ok = TgiOptions::required("http://localhost:8080", "model");
        assert!(TgiClient::new(&ok).is_ok());
    }
}
