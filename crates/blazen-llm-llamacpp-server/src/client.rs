//! Thin wrapper over [`reqwest::Client`] tailored to the `llama-server`
//! HTTP API.
//!
//! Covers the endpoints the proxy provider needs:
//!
//! - `POST /v1/chat/completions`     — OpenAI-compat chat (JSON or SSE)
//! - `POST /v1/completions`          — OpenAI-compat text completion
//! - `POST /v1/embeddings`           — OpenAI-compat embeddings
//! - `GET  /v1/models`               — base model listing
//! - `POST /completion`              — llama.cpp-native completion
//! - `GET  /health`                  — readiness probe
//! - `GET  /slots`                   — running-slot introspection
//! - `GET  /lora-adapters`           — list preloaded adapters + scales
//! - `POST /lora-adapters`           — toggle active set / scales
//!
//! Bodies are `serde_json::Value` rather than typed structs because
//! Blazen's provider-agnostic [`ModelRequest`] / [`ModelResponse`]
//! types live in `blazen-llm` (which this crate cannot depend on without
//! creating a cycle). The `backends/llamacpp_server.rs` bridge does the
//! typed conversion.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::LlamacppServerError;
use crate::options::LlamacppServerOptions;

/// Reusable HTTP client wrapping `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct LlamacppServerClient {
    inner: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    request_timeout: Duration,
    adapter_timeout: Duration,
}

/// One row from `GET /v1/models` (OpenAI-shaped). `llama-server` reports
/// the loaded model under this surface; only fields the proxy reads are
/// typed.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamacppServerModelEntry {
    /// Model identifier (e.g. `"llama-3.2"`, set by `--alias` or
    /// defaulting to the basename of `--model`).
    pub id: String,
    /// `"model"` for OpenAI-compat rows; reserved for forward compat.
    #[serde(default)]
    pub object: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<LlamacppServerModelEntry>,
}

/// Response from `GET /health`. `llama-server` returns
/// `{"status": "ok"}` on a ready server and a non-2xx body when the
/// model is still loading or has crashed.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamacppServerHealth {
    /// `"ok"`, `"loading model"`, `"no slot available"`, etc.
    pub status: String,
}

/// One slot row from `GET /slots`. `llama-server` exposes a free-form
/// payload per slot; only the always-present integer id is typed and
/// the rest passed through as [`Value`] via [`Self::raw`].
#[derive(Debug, Clone, Deserialize)]
pub struct LlamacppServerSlot {
    /// Integer slot id (0-indexed up to `--parallel`).
    pub id: u32,
    /// Full upstream payload for callers needing fields not typed here.
    #[serde(flatten)]
    pub raw: Value,
}

/// One row from `GET /lora-adapters`. `llama-server` (>= b4334) reports
/// each adapter preloaded at startup via `--lora` / `--lora-scaled`.
/// The integer `id` is the index used by the toggle endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlamacppServerLoraAdapter {
    /// Integer adapter id (0-indexed, in CLI declaration order).
    pub id: u32,
    /// Filesystem path the adapter was loaded from.
    pub path: String,
    /// Current scale. `0.0` means inactive; positive values are blended
    /// into the base weights on each forward pass.
    pub scale: f32,
}

/// Toggle body for `POST /lora-adapters`. The wire format is a top-level
/// array of `{id, scale}` objects; this struct is serialised as one
/// array element.
#[derive(Debug, Clone, Serialize)]
pub struct LlamacppServerLoraToggle {
    /// Integer adapter id (must match a row from `GET /lora-adapters`).
    pub id: u32,
    /// New scale. `0.0` disables the adapter; any positive value
    /// activates it at that weight.
    pub scale: f32,
}

impl LlamacppServerClient {
    /// Build a client from validated options.
    ///
    /// # Errors
    /// Returns [`LlamacppServerError::Init`] if the underlying
    /// `reqwest::Client` cannot be built (e.g. TLS init failed).
    pub fn new(opts: &LlamacppServerOptions) -> Result<Self, LlamacppServerError> {
        opts.validate()?;
        // The per-call timeout is set on each `RequestBuilder`, not on
        // the Client itself, because adapter calls use a different
        // budget than completions.
        let inner = reqwest::Client::builder()
            .build()
            .map_err(|e| LlamacppServerError::Init(e.to_string()))?;
        Ok(Self {
            inner,
            endpoint: opts.endpoint_trimmed().to_string(),
            api_key: opts.api_key.clone(),
            request_timeout: opts.request_timeout,
            adapter_timeout: opts.adapter_timeout,
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

    // -----------------------------------------------------------------
    // OpenAI-compatible surface
    // -----------------------------------------------------------------

    /// POST `/v1/chat/completions` with `stream: false`. The supplied
    /// body must include `model` and `messages`; the helper does NOT
    /// override the `stream` field — the caller composes the body.
    ///
    /// # Errors
    /// - [`LlamacppServerError::Request`] on transport failure.
    /// - [`LlamacppServerError::NotFound`] on HTTP 404.
    /// - [`LlamacppServerError::Http`] on other non-2xx responses.
    /// - [`LlamacppServerError::Decode`] when the body isn't valid JSON.
    pub async fn chat_completions(&self, body: &Value) -> Result<Value, LlamacppServerError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// POST `/v1/chat/completions` with `stream: true`. Returns the raw
    /// `reqwest::Response` so the caller can drive the SSE parser
    /// (`data: <json>\n\n` framed, terminated by `data: [DONE]`).
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn chat_completions_stream(
        &self,
        body: &Value,
    ) -> Result<reqwest::Response, LlamacppServerError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        self.post_stream(&url, body, self.request_timeout).await
    }

    /// POST `/v1/completions` (OpenAI-compat legacy text completion).
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn completions(&self, body: &Value) -> Result<Value, LlamacppServerError> {
        let url = format!("{}/v1/completions", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// POST `/v1/embeddings`. The body must include `model` and `input`
    /// (a single string or an array of strings).
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn embeddings(&self, body: &Value) -> Result<Value, LlamacppServerError> {
        let url = format!("{}/v1/embeddings", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// GET `/v1/models`. `llama-server` typically returns a single row
    /// for the currently-loaded model.
    ///
    /// # Errors
    /// - [`LlamacppServerError::Http`] on non-2xx.
    /// - [`LlamacppServerError::Decode`] when the payload doesn't match
    ///   `{ data: [...] }`.
    pub async fn list_models(&self) -> Result<Vec<LlamacppServerModelEntry>, LlamacppServerError> {
        let url = format!("{}/v1/models", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
        }
        let parsed: ModelsListResponse = serde_json::from_str(&text)?;
        Ok(parsed.data)
    }

    // -----------------------------------------------------------------
    // llama.cpp-native surface
    // -----------------------------------------------------------------

    /// POST `/completion` (llama.cpp's native completion endpoint, NOT
    /// OpenAI-compat). The body shape is `{prompt, n_predict, ...}`.
    /// Streaming uses the `stream` flag — when `true` the response is
    /// SSE-framed with `data: <json>\n\n` lines (terminated by an
    /// empty JSON object).
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn completion(&self, body: &Value) -> Result<Value, LlamacppServerError> {
        let url = format!("{}/completion", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// GET `/health`. Used for readiness probes.
    ///
    /// # Errors
    /// - [`LlamacppServerError::Http`] when the server is up but the
    ///   model is still loading (`llama-server` responds with HTTP 503
    ///   and a `{"status":"loading model"}` body).
    /// - [`LlamacppServerError::Decode`] on schema mismatch.
    pub async fn health(&self) -> Result<LlamacppServerHealth, LlamacppServerError> {
        let url = format!("{}/health", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
        }
        let parsed: LlamacppServerHealth = serde_json::from_str(&text)?;
        Ok(parsed)
    }

    /// GET `/slots`. Returns one row per running slot (0-indexed up to
    /// `--parallel`). `llama-server` may disable this endpoint via
    /// `--no-slots`; in that case it returns HTTP 501.
    ///
    /// # Errors
    /// - [`LlamacppServerError::Http`] on non-2xx (including 501 when
    ///   slots-introspection is disabled).
    /// - [`LlamacppServerError::Decode`] on schema mismatch.
    pub async fn slots(&self) -> Result<Vec<LlamacppServerSlot>, LlamacppServerError> {
        let url = format!("{}/slots", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
        }
        let parsed: Vec<LlamacppServerSlot> = serde_json::from_str(&text)?;
        Ok(parsed)
    }

    /// GET `/lora-adapters`. Returns the set of `LoRA` adapters
    /// preloaded at server startup via `--lora` / `--lora-scaled`,
    /// each with its current scale.
    ///
    /// # Errors
    /// - [`LlamacppServerError::Http`] on non-2xx.
    /// - [`LlamacppServerError::Decode`] on schema mismatch.
    pub async fn list_lora_adapters(
        &self,
    ) -> Result<Vec<LlamacppServerLoraAdapter>, LlamacppServerError> {
        let url = format!("{}/lora-adapters", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.adapter_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
        }
        let parsed: Vec<LlamacppServerLoraAdapter> = serde_json::from_str(&text)?;
        Ok(parsed)
    }

    /// POST `/lora-adapters`. The body is a JSON array of
    /// `{id, scale}` objects describing the *new* active set — entries
    /// not present default to `scale: 0.0` (inactive). Pass an empty
    /// slice to disable every adapter.
    ///
    /// # Errors
    /// - [`LlamacppServerError::AdapterFailed`] when `llama-server`
    ///   returns non-2xx (e.g. id out of range, scale invalid).
    /// - [`LlamacppServerError::Request`] on transport failure.
    pub async fn set_lora_adapters(
        &self,
        toggles: &[LlamacppServerLoraToggle],
    ) -> Result<(), LlamacppServerError> {
        let url = format!("{}/lora-adapters", self.endpoint);
        let resp = self
            .auth(self.inner.post(&url).timeout(self.adapter_timeout))
            .json(toggles)
            .send()
            .await?;
        let status = resp.status();
        if status.is_success() {
            return Ok(());
        }
        let text = resp.text().await.unwrap_or_default();
        Err(LlamacppServerError::AdapterFailed(format!(
            "POST /lora-adapters -> HTTP {status}: {}",
            cap(text, 4 * 1024)
        )))
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
    ) -> Result<Value, LlamacppServerError> {
        let resp = self
            .auth(self.inner.post(url).timeout(timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
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
    ) -> Result<reqwest::Response, LlamacppServerError> {
        let resp = self
            .auth(self.inner.post(url).timeout(timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(http_or_not_found(status.as_u16(), text));
        }
        Ok(resp)
    }
}

/// Map a non-2xx status to the right error variant. 404 becomes
/// [`LlamacppServerError::NotFound`] so the bridge can return
/// `BlazenError::ModelNotFound`.
fn http_or_not_found(status: u16, body: String) -> LlamacppServerError {
    if status == 404 {
        LlamacppServerError::NotFound(cap(body, 4 * 1024))
    } else {
        LlamacppServerError::Http {
            status,
            body: cap(body, 4 * 1024),
        }
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
    fn http_or_not_found_routes_404() {
        let e = http_or_not_found(404, "model not found".into());
        assert!(matches!(e, LlamacppServerError::NotFound(_)));
    }

    #[test]
    fn http_or_not_found_routes_other() {
        let e = http_or_not_found(500, "boom".into());
        assert!(matches!(e, LlamacppServerError::Http { status: 500, .. }));
    }

    #[test]
    fn client_build_validates_options() {
        let bad = LlamacppServerOptions::required("", "model");
        assert!(LlamacppServerClient::new(&bad).is_err());
        let ok = LlamacppServerOptions::required("http://localhost:8080", "model");
        assert!(LlamacppServerClient::new(&ok).is_ok());
    }
}
