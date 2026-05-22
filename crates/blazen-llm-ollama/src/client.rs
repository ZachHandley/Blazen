//! Thin wrapper over [`reqwest::Client`] tailored to the Ollama HTTP API.
//!
//! Covers the eight endpoints the proxy provider needs:
//!
//! - `POST  /api/generate`     — completion (NDJSON stream)
//! - `POST  /api/chat`         — chat completion (NDJSON stream)
//! - `POST  /api/embeddings`   — vector embedding
//! - `GET   /api/tags`         — list installed models
//! - `POST  /api/show`         — model metadata (template, parameters, ...)
//! - `POST  /api/pull`         — pull a model from the registry (NDJSON
//!   progress stream)
//! - `POST  /api/create`       — create a derived model from a
//!   Modelfile (used to mount adapters via the `ADAPTER` directive)
//! - `DELETE /api/delete`      — remove an installed model
//!
//! Bodies are `serde_json::Value` rather than typed structs because
//! Blazen's provider-agnostic [`ModelRequest`] / [`ModelResponse`]
//! types live in `blazen-llm` (which this crate cannot depend on without
//! creating a cycle). The `backends/ollama.rs` bridge does the typed
//! conversion.

use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::OllamaError;
use crate::options::OllamaOptions;

/// Reusable HTTP client wrapping `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    inner: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    request_timeout: Duration,
    adapter_timeout: Duration,
}

/// One row from `GET /api/tags`. Ollama returns more fields (size,
/// digest, `modified_at`, ...) but the proxy only needs the name.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelEntry {
    /// Fully-qualified model tag, e.g. `"llama3.2:3b"`.
    pub name: String,
    /// SHA256 digest of the underlying manifest, when present.
    #[serde(default)]
    pub digest: Option<String>,
    /// Size in bytes on disk, when present.
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<OllamaModelEntry>,
}

/// Response from `POST /api/show`. Only the fields the bridge actually
/// reads are typed; Ollama may emit additional keys we forward as
/// [`serde_json::Value`] via [`Self::raw`].
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaShowResponse {
    /// Modelfile-derived template string.
    #[serde(default)]
    pub template: Option<String>,
    /// Raw parameter block (free-form key/value).
    #[serde(default)]
    pub parameters: Option<String>,
    /// Full upstream payload for callers that need fields we don't type.
    #[serde(flatten)]
    pub raw: Value,
}

/// One frame from an `/api/pull` NDJSON stream.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaPullProgress {
    /// Human-readable status (`"pulling manifest"`, `"verifying sha256"`,
    /// `"success"`, ...).
    pub status: String,
    /// SHA256 of the current layer being downloaded, when present.
    #[serde(default)]
    pub digest: Option<String>,
    /// Total bytes in the current layer.
    #[serde(default)]
    pub total: Option<u64>,
    /// Bytes received so far.
    #[serde(default)]
    pub completed: Option<u64>,
}

/// One frame from a `/api/generate` or `/api/chat` NDJSON stream.
///
/// Mid-stream frames carry partial output; the terminal frame sets
/// `done: true` and may include `total_duration`, `eval_count`,
/// `prompt_eval_count`, etc. The bridge passes those through verbatim
/// via [`Self::raw`].
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaStreamPart {
    /// Echoed model name.
    #[serde(default)]
    pub model: Option<String>,
    /// Mid-stream token delta for `/api/generate`.
    #[serde(default)]
    pub response: Option<String>,
    /// Mid-stream message delta for `/api/chat`. Ollama wraps the delta
    /// in `{role, content}` even mid-stream; only `content` carries
    /// useful bytes.
    #[serde(default)]
    pub message: Option<OllamaChatMessageDelta>,
    /// `true` on the terminal frame.
    #[serde(default)]
    pub done: bool,
    /// Why the stream ended (`"stop"`, `"length"`, `"load"`, ...). Only
    /// present on the terminal frame.
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Tokens consumed by the prompt (terminal-frame only).
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    /// Tokens produced (terminal-frame only).
    #[serde(default)]
    pub eval_count: Option<u32>,
    /// Full upstream payload for callers needing fields not typed above.
    #[serde(flatten)]
    pub raw: Value,
}

/// Chat-message delta nested inside [`OllamaStreamPart::message`].
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatMessageDelta {
    /// `"assistant"`, `"user"`, `"tool"`, ... .
    pub role: String,
    /// Token chunk for this frame.
    #[serde(default)]
    pub content: String,
}

impl OllamaClient {
    /// Build a client from validated options.
    ///
    /// # Errors
    /// Returns [`OllamaError::Init`] if the underlying `reqwest::Client`
    /// cannot be built (e.g. TLS init failed).
    pub fn new(opts: &OllamaOptions) -> Result<Self, OllamaError> {
        opts.validate()?;
        // The per-call timeout is set on each `RequestBuilder`, not on
        // the Client itself, because adapter / pull calls use a different
        // budget than completions.
        let inner = reqwest::Client::builder()
            .build()
            .map_err(|e| OllamaError::Init(e.to_string()))?;
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

    /// POST `/api/generate` with `stream: false`.
    ///
    /// The supplied body must include `model` and `prompt`; the helper
    /// forces `stream: false` so a single JSON object comes back.
    ///
    /// # Errors
    /// - [`OllamaError::Request`] on transport failure.
    /// - [`OllamaError::NotFound`] on HTTP 404.
    /// - [`OllamaError::Http`] on other non-2xx responses.
    /// - [`OllamaError::Decode`] when the response body isn't valid JSON.
    pub async fn generate(&self, body: &Value) -> Result<Value, OllamaError> {
        let url = format!("{}/api/generate", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(false);
        self.post_json_single(&url, &body, self.request_timeout)
            .await
    }

    /// POST `/api/generate` with `stream: true`. Returns the raw
    /// `reqwest::Response` so the caller can drive the NDJSON parser.
    ///
    /// # Errors
    /// - [`OllamaError::Request`] on transport failure.
    /// - [`OllamaError::NotFound`] on HTTP 404.
    /// - [`OllamaError::Http`] on other non-2xx responses.
    pub async fn generate_stream(&self, body: &Value) -> Result<reqwest::Response, OllamaError> {
        let url = format!("{}/api/generate", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(true);
        self.post_stream(&url, &body, self.request_timeout).await
    }

    /// POST `/api/chat` with `stream: false`. Forces `stream: false`.
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn chat(&self, body: &Value) -> Result<Value, OllamaError> {
        let url = format!("{}/api/chat", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(false);
        self.post_json_single(&url, &body, self.request_timeout)
            .await
    }

    /// POST `/api/chat` with `stream: true`. Returns the raw response
    /// so the caller can drive the NDJSON parser.
    ///
    /// # Errors
    /// As [`Self::generate_stream`].
    pub async fn chat_stream(&self, body: &Value) -> Result<reqwest::Response, OllamaError> {
        let url = format!("{}/api/chat", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(true);
        self.post_stream(&url, &body, self.request_timeout).await
    }

    /// POST `/api/embeddings`. The body must include `model` and either
    /// `prompt` (single input) or `input` (Ollama 0.1.40+ batch shape).
    ///
    /// # Errors
    /// As [`Self::generate`].
    pub async fn embeddings(&self, body: &Value) -> Result<Value, OllamaError> {
        let url = format!("{}/api/embeddings", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// GET `/api/tags`. Returns every installed model row including base
    /// models and derived adapter-mounted models.
    ///
    /// # Errors
    /// - [`OllamaError::Http`] on non-2xx.
    /// - [`OllamaError::Decode`] when the payload doesn't match the
    ///   `{ models: [...] }` schema.
    pub async fn tags(&self) -> Result<Vec<OllamaModelEntry>, OllamaError> {
        let url = format!("{}/api/tags", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
        }
        let parsed: TagsResponse = serde_json::from_str(&text)?;
        Ok(parsed.models)
    }

    /// POST `/api/show` for the given model name.
    ///
    /// # Errors
    /// As [`Self::tags`], plus [`OllamaError::NotFound`] when Ollama
    /// reports HTTP 404 for an unknown model.
    pub async fn show(&self, model: &str) -> Result<OllamaShowResponse, OllamaError> {
        let url = format!("{}/api/show", self.endpoint);
        let body = serde_json::json!({ "name": model });
        let resp = self
            .auth(self.inner.post(&url).timeout(self.request_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_or_not_found(status.as_u16(), text));
        }
        let parsed: OllamaShowResponse = serde_json::from_str(&text)?;
        Ok(parsed)
    }

    /// POST `/api/pull` with `stream: true`. Returns the raw response so
    /// the caller can drain the NDJSON progress frames.
    ///
    /// # Errors
    /// As [`Self::generate_stream`].
    pub async fn pull_stream(&self, model: &str) -> Result<reqwest::Response, OllamaError> {
        let url = format!("{}/api/pull", self.endpoint);
        let body = serde_json::json!({ "name": model, "stream": true });
        self.post_stream(&url, &body, self.adapter_timeout).await
    }

    /// Convenience wrapper around [`Self::pull_stream`] that drains the
    /// NDJSON progress frames and invokes `on_progress` for each one.
    ///
    /// Returns once the upstream emits a `{status: "success"}` frame or
    /// the connection closes.
    ///
    /// # Errors
    /// As [`Self::pull_stream`], plus [`OllamaError::AdapterFailed`] when
    /// any progress frame carries `{status: "error", ...}`.
    pub async fn pull(
        &self,
        model: &str,
        mut on_progress: impl FnMut(&OllamaPullProgress),
    ) -> Result<(), OllamaError> {
        let resp = self.pull_stream(model).await?;
        let body = resp.text().await?;
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parsed: OllamaPullProgress = serde_json::from_str(trimmed)?;
            if parsed.status == "error" {
                return Err(OllamaError::AdapterFailed(format!(
                    "pull '{model}' reported error: {trimmed}"
                )));
            }
            on_progress(&parsed);
        }
        Ok(())
    }

    /// POST `/api/create` with a Modelfile body. Stream mode is enabled
    /// so the caller can observe build progress; this helper drains the
    /// stream and returns once the upstream emits a terminal status.
    ///
    /// `name` is the derived-model name being created (e.g.
    /// `"llama3.2-sql-lora"`). `modelfile` is the raw Modelfile text
    /// — typically `FROM <base>\nADAPTER <path-or-hf-ref>` for the
    /// adapter-mount case.
    ///
    /// # Errors
    /// - [`OllamaError::AdapterFailed`] when any progress frame carries
    ///   `{status: "error", ...}` or the upstream returns non-2xx.
    /// - [`OllamaError::Request`] on transport failure.
    pub async fn create(&self, name: &str, modelfile: &str) -> Result<(), OllamaError> {
        let url = format!("{}/api/create", self.endpoint);
        let body = serde_json::json!({
            "name": name,
            "modelfile": modelfile,
            "stream": true,
        });
        let resp = self
            .auth(self.inner.post(&url).timeout(self.adapter_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(OllamaError::AdapterFailed(format!(
                "POST /api/create -> HTTP {status}: {}",
                cap(text, 4 * 1024)
            )));
        }
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Frames look like {"status":"reading model metadata"} until
            // either {"status":"success"} or {"status":"error","error":...}.
            let frame: Value = serde_json::from_str(trimmed)?;
            if frame.get("status").and_then(Value::as_str) == Some("error") {
                let msg = frame
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified error");
                return Err(OllamaError::AdapterFailed(format!(
                    "/api/create reported error for '{name}': {msg}"
                )));
            }
        }
        Ok(())
    }

    /// DELETE `/api/delete` for the given model name (base model or
    /// derived adapter-mounted model).
    ///
    /// # Errors
    /// - [`OllamaError::NotFound`] when Ollama returns HTTP 404.
    /// - [`OllamaError::Http`] on other non-2xx responses.
    pub async fn delete(&self, model: &str) -> Result<(), OllamaError> {
        let url = format!("{}/api/delete", self.endpoint);
        let body = serde_json::json!({ "name": model });
        let resp = self
            .auth(self.inner.delete(&url).timeout(self.adapter_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if status.is_success() {
            return Ok(());
        }
        let text = resp.text().await.unwrap_or_default();
        Err(http_or_not_found(status.as_u16(), text))
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
    ) -> Result<Value, OllamaError> {
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
    ) -> Result<reqwest::Response, OllamaError> {
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
/// [`OllamaError::NotFound`] so the bridge can return
/// `BlazenError::ModelNotFound`.
fn http_or_not_found(status: u16, body: String) -> OllamaError {
    if status == 404 {
        OllamaError::NotFound(cap(body, 4 * 1024))
    } else {
        OllamaError::Http {
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
        assert!(matches!(e, OllamaError::NotFound(_)));
    }

    #[test]
    fn http_or_not_found_routes_other() {
        let e = http_or_not_found(500, "boom".into());
        assert!(matches!(e, OllamaError::Http { status: 500, .. }));
    }

    #[test]
    fn client_build_validates_options() {
        let bad = OllamaOptions::required("", "model");
        assert!(OllamaClient::new(&bad).is_err());
        let ok = OllamaOptions::required("http://localhost:11434", "model");
        assert!(OllamaClient::new(&ok).is_ok());
    }
}
