//! Thin wrapper over [`reqwest::Client`] tailored to the LM Studio
//! HTTP API.
//!
//! Covers the seven endpoints the proxy provider needs:
//!
//! - `POST  /v1/chat/completions`   — OAI-shaped chat completion
//! - `POST  /v1/completions`        — OAI-shaped legacy completion
//! - `POST  /v1/embeddings`         — OAI-shaped embeddings
//! - `GET   /v1/models`             — OAI-shaped model listing
//! - `GET   /api/v0/models`         — native listing with load status
//! - `POST  /api/v0/models/load`    — load a model
//! - `POST  /api/v0/models/unload`  — unload a model
//!
//! Bodies are `serde_json::Value` rather than typed structs because
//! Blazen's provider-agnostic [`ModelRequest`] / [`ModelResponse`]
//! types live in `blazen-llm` (which this crate cannot depend on without
//! creating a cycle). The `backends/lmstudio.rs` bridge does the typed
//! conversion.

use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::LmStudioError;
use crate::options::LmStudioOptions;

/// Reusable HTTP client wrapping `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct LmStudioClient {
    inner: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    request_timeout: Duration,
    load_timeout: Duration,
}

/// One row from `GET /v1/models` (OAI-shaped). LM Studio includes more
/// fields (object, `owned_by`, ...) but the proxy only needs the id.
#[derive(Debug, Clone, Deserialize)]
pub struct LmStudioModelEntry {
    /// LM Studio model identifier — the value clients put in the
    /// OAI `model` field at request time.
    pub id: String,
    /// `"model"` for entries; included verbatim so callers can filter.
    #[serde(default)]
    pub object: Option<String>,
    /// LM Studio sets this when the row represents an organisation/owner.
    #[serde(default)]
    pub owned_by: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<LmStudioModelEntry>,
}

/// Load state of a model in the LM Studio native listing.
///
/// LM Studio reports `state: "loaded" | "not-loaded"` on each entry of
/// `GET /api/v0/models`. Unknown states are mapped to
/// [`Self::Other`] verbatim so the proxy doesn't break against forward-
/// compatible additions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LmStudioNativeModelState {
    /// The model is currently loaded and ready to serve requests.
    Loaded,
    /// The model is installed but not loaded.
    NotLoaded,
    /// Any other state string LM Studio reports.
    Other(String),
}

impl LmStudioNativeModelState {
    fn from_str(s: &str) -> Self {
        match s {
            "loaded" => Self::Loaded,
            "not-loaded" => Self::NotLoaded,
            other => Self::Other(other.to_string()),
        }
    }
}

/// One row from `GET /api/v0/models` (LM-Studio native). Includes load
/// state and other native fields (type, arch, ...) the OAI surface
/// does not expose.
#[derive(Debug, Clone)]
pub struct LmStudioNativeModelEntry {
    /// LM Studio model identifier.
    pub id: String,
    /// Whether the model is currently loaded.
    pub state: LmStudioNativeModelState,
    /// Native-only model type, e.g. `"llm"`, `"embeddings"`,
    /// `"vlm"`, when present.
    pub r#type: Option<String>,
    /// Native-only base architecture, e.g. `"qwen2"`, `"llama"`,
    /// when present.
    pub arch: Option<String>,
    /// Full upstream payload for callers needing fields we don't type.
    pub raw: Value,
}

#[derive(Debug, Deserialize)]
struct NativeModelRow {
    id: String,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    arch: Option<String>,
    #[serde(flatten)]
    raw: Value,
}

#[derive(Debug, Deserialize)]
struct NativeModelsListResponse {
    data: Vec<NativeModelRow>,
}

impl LmStudioClient {
    /// Build a client from validated options.
    ///
    /// # Errors
    /// Returns [`LmStudioError::Init`] if the underlying
    /// `reqwest::Client` cannot be built (e.g. TLS init failed).
    pub fn new(opts: &LmStudioOptions) -> Result<Self, LmStudioError> {
        opts.validate()?;
        // The per-call timeout is set on each `RequestBuilder`, not on
        // the Client itself, because load/unload calls use a different
        // budget than completions.
        let inner = reqwest::Client::builder()
            .build()
            .map_err(|e| LmStudioError::Init(e.to_string()))?;
        Ok(Self {
            inner,
            endpoint: opts.endpoint_trimmed().to_string(),
            api_key: opts.api_key.clone(),
            request_timeout: opts.request_timeout,
            load_timeout: opts.load_timeout,
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

    /// POST `/v1/chat/completions` with `stream: false`. The caller-
    /// supplied body must be OAI-shaped; this helper forces
    /// `stream: false`.
    ///
    /// # Errors
    /// - [`LmStudioError::Request`] on transport failure.
    /// - [`LmStudioError::NotFound`] on HTTP 404 (model missing).
    /// - [`LmStudioError::NoModelLoaded`] on HTTP 409/422 (no model loaded).
    /// - [`LmStudioError::Http`] on other non-2xx responses.
    /// - [`LmStudioError::Decode`] when the response body isn't valid JSON.
    pub async fn chat_completions(&self, body: &Value) -> Result<Value, LmStudioError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(false);
        self.post_json_single(&url, &body, self.request_timeout)
            .await
    }

    /// POST `/v1/chat/completions` with `stream: true`. Returns the raw
    /// `reqwest::Response` so the caller can drive the SSE parser
    /// (LM Studio uses OAI-shaped Server-Sent Events).
    ///
    /// # Errors
    /// As [`Self::chat_completions`], minus `Decode`.
    pub async fn chat_completions_stream(
        &self,
        body: &Value,
    ) -> Result<reqwest::Response, LmStudioError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(true);
        self.post_stream(&url, &body, self.request_timeout).await
    }

    /// POST `/v1/completions` (legacy single-prompt completion).
    /// Forces `stream: false`.
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn completions(&self, body: &Value) -> Result<Value, LmStudioError> {
        let url = format!("{}/v1/completions", self.endpoint);
        let mut body = body.clone();
        body["stream"] = Value::Bool(false);
        self.post_json_single(&url, &body, self.request_timeout)
            .await
    }

    /// POST `/v1/embeddings`. The body must include `model` and `input`
    /// (string or array of strings).
    ///
    /// # Errors
    /// As [`Self::chat_completions`].
    pub async fn embeddings(&self, body: &Value) -> Result<Value, LmStudioError> {
        let url = format!("{}/v1/embeddings", self.endpoint);
        self.post_json_single(&url, body, self.request_timeout)
            .await
    }

    /// GET `/v1/models`. Returns every model row LM Studio knows about.
    /// The OAI listing is flat — to tell loaded from unloaded use
    /// [`Self::native_models`] instead.
    ///
    /// # Errors
    /// - [`LmStudioError::Http`] on non-2xx.
    /// - [`LmStudioError::Decode`] when the payload doesn't match the
    ///   `{ data: [...] }` schema.
    pub async fn list_models(&self) -> Result<Vec<LmStudioModelEntry>, LmStudioError> {
        let url = format!("{}/v1/models", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_to_error(status.as_u16(), text));
        }
        let parsed: ModelsListResponse = serde_json::from_str(&text)?;
        Ok(parsed.data)
    }

    /// GET `/api/v0/models`. Returns the LM-Studio-native listing,
    /// including each model's load state, type, and architecture.
    ///
    /// # Errors
    /// As [`Self::list_models`].
    pub async fn native_models(&self) -> Result<Vec<LmStudioNativeModelEntry>, LmStudioError> {
        let url = format!("{}/api/v0/models", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_to_error(status.as_u16(), text));
        }
        let parsed: NativeModelsListResponse = serde_json::from_str(&text)?;
        Ok(parsed
            .data
            .into_iter()
            .map(|row| LmStudioNativeModelEntry {
                id: row.id,
                state: row
                    .state
                    .as_deref()
                    .map_or(LmStudioNativeModelState::NotLoaded, |s| {
                        LmStudioNativeModelState::from_str(s)
                    }),
                r#type: row.r#type,
                arch: row.arch,
                raw: row.raw,
            })
            .collect())
    }

    /// POST `/api/v0/models/load`. Asks LM Studio to load the named
    /// model into memory.
    ///
    /// # Errors
    /// - [`LmStudioError::NotFound`] when LM Studio returns HTTP 404
    ///   for an unknown model id.
    /// - [`LmStudioError::LoadFailed`] for any other non-2xx response.
    /// - [`LmStudioError::Request`] on transport failure.
    pub async fn load_model(&self, model: &str) -> Result<(), LmStudioError> {
        let url = format!("{}/api/v0/models/load", self.endpoint);
        let body = serde_json::json!({ "model": model });
        let resp = self
            .auth(self.inner.post(&url).timeout(self.load_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.is_success() {
            return Ok(());
        }
        if status.as_u16() == 404 {
            return Err(LmStudioError::NotFound(cap(text, 4 * 1024)));
        }
        Err(LmStudioError::LoadFailed(format!(
            "POST /api/v0/models/load -> HTTP {status}: {}",
            cap(text, 4 * 1024)
        )))
    }

    /// POST `/api/v0/models/unload`. Asks LM Studio to evict the named
    /// model from memory.
    ///
    /// # Errors
    /// As [`Self::load_model`].
    pub async fn unload_model(&self, model: &str) -> Result<(), LmStudioError> {
        let url = format!("{}/api/v0/models/unload", self.endpoint);
        let body = serde_json::json!({ "model": model });
        let resp = self
            .auth(self.inner.post(&url).timeout(self.load_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.is_success() {
            return Ok(());
        }
        if status.as_u16() == 404 {
            return Err(LmStudioError::NotFound(cap(text, 4 * 1024)));
        }
        Err(LmStudioError::LoadFailed(format!(
            "POST /api/v0/models/unload -> HTTP {status}: {}",
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
    ) -> Result<Value, LmStudioError> {
        let resp = self
            .auth(self.inner.post(url).timeout(timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(http_to_error(status.as_u16(), text));
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
    ) -> Result<reqwest::Response, LmStudioError> {
        let resp = self
            .auth(self.inner.post(url).timeout(timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(http_to_error(status.as_u16(), text));
        }
        Ok(resp)
    }
}

/// Map a non-2xx status to the right error variant. 404 -> `NotFound`,
/// 409/422 -> `NoModelLoaded`, anything else -> `Http`.
fn http_to_error(status: u16, body: String) -> LmStudioError {
    match status {
        404 => LmStudioError::NotFound(cap(body, 4 * 1024)),
        409 | 422 => LmStudioError::NoModelLoaded(cap(body, 4 * 1024)),
        _ => LmStudioError::Http {
            status,
            body: cap(body, 4 * 1024),
        },
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
    fn http_to_error_routes_404_to_not_found() {
        let e = http_to_error(404, "model not found".into());
        assert!(matches!(e, LmStudioError::NotFound(_)));
    }

    #[test]
    fn http_to_error_routes_409_to_no_model_loaded() {
        let e = http_to_error(409, "no model loaded".into());
        assert!(matches!(e, LmStudioError::NoModelLoaded(_)));
    }

    #[test]
    fn http_to_error_routes_422_to_no_model_loaded() {
        let e = http_to_error(422, "no model loaded".into());
        assert!(matches!(e, LmStudioError::NoModelLoaded(_)));
    }

    #[test]
    fn http_to_error_routes_other_to_http() {
        let e = http_to_error(500, "boom".into());
        assert!(matches!(e, LmStudioError::Http { status: 500, .. }));
    }

    #[test]
    fn native_state_parses_known_strings() {
        assert_eq!(
            LmStudioNativeModelState::from_str("loaded"),
            LmStudioNativeModelState::Loaded
        );
        assert_eq!(
            LmStudioNativeModelState::from_str("not-loaded"),
            LmStudioNativeModelState::NotLoaded
        );
    }

    #[test]
    fn native_state_keeps_unknown_strings_verbatim() {
        let s = LmStudioNativeModelState::from_str("loading");
        assert_eq!(s, LmStudioNativeModelState::Other("loading".into()));
    }

    #[test]
    fn client_build_validates_options() {
        let bad = LmStudioOptions::required("", "model");
        assert!(LmStudioClient::new(&bad).is_err());
        let ok = LmStudioOptions::required("http://localhost:1234", "model");
        assert!(LmStudioClient::new(&ok).is_ok());
    }
}
