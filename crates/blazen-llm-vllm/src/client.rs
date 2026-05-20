//! Thin wrapper over [`reqwest::Client`] tailored to the vLLM HTTP API.
//!
//! Implements exactly the four endpoints the proxy provider needs:
//!
//! - `POST /v1/chat/completions`           — chat completion (OAI compat)
//! - `POST /v1/load_lora_adapter`          — runtime `LoRA` mount
//! - `POST /v1/unload_lora_adapter`        — runtime `LoRA` unmount
//! - `GET  /v1/models`                     — list base + mounted adapters
//!
//! All bodies are `serde_json::Value` rather than typed structs because
//! Blazen's provider-agnostic [`CompletionRequest`] / [`CompletionResponse`]
//! types live in `blazen-llm` (which this crate cannot depend on without
//! creating a cycle). The `backends/vllm.rs` bridge module does the
//! typed-shape conversion.

use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::VllmError;
use crate::options::VllmOptions;

/// Reusable HTTP client wrapping `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct VllmClient {
    inner: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    request_timeout: Duration,
    adapter_timeout: Duration,
}

/// Minimal `/v1/models` row. Only the fields the proxy actually consumes
/// are decoded; vLLM may emit additional keys we ignore.
#[derive(Debug, Clone, Deserialize)]
pub struct VllmModelEntry {
    pub id: String,
    /// vLLM (>= 0.10) sets this on LoRA-adapter rows so callers can tell
    /// which entries are adapters and which is the base model.
    #[serde(default)]
    pub parent: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<VllmModelEntry>,
}

impl VllmClient {
    /// Build a client from validated options.
    ///
    /// # Errors
    /// Returns [`VllmError::Init`] if the underlying `reqwest::Client`
    /// cannot be built (e.g. TLS init failed).
    pub fn new(opts: &VllmOptions) -> Result<Self, VllmError> {
        opts.validate()?;
        // The per-call timeout is set on each `RequestBuilder`, not on
        // the Client itself, because adapter calls use a different
        // budget than completions.
        let inner = reqwest::Client::builder()
            .build()
            .map_err(|e| VllmError::Init(e.to_string()))?;
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

    /// Apply bearer-auth and JSON content-type to a request builder.
    fn auth(&self, mut rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            rb = rb.bearer_auth(key);
        }
        rb
    }

    /// POST `/v1/chat/completions` with the caller-supplied JSON body.
    ///
    /// `body` must be a fully-formed OpenAI-shaped chat request — the
    /// vLLM server accepts the standard OAI surface (messages, tools,
    /// temperature, stream, `response_format`, ...). Per-request adapter
    /// selection happens by setting `body["model"]` to the adapter's
    /// `lora_name` (vLLM surfaces adapters as `/v1/models` rows with
    /// `parent == base_model`).
    ///
    /// # Errors
    /// - [`VllmError::Request`] on transport failure.
    /// - [`VllmError::Http`] on non-2xx responses (body captured).
    /// - [`VllmError::Decode`] when the response body isn't valid JSON.
    pub async fn chat_completions(&self, body: &Value) -> Result<Value, VllmError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let resp = self
            .auth(self.inner.post(&url).timeout(self.request_timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(VllmError::Http {
                status: status.as_u16(),
                body: cap(text, 4 * 1024),
            });
        }
        Ok(serde_json::from_str(&text)?)
    }

    /// POST `/v1/chat/completions` with `stream: true`. Returns the raw
    /// `reqwest::Response` so the caller can drive the SSE parser
    /// (the streaming wire format is OpenAI-shaped Server-Sent Events).
    ///
    /// # Errors
    /// - [`VllmError::Request`] on transport failure.
    /// - [`VllmError::Http`] on non-2xx responses (body captured).
    pub async fn chat_completions_stream(
        &self,
        body: &Value,
    ) -> Result<reqwest::Response, VllmError> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let resp = self
            .auth(self.inner.post(&url).timeout(self.request_timeout))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VllmError::Http {
                status: status.as_u16(),
                body: cap(text, 4 * 1024),
            });
        }
        Ok(resp)
    }

    /// POST `/v1/load_lora_adapter`.
    ///
    /// The `lora_path` must be a path the vLLM server's process can
    /// read; vLLM does not download the adapter for you (unless the
    /// `lora_huggingface_resolver` plugin is loaded and `lora_path`
    /// starts with `hf://` — that is the caller's contract, not ours).
    ///
    /// # Errors
    /// - [`VllmError::AdapterFailed`] when vLLM returns non-2xx.
    /// - [`VllmError::Request`] on transport failure.
    pub async fn load_lora_adapter(
        &self,
        lora_name: &str,
        lora_path: &str,
    ) -> Result<(), VllmError> {
        let url = format!("{}/v1/load_lora_adapter", self.endpoint);
        let body = serde_json::json!({
            "lora_name": lora_name,
            "lora_path": lora_path,
        });
        let resp = self
            .auth(self.inner.post(&url).timeout(self.adapter_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(VllmError::AdapterFailed(format!(
                "POST /v1/load_lora_adapter -> HTTP {status}: {}",
                cap(text, 4 * 1024)
            )));
        }
        Ok(())
    }

    /// POST `/v1/unload_lora_adapter`.
    ///
    /// # Errors
    /// Same shape as [`Self::load_lora_adapter`].
    pub async fn unload_lora_adapter(&self, lora_name: &str) -> Result<(), VllmError> {
        let url = format!("{}/v1/unload_lora_adapter", self.endpoint);
        let body = serde_json::json!({ "lora_name": lora_name });
        let resp = self
            .auth(self.inner.post(&url).timeout(self.adapter_timeout))
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(VllmError::AdapterFailed(format!(
                "POST /v1/unload_lora_adapter -> HTTP {status}: {}",
                cap(text, 4 * 1024)
            )));
        }
        Ok(())
    }

    /// GET `/v1/models`. Returns every row vLLM lists, including the
    /// base model and any mounted `LoRA` adapters.
    ///
    /// # Errors
    /// - [`VllmError::Http`] on non-2xx.
    /// - [`VllmError::Decode`] when the payload doesn't match the
    ///   `{ data: [...] }` schema.
    pub async fn list_models(&self) -> Result<Vec<VllmModelEntry>, VllmError> {
        let url = format!("{}/v1/models", self.endpoint);
        let resp = self
            .auth(self.inner.get(&url).timeout(self.request_timeout))
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(VllmError::Http {
                status: status.as_u16(),
                body: cap(text, 4 * 1024),
            });
        }
        let parsed: ModelsListResponse = serde_json::from_str(&text)?;
        Ok(parsed.data)
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
    fn client_build_validates_options() {
        let bad = VllmOptions::required("", "model");
        assert!(VllmClient::new(&bad).is_err());
        let ok = VllmOptions::required("http://localhost:8000", "model");
        assert!(VllmClient::new(&ok).is_ok());
    }
}
