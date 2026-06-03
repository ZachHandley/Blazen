//! Bridge between [`blazen_llm_ollama::OllamaProvider`] and the Blazen
//! [`Model`](crate::Model) +
//! [`LocalModel`](crate::LocalModel) traits.
//!
//! Ollama exposes its own request/response shapes (not OpenAI-compat),
//! so the bridge has its own typed wire structs rather than reusing the
//! ones from `providers/openai_compat.rs`. Streaming is NDJSON
//! (`application/x-ndjson` — one JSON object per `\n`-terminated line)
//! rather than SSE.
//!
//! The crate dependency edge is one-way: `blazen-llm` depends on
//! `blazen-llm-ollama`, never the reverse. All trait impls live here.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_llm_ollama::{MountedAdapter, OllamaError, OllamaProvider};
use futures_util::Stream;
use serde::Deserialize;
use serde_json::Value;

use crate::error::BlazenError;
use crate::traits::Model;
use crate::types::{
    ChatMessage, ModelRequest, ModelResponse, RequestTiming, Role, StreamChunk, TokenUsage,
};

// ---------------------------------------------------------------------------
// Wire shapes (decode-only)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WireChatResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    message: Option<WireChatMessage>,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct WireChatMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct WireStreamFrame {
    #[serde(default)]
    message: Option<WireChatMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn role_to_str(r: &Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn message_to_value(msg: &ChatMessage) -> Value {
    let role = role_to_str(&msg.role);
    let content = msg.content.as_text().unwrap_or_default().to_string();
    serde_json::json!({
        "role": role,
        "content": content,
    })
}

/// Translate a Blazen `ModelRequest` into an Ollama `/api/chat` body.
///
/// Sampling knobs map into the `options` sub-object Ollama uses for
/// generation parameters (`temperature`, `top_p`, `num_predict` — the
/// `max_tokens` analogue). Streaming is controlled by the caller (the
/// proxy client forces the flag onto the body either way).
fn build_body(default_model: &str, req: &ModelRequest, stream: bool) -> Value {
    let model = req.model.as_deref().unwrap_or(default_model);
    let messages: Vec<Value> = req.messages.iter().map(message_to_value).collect();
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });
    let mut options = serde_json::Map::new();
    if let Some(t) = req.temperature {
        options.insert("temperature".to_string(), serde_json::json!(t));
    }
    if let Some(p) = req.top_p {
        options.insert("top_p".to_string(), serde_json::json!(p));
    }
    if let Some(m) = req.max_tokens {
        options.insert("num_predict".to_string(), serde_json::json!(m));
    }
    if !options.is_empty() {
        body["options"] = Value::Object(options);
    }
    if let Some(ref fmt) = req.response_format {
        // Ollama accepts a JSON schema directly under `format`.
        body["format"] = fmt.clone();
    }
    body
}

fn decode_response(value: Value) -> Result<ModelResponse, BlazenError> {
    let parsed: WireChatResponse = serde_json::from_value(value)
        .map_err(|e| BlazenError::provider("ollama", format!("decode response: {e}")))?;
    let content = parsed.message.map(|m| m.content);
    let usage = TokenUsage {
        prompt_tokens: parsed.prompt_eval_count.unwrap_or(0),
        completion_tokens: parsed.eval_count.unwrap_or(0),
        total_tokens: parsed.prompt_eval_count.unwrap_or(0) + parsed.eval_count.unwrap_or(0),
        ..Default::default()
    };
    Ok(ModelResponse {
        content,
        tool_calls: Vec::new(),
        reasoning: None,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage: Some(usage),
        model: parsed.model.unwrap_or_default(),
        finish_reason: parsed.done_reason,
        cost: None,
        timing: Some(RequestTiming {
            queue_ms: None,
            execution_ms: None,
            total_ms: None,
        }),
        images: Vec::new(),
        audio: Vec::new(),
        videos: Vec::new(),
        metadata: Value::Null,
    })
}

fn ollama_to_blazen(err: OllamaError) -> BlazenError {
    match err {
        OllamaError::Unsupported(msg) | OllamaError::InvalidOptions(msg) => {
            BlazenError::unsupported(format!("ollama: {msg}"))
        }
        other => BlazenError::provider("ollama", other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// NDJSON parser — minimal, scoped to the ollama bridge
// ---------------------------------------------------------------------------

/// Drain an `application/x-ndjson` byte stream into [`StreamChunk`]s.
/// Each line is one JSON object; the terminal frame carries `done: true`.
fn ndjson_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>> {
    let byte_stream = response.bytes_stream();
    Box::pin(NdjsonChunkStream {
        inner: Box::pin(byte_stream),
        buffer: String::new(),
        done: false,
    })
}

struct NdjsonChunkStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buffer: String,
    done: bool,
}

impl Stream for NdjsonChunkStream {
    type Item = Result<StreamChunk, BlazenError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.done {
                return std::task::Poll::Ready(None);
            }
            if let Some(line) = pop_line(&mut this.buffer) {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match serde_json::from_str::<WireStreamFrame>(trimmed) {
                    Ok(frame) => {
                        let delta = frame.message.map(|m| m.content);
                        let finish = frame.done_reason.clone();
                        let terminal = frame.done;
                        if terminal {
                            this.done = true;
                        }
                        // Emit content deltas always; emit a final chunk
                        // with finish_reason set when the terminal frame
                        // arrives (even if delta is empty).
                        return std::task::Poll::Ready(Some(Ok(StreamChunk {
                            delta,
                            tool_calls: Vec::new(),
                            finish_reason: finish,
                            reasoning_delta: None,
                            citations: Vec::new(),
                            artifacts: Vec::new(),
                        })));
                    }
                    Err(e) => {
                        return std::task::Poll::Ready(Some(Err(BlazenError::stream_error(
                            format!("ollama ndjson decode: {e}"),
                        ))));
                    }
                }
            }

            match this.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(BlazenError::stream_error(
                        e.to_string(),
                    ))));
                }
                std::task::Poll::Ready(None) => {
                    // Flush any remaining buffered line (servers don't
                    // always end with a trailing newline).
                    let leftover = std::mem::take(&mut this.buffer);
                    this.done = true;
                    let trimmed = leftover.trim();
                    if trimmed.is_empty() {
                        return std::task::Poll::Ready(None);
                    }
                    match serde_json::from_str::<WireStreamFrame>(trimmed) {
                        Ok(frame) => {
                            return std::task::Poll::Ready(Some(Ok(StreamChunk {
                                delta: frame.message.map(|m| m.content),
                                tool_calls: Vec::new(),
                                finish_reason: frame.done_reason,
                                reasoning_delta: None,
                                citations: Vec::new(),
                                artifacts: Vec::new(),
                            })));
                        }
                        Err(e) => {
                            return std::task::Poll::Ready(Some(Err(BlazenError::stream_error(
                                format!("ollama ndjson trailing-frame decode: {e}"),
                            ))));
                        }
                    }
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

/// Pop the next `\n`-terminated line from `buffer`, if one exists. Does
/// not include the trailing newline.
fn pop_line(buffer: &mut String) -> Option<String> {
    let idx = buffer.find('\n')?;
    let line = buffer[..idx].to_string();
    buffer.drain(..=idx);
    Some(line)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[async_trait]
impl Model for OllamaProvider {
    fn model_id(&self) -> &str {
        OllamaProvider::model_id(self)
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        let body = build_body(self.model_id(), &request, false);
        let value = self.chat(body).await.map_err(ollama_to_blazen)?;
        decode_response(value)
    }

    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let body = build_body(self.model_id(), &request, true);
        let response = self.chat_stream(body).await.map_err(ollama_to_blazen)?;
        Ok(ndjson_stream(response))
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::LocalModel for OllamaProvider {
    /// Ollama owns its own model lifecycle — `load` verifies the upstream
    /// can reach the base model via `GET /api/tags` and that the
    /// provider's `model` is in the listing.
    async fn load(&self) -> Result<(), BlazenError> {
        let listing = self.tags().await.map_err(ollama_to_blazen)?;
        let base = self.model_id();
        if listing.iter().any(|m| m.name == base) {
            Ok(())
        } else {
            Err(BlazenError::provider(
                "ollama",
                format!("base model '{base}' is not installed on the Ollama server"),
            ))
        }
    }

    /// Ollama has no public unload API for its base model; this is a
    /// no-op success (idempotent contract).
    async fn unload(&self) -> Result<(), BlazenError> {
        Ok(())
    }

    /// Probes by issuing a cheap `GET /api/tags`.
    async fn is_loaded(&self) -> bool {
        self.tags().await.is_ok()
    }

    fn device(&self) -> crate::device::Device {
        crate::device::Device::Remote {
            endpoint: self.options().endpoint.clone(),
        }
    }

    /// Always zero — the weights live in another process; the proxy
    /// does not charge against the local `ModelManager` pool budgets.
    async fn memory_bytes(&self) -> Option<u64> {
        Some(0)
    }

    async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        let mounted: MountedAdapter = self
            .load_adapter(options.adapter_id, adapter_dir)
            .await
            .map_err(ollama_to_blazen)?;
        Ok(crate::AdapterHandle {
            adapter_id: mounted.adapter_id,
            // Adapter memory lives upstream; never count against local pools.
            memory_bytes: 0,
            mount_strategy: crate::AdapterMountStrategy::Attached,
        })
    }

    async fn unload_adapter(&self, handle: &crate::AdapterHandle) -> Result<(), BlazenError> {
        self.unload_adapter(&handle.adapter_id)
            .await
            .map_err(ollama_to_blazen)
    }

    async fn list_adapters(&self) -> Vec<crate::AdapterStatus> {
        self.list_adapters()
            .await
            .into_iter()
            .map(|m| crate::AdapterStatus {
                adapter_id: m.adapter_id,
                scale: m.scale,
                source_dir: m.source_dir,
                memory_bytes: 0,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, MessageContent};

    fn make_req(text: &str) -> ModelRequest {
        ModelRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: MessageContent::Text(text.into()),
                tool_call_id: None,
                name: None,
                tool_calls: Vec::new(),
                tool_result: None,
            }],
            tools: vec![],
            temperature: Some(0.3),
            max_tokens: Some(128),
            top_p: Some(0.9),
            response_format: None,
            model: None,
            modalities: None,
            image_config: None,
            audio_config: None,
            tool_choice: None,
        }
    }

    #[test]
    fn build_body_uses_provider_default_model_when_request_omits_one() {
        let body = build_body("llama3.2", &make_req("hi"), false);
        assert_eq!(body["model"], "llama3.2");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        // Sampling knobs go into the options sub-object.
        let opts = body.get("options").expect("options block present");
        assert!(opts["temperature"].as_f64().is_some());
        assert_eq!(opts["num_predict"], 128);
    }

    #[test]
    fn build_body_honours_request_model_override() {
        let mut req = make_req("hi");
        req.model = Some("llama3.2-sql-lora".into());
        let body = build_body("llama3.2", &req, true);
        assert_eq!(body["model"], "llama3.2-sql-lora");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn build_body_omits_options_block_when_no_sampling_knobs_set() {
        let mut req = make_req("hi");
        req.temperature = None;
        req.top_p = None;
        req.max_tokens = None;
        let body = build_body("llama3.2", &req, false);
        assert!(body.get("options").is_none());
    }

    #[test]
    fn decode_response_pulls_content_and_usage() {
        let raw = serde_json::json!({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hello back"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 7,
        });
        let resp = decode_response(raw).expect("decode ok");
        assert_eq!(resp.content.as_deref(), Some("hello back"));
        assert_eq!(resp.model, "llama3.2");
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 12);
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn decode_response_handles_empty_message_gracefully() {
        let raw = serde_json::json!({
            "model": "llama3.2",
            "done": true,
        });
        let resp = decode_response(raw).expect("decode ok");
        assert!(resp.content.is_none());
    }

    #[test]
    fn pop_line_extracts_first_line_and_consumes_buffer() {
        let mut buf = "first\nsecond\n".to_string();
        assert_eq!(pop_line(&mut buf).as_deref(), Some("first"));
        assert_eq!(buf, "second\n");
    }

    #[test]
    fn pop_line_returns_none_when_no_newline() {
        let mut buf = "partial".to_string();
        assert!(pop_line(&mut buf).is_none());
        assert_eq!(buf, "partial");
    }

    #[test]
    fn ollama_to_blazen_routes_unsupported_to_unsupported_variant() {
        let e = ollama_to_blazen(OllamaError::Unsupported("nope".into()));
        // Cannot assert variant equality without exposing internals; just
        // verify the display contains the marker substring.
        assert!(format!("{e:?}").to_lowercase().contains("unsupported"));
    }

    #[test]
    fn ollama_to_blazen_routes_other_to_provider_variant() {
        let e = ollama_to_blazen(OllamaError::Http {
            status: 500,
            body: "boom".into(),
        });
        assert!(format!("{e:?}").to_lowercase().contains("provider"));
    }
}
