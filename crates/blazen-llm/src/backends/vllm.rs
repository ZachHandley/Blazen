//! Bridge between [`blazen_llm_vllm::VllmProvider`] and the Blazen
//! [`Model`](crate::Model) + [`LocalModel`](crate::LocalModel)
//! traits.
//!
//! vLLM exposes the `OpenAI` chat-completion shape on the wire, so the
//! request / response conversion here is the lightweight subset of
//! `providers/openai_compat.rs` that the proxy actually needs.
//!
//! The crate dependency edge is one-way: `blazen-llm` depends on
//! `blazen-llm-vllm`, never the reverse. All trait impls live in this
//! module.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_llm_vllm::{MountedAdapter, VllmAdapterTransport, VllmError, VllmProvider};
use futures_util::Stream;
use serde::Deserialize;
use serde_json::Value;

use crate::AdapterTransport;
use crate::error::BlazenError;
use crate::traits::Model;
use crate::types::{
    ChatMessage, ModelRequest, ModelResponse, RequestTiming, Role, StreamChunk, TokenUsage,
    ToolCall,
};

// ---------------------------------------------------------------------------
// Wire shapes (decode-only)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WireResponse {
    choices: Vec<WireChoice>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    message: WireMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Debug, Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFunctionCall,
}

#[derive(Debug, Deserialize)]
struct WireFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize, Default)]
#[allow(clippy::struct_field_names)] // mirrors the OAI wire field names verbatim
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

// Streaming delta shape.

#[derive(Debug, Deserialize)]
struct WireStreamChunk {
    choices: Vec<WireStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct WireStreamChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireStreamToolCall>,
}

#[derive(Debug, Deserialize)]
struct WireStreamToolCall {
    #[serde(default)]
    id: Option<String>,
    function: WireStreamFunctionCall,
}

#[derive(Debug, Deserialize)]
struct WireStreamFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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
    // Proxies don't need the full multimodal payload — vLLM accepts a
    // string content for chat completions. Future enhancement: route
    // image parts through the OAI content-array shape.
    let content = msg.content.as_text().unwrap_or_default().to_string();
    let mut m = serde_json::json!({
        "role": role,
        "content": content,
    });
    if let Some(ref tc_id) = msg.tool_call_id {
        m["tool_call_id"] = Value::String(tc_id.clone());
    }
    if !msg.tool_calls.is_empty() {
        m["tool_calls"] = Value::Array(
            msg.tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string(),
                        }
                    })
                })
                .collect(),
        );
    }
    m
}

/// Translate a Blazen `ModelRequest` into a vLLM OpenAI-compat body.
fn build_body(default_model: &str, req: &ModelRequest, stream: bool) -> Value {
    let model = req.model.as_deref().unwrap_or(default_model);
    let messages: Vec<Value> = req.messages.iter().map(message_to_value).collect();
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });
    if let Some(t) = req.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = req.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(m) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(m);
    }
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect(),
        );
    }
    if let Some(ref fmt) = req.response_format {
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "blazen_structured", "schema": fmt },
        });
    }
    body
}

fn parse_finish_reason(reason: &str) -> String {
    // `ModelResponse::finish_reason` and `StreamChunk::finish_reason`
    // are both `Option<String>`. The bridge keeps the upstream wire string
    // verbatim — downstream consumers do the semantic mapping via the
    // separate `FinishReason` enum exported from `blazen-llm`. Caller
    // wraps the result in `Some(...)` because vLLM only ever sends
    // `finish_reason` on a terminal choice.
    reason.to_string()
}

fn decode_response(value: Value) -> Result<ModelResponse, BlazenError> {
    let parsed: WireResponse = serde_json::from_value(value)
        .map_err(|e| BlazenError::provider("vllm", format!("decode response: {e}")))?;
    let first = parsed.choices.into_iter().next();
    let (content, tool_calls, finish_reason) = match first {
        Some(c) => {
            let tool_calls: Vec<ToolCall> = c
                .message
                .tool_calls
                .into_iter()
                .map(|tc| {
                    let args = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(Value::String(tc.function.arguments));
                    ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: args,
                    }
                })
                .collect();
            let finish = c.finish_reason.as_deref().map(parse_finish_reason);
            (c.message.content, tool_calls, finish)
        }
        None => (None, Vec::new(), None),
    };

    let usage = parsed.usage.map(|u| TokenUsage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
        ..Default::default()
    });

    Ok(ModelResponse {
        content,
        tool_calls,
        reasoning: None,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage,
        model: parsed.model.unwrap_or_default(),
        finish_reason,
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

fn vllm_to_blazen(err: VllmError) -> BlazenError {
    match err {
        VllmError::UnsupportedTransport(msg) | VllmError::InvalidOptions(msg) => {
            BlazenError::unsupported(format!("vllm: {msg}"))
        }
        other => BlazenError::provider("vllm", other.to_string()),
    }
}

fn convert_transport(t: &AdapterTransport) -> VllmAdapterTransport {
    match t {
        AdapterTransport::LocalFs(p) => VllmAdapterTransport::LocalFs(p.clone()),
        AdapterTransport::HfHub { repo, revision } => VllmAdapterTransport::HfHub {
            repo: repo.clone(),
            revision: revision.clone(),
        },
        AdapterTransport::HttpPush(bytes) => VllmAdapterTransport::HttpPush(bytes.clone()),
    }
}

// ---------------------------------------------------------------------------
// SSE parser — minimal, scoped to the vllm bridge
// ---------------------------------------------------------------------------

/// Drains an `data: <json>\n\n`-framed SSE byte stream into
/// [`StreamChunk`]s. Terminator is `data: [DONE]`.
fn sse_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>> {
    let byte_stream = response.bytes_stream();
    let parser = SseChunkStream {
        inner: Box::pin(byte_stream),
        buffer: String::new(),
        done: false,
    };
    Box::pin(parser)
}

struct SseChunkStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buffer: String,
    done: bool,
}

impl Stream for SseChunkStream {
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
            // Try to drain a complete event from the buffer.
            if let Some(chunk_or_eof) = pop_event(&mut this.buffer) {
                match chunk_or_eof {
                    Event::Done => {
                        this.done = true;
                        return std::task::Poll::Ready(None);
                    }
                    Event::Data(data) => match serde_json::from_str::<WireStreamChunk>(&data) {
                        Ok(parsed) => {
                            let chunk = wire_to_chunk(parsed);
                            return std::task::Poll::Ready(Some(Ok(chunk)));
                        }
                        Err(e) => {
                            return std::task::Poll::Ready(Some(Err(BlazenError::stream_error(
                                format!("vllm sse decode: {e}"),
                            ))));
                        }
                    },
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
                    this.done = true;
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

enum Event {
    Data(String),
    Done,
}

fn pop_event(buffer: &mut String) -> Option<Event> {
    let split = buffer.find("\n\n")?;
    let event_block = buffer[..split].to_string();
    buffer.drain(..split + 2);
    for line in event_block.lines() {
        if let Some(payload) = line.strip_prefix("data:") {
            let trimmed = payload.trim();
            if trimmed == "[DONE]" {
                return Some(Event::Done);
            }
            return Some(Event::Data(trimmed.to_string()));
        }
    }
    // Heartbeat / unknown — recurse on the remaining buffer.
    pop_event(buffer)
}

fn wire_to_chunk(parsed: WireStreamChunk) -> StreamChunk {
    let first = parsed.choices.into_iter().next();
    let (delta, tool_calls, finish) = match first {
        Some(c) => {
            let tcs: Vec<ToolCall> = c
                .delta
                .tool_calls
                .into_iter()
                .map(|t| {
                    let args = t.function.arguments.as_deref().map_or(Value::Null, |s| {
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()))
                    });
                    ToolCall {
                        id: t.id.unwrap_or_default(),
                        name: t.function.name.unwrap_or_default(),
                        arguments: args,
                    }
                })
                .collect();
            (
                c.delta.content,
                tcs,
                c.finish_reason.as_deref().map(parse_finish_reason),
            )
        }
        None => (None, Vec::new(), None),
    };
    StreamChunk {
        delta,
        tool_calls,
        finish_reason: finish,
        reasoning_delta: None,
        citations: Vec::new(),
        artifacts: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[async_trait]
impl Model for VllmProvider {
    fn model_id(&self) -> &str {
        VllmProvider::model_id(self)
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        let body = build_body(self.model_id(), &request, false);
        let value = self.complete(body).await.map_err(vllm_to_blazen)?;
        decode_response(value)
    }

    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let body = build_body(self.model_id(), &request, true);
        let response = self.stream(body).await.map_err(vllm_to_blazen)?;
        Ok(sse_stream(response))
    }
}

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::LocalModel for VllmProvider {
    /// vLLM owns its own model lifecycle — `load` verifies the upstream
    /// can reach the base model via `GET /v1/models`.
    async fn load(&self) -> Result<(), BlazenError> {
        self.refresh_adapters_from_server()
            .await
            .map(|_| ())
            .map_err(vllm_to_blazen)
    }

    /// vLLM has no public unload API for its base model; this is a no-op
    /// success (idempotent contract).
    async fn unload(&self) -> Result<(), BlazenError> {
        Ok(())
    }

    /// Probes by issuing a cheap `GET /v1/models`.
    async fn is_loaded(&self) -> bool {
        self.refresh_adapters_from_server().await.is_ok()
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
        // Mirror the caller's AdapterTransport onto the provider for this
        // call by cloning options. We do not mutate the provider — the
        // bridge stays stateless w.r.t. transport so two concurrent
        // callers don't race.
        let transport = convert_transport(&AdapterTransport::default());
        let _ = transport; // Reserved: future per-call transport.

        let mounted: MountedAdapter = self
            .load_adapter(options.adapter_id, adapter_dir)
            .await
            .map_err(vllm_to_blazen)?;
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
            .map_err(vllm_to_blazen)
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
    use crate::types::{ChatMessage, MessageContent, ToolDefinition};

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
            temperature: Some(0.2),
            max_tokens: Some(64),
            top_p: None,
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
        let body = build_body("llama-3.2", &make_req("hi"), false);
        assert_eq!(body["model"], "llama-3.2");
        assert_eq!(body["stream"], false);
        // temperature was set as f32 0.2; serde widens to f64 with the
        // usual cast residue, so compare with tolerance rather than
        // assert exact equality against the literal 0.2.
        let t = body["temperature"].as_f64().unwrap();
        assert!((t - 0.2).abs() < 1e-6, "temperature ≈ 0.2, got {t}");
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
    }

    #[test]
    fn build_body_honours_request_model_override() {
        let mut req = make_req("hi");
        req.model = Some("sql-lora".into());
        let body = build_body("llama-3.2", &req, true);
        assert_eq!(body["model"], "sql-lora");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn build_body_serialises_tools() {
        let mut req = make_req("hi");
        req.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Look up weather".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let body = build_body("llama", &req, false);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn decode_response_pulls_content_and_usage() {
        let raw = serde_json::json!({
            "model": "llama-3.2",
            "choices": [{
                "message": {"content": "hello back"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        });
        let resp = decode_response(raw).expect("decode ok");
        assert_eq!(resp.content.as_deref(), Some("hello back"));
        assert_eq!(resp.model, "llama-3.2");
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 12);
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn decode_response_parses_tool_calls() {
        let raw = serde_json::json!({
            "model": "m",
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "sum",
                            "arguments": "{\"a\":1,\"b\":2}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let resp = decode_response(raw).expect("decode ok");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "sum");
        assert_eq!(resp.tool_calls[0].arguments["a"], 1);
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn pop_event_extracts_data_line_and_consumes_buffer() {
        let mut buf = "data: {\"choices\":[]}\n\nnext".to_string();
        match pop_event(&mut buf).expect("event") {
            Event::Data(s) => assert_eq!(s, "{\"choices\":[]}"),
            Event::Done => panic!("expected data"),
        }
        assert_eq!(buf, "next");
    }

    #[test]
    fn pop_event_recognises_done_terminator() {
        let mut buf = "data: [DONE]\n\n".to_string();
        assert!(matches!(pop_event(&mut buf), Some(Event::Done)));
    }

    #[test]
    fn pop_event_returns_none_when_buffer_incomplete() {
        let mut buf = "data: {\"x".to_string();
        assert!(pop_event(&mut buf).is_none());
    }

    #[test]
    fn convert_transport_passes_through_each_variant() {
        let lf = convert_transport(&AdapterTransport::LocalFs("/srv/a".into()));
        assert!(matches!(lf, VllmAdapterTransport::LocalFs(_)));
        let hf = convert_transport(&AdapterTransport::HfHub {
            repo: "org/repo".into(),
            revision: Some("v1".into()),
        });
        assert!(matches!(hf, VllmAdapterTransport::HfHub { .. }));
        let push = convert_transport(&AdapterTransport::HttpPush(vec![1, 2]));
        assert!(matches!(push, VllmAdapterTransport::HttpPush(_)));
    }
}
