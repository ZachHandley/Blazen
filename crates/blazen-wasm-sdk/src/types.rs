//! `wasm-bindgen` wrappers for response types.
//!
//! Exposes [`CompletionResponse`], [`StreamChunk`], [`TokenUsage`], and
//! [`ToolCall`] to JavaScript with getter-based property access.

use wasm_bindgen::prelude::*;

use blazen_llm::types::{
    CompletionResponse as InnerCompletionResponse,
    StreamChunk as InnerStreamChunk,
    TokenUsage as InnerTokenUsage,
    ToolCall as InnerToolCall,
};

// ---------------------------------------------------------------------------
// WasmCompletionResponse
// ---------------------------------------------------------------------------

/// The result of a non-streaming chat completion.
#[wasm_bindgen(js_name = "CompletionResponse")]
pub struct WasmCompletionResponse {
    inner: InnerCompletionResponse,
}

#[wasm_bindgen(js_class = "CompletionResponse")]
impl WasmCompletionResponse {
    /// The text content of the assistant's reply, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn content(&self) -> Option<String> {
        self.inner.content.clone()
    }

    /// The model that produced this response.
    #[wasm_bindgen(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// The reason the model stopped generating (e.g. `"stop"`, `"tool_use"`).
    #[wasm_bindgen(getter, js_name = "finishReason")]
    pub fn finish_reason(&self) -> Option<String> {
        self.inner.finish_reason.clone()
    }

    /// Estimated cost for this request in USD, if available.
    #[wasm_bindgen(getter)]
    pub fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Token usage statistics as a `TokenUsage` object, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn usage(&self) -> Option<WasmTokenUsage> {
        self.inner.usage.clone().map(|u| WasmTokenUsage { inner: u })
    }

    /// Tool invocations requested by the model, as a JSON array.
    #[wasm_bindgen(getter, js_name = "toolCalls")]
    pub fn tool_calls(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.tool_calls)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Whether the model requested tool invocations.
    #[wasm_bindgen(getter, js_name = "hasToolCalls")]
    pub fn has_tool_calls(&self) -> bool {
        !self.inner.tool_calls.is_empty()
    }

    /// Provider-specific metadata as a JSON value.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.metadata)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl From<InnerCompletionResponse> for WasmCompletionResponse {
    fn from(inner: InnerCompletionResponse) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// WasmStreamChunk
// ---------------------------------------------------------------------------

/// A single chunk from a streaming completion response.
#[wasm_bindgen(js_name = "StreamChunk")]
pub struct WasmStreamChunk {
    inner: InnerStreamChunk,
}

#[wasm_bindgen(js_class = "StreamChunk")]
impl WasmStreamChunk {
    /// Incremental text content, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn delta(&self) -> Option<String> {
        self.inner.delta.clone()
    }

    /// Present in the final chunk to indicate why generation stopped.
    #[wasm_bindgen(getter, js_name = "finishReason")]
    pub fn finish_reason(&self) -> Option<String> {
        self.inner.finish_reason.clone()
    }

    /// Tool invocations completed in this chunk, as a JSON array.
    #[wasm_bindgen(getter, js_name = "toolCalls")]
    pub fn tool_calls(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.tool_calls)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl From<InnerStreamChunk> for WasmStreamChunk {
    fn from(inner: InnerStreamChunk) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// WasmTokenUsage
// ---------------------------------------------------------------------------

/// Token usage statistics for a completion request.
#[wasm_bindgen(js_name = "TokenUsage")]
pub struct WasmTokenUsage {
    inner: InnerTokenUsage,
}

#[wasm_bindgen(js_class = "TokenUsage")]
impl WasmTokenUsage {
    /// Number of tokens in the prompt / input.
    #[wasm_bindgen(getter, js_name = "promptTokens")]
    pub fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Number of tokens in the completion / output.
    #[wasm_bindgen(getter, js_name = "completionTokens")]
    pub fn completion_tokens(&self) -> u32 {
        self.inner.completion_tokens
    }

    /// Total tokens consumed (prompt + completion).
    #[wasm_bindgen(getter, js_name = "totalTokens")]
    pub fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }
}

// ---------------------------------------------------------------------------
// WasmToolCall
// ---------------------------------------------------------------------------

/// A tool invocation requested by the model.
#[wasm_bindgen(js_name = "ToolCall")]
pub struct WasmToolCall {
    inner: InnerToolCall,
}

#[wasm_bindgen(js_class = "ToolCall")]
impl WasmToolCall {
    /// Provider-assigned identifier for this invocation.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// The name of the tool to invoke.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The arguments to pass, as a JSON value.
    #[wasm_bindgen(getter)]
    pub fn arguments(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.arguments)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl From<InnerToolCall> for WasmToolCall {
    fn from(inner: InnerToolCall) -> Self {
        Self { inner }
    }
}
