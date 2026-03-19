//! Completion response and options types, plus the `build_response` helper.

use napi_derive::napi;

use blazen_llm::types::CompletionResponse;

use super::tool::{JsToolCall, JsToolDefinition};
use super::usage::{JsRequestTiming, JsTokenUsage};

/// The result of a chat completion.
#[napi(object)]
pub struct JsCompletionResponse {
    pub content: Option<String>,
    #[napi(js_name = "toolCalls")]
    pub tool_calls: Vec<JsToolCall>,
    pub usage: Option<JsTokenUsage>,
    pub model: String,
    #[napi(js_name = "finishReason")]
    pub finish_reason: Option<String>,
    pub cost: Option<f64>,
    pub timing: Option<JsRequestTiming>,
    pub images: Vec<serde_json::Value>,
    pub audio: Vec<serde_json::Value>,
    pub videos: Vec<serde_json::Value>,
    pub metadata: serde_json::Value,
}

/// Options for a chat completion request.
#[napi(object)]
pub struct JsCompletionOptions {
    pub temperature: Option<f64>,
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<i64>,
    #[napi(js_name = "topP")]
    pub top_p: Option<f64>,
    pub model: Option<String>,
    pub tools: Option<Vec<JsToolDefinition>>,
    /// JSON Schema for structured output / response format.
    #[napi(js_name = "responseFormat")]
    pub response_format: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Helper: build a JsCompletionResponse from the internal type
// ---------------------------------------------------------------------------

pub(crate) fn build_response(response: CompletionResponse) -> JsCompletionResponse {
    JsCompletionResponse {
        content: response.content,
        tool_calls: response
            .tool_calls
            .into_iter()
            .map(|tc| JsToolCall {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
            })
            .collect(),
        usage: response.usage.map(|u| JsTokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }),
        model: response.model,
        finish_reason: response.finish_reason,
        cost: response.cost,
        #[allow(clippy::cast_possible_wrap)]
        timing: response.timing.map(|t| JsRequestTiming {
            queue_ms: t.queue_ms.map(|v| v as i64),
            execution_ms: t.execution_ms.map(|v| v as i64),
            total_ms: t.total_ms.map(|v| v as i64),
        }),
        images: response
            .images
            .iter()
            .map(|img| serde_json::to_value(img).unwrap_or_default())
            .collect(),
        audio: response
            .audio
            .iter()
            .map(|a| serde_json::to_value(a).unwrap_or_default())
            .collect(),
        videos: response
            .videos
            .iter()
            .map(|v| serde_json::to_value(v).unwrap_or_default())
            .collect(),
        metadata: response.metadata,
    }
}
