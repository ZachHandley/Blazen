//! Completion response and options types, plus the `build_response` helper.

use napi_derive::napi;

use blazen_llm::types::CompletionResponse;

use super::artifact::JsArtifact;
use super::citation::JsCitation;
use super::reasoning::JsReasoningTrace;
use crate::generated::{JsRequestTiming, JsTokenUsage, JsToolCall, JsToolDefinition};

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
    pub reasoning: Option<JsReasoningTrace>,
    pub citations: Vec<JsCitation>,
    pub artifacts: Vec<JsArtifact>,
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
        tool_calls: response.tool_calls.into_iter().map(Into::into).collect(),
        usage: response.usage.map(Into::into),
        model: response.model,
        finish_reason: response.finish_reason,
        cost: response.cost,
        timing: response.timing.map(Into::into),
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
        reasoning: response.reasoning.as_ref().map(JsReasoningTrace::from),
        citations: response.citations.iter().map(JsCitation::from).collect(),
        artifacts: response.artifacts.iter().map(JsArtifact::from).collect(),
        metadata: response.metadata,
    }
}
