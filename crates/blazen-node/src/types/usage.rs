//! Token usage and request timing types.

use napi_derive::napi;

/// Token usage statistics for a completion request.
#[napi(object)]
pub struct JsTokenUsage {
    #[napi(js_name = "promptTokens")]
    pub prompt_tokens: u32,
    #[napi(js_name = "completionTokens")]
    pub completion_tokens: u32,
    #[napi(js_name = "totalTokens")]
    pub total_tokens: u32,
}

/// Timing metadata for a completion request.
#[napi(object)]
pub struct JsRequestTiming {
    #[napi(js_name = "queueMs")]
    pub queue_ms: Option<i64>,
    #[napi(js_name = "executionMs")]
    pub execution_ms: Option<i64>,
    #[napi(js_name = "totalMs")]
    pub total_ms: Option<i64>,
}
