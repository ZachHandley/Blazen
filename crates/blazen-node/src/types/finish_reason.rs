//! Node wrapper for `FinishReason`.

use napi_derive::napi;

use blazen_llm::FinishReason;

/// Normalized finish reason across providers.
///
/// Maps provider-specific finish-reason strings (`"stop"`, `"end_turn"`,
/// `"STOP"`, `"length"`, `"tool_calls"`, `"max_tokens"`, etc.) into a canonical
/// set. Unknown values fall through to `kind == "other"`.
///
/// Use `kind` for the canonical category and `value` for the original
/// provider string.
#[napi(object)]
pub struct JsFinishReason {
    /// Canonical category. One of: `"stop"`, `"length"`, `"tool_calls"`,
    /// `"content_filter"`, `"safety"`, `"end_turn"`, `"stop_sequence"`,
    /// `"max_tokens"`, `"error"`, or `"other"`.
    pub kind: String,
    /// Raw provider string. Equals `kind` for canonical variants; for `Other(s)`
    /// it preserves the original string.
    pub value: String,
}

impl From<&FinishReason> for JsFinishReason {
    fn from(fr: &FinishReason) -> Self {
        let (kind, value) = match fr {
            FinishReason::Stop => ("stop", "stop".to_owned()),
            FinishReason::Length => ("length", "length".to_owned()),
            FinishReason::ToolCalls => ("tool_calls", "tool_calls".to_owned()),
            FinishReason::ContentFilter => ("content_filter", "content_filter".to_owned()),
            FinishReason::Safety => ("safety", "safety".to_owned()),
            FinishReason::EndTurn => ("end_turn", "end_turn".to_owned()),
            FinishReason::StopSequence => ("stop_sequence", "stop_sequence".to_owned()),
            FinishReason::MaxTokens => ("max_tokens", "max_tokens".to_owned()),
            FinishReason::Error => ("error", "error".to_owned()),
            FinishReason::Other(s) => ("other", s.clone()),
        };
        Self {
            kind: kind.to_owned(),
            value,
        }
    }
}

impl From<FinishReason> for JsFinishReason {
    fn from(fr: FinishReason) -> Self {
        (&fr).into()
    }
}
