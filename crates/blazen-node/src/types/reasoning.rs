//! Node wrapper for `ReasoningTrace`.

use napi_derive::napi;

use blazen_llm::ReasoningTrace;

/// Chain-of-thought / extended-thinking trace from a model that exposes one.
///
/// Populated by Anthropic extended thinking, `DeepSeek` R1 `reasoning_content`,
/// `OpenAI` o-series, xAI Grok reasoning, and Gemini thoughts.
#[napi(object)]
pub struct JsReasoningTrace {
    /// Plain-text rendering of the reasoning content.
    pub text: String,
    /// Provider-specific signature/redaction handle, if any (Anthropic).
    pub signature: Option<String>,
    /// Whether the trace was redacted by the provider.
    pub redacted: bool,
    /// Reasoning effort level if the provider exposes one
    /// (e.g. `"low"`, `"medium"`, `"high"`).
    pub effort: Option<String>,
}

impl From<&ReasoningTrace> for JsReasoningTrace {
    fn from(t: &ReasoningTrace) -> Self {
        Self {
            text: t.text.clone(),
            signature: t.signature.clone(),
            redacted: t.redacted,
            effort: t.effort.clone(),
        }
    }
}

impl From<ReasoningTrace> for JsReasoningTrace {
    fn from(t: ReasoningTrace) -> Self {
        Self {
            text: t.text,
            signature: t.signature,
            redacted: t.redacted,
            effort: t.effort,
        }
    }
}
