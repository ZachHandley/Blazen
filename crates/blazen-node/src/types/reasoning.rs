//! Node wrapper for `ReasoningTrace`.
//!
//! Exposes both a plain `#[napi(object)]` shape ([`JsReasoningTrace`]) used
//! inside response objects, and a class wrapper ([`JsReasoningTraceClass`],
//! JS name `ReasoningTrace`) with getters and a constructor.

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

// ---------------------------------------------------------------------------
// JsReasoningTraceClass — class wrapper exposed to JS as `ReasoningTrace`
// ---------------------------------------------------------------------------

/// Options for constructing a [`JsReasoningTraceClass`] from JavaScript.
#[napi(object)]
pub struct ReasoningTraceOptions {
    /// Plain-text rendering of the reasoning content.
    pub text: String,
    /// Provider-specific signature/redaction handle, if any (Anthropic).
    pub signature: Option<String>,
    /// Whether the trace was redacted by the provider.
    pub redacted: Option<bool>,
    /// Reasoning effort level if the provider exposes one
    /// (e.g. `"low"`, `"medium"`, `"high"`).
    pub effort: Option<String>,
}

/// Class wrapper around [`ReasoningTrace`].
///
/// ```typescript
/// import { ReasoningTrace } from 'blazen';
///
/// const r = new ReasoningTrace({ text: "step-by-step thinking", effort: "high" });
/// console.log(r.text, r.effort);
/// ```
#[napi(js_name = "ReasoningTrace")]
pub struct JsReasoningTraceClass {
    pub(crate) inner: ReasoningTrace,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::needless_pass_by_value)]
impl JsReasoningTraceClass {
    /// Construct a reasoning trace.
    #[napi(constructor)]
    pub fn new(options: ReasoningTraceOptions) -> Self {
        Self {
            inner: ReasoningTrace {
                text: options.text,
                signature: options.signature,
                redacted: options.redacted.unwrap_or(false),
                effort: options.effort,
            },
        }
    }

    /// Plain-text rendering of the reasoning content.
    #[napi(getter)]
    pub fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Provider-specific signature/redaction handle, if any.
    #[napi(getter)]
    pub fn signature(&self) -> Option<String> {
        self.inner.signature.clone()
    }

    /// Whether the trace was redacted by the provider.
    #[napi(getter)]
    pub fn redacted(&self) -> bool {
        self.inner.redacted
    }

    /// Reasoning effort level if the provider exposes one.
    #[napi(getter)]
    pub fn effort(&self) -> Option<String> {
        self.inner.effort.clone()
    }
}

impl From<ReasoningTrace> for JsReasoningTraceClass {
    fn from(inner: ReasoningTrace) -> Self {
        Self { inner }
    }
}

impl From<&ReasoningTrace> for JsReasoningTraceClass {
    fn from(inner: &ReasoningTrace) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}
