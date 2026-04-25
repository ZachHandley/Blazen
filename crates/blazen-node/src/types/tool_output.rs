//! napi wrappers for the structured tool-output types.
//!
//! These mirror [`blazen_llm::types::ToolOutput`] and
//! [`blazen_llm::types::LlmPayload`] for the JS/TS surface so callers can
//! return a structured `{ data, llmOverride }` object from a tool handler
//! and inspect `msg.toolResult` on a returned `ChatMessage`.

use blazen_llm::types::{LlmPayload as RustLlmPayload, ProviderId, ToolOutput as RustToolOutput};
use napi_derive::napi;

/// `ToolOutput` carrying both the user-visible `data` and an optional
/// `llmOverride` controlling what the LLM sees on the next turn.
///
/// Either return one from a JS tool handler:
///   `return { data: { items: [1,2,3] }, llmOverride: { kind: 'text', text: 'summary' } };`
/// Or just return a plain value, which will be auto-wrapped with no override.
#[napi(object)]
pub struct ToolOutput {
    /// The full structured payload the caller sees programmatically.
    pub data: serde_json::Value,
    /// Optional override for what the LLM sees on the next turn.
    /// `None`/`undefined` means the provider applies its default conversion
    /// from `data`.
    #[napi(js_name = "llmOverride")]
    pub llm_override: Option<LlmPayload>,
}

/// Variant-tagged provider-aware override for what the LLM sees.
///
/// `kind` is one of `"text"`, `"json"`, `"parts"`, or `"provider_raw"`.
/// Other fields are populated based on the variant:
/// - `text` for `kind: "text"`
/// - `value` for `kind: "json"` or `kind: "provider_raw"`
/// - `parts` for `kind: "parts"` (serialized `ContentPart[]`)
/// - `provider` for `kind: "provider_raw"` — one of `"openai"`,
///   `"openai_compat"`, `"azure"`, `"anthropic"`, `"gemini"`, `"responses"`,
///   or `"fal"`.
#[napi(object)]
pub struct LlmPayload {
    /// Which variant: `"text"`, `"json"`, `"parts"`, or `"provider_raw"`.
    pub kind: String,
    /// Plain text body. Required for `kind: "text"`.
    pub text: Option<String>,
    /// Structured JSON value. Required for `kind: "json"` and
    /// `kind: "provider_raw"`.
    pub value: Option<serde_json::Value>,
    /// Multimodal content parts (serialized `ContentPart[]`).
    /// Required for `kind: "parts"`.
    pub parts: Option<serde_json::Value>,
    /// Provider id string. Required for `kind: "provider_raw"`.
    pub provider: Option<String>,
}

impl LlmPayload {
    pub(crate) fn from_rust(payload: &RustLlmPayload) -> Self {
        match payload {
            RustLlmPayload::Text { text } => Self {
                kind: "text".into(),
                text: Some(text.clone()),
                value: None,
                parts: None,
                provider: None,
            },
            RustLlmPayload::Json { value } => Self {
                kind: "json".into(),
                text: None,
                value: Some(value.clone()),
                parts: None,
                provider: None,
            },
            RustLlmPayload::Parts { parts } => Self {
                kind: "parts".into(),
                text: None,
                value: None,
                parts: Some(serde_json::to_value(parts).unwrap_or_default()),
                provider: None,
            },
            RustLlmPayload::ProviderRaw { provider, value } => Self {
                kind: "provider_raw".into(),
                text: None,
                value: Some(value.clone()),
                parts: None,
                provider: Some(provider_to_str(*provider).into()),
            },
        }
    }
}

impl ToolOutput {
    pub(crate) fn from_rust(out: &RustToolOutput<serde_json::Value>) -> Self {
        Self {
            data: out.data.clone(),
            llm_override: out.llm_override.as_ref().map(LlmPayload::from_rust),
        }
    }
}

fn provider_to_str(p: ProviderId) -> &'static str {
    match p {
        ProviderId::OpenAi => "openai",
        ProviderId::OpenAiCompat => "openai_compat",
        ProviderId::Azure => "azure",
        ProviderId::Anthropic => "anthropic",
        ProviderId::Gemini => "gemini",
        ProviderId::Responses => "responses",
        ProviderId::Fal => "fal",
    }
}
