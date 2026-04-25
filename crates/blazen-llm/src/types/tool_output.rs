//! Two-channel tool output: what the user/caller sees vs. what the LLM
//! sees on the next turn. See the workspace docs for the design rationale.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::message::ContentPart;

/// What a tool returns. Carries two channels:
/// - `data`: the typed value the caller sees programmatically (full structured payload).
/// - `llm_override`: explicit override for what to send to the LLM on the next turn.
///   `None` means each provider applies its default conversion from `data`.
///
/// Generic over `T` (defaults to `Value`) so [`crate::TypedTool`] can carry a
/// concrete `Output` type up to the trait boundary, where it gets type-erased
/// to `ToolOutput<Value>` for `dyn Tool` heterogeneity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
pub struct ToolOutput<T = Value> {
    pub data: T,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub llm_override: Option<LlmPayload>,
}

impl<T> ToolOutput<T> {
    pub fn new(data: T) -> Self {
        Self {
            data,
            llm_override: None,
        }
    }

    pub fn with_override(data: T, override_payload: LlmPayload) -> Self {
        Self {
            data,
            llm_override: Some(override_payload),
        }
    }
}

/// Ergonomic conversion: `Ok(value.into())` from a tool returns
/// `ToolOutput { data: value, llm_override: None }`.
impl From<Value> for ToolOutput<Value> {
    fn from(data: Value) -> Self {
        Self::new(data)
    }
}

/// What to send to the LLM as the tool result body. Provider-aware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmPayload {
    /// Plain text — works on every provider universally.
    Text { text: String },

    /// Structured JSON. Anthropic and Gemini consume natively;
    /// `OpenAI` / Responses stringify at the wire boundary.
    Json { value: Value },

    /// Multimodal content blocks (Anthropic supports natively as
    /// `tool_result.content` blocks; `OpenAI` falls back to text;
    /// Gemini falls back to a JSON object).
    Parts { parts: Vec<ContentPart> },

    /// Provider-specific escape hatch. The named provider gets `value`
    /// inserted verbatim into its tool-result body; every other provider
    /// falls back to the default conversion from `ToolOutput::data`.
    ProviderRaw { provider: ProviderId, value: Value },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(tsify_next::Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "snake_case")]
pub enum ProviderId {
    /// `openai` on the wire (single word, matching the canonical product name).
    #[serde(rename = "openai")]
    OpenAi,
    /// `openai_compat` on the wire (any OpenAI-compatible endpoint).
    #[serde(rename = "openai_compat")]
    OpenAiCompat,
    Azure,
    Anthropic,
    Gemini,
    Responses,
    Fal,
}
