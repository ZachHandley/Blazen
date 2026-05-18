//! napi wrappers for the structured tool-output types.
//!
//! These mirror [`blazen_llm::types::ToolOutput`] and
//! [`blazen_llm::types::LlmPayload`] for the JS/TS surface so callers can
//! return a structured `{ data, llmOverride }` object from a tool handler
//! and inspect `msg.toolResult` on a returned `ChatMessage`.

use blazen_llm::types::{LlmPayload as RustLlmPayload, ProviderId, ToolOutput as RustToolOutput};
use napi_derive::napi;

use crate::types::message::JsContentPart;

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
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmPayload {
    /// Which variant: `"text"`, `"json"`, `"parts"`, or `"provider_raw"`.
    pub kind: String,
    /// Plain text body. Required for `kind: "text"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Structured JSON value. Required for `kind: "json"` and
    /// `kind: "provider_raw"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    /// Multimodal content parts. Required for `kind: "parts"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<JsContentPart>>,
    /// Provider id string. Required for `kind: "provider_raw"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
            RustLlmPayload::Parts { parts } => {
                let js_parts = parts
                    .iter()
                    .filter_map(|p| {
                        use blazen_llm::types::ContentPart;
                        let part_type = match p {
                            ContentPart::Text { .. } => "text",
                            ContentPart::Image(_) => "image",
                            ContentPart::File(_) => "file",
                            ContentPart::Audio(_) => "audio",
                            ContentPart::Video(_) => "video",
                        };
                        let mut jcp = JsContentPart {
                            part_type: part_type.into(),
                            text: None,
                            image: None,
                            file: None,
                            audio: None,
                            video: None,
                        };
                        // Convert payload-specific fields. We bridge the inner
                        // ImageContent/FileContent/etc. structs via serde
                        // round-trip — they all derive Serialize/Deserialize
                        // on both sides. Parts that fail to round-trip (which
                        // should never happen in practice) are silently
                        // dropped by `filter_map`.
                        match p {
                            ContentPart::Text { text } => jcp.text = Some(text.clone()),
                            ContentPart::Image(img) => {
                                jcp.image =
                                    serde_json::from_value(serde_json::to_value(img).ok()?).ok();
                            }
                            ContentPart::File(f) => {
                                jcp.file =
                                    serde_json::from_value(serde_json::to_value(f).ok()?).ok();
                            }
                            ContentPart::Audio(a) => {
                                jcp.audio =
                                    serde_json::from_value(serde_json::to_value(a).ok()?).ok();
                            }
                            ContentPart::Video(v) => {
                                jcp.video =
                                    serde_json::from_value(serde_json::to_value(v).ok()?).ok();
                            }
                        }
                        Some(jcp)
                    })
                    .collect::<Vec<_>>();
                Self {
                    kind: "parts".into(),
                    text: None,
                    value: None,
                    parts: Some(js_parts),
                    provider: None,
                }
            }
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

/// Convert a deserialized napi [`LlmPayload`] into the core
/// [`blazen_llm::types::LlmPayload`] enum.
///
/// The napi struct is a flat `{ kind, text?, value?, parts?, provider? }`
/// shape (napi `#[napi(object)]` types can't be tagged enums on the FFI
/// surface). This function dispatches on `kind` and validates that the
/// variant-specific fields are present.
pub(crate) fn js_llm_payload_to_rust(p: LlmPayload) -> napi::Result<blazen_llm::types::LlmPayload> {
    use blazen_llm::types::{LlmPayload as Rust, ProviderId};
    match p.kind.as_str() {
        "text" => Ok(Rust::Text {
            text: p
                .text
                .ok_or_else(|| napi::Error::from_reason("llmOverride kind=text requires `text`"))?,
        }),
        "json" => Ok(Rust::Json {
            value: p.value.ok_or_else(|| {
                napi::Error::from_reason("llmOverride kind=json requires `value`")
            })?,
        }),
        "parts" => {
            let parts = p.parts.unwrap_or_default();
            let rust_parts = crate::types::message::convert_js_parts(parts)?;
            Ok(Rust::Parts { parts: rust_parts })
        }
        "provider_raw" => {
            let provider_str = p.provider.ok_or_else(|| {
                napi::Error::from_reason("llmOverride kind=provider_raw requires `provider`")
            })?;
            let provider = match provider_str.as_str() {
                "openai" => ProviderId::OpenAi,
                "openai_compat" => ProviderId::OpenAiCompat,
                "azure" => ProviderId::Azure,
                "anthropic" => ProviderId::Anthropic,
                "gemini" => ProviderId::Gemini,
                "responses" => ProviderId::Responses,
                "fal" => ProviderId::Fal,
                other => {
                    return Err(napi::Error::from_reason(format!(
                        "llmOverride: unknown provider `{other}`"
                    )));
                }
            };
            Ok(Rust::ProviderRaw {
                provider,
                value: p.value.ok_or_else(|| {
                    napi::Error::from_reason("llmOverride kind=provider_raw requires `value`")
                })?,
            })
        }
        other => Err(napi::Error::from_reason(format!(
            "llmOverride: unknown kind `{other}`"
        ))),
    }
}
