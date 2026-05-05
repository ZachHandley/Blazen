//! Typed mirrors of [`blazen_llm::CompletionRequest`],
//! [`blazen_llm::StructuredResponse`], [`blazen_llm::types::MessageContent`],
//! and [`blazen_llm::types::FileContent`].
//!
//! Adds plain-data shapes for callers who want to construct request
//! envelopes outside the [`crate::providers::JsCompletionModel`] factory
//! methods, plus the file-content variant the existing
//! [`crate::types::message`] module did not surface.

use napi_derive::napi;

use blazen_llm::types::FileContent as RustFileContent;

use super::artifact::JsArtifact;
use super::citation::JsCitation;
use super::message::{JsContentPart, JsImageSource, rust_source_to_js};
use super::reasoning::JsReasoningTrace;
use crate::generated::{JsRequestTiming, JsTokenUsage, JsToolDefinition};

// ---------------------------------------------------------------------------
// JsFileContent
// ---------------------------------------------------------------------------

/// File / document content (PDF, generic file, etc.) for multimodal
/// messages. Mirrors [`blazen_llm::types::FileContent`].
#[napi(object, js_name = "FileContent")]
pub struct JsFileContent {
    /// The source of the file data.
    pub source: JsImageSource,
    /// The MIME type of the file (e.g. `"application/pdf"`).
    #[napi(js_name = "mediaType")]
    pub media_type: String,
    /// An optional filename for display purposes.
    pub filename: Option<String>,
}

impl From<&RustFileContent> for JsFileContent {
    fn from(f: &RustFileContent) -> Self {
        Self {
            source: rust_source_to_js(&f.source),
            media_type: f.media_type.clone(),
            filename: f.filename.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// JsMessageContent
// ---------------------------------------------------------------------------

/// Tagged-union mirror of [`blazen_llm::types::MessageContent`].
///
/// `kind` is one of:
/// - `"text"` -- `text` is populated with the string content.
/// - `"image"` -- `image` is populated with the [`JsContentPart`] image.
/// - `"parts"` -- `parts` is populated with the multi-part content list.
#[napi(object, js_name = "MessageContent")]
pub struct JsMessageContent {
    /// Discriminant: `"text"`, `"image"`, or `"parts"`.
    pub kind: String,
    /// Plain text body. Required for `kind: "text"`.
    pub text: Option<String>,
    /// Single image content. Required for `kind: "image"`.
    pub image: Option<JsContentPart>,
    /// Multi-part content. Required for `kind: "parts"`.
    pub parts: Option<Vec<JsContentPart>>,
}

// ---------------------------------------------------------------------------
// JsCompletionRequest
// ---------------------------------------------------------------------------

/// Provider-agnostic request for a chat completion.
///
/// Mirrors [`blazen_llm::CompletionRequest`]. Most callers reach for the
/// [`crate::providers::JsCompletionModel`] factory + per-call options
/// path; this typed shape exists for callers who need to build a request
/// envelope explicitly (e.g. forwarding the same request through multiple
/// middleware layers).
#[napi(object, js_name = "CompletionRequest")]
pub struct JsCompletionRequest {
    /// The conversation history as JSON-serialized `ChatMessage` values.
    ///
    /// Each entry must round-trip through `serde_json` into a Rust
    /// [`blazen_llm::ChatMessage`]. Use the `ChatMessage` class to build
    /// these in JS.
    pub messages: Vec<serde_json::Value>,
    /// Tools available for the model to invoke.
    pub tools: Option<Vec<JsToolDefinition>>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    #[napi(js_name = "topP")]
    pub top_p: Option<f64>,
    /// JSON-encoded response format hint (raw, matching the `OpenAI` shape
    /// or the typed [`crate::types::JsResponseFormat`] when serialized).
    #[napi(js_name = "responseFormat")]
    pub response_format: Option<serde_json::Value>,
    /// Override the provider's default model for this request.
    pub model: Option<String>,
    /// Output modalities (e.g., `["text"]`, `["image", "text"]`).
    pub modalities: Option<Vec<String>>,
    /// Image generation configuration (model-specific).
    #[napi(js_name = "imageConfig")]
    pub image_config: Option<serde_json::Value>,
    /// Audio output configuration (voice, format, etc.).
    #[napi(js_name = "audioConfig")]
    pub audio_config: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// JsStructuredResponse
// ---------------------------------------------------------------------------

/// Result of a structured-output extraction call.
///
/// Mirrors [`blazen_llm::types::StructuredResponse`]. The extracted typed
/// data is exposed as `data: serde_json::Value` for JS interop -- callers
/// can `JSON.parse` further or pass it through a typed Zod/io-ts schema.
#[napi(object, js_name = "StructuredResponse")]
pub struct JsStructuredResponse {
    /// The extracted structured data.
    pub data: serde_json::Value,
    /// Token usage statistics.
    pub usage: Option<JsTokenUsage>,
    /// The model that produced this response.
    pub model: String,
    /// Estimated cost in USD.
    pub cost: Option<f64>,
    /// Request timing.
    pub timing: Option<JsRequestTiming>,
    /// Provider-specific metadata.
    pub metadata: serde_json::Value,
    /// Chain-of-thought / extended-thinking trace, if exposed by the provider.
    pub reasoning: Option<JsReasoningTrace>,
    /// Citations backing the model's response.
    pub citations: Vec<JsCitation>,
    /// Typed artifacts extracted from or returned by the model.
    pub artifacts: Vec<JsArtifact>,
}

impl<T: serde::Serialize> From<blazen_llm::types::StructuredResponse<T>> for JsStructuredResponse {
    fn from(r: blazen_llm::types::StructuredResponse<T>) -> Self {
        Self {
            data: serde_json::to_value(&r.data).unwrap_or(serde_json::Value::Null),
            usage: r.usage.map(Into::into),
            model: r.model,
            cost: r.cost,
            timing: r.timing.map(Into::into),
            metadata: r.metadata,
            reasoning: r.reasoning.as_ref().map(JsReasoningTrace::from),
            citations: r.citations.iter().map(JsCitation::from).collect(),
            artifacts: r.artifacts.iter().map(JsArtifact::from).collect(),
        }
    }
}
