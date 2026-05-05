//! Google Gemini API provider.
//!
//! The Gemini API has a completely different wire format from `OpenAI`:
//!
//! - Auth: `x-goog-api-key` header
//! - Generate: `POST /v1beta/models/{model}:generateContent`
//! - Stream: `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`
//! - List models: `GET /v1beta/models`
//! - Request body uses `contents` with `parts` instead of `messages`
//! - System prompt is `systemInstruction` top-level field
//! - Tools use `functionDeclarations` format

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use tracing::{debug, warn};

use super::openai_format::parse_retry_after;
use super::{provider_http_error, provider_http_error_parts};
use crate::error::BlazenError;
use crate::http::{ByteStream, HttpClient, HttpRequest, HttpResponse};
use crate::retry::RetryConfig;
use crate::traits::{ModelCapabilities, ModelInfo, ModelRegistry};
use crate::types::{
    Citation, CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource,
    MessageContent, ReasoningTrace, Role, StreamChunk, TokenUsage, ToolCall,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// ---------------------------------------------------------------------------
// Multimodal helpers
// ---------------------------------------------------------------------------

/// Convert an [`ImageContent`] to a Gemini `parts` element.
fn image_content_to_gemini(img: &ImageContent) -> serde_json::Value {
    match &img.source {
        ImageSource::Base64 { data } => {
            let mime_type = img.media_type.as_deref().unwrap_or("image/png");
            serde_json::json!({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": data,
                }
            })
        }
        ImageSource::Url { url } => {
            let mime_type = img.media_type.as_deref().unwrap_or("image/png");
            serde_json::json!({
                "fileData": {
                    "mimeType": mime_type,
                    "fileUri": url,
                }
            })
        }
        ImageSource::File { .. } => {
            tracing::warn!(
                "gemini: local file source is not supported — use a URL or base64 source instead; \
                 image content dropped."
            );
            serde_json::Value::Null
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(provider, crate::types::ProviderId::Gemini) {
                // Gemini's Files API returns a fileUri like
                // `https://generativelanguage.googleapis.com/v1beta/files/<id>`.
                // We accept the bare ID and pass it through as `fileUri`;
                // callers responsible for upload should store the full URI in
                // the ID field.
                let mime_type = img.media_type.as_deref().unwrap_or("image/png");
                serde_json::json!({
                    "fileData": {
                        "mimeType": mime_type,
                        "fileUri": id,
                    }
                })
            } else {
                crate::content::render::warn_provider_file_mismatch(
                    crate::types::ProviderId::Gemini,
                    *provider,
                    id,
                    crate::content::render::MediaKindLabel::Image,
                );
                serde_json::Value::Null
            }
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Gemini,
                handle,
                crate::content::render::MediaKindLabel::Image,
            );
            serde_json::Value::Null
        }
    }
}

/// Convert a single [`ContentPart`] to a Gemini parts element.
fn content_part_to_gemini(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => serde_json::json!({ "text": text }),
        ContentPart::Image(img) => image_content_to_gemini(img),
        ContentPart::File(file) => file_part_to_gemini(file),
        ContentPart::Audio(audio) => audio_part_to_gemini(audio),
        ContentPart::Video(video) => video_part_to_gemini(video),
    }
}

fn file_part_to_gemini(file: &crate::types::FileContent) -> serde_json::Value {
    match &file.source {
        ImageSource::Base64 { data } => {
            serde_json::json!({
                "inlineData": {
                    "mimeType": file.media_type,
                    "data": data,
                }
            })
        }
        ImageSource::Url { url } => {
            serde_json::json!({
                "fileData": {
                    "mimeType": file.media_type,
                    "fileUri": url,
                }
            })
        }
        ImageSource::File { .. } => {
            tracing::warn!(
                "gemini: local file source is not supported — use a URL or base64 source \
                 instead; file content dropped."
            );
            serde_json::Value::Null
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(provider, crate::types::ProviderId::Gemini) {
                serde_json::json!({
                    "fileData": {
                        "mimeType": file.media_type,
                        "fileUri": id,
                    }
                })
            } else {
                crate::content::render::warn_provider_file_mismatch(
                    crate::types::ProviderId::Gemini,
                    *provider,
                    id,
                    crate::content::render::MediaKindLabel::File,
                );
                serde_json::Value::Null
            }
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Gemini,
                handle,
                crate::content::render::MediaKindLabel::File,
            );
            serde_json::Value::Null
        }
    }
}

fn audio_part_to_gemini(audio: &crate::types::AudioContent) -> serde_json::Value {
    // Gemini natively accepts audio via the same inlineData/fileData
    // shape used for images and files. Default mime: audio/mp3.
    let mime = audio.media_type.as_deref().unwrap_or("audio/mp3");
    match &audio.source {
        ImageSource::Base64 { data } => serde_json::json!({
            "inlineData": { "mimeType": mime, "data": data }
        }),
        ImageSource::Url { url } => serde_json::json!({
            "fileData": { "mimeType": mime, "fileUri": url }
        }),
        ImageSource::File { .. } => {
            tracing::warn!(
                "gemini: local file source is not supported — use a URL or base64 source \
                 instead; audio content dropped."
            );
            serde_json::Value::Null
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(provider, crate::types::ProviderId::Gemini) {
                serde_json::json!({
                    "fileData": { "mimeType": mime, "fileUri": id }
                })
            } else {
                crate::content::render::warn_provider_file_mismatch(
                    crate::types::ProviderId::Gemini,
                    *provider,
                    id,
                    crate::content::render::MediaKindLabel::Audio,
                );
                serde_json::Value::Null
            }
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Gemini,
                handle,
                crate::content::render::MediaKindLabel::Audio,
            );
            serde_json::Value::Null
        }
    }
}

fn video_part_to_gemini(video: &crate::types::VideoContent) -> serde_json::Value {
    // Gemini natively accepts video via the same inlineData/fileData
    // shape. Default mime: video/mp4.
    let mime = video.media_type.as_deref().unwrap_or("video/mp4");
    match &video.source {
        ImageSource::Base64 { data } => serde_json::json!({
            "inlineData": { "mimeType": mime, "data": data }
        }),
        ImageSource::Url { url } => serde_json::json!({
            "fileData": { "mimeType": mime, "fileUri": url }
        }),
        ImageSource::File { .. } => {
            tracing::warn!(
                "gemini: local file source is not supported — use a URL or base64 source \
                 instead; video content dropped."
            );
            serde_json::Value::Null
        }
        ImageSource::ProviderFile { provider, id } => {
            if matches!(provider, crate::types::ProviderId::Gemini) {
                serde_json::json!({
                    "fileData": { "mimeType": mime, "fileUri": id }
                })
            } else {
                crate::content::render::warn_provider_file_mismatch(
                    crate::types::ProviderId::Gemini,
                    *provider,
                    id,
                    crate::content::render::MediaKindLabel::Video,
                );
                serde_json::Value::Null
            }
        }
        ImageSource::Handle { handle } => {
            crate::content::render::warn_handle_unresolved(
                crate::types::ProviderId::Gemini,
                handle,
                crate::content::render::MediaKindLabel::Video,
            );
            serde_json::Value::Null
        }
    }
}

/// Convert [`MessageContent`] to a Gemini `parts` array.
fn content_to_gemini_parts(content: &MessageContent) -> Vec<serde_json::Value> {
    match content {
        MessageContent::Text(t) => vec![serde_json::json!({ "text": t })],
        MessageContent::Image(img) => vec![image_content_to_gemini(img)],
        MessageContent::Parts(parts) => parts.iter().map(content_part_to_gemini).collect(),
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A Google Gemini API provider.
///
/// # Examples
///
/// ```rust,no_run
/// use blazen_llm::providers::gemini::GeminiProvider;
///
/// let provider = GeminiProvider::new("AIza...")
///     .with_model("gemini-2.5-pro");
/// ```
pub struct GeminiProvider {
    client: Arc<dyn HttpClient>,
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl std::fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

impl Clone for GeminiProvider {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            retry_config: self.retry_config.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            default_model: self.default_model.clone(),
        }
    }
}

impl GeminiProvider {
    /// Create a new Gemini provider with the given API key.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: crate::default_http_client(),
            retry_config: None,
            api_key: api_key.into(),
            base_url: GEMINI_BASE_URL.to_owned(),
            default_model: "gemini-2.5-flash".to_owned(),
        }
    }

    /// Create a new Gemini provider with an explicit HTTP client backend.
    #[must_use]
    pub fn new_with_client(api_key: impl Into<String>, client: Arc<dyn HttpClient>) -> Self {
        Self {
            client,
            retry_config: None,
            api_key: api_key.into(),
            base_url: GEMINI_BASE_URL.to_owned(),
            default_model: "gemini-2.5-flash".to_owned(),
        }
    }

    /// Override the default model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Override the base URL (e.g. for Vertex AI or proxies).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Use a custom HTTP client backend.
    #[must_use]
    pub fn with_http_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// Return a clone of the underlying HTTP client.
    ///
    /// Escape hatch for power users who need to issue raw HTTP requests
    /// (custom headers, endpoints not yet covered by Blazen's typed
    /// surface, debugging) while reusing the same connection pool, TLS
    /// config, and timeouts as this provider.
    #[must_use]
    pub fn http_client(&self) -> Arc<dyn HttpClient> {
        Arc::clone(&self.client)
    }

    /// Set the provider-level default retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Build the JSON request body for the Gemini `generateContent` endpoint.
    #[allow(clippy::unused_self, clippy::too_many_lines)]
    fn build_body(&self, request: &CompletionRequest) -> serde_json::Value {
        // Separate system instructions from conversation messages.
        let mut system_parts: Vec<String> = Vec::new();
        let mut contents: Vec<serde_json::Value> = Vec::new();

        for msg in &request.messages {
            if msg.role == Role::System {
                if let Some(text) = msg.content.text_content() {
                    system_parts.push(text);
                }
            } else if msg.role == Role::Tool && msg.tool_call_id.is_some() {
                // Gemini expects tool results as functionResponse parts.
                // The `response` field requires a JSON object — multimodal
                // content cannot live there. We split text/JSON from any
                // image / audio / file / video parts and emit the latter as
                // a follow-up `Content` so the model sees them attached to
                // the tool turn.
                let split = tool_result_to_gemini_split(msg);
                contents.push(serde_json::json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.name.as_deref().unwrap_or("unknown"),
                            "response": split.response,
                        }
                    }],
                }));

                if !split.follow_up_parts.is_empty() {
                    let parts: Vec<serde_json::Value> = split
                        .follow_up_parts
                        .iter()
                        .map(content_part_to_gemini)
                        .filter(|v| !v.is_null())
                        .collect();
                    if !parts.is_empty() {
                        contents.push(serde_json::json!({
                            "role": "user",
                            "parts": parts,
                        }));
                    }
                }
            } else if msg.role == Role::Assistant && !msg.tool_calls.is_empty() {
                // Gemini represents tool calls as functionCall parts.
                let mut parts = content_to_gemini_parts(&msg.content);
                // Only keep non-empty text parts.
                parts.retain(|p| {
                    p.get("text")
                        .and_then(|t| t.as_str())
                        .is_none_or(|s| !s.is_empty())
                });
                for tc in &msg.tool_calls {
                    parts.push(serde_json::json!({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    }));
                }
                contents.push(serde_json::json!({
                    "role": "model",
                    "parts": parts,
                }));
            } else {
                let role = match msg.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "model",
                    Role::System => continue, // Already handled above
                };
                let parts = content_to_gemini_parts(&msg.content);
                contents.push(serde_json::json!({
                    "role": role,
                    "parts": parts,
                }));
            }
        }

        let mut body = serde_json::json!({
            "contents": contents,
        });

        // System instruction as top-level field.
        if !system_parts.is_empty() {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": system_parts.join("\n\n") }],
            });
        }

        // Generation config.
        let mut gen_config = serde_json::Map::new();
        if let Some(temp) = request.temperature {
            gen_config.insert("temperature".into(), serde_json::json!(temp));
        }
        if let Some(max) = request.max_tokens {
            gen_config.insert("maxOutputTokens".into(), serde_json::json!(max));
        }
        if let Some(top_p) = request.top_p {
            gen_config.insert("topP".into(), serde_json::json!(top_p));
        }
        if let Some(ref fmt) = request.response_format {
            gen_config.insert(
                "responseMimeType".into(),
                serde_json::json!("application/json"),
            );
            let schema = if fmt.get("type").and_then(|v| v.as_str()) == Some("json_schema") {
                fmt.get("json_schema")
                    .and_then(|js| js.get("schema"))
                    .cloned()
                    .unwrap_or_else(|| fmt.clone())
            } else {
                fmt.clone()
            };
            gen_config.insert("responseSchema".into(), schema);
        }
        if !gen_config.is_empty() {
            body["generationConfig"] = serde_json::Value::Object(gen_config);
        }

        // Tools (function declarations).
        if !request.tools.is_empty() {
            let function_declarations: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!([{
                "functionDeclarations": function_declarations,
            }]);
        }

        body
    }

    /// Resolve the model to use for this request, returning an owned String.
    fn resolve_model(&self, request: &CompletionRequest) -> String {
        request
            .model
            .clone()
            .unwrap_or_else(|| self.default_model.clone())
    }

    /// Build an [`HttpRequest`] for a Gemini endpoint.
    fn build_http_request(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<HttpRequest, BlazenError> {
        HttpRequest::post(url)
            .header("x-goog-api-key", &self.api_key)
            .json_body(body)
    }

    /// Send a request to the Gemini API, handling common errors.
    async fn send_request(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<HttpResponse, BlazenError> {
        let request = self.build_http_request(url, body)?;
        let response = self.client.send(request).await?;

        if response.is_success() {
            return Ok(response);
        }

        match response.status {
            401 | 403 => Err(BlazenError::auth("authentication failed")),
            404 => Err(BlazenError::model_not_found(response.text())),
            429 => Err(BlazenError::RateLimit {
                retry_after_ms: parse_retry_after(&response.headers),
            }),
            _ => Err(provider_http_error("gemini", url, &response)),
        }
    }
}

super::impl_simple_from_options!(GeminiProvider, "gemini");

// ---------------------------------------------------------------------------
// Wire types (response deserialization)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    usage_metadata: Option<GeminiUsageMetadata>,
    model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
    #[serde(default, rename = "groundingMetadata")]
    grounding_metadata: Option<GroundingMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
    #[allow(dead_code)]
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    text: Option<String>,
    function_call: Option<GeminiFunctionCall>,
    #[serde(default)]
    thought: bool,
    #[serde(default, rename = "thoughtSignature")]
    thought_signature: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GroundingMetadata {
    #[serde(default)]
    pub web_search_queries: Vec<String>,
    #[serde(default)]
    pub grounding_chunks: Vec<GroundingChunk>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GroundingChunk {
    #[serde(default)]
    pub web: Option<GroundingWeb>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct GroundingWeb {
    pub uri: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(clippy::struct_field_names)]
struct GeminiUsageMetadata {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
    total_token_count: Option<u32>,
}

// Model listing wire types

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModelsResponse {
    models: Option<Vec<GeminiModelEntry>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModelEntry {
    /// Full resource name, e.g. "models/gemini-2.5-flash"
    name: String,
    display_name: Option<String>,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
    input_token_limit: Option<u64>,
    #[allow(dead_code)]
    output_token_limit: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper: parse a Gemini response into our types
// ---------------------------------------------------------------------------

fn parse_gemini_response(response: GeminiResponse) -> Result<CompletionResponse, BlazenError> {
    let candidates = response
        .candidates
        .ok_or_else(|| BlazenError::invalid_response("no candidates in response"))?;

    let candidate = candidates
        .into_iter()
        .next()
        .ok_or_else(|| BlazenError::invalid_response("empty candidates array"))?;

    let mut text_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut last_thought_signature: Option<String> = None;
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    if let Some(content) = candidate.content.as_ref()
        && let Some(parts) = content.parts.as_ref()
    {
        for (i, part) in parts.iter().enumerate() {
            if let Some(text) = part.text.as_ref() {
                if part.thought {
                    reasoning_parts.push(text.clone());
                    if let Some(sig) = part.thought_signature.as_ref() {
                        last_thought_signature = Some(sig.clone());
                    }
                } else {
                    text_parts.push(text.clone());
                }
            } else if part.thought
                && let Some(sig) = part.thought_signature.as_ref()
            {
                last_thought_signature = Some(sig.clone());
            }
            if let Some(fc) = part.function_call.as_ref() {
                tool_calls.push(ToolCall {
                    id: format!("gemini_call_{i}"),
                    name: fc.name.clone(),
                    arguments: fc.args.clone().unwrap_or(serde_json::Value::Null),
                });
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(ReasoningTrace {
            text: reasoning_parts.join("\n"),
            signature: last_thought_signature,
            redacted: false,
            effort: None,
        })
    };

    let mut citations: Vec<Citation> = Vec::new();
    if let Some(gm) = candidate.grounding_metadata.as_ref() {
        for chunk in &gm.grounding_chunks {
            if let Some(web) = chunk.web.as_ref() {
                citations.push(Citation {
                    url: web.uri.clone(),
                    title: web.title.clone(),
                    snippet: None,
                    start: None,
                    end: None,
                    document_id: None,
                    metadata: serde_json::json!({
                        "web_search_queries": gm.web_search_queries,
                    }),
                });
            }
        }
    }

    let usage = response.usage_metadata.map(|u| TokenUsage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.total_token_count.unwrap_or(0),
        ..Default::default()
    });

    let model = response
        .model_version
        .unwrap_or_else(|| "gemini".to_owned());

    Ok(CompletionResponse {
        content,
        tool_calls,
        reasoning,
        citations,
        artifacts: vec![],
        usage,
        model,
        finish_reason: candidate.finish_reason,
        cost: None,
        timing: None,
        images: vec![],
        audio: vec![],
        videos: vec![],
        metadata: serde_json::Value::Null,
    })
}

// ---------------------------------------------------------------------------
// CompletionModel implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl crate::traits::CompletionModel for GeminiProvider {
    fn model_id(&self) -> &str {
        &self.default_model
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config.as_ref()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        Some(Self::http_client(self))
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        let model = self.resolve_model(&request);
        let span = tracing::info_span!(
            "llm.complete",
            provider = "gemini",
            model = %model,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let url = format!("{}/models/{}:generateContent", self.base_url, model);
        let body = self.build_body(&request);
        debug!(%model, "Gemini completion request");

        let response = self.send_request(&url, &body).await?;
        let gemini: GeminiResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let mut result = parse_gemini_response(gemini)?;

        result.cost = result
            .usage
            .as_ref()
            .and_then(|u| crate::pricing::compute_cost(&result.model, u));

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        if let Some(ref u) = result.usage {
            span.record("prompt_tokens", u.prompt_tokens);
            span.record("completion_tokens", u.completion_tokens);
            span.record("total_tokens", u.total_tokens);
        }
        if let Some(ref reason) = result.finish_reason {
            span.record("finish_reason", reason.as_str());
        }

        Ok(result)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        let model = self.resolve_model(&request);
        let span = tracing::info_span!(
            "llm.stream",
            provider = "gemini",
            model = %model,
            duration_ms = tracing::field::Empty,
            chunk_count = tracing::field::Empty,
        );
        let _enter = span.enter();
        let start = Instant::now();

        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        );
        let body = self.build_body(&request);
        debug!(%model, "Gemini streaming request");

        let http_request = self.build_http_request(&url, &body)?;
        let (status, headers, byte_stream) = self.client.send_streaming(http_request).await?;

        if !(200..300).contains(&status) {
            match status {
                401 | 403 => return Err(BlazenError::auth("authentication failed")),
                404 => return Err(BlazenError::model_not_found("model not found")),
                429 => {
                    return Err(BlazenError::RateLimit {
                        retry_after_ms: parse_retry_after(&headers),
                    });
                }
                _ => {
                    return Err(provider_http_error_parts(
                        "gemini", &url, status, &headers, "",
                    ));
                }
            }
        }

        span.record(
            "duration_ms",
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        );

        let stream = GeminiSseParser::new(byte_stream);
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ModelRegistry for GeminiProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, BlazenError> {
        let url = format!("{}/models", self.base_url);
        let request = HttpRequest::get(&url).header("x-goog-api-key", &self.api_key);
        let response = self.client.send(request).await?;

        if !response.is_success() {
            return Err(provider_http_error("gemini", &url, &response));
        }

        let models_response: GeminiModelsResponse = response
            .json()
            .map_err(|e| BlazenError::invalid_response(e.to_string()))?;

        let models = models_response
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|entry| {
                // Strip the "models/" prefix to get the model id.
                let id = entry
                    .name
                    .strip_prefix("models/")
                    .unwrap_or(&entry.name)
                    .to_owned();

                let has_generate = entry
                    .supported_generation_methods
                    .iter()
                    .any(|m| m == "generateContent");
                let has_embeddings = entry
                    .supported_generation_methods
                    .iter()
                    .any(|m| m == "embedContent");

                ModelInfo {
                    id,
                    name: entry.display_name,
                    provider: "gemini".into(),
                    context_length: entry.input_token_limit,
                    pricing: None, // Gemini model listing does not include pricing.
                    capabilities: ModelCapabilities {
                        chat: has_generate,
                        streaming: has_generate,
                        tool_use: has_generate,
                        structured_output: has_generate,
                        vision: has_generate, // Most Gemini models support vision.
                        image_generation: false,
                        embeddings: has_embeddings,
                        ..Default::default()
                    },
                }
            })
            .collect();

        Ok(models)
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, BlazenError> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.id == model_id))
    }
}

// ---------------------------------------------------------------------------
// SSE stream parser for Gemini
// ---------------------------------------------------------------------------

/// Parses an SSE byte stream from the Gemini `streamGenerateContent` endpoint
/// into [`StreamChunk`]s.
///
/// Gemini SSE uses `data: <json>` lines like `OpenAI`, but the JSON payload is
/// a full `GeminiResponse` per chunk (not the `OpenAI` delta format).
struct GeminiSseParser {
    inner: ByteStream,
    buffer: String,
}

impl GeminiSseParser {
    fn new(inner: ByteStream) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl Stream for GeminiSseParser {
    type Item = Result<StreamChunk, BlazenError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if let Some(chunk) = parse_gemini_sse_event(&mut this.buffer) {
                return Poll::Ready(Some(chunk));
            }

            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes);
                    this.buffer.push_str(&text);
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(BlazenError::stream_error(e.to_string()))));
                }
                Poll::Ready(None) => {
                    if !this.buffer.is_empty()
                        && let Some(chunk) = parse_gemini_sse_event(&mut this.buffer)
                    {
                        return Poll::Ready(Some(chunk));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

/// Try to extract the next SSE event from the Gemini buffer.
///
/// Each `data:` line contains a full `GeminiResponse` JSON object. We parse
/// it and convert to a `StreamChunk`.
fn parse_gemini_sse_event(buffer: &mut String) -> Option<Result<StreamChunk, BlazenError>> {
    loop {
        let newline_pos = buffer.find('\n')?;
        let line = buffer[..newline_pos].trim().to_owned();
        buffer.drain(..=newline_pos);

        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data: ") {
            let data = data.trim();

            if data == "[DONE]" {
                return Some(Ok(StreamChunk {
                    delta: None,
                    tool_calls: Vec::new(),
                    finish_reason: Some("stop".to_owned()),
                    ..Default::default()
                }));
            }

            match serde_json::from_str::<GeminiResponse>(data) {
                Ok(response) => {
                    let Some(candidates) = response.candidates else {
                        continue;
                    };

                    let Some(candidate) = candidates.into_iter().next() else {
                        continue;
                    };

                    let mut text_delta: Option<String> = None;
                    let mut tool_calls: Vec<ToolCall> = Vec::new();

                    if let Some(content) = candidate.content
                        && let Some(parts) = content.parts
                    {
                        for (i, part) in parts.into_iter().enumerate() {
                            if let Some(text) = part.text {
                                text_delta = Some(text);
                            }
                            if let Some(fc) = part.function_call {
                                tool_calls.push(ToolCall {
                                    id: format!("gemini_call_{i}"),
                                    name: fc.name,
                                    arguments: fc.args.unwrap_or(serde_json::Value::Null),
                                });
                            }
                        }
                    }

                    return Some(Ok(StreamChunk {
                        delta: text_delta,
                        tool_calls,
                        finish_reason: candidate.finish_reason,
                        ..Default::default()
                    }));
                }
                Err(e) => {
                    warn!(error = %e, data, "failed to parse Gemini SSE chunk");
                    return Some(Err(BlazenError::stream_error(format!(
                        "failed to parse Gemini SSE chunk: {e}"
                    ))));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ProviderInfo implementation
// ---------------------------------------------------------------------------

impl crate::traits::ProviderInfo for GeminiProvider {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: true,
            model_listing: true,
            embeddings: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool-result helpers
// ---------------------------------------------------------------------------

/// Result of splitting a tool-result message for Gemini serialization.
///
/// Gemini's `functionResponse.response` field requires a JSON object; it
/// does not accept multimodal content. Image / audio / file / video parts
/// are emitted as a separate immediately-following `Content` with role
/// `user` carrying `inlineData` / `fileData` parts so the model can see
/// them in connection with the tool output.
pub(super) struct GeminiToolResultSplit {
    /// The `functionResponse.response` JSON object.
    pub response: serde_json::Value,
    /// Non-text content parts to emit as a follow-up `Content` after the
    /// `functionResponse` Content. Empty if the tool result is text-only.
    pub follow_up_parts: Vec<crate::types::ContentPart>,
}

/// Render a tool-result payload for Gemini, splitting any multimodal
/// content into a separate follow-up `Content`. The `response` field is
/// always a JSON object as Gemini requires; scalar text is wrapped as
/// `{"result": <text>}`.
///
/// Considers both `tool_result.llm_override = LlmPayload::Parts` and
/// `MessageContent::Parts` (from
/// [`crate::types::ChatMessage::tool_result_parts`]) as multimodal sources.
pub(super) fn tool_result_to_gemini_split(
    msg: &crate::types::ChatMessage,
) -> GeminiToolResultSplit {
    use crate::types::{LlmPayload, MessageContent, Role};

    if msg.role != Role::Tool {
        return GeminiToolResultSplit {
            response: serde_json::json!({}),
            follow_up_parts: Vec::new(),
        };
    }

    if let Some(out) = &msg.tool_result {
        if let Some(payload) = &out.llm_override {
            return match payload {
                LlmPayload::Text { text } => GeminiToolResultSplit {
                    response: serde_json::json!({"result": text}),
                    follow_up_parts: Vec::new(),
                },
                LlmPayload::Json { value } => GeminiToolResultSplit {
                    response: match value {
                        serde_json::Value::Object(_) => value.clone(),
                        scalar => serde_json::json!({"result": scalar}),
                    },
                    follow_up_parts: Vec::new(),
                },
                LlmPayload::Parts { parts } => split_parts_for_gemini(parts),
                LlmPayload::ProviderRaw { provider, value }
                    if *provider == crate::types::ProviderId::Gemini =>
                {
                    GeminiToolResultSplit {
                        response: value.clone(),
                        follow_up_parts: Vec::new(),
                    }
                }
                LlmPayload::ProviderRaw { .. } => GeminiToolResultSplit {
                    response: default_gemini_response(&out.data),
                    follow_up_parts: Vec::new(),
                },
            };
        }
        return GeminiToolResultSplit {
            response: default_gemini_response(&out.data),
            follow_up_parts: Vec::new(),
        };
    }

    // `tool_result` is None; payload lives in `content`.
    match &msg.content {
        MessageContent::Parts(parts) => split_parts_for_gemini(parts),
        MessageContent::Text(s) => GeminiToolResultSplit {
            response: serde_json::json!({"result": s}),
            follow_up_parts: Vec::new(),
        },
        MessageContent::Image(img) => GeminiToolResultSplit {
            response: serde_json::json!({}),
            follow_up_parts: vec![crate::types::ContentPart::Image(img.clone())],
        },
    }
}

fn split_parts_for_gemini(parts: &[crate::types::ContentPart]) -> GeminiToolResultSplit {
    let mut text_chunks: Vec<String> = Vec::new();
    let mut non_text: Vec<crate::types::ContentPart> = Vec::new();
    for part in parts {
        match part {
            crate::types::ContentPart::Text { text } => text_chunks.push(text.clone()),
            other => non_text.push(other.clone()),
        }
    }
    let response = if text_chunks.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::json!({"result": text_chunks.join("\n")})
    };
    GeminiToolResultSplit {
        response,
        follow_up_parts: non_text,
    }
}

fn default_gemini_response(data: &serde_json::Value) -> serde_json::Value {
    match data {
        serde_json::Value::Object(_) => data.clone(),
        scalar => serde_json::json!({"result": scalar}),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ToolDefinition};

    #[test]
    fn default_model() {
        let provider = GeminiProvider::new("test-key");
        assert_eq!(provider.default_model, "gemini-2.5-flash");
    }

    #[test]
    fn with_model_override() {
        let provider = GeminiProvider::new("test-key").with_model("gemini-2.5-pro");
        assert_eq!(provider.default_model, "gemini-2.5-pro");
    }

    #[test]
    fn build_body_basic() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello");
    }

    #[test]
    fn build_body_with_system() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ]);

        let body = provider.build_body(&request);

        // System should be in systemInstruction, not contents.
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            "You are helpful."
        );
    }

    #[test]
    fn build_body_with_generation_config() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")])
            .with_temperature(0.7)
            .with_max_tokens(200)
            .with_top_p(0.9);

        let body = provider.build_body(&request);
        // f32 -> f64 conversion causes slight precision loss, so compare approximately.
        let temp = body["generationConfig"]["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature was {temp}");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 200);
        let top_p = body["generationConfig"]["topP"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 0.001, "topP was {top_p}");
    }

    #[test]
    fn build_body_with_tools() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]).with_tools(vec![
            ToolDefinition {
                name: "search".to_owned(),
                description: "Search the web".to_owned(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    }
                }),
            },
        ]);

        let body = provider.build_body(&request);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let decls = tools[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], "search");
    }

    #[test]
    fn test_text_backward_compat() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")]);

        let body = provider.build_body(&request);
        let contents = body["contents"].as_array().unwrap();
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "Hello");
    }

    #[test]
    fn test_build_body_image_url() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_url(
            "What is this?",
            "https://example.com/cat.jpg",
            Some("image/jpeg"),
        )]);

        let body = provider.build_body(&request);
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "What is this?");
        assert_eq!(parts[1]["fileData"]["mimeType"], "image/jpeg");
        assert_eq!(
            parts[1]["fileData"]["fileUri"],
            "https://example.com/cat.jpg"
        );
    }

    #[test]
    fn test_build_body_base64_image() {
        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user_image_base64(
            "Describe this",
            "abc123base64data",
            "image/png",
        )]);

        let body = provider.build_body(&request);
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "abc123base64data");
    }

    #[test]
    fn test_build_body_multipart() {
        use crate::types::{ContentPart, ImageContent, ImageSource};

        let provider = GeminiProvider::new("test-key");
        let request = CompletionRequest::new(vec![ChatMessage::user_parts(vec![
            ContentPart::Text {
                text: "First".into(),
            },
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/a.png".into(),
                },
                media_type: Some("image/png".into()),
            }),
            ContentPart::Text {
                text: "Second".into(),
            },
        ])]);

        let body = provider.build_body(&request);
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0]["text"], "First");
        assert!(parts[1].get("fileData").is_some());
        assert_eq!(parts[2]["text"], "Second");
    }

    #[test]
    fn parse_gemini_response_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8
            },
            "modelVersion": "gemini-2.5-flash"
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let result = parse_gemini_response(response).unwrap();

        assert_eq!(result.content.as_deref(), Some("Hello there!"));
        assert_eq!(result.finish_reason.as_deref(), Some("STOP"));
        assert_eq!(result.model, "gemini-2.5-flash");
        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 8);
    }

    #[test]
    fn parse_gemini_response_function_call() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        }"#;

        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let result = parse_gemini_response(response).unwrap();

        assert!(result.content.is_none());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].arguments["city"], "NYC");
    }

    #[test]
    fn parse_gemini_sse_text() {
        let mut buf =
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"finishReason\":null}]}\n\n"
                .to_owned();

        let result = parse_gemini_sse_event(&mut buf).unwrap().unwrap();
        assert_eq!(result.delta.as_deref(), Some("Hi"));
    }

    #[test]
    fn parse_gemini_sse_done() {
        let mut buf = "data: [DONE]\n\n".to_owned();

        let result = parse_gemini_sse_event(&mut buf).unwrap().unwrap();
        assert!(result.delta.is_none());
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn parse_gemini_model_list() {
        let json = r#"{
            "models": [{
                "name": "models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 8192
            }]
        }"#;

        let response: GeminiModelsResponse = serde_json::from_str(json).unwrap();
        let models = response.models.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "models/gemini-2.5-flash");
        assert_eq!(models[0].display_name.as_deref(), Some("Gemini 2.5 Flash"));
        assert!(
            models[0]
                .supported_generation_methods
                .contains(&"generateContent".to_owned())
        );
    }

    #[test]
    fn test_parse_grounding_metadata_to_citations() {
        let json_body = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "answer based on sources"}],
                    "role": "model"
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "webSearchQueries": ["query1"],
                    "groundingChunks": [
                        {"web": {"uri": "https://example.com/a", "title": "Source A"}},
                        {"web": {"uri": "https://example.com/b"}}
                    ]
                }
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        }"#;
        let parsed: GeminiResponse = serde_json::from_str(json_body).unwrap();
        let candidates = parsed.candidates.as_ref().unwrap();
        let candidate = &candidates[0];
        assert!(candidate.grounding_metadata.is_some());
        let gm = candidate.grounding_metadata.as_ref().unwrap();
        assert_eq!(gm.grounding_chunks.len(), 2);
        assert_eq!(
            gm.grounding_chunks[0].web.as_ref().unwrap().uri,
            "https://example.com/a"
        );

        // Also exercise the full parser to ensure citations are populated.
        let result = parse_gemini_response(parsed).unwrap();
        assert_eq!(result.citations.len(), 2);
        assert_eq!(result.citations[0].url, "https://example.com/a");
        assert_eq!(result.citations[0].title.as_deref(), Some("Source A"));
        assert_eq!(result.citations[1].url, "https://example.com/b");
        assert!(result.citations[1].title.is_none());
        assert_eq!(
            result.citations[0].metadata["web_search_queries"][0],
            "query1"
        );
    }

    #[test]
    fn test_parse_thoughts_to_reasoning_trace() {
        let json_body = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me think about this", "thought": true, "thoughtSignature": "sig123"},
                        {"text": "The answer is 42"}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        }"#;
        let parsed: GeminiResponse = serde_json::from_str(json_body).unwrap();
        let candidates = parsed.candidates.as_ref().unwrap();
        let parts = candidates[0]
            .content
            .as_ref()
            .unwrap()
            .parts
            .as_ref()
            .unwrap();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].thought);
        assert!(!parts[1].thought);
        assert_eq!(parts[0].thought_signature.as_deref(), Some("sig123"));

        // Exercise the full parser to ensure ReasoningTrace is built.
        let result = parse_gemini_response(parsed).unwrap();
        assert_eq!(result.content.as_deref(), Some("The answer is 42"));
        let reasoning = result.reasoning.expect("reasoning trace should be set");
        assert_eq!(reasoning.text, "Let me think about this");
        assert_eq!(reasoning.signature.as_deref(), Some("sig123"));
        assert!(!reasoning.redacted);
        assert!(reasoning.effort.is_none());
    }

    // -----------------------------------------------------------------------
    // tool_result_to_gemini_split helper
    // -----------------------------------------------------------------------

    #[test]
    fn gemini_helper_object_data_passes_through() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!({"k":"v"}));
        let split = super::tool_result_to_gemini_split(&msg);
        assert_eq!(split.response, serde_json::json!({"k":"v"}));
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn gemini_helper_scalar_wraps_in_result() {
        let msg = ChatMessage::tool_result("call_1", "search", serde_json::json!("hello"));
        let split = super::tool_result_to_gemini_split(&msg);
        assert_eq!(split.response, serde_json::json!({"result":"hello"}));
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn gemini_helper_text_override_wraps_in_result() {
        use crate::types::{LlmPayload, ToolOutput};
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({"k":"v"}),
                LlmPayload::Text {
                    text: "summary".into(),
                },
            ),
        );
        let split = super::tool_result_to_gemini_split(&msg);
        assert_eq!(split.response, serde_json::json!({"result":"summary"}));
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn gemini_helper_no_from_str_round_trip() {
        let original = serde_json::json!({"weather": {"temp": 72, "humidity": 0.5}});
        let msg = ChatMessage::tool_result("call_1", "search", original.clone());
        let split = super::tool_result_to_gemini_split(&msg);
        assert_eq!(split.response, original);
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn gemini_helper_parts_split_text_and_image() {
        use crate::types::{ContentPart, LlmPayload, ToolOutput};
        let msg = ChatMessage::tool_result(
            "call_1",
            "render",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![
                        ContentPart::text("rendered"),
                        ContentPart::image_base64("AAAA", "image/png"),
                    ],
                },
            ),
        );
        let split = super::tool_result_to_gemini_split(&msg);
        assert_eq!(split.response, serde_json::json!({"result":"rendered"}));
        assert_eq!(split.follow_up_parts.len(), 1);
    }

    #[test]
    fn gemini_helper_parts_text_only_yields_no_follow_up() {
        use crate::types::{ContentPart, LlmPayload, ToolOutput};
        let msg = ChatMessage::tool_result(
            "call_1",
            "search",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![ContentPart::text("a"), ContentPart::text("b")],
                },
            ),
        );
        let split = super::tool_result_to_gemini_split(&msg);
        assert_eq!(split.response, serde_json::json!({"result":"a\nb"}));
        assert!(split.follow_up_parts.is_empty());
    }

    #[test]
    fn gemini_build_body_emits_follow_up_for_image_tool_result() {
        use crate::types::{ContentPart, LlmPayload, ToolOutput};
        let provider = GeminiProvider::new("test-key");
        let tool_msg = ChatMessage::tool_result(
            "call_1",
            "render",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![
                        ContentPart::text("rendered"),
                        ContentPart::image_base64("AAAA", "image/png"),
                    ],
                },
            ),
        );
        let request = CompletionRequest::new(vec![tool_msg]);
        let body = provider.build_body(&request);
        let contents = body["contents"].as_array().unwrap();
        // functionResponse Content + follow-up user Content with inlineData.
        assert_eq!(contents.len(), 2);
        assert!(contents[0]["parts"][0].get("functionResponse").is_some());
        assert_eq!(
            contents[0]["parts"][0]["functionResponse"]["response"],
            serde_json::json!({"result":"rendered"})
        );
        assert_eq!(contents[1]["role"], "user");
        let follow_up_parts = contents[1]["parts"].as_array().unwrap();
        assert_eq!(follow_up_parts.len(), 1);
        assert!(follow_up_parts[0].get("inlineData").is_some());
        assert_eq!(follow_up_parts[0]["inlineData"]["mimeType"], "image/png");
    }

    #[test]
    fn gemini_build_body_text_only_tool_result_no_follow_up() {
        let provider = GeminiProvider::new("test-key");
        let tool_msg = ChatMessage::tool_result("call_1", "search", serde_json::json!("ok"));
        let request = CompletionRequest::new(vec![tool_msg]);
        let body = provider.build_body(&request);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert!(contents[0]["parts"][0].get("functionResponse").is_some());
    }

    #[test]
    fn gemini_build_body_no_bogus_parts_envelope() {
        // Regression guard: the old implementation emitted
        // `{"parts": parts}` inside `functionResponse.response`. This shape
        // is not recognized by Gemini and must never appear.
        use crate::types::{ContentPart, LlmPayload, ToolOutput};
        let provider = GeminiProvider::new("test-key");
        let tool_msg = ChatMessage::tool_result(
            "call_1",
            "render",
            ToolOutput::with_override(
                serde_json::json!({}),
                LlmPayload::Parts {
                    parts: vec![ContentPart::image_base64("AAAA", "image/png")],
                },
            ),
        );
        let request = CompletionRequest::new(vec![tool_msg]);
        let body = provider.build_body(&request);
        let response = &body["contents"][0]["parts"][0]["functionResponse"]["response"];
        assert!(
            response.get("parts").is_none(),
            "functionResponse.response must not contain a `parts` envelope; got: {response}",
        );
    }
}
