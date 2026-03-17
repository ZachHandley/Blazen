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
use std::task::{Context, Poll};
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::LlmError;
use crate::traits::{ModelCapabilities, ModelInfo, ModelRegistry};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentPart, ImageContent, ImageSource, MessageContent,
    Role, StreamChunk, TokenUsage, ToolCall,
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
    }
}

/// Convert a single [`ContentPart`] to a Gemini parts element.
fn content_part_to_gemini(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => serde_json::json!({ "text": text }),
        ContentPart::Image(img) => image_content_to_gemini(img),
        ContentPart::File(file) => match &file.source {
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
        },
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
#[derive(Debug, Clone)]
pub struct GeminiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl GeminiProvider {
    /// Create a new Gemini provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
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

    /// Build the JSON request body for the Gemini `generateContent` endpoint.
    #[allow(clippy::unused_self)]
    fn build_body(&self, request: &CompletionRequest) -> serde_json::Value {
        // Separate system instructions from conversation messages.
        let mut system_parts: Vec<String> = Vec::new();
        let mut contents: Vec<serde_json::Value> = Vec::new();

        for msg in &request.messages {
            if msg.role == Role::System {
                if let Some(text) = msg.content.text_content() {
                    system_parts.push(text);
                }
            } else {
                let role = match msg.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "model",
                    Role::System => unreachable!(),
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
            gen_config.insert("responseSchema".into(), fmt.clone());
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

    /// Send a request to the Gemini API, handling common errors.
    async fn send_request(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        let response = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let error_body = response
            .text()
            .await
            .unwrap_or_else(|_| String::from("<unable to read error body>"));

        match status.as_u16() {
            401 | 403 => Err(LlmError::AuthFailed),
            404 => Err(LlmError::ModelNotFound(error_body)),
            429 => Err(LlmError::RateLimited { retry_after: None }),
            _ => Err(LlmError::RequestFailed(format!(
                "HTTP {status}: {error_body}"
            ))),
        }
    }
}

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

fn parse_gemini_response(response: GeminiResponse) -> Result<CompletionResponse, LlmError> {
    let candidates = response
        .candidates
        .ok_or_else(|| LlmError::InvalidResponse("no candidates in response".into()))?;

    let candidate = candidates
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::InvalidResponse("empty candidates array".into()))?;

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    if let Some(content) = candidate.content
        && let Some(parts) = content.parts
    {
        for (i, part) in parts.into_iter().enumerate() {
            if let Some(text) = part.text {
                text_parts.push(text);
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

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let usage = response.usage_metadata.map(|u| TokenUsage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.total_token_count.unwrap_or(0),
    });

    let model = response
        .model_version
        .unwrap_or_else(|| "gemini".to_owned());

    Ok(CompletionResponse {
        content,
        tool_calls,
        usage,
        model,
        finish_reason: candidate.finish_reason,
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

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
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
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let result = parse_gemini_response(gemini)?;

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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError> {
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

        let response = self.send_request(&url, &body).await?;
        let byte_stream = response.bytes_stream();

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
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unable to read error body>"));
            return Err(LlmError::RequestFailed(format!(
                "HTTP {status}: {error_body}"
            )));
        }

        let models_response: GeminiModelsResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

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
                    },
                }
            })
            .collect();

        Ok(models)
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, LlmError> {
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
struct GeminiSseParser<S> {
    inner: S,
    buffer: String,
}

impl<S> GeminiSseParser<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl<S> Stream for GeminiSseParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
{
    type Item = Result<StreamChunk, LlmError>;

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
                    return Poll::Ready(Some(Err(LlmError::Stream(e.to_string()))));
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
fn parse_gemini_sse_event(buffer: &mut String) -> Option<Result<StreamChunk, LlmError>> {
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
                    }));
                }
                Err(e) => {
                    warn!(error = %e, data, "failed to parse Gemini SSE chunk");
                    return Some(Err(LlmError::Stream(format!(
                        "failed to parse Gemini SSE chunk: {e}"
                    ))));
                }
            }
        }
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
}
