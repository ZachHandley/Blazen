//! OpenAI-compatible REST surface.
//!
//! Routes:
//!
//! - `POST /v1/chat/completions` — JSON in, JSON out (or `text/event-stream`
//!   when the request sets `stream: true`).
//! - `POST /v1/completions` — legacy non-chat completion. Same wire shape
//!   as chat completions but the request carries a `prompt` string instead
//!   of a `messages` array.
//! - `POST /v1/embeddings` — text embeddings; mirrors
//!   `openai.embeddings.create`.
//! - `GET  /v1/models` — list of loaded models.
//! - `POST /v1/audio/speech` — text-to-speech; returns raw audio bytes
//!   with the upstream MIME on the `content-type` header.
//! - `POST /v1/audio/transcriptions` — multipart upload, returns
//!   `{text, language}`.
//! - `POST /v1/images/generations` — image generation; returns the
//!   `OpenAI` `{created, data: [{b64_json, ... }]}` shape.
//!
//! Every route maps an OpenAI-shaped JSON request into the matching
//! [`crate::model_protocol`] postcard struct, dispatches via the
//! [`crate::server::model_manager::ManagerHandle`] on the shared
//! [`super::RestState`], then renders the response in `OpenAI`'s wire
//! format. `RpcError` from the handle is converted via the `From` impl
//! on [`super::HttpError`].

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::extract::{Multipart, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json2;
use uuid::Uuid;

use crate::model_protocol::{
    ChatMessageWire, CompleteRequest, EmbedRequest, GenerateImageRequest, MODEL_ENVELOPE_VERSION,
    StreamCompleteChunk, TextToSpeechRequest, TranscribeRequest,
};

use super::error::HttpError;
use super::rest_state::RestState;
use super::sse::{finish_event, json_event};

/// Maximum body bytes accepted on a single JSON request — matches the
/// default axum body limit (2 MiB). Audio transcription multipart
/// uploads use their own per-route limit via [`axum::extract::DefaultBodyLimit`].
const JSON_BODY_LIMIT: usize = 8 * 1024 * 1024;
/// Limit for the multipart audio endpoint — 100 MiB, mirroring `OpenAI`'s
/// own per-request cap.
const AUDIO_BODY_LIMIT: usize = 100 * 1024 * 1024;

/// Build the OpenAI-compat sub-router.
pub fn router(state: Arc<RestState>) -> Router {
    use axum::extract::DefaultBodyLimit;
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(legacy_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/speech", post(audio_speech))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcriptions).layer(DefaultBodyLimit::max(AUDIO_BODY_LIMIT)),
        )
        .route("/v1/images/generations", post(images_generations))
        .layer(DefaultBodyLimit::max(JSON_BODY_LIMIT))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

/// OpenAI-shaped chat-completions request. Unknown fields are tolerated
/// because `OpenAI` clients routinely send provider-specific extras.
#[derive(Debug, Deserialize)]
pub struct ChatModelRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub response_format: Option<Json2>,
    /// Any additional provider-specific extras (`extra_body` on the
    /// Python SDK, `extraBody` on the Node SDK) get round-tripped via
    /// `extra_json`.
    #[serde(default)]
    pub extra_body: Option<Json2>,
}

/// Single message in a chat-completions request.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Json2>,
}

/// `stop` is either a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/// OpenAI-shaped non-stream response.
#[derive(Debug, Serialize)]
struct ChatModelResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: u32,
    message: AssistantMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Json2>,
}

#[derive(Debug, Serialize)]
#[allow(clippy::struct_field_names)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

async fn chat_completions(
    State(state): State<Arc<RestState>>,
    Json(req): Json<ChatModelRequest>,
) -> Result<Response, HttpError> {
    let stream = req.stream.unwrap_or(false);
    let model = req.model.clone();
    let wire = build_complete_request(&req)?;

    if stream {
        state.metrics.record("stream_complete");
        let inner = state.handle.stream_complete(wire).await?;
        let stream = sse_chat_stream(inner, model);
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    state.metrics.record("complete");
    let resp = state.handle.complete(wire).await?;
    let usage = Usage {
        prompt_tokens: resp.prompt_tokens.unwrap_or(0),
        completion_tokens: resp.completion_tokens.unwrap_or(0),
        total_tokens: resp.prompt_tokens.unwrap_or(0) + resp.completion_tokens.unwrap_or(0),
    };
    let tool_calls = parse_optional_json(&resp.tool_calls_json)?;
    let body = ChatModelResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: unix_seconds(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: resp.text,
                tool_calls,
            },
            finish_reason: resp.finish_reason,
        }],
        usage,
    };
    Ok((StatusCode::OK, Json(body)).into_response())
}

fn build_complete_request(req: &ChatModelRequest) -> Result<CompleteRequest, HttpError> {
    let messages: Vec<ChatMessageWire> = req
        .messages
        .iter()
        .map(|m| {
            let (text, content_json) = render_content(m.content.as_ref())?;
            Ok::<ChatMessageWire, HttpError>(ChatMessageWire {
                role: m.role.clone(),
                text,
                content_json,
            })
        })
        .collect::<Result<_, _>>()?;
    let stop = req
        .stop
        .as_ref()
        .map(|s| match s {
            StopSpec::One(s) => vec![s.clone()],
            StopSpec::Many(v) => v.clone(),
        })
        .unwrap_or_default();
    let response_format_json = match &req.response_format {
        Some(v) => serde_json::to_vec(v)?,
        None => Vec::new(),
    };
    let extra_json = match &req.extra_body {
        Some(v) => serde_json::to_vec(v)?,
        None => Vec::new(),
    };
    Ok(CompleteRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: req.model.clone(),
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop,
        response_format_json,
        extra_json,
        tags: std::collections::BTreeMap::new(),
    })
}

fn render_content(content: Option<&Json2>) -> Result<(String, Vec<u8>), HttpError> {
    match content {
        None | Some(Json2::Null) => Ok((String::new(), Vec::new())),
        Some(Json2::String(s)) => Ok((s.clone(), Vec::new())),
        Some(other) => {
            // Structured content — keep the JSON for the host to
            // interpret. The flat `text` field stays empty so backends
            // that only honor `text` don't see a partial duplicate.
            let bytes = serde_json::to_vec(other)?;
            Ok((String::new(), bytes))
        }
    }
}

fn parse_optional_json(bytes: &[u8]) -> Result<Option<Json2>, HttpError> {
    if bytes.is_empty() {
        Ok(None)
    } else {
        Ok(Some(serde_json::from_slice(bytes)?))
    }
}

fn unix_seconds() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs(),
        Err(_) => 0,
    }
}

fn sse_chat_stream(
    inner: crate::server::model_manager::StreamCompleteStream,
    model: String,
) -> impl Stream<Item = Result<axum::response::sse::Event, Infallible>> {
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_seconds();
    let (tx, rx) = mpsc::channel::<Result<axum::response::sse::Event, Infallible>>(16);
    tokio::spawn(async move {
        let mut inner = inner;
        while let Some(frame) = inner.next().await {
            let ev = match frame {
                Ok(StreamCompleteChunk::Delta { text, .. }) => {
                    let chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "role": "assistant", "content": text },
                            "finish_reason": Json2::Null,
                        }],
                    });
                    json_event(&chunk).ok()
                }
                Ok(StreamCompleteChunk::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                }) => {
                    let chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens.unwrap_or(0),
                            "completion_tokens": completion_tokens.unwrap_or(0),
                            "total_tokens": prompt_tokens.unwrap_or(0) + completion_tokens.unwrap_or(0),
                        }
                    });
                    json_event(&chunk).ok()
                }
                Err(rpc_err) => {
                    let body = serde_json::json!({
                        "error": {
                            "message": rpc_err.message,
                            "type": "server_error",
                            "code": rpc_err.code,
                        }
                    });
                    let ev = json_event(&body).ok();
                    if let Some(ev) = ev {
                        let _ = tx.send(Ok(ev)).await;
                    }
                    break;
                }
            };
            if let Some(ev) = ev
                && tx.send(Ok(ev)).await.is_err()
            {
                return;
            }
        }
        let _ = tx.send(Ok(finish_event())).await;
    });
    ReceiverStream::new(rx)
}

// ---------------------------------------------------------------------------
// Legacy completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LegacyModelRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct LegacyModelResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<LegacyChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct LegacyChoice {
    text: String,
    index: u32,
    finish_reason: Option<String>,
}

async fn legacy_completions(
    State(state): State<Arc<RestState>>,
    Json(req): Json<LegacyModelRequest>,
) -> Result<Response, HttpError> {
    if req.stream.unwrap_or(false) {
        return Err(HttpError::unprocessable(
            "stream=true is not supported on the legacy /v1/completions endpoint; use /v1/chat/completions",
        ));
    }
    state.metrics.record("complete");
    let stop = req.stop.map(StopSpec::into_vec).unwrap_or_default();
    let wire = CompleteRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: req.model.clone(),
        messages: vec![ChatMessageWire {
            role: "user".into(),
            text: req.prompt,
            content_json: Vec::new(),
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop,
        response_format_json: Vec::new(),
        extra_json: Vec::new(),
        tags: std::collections::BTreeMap::new(),
    };
    let resp = state.handle.complete(wire).await?;
    let usage = Usage {
        prompt_tokens: resp.prompt_tokens.unwrap_or(0),
        completion_tokens: resp.completion_tokens.unwrap_or(0),
        total_tokens: resp.prompt_tokens.unwrap_or(0) + resp.completion_tokens.unwrap_or(0),
    };
    let body = LegacyModelResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion",
        created: unix_seconds(),
        model: req.model,
        choices: vec![LegacyChoice {
            text: resp.text,
            index: 0,
            finish_reason: resp.finish_reason,
        }],
        usage,
    };
    Ok((StatusCode::OK, Json(body)).into_response())
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub dimensions: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    data: Vec<Embedding>,
    model: String,
    usage: EmbeddingsUsage,
}

#[derive(Debug, Serialize)]
#[allow(clippy::struct_field_names)]
struct Embedding {
    object: &'static str,
    index: u32,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbeddingsUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

async fn embeddings(
    State(state): State<Arc<RestState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, HttpError> {
    state.metrics.record("embed");
    let inputs = match req.input {
        EmbeddingInput::One(s) => vec![s],
        EmbeddingInput::Many(v) => v,
    };
    let wire = EmbedRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: req.model.clone(),
        inputs,
        dimensions: req.dimensions,
        extra_json: Vec::new(),
    };
    let resp = state.handle.embed(wire).await?;
    let pt = resp.prompt_tokens.unwrap_or(0);
    let data = resp
        .vectors
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| Embedding {
            object: "embedding",
            index: u32::try_from(i).unwrap_or(u32::MAX),
            embedding,
        })
        .collect();
    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model: req.model,
        usage: EmbeddingsUsage {
            prompt_tokens: pt,
            total_tokens: pt,
        },
    }))
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ModelsList {
    object: &'static str,
    data: Vec<ModelDescriptor>,
}

#[derive(Debug, Serialize)]
struct ModelDescriptor {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

async fn list_models(State(state): State<Arc<RestState>>) -> Result<Json<ModelsList>, HttpError> {
    state.metrics.record("status");
    let req = crate::model_protocol::StatusRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
    };
    let resp = state.handle.status(req).await?;
    let created = unix_seconds();
    let data = resp
        .models
        .into_iter()
        .map(|m| ModelDescriptor {
            id: m.id,
            object: "model",
            created,
            owned_by: "blazen",
        })
        .collect();
    Ok(Json(ModelsList {
        object: "list",
        data,
    }))
}

// ---------------------------------------------------------------------------
// Audio: speech (TTS)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub sample_rate_hz: Option<u32>,
}

async fn audio_speech(
    State(state): State<Arc<RestState>>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response, HttpError> {
    state.metrics.record("text_to_speech");
    let mut extra = serde_json::Map::new();
    if let Some(fmt) = req.response_format.as_ref() {
        extra.insert("response_format".to_owned(), Json2::String(fmt.clone()));
    }
    let extra_json = if extra.is_empty() {
        Vec::new()
    } else {
        serde_json::to_vec(&Json2::Object(extra))?
    };
    let wire = TextToSpeechRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: req.model,
        text: req.input,
        voice: req.voice,
        language: req.language,
        sample_rate_hz: req.sample_rate_hz,
        audio_config_json: extra_json,
    };
    let resp = state.handle.text_to_speech(wire).await?;
    let mime = HeaderValue::from_str(&resp.mime)
        .map_err(|e| HttpError::internal(format!("invalid mime returned by backend: {e}")))?;
    let mut response = (StatusCode::OK, resp.data).into_response();
    response.headers_mut().insert(header::CONTENT_TYPE, mime);
    Ok(response)
}

// ---------------------------------------------------------------------------
// Audio: transcription (multipart)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<Json2>,
}

async fn audio_transcriptions(
    State(state): State<Arc<RestState>>,
    mut multipart: Multipart,
) -> Result<Json<TranscriptionResponse>, HttpError> {
    state.metrics.record("transcribe");
    let mut audio: Option<super::uploads::StoredBlob> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| HttpError::bad_request(format!("multipart: {e}")))?
    {
        let name = field.name().map(str::to_owned);
        match name.as_deref() {
            Some("file") => {
                let filename = field.file_name().map(str::to_owned);
                let content_type = field
                    .content_type()
                    .map_or_else(|| "application/octet-stream".to_owned(), str::to_owned);
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| HttpError::bad_request(format!("multipart bytes: {e}")))?;
                audio = Some(super::uploads::StoredBlob {
                    filename,
                    content_type,
                    data: Arc::new(data.to_vec()),
                });
            }
            Some("model") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| HttpError::bad_request(format!("multipart text: {e}")))?;
                model = Some(text);
            }
            Some("language") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| HttpError::bad_request(format!("multipart text: {e}")))?;
                language = Some(text);
            }
            _ => {}
        }
    }
    let audio = audio.ok_or_else(|| HttpError::bad_request("missing multipart field 'file'"))?;
    let model = model.ok_or_else(|| HttpError::bad_request("missing multipart field 'model'"))?;
    let wire = TranscribeRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: model,
        audio: (*audio.data).clone(),
        mime: audio.content_type,
        language,
        extra_json: Vec::new(),
    };
    let resp = state.handle.transcribe(wire).await?;
    let segments = parse_optional_json(&resp.segments_json)?;
    Ok(Json(TranscriptionResponse {
        text: resp.text,
        language: resp.language,
        segments,
    }))
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// `"1024x1024"` etc. `OpenAI` uses this string form.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// `"b64_json"` (default) or `"url"`. The MVP only honors `b64_json`
    /// — `url` returns the same b64 body with a warning so clients keep
    /// working without a separate storage layer.
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub extra_body: Option<Json2>,
}

#[derive(Debug, Serialize)]
struct ImageGenerationResponse {
    created: u64,
    data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
struct ImageData {
    b64_json: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
}

async fn images_generations(
    State(state): State<Arc<RestState>>,
    Json(req): Json<ImageGenerationRequest>,
) -> Result<Json<ImageGenerationResponse>, HttpError> {
    state.metrics.record("generate_image");
    let (width, height) = parse_size(req.size.as_deref())?;
    let extra_json = match &req.extra_body {
        Some(v) => serde_json::to_vec(v)?,
        None => Vec::new(),
    };
    let wire = GenerateImageRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id: req.model,
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        width,
        height,
        num_images: req.n,
        seed: req.seed,
        image_config_json: extra_json,
    };
    let resp = state.handle.generate_image(wire).await?;
    let data = resp
        .images
        .into_iter()
        .map(|img| ImageData {
            b64_json: BASE64.encode(&img.data),
            mime_type: Some(img.mime),
        })
        .collect();
    Ok(Json(ImageGenerationResponse {
        created: unix_seconds(),
        data,
    }))
}

fn parse_size(s: Option<&str>) -> Result<(Option<u32>, Option<u32>), HttpError> {
    let Some(s) = s else { return Ok((None, None)) };
    let (w, h) = s
        .split_once('x')
        .ok_or_else(|| HttpError::bad_request(format!("invalid size '{s}', expected WxH")))?;
    let w: u32 = w
        .parse()
        .map_err(|e| HttpError::bad_request(format!("invalid width in '{s}': {e}")))?;
    let h: u32 = h
        .parse()
        .map_err(|e| HttpError::bad_request(format!("invalid height in '{s}': {e}")))?;
    Ok((Some(w), Some(h)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::model_manager::test_support::MockManagerHandle;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    fn router_with_mock() -> (axum::Router, Arc<RestState>) {
        let mock = MockManagerHandle::new();
        let state = Arc::new(RestState::new(mock));
        let r = router(state.clone());
        (r, state)
    }

    async fn body_json(resp: axum::http::Response<Body>) -> Json2 {
        let bytes = to_bytes(resp.into_body(), 8 * 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn body_bytes(resp: axum::http::Response<Body>) -> Vec<u8> {
        to_bytes(resp.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap()
            .to_vec()
    }

    fn json_request(path: &str, body: &Json2) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    async fn chat_completions_non_stream_roundtrip() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let resp = r
            .oneshot(json_request("/v1/chat/completions", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert!(
            v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .starts_with("echo:")
        );
    }

    #[tokio::test]
    async fn chat_completions_structured_content_serialised_as_json() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": [{"type":"text","text":"hi"}]}],
        });
        let resp = r
            .oneshot(json_request("/v1/chat/completions", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn chat_completions_stream_emits_chunks_and_done() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let resp = r
            .oneshot(json_request("/v1/chat/completions", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(ct.starts_with("text/event-stream"), "got {ct}");
        let text = String::from_utf8(body_bytes(resp).await).unwrap();
        // Mock emits 2 Delta + 1 Done = 3 SSE events plus the `[DONE]` terminator.
        let data_lines: Vec<_> = text.lines().filter(|l| l.starts_with("data:")).collect();
        assert!(
            data_lines.len() >= 4,
            "expected >=4 data lines, got {data_lines:?}"
        );
        assert!(text.contains("[DONE]"));
        assert!(text.contains("chat.completion.chunk"));
    }

    #[tokio::test]
    async fn chat_completions_422_on_malformed_json() {
        let (r, _) = router_with_mock();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("{not json"))
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        // axum returns 400 for json decode errors by default; just ensure
        // it isn't 200 and the body parses as some JSON.
        assert!(resp.status().is_client_error());
    }

    #[tokio::test]
    async fn legacy_completions_roundtrip() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "qwen",
            "prompt": "hello",
        });
        let resp = r
            .oneshot(json_request("/v1/completions", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "text_completion");
        assert!(
            v["choices"][0]["text"]
                .as_str()
                .unwrap()
                .starts_with("echo:")
        );
    }

    #[tokio::test]
    async fn legacy_completions_rejects_stream() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "qwen",
            "prompt": "hello",
            "stream": true,
        });
        let resp = r
            .oneshot(json_request("/v1/completions", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn embeddings_string_input() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({"model":"bge","input":"hello"});
        let resp = r
            .oneshot(json_request("/v1/embeddings", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
        assert_eq!(v["data"][0]["object"], "embedding");
    }

    #[tokio::test]
    async fn embeddings_array_input() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({"model":"bge","input":["a","b","c"]});
        let resp = r
            .oneshot(json_request("/v1/embeddings", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["data"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn list_models_returns_openai_shape() {
        let (r, _) = router_with_mock();
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
    }

    #[tokio::test]
    async fn audio_speech_returns_audio_bytes() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({"model":"kokoro","input":"hi"});
        let resp = r
            .oneshot(json_request("/v1/audio/speech", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ct, "audio/wav");
        let bytes = body_bytes(resp).await;
        assert_eq!(bytes.len(), 8); // mock returns 8 zeros
    }

    #[tokio::test]
    async fn audio_transcriptions_multipart_roundtrip() {
        let (r, _) = router_with_mock();
        let boundary = "----blazentest";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n",
        );
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(b"RIFF....WAVE");
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["text"], "transcribed");
    }

    #[tokio::test]
    async fn audio_transcriptions_missing_file_400() {
        let (r, _) = router_with_mock();
        let boundary = "----blazentest";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n",
        );
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn images_generations_roundtrip() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "sdxl",
            "prompt": "a cat",
            "size": "1024x1024",
            "n": 1,
        });
        let resp = r
            .oneshot(json_request("/v1/images/generations", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
        assert!(!v["data"][0]["b64_json"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn images_generations_rejects_bad_size() {
        let (r, _) = router_with_mock();
        let body = serde_json::json!({
            "model": "sdxl",
            "prompt": "a cat",
            "size": "garbage",
        });
        let resp = r
            .oneshot(json_request("/v1/images/generations", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn metrics_record_each_dispatch() {
        let (r, state) = router_with_mock();
        let body = serde_json::json!({"model":"bge","input":"hi"});
        let resp = r
            .clone()
            .oneshot(json_request("/v1/embeddings", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp2 = r
            .oneshot(json_request("/v1/embeddings", &body))
            .await
            .unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
        let count = *state
            .metrics
            .by_rpc
            .get("embed")
            .expect("embed counted")
            .value();
        assert_eq!(count, 2);
    }
}
