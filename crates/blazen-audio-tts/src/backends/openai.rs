//! HTTP client for `OpenAI`-compatible text-to-speech servers.
//!
//! Targets:
//! - `POST /v1/audio/speech` — synthesize speech from text. Standard `OpenAI`
//!   endpoint; also accepted by every `OpenAI`-compat TTS service (Groq's
//!   `PlayHT` bridge, Together, in-house `OpenAI`-compat TTS deployments,
//!   and a long tail of self-hosted `VoxCPM` / XTTS / Piper bridges).
//! - `GET    /v1/voices` — list voices. Not in the official spec but a
//!   de-facto extension implemented by basically every compat server.
//! - `POST   /v1/voices/clone` — multipart upload of a reference clip,
//!   returns a voice handle usable as the `voice` field on a synthesis
//!   request.
//! - `POST   /v1/voices/design` — text-only voice designer.
//! - `DELETE /v1/voices/{id}` — remove a previously-created voice.
//!
//! The backend is a pure HTTP client — no native dependencies and no
//! engine state. It is constructed from a base URL + optional bearer
//! token + model + default voice and is `Send + Sync`.

use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use blazen_audio::{
    AudioBackend, AudioFormat, CloneVoiceRequest, DesignVoiceRequest, GeneratedAudio,
    ListVoicesRequest, ListVoicesResponse as SharedListVoicesResponse, VoiceDto as SharedVoiceDto,
    VoiceHandle, VoiceKind,
};
use reqwest::{
    Client, ClientBuilder, StatusCode,
    multipart::{Form, Part},
};
use serde::de::DeserializeOwned;

use crate::backends::openai_types::{
    CloneVoiceResponse, DesignVoiceRequestDto, DesignVoiceResponse, ErrorEnvelope,
    ListVoicesResponse, SpeechRequestDto, VoiceDto,
};
use crate::traits::TtsBackend;
use crate::{TtsError, TtsOptions};

/// Default model identifier used when the caller does not supply one.
///
/// Matches `OpenAI`'s lowest-tier TTS model name. Compat servers either
/// accept this verbatim or route any value to their single available
/// model — both behaviors are fine for our purposes.
pub const DEFAULT_MODEL: &str = "tts-1";

/// Default voice identifier used when the caller does not supply one.
///
/// `alloy` is one of `OpenAI`'s preset voices and is also the most common
/// fallback name in compat servers.
pub const DEFAULT_VOICE: &str = "alloy";

/// Default response format. `mp3` is the smallest and most widely
/// supported across compat servers.
pub const DEFAULT_RESPONSE_FORMAT: &str = "mp3";

/// Default request timeout when the caller does not customize the
/// underlying [`reqwest::Client`].
const DEFAULT_TIMEOUT: Duration = Duration::from_mins(1);

// ---------------------------------------------------------------------------
// Public request / response types
// ---------------------------------------------------------------------------

/// Inputs for [`OpenAiTtsBackend::text_to_speech`].
///
/// This is the OpenAI-flavored shape — keeps the field names matched to
/// the wire request. The [`TtsBackend::synthesize`] entry point accepts
/// the capability-agnostic [`TtsOptions`] instead.
#[derive(Debug, Clone, Default)]
pub struct OpenAiTtsSpeechRequest {
    /// Text to synthesize. Required.
    pub text: String,
    /// Voice identifier override. `None` => backend default voice.
    pub voice: Option<String>,
    /// Model identifier override. `None` => backend default model.
    pub model: Option<String>,
    /// Speech speed multiplier (`1.0` = normal). `None` omits the field.
    pub speed: Option<f32>,
    /// Output container. `None` => `"mp3"`.
    pub response_format: Option<String>,
    /// Whether to request streamed audio. Currently informational only
    /// — the buffered HTTP path is used regardless.
    pub stream: Option<bool>,
}

/// Output of [`OpenAiTtsBackend::text_to_speech`].
#[derive(Debug, Clone)]
pub struct OpenAiTtsSpeechResponse {
    /// Raw audio bytes (container determined by `response_format`).
    pub audio_bytes: Vec<u8>,
    /// MIME type reported by the server. Falls back to `audio/mpeg` if
    /// the server did not set `Content-Type`.
    pub content_type: String,
}

impl OpenAiTtsSpeechResponse {
    /// Base64-encode the audio bytes (standard alphabet, with padding).
    #[must_use]
    pub fn audio_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.audio_bytes)
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Configuration for [`OpenAiTtsBackend`].
#[derive(Debug, Clone)]
pub struct OpenAiTtsConfig {
    /// Base URL ending at the `v1` segment, e.g. `https://api.openai.com/v1`
    /// or `http://localhost:8900/v1`. Trailing slashes are tolerated.
    pub base_url: String,
    /// Bearer token. Empty string => no `Authorization` header sent
    /// (useful for unauth'd local deployments).
    pub api_key: String,
    /// Default model. Falls back to [`DEFAULT_MODEL`] if empty.
    pub model: String,
    /// Default voice. Falls back to [`DEFAULT_VOICE`] if empty.
    pub default_voice: String,
}

impl OpenAiTtsConfig {
    /// Build a config with all defaults except `base_url` and `api_key`.
    #[must_use]
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            default_voice: DEFAULT_VOICE.to_owned(),
        }
    }

    /// Override the default model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the default voice.
    #[must_use]
    pub fn with_default_voice(mut self, voice: impl Into<String>) -> Self {
        self.default_voice = voice.into();
        self
    }
}

/// HTTP backend for OpenAI-compatible TTS servers.
#[derive(Debug, Clone)]
pub struct OpenAiTtsBackend {
    base_url: String,
    api_key: String,
    model: String,
    default_voice: String,
    id: String,
    client: Client,
}

impl OpenAiTtsBackend {
    /// Build a backend with the default [`reqwest::Client`].
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Http`] if the underlying HTTP client could
    /// not be constructed (e.g. native TLS init failure).
    pub fn new(config: OpenAiTtsConfig) -> Result<Self, TtsError> {
        let client = ClientBuilder::new()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|e| TtsError::Http(e.to_string()))?;
        Self::with_client(config, client)
    }

    /// Build a backend using a caller-supplied [`reqwest::Client`].
    ///
    /// Useful when sharing a connection pool across backends or when
    /// the caller has already configured TLS, proxies, or timeouts.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::InvalidOptions`] if `base_url` is empty.
    pub fn with_client(config: OpenAiTtsConfig, client: Client) -> Result<Self, TtsError> {
        if config.base_url.trim().is_empty() {
            return Err(TtsError::InvalidOptions("base_url is empty".to_owned()));
        }
        let model = if config.model.is_empty() {
            DEFAULT_MODEL.to_owned()
        } else {
            config.model
        };
        let default_voice = if config.default_voice.is_empty() {
            DEFAULT_VOICE.to_owned()
        } else {
            config.default_voice
        };
        let id = format!("openai:{model}");
        Ok(Self {
            base_url: config.base_url.trim_end_matches('/').to_owned(),
            api_key: config.api_key,
            model,
            default_voice,
            id,
            client,
        })
    }

    /// Backend identifier — `"openai:<model>"`.
    #[must_use]
    pub fn provider_id(&self) -> &str {
        &self.id
    }

    /// Base URL the backend targets.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Default model identifier.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Default voice identifier.
    #[must_use]
    pub fn default_voice(&self) -> &str {
        &self.default_voice
    }

    // -----------------------------------------------------------------
    // Endpoints — inherent methods (OpenAI-shaped surface)
    // -----------------------------------------------------------------

    /// `POST /v1/audio/speech` — synthesize speech and return raw audio
    /// bytes plus the server-reported content type.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn text_to_speech(
        &self,
        request: OpenAiTtsSpeechRequest,
    ) -> Result<OpenAiTtsSpeechResponse, TtsError> {
        if request.text.is_empty() {
            return Err(TtsError::InvalidOptions(
                "OpenAiTtsSpeechRequest.text is empty".to_owned(),
            ));
        }

        let url = format!("{}/audio/speech", self.base_url);
        let model = request.model.as_deref().unwrap_or(&self.model);
        let voice = request.voice.as_deref().unwrap_or(&self.default_voice);
        let response_format = request
            .response_format
            .as_deref()
            .unwrap_or(DEFAULT_RESPONSE_FORMAT);

        let body = SpeechRequestDto {
            model,
            input: &request.text,
            voice,
            response_format,
            speed: request.speed,
            stream: request.stream,
        };

        let mut req = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let response = req.send().await.map_err(|e| reqwest_to_tts(&e))?;
        let status = response.status();
        let headers = response.headers().clone();

        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(map_error(status, &headers, &body, &url));
        }

        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("audio/mpeg")
            .to_owned();

        let audio_bytes = response
            .bytes()
            .await
            .map_err(|e| reqwest_to_tts(&e))?
            .to_vec();
        Ok(OpenAiTtsSpeechResponse {
            audio_bytes,
            content_type,
        })
    }

    /// `GET /v1/voices` — list all voices known to the server.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn list_voices_raw(&self) -> Result<Vec<VoiceDto>, TtsError> {
        let url = format!("{}/voices", self.base_url);
        let mut req = self.client.get(&url);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req.send().await.map_err(|e| reqwest_to_tts(&e))?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(map_error(status, &headers, &body, &url));
        }
        let parsed: ListVoicesResponse = decode_json(response).await?;
        Ok(parsed.voices)
    }

    /// `POST /v1/voices/clone` — upload a reference audio clip and a
    /// human-readable name, optionally with the spoken transcript. The
    /// returned [`VoiceDto`] carries the new voice id.
    ///
    /// `audio_bytes` is sent as a multipart `audio` file part with
    /// `Content-Type: application/octet-stream`. Servers that need a
    /// specific MIME (e.g. `audio/wav`) typically sniff the bytes
    /// themselves; passing octet-stream maximizes compatibility.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn clone_voice_raw(
        &self,
        audio_bytes: Vec<u8>,
        name: &str,
        transcript: Option<&str>,
    ) -> Result<VoiceDto, TtsError> {
        if name.trim().is_empty() {
            return Err(TtsError::InvalidOptions(
                "clone_voice: name is empty".to_owned(),
            ));
        }
        if audio_bytes.is_empty() {
            return Err(TtsError::InvalidOptions(
                "clone_voice: audio_bytes is empty".to_owned(),
            ));
        }

        let url = format!("{}/voices/clone", self.base_url);
        let audio_part = Part::bytes(audio_bytes)
            .file_name("reference.wav")
            .mime_str("application/octet-stream")
            .map_err(|e| reqwest_to_tts(&e))?;
        let mut form = Form::new()
            .text("name", name.to_owned())
            .part("audio", audio_part);
        if let Some(t) = transcript {
            form = form.text("transcript", t.to_owned());
        }

        let mut req = self.client.post(&url).multipart(form);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req.send().await.map_err(|e| reqwest_to_tts(&e))?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(map_error(status, &headers, &body, &url));
        }
        let parsed: CloneVoiceResponse = decode_json(response).await?;
        Ok(parsed.into_voice())
    }

    /// `POST /v1/voices/design` — create a new voice from a natural-
    /// language description (no reference audio needed). Returns the
    /// new voice handle.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn design_voice_raw(
        &self,
        name: &str,
        description: &str,
    ) -> Result<VoiceDto, TtsError> {
        if name.trim().is_empty() {
            return Err(TtsError::InvalidOptions(
                "design_voice: name is empty".to_owned(),
            ));
        }
        if description.trim().is_empty() {
            return Err(TtsError::InvalidOptions(
                "design_voice: description is empty".to_owned(),
            ));
        }

        let url = format!("{}/voices/design", self.base_url);
        let body = DesignVoiceRequestDto { name, description };
        let mut req = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req.send().await.map_err(|e| reqwest_to_tts(&e))?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(map_error(status, &headers, &body, &url));
        }
        let parsed: DesignVoiceResponse = decode_json(response).await?;
        Ok(parsed.into_voice())
    }

    /// `DELETE /v1/voices/{id}` — remove a previously-created voice.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn delete_voice(&self, voice_id: &str) -> Result<(), TtsError> {
        if voice_id.trim().is_empty() {
            return Err(TtsError::InvalidOptions(
                "delete_voice: voice_id is empty".to_owned(),
            ));
        }
        let url = format!("{}/voices/{voice_id}", self.base_url);
        let mut req = self.client.delete(&url);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req.send().await.map_err(|e| reqwest_to_tts(&e))?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            let body = response.bytes().await.unwrap_or_default();
            return Err(map_error(status, &headers, &body, &url));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

fn audio_format_for(response_format: Option<&str>, content_type: &str) -> AudioFormat {
    let lc = response_format.map_or_else(
        || content_type.to_ascii_lowercase(),
        str::to_ascii_lowercase,
    );
    if lc.contains("wav") {
        AudioFormat::Wav
    } else if lc.contains("flac") {
        AudioFormat::Flac
    } else if lc.contains("opus") || lc.contains("ogg") {
        AudioFormat::Opus
    } else if lc.contains("pcm") {
        AudioFormat::Pcm
    } else {
        AudioFormat::Mp3
    }
}

fn voice_dto_to_shared(dto: VoiceDto) -> SharedVoiceDto {
    SharedVoiceDto {
        id: dto.id,
        name: dto.name,
        language: dto.language,
        kind: VoiceKind::Preset,
    }
}

fn voice_dto_to_handle(dto: VoiceDto) -> VoiceHandle {
    VoiceHandle {
        id: dto.id,
        provider: "openai".to_owned(),
    }
}

#[async_trait]
impl AudioBackend for OpenAiTtsBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "tts"
    }
    // Default load/unload/is_loaded are fine — this is a stateless HTTP client.
}

#[async_trait]
impl TtsBackend for OpenAiTtsBackend {
    async fn synthesize(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        let response_format = options.response_format.clone();
        let speech_request = OpenAiTtsSpeechRequest {
            text: text.to_owned(),
            voice: options.voice.clone(),
            model: options.model_id.clone(),
            speed: options.speed,
            response_format: response_format.clone(),
            stream: None,
        };
        let response = self.text_to_speech(speech_request).await?;
        let format = audio_format_for(response_format.as_deref(), &response.content_type);
        Ok(GeneratedAudio {
            bytes: response.audio_bytes,
            format,
            sample_rate: options.sample_rate.unwrap_or(24_000),
            channels: 1,
            duration_seconds: None,
        })
    }

    async fn list_voices(
        &self,
        _request: &ListVoicesRequest,
    ) -> Result<SharedListVoicesResponse, TtsError> {
        let voices = self.list_voices_raw().await?;
        Ok(SharedListVoicesResponse {
            voices: voices.into_iter().map(voice_dto_to_shared).collect(),
        })
    }

    async fn clone_voice(&self, request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        let dto = self
            .clone_voice_raw(
                request.audio_bytes,
                &request.name,
                request.transcript.as_deref(),
            )
            .await?;
        Ok(voice_dto_to_handle(dto))
    }

    async fn design_voice(&self, request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        let dto = self
            .design_voice_raw(&request.name, &request.description)
            .await?;
        Ok(voice_dto_to_handle(dto))
    }

    async fn delete_voice(&self, voice_id: &str) -> Result<(), TtsError> {
        OpenAiTtsBackend::delete_voice(self, voice_id).await
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn reqwest_to_tts(err: &reqwest::Error) -> TtsError {
    TtsError::Http(err.to_string())
}

/// Map an error HTTP response onto [`TtsError`].
///
/// Tries to extract a message from the OpenAI-style
/// `{ "error": { "message": ... } }` envelope; falls back to the raw
/// body as UTF-8.
fn map_error(
    status: StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &[u8],
    url: &str,
) -> TtsError {
    let message = parse_error_message(body);
    match status.as_u16() {
        401 => TtsError::Auth(message),
        429 => TtsError::RateLimit {
            retry_after_seconds: parse_retry_after(headers),
            message,
        },
        code => TtsError::ServerError {
            status: code,
            url: url.to_owned(),
            message,
        },
    }
}

/// Extract a human-readable error message from a response body.
fn parse_error_message(body: &[u8]) -> String {
    if let Ok(env) = serde_json::from_slice::<ErrorEnvelope>(body) {
        return env.error.message;
    }
    String::from_utf8_lossy(body).into_owned()
}

/// Parse a `Retry-After: <seconds>` header. Returns `None` for missing,
/// non-numeric (HTTP-date), or unparseable values.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Decode a successful response body as JSON, surfacing decode errors
/// as [`TtsError::Decode`] rather than the raw `reqwest::Error`
/// (which loses the body context).
async fn decode_json<T: DeserializeOwned>(response: reqwest::Response) -> Result<T, TtsError> {
    let bytes = response.bytes().await.map_err(|e| reqwest_to_tts(&e))?;
    serde_json::from_slice::<T>(&bytes)
        .map_err(|e| TtsError::Decode(format!("{e}: body was {}", String::from_utf8_lossy(&bytes))))
}

// ---------------------------------------------------------------------------
// Tests — unit coverage for the helpers and validation paths. Wire
// coverage against a mock server lives in `tests/openai_smoke.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> OpenAiTtsBackend {
        OpenAiTtsBackend::new(OpenAiTtsConfig::new("http://localhost:9999/v1", "sk-test"))
            .expect("backend should build")
    }

    #[test]
    fn config_with_overrides() {
        let cfg = OpenAiTtsConfig::new("https://api.openai.com/v1", "sk")
            .with_model("tts-1-hd")
            .with_default_voice("nova");
        assert_eq!(cfg.model, "tts-1-hd");
        assert_eq!(cfg.default_voice, "nova");
    }

    #[test]
    fn new_rejects_empty_base_url() {
        let cfg = OpenAiTtsConfig::new("", "sk");
        let err = OpenAiTtsBackend::new(cfg).unwrap_err();
        assert!(matches!(err, TtsError::InvalidOptions(_)));
    }

    #[test]
    fn new_trims_trailing_slash_from_base_url() {
        let cfg = OpenAiTtsConfig::new("https://api.openai.com/v1/", "sk");
        let p = OpenAiTtsBackend::new(cfg).unwrap();
        assert_eq!(p.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn new_substitutes_defaults_for_empty_model_and_voice() {
        let cfg = OpenAiTtsConfig {
            base_url: "http://x/v1".to_owned(),
            api_key: String::new(),
            model: String::new(),
            default_voice: String::new(),
        };
        let p = OpenAiTtsBackend::new(cfg).unwrap();
        assert_eq!(p.model(), DEFAULT_MODEL);
        assert_eq!(p.default_voice(), DEFAULT_VOICE);
    }

    #[tokio::test]
    async fn text_to_speech_rejects_empty_text() {
        let p = backend();
        let err = p
            .text_to_speech(OpenAiTtsSpeechRequest::default())
            .await
            .unwrap_err();
        assert!(matches!(err, TtsError::InvalidOptions(_)));
    }

    #[tokio::test]
    async fn clone_voice_rejects_empty_name() {
        let p = backend();
        let err = p
            .clone_voice_raw(b"audio".to_vec(), "  ", None)
            .await
            .unwrap_err();
        assert!(matches!(err, TtsError::InvalidOptions(_)));
    }

    #[tokio::test]
    async fn clone_voice_rejects_empty_audio() {
        let p = backend();
        let err = p
            .clone_voice_raw(Vec::new(), "MyVoice", None)
            .await
            .unwrap_err();
        assert!(matches!(err, TtsError::InvalidOptions(_)));
    }

    #[tokio::test]
    async fn design_voice_rejects_empty_description() {
        let p = backend();
        let err = p.design_voice_raw("Name", "").await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidOptions(_)));
    }

    #[tokio::test]
    async fn delete_voice_rejects_empty_id() {
        let p = backend();
        let err = p.delete_voice("").await.unwrap_err();
        assert!(matches!(err, TtsError::InvalidOptions(_)));
    }

    #[test]
    fn provider_id_includes_model() {
        assert_eq!(backend().provider_id(), "openai:tts-1");
    }

    #[test]
    fn parse_retry_after_seconds() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, "42".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(42));
    }

    #[test]
    fn parse_retry_after_missing_returns_none() {
        let h = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn parse_retry_after_http_date_returns_none() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn parse_error_message_extracts_openai_envelope() {
        let body = br#"{"error":{"message":"invalid voice","type":"invalid_request_error"}}"#;
        assert_eq!(parse_error_message(body), "invalid voice");
    }

    #[test]
    fn parse_error_message_falls_back_to_raw_body() {
        let body = b"plain text upstream error";
        assert_eq!(parse_error_message(body), "plain text upstream error");
    }

    #[test]
    fn speech_response_base64_round_trips() {
        let r = OpenAiTtsSpeechResponse {
            audio_bytes: b"hello".to_vec(),
            content_type: "audio/mpeg".to_owned(),
        };
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(r.audio_base64())
            .unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn audio_format_for_picks_mp3_by_default() {
        assert_eq!(audio_format_for(None, "application/json"), AudioFormat::Mp3);
        assert_eq!(
            audio_format_for(Some("wav"), "audio/mpeg"),
            AudioFormat::Wav
        );
        assert_eq!(audio_format_for(None, "audio/wav"), AudioFormat::Wav);
        assert_eq!(audio_format_for(None, "audio/ogg"), AudioFormat::Opus);
        assert_eq!(audio_format_for(Some("flac"), ""), AudioFormat::Flac);
    }
}
