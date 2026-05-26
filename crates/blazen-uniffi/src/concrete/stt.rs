//! STT per-engine `#[uniffi::Object]` providers — P4.2.stt.
//!
//! One concrete `<Engine>Provider` per STT backend. Each wraps the
//! upstream `blazen_llm::providers::concrete::stt::<Engine>Provider`
//! handle in an `Arc` so UniFFI's foreign-language bindgens emit a real
//! per-engine class (Kotlin `class WhisperCppProvider`, Swift
//! `class WhisperCppProvider`, Go `*WhisperCppProvider`,
//! Ruby `Blazen::WhisperCppProvider`).
//!
//! The whole module is implicitly gated by the parent
//! `#[cfg(feature = "whispercpp")]` declaration in [`super`], because
//! the upstream `blazen_llm::providers::concrete::stt` module itself
//! lives under that same gate. Individual engines that require an
//! additional feature (`audio-stt-faster-whisper`,
//! `audio-stt-whisper-streaming`) carry their own `#[cfg]` on top.

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use crate::compute::SttResult;
use crate::errors::BlazenError;
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a `TranscriptionRequest` from the flat uniffi-friendly args.
///
/// `audio_source` is interpreted as either a URL (when it parses with a
/// scheme) or a local file path. Local backends (whisper.cpp, faster-
/// whisper, whisper-streaming) read from disk; hosted backends (fal.ai)
/// fetch the URL.
fn build_transcription_request(
    audio_source: String,
    language: Option<String>,
) -> blazen_llm::compute::requests::TranscriptionRequest {
    use blazen_llm::compute::requests::TranscriptionRequest;

    let is_url = audio_source.starts_with("http://")
        || audio_source.starts_with("https://")
        || audio_source.starts_with("data:");
    let mut request = if is_url {
        TranscriptionRequest::new(audio_source)
    } else {
        TranscriptionRequest::from_file(std::path::PathBuf::from(audio_source))
    };
    request.language = language;
    request
}

/// Map a `blazen_llm::compute::results::TranscriptionResult` into the
/// uniffi-surface [`SttResult`] DTO.
fn map_transcription_result(
    result: blazen_llm::compute::results::TranscriptionResult,
) -> SttResult {
    let duration_ms = result.timing.total_ms.map(u64::from).unwrap_or_default();
    SttResult {
        transcript: result.text,
        language: result.language.unwrap_or_default(),
        duration_ms,
    }
}

/// Build a `BlazenError::Provider` with the canonical sentinel shape used
/// across the binding surface.
fn provider_err(kind: &str, provider: &str, err: impl std::fmt::Display) -> BlazenError {
    BlazenError::Provider {
        kind: kind.to_string(),
        message: err.to_string(),
        provider: Some(provider.to_string()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

// ---------------------------------------------------------------------------
// WhisperCppProvider — local whisper.cpp
// ---------------------------------------------------------------------------

/// Local whisper.cpp speech-to-text provider.
///
/// Construct with [`WhisperCppProvider::new`] (sync — option validation
/// only; weight loading is lazy on the first transcribe call). Use
/// [`transcribe`](Self::transcribe) (async) or
/// [`transcribe_blocking`](Self::transcribe_blocking) (sync) afterwards.
#[derive(uniffi::Object)]
pub struct WhisperCppProvider {
    inner: Arc<blazen_llm::providers::concrete::stt::WhisperCppProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl WhisperCppProvider {
    /// Build a `WhisperCppProvider`.
    ///
    /// `model` selects the whisper.cpp variant (`"tiny"`, `"base"`,
    /// `"small"` (default), `"medium"`, `"large-v3"`). `device` picks
    /// the runtime device (`"cpu"`, `"cuda"`, etc.). `language` is an
    /// optional ISO-639-1 default-language hint (overridden per call).
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "WhisperCppInit", ... }`
    /// when option validation fails.
    #[uniffi::constructor]
    pub fn new(
        model: Option<String>,
        device: Option<String>,
        language: Option<String>,
    ) -> Result<Arc<Self>, BlazenError> {
        let inner =
            blazen_llm::providers::concrete::stt::WhisperCppProvider::new(model, device, language)
                .map_err(|e| provider_err("WhisperCppInit", "whispercpp", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Transcribe audio at `audio_source` and return the transcript.
    ///
    /// `audio_source` is a local file path (16-bit PCM mono WAV at
    /// 16 kHz) or an `http(s)://` / `data:` URL. `language` is an
    /// optional per-call ISO-639-1 override; when omitted the
    /// constructor's `language` hint (if any) is used.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "SttTranscribe", ... }`
    /// when the backend fails to decode or transcribe.
    pub async fn transcribe(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "whispercpp", e))?;
        Ok(map_transcription_result(result))
    }
}

#[uniffi::export]
impl WhisperCppProvider {
    /// Synchronous variant of [`transcribe`](Self::transcribe).
    pub fn transcribe_blocking(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.transcribe(audio_source, language).await })
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for WhisperCppProvider {
    fn provider_id(&self) -> String {
        "whisper-cpp".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::SttProvider for WhisperCppProvider {
    fn provider_id(&self) -> String {
        "whisper-cpp".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }

    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "whisper-cpp", e))?;
        Ok(map_transcription_result(result))
    }
}

// ---------------------------------------------------------------------------
// FasterWhisperProvider — local faster-whisper (CTranslate2 / ct2rs)
// ---------------------------------------------------------------------------

/// Local faster-whisper (`CTranslate2`) speech-to-text provider.
///
/// Construct with [`FasterWhisperProvider::new`] (sync — cheap). The
/// ct2rs decoder is materialised lazily on the first transcribe call
/// (Hugging Face download + `CTranslate2` model load).
#[cfg(feature = "audio-stt-faster-whisper")]
#[derive(uniffi::Object)]
pub struct FasterWhisperProvider {
    inner: Arc<blazen_llm::providers::concrete::stt::FasterWhisperProvider>,
}

#[cfg(feature = "audio-stt-faster-whisper")]
#[uniffi::export(async_runtime = "tokio")]
impl FasterWhisperProvider {
    /// Build a `FasterWhisperProvider`.
    ///
    /// `model_id` selects a Hugging Face bundle id (default
    /// `"Systran/faster-whisper-tiny"`). `model_dir` provides a pre-
    /// resolved local `CTranslate2` bundle directory; when supplied the
    /// HF download is skipped. `revision` pins a specific branch / tag /
    /// commit on the HF repo (default `"main"`).
    #[uniffi::constructor]
    pub fn new(
        model_id: Option<String>,
        model_dir: Option<String>,
        revision: Option<String>,
    ) -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::stt::FasterWhisperProvider::new(
            model_id, model_dir, revision,
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Transcribe audio at `audio_source` and return the transcript.
    ///
    /// See [`WhisperCppProvider::transcribe`] for the `audio_source` /
    /// `language` semantics.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "SttTranscribe", ... }`
    /// when the backend fails to load or transcribe.
    pub async fn transcribe(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "faster-whisper", e))?;
        Ok(map_transcription_result(result))
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
#[uniffi::export]
impl FasterWhisperProvider {
    /// Synchronous variant of [`transcribe`](Self::transcribe).
    pub fn transcribe_blocking(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.transcribe(audio_source, language).await })
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FasterWhisperProvider {
    fn provider_id(&self) -> String {
        "faster-whisper".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
#[async_trait::async_trait]
impl crate::concrete::bases::SttProvider for FasterWhisperProvider {
    fn provider_id(&self) -> String {
        "faster-whisper".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }

    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "faster-whisper", e))?;
        Ok(map_transcription_result(result))
    }
}

// ---------------------------------------------------------------------------
// WhisperStreamingProvider — chunked candle Whisper + Silero VAD
// ---------------------------------------------------------------------------

/// Streaming speech-to-text provider (chunked candle Whisper fronted by
/// Silero VAD).
///
/// Note: the underlying backend's blocking `transcribe` entry point
/// returns `Unsupported` — only the streaming surface is functional in
/// the upstream backend. The async / blocking methods below are exposed
/// for API parity but will surface that error to the caller. Wire the
/// streaming entrypoint directly via the backend if you need it.
#[cfg(feature = "audio-stt-whisper-streaming")]
#[derive(uniffi::Object)]
pub struct WhisperStreamingProvider {
    inner: Arc<blazen_llm::providers::concrete::stt::WhisperStreamingProvider>,
}

#[cfg(feature = "audio-stt-whisper-streaming")]
#[uniffi::export(async_runtime = "tokio")]
impl WhisperStreamingProvider {
    /// Build a `WhisperStreamingProvider`.
    ///
    /// `model_id` selects the underlying candle Whisper HF model
    /// (default `"openai/whisper-base"`). `vad_model_path` overrides
    /// the Silero VAD ONNX location (default: download from HF on
    /// first use). `chunk_seconds` / `chunk_overlap_seconds` tune the
    /// sliding-window geometry; pass `None` for the defaults (`30.0` /
    /// `5.0`).
    #[uniffi::constructor]
    pub fn new(
        model_id: Option<String>,
        vad_model_path: Option<String>,
        chunk_seconds: Option<f32>,
        chunk_overlap_seconds: Option<f32>,
    ) -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::stt::WhisperStreamingProvider::new(
            model_id,
            vad_model_path,
            chunk_seconds,
            chunk_overlap_seconds,
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Transcribe audio at `audio_source` and return the transcript.
    ///
    /// See [`WhisperCppProvider::transcribe`] for `audio_source` /
    /// `language` semantics.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "SttTranscribe", ... }`
    /// — typically `Unsupported` since the streaming backend's blocking
    /// path is intentionally not wired.
    pub async fn transcribe(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "whisper-streaming", e))?;
        Ok(map_transcription_result(result))
    }
}

#[cfg(feature = "audio-stt-whisper-streaming")]
#[uniffi::export]
impl WhisperStreamingProvider {
    /// Synchronous variant of [`transcribe`](Self::transcribe).
    pub fn transcribe_blocking(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.transcribe(audio_source, language).await })
    }
}

#[cfg(feature = "audio-stt-whisper-streaming")]
#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for WhisperStreamingProvider {
    fn provider_id(&self) -> String {
        "whisper-streaming".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }
}

#[cfg(feature = "audio-stt-whisper-streaming")]
#[async_trait::async_trait]
impl crate::concrete::bases::SttProvider for WhisperStreamingProvider {
    fn provider_id(&self) -> String {
        "whisper-streaming".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }

    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "whisper-streaming", e))?;
        Ok(map_transcription_result(result))
    }
}

// ---------------------------------------------------------------------------
// FalSttProvider — hosted fal.ai Whisper / Wizper endpoints
// ---------------------------------------------------------------------------

/// Hosted fal.ai speech-to-text provider.
///
/// Thin wrapper around `blazen_llm::providers::concrete::stt::FalSttProvider`
/// that surfaces only the [`SttProvider`] capability. Construction is
/// cheap; the underlying fal client handles endpoint selection,
/// queueing, and result polling per call.
#[derive(uniffi::Object)]
pub struct FalSttProvider {
    inner: Arc<blazen_llm::providers::concrete::stt::FalSttProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalSttProvider {
    /// Build a `FalSttProvider` with the given fal.ai API key.
    #[uniffi::constructor]
    pub fn new(api_key: String) -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::stt::FalSttProvider::new(api_key);
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Transcribe audio at `audio_source` and return the transcript.
    ///
    /// `audio_source` should be an `http(s)://` URL or a `data:` URI
    /// reachable by fal.ai's workers. `language` is an optional
    /// ISO-639-1 hint; when omitted fal performs language detection.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "SttTranscribe", ... }`
    /// on HTTP / fal queue / decoding errors.
    pub async fn transcribe(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "fal", e))?;
        Ok(map_transcription_result(result))
    }
}

#[uniffi::export]
impl FalSttProvider {
    /// Synchronous variant of [`transcribe`](Self::transcribe).
    pub fn transcribe_blocking(
        self: Arc<Self>,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.transcribe(audio_source, language).await })
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::BaseProvider for FalSttProvider {
    fn provider_id(&self) -> String {
        "fal-stt".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::SttProvider for FalSttProvider {
    fn provider_id(&self) -> String {
        "fal-stt".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Stt
    }

    async fn transcribe(
        &self,
        audio_source: String,
        language: Option<String>,
    ) -> Result<SttResult, BlazenError> {
        use blazen_llm::providers::capabilities::SttProvider as _;
        let request = build_transcription_request(audio_source, language);
        let result = self
            .inner
            .transcribe(request)
            .await
            .map_err(|e| provider_err("SttTranscribe", "fal-stt", e))?;
        Ok(map_transcription_result(result))
    }
}
