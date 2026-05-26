//! Per-engine TTS `#[uniffi::Object]` providers.
//!
//! Each `<Engine>Provider` here is a thin UniFFI-exported wrapper around
//! the matching canonical concrete provider in
//! [`blazen_llm::providers::concrete::tts`]. Foreign bindgen (Go / Swift /
//! Kotlin / Ruby) emits a real class per engine — `PiperProvider`,
//! `KokoroProvider`, `VibeVoiceProvider`, `Qwen3TtsProvider`,
//! `SparkTtsProvider`, `BarkProvider`, `F5Provider`, `FalTtsProvider` —
//! rather than overloading the central `TtsModel` factory functions in
//! [`crate::compute`].
//!
//! Synthesis routes through [`blazen_llm::TtsProvider::synthesize`]
//! using a freshly-built [`blazen_llm::compute::requests::SpeechRequest`]
//! from the flat `(text, voice, language)` UniFFI signature, then
//! collapses the upstream [`blazen_llm::compute::results::AudioResult`]
//! into the existing [`crate::compute::TtsResult`] record.

#![allow(unused_imports)]

use std::sync::Arc;

use crate::compute::TtsResult;
use crate::errors::BlazenError;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a fresh [`blazen_llm::compute::requests::SpeechRequest`] from the
/// flat `(text, voice, language)` UniFFI signature.
#[allow(dead_code)]
fn build_request(
    text: String,
    voice: Option<String>,
    language: Option<String>,
) -> blazen_llm::compute::requests::SpeechRequest {
    let mut req = blazen_llm::compute::requests::SpeechRequest::new(text);
    if let Some(v) = voice {
        req = req.with_voice(v);
    }
    if let Some(lang) = language {
        req = req.with_language(lang);
    }
    req
}

/// Collapse an upstream [`blazen_llm::compute::results::AudioResult`] into
/// the UniFFI [`TtsResult`] record.
///
/// Picks the first clip in `audio` (TTS providers return exactly one).
/// Prefers `base64`, falling back to the clip URL when only that is
/// populated — matches the [`TtsResult`] doc-comment contract.
#[allow(dead_code)]
fn audio_to_tts_result(
    audio: blazen_llm::compute::results::AudioResult,
    provider: &str,
) -> Result<TtsResult, BlazenError> {
    let Some(clip) = audio.audio.into_iter().next() else {
        return Err(BlazenError::Provider {
            kind: "TtsSynthesis".into(),
            message: "empty audio result".into(),
            provider: Some(provider.into()),
            status: None,
            endpoint: None,
            request_id: None,
            detail: None,
            retry_after_ms: None,
        });
    };
    let mime_type = clip.media.media_type.mime().to_owned();
    let audio_base64 = clip.media.base64.or(clip.media.url).unwrap_or_default();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let duration_ms = (f64::from(clip.duration_seconds.unwrap_or(0.0)) * 1000.0).round() as u64;
    Ok(TtsResult {
        audio_base64,
        mime_type,
        duration_ms,
    })
}

/// Build a UniFFI `Provider` error for a TTS synthesis failure.
#[allow(dead_code)]
fn synth_error(provider: &str, err: impl std::fmt::Display) -> BlazenError {
    BlazenError::Provider {
        kind: "TtsSynthesis".into(),
        message: err.to_string(),
        provider: Some(provider.into()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

/// Build a UniFFI `Provider` error for a TTS construction failure.
#[allow(dead_code)]
fn init_error(provider: &str, err: impl std::fmt::Display) -> BlazenError {
    BlazenError::Provider {
        kind: format!("{provider}Init"),
        message: err.to_string(),
        provider: Some(provider.into()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

// ---------------------------------------------------------------------------
// PiperProvider — local Piper ONNX
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local Piper ONNX engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::PiperProvider`].
#[cfg(feature = "audio-tts-piper")]
#[derive(uniffi::Object)]
pub struct PiperProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::PiperProvider>,
}

#[cfg(feature = "audio-tts-piper")]
#[uniffi::export(async_runtime = "tokio")]
impl PiperProvider {
    /// Construct from a Piper voice id + already-resolved local file paths.
    ///
    /// `onnx_path` points at the `<voice>.onnx` weights; `config_path` is
    /// the sidecar `<voice>.onnx.json` (pass `None` to derive automatically
    /// by appending `.json` to the onnx path). `default_speaker_id` is
    /// used at synthesis time when [`blazen_llm::compute::requests::SpeechRequest::voice`]
    /// is `None` — typical for multi-speaker voices.
    #[uniffi::constructor]
    pub fn new(
        voice_id: String,
        onnx_path: String,
        config_path: Option<String>,
        default_speaker_id: Option<i64>,
    ) -> Result<Arc<Self>, BlazenError> {
        let inner = blazen_llm::providers::concrete::tts::PiperProvider::new(
            voice_id,
            std::path::PathBuf::from(onnx_path),
            config_path.map(std::path::PathBuf::from),
            default_speaker_id,
        )
        .map_err(|e| init_error("Piper", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("piper", e))?;
        audio_to_tts_result(audio, "piper")
    }
}

#[cfg(feature = "audio-tts-piper")]
#[uniffi::export]
impl PiperProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize) — blocks on
    /// the shared Tokio runtime.
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// KokoroProvider — local Kokoro-82M via AnyTtsBackend
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local Kokoro-82M engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::KokoroProvider`].
#[cfg(feature = "tts")]
#[derive(uniffi::Object)]
pub struct KokoroProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::KokoroProvider>,
}

#[cfg(feature = "tts")]
#[uniffi::export(async_runtime = "tokio")]
impl KokoroProvider {
    /// Construct with optional voice / language / sample-rate overrides.
    #[uniffi::constructor]
    pub fn new(
        voice: Option<String>,
        language: Option<String>,
        sample_rate: Option<u32>,
    ) -> Result<Arc<Self>, BlazenError> {
        let inner = blazen_llm::providers::concrete::tts::KokoroProvider::new(
            voice,
            language,
            sample_rate,
        )
        .map_err(|e| init_error("Kokoro", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("kokoro", e))?;
        audio_to_tts_result(audio, "kokoro")
    }
}

#[cfg(feature = "tts")]
#[uniffi::export]
impl KokoroProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// VibeVoiceProvider — local VibeVoice via AnyTtsBackend
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local VibeVoice engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::VibeVoiceProvider`].
#[cfg(feature = "tts")]
#[derive(uniffi::Object)]
pub struct VibeVoiceProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::VibeVoiceProvider>,
}

#[cfg(feature = "tts")]
#[uniffi::export(async_runtime = "tokio")]
impl VibeVoiceProvider {
    /// Construct with optional voice / language / sample-rate overrides.
    #[uniffi::constructor]
    pub fn new(
        voice: Option<String>,
        language: Option<String>,
        sample_rate: Option<u32>,
    ) -> Result<Arc<Self>, BlazenError> {
        let inner = blazen_llm::providers::concrete::tts::VibeVoiceProvider::new(
            voice,
            language,
            sample_rate,
        )
        .map_err(|e| init_error("VibeVoice", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("vibevoice", e))?;
        audio_to_tts_result(audio, "vibevoice")
    }
}

#[cfg(feature = "tts")]
#[uniffi::export]
impl VibeVoiceProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// Qwen3TtsProvider — local Qwen3-TTS via AnyTtsBackend
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local Qwen3-TTS engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::Qwen3TtsProvider`].
#[cfg(feature = "tts")]
#[derive(uniffi::Object)]
pub struct Qwen3TtsProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::Qwen3TtsProvider>,
}

#[cfg(feature = "tts")]
#[uniffi::export(async_runtime = "tokio")]
impl Qwen3TtsProvider {
    /// Construct with optional voice / language / sample-rate overrides.
    #[uniffi::constructor]
    pub fn new(
        voice: Option<String>,
        language: Option<String>,
        sample_rate: Option<u32>,
    ) -> Result<Arc<Self>, BlazenError> {
        let inner = blazen_llm::providers::concrete::tts::Qwen3TtsProvider::new(
            voice,
            language,
            sample_rate,
        )
        .map_err(|e| init_error("Qwen3Tts", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("qwen3-tts", e))?;
        audio_to_tts_result(audio, "qwen3-tts")
    }
}

#[cfg(feature = "tts")]
#[uniffi::export]
impl Qwen3TtsProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// SparkTtsProvider — local SparkAudio Spark-TTS
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local Spark-TTS engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::SparkTtsProvider`].
/// **CC-BY-NC-SA-4.0** — non-commercial use only.
#[cfg(feature = "audio-tts-spark")]
#[derive(uniffi::Object)]
pub struct SparkTtsProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::SparkTtsProvider>,
}

#[cfg(feature = "audio-tts-spark")]
#[uniffi::export(async_runtime = "tokio")]
impl SparkTtsProvider {
    /// Construct with optional model id / bundle dir / revision overrides.
    #[uniffi::constructor]
    pub fn new(
        model_id: Option<String>,
        model_dir: Option<String>,
        revision: Option<String>,
    ) -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::tts::SparkTtsProvider::new(
            model_id, model_dir, revision,
        );
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("spark-tts", e))?;
        audio_to_tts_result(audio, "spark-tts")
    }
}

#[cfg(feature = "audio-tts-spark")]
#[uniffi::export]
impl SparkTtsProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// BarkProvider — local Suno Bark
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local Bark engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::BarkProvider`].
#[cfg(feature = "audio-tts-bark")]
#[derive(uniffi::Object)]
pub struct BarkProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::BarkProvider>,
}

#[cfg(feature = "audio-tts-bark")]
#[uniffi::export(async_runtime = "tokio")]
impl BarkProvider {
    /// Construct a Bark provider with default configuration.
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::tts::BarkProvider::new();
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("bark", e))?;
        audio_to_tts_result(audio, "bark")
    }
}

#[cfg(feature = "audio-tts-bark")]
#[uniffi::export]
impl BarkProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// F5Provider — local F5-TTS
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by the local F5-TTS engine.
///
/// Wraps [`blazen_llm::providers::concrete::tts::F5Provider`].
#[cfg(feature = "audio-tts-f5")]
#[derive(uniffi::Object)]
pub struct F5Provider {
    inner: Arc<blazen_llm::providers::concrete::tts::F5Provider>,
}

#[cfg(feature = "audio-tts-f5")]
#[uniffi::export(async_runtime = "tokio")]
impl F5Provider {
    /// Construct an F5-TTS provider with default configuration.
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::tts::F5Provider::new();
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("f5-tts", e))?;
        audio_to_tts_result(audio, "f5-tts")
    }
}

#[cfg(feature = "audio-tts-f5")]
#[uniffi::export]
impl F5Provider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}

// ---------------------------------------------------------------------------
// FalTtsProvider — fal.ai cloud TTS (no feature gate)
// ---------------------------------------------------------------------------

/// Concrete TTS provider backed by fal.ai's hosted TTS endpoints.
///
/// Wraps [`blazen_llm::providers::concrete::tts::FalTtsProvider`].
#[derive(uniffi::Object)]
pub struct FalTtsProvider {
    inner: Arc<blazen_llm::providers::concrete::tts::FalTtsProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalTtsProvider {
    /// Construct from a fal.ai API key. An empty `api_key` falls back to
    /// the `FAL_KEY` environment variable.
    #[uniffi::constructor]
    pub fn new(api_key: String) -> Result<Arc<Self>, BlazenError> {
        let inner = blazen_llm::providers::concrete::tts::FalTtsProvider::new(api_key)
            .map_err(|e| init_error("FalTts", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Construct with an explicit default fal TTS endpoint
    /// (e.g. `"fal-ai/dia-tts"`).
    #[uniffi::constructor]
    pub fn with_model(
        api_key: String,
        default_model: Option<String>,
    ) -> Result<Arc<Self>, BlazenError> {
        let inner = blazen_llm::providers::concrete::tts::FalTtsProvider::with_model(
            api_key,
            default_model,
        )
        .map_err(|e| init_error("FalTts", e))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Synthesize `text` into an audio payload.
    pub async fn synthesize(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        use blazen_llm::TtsProvider as _;
        let req = build_request(text, voice, language);
        let audio = self
            .inner
            .synthesize(req)
            .await
            .map_err(|e| synth_error("fal", e))?;
        audio_to_tts_result(audio, "fal")
    }
}

#[uniffi::export]
impl FalTtsProvider {
    /// Synchronous variant of [`synthesize`](Self::synthesize).
    pub fn synthesize_blocking(
        self: Arc<Self>,
        text: String,
        voice: Option<String>,
        language: Option<String>,
    ) -> Result<TtsResult, BlazenError> {
        let this = Arc::clone(&self);
        crate::runtime::runtime()
            .block_on(async move { this.synthesize(text, voice, language).await })
    }
}
