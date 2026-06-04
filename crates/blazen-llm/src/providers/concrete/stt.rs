//! STT concrete provider classes.
//!
//! One concrete `<Engine>SttProvider` (or `Provider`) per STT backend. Each:
//!
//! - Holds an inner backend handle erased into the appropriate dyn type
//!   ([`blazen_audio_stt::DynSttProvider`] for local engines; the live
//!   [`crate::providers::fal::FalProvider`] for the hosted fal.ai surface).
//! - Stamps a [`crate::providers::ProviderMetadata`] at construction so
//!   [`crate::providers::BaseProvider::metadata`] /
//!   [`crate::providers::BaseProvider::provider_id`] /
//!   [`crate::providers::BaseProvider::capability`] are O(1) lookups.
//! - Implements [`crate::providers::SttProvider`] by delegating to the
//!   existing `Transcription` impl on the inner handle (the
//!   `DynSttProvider` -> `Transcription` bridge lives in
//!   [`crate::backends::audio_stt`]; the FAL impl lives in
//!   [`crate::providers::fal`]).
//!
//! These types are the consumer-facing public API and are the basis for
//! the binding-side `WhisperCppProvider` / `FasterWhisperProvider` /
//! `WhisperStreamingProvider` / `FalSttProvider` exported through
//! napi-rs / `PyO3` / `UniFFI` / cabi / Ruby / WASM.

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use async_trait::async_trait;

use crate::compute::traits::Transcription;
use crate::compute::{TranscriptionRequest, TranscriptionResult};
use crate::error::BlazenError;
use crate::providers::capabilities::SttProvider;
use crate::providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// WhisperCppProvider — local whisper.cpp via blazen-audio-stt
// ---------------------------------------------------------------------------

/// Local whisper.cpp speech-to-text provider.
///
/// Wraps a [`blazen_audio_stt::DynSttProvider`] constructed from a
/// [`blazen_audio_stt::backends::whispercpp::WhisperCppBackend`]. The
/// pre-existing legacy alias [`crate::compat::whisper::WhisperCppProvider`]
/// stays in place for binding back-compat — this is the new
/// [`BaseProvider`]-aware shape that lives under
/// `crate::providers::concrete::stt`.
pub struct WhisperCppProvider {
    inner: Arc<blazen_audio_stt::DynSttProvider>,
    metadata: ProviderMetadata,
}

impl WhisperCppProvider {
    /// Build a `WhisperCppProvider` from the standard whisper.cpp options
    /// (model size variant, optional device, optional default language).
    ///
    /// Mirrors the [`crate::compat::whisper::WhisperCppProvider::from_options`]
    /// construction shape — see the uniffi factory
    /// `new_whisper_stt_model` in `blazen-uniffi::compute` for the
    /// canonical call site.
    ///
    /// # Errors
    ///
    /// Forwards any [`blazen_audio_stt::SttError`] from
    /// [`blazen_audio_stt::backends::whispercpp::WhisperCppBackend::new`]
    /// (option validation only; weight loading is lazy).
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        model: Option<String>,
        device: Option<String>,
        language: Option<String>,
    ) -> Result<Self, BlazenError> {
        use blazen_audio_stt::SttBackendHandle;
        use blazen_audio_stt::backends::whispercpp::{
            WhisperCppBackend, WhisperCppOptions, WhisperModel,
        };

        let model_variant = match model.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("tiny") => WhisperModel::Tiny,
            Some("base") => WhisperModel::Base,
            Some("medium") => WhisperModel::Medium,
            Some("large-v3" | "largev3" | "large_v3" | "large") => WhisperModel::LargeV3,
            _ => WhisperModel::Small,
        };
        let version_pin = format!("{model_variant:?}");
        let opts = WhisperCppOptions {
            model: model_variant,
            device,
            language,
            diarize: None,
            cache_dir: None,
        };
        let backend = WhisperCppBackend::new(opts)
            .map_err(|e| BlazenError::provider("whispercpp", e.to_string()))?;
        let inner = Arc::new(SttBackendHandle::new(backend).into_dyn());
        let metadata = ProviderMetadata::new("whispercpp", CapabilityKind::Stt)
            .with_display_name("whisper.cpp")
            .with_version(version_pin);
        Ok(Self { inner, metadata })
    }
}

impl std::fmt::Debug for WhisperCppProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperCppProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl BaseProvider for WhisperCppProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[async_trait]
impl SttProvider for WhisperCppProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        Transcription::transcribe(self.inner.as_ref(), request).await
    }
}

// ---------------------------------------------------------------------------
// FasterWhisperProvider — local faster-whisper (CTranslate2 / ct2rs)
// ---------------------------------------------------------------------------

/// Local faster-whisper (`CTranslate2`) speech-to-text provider.
///
/// Wraps a [`blazen_audio_stt::DynSttProvider`] backed by a
/// [`blazen_audio_stt::backends::faster_whisper::FasterWhisperBackend`].
/// Construction is cheap and synchronous — the ct2rs decoder is
/// materialised lazily on the first transcription call (HF download +
/// `CTranslate2` model load).
#[cfg(feature = "audio-stt-faster-whisper")]
pub struct FasterWhisperProvider {
    inner: Arc<blazen_audio_stt::DynSttProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-stt-faster-whisper")]
impl FasterWhisperProvider {
    /// Build a `FasterWhisperProvider`.
    ///
    /// `model_id` selects a Hugging Face bundle id (default
    /// `"Systran/faster-whisper-tiny"`). `model_dir` provides a
    /// pre-resolved local `CTranslate2` bundle directory; when supplied
    /// the HF download is skipped. `revision` pins a specific branch /
    /// tag / commit on the repo (default `main`). Mirrors the uniffi
    /// factory `new_faster_whisper_stt_model`.
    #[must_use]
    pub fn new(
        model_id: Option<String>,
        model_dir: Option<String>,
        revision: Option<String>,
    ) -> Self {
        use blazen_audio_stt::SttBackendHandle;
        use blazen_audio_stt::backends::faster_whisper::{
            FasterWhisperBackend, FasterWhisperConfig,
        };
        use std::path::PathBuf;

        let mut config = FasterWhisperConfig::default();
        if let Some(id) = model_id {
            config.model_id = id;
        }
        config.model_dir = model_dir.map(PathBuf::from);
        config.revision = revision;
        let version_pin = config.model_id.clone();
        let backend = FasterWhisperBackend::new(config);
        let inner = Arc::new(SttBackendHandle::new(backend).into_dyn());
        let metadata = ProviderMetadata::new("faster-whisper", CapabilityKind::Stt)
            .with_display_name("faster-whisper (CTranslate2)")
            .with_version(version_pin);
        Self { inner, metadata }
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
impl std::fmt::Debug for FasterWhisperProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FasterWhisperProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
impl BaseProvider for FasterWhisperProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-stt-faster-whisper")]
#[async_trait]
impl SttProvider for FasterWhisperProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        Transcription::transcribe(self.inner.as_ref(), request).await
    }
}

// ---------------------------------------------------------------------------
// WhisperStreamingProvider — chunked candle Whisper + Silero VAD
// ---------------------------------------------------------------------------

/// Streaming speech-to-text provider using a chunked candle Whisper
/// decoder fronted by Silero VAD.
///
/// Wraps a [`blazen_audio_stt::DynSttProvider`] backed by a
/// [`blazen_audio_stt::WhisperStreamingBackend`]. Note: the underlying
/// backend's `transcribe` returns `Unsupported` — only the streaming
/// surface is functional. The blocking [`SttProvider::transcribe`] entry
/// point is exposed for API parity but will surface that error to the
/// caller. Wire streaming via the backend directly if you need it.
#[cfg(feature = "audio-stt-whisper-streaming")]
pub struct WhisperStreamingProvider {
    inner: Arc<blazen_audio_stt::DynSttProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-stt-whisper-streaming")]
impl WhisperStreamingProvider {
    /// Build a `WhisperStreamingProvider`.
    ///
    /// `model_id` selects the underlying candle Whisper HF model
    /// (default `"openai/whisper-base"`). `vad_model_path` is accepted for
    /// backwards compatibility but **ignored** — the Silero VAD model is
    /// now embedded in the binary (control-flow-free, runs under ort or
    /// tract), so there is no external ONNX path to override.
    /// `chunk_seconds` / `chunk_overlap_seconds` tune the sliding window
    /// geometry; pass `None` for the defaults (`30.0` / `5.0`).
    #[must_use]
    pub fn new(
        model_id: Option<String>,
        vad_model_path: Option<String>,
        chunk_seconds: Option<f32>,
        chunk_overlap_seconds: Option<f32>,
    ) -> Self {
        use blazen_audio_stt::SttBackendHandle;
        use blazen_audio_stt::{WhisperStreamingBackend, WhisperStreamingConfig};

        // The Silero VAD is embedded; a caller-supplied path no longer
        // applies (kept in the signature for ABI/binding stability).
        let _ = vad_model_path;

        let mut config = WhisperStreamingConfig::default();
        if let Some(id) = model_id {
            config.model_id = id;
        }
        if let Some(s) = chunk_seconds {
            config.chunk_seconds = s;
        }
        if let Some(s) = chunk_overlap_seconds {
            config.chunk_overlap_seconds = s;
        }
        let version_pin = config.model_id.clone();
        let backend = WhisperStreamingBackend::new(config);
        let inner = Arc::new(SttBackendHandle::new(backend).into_dyn());
        let metadata = ProviderMetadata::new("whisper-streaming", CapabilityKind::Stt)
            .with_display_name("whisper-streaming (candle + Silero VAD)")
            .with_version(version_pin);
        Self { inner, metadata }
    }
}

#[cfg(feature = "audio-stt-whisper-streaming")]
impl std::fmt::Debug for WhisperStreamingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperStreamingProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-stt-whisper-streaming")]
impl BaseProvider for WhisperStreamingProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-stt-whisper-streaming")]
#[async_trait]
impl SttProvider for WhisperStreamingProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        Transcription::transcribe(self.inner.as_ref(), request).await
    }
}

// ---------------------------------------------------------------------------
// FalSttProvider — hosted fal.ai Whisper / Wizper endpoints
// ---------------------------------------------------------------------------

/// Hosted fal.ai speech-to-text provider.
///
/// Thin wrapper around [`crate::providers::fal::FalProvider`] that
/// surfaces only the [`SttProvider`] capability. Construction is cheap;
/// the underlying [`crate::providers::fal::FalProvider`] handles
/// endpoint selection, queueing, and result polling per call.
///
/// Note: this concrete is technically gated by the parent module's
/// `whispercpp` cfg (see [`super`]); when the new provider hierarchy
/// graduates from preview the gate will be relaxed so `FalSttProvider`
/// is reachable without a local Whisper backend. Until then build
/// with `--all-features` (CI default) to use it.
pub struct FalSttProvider {
    inner: Arc<crate::providers::fal::FalProvider>,
    metadata: ProviderMetadata,
}

impl FalSttProvider {
    /// Build a `FalSttProvider` with the given fal.ai API key.
    ///
    /// Uses the default HTTP client wired into the crate (matches the
    /// existing [`crate::providers::fal::FalProvider::new`] shape).
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = Arc::new(crate::providers::fal::FalProvider::new(api_key));
        let metadata =
            ProviderMetadata::new("fal", CapabilityKind::Stt).with_display_name("fal.ai Whisper");
        Self { inner, metadata }
    }

    /// Build a `FalSttProvider` with an explicit HTTP client.
    ///
    /// Available on every target — useful for tests, custom transport,
    /// and binding surfaces that supply their own HTTP plumbing.
    #[must_use]
    pub fn new_with_client(
        api_key: impl Into<String>,
        client: Arc<dyn crate::http::HttpClient>,
    ) -> Self {
        let inner = Arc::new(crate::providers::fal::FalProvider::new_with_client(
            api_key, client,
        ));
        let metadata =
            ProviderMetadata::new("fal", CapabilityKind::Stt).with_display_name("fal.ai Whisper");
        Self { inner, metadata }
    }
}

impl std::fmt::Debug for FalSttProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalSttProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl BaseProvider for FalSttProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[async_trait]
impl SttProvider for FalSttProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        Transcription::transcribe(self.inner.as_ref(), request).await
    }
}
