//! Music per-engine `#[uniffi::Object]` providers — populated by P4.2.music.
//!
//! Each provider class wraps the canonical
//! [`blazen_llm::providers::concrete::music`] concrete provider, exposing it
//! as a foreign-language class (Kotlin `class MusicGenProvider`, Swift
//! `class MusicGenProvider`, Go `*MusicGenProvider`, Ruby
//! `Blazen::MusicGenProvider`). Construction parameters mirror the
//! upstream provider's constructor; both async and `_blocking` variants of
//! `generate_music` / `generate_sfx` are emitted so callers picking either
//! style get a clean ergonomic surface.
//!
//! `generate_sfx` is only exposed for engines whose upstream provider
//! actually overrides it — `MusicGenProvider` does NOT expose it (MusicGen
//! is music-only; the trait's `Unsupported` default would be the only
//! callable shape, which is worse UX than the method just not existing).

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use crate::compute_music::MusicResult;
use crate::errors::{BlazenError, BlazenResult};
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Map a `blazen_llm::BlazenError` returned by the upstream music provider
/// into the UniFFI surface `Provider` variant with a stable `kind` tag.
/// Matches the helper pattern in [`crate::compute_music`].
fn provider_err(kind: &str, provider: &str, err: blazen_llm::BlazenError) -> BlazenError {
    BlazenError::Provider {
        kind: kind.to_owned(),
        message: err.to_string(),
        provider: Some(provider.to_owned()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

/// Wrap a constructor-side message into the `Provider` error variant.
fn init_err(kind: &str, provider: &str, message: impl Into<String>) -> BlazenError {
    BlazenError::Provider {
        kind: kind.to_owned(),
        message: message.into(),
        provider: Some(provider.to_owned()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

/// Convert an upstream [`blazen_llm::compute::AudioResult`] into the
/// FFI-shaped [`MusicResult`]. Picks the first clip and pulls
/// bytes / url / sample-rate / channels / duration off it. Mirrors the
/// `FalMusicAdapter::result_from_audio` helper in [`crate::compute_music`].
fn audio_result_to_music_result(result: blazen_llm::compute::AudioResult) -> MusicResult {
    let (bytes, mime_type, sample_rate, channels, duration_seconds, url) = result
        .audio
        .first()
        .map(|clip| {
            let mime = clip.media.media_type.mime().to_owned();
            let bytes = clip
                .media
                .base64
                .as_deref()
                .map(|s| {
                    use base64::Engine as _;
                    base64::engine::general_purpose::STANDARD
                        .decode(s)
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            let url = clip.media.url.clone().unwrap_or_default();
            let sr = clip.sample_rate.unwrap_or(0);
            let ch = u32::from(clip.channels.unwrap_or(0));
            let dur = clip.duration_seconds.unwrap_or(0.0);
            (bytes, mime, sr, ch, dur, url)
        })
        .unwrap_or_else(|| (Vec::new(), String::new(), 0, 0, 0.0, String::new()));
    MusicResult {
        bytes,
        mime_type,
        sample_rate,
        channels,
        duration_seconds,
        url,
    }
}

/// Build a [`blazen_llm::compute::MusicRequest`] from a `(prompt, duration)`
/// pair. The provider applies its own model defaults; we don't override
/// here — callers wanting a specific model construct one via the upstream
/// type directly (the per-engine providers wrap a fixed engine, so the
/// model selection happens at construction time).
fn build_request(prompt: String, duration_seconds: f32) -> blazen_llm::compute::MusicRequest {
    blazen_llm::compute::MusicRequest::new(prompt).with_duration(duration_seconds)
}

// ===========================================================================
// MusicGenProvider — Meta MusicGen text-to-music (native)
// ===========================================================================

/// Concrete provider class for Meta's `MusicGen` text-to-music model.
///
/// Wraps [`blazen_llm::providers::concrete::music::MusicGenProvider`].
/// Only `generate_music` is exposed — `MusicGen` is music-only and the
/// upstream trait's `generate_sfx` would surface `Unsupported`, so we omit
/// it from the FFI surface entirely.
#[cfg(feature = "audio-music-musicgen")]
#[derive(uniffi::Object)]
pub struct MusicGenProvider {
    inner: Arc<blazen_llm::providers::concrete::music::MusicGenProvider>,
}

#[cfg(feature = "audio-music-musicgen")]
#[uniffi::export(async_runtime = "tokio")]
impl MusicGenProvider {
    /// Build a new `MusicGen`-backed provider.
    ///
    /// `variant` selects the checkpoint (`"small"` / `"medium"` /
    /// `"large"`, case-insensitive); unrecognised values default to
    /// `Small`. `device` accepts `"cpu"`, `"cuda"`, `"cuda:N"`,
    /// `"metal"`, or `"metal:N"`; `None` lets the backend auto-detect.
    /// `cache_dir` overrides the Hugging Face Hub cache.
    /// `max_duration_seconds` overrides the default 30 s per-call safety
    /// cap.
    #[uniffi::constructor]
    pub fn new(
        variant: Option<String>,
        device: Option<String>,
        cache_dir: Option<String>,
        max_duration_seconds: Option<f32>,
    ) -> BlazenResult<Arc<Self>> {
        let inner = blazen_llm::providers::concrete::music::MusicGenProvider::new(
            variant,
            device,
            cache_dir,
            max_duration_seconds,
        )
        .map_err(|e| init_err("MusicGenInit", "musicgen", e.to_string()))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Generate `duration_seconds` of music conditioned on `prompt`.
    pub async fn generate_music(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "musicgen", e))?;
        Ok(audio_result_to_music_result(result))
    }
}

#[cfg(feature = "audio-music-musicgen")]
#[uniffi::export]
impl MusicGenProvider {
    /// Synchronous variant of [`generate_music`](Self::generate_music).
    pub fn generate_music_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_music(prompt, duration_seconds).await })
    }
}

// ===========================================================================
// AudioGenProvider — Meta AudioGen text-to-sfx (native)
// ===========================================================================

/// Concrete provider class for Meta's `AudioGen` text-to-sfx model.
///
/// Wraps [`blazen_llm::providers::concrete::music::AudioGenProvider`].
/// Both `generate_music` and `generate_sfx` are exposed (the underlying
/// backend routes both through the same dispatch trait).
#[cfg(feature = "audio-music-audiogen")]
#[derive(uniffi::Object)]
pub struct AudioGenProvider {
    inner: Arc<blazen_llm::providers::concrete::music::AudioGenProvider>,
}

#[cfg(feature = "audio-music-audiogen")]
#[uniffi::export(async_runtime = "tokio")]
impl AudioGenProvider {
    /// Build a new `AudioGen`-backed provider.
    ///
    /// `repo_id` overrides the default Hugging Face repo (defaults to
    /// `facebook/audiogen-medium`). `revision` pins a specific commit /
    /// tag. `device` follows the same format as
    /// [`MusicGenProvider::new`]. `cache_dir` overrides the cache.
    /// `max_duration_seconds` overrides the default 30 s safety cap.
    #[uniffi::constructor]
    pub fn new(
        repo_id: Option<String>,
        revision: Option<String>,
        device: Option<String>,
        cache_dir: Option<String>,
        max_duration_seconds: Option<f32>,
    ) -> BlazenResult<Arc<Self>> {
        let inner = blazen_llm::providers::concrete::music::AudioGenProvider::new(
            repo_id,
            revision,
            device,
            cache_dir,
            max_duration_seconds,
        )
        .map_err(|e| init_err("AudioGenInit", "audiogen", e.to_string()))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Generate `duration_seconds` of music conditioned on `prompt`.
    pub async fn generate_music(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "audiogen", e))?;
        Ok(audio_result_to_music_result(result))
    }

    /// Generate `duration_seconds` of sound-effect audio conditioned on
    /// `prompt`.
    pub async fn generate_sfx(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_sfx(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "audiogen", e))?;
        Ok(audio_result_to_music_result(result))
    }
}

#[cfg(feature = "audio-music-audiogen")]
#[uniffi::export]
impl AudioGenProvider {
    /// Synchronous variant of [`generate_music`](Self::generate_music).
    pub fn generate_music_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_music(prompt, duration_seconds).await })
    }

    /// Synchronous variant of [`generate_sfx`](Self::generate_sfx).
    pub fn generate_sfx_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_sfx(prompt, duration_seconds).await })
    }
}

// ===========================================================================
// StableAudioProvider — Stability AI Stable Audio Open (native)
// ===========================================================================

/// Concrete provider class for Stability AI's Stable Audio Open
/// text-to-audio model.
///
/// Wraps [`blazen_llm::providers::concrete::music::StableAudioProvider`].
/// Stable Audio Open generates both music AND sfx, so both methods are
/// wired through. The constructor is async because Stable Audio loads its
/// weights at construction time — the sync `new_blocking` shim drives the
/// shared Tokio runtime for non-async callers.
#[cfg(feature = "audio-music-stable-audio")]
#[derive(uniffi::Object)]
pub struct StableAudioProvider {
    inner: Arc<blazen_llm::providers::concrete::music::StableAudioProvider>,
}

#[cfg(feature = "audio-music-stable-audio")]
#[uniffi::export(async_runtime = "tokio")]
impl StableAudioProvider {
    /// Build a new Stable Audio Open-backed provider.
    ///
    /// `variant` selects the checkpoint (`"small"`, `"open-1.0"` /
    /// `"open1.0"` / `"open"` / `"1.0"`); unrecognised values default
    /// to `Small`. `tokenizer_path` must point at the T5 `SentencePiece`
    /// `tokenizer.json` shipped with the Stable Audio Open repo —
    /// required because Stable Audio's tokenizer is not auto-downloaded.
    /// `device` follows the same format as the other providers; `None`
    /// defaults to CPU. `max_duration_seconds` is accepted for API
    /// symmetry but Stable Audio enforces its own variant-dependent
    /// ceiling internally.
    #[uniffi::constructor]
    pub async fn new(
        variant: Option<String>,
        tokenizer_path: String,
        device: Option<String>,
        max_duration_seconds: Option<f32>,
    ) -> BlazenResult<Arc<Self>> {
        let inner = blazen_llm::providers::concrete::music::StableAudioProvider::new(
            variant,
            tokenizer_path,
            device,
            max_duration_seconds,
        )
        .await
        .map_err(|e| init_err("StableAudioInit", "stable-audio", e.to_string()))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
        }))
    }

    /// Generate `duration_seconds` of music conditioned on `prompt`.
    pub async fn generate_music(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "stable-audio", e))?;
        Ok(audio_result_to_music_result(result))
    }

    /// Generate `duration_seconds` of sound-effect audio conditioned on
    /// `prompt`.
    pub async fn generate_sfx(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_sfx(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "stable-audio", e))?;
        Ok(audio_result_to_music_result(result))
    }
}

#[cfg(feature = "audio-music-stable-audio")]
#[uniffi::export]
impl StableAudioProvider {
    /// Synchronous variant of [`generate_music`](Self::generate_music).
    pub fn generate_music_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_music(prompt, duration_seconds).await })
    }

    /// Synchronous variant of [`generate_sfx`](Self::generate_sfx).
    pub fn generate_sfx_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_sfx(prompt, duration_seconds).await })
    }
}

// ===========================================================================
// FalMusicProvider — fal.ai cloud music + sfx
// ===========================================================================

/// Concrete provider class for fal.ai's music + sfx endpoints.
///
/// Wraps [`blazen_llm::providers::concrete::music::FalMusicProvider`].
/// Both `generate_music` and `generate_sfx` are routed through it
/// (fal's per-endpoint dispatch handles the underlying model selection).
///
/// This type is part of the music concrete module, which is itself feature
/// -gated behind `audio-music-musicgen` — when MusicGen is disabled the
/// whole `concrete::music` module disappears, so fal.ai music callers fall
/// back to the central [`crate::compute_music::new_fal_music_model`]
/// factory which has no feature gate.
#[derive(uniffi::Object)]
pub struct FalMusicProvider {
    inner: Arc<blazen_llm::providers::concrete::music::FalMusicProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalMusicProvider {
    /// Build a new fal.ai-backed music provider.
    ///
    /// `api_key` may be empty when the provider resolves it from the
    /// `FAL_KEY` environment variable.
    #[uniffi::constructor]
    pub fn new(api_key: String) -> Arc<Self> {
        let inner = blazen_llm::providers::concrete::music::FalMusicProvider::new(api_key);
        Arc::new(Self {
            inner: Arc::new(inner),
        })
    }

    /// Generate `duration_seconds` of music conditioned on `prompt`.
    pub async fn generate_music(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "fal", e))?;
        Ok(audio_result_to_music_result(result))
    }

    /// Generate `duration_seconds` of sound-effect audio conditioned on
    /// `prompt`.
    pub async fn generate_sfx(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_sfx(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "fal", e))?;
        Ok(audio_result_to_music_result(result))
    }
}

#[uniffi::export]
impl FalMusicProvider {
    /// Synchronous variant of [`generate_music`](Self::generate_music).
    pub fn generate_music_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_music(prompt, duration_seconds).await })
    }

    /// Synchronous variant of [`generate_sfx`](Self::generate_sfx).
    pub fn generate_sfx_blocking(
        self: Arc<Self>,
        prompt: String,
        duration_seconds: f32,
    ) -> BlazenResult<MusicResult> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.generate_sfx(prompt, duration_seconds).await })
    }
}

// ===========================================================================
// Polymorphic capability-base trait impls (P4.2.x.3.music)
// ===========================================================================
//
// Each `<Engine>Provider` implements `BaseProvider` + `MusicProvider`
// from `crate::concrete::bases` so foreign callers can hold a
// `MusicProvider` reference / `Arc<dyn BaseProvider>` and dispatch
// polymorphically. The trait methods use `&self` receivers (UniFFI 0.31
// requirement); the existing `Arc<Self>`-receiver inherent methods on
// each class remain in place as the per-class ergonomic surface, and
// Rust's inherent-first method resolution keeps them the default
// callable for `self.generate_music(...)` call sites.
//
// Engines that don't natively support a method (MusicGen has no SFX,
// AudioGen has no music) explicitly return `BlazenError::Unsupported`
// because UniFFI 0.31 forbids default bodies on `#[uniffi::export]`'d
// trait methods.

#[cfg(feature = "audio-music-musicgen")]
impl crate::concrete::bases::BaseProvider for MusicGenProvider {
    fn provider_id(&self) -> String {
        "musicgen".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }
}

#[cfg(feature = "audio-music-musicgen")]
#[async_trait::async_trait]
impl crate::concrete::bases::MusicProvider for MusicGenProvider {
    fn provider_id(&self) -> String {
        "musicgen".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }

    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "musicgen", e))?;
        Ok(audio_result_to_music_result(result))
    }

    async fn generate_sfx(
        &self,
        _prompt: String,
        _duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "MusicGenProvider does not support generate_sfx — MusicGen is music-only. \
                      Use AudioGenProvider or StableAudioProvider for SFX."
                .to_string(),
        })
    }
}

#[cfg(feature = "audio-music-audiogen")]
impl crate::concrete::bases::BaseProvider for AudioGenProvider {
    fn provider_id(&self) -> String {
        "audiogen".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }
}

#[cfg(feature = "audio-music-audiogen")]
#[async_trait::async_trait]
impl crate::concrete::bases::MusicProvider for AudioGenProvider {
    fn provider_id(&self) -> String {
        "audiogen".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }

    async fn generate_music(
        &self,
        _prompt: String,
        _duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        Err(BlazenError::Unsupported {
            message: "AudioGenProvider does not support generate_music — AudioGen is sfx-only. \
                      Use MusicGenProvider or StableAudioProvider for music."
                .to_string(),
        })
    }

    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_sfx(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "audiogen", e))?;
        Ok(audio_result_to_music_result(result))
    }
}

#[cfg(feature = "audio-music-stable-audio")]
impl crate::concrete::bases::BaseProvider for StableAudioProvider {
    fn provider_id(&self) -> String {
        "stable-audio".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }
}

#[cfg(feature = "audio-music-stable-audio")]
#[async_trait::async_trait]
impl crate::concrete::bases::MusicProvider for StableAudioProvider {
    fn provider_id(&self) -> String {
        "stable-audio".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }

    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "stable-audio", e))?;
        Ok(audio_result_to_music_result(result))
    }

    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_sfx(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "stable-audio", e))?;
        Ok(audio_result_to_music_result(result))
    }
}

impl crate::concrete::bases::BaseProvider for FalMusicProvider {
    fn provider_id(&self) -> String {
        "fal-music".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }
}

#[async_trait::async_trait]
impl crate::concrete::bases::MusicProvider for FalMusicProvider {
    fn provider_id(&self) -> String {
        "fal-music".to_string()
    }
    fn capability(&self) -> crate::concrete::bases::CapabilityKind {
        crate::concrete::bases::CapabilityKind::Music
    }

    async fn generate_music(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_music(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "fal", e))?;
        Ok(audio_result_to_music_result(result))
    }

    async fn generate_sfx(
        &self,
        prompt: String,
        duration_seconds: f32,
    ) -> Result<MusicResult, BlazenError> {
        use blazen_llm::MusicProvider as _;
        let req = build_request(prompt, duration_seconds);
        let result = self
            .inner
            .generate_sfx(req)
            .await
            .map_err(|e| provider_err("MusicGeneration", "fal", e))?;
        Ok(audio_result_to_music_result(result))
    }
}
