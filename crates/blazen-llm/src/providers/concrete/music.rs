//! Music concrete provider classes — populated by P4.1.f-music.
//!
//! Four consumer-facing music / sfx provider classes that implement the
//! polymorphic [`crate::providers::BaseProvider`] root + the
//! [`crate::providers::capabilities::MusicProvider`] capability sub-trait:
//!
//! - [`MusicGenProvider`] — Meta's `MusicGen` text-to-music (native,
//!   gated by `audio-music-musicgen`).
//! - [`AudioGenProvider`] — Meta's `AudioGen` text-to-sfx (native,
//!   gated by `audio-music-audiogen`).
//! - [`StableAudioProvider`] — Stability AI's Stable Audio Open
//!   (native, gated by `audio-music-stable-audio`).
//! - [`FalMusicProvider`] — fal.ai-backed music + sfx (cloud, no
//!   feature gate beyond the module's `audio-music-musicgen` gate that
//!   pulls this file into the build).
//!
//! Each wraps either a [`blazen_audio_music::DynMusicProvider`]
//! (`Arc<dyn MusicBackend>`) — which already implements
//! [`crate::compute::traits::AudioGeneration`] via the bridge in
//! [`crate::backends::audio_music`] — or the existing
//! [`crate::providers::fal::FalProvider`], delegating the request
//! shape directly.

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use async_trait::async_trait;

use crate::compute::requests::MusicRequest;
use crate::compute::results::AudioResult;
use crate::error::BlazenError;
use crate::providers::capabilities::MusicProvider;
use crate::providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Parse a device spec string (`"cpu"`, `"cuda"`, `"cuda:N"`,
/// `"metal"`, `"metal:N"`) into a `candle_core::Device`. Returns
/// `Ok(None)` when `spec` is `None` / empty so the caller can defer to
/// the backend's auto-detection logic.
#[cfg(any(
    feature = "audio-music-musicgen",
    feature = "audio-music-stable-audio",
    feature = "audio-music-audiogen"
))]
fn parse_device(spec: Option<&str>) -> Result<Option<candle_core::Device>, BlazenError> {
    let Some(raw) = spec else {
        return Ok(None);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Ok(None);
    }
    if normalized == "cpu" {
        return Ok(Some(candle_core::Device::Cpu));
    }
    let (kind, idx) = match normalized.split_once(':') {
        Some((k, rest)) => {
            let parsed = rest.parse::<usize>().map_err(|e| {
                BlazenError::validation(format!(
                    "music device {raw:?} has non-numeric index {rest:?}: {e}"
                ))
            })?;
            (k, parsed)
        }
        None => (normalized.as_str(), 0),
    };
    match kind {
        "cuda" => candle_core::Device::new_cuda(idx)
            .map(Some)
            .map_err(|e| BlazenError::validation(format!("cuda:{idx} unavailable: {e}"))),
        "metal" => candle_core::Device::new_metal(idx)
            .map(Some)
            .map_err(|e| BlazenError::validation(format!("metal:{idx} unavailable: {e}"))),
        other => Err(BlazenError::validation(format!(
            "unknown music device {other:?} (want one of: cpu, cuda[:N], metal[:N])"
        ))),
    }
}

// ===========================================================================
// MusicGenProvider — Meta MusicGen text-to-music (native)
// ===========================================================================

/// Concrete provider class for Meta's `MusicGen` text-to-music model.
///
/// Wraps the [`blazen_audio_music::DynMusicProvider`] handle produced
/// from a [`blazen_audio_music::MusicgenBackend`]. `generate_sfx`
/// is intentionally left at the trait's default (`Unsupported`) —
/// `MusicGen` is music-only; use [`AudioGenProvider`] for sfx.
#[cfg(feature = "audio-music-musicgen")]
pub struct MusicGenProvider {
    inner: Arc<blazen_audio_music::DynMusicProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-music-musicgen")]
impl MusicGenProvider {
    /// Build a new `MusicGen`-backed provider.
    ///
    /// `variant` selects the checkpoint (`"small"` / `"medium"` /
    /// `"large"`, case-insensitive); unrecognised values default to
    /// `Small`. `device` accepts `"cpu"`, `"cuda"`, `"cuda:N"`,
    /// `"metal"`, or `"metal:N"`; `None` lets the backend auto-detect.
    /// `cache_dir` overrides the Hugging Face Hub cache.
    /// `max_duration_seconds` overrides the default 30 s per-call
    /// safety cap.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::provider`] when `device` cannot be parsed
    /// into a valid backend device specifier.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        variant: Option<String>,
        device: Option<String>,
        cache_dir: Option<String>,
        max_duration_seconds: Option<f32>,
    ) -> Result<Self, BlazenError> {
        let variant_kind = match variant.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("medium") => crate::MusicgenVariant::Medium,
            Some("large") => crate::MusicgenVariant::Large,
            _ => crate::MusicgenVariant::Small,
        };
        let device = parse_device(device.as_deref())?;
        let config = crate::MusicgenConfig {
            variant: variant_kind,
            device,
            cache_dir: cache_dir.map(std::path::PathBuf::from),
            max_duration_seconds: max_duration_seconds.unwrap_or(30.0),
        };
        let backend = crate::MusicgenBackend::new(config);
        let dyn_provider: blazen_audio_music::DynMusicProvider = Arc::new(backend);
        let version = match variant_kind {
            crate::MusicgenVariant::Small => "facebook/musicgen-small",
            crate::MusicgenVariant::Medium => "facebook/musicgen-medium",
            crate::MusicgenVariant::Large => "facebook/musicgen-large",
        };
        Ok(Self {
            inner: Arc::new(dyn_provider),
            metadata: ProviderMetadata::new("musicgen", CapabilityKind::Music)
                .with_display_name("MusicGen")
                .with_version(version),
        })
    }
}

#[cfg(feature = "audio-music-musicgen")]
impl std::fmt::Debug for MusicGenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicGenProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-music-musicgen")]
impl BaseProvider for MusicGenProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-music-musicgen")]
#[async_trait]
impl MusicProvider for MusicGenProvider {
    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_music(request).await
    }
    // generate_sfx — defaults to Unsupported via the trait. MusicGen is
    // music-only; callers wanting SFX should use AudioGenProvider.
}

// ===========================================================================
// AudioGenProvider — Meta AudioGen text-to-sfx (native)
// ===========================================================================

/// Concrete provider class for Meta's `AudioGen` text-to-sfx model.
///
/// `AudioGen` is sfx-primary, so [`MusicProvider::generate_sfx`] is the
/// canonical entry; [`MusicProvider::generate_music`] is also wired
/// through (the underlying backend routes both via the same trait).
#[cfg(feature = "audio-music-audiogen")]
pub struct AudioGenProvider {
    inner: Arc<blazen_audio_music::DynMusicProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-music-audiogen")]
impl AudioGenProvider {
    /// Build a new `AudioGen`-backed provider.
    ///
    /// `repo_id` overrides the default Hugging Face repo (defaults to
    /// `facebook/audiogen-medium`). `revision` pins a specific
    /// commit / tag. `device` follows the same format as
    /// [`MusicGenProvider::new`]. `cache_dir` overrides the cache.
    /// `max_duration_seconds` overrides the default 30 s safety cap.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::provider`] when `device` cannot be parsed
    /// into a valid backend device specifier.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        repo_id: Option<String>,
        revision: Option<String>,
        device: Option<String>,
        cache_dir: Option<String>,
        max_duration_seconds: Option<f32>,
    ) -> Result<Self, BlazenError> {
        let device = parse_device(device.as_deref())?;
        let repo = repo_id.unwrap_or_else(|| "facebook/audiogen-medium".to_string());
        let config = crate::AudioGenConfig {
            repo_id: repo.clone(),
            revision: revision.clone(),
            device,
            cache_dir: cache_dir.map(std::path::PathBuf::from),
            max_duration_seconds: max_duration_seconds.unwrap_or(30.0),
        };
        let backend = crate::AudioGenBackend::new(config);
        let dyn_provider: blazen_audio_music::DynMusicProvider = Arc::new(backend);
        let version = match revision {
            Some(rev) => format!("{repo}@{rev}"),
            None => repo,
        };
        Ok(Self {
            inner: Arc::new(dyn_provider),
            metadata: ProviderMetadata::new("audiogen", CapabilityKind::Music)
                .with_display_name("AudioGen")
                .with_version(version),
        })
    }
}

#[cfg(feature = "audio-music-audiogen")]
impl std::fmt::Debug for AudioGenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioGenProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-music-audiogen")]
impl BaseProvider for AudioGenProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-music-audiogen")]
#[async_trait]
impl MusicProvider for AudioGenProvider {
    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_music(request).await
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_sfx(request).await
    }
}

// ===========================================================================
// StableAudioProvider — Stability AI Stable Audio Open (native)
// ===========================================================================

/// Concrete provider class for Stability AI's Stable Audio Open
/// text-to-audio model.
///
/// Stable Audio Open generates both music AND sfx, so both
/// [`MusicProvider::generate_music`] and [`MusicProvider::generate_sfx`]
/// are wired through.
#[cfg(feature = "audio-music-stable-audio")]
pub struct StableAudioProvider {
    inner: Arc<blazen_audio_music::DynMusicProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-music-stable-audio")]
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
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::provider`] when `device` cannot be parsed
    /// or when the Stable Audio backend fails to load weights from the
    /// configured repo / local path.
    pub async fn new(
        variant: Option<String>,
        tokenizer_path: String,
        device: Option<String>,
        _max_duration_seconds: Option<f32>,
    ) -> Result<Self, BlazenError> {
        let variant_kind = match variant.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("open-1.0" | "open1.0" | "open_1_0" | "open" | "1.0") => {
                crate::StableAudioVariant::Open1_0
            }
            _ => crate::StableAudioVariant::Small,
        };
        let device = parse_device(device.as_deref())?.unwrap_or(candle_core::Device::Cpu);
        let hf_repo = variant_kind.hf_repo().to_string();
        let config = crate::StableAudioConfig {
            hf_repo: hf_repo.clone(),
            local_weights_path: None,
            tokenizer_path: std::path::PathBuf::from(tokenizer_path),
            device,
            dtype: candle_core::DType::F32,
            variant: variant_kind,
        };
        let backend = crate::StableAudioBackend::load(config)
            .await
            .map_err(|e| {
                BlazenError::provider("stable-audio", format!("StableAudioInit: {e}"))
            })?;
        let dyn_provider: blazen_audio_music::DynMusicProvider = Arc::new(backend);
        Ok(Self {
            inner: Arc::new(dyn_provider),
            metadata: ProviderMetadata::new("stable-audio", CapabilityKind::Music)
                .with_display_name("Stable Audio Open")
                .with_version(hf_repo),
        })
    }
}

#[cfg(feature = "audio-music-stable-audio")]
impl std::fmt::Debug for StableAudioProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StableAudioProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-music-stable-audio")]
impl BaseProvider for StableAudioProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-music-stable-audio")]
#[async_trait]
impl MusicProvider for StableAudioProvider {
    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_music(request).await
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_sfx(request).await
    }
}

// ===========================================================================
// FalMusicProvider — fal.ai cloud music + sfx (no feature gate)
// ===========================================================================

/// Concrete provider class for fal.ai's music + sfx endpoints.
///
/// Wraps the existing [`crate::providers::fal::FalProvider`]; both
/// [`MusicProvider::generate_music`] and [`MusicProvider::generate_sfx`]
/// are routed through it (fal's per-endpoint dispatch handles the
/// underlying model selection).
pub struct FalMusicProvider {
    inner: Arc<crate::providers::fal::FalProvider>,
    metadata: ProviderMetadata,
}

impl FalMusicProvider {
    /// Build a new fal.ai-backed music provider.
    ///
    /// `api_key` may be empty when the provider resolves it from the
    /// `FAL_KEY` environment variable.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        let provider = crate::providers::fal::FalProvider::new(api_key);
        Self {
            inner: Arc::new(provider),
            metadata: ProviderMetadata::new("fal", CapabilityKind::Music)
                .with_display_name("fal.ai (music)"),
        }
    }
}

impl std::fmt::Debug for FalMusicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalMusicProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl BaseProvider for FalMusicProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[async_trait]
impl MusicProvider for FalMusicProvider {
    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_music(request).await
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        use crate::compute::traits::AudioGeneration;
        self.inner.generate_sfx(request).await
    }
}
