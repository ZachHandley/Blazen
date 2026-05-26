//! Concrete per-engine TTS provider classes.
//!
//! Each `<Engine>Provider` newtype wraps the matching upstream backend
//! (Piper, Kokoro-82M, `VibeVoice`, Qwen3-TTS, Spark-TTS, Bark, F5,
//! fal.ai cloud TTS) and implements the polymorphic
//! [`crate::providers::root::BaseProvider`] root plus the
//! [`crate::providers::capabilities::TtsProvider`] capability sub-trait.
//!
//! Construction patterns mirror the existing `new_*_tts_model`
//! factories in `crates/blazen-uniffi/src/compute.rs` so the binding
//! crates (napi-rs / `PyO3` / `UniFFI` / cabi / Ruby / WASM) can wrap
//! these structs directly without re-deriving the HF-download / config
//! plumbing.
//!
//! All providers route synthesis through
//! [`blazen_audio_tts::DynTtsProvider`] — which already carries the
//! [`crate::compute::traits::AudioGeneration`] bridge in
//! [`crate::backends::tts`] — so the request / result translation is
//! shared.

#![allow(unused_imports)]

use std::sync::Arc;

use async_trait::async_trait;

use crate::compute::requests::SpeechRequest;
use crate::compute::results::AudioResult;
use crate::compute::traits::AudioGeneration;
use crate::error::BlazenError;
use crate::providers::capabilities::TtsProvider;
use crate::providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};

// ---------------------------------------------------------------------------
// PiperProvider — local Piper ONNX text-to-speech
// ---------------------------------------------------------------------------

/// Concrete [`TtsProvider`] backed by the local
/// [`blazen_audio_tts::PiperBackend`] engine.
///
/// Resolves a Piper voice id like `"en_US-amy-medium"` into the
/// corresponding `rhasspy/piper-voices` repo path, downloads the
/// `<voice>.onnx` + `<voice>.onnx.json` pair through
/// [`blazen_model_cache::ModelCache`], then constructs the backend with
/// [`blazen_audio_tts::PiperBackend::with_voice`].
#[cfg(feature = "audio-tts-piper")]
pub struct PiperProvider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-piper")]
impl PiperProvider {
    /// Construct from pre-resolved Piper voice files.
    ///
    /// `onnx_path` points at the `<voice>.onnx` weights and
    /// `config_path` at the sidecar `<voice>.onnx.json` (when `None`,
    /// [`blazen_audio_tts::PiperBackend::with_voice`] derives it by
    /// appending `.json` to the onnx path). `default_speaker_id` is
    /// used at synthesis time when [`SpeechRequest`]'s
    /// `speaker_id` is `None` — typical for multi-speaker voices like
    /// `en_US-libritts_r-medium`.
    ///
    /// `voice_id` is the canonical `"<lang>_<region>-<speaker>-<quality>"`
    /// identifier used as the metadata version pin (e.g.
    /// `"en_US-amy-medium"`); it is not interpreted as a path.
    ///
    /// HF-download / cache resolution lives in the binding crates that
    /// pull in `blazen-model-cache` (see `new_piper_tts_model` in
    /// `crates/blazen-uniffi/src/compute.rs`); `blazen-llm` itself does
    /// not depend on the cache crate, so callers pass concrete paths.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] if the Piper voice files fail
    /// to load (typically a missing `espeak-ng` binary on the host).
    pub fn new(
        voice_id: impl Into<String>,
        onnx_path: std::path::PathBuf,
        config_path: Option<std::path::PathBuf>,
        default_speaker_id: Option<i64>,
    ) -> Result<Self, BlazenError> {
        use blazen_audio_tts::PiperBackend;

        let voice_id = voice_id.into();
        let backend = PiperBackend::with_voice(onnx_path, config_path, default_speaker_id)
            .map_err(|e| BlazenError::provider("piper", e.to_string()))?;

        let metadata = ProviderMetadata::new("piper", CapabilityKind::Tts).with_version(voice_id);

        Ok(Self {
            inner: Arc::new(blazen_audio_tts::DynTtsProvider::erase(backend)),
            metadata,
        })
    }

    /// Parse a Piper voice id like `"en_US-amy-medium"` into the
    /// corresponding repo-relative path stem inside the
    /// `rhasspy/piper-voices` Hugging Face repo
    /// (`en/en_US/amy/medium/en_US-amy-medium`). Returns `None` if the
    /// id doesn't split into exactly three `-`-delimited segments.
    ///
    /// Exposed as an associated helper so the binding crates (which do
    /// own the HF cache surface) can re-use the parser without
    /// duplicating it.
    #[must_use]
    pub fn voice_id_to_hf_path(voice_id: &str) -> Option<String> {
        piper_voice_id_to_hf_path(voice_id)
    }
}

#[cfg(feature = "audio-tts-piper")]
impl std::fmt::Debug for PiperProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PiperProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-piper")]
impl BaseProvider for PiperProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-piper")]
#[async_trait]
impl TtsProvider for PiperProvider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

/// Parse a Piper voice id like `"en_US-amy-medium"` into the
/// corresponding repo-relative path stem inside the
/// `rhasspy/piper-voices` Hugging Face repo. Mirrors the helper in
/// `blazen-uniffi/src/compute.rs`.
#[cfg(feature = "audio-tts-piper")]
fn piper_voice_id_to_hf_path(voice_id: &str) -> Option<String> {
    let parts: Vec<&str> = voice_id.splitn(3, '-').collect();
    if parts.len() != 3 {
        return None;
    }
    let lang_region = parts[0];
    let lang = lang_region.split('_').next()?;
    let speaker = parts[1];
    let quality = parts[2];
    Some(format!(
        "{lang}/{lang_region}/{speaker}/{quality}/{voice_id}"
    ))
}

// ---------------------------------------------------------------------------
// any-tts engines: Kokoro-82M, VibeVoice, Qwen3-TTS
// ---------------------------------------------------------------------------

/// Build a [`blazen_audio_tts::DynTtsProvider`] wrapping
/// [`blazen_audio_tts::AnyTtsBackend`] configured for the requested
/// [`blazen_audio_tts::TtsModel`].
///
/// Shared helper used by the Kokoro / `VibeVoice` / Qwen3 concrete
/// constructors so the option-plumbing lives in one place.
#[cfg(feature = "audio-tts-anytts")]
fn build_anytts_dyn(
    model: blazen_audio_tts::TtsModel,
    voice: Option<String>,
    language: Option<String>,
    sample_rate: Option<u32>,
    provider_id: &str,
) -> Result<Arc<blazen_audio_tts::DynTtsProvider>, BlazenError> {
    let opts = blazen_audio_tts::TtsOptions {
        model: Some(model),
        voice,
        language,
        sample_rate,
        cache_dir: None,
        ..blazen_audio_tts::TtsOptions::default()
    };
    let backend = blazen_audio_tts::AnyTtsBackend::from_options(opts)
        .map_err(|e| BlazenError::provider(provider_id.to_owned(), e.to_string()))?;
    Ok(Arc::new(blazen_audio_tts::DynTtsProvider::erase(backend)))
}

/// Concrete [`TtsProvider`] backed by the local Kokoro-82M engine via
/// [`blazen_audio_tts::AnyTtsBackend`].
#[cfg(feature = "audio-tts-anytts")]
pub struct KokoroProvider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-anytts")]
impl KokoroProvider {
    /// Construct a Kokoro-82M provider with optional voice / language /
    /// sample-rate overrides.
    ///
    /// `voice` selects a Kokoro speaker preset (e.g. `"af_bella"`).
    /// `sample_rate` overrides the model's native rate (24 kHz).
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] if
    /// [`blazen_audio_tts::AnyTtsBackend::from_options`] fails.
    pub fn new(
        voice: Option<String>,
        language: Option<String>,
        sample_rate: Option<u32>,
    ) -> Result<Self, BlazenError> {
        let inner = build_anytts_dyn(
            blazen_audio_tts::TtsModel::Kokoro82m,
            voice,
            language,
            sample_rate,
            "kokoro",
        )?;
        Ok(Self {
            inner,
            metadata: ProviderMetadata::new("kokoro", CapabilityKind::Tts)
                .with_version("kokoro82m"),
        })
    }
}

#[cfg(feature = "audio-tts-anytts")]
impl std::fmt::Debug for KokoroProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KokoroProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-anytts")]
impl BaseProvider for KokoroProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-anytts")]
#[async_trait]
impl TtsProvider for KokoroProvider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

/// Concrete [`TtsProvider`] backed by the local `VibeVoice` engine via
/// [`blazen_audio_tts::AnyTtsBackend`].
#[cfg(feature = "audio-tts-anytts")]
pub struct VibeVoiceProvider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-anytts")]
impl VibeVoiceProvider {
    /// Construct a `VibeVoice` provider with optional voice / language /
    /// sample-rate overrides.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] if
    /// [`blazen_audio_tts::AnyTtsBackend::from_options`] fails.
    pub fn new(
        voice: Option<String>,
        language: Option<String>,
        sample_rate: Option<u32>,
    ) -> Result<Self, BlazenError> {
        let inner = build_anytts_dyn(
            blazen_audio_tts::TtsModel::VibeVoice,
            voice,
            language,
            sample_rate,
            "vibevoice",
        )?;
        Ok(Self {
            inner,
            metadata: ProviderMetadata::new("vibevoice", CapabilityKind::Tts)
                .with_version("vibevoice"),
        })
    }
}

#[cfg(feature = "audio-tts-anytts")]
impl std::fmt::Debug for VibeVoiceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VibeVoiceProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-anytts")]
impl BaseProvider for VibeVoiceProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-anytts")]
#[async_trait]
impl TtsProvider for VibeVoiceProvider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

/// Concrete [`TtsProvider`] backed by the local Qwen3-TTS engine via
/// [`blazen_audio_tts::AnyTtsBackend`].
#[cfg(feature = "audio-tts-anytts")]
pub struct Qwen3TtsProvider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-anytts")]
impl Qwen3TtsProvider {
    /// Construct a Qwen3-TTS provider with optional voice / language /
    /// sample-rate overrides.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] if
    /// [`blazen_audio_tts::AnyTtsBackend::from_options`] fails.
    pub fn new(
        voice: Option<String>,
        language: Option<String>,
        sample_rate: Option<u32>,
    ) -> Result<Self, BlazenError> {
        let inner = build_anytts_dyn(
            blazen_audio_tts::TtsModel::Qwen3Tts,
            voice,
            language,
            sample_rate,
            "qwen3-tts",
        )?;
        Ok(Self {
            inner,
            metadata: ProviderMetadata::new("qwen3-tts", CapabilityKind::Tts)
                .with_version("qwen3_tts"),
        })
    }
}

#[cfg(feature = "audio-tts-anytts")]
impl std::fmt::Debug for Qwen3TtsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-anytts")]
impl BaseProvider for Qwen3TtsProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-anytts")]
#[async_trait]
impl TtsProvider for Qwen3TtsProvider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

// ---------------------------------------------------------------------------
// SparkTtsProvider — local SparkAudio Spark-TTS
// ---------------------------------------------------------------------------

/// Concrete [`TtsProvider`] backed by the local
/// [`blazen_audio_tts::backends::spark::SparkTtsBackend`] engine
/// (`SparkAudio/Spark-TTS-0.5B` `BiCodec` + Qwen2.5-0.5B AR decoder).
///
/// The bundle ships under **CC-BY-NC-SA-4.0** — non-commercial use only;
/// the backend emits a one-shot warning on first synthesis.
#[cfg(feature = "audio-tts-spark")]
pub struct SparkTtsProvider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-spark")]
impl SparkTtsProvider {
    /// Construct a Spark-TTS provider.
    ///
    /// `model_id` selects a Hugging Face bundle id (defaults to
    /// `"SparkAudio/Spark-TTS-0.5B"`). `model_dir` provides a
    /// pre-resolved local bundle directory containing the `LLM/` +
    /// `BiCodec/` subtrees; when supplied, the HF download is skipped.
    /// `revision` pins a specific branch / tag / commit.
    #[must_use]
    pub fn new(
        model_id: Option<String>,
        model_dir: Option<String>,
        revision: Option<String>,
    ) -> Self {
        use blazen_audio_tts::backends::spark::{SparkTtsBackend, SparkTtsConfig};
        use std::path::PathBuf;

        let mut config = SparkTtsConfig::default();
        if let Some(id) = model_id {
            config.model_id = id;
        }
        config.model_dir = model_dir.map(PathBuf::from);
        config.revision = revision;

        let version = config.model_id.clone();
        let backend = SparkTtsBackend::new(config);
        let inner = Arc::new(blazen_audio_tts::DynTtsProvider::erase(backend));
        let metadata =
            ProviderMetadata::new("spark-tts", CapabilityKind::Tts).with_version(version);

        Self { inner, metadata }
    }
}

#[cfg(feature = "audio-tts-spark")]
impl std::fmt::Debug for SparkTtsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparkTtsProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-spark")]
impl BaseProvider for SparkTtsProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-spark")]
#[async_trait]
impl TtsProvider for SparkTtsProvider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

// ---------------------------------------------------------------------------
// BarkProvider — local Suno Bark
// ---------------------------------------------------------------------------

/// Concrete [`TtsProvider`] backed by the local
/// [`blazen_audio_tts::BarkBackend`] (Suno Bark) engine.
#[cfg(feature = "audio-tts-bark")]
pub struct BarkProvider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-bark")]
impl BarkProvider {
    /// Construct a Bark provider with default configuration.
    ///
    /// The underlying [`blazen_audio_tts::BarkConfig`] knobs (HF repo,
    /// device, sample rate) are taken from
    /// [`blazen_audio_tts::BarkConfig::default`]. Wave F.1 scaffolding —
    /// synthesis returns the upstream "engine not wired" error until the
    /// implementation lands.
    #[must_use]
    pub fn new() -> Self {
        let backend = blazen_audio_tts::BarkBackend::new(blazen_audio_tts::BarkConfig::default());
        let inner = Arc::new(blazen_audio_tts::DynTtsProvider::erase(backend));
        let metadata =
            ProviderMetadata::new("bark", CapabilityKind::Tts).with_version("suno/bark");
        Self { inner, metadata }
    }
}

#[cfg(feature = "audio-tts-bark")]
impl Default for BarkProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "audio-tts-bark")]
impl std::fmt::Debug for BarkProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarkProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-bark")]
impl BaseProvider for BarkProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-bark")]
#[async_trait]
impl TtsProvider for BarkProvider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

// ---------------------------------------------------------------------------
// F5Provider — local F5-TTS
// ---------------------------------------------------------------------------

/// Concrete [`TtsProvider`] backed by the local
/// [`blazen_audio_tts::F5Backend`] (F5-TTS) engine.
#[cfg(feature = "audio-tts-f5")]
pub struct F5Provider {
    inner: Arc<blazen_audio_tts::DynTtsProvider>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-tts-f5")]
impl F5Provider {
    /// Construct an F5-TTS provider with default configuration.
    ///
    /// The underlying [`blazen_audio_tts::F5Config`] knobs (HF repo,
    /// device, sample rate) are taken from
    /// [`blazen_audio_tts::F5Config::default`]. Wave F.1 scaffolding —
    /// synthesis returns the upstream "engine not wired" error until the
    /// implementation lands.
    #[must_use]
    pub fn new() -> Self {
        let backend = blazen_audio_tts::F5Backend::new(blazen_audio_tts::F5Config::default());
        let inner = Arc::new(blazen_audio_tts::DynTtsProvider::erase(backend));
        let metadata = ProviderMetadata::new("f5-tts", CapabilityKind::Tts).with_version("f5-tts");
        Self { inner, metadata }
    }
}

#[cfg(feature = "audio-tts-f5")]
impl Default for F5Provider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "audio-tts-f5")]
impl std::fmt::Debug for F5Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F5Provider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-tts-f5")]
impl BaseProvider for F5Provider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-tts-f5")]
#[async_trait]
impl TtsProvider for F5Provider {
    async fn synthesize(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        AudioGeneration::text_to_speech(&*self.inner, request).await
    }
}

// ---------------------------------------------------------------------------
// FalTtsProvider — fal.ai cloud TTS
// ---------------------------------------------------------------------------

/// Concrete [`TtsProvider`] backed by fal.ai's hosted TTS endpoints via
/// [`crate::providers::fal::FalProvider`].
///
/// Mirrors the `FalTtsAdapter` in `blazen-uniffi/src/compute.rs`: a
/// single [`crate::providers::fal::FalProvider`] instance carries the
/// API key + base URL, and an optional default model id is injected
/// onto each [`SpeechRequest`] that doesn't already carry one.
pub struct FalTtsProvider {
    inner: Arc<crate::providers::fal::FalProvider>,
    default_model: Option<String>,
    metadata: ProviderMetadata,
}

impl FalTtsProvider {
    /// Construct from a fal.ai API key.
    ///
    /// An empty `api_key` falls back to the `FAL_KEY` environment
    /// variable through [`crate::keys::resolve_api_key`].
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] if
    /// [`crate::providers::fal::FalProvider::from_options`] fails (e.g.
    /// the API key cannot be resolved from the environment).
    pub fn new(api_key: impl Into<String>) -> Result<Self, BlazenError> {
        Self::with_model(api_key, None)
    }

    /// Construct with an explicit default fal TTS endpoint
    /// (e.g. `"fal-ai/dia-tts"`).
    ///
    /// When `default_model` is `None`, the per-call `voice` / `language`
    /// arguments decide which endpoint fal routes to.
    ///
    /// # Errors
    ///
    /// Returns [`BlazenError::Provider`] if
    /// [`crate::providers::fal::FalProvider::from_options`] fails.
    pub fn with_model(
        api_key: impl Into<String>,
        default_model: Option<String>,
    ) -> Result<Self, BlazenError> {
        let api_key = api_key.into();
        let opts = crate::types::provider_options::FalOptions {
            base: crate::types::provider_options::ProviderOptions {
                api_key: if api_key.is_empty() {
                    None
                } else {
                    Some(api_key)
                },
                model: None,
                base_url: None,
            },
            endpoint: None,
            enterprise: false,
            auto_route_modality: true,
        };
        let provider = crate::providers::fal::FalProvider::from_options(opts)
            .map_err(|e| BlazenError::provider("fal", e.to_string()))?;

        let mut metadata = ProviderMetadata::new("fal", CapabilityKind::Tts);
        if let Some(m) = default_model.as_ref() {
            metadata = metadata.with_version(m.clone());
        }

        Ok(Self {
            inner: Arc::new(provider),
            default_model,
            metadata,
        })
    }
}

impl std::fmt::Debug for FalTtsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalTtsProvider")
            .field("metadata", &self.metadata)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

impl BaseProvider for FalTtsProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[async_trait]
impl TtsProvider for FalTtsProvider {
    async fn synthesize(&self, mut request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        // Inject the provider's default fal endpoint id when the caller
        // didn't supply one — matches the FalTtsAdapter pattern in
        // `blazen-uniffi/src/compute.rs`.
        if request.model.is_none()
            && let Some(m) = self.default_model.clone()
        {
            request = request.with_model(m);
        }
        AudioGeneration::text_to_speech(self.inner.as_ref(), request).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fal_tts_provider_metadata() {
        let provider = FalTtsProvider::with_model("test-key", Some("fal-ai/dia-tts".into()))
            .expect("explicit api_key should construct fine");
        assert_eq!(provider.provider_id(), "fal");
        assert_eq!(provider.capability(), CapabilityKind::Tts);
        assert_eq!(
            provider.metadata().version.as_deref(),
            Some("fal-ai/dia-tts")
        );
    }

    #[cfg(feature = "audio-tts-piper")]
    #[test]
    fn piper_voice_id_parses_three_segments() {
        assert_eq!(
            piper_voice_id_to_hf_path("en_US-amy-medium"),
            Some("en/en_US/amy/medium/en_US-amy-medium".to_owned())
        );
        assert_eq!(piper_voice_id_to_hf_path("invalid"), None);
        assert_eq!(piper_voice_id_to_hf_path("only-two"), None);
    }

    #[cfg(feature = "audio-tts-bark")]
    #[test]
    fn bark_provider_metadata() {
        let p = BarkProvider::new();
        assert_eq!(p.provider_id(), "bark");
        assert_eq!(p.capability(), CapabilityKind::Tts);
    }

    #[cfg(feature = "audio-tts-f5")]
    #[test]
    fn f5_provider_metadata() {
        let p = F5Provider::new();
        assert_eq!(p.provider_id(), "f5-tts");
        assert_eq!(p.capability(), CapabilityKind::Tts);
    }

    #[cfg(feature = "audio-tts-spark")]
    #[test]
    fn spark_provider_metadata() {
        let p = SparkTtsProvider::new(None, None, None);
        assert_eq!(p.provider_id(), "spark-tts");
        assert_eq!(p.capability(), CapabilityKind::Tts);
    }
}
