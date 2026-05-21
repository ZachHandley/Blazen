//! JavaScript bindings for the local TTS provider (`any-tts`).
//!
//! Exposes [`JsTtsProvider`] as a NAPI class with a factory constructor.
//! The underlying engine is `blazen-audio-tts`, which wraps the
//! `any-tts` crate (Kokoro-82M, `VibeVoice`, Qwen3-TTS). Without the
//! `engine` feature on `blazen-audio-tts`, the provider can still be
//! constructed but synthesis calls return `EngineNotAvailable`.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::{AnyTtsBackend, DynTtsProvider, TtsModel, TtsOptions};

// ---------------------------------------------------------------------------
// JsTtsModel
// ---------------------------------------------------------------------------

/// Which underlying TTS model to load. Maps onto
/// [`blazen_llm::TtsModel`].
#[napi(string_enum)]
#[derive(Debug, Clone, Copy)]
pub enum JsTtsModel {
    /// Kokoro-82M (default; small, CPU-friendly).
    Kokoro82m,
    /// VibeVoice-1.5B (Microsoft).
    VibeVoice,
    /// Qwen3-TTS-12Hz-1.7B (`CustomVoice` variant).
    Qwen3Tts,
}

impl From<JsTtsModel> for TtsModel {
    fn from(value: JsTtsModel) -> Self {
        match value {
            JsTtsModel::Kokoro82m => Self::Kokoro82m,
            JsTtsModel::VibeVoice => Self::VibeVoice,
            JsTtsModel::Qwen3Tts => Self::Qwen3Tts,
        }
    }
}

// ---------------------------------------------------------------------------
// JsTtsOptions
// ---------------------------------------------------------------------------

/// Options for the local TTS backend.
///
/// All fields are optional. `model` selects the backend (defaults to
/// Kokoro-82M); `voice` selects the speaker preset.
///
/// ```javascript
/// const provider = TtsProvider.create({
///   model: "kokoro82m",
///   voice: "af_bella",
/// });
/// ```
#[napi(object)]
pub struct JsTtsOptions {
    /// TTS model to load. Defaults to `"kokoro82m"`.
    pub model: Option<JsTtsModel>,
    /// Voice / speaker preset name.
    pub voice: Option<String>,
    /// Language ISO 639-1 code (e.g. `"en"`, `"ja"`).
    pub language: Option<String>,
    /// Output audio sample rate in Hz.
    #[napi(js_name = "sampleRate")]
    pub sample_rate: Option<u32>,
    /// Path to cache downloaded model weights.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

impl From<JsTtsOptions> for TtsOptions {
    fn from(val: JsTtsOptions) -> Self {
        Self {
            model: val.model.map(Into::into),
            voice: val.voice,
            language: val.language,
            sample_rate: val.sample_rate,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
            ..TtsOptions::default()
        }
    }
}

// ---------------------------------------------------------------------------
// JsTtsProvider NAPI class
// ---------------------------------------------------------------------------

/// A local TTS provider backed by `any-tts`.
///
/// ```javascript
/// const provider = TtsProvider.create({
///   model: "kokoro82m",
///   voice: "af_bella",
/// });
/// ```
#[napi(js_name = "TtsProvider")]
pub struct JsTtsProvider {
    // Held to keep the backend alive for the lifetime of the JS object —
    // future JS-side synthesize methods will reach for this; today the
    // ModelManager binding drives synthesis through its own typed handle.
    #[allow(dead_code)]
    inner: Arc<DynTtsProvider>,
    model_str: String,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsTtsProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new TTS provider.
    #[napi(factory)]
    pub fn create(options: Option<JsTtsOptions>) -> Result<Self> {
        let opts: TtsOptions = options.map(Into::into).unwrap_or_default();
        let model_str = opts.model.unwrap_or_default().as_str().to_owned();
        let backend = AnyTtsBackend::from_options(opts)
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(Self {
            inner: Arc::new(DynTtsProvider::erase(backend)),
            model_str,
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// The configured model kind, as a string (`"kokoro"`, `"vibevoice"`, `"qwen3_tts"`).
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.model_str.clone()
    }

    /// Whether the engine feature is compiled in. When the `anytts`
    /// feature is on, this returns `true` — the provider can be
    /// constructed regardless of the runtime model-load outcome.
    #[napi(js_name = "engineAvailable", getter)]
    pub fn engine_available(&self) -> bool {
        true
    }
}
