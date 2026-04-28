//! JavaScript bindings for the local Piper TTS provider.
//!
//! Exposes [`JsPiperProvider`] as a NAPI class with a factory constructor.
//!
//! Piper is a fast, on-device neural text-to-speech engine. The actual
//! audio synthesis surface (`AudioGeneration::text_to_speech`) is not yet
//! implemented in [`blazen_audio_piper::PiperProvider`] -- the upstream
//! crate currently validates options and stages the engine handle for the
//! Phase 9 wiring. Once that lands, `textToSpeech` will be added here.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::{PiperOptions, PiperProvider};

// ---------------------------------------------------------------------------
// JsPiperOptions
// ---------------------------------------------------------------------------

/// Options for the local Piper TTS backend.
///
/// All fields are optional. `modelId` selects the voice (e.g.
/// `"en_US-amy-medium"`); when `null`, callers must set it before
/// synthesis can run.
///
/// ```javascript
/// const provider = PiperProvider.create({
///   modelId: "en_US-amy-medium",
///   sampleRate: 22050,
/// });
/// ```
#[napi(object)]
pub struct JsPiperOptions {
    /// Piper voice model identifier.
    #[napi(js_name = "modelId")]
    pub model_id: Option<String>,
    /// Speaker ID for multi-speaker models.
    #[napi(js_name = "speakerId")]
    pub speaker_id: Option<u32>,
    /// Output audio sample rate in Hz.
    #[napi(js_name = "sampleRate")]
    pub sample_rate: Option<u32>,
    /// Path to cache downloaded voice models.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

impl From<JsPiperOptions> for PiperOptions {
    fn from(val: JsPiperOptions) -> Self {
        Self {
            model_id: val.model_id,
            speaker_id: val.speaker_id,
            sample_rate: val.sample_rate,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
        }
    }
}

// ---------------------------------------------------------------------------
// JsPiperProvider NAPI class
// ---------------------------------------------------------------------------

/// A local Piper TTS provider.
///
/// ```javascript
/// const provider = PiperProvider.create({
///   modelId: "en_US-amy-medium",
/// });
/// ```
#[napi(js_name = "PiperProvider")]
pub struct JsPiperProvider {
    inner: Arc<PiperProvider>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsPiperProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new Piper provider.
    #[napi(factory)]
    pub fn create(options: Option<JsPiperOptions>) -> Result<Self> {
        let opts: PiperOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                PiperProvider::from_options(opts)
                    .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the configured voice model identifier, if any.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> Option<String> {
        self.inner.model_id().map(str::to_owned)
    }

    /// Whether the engine feature is compiled in. When `false`,
    /// synthesis methods will return errors.
    #[napi(js_name = "engineAvailable", getter)]
    pub fn engine_available(&self) -> bool {
        self.inner.engine_available()
    }
}
