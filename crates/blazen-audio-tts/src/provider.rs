//! The [`TtsProvider`] type — local TTS via [`any-tts`](https://crates.io/crates/any-tts).
//!
//! Without the `engine` feature this crate compiles as a stub: the provider
//! can be constructed and its options inspected, but every `synthesize` call
//! returns [`TtsError::EngineNotAvailable`]. With the feature enabled the
//! provider lazy-loads the underlying any-tts model on first synthesis and
//! caches the loaded `Box<dyn TtsModel>` behind a `tokio::sync::Mutex`.

use std::fmt;
#[cfg(feature = "engine")]
use std::sync::Arc;

#[cfg(feature = "engine")]
use tokio::sync::Mutex;

use crate::{TtsModel, TtsOptions};

/// Error type for the local TTS backend.
#[derive(Debug)]
pub enum TtsError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The voice model could not be downloaded or loaded.
    ModelLoad(String),
    /// A synthesis operation failed.
    Synthesis(String),
    /// The `engine` feature is not enabled.
    EngineNotAvailable,
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "tts invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "tts model load failed: {msg}"),
            Self::Synthesis(msg) => write!(f, "tts synthesis failed: {msg}"),
            Self::EngineNotAvailable => write!(
                f,
                "tts engine not available: compile blazen-audio-tts with the `engine` feature"
            ),
        }
    }
}

impl std::error::Error for TtsError {}

#[cfg(feature = "engine")]
impl From<any_tts::TtsError> for TtsError {
    fn from(value: any_tts::TtsError) -> Self {
        Self::ModelLoad(value.to_string())
    }
}

/// Result of a successful synthesis call.
///
/// `wav_bytes` is a complete RIFF/WAV byte stream (16-bit PCM) that can be
/// written to disk or played back directly. `sample_rate_hz` is the rate of
/// the underlying PCM data (matches the WAV header).
#[derive(Debug, Clone)]
pub struct SynthesizedAudio {
    /// Complete WAV bytes (RIFF header + PCM data).
    pub wav_bytes: Vec<u8>,
    /// Sample rate in Hz of the underlying PCM stream.
    pub sample_rate_hz: u32,
    /// Approximate duration in seconds.
    pub duration_secs: f32,
}

/// A local TTS provider backed by `any-tts` (Kokoro-82M default).
///
/// Construct via [`TtsProvider::from_options`]. Synthesis is exposed
/// through [`TtsProvider::synthesize`]; the blazen-llm bridge layer wraps
/// that into the `AudioGeneration::text_to_speech` trait method.
pub struct TtsProvider {
    /// Resolved options (model, voice, language, …).
    options: TtsOptions,
    /// Lazy-loaded any-tts model handle. Held behind an async mutex so
    /// concurrent first-synthesis calls cooperatively wait for the load.
    #[cfg(feature = "engine")]
    model: Arc<Mutex<Option<Box<dyn any_tts::TtsModel + Send + Sync>>>>,
}

impl TtsProvider {
    /// Create a new provider from the given options.
    ///
    /// Performs cheap up-front validation; the underlying any-tts model is
    /// not loaded until the first call to [`Self::synthesize`].
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::InvalidOptions`] if a string field is present
    /// but empty.
    pub fn from_options(opts: TtsOptions) -> Result<Self, TtsError> {
        if let Some(ref voice) = opts.voice
            && voice.is_empty()
        {
            return Err(TtsError::InvalidOptions(
                "voice must not be empty when specified".into(),
            ));
        }
        if let Some(ref language) = opts.language
            && language.is_empty()
        {
            return Err(TtsError::InvalidOptions(
                "language must not be empty when specified".into(),
            ));
        }
        if let Some(rate) = opts.sample_rate
            && rate == 0
        {
            return Err(TtsError::InvalidOptions(
                "sample_rate must be > 0 when specified".into(),
            ));
        }

        Ok(Self {
            #[cfg(feature = "engine")]
            model: Arc::new(Mutex::new(None)),
            options: opts,
        })
    }

    /// The resolved options used at construction time.
    #[must_use]
    pub fn options(&self) -> &TtsOptions {
        &self.options
    }

    /// The model that this provider was configured to load.
    #[must_use]
    pub fn model_kind(&self) -> TtsModel {
        self.options.model.unwrap_or_default()
    }

    /// Whether the engine feature is compiled in.
    #[must_use]
    pub fn engine_available(&self) -> bool {
        cfg!(feature = "engine")
    }

    /// Synthesize speech for `text`, returning a complete WAV-encoded
    /// audio buffer.
    ///
    /// On first call this lazy-loads the underlying any-tts model and
    /// caches it for the lifetime of the provider.
    ///
    /// # Errors
    ///
    /// * [`TtsError::EngineNotAvailable`] when the `engine` feature is
    ///   not enabled.
    /// * [`TtsError::ModelLoad`] when the model weights cannot be fetched
    ///   or initialised.
    /// * [`TtsError::Synthesis`] when the model itself errors during
    ///   synthesis.
    pub async fn synthesize(&self, text: &str) -> Result<SynthesizedAudio, TtsError> {
        if text.is_empty() {
            return Err(TtsError::Synthesis(
                "synthesize called with empty text".into(),
            ));
        }
        self.synthesize_inner(text).await
    }

    #[cfg(not(feature = "engine"))]
    async fn synthesize_inner(&self, _text: &str) -> Result<SynthesizedAudio, TtsError> {
        Err(TtsError::EngineNotAvailable)
    }

    #[cfg(feature = "engine")]
    #[allow(clippy::cast_precision_loss)]
    async fn synthesize_inner(&self, text: &str) -> Result<SynthesizedAudio, TtsError> {
        use any_tts::{ModelType, SynthesisRequest, TtsConfig};

        // Choose voice: explicit > model default.
        let voice = self
            .options
            .voice
            .clone()
            .or_else(|| self.model_kind().default_voice().map(str::to_string));

        let language = self.options.language.clone();
        let model_kind = self.model_kind();
        let cache_dir = self.options.cache_dir.clone();

        // Lazy-load the model inside the mutex.
        let model = Arc::clone(&self.model);
        let text = text.to_string();
        let audio = tokio::task::spawn_blocking(move || -> Result<_, TtsError> {
            let mut guard = model.blocking_lock();
            if guard.is_none() {
                let model_type = match model_kind {
                    TtsModel::Kokoro82m => ModelType::Kokoro,
                    TtsModel::VibeVoice => ModelType::VibeVoice,
                    TtsModel::Qwen3Tts => ModelType::Qwen3Tts,
                };
                let mut config = TtsConfig::new(model_type);
                if let Some(dir) = cache_dir.as_ref() {
                    config = config.with_model_path(dir.to_string_lossy().into_owned());
                }
                let loaded = any_tts::load_model(config)?;
                *guard = Some(loaded);
            }
            // `guard` now holds Some(model); build the request and synthesize.
            let mut req = SynthesisRequest::new(text);
            if let Some(v) = voice {
                req = req.with_voice(v);
            }
            if let Some(l) = language {
                req = req.with_language(l);
            }
            let model_ref = guard.as_ref().expect("just-loaded model is Some");
            let audio = model_ref
                .synthesize(&req)
                .map_err(|e| TtsError::Synthesis(e.to_string()))?;
            Ok(audio)
        })
        .await
        .map_err(|e| TtsError::Synthesis(format!("synthesis task panicked: {e}")))??;

        let sample_rate_hz = audio.sample_rate;
        let duration_secs = if sample_rate_hz == 0 {
            0.0
        } else {
            audio.samples.len() as f32 / sample_rate_hz as f32
        };
        let wav_bytes = audio.get_wav();
        Ok(SynthesizedAudio {
            wav_bytes,
            sample_rate_hz,
            duration_secs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TtsOptions;

    #[tokio::test]
    async fn from_options_with_defaults() {
        let provider = TtsProvider::from_options(TtsOptions::default()).expect("should succeed");
        assert_eq!(provider.model_kind(), TtsModel::Kokoro82m);
    }

    #[tokio::test]
    async fn from_options_rejects_empty_voice() {
        let opts = TtsOptions {
            voice: Some(String::new()),
            ..TtsOptions::default()
        };
        assert!(TtsProvider::from_options(opts).is_err());
    }

    #[tokio::test]
    async fn from_options_rejects_zero_sample_rate() {
        let opts = TtsOptions {
            sample_rate: Some(0),
            ..TtsOptions::default()
        };
        assert!(TtsProvider::from_options(opts).is_err());
    }

    #[tokio::test]
    async fn synthesize_empty_text_rejected() {
        let provider = TtsProvider::from_options(TtsOptions::default()).unwrap();
        let err = provider.synthesize("").await.unwrap_err();
        assert!(matches!(err, TtsError::Synthesis(_)));
    }

    #[tokio::test]
    async fn synthesize_without_engine_feature_errors() {
        // Without the `engine` feature on this crate, every synthesize call
        // surfaces EngineNotAvailable. CI may or may not have the feature
        // active, so only assert when it's off.
        if cfg!(not(feature = "engine")) {
            let provider = TtsProvider::from_options(TtsOptions::default()).unwrap();
            let err = provider.synthesize("hello").await.unwrap_err();
            assert!(matches!(err, TtsError::EngineNotAvailable));
        }
    }

    #[test]
    fn engine_not_available_display() {
        let err = TtsError::EngineNotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("engine"), "should mention engine: {msg}");
    }

    #[tokio::test]
    async fn engine_available_reflects_feature() {
        let provider = TtsProvider::from_options(TtsOptions::default()).unwrap();
        if cfg!(feature = "engine") {
            assert!(provider.engine_available());
        } else {
            assert!(!provider.engine_available());
        }
    }
}
