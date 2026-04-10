//! The [`WhisperCppProvider`] type -- local speech-to-text via whisper.cpp.
//!
//! When the `engine` feature is enabled, this module loads a GGML model via
//! `whisper-rs` and runs fully-local transcription.  Without the feature, the
//! struct still compiles (for type-checking downstream crates) but all
//! transcription calls return [`WhisperError::EngineNotAvailable`].

use std::fmt;
#[cfg(feature = "engine")]
use std::path::PathBuf;
#[cfg(feature = "engine")]
use std::sync::Arc;

#[cfg(feature = "engine")]
use tokio::sync::OnceCell;

use crate::WhisperOptions;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error type for whisper.cpp operations.
#[derive(Debug)]
pub enum WhisperError {
    /// The `engine` feature is not enabled -- whisper-rs is not linked.
    EngineNotAvailable,
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// A transcription operation failed.
    Transcription(String),
    /// An I/O error (e.g. reading an audio file).
    Io(std::io::Error),
}

impl fmt::Display for WhisperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EngineNotAvailable => write!(
                f,
                "whisper.cpp engine not available -- build with the `engine` feature"
            ),
            Self::InvalidOptions(msg) => write!(f, "whisper.cpp invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "whisper.cpp model load failed: {msg}"),
            Self::Transcription(msg) => write!(f, "whisper.cpp transcription failed: {msg}"),
            Self::Io(e) => write!(f, "whisper.cpp I/O error: {e}"),
        }
    }
}

impl std::error::Error for WhisperError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for WhisperError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Result types (provider-level, independent of blazen-llm)
// ---------------------------------------------------------------------------

/// The result of a transcription operation.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// The full transcribed text.
    pub text: String,
    /// Time-aligned segments.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected or specified language code.
    pub language: Option<String>,
}

/// A single time-aligned segment within a transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// The transcribed text for this segment.
    pub text: String,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// A local speech-to-text provider backed by
/// [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp).
///
/// Constructed via [`WhisperCppProvider::from_options`].  When the `engine`
/// feature is active the provider defers the model download
/// (via `blazen-model-cache`) and context creation until the first
/// [`WhisperCppProvider::transcribe`] call, so validation-only callers do not
/// pay the ~500MB download cost.  Without the feature the provider is a
/// lightweight stub.
pub struct WhisperCppProvider {
    /// The resolved model size that was requested.
    model: crate::WhisperModel,
    /// Full options preserved for transcription calls.
    #[cfg_attr(not(feature = "engine"), allow(dead_code))]
    options: WhisperOptions,
    /// Lazily-loaded whisper context.  Populated on the first successful
    /// `transcribe()` call (only present with the `engine` feature).
    #[cfg(feature = "engine")]
    engine: Arc<OnceCell<Arc<whisper_rs::WhisperContext>>>,
}

impl WhisperCppProvider {
    /// Create a new provider from the given options.
    ///
    /// This only validates the supplied options -- it does **not** download
    /// the GGML model or create the `WhisperContext`.  Model loading is
    /// deferred until the first [`WhisperCppProvider::transcribe`] call so
    /// that validation-only code paths (and parallel unit tests) never pay
    /// the ~500MB download cost.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidOptions`] for bad option values.
    /// [`WhisperError::ModelLoad`] and [`WhisperError::EngineNotAvailable`]
    /// are produced from transcription methods, not from this constructor.
    // This fn is `async` to keep the API stable with any future async
    // validation work and with the existing call sites in `blazen-llm`.
    #[allow(clippy::unused_async)]
    pub async fn from_options(opts: WhisperOptions) -> Result<Self, WhisperError> {
        // --- Validate common options ---
        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(WhisperError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        if let Some(ref lang) = opts.language
            && lang.is_empty()
        {
            return Err(WhisperError::InvalidOptions(
                "language must not be empty when specified".into(),
            ));
        }

        #[cfg(feature = "engine")]
        {
            Ok(Self {
                model: opts.model,
                options: opts,
                engine: Arc::new(OnceCell::new()),
            })
        }

        #[cfg(not(feature = "engine"))]
        {
            Ok(Self {
                model: opts.model,
                options: opts,
            })
        }
    }

    /// Lazily initialize (or return the cached) whisper context.
    ///
    /// The first successful call downloads (or locates in cache) the GGML
    /// model and calls `WhisperContext::new_with_params`.  Subsequent calls
    /// return a clone of the cached `Arc<WhisperContext>`.
    #[cfg(feature = "engine")]
    async fn ensure_engine(&self) -> Result<Arc<whisper_rs::WhisperContext>, WhisperError> {
        self.engine
            .get_or_try_init(|| async {
                let model_path = Self::resolve_model_path(&self.options).await?;
                tracing::info!(
                    model = %self.options.model,
                    path = %model_path.display(),
                    "loading whisper.cpp model"
                );

                let params = whisper_rs::WhisperContextParameters::default();
                let ctx = whisper_rs::WhisperContext::new_with_params(
                    model_path.to_str().ok_or_else(|| {
                        WhisperError::ModelLoad("model path contains invalid UTF-8".into())
                    })?,
                    params,
                )
                .map_err(|e| WhisperError::ModelLoad(e.to_string()))?;

                Ok(Arc::new(ctx))
            })
            .await
            .map(Arc::clone)
    }

    /// Download (or locate in cache) the GGML model file.
    #[cfg(feature = "engine")]
    async fn resolve_model_path(opts: &WhisperOptions) -> Result<PathBuf, WhisperError> {
        let cache = if let Some(ref dir) = opts.cache_dir {
            blazen_model_cache::ModelCache::with_dir(dir)
        } else {
            blazen_model_cache::ModelCache::new()
                .map_err(|e| WhisperError::ModelLoad(e.to_string()))?
        };

        let repo_id = opts.model.as_model_id();
        let filename = opts.model.as_ggml_filename();

        tracing::info!(repo = repo_id, file = filename, "downloading whisper model");

        cache
            .download(repo_id, filename, None)
            .await
            .map_err(|e| WhisperError::ModelLoad(e.to_string()))
    }

    /// Transcribe an audio file (WAV, 16-bit PCM, mono, 16 kHz).
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the WAV file on disk.
    /// * `language` - Optional ISO 639-1 code. Overrides the language set in
    ///   options for this single call.  When `None`, uses the provider-level
    ///   language or lets whisper auto-detect.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::EngineNotAvailable`] when compiled without the
    /// `engine` feature.
    // Same reason as `from_options` above: the API must be async to keep a
    // stable signature across feature flags.
    #[cfg_attr(not(feature = "engine"), allow(clippy::unused_async))]
    pub async fn transcribe(
        &self,
        audio_path: &std::path::Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, WhisperError> {
        #[cfg(feature = "engine")]
        {
            self.transcribe_engine(audio_path, language).await
        }

        #[cfg(not(feature = "engine"))]
        {
            let _ = (audio_path, language);
            Err(WhisperError::EngineNotAvailable)
        }
    }

    /// Engine-backed transcription (only compiled with `engine` feature).
    #[cfg(feature = "engine")]
    async fn transcribe_engine(
        &self,
        audio_path: &std::path::Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, WhisperError> {
        use tokio::task;

        // Read the WAV file bytes on the async runtime, then hand off to
        // a blocking thread for the CPU-intensive whisper inference.
        // Validate the audio file exists/reads before triggering the
        // (potentially very expensive) model download.
        let raw_bytes = tokio::fs::read(audio_path).await?;

        // Only AFTER audio validation succeeds do we load the engine (which
        // may download a ~500MB model on first call).
        let ctx = self.ensure_engine().await?;
        let lang = language
            .map(String::from)
            .or_else(|| self.options.language.clone());

        task::spawn_blocking(move || Self::run_whisper(&ctx, &raw_bytes, lang.as_deref()))
            .await
            .map_err(|e| WhisperError::Transcription(format!("join error: {e}")))?
    }

    /// Synchronous whisper inference -- runs on a blocking thread.
    #[cfg(feature = "engine")]
    fn run_whisper(
        ctx: &whisper_rs::WhisperContext,
        raw_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, WhisperError> {
        // Decode the WAV into f32 samples (whisper expects 16 kHz mono f32).
        let samples = Self::decode_wav_to_f32(raw_bytes)?;

        let mut state = ctx
            .create_state()
            .map_err(|e| WhisperError::Transcription(e.to_string()))?;

        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

        // Language setting
        if let Some(lang) = language {
            params.set_language(Some(lang));
        } else {
            // Auto-detect
            params.set_language(None);
        }

        // Disable printing to stdout
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, &samples)
            .map_err(|e| WhisperError::Transcription(e.to_string()))?;

        // Collect segments via the 0.16 segment API
        let n_segments = state.full_n_segments();

        let mut full_text = String::new();
        let mut segments = Vec::with_capacity(usize::try_from(n_segments).unwrap_or(0));

        for i in 0..n_segments {
            let seg = state
                .get_segment(i)
                .ok_or_else(|| WhisperError::Transcription(format!("segment {i} out of bounds")))?;

            let text = seg
                .to_str()
                .map_err(|e| WhisperError::Transcription(format!("segment {i} text: {e}")))?;

            // whisper.cpp returns timestamps in centiseconds (100ths of a
            // second), convert to milliseconds.
            segments.push(TranscriptionSegment {
                start_ms: seg.start_timestamp() * 10,
                end_ms: seg.end_timestamp() * 10,
                text: text.to_string(),
            });

            full_text.push_str(text);
        }

        // Detect language from the whisper state.
        let detected_lang = if language.is_some() {
            language.map(String::from)
        } else {
            let lang_id = state.full_lang_id_from_state();
            whisper_rs::get_lang_str(lang_id).map(String::from)
        };

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            segments,
            language: detected_lang,
        })
    }

    /// Decode a WAV file (16-bit PCM, mono, 16 kHz) into f32 samples in
    /// [-1.0, 1.0].
    ///
    /// whisper.cpp / whisper-rs expects f32 samples at 16 kHz, mono channel.
    /// This is the standard format for Whisper input.
    #[cfg(feature = "engine")]
    fn decode_wav_to_f32(raw_bytes: &[u8]) -> Result<Vec<f32>, WhisperError> {
        // Minimal WAV parser: skip 44-byte header, read 16-bit LE PCM samples.
        // For production usage, a proper WAV parser (e.g. hound) would be
        // better, but whisper-rs also provides `convert_integer_to_float_audio`
        // which we can use.
        if raw_bytes.len() < 44 {
            return Err(WhisperError::Transcription(
                "audio file too small to be a valid WAV".into(),
            ));
        }

        // Check RIFF header
        if &raw_bytes[..4] != b"RIFF" || &raw_bytes[8..12] != b"WAVE" {
            return Err(WhisperError::Transcription(
                "audio file is not a valid WAV (missing RIFF/WAVE header)".into(),
            ));
        }

        // Read 16-bit PCM data after the header.
        // A real parser would walk the chunk list, but for the common case
        // the data chunk starts at offset 44.
        let pcm_data = &raw_bytes[44..];
        let sample_count = pcm_data.len() / 2;
        let mut integer_samples = Vec::with_capacity(sample_count);

        for chunk in pcm_data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            integer_samples.push(sample);
        }

        // Use whisper-rs helper to convert i16 -> f32
        let mut float_output = vec![0.0f32; integer_samples.len()];
        whisper_rs::convert_integer_to_float_audio(&integer_samples, &mut float_output)
            .map_err(|e| WhisperError::Transcription(format!("audio conversion failed: {e}")))?;

        Ok(float_output)
    }

    /// The model size that was configured at construction time.
    #[must_use]
    pub const fn model(&self) -> crate::WhisperModel {
        self.model
    }

    /// The `HuggingFace` repository ID for the configured model.
    #[must_use]
    pub const fn model_id(&self) -> &'static str {
        self.model.as_model_id()
    }

    /// The GGML filename for the configured model.
    #[must_use]
    pub const fn model_filename(&self) -> &'static str {
        self.model.as_ggml_filename()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WhisperOptions;

    // ---- Option validation (engine-independent) ------------------------
    //
    // These tests exercise the validation arm of `from_options` which runs
    // *before* any model loading, so they work regardless of whether the
    // `engine` feature is enabled.

    #[tokio::test]
    async fn from_options_rejects_empty_device() {
        let opts = WhisperOptions {
            device: Some(String::new()),
            ..WhisperOptions::default()
        };
        let result = WhisperCppProvider::from_options(opts).await;
        assert!(matches!(result, Err(WhisperError::InvalidOptions(_))));
    }

    #[tokio::test]
    async fn from_options_rejects_empty_language() {
        let opts = WhisperOptions {
            language: Some(String::new()),
            ..WhisperOptions::default()
        };
        let result = WhisperCppProvider::from_options(opts).await;
        assert!(matches!(result, Err(WhisperError::InvalidOptions(_))));
    }

    // ---- Stub-path tests (no `engine` feature) --------------------------
    //
    // Without the `engine` feature, `from_options` does not download or
    // load any model, so successful construction can be asserted directly.

    #[cfg(not(feature = "engine"))]
    #[tokio::test]
    async fn from_options_with_defaults() {
        use crate::WhisperModel;

        let opts = WhisperOptions::default();
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .expect("should succeed");
        assert_eq!(provider.model(), WhisperModel::Small);
        assert_eq!(provider.model_id(), "ggerganov/whisper.cpp");
        assert_eq!(provider.model_filename(), "ggml-small.bin");
    }

    #[cfg(not(feature = "engine"))]
    #[tokio::test]
    async fn from_options_with_large_v3() {
        use crate::WhisperModel;

        let opts = WhisperOptions {
            model: WhisperModel::LargeV3,
            ..WhisperOptions::default()
        };
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .expect("should succeed");
        assert_eq!(provider.model(), WhisperModel::LargeV3);
        assert_eq!(provider.model_filename(), "ggml-large-v3.bin");
    }

    #[cfg(not(feature = "engine"))]
    #[tokio::test]
    async fn from_options_accepts_valid_device() {
        use crate::WhisperModel;

        let opts = WhisperOptions {
            device: Some("cuda:0".into()),
            ..WhisperOptions::default()
        };
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .expect("should succeed");
        assert_eq!(provider.model(), WhisperModel::Small);
    }

    #[cfg(not(feature = "engine"))]
    #[tokio::test]
    async fn from_options_accepts_valid_language() {
        use crate::WhisperModel;

        let opts = WhisperOptions {
            language: Some("en".into()),
            ..WhisperOptions::default()
        };
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .expect("should succeed");
        assert_eq!(provider.model(), WhisperModel::Small);
    }

    /// Without the `engine` feature, transcription must return
    /// `EngineNotAvailable`.
    #[cfg(not(feature = "engine"))]
    #[tokio::test]
    async fn transcribe_without_engine_returns_error() {
        let opts = WhisperOptions::default();
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .expect("construction should succeed");
        let err = provider
            .transcribe(std::path::Path::new("/tmp/dummy.wav"), None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, WhisperError::EngineNotAvailable),
            "expected EngineNotAvailable, got: {err}"
        );
    }

    // ---- Engine-path tests (network + disk, ignored by default) ---------
    //
    // With `engine` enabled, successful `from_options` downloads a ~500MB
    // model from HuggingFace and loads it into whisper.cpp.  This is too
    // expensive to run in regular unit tests; mark them `#[ignore]` so
    // developers can run them explicitly with
    // `cargo test -p blazen-audio-whispercpp --features engine -- --ignored`.

    #[cfg(feature = "engine")]
    #[tokio::test]
    #[ignore = "downloads ~500MB whisper-small model from HuggingFace"]
    async fn engine_loads_small_model() {
        use crate::WhisperModel;

        let opts = WhisperOptions::default();
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .expect("engine should load small model");
        assert_eq!(provider.model(), WhisperModel::Small);
    }
}
