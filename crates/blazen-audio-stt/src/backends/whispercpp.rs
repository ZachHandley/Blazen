//! Local whisper.cpp [`SttBackend`] implementation.
//!
//! Ports the code from `crates/blazen-audio-whispercpp` into the
//! multi-backend `blazen-audio-stt` shape. The legacy crate remains in
//! place until the cutover wave (W11) so existing call sites keep
//! compiling.
//!
//! When the `whispercpp` feature is enabled this module compiles in the
//! `whisper-rs` Rust bindings to whisper.cpp and `blazen-model-cache` for
//! GGML weight downloads.
//!
//! Platform-specific acceleration features (`cuda`, `metal`, `coreml`) on
//! this crate are intentional no-ops at the library level — see the
//! `Cargo.toml` comments for the rationale and the override recipe.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::SttError;
use crate::traits::{SttBackend, TranscriptionResult, TranscriptionSegment};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// The Whisper model size variants hosted in the `ggerganov/whisper.cpp`
/// `HuggingFace` repository.
///
/// Each variant maps to a specific GGML file in that repository; larger
/// models produce more accurate transcriptions at higher memory / compute
/// cost.
///
/// | Model    | Parameters | English-only WER | Multilingual WER | RAM  |
/// |----------|-----------|-------------------|------------------|------|
/// | Tiny     | 39M       | ~8%               | ~12%             | ~1GB |
/// | Base     | 74M       | ~6%               | ~10%             | ~1GB |
/// | Small    | 244M      | ~4%               | ~7%              | ~2GB |
/// | Medium   | 769M      | ~3%               | ~5%              | ~5GB |
/// | LargeV3  | 1.5B      | ~2%               | ~3%              | ~10GB|
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum WhisperModel {
    /// Fastest, lowest accuracy (~39M parameters).
    Tiny,
    /// Fast, moderate accuracy (~74M parameters).
    Base,
    /// Good balance of speed and accuracy (~244M parameters).
    #[default]
    Small,
    /// High accuracy, slower (~769M parameters).
    Medium,
    /// Highest accuracy, most resource-intensive (~1.5B parameters).
    LargeV3,
}

impl WhisperModel {
    /// The `HuggingFace` repository ID for the GGML weights.
    ///
    /// All sizes share the single `ggerganov/whisper.cpp` repository — the
    /// specific file (`ggml-tiny.bin`, `ggml-base.bin`, ...) is selected
    /// by [`Self::as_ggml_filename`].
    #[must_use]
    pub const fn as_model_id(&self) -> &'static str {
        "ggerganov/whisper.cpp"
    }

    /// The GGML filename in the `HuggingFace` repository for this size.
    #[must_use]
    pub const fn as_ggml_filename(&self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
            Self::LargeV3 => "ggml-large-v3.bin",
        }
    }
}

impl std::fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Tiny => "tiny",
            Self::Base => "base",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::LargeV3 => "large-v3",
        };
        f.write_str(name)
    }
}

/// Construction-time options for [`WhisperCppBackend`].
///
/// All fields default to a sensible CPU build of the `Small` model with
/// auto-detect language. Override fields with struct-update syntax.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhisperCppOptions {
    /// Which Whisper model size to use. Defaults to [`WhisperModel::Small`].
    #[serde(default)]
    pub model: WhisperModel,

    /// Hardware device specifier (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    /// `None` is treated as CPU.
    pub device: Option<String>,

    /// ISO 639-1 language code (e.g. `"en"`). `None` lets whisper.cpp
    /// auto-detect the spoken language.
    pub language: Option<String>,

    /// Enable speaker diarization (`whisper.cpp` tinydiarize models).
    /// Currently advisory — the backend does not branch on this flag yet.
    pub diarize: Option<bool>,

    /// Optional override for the GGML cache directory. Defaults to the
    /// `blazen-model-cache` default (`$BLAZEN_CACHE_DIR` or
    /// `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Local speech-to-text backend powered by
/// [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp).
///
/// Construct with [`WhisperCppBackend::new`]; the GGML model is downloaded
/// and the `WhisperContext` is created lazily on the first
/// [`SttBackend::transcribe`] call (or eagerly via [`AudioBackend::load`]).
pub struct WhisperCppBackend {
    /// Stable backend identifier, e.g. `"whispercpp:small"`.
    id: String,
    /// Full options preserved for transcription calls.
    options: WhisperCppOptions,
    /// Lazily-loaded whisper context. `WhisperContext` is not `Clone`, so
    /// it lives behind an inner `Arc` that we clone out of the read lock
    /// for each transcription call. This lets blocking inference (run on
    /// `spawn_blocking`) own a clone of the handle without holding the
    /// async read guard across the blocking boundary.
    engine: Arc<RwLock<Option<Arc<whisper_rs::WhisperContext>>>>,
}

impl std::fmt::Debug for WhisperCppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `WhisperContext` is not `Debug`; render the safely-printable
        // fields only and use `finish_non_exhaustive` to flag the
        // intentional omission of the lazily-loaded engine handle.
        f.debug_struct("WhisperCppBackend")
            .field("id", &self.id)
            .field("model", &self.options.model)
            .field("device", &self.options.device)
            .field("language", &self.options.language)
            .finish_non_exhaustive()
    }
}

impl WhisperCppBackend {
    /// Build a new backend from options. Validates the supplied option
    /// values but does **not** download weights or create the
    /// `WhisperContext` — that happens on first
    /// [`SttBackend::transcribe`] call (or on an explicit
    /// [`AudioBackend::load`]).
    ///
    /// # Errors
    ///
    /// Returns [`SttError::InvalidOptions`] when `device` or `language`
    /// are present but empty.
    pub fn new(options: WhisperCppOptions) -> Result<Self, SttError> {
        if let Some(ref device) = options.device
            && device.is_empty()
        {
            return Err(SttError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }
        if let Some(ref lang) = options.language
            && lang.is_empty()
        {
            return Err(SttError::InvalidOptions(
                "language must not be empty when specified".into(),
            ));
        }
        let id = format!("whispercpp:{}", options.model);
        Ok(Self {
            id,
            options,
            engine: Arc::new(RwLock::new(None)),
        })
    }

    /// The model size that was configured at construction time.
    #[must_use]
    pub const fn model(&self) -> WhisperModel {
        self.options.model
    }

    /// The `HuggingFace` repository ID for the configured model.
    #[must_use]
    pub const fn model_id(&self) -> &'static str {
        self.options.model.as_model_id()
    }

    /// The GGML filename for the configured model.
    #[must_use]
    pub const fn model_filename(&self) -> &'static str {
        self.options.model.as_ggml_filename()
    }

    /// The configured device string, if any.
    #[must_use]
    pub fn device_str(&self) -> Option<&str> {
        self.options.device.as_deref()
    }

    /// Return the loaded engine, loading it on first use.
    ///
    /// Uses a double-checked `RwLock` pattern: the fast path takes a read
    /// lock, clones the existing `Arc<WhisperContext>` and returns. If
    /// not yet loaded, drops the read lock, acquires a write lock,
    /// re-checks (another task may have loaded concurrently), and finally
    /// builds the engine.
    async fn get_or_load_engine(&self) -> Result<Arc<whisper_rs::WhisperContext>, SttError> {
        {
            let guard = self.engine.read().await;
            if let Some(ctx) = guard.as_ref() {
                return Ok(Arc::clone(ctx));
            }
        }
        let mut guard = self.engine.write().await;
        if guard.is_none() {
            let ctx = build_engine(&self.options).await?;
            *guard = Some(Arc::new(ctx));
        }
        let ctx = guard.as_ref().expect("engine loaded above");
        Ok(Arc::clone(ctx))
    }

    /// Synchronous whisper inference — runs on a blocking thread.
    fn run_whisper(
        ctx: &whisper_rs::WhisperContext,
        raw_bytes: &[u8],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        let samples = Self::decode_wav_to_f32(raw_bytes)?;

        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Transcription(e.to_string()))?;

        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

        if let Some(lang) = language {
            params.set_language(Some(lang));
        } else {
            params.set_language(None);
        }

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, &samples)
            .map_err(|e| SttError::Transcription(e.to_string()))?;

        let n_segments = state.full_n_segments();

        let mut full_text = String::new();
        let mut segments = Vec::with_capacity(usize::try_from(n_segments).unwrap_or(0));

        for i in 0..n_segments {
            let seg = state
                .get_segment(i)
                .ok_or_else(|| SttError::Transcription(format!("segment {i} out of bounds")))?;

            let text = seg
                .to_str()
                .map_err(|e| SttError::Transcription(format!("segment {i} text: {e}")))?;

            // whisper.cpp returns timestamps in centiseconds; convert to ms.
            segments.push(TranscriptionSegment {
                start_ms: seg.start_timestamp() * 10,
                end_ms: seg.end_timestamp() * 10,
                text: text.to_string(),
            });

            full_text.push_str(text);
        }

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
    /// `[-1.0, 1.0]`. This is the standard Whisper input format.
    fn decode_wav_to_f32(raw_bytes: &[u8]) -> Result<Vec<f32>, SttError> {
        if raw_bytes.len() < 44 {
            return Err(SttError::Transcription(
                "audio file too small to be a valid WAV".into(),
            ));
        }
        if &raw_bytes[..4] != b"RIFF" || &raw_bytes[8..12] != b"WAVE" {
            return Err(SttError::Transcription(
                "audio file is not a valid WAV (missing RIFF/WAVE header)".into(),
            ));
        }

        let pcm_data = &raw_bytes[44..];
        let sample_count = pcm_data.len() / 2;
        let mut integer_samples = Vec::with_capacity(sample_count);

        for chunk in pcm_data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            integer_samples.push(sample);
        }

        let mut float_output = vec![0.0f32; integer_samples.len()];
        whisper_rs::convert_integer_to_float_audio(&integer_samples, &mut float_output)
            .map_err(|e| SttError::Transcription(format!("audio conversion failed: {e}")))?;

        Ok(float_output)
    }
}

#[async_trait]
impl AudioBackend for WhisperCppBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "stt"
    }

    async fn load(&self) -> Result<(), AudioError> {
        let _ = self
            .get_or_load_engine()
            .await
            .map_err(|e| AudioError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn unload(&self) -> Result<(), AudioError> {
        let mut guard = self.engine.write().await;
        *guard = None;
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        self.engine.read().await.is_some()
    }
}

#[async_trait]
impl SttBackend for WhisperCppBackend {
    async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        // Read the WAV bytes before triggering the (potentially expensive)
        // model download — if the input file is missing the user gets a
        // fast Io error instead of waiting on a 500 MB GGML download.
        let raw_bytes = tokio::fs::read(audio_path).await?;

        let ctx = self.get_or_load_engine().await?;
        let lang = language
            .map(String::from)
            .or_else(|| self.options.language.clone());

        tokio::task::spawn_blocking(move || Self::run_whisper(&ctx, &raw_bytes, lang.as_deref()))
            .await
            .map_err(|e| SttError::Transcription(format!("join error: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Engine construction helpers
// ---------------------------------------------------------------------------

/// Download (or locate in cache) the GGML weights and build a fresh
/// `whisper_rs::WhisperContext`.
async fn build_engine(opts: &WhisperCppOptions) -> Result<whisper_rs::WhisperContext, SttError> {
    let model_path = resolve_model_path(opts).await?;

    tracing::info!(
        model = %opts.model,
        path = %model_path.display(),
        "loading whisper.cpp model"
    );

    let params = whisper_rs::WhisperContextParameters::default();
    whisper_rs::WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| SttError::ModelLoad("model path contains invalid UTF-8".into()))?,
        params,
    )
    .map_err(|e| SttError::ModelLoad(e.to_string()))
}

/// Resolve the GGML model file path, downloading via `blazen-model-cache`
/// if it isn't already in the cache.
async fn resolve_model_path(opts: &WhisperCppOptions) -> Result<PathBuf, SttError> {
    let cache = if let Some(ref dir) = opts.cache_dir {
        blazen_model_cache::ModelCache::with_dir(dir)
    } else {
        blazen_model_cache::ModelCache::new().map_err(|e| SttError::ModelLoad(e.to_string()))?
    };

    let repo_id = opts.model.as_model_id();
    let filename = opts.model.as_ggml_filename();

    tracing::info!(repo = repo_id, file = filename, "downloading whisper model");

    cache
        .download(repo_id, filename, None)
        .await
        .map_err(|e| SttError::ModelLoad(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Options / model-enum surface ---------------------------------

    #[test]
    fn default_model_is_small() {
        assert_eq!(WhisperModel::default(), WhisperModel::Small);
    }

    #[test]
    fn default_options_uses_small_model() {
        let opts = WhisperCppOptions::default();
        assert_eq!(opts.model, WhisperModel::Small);
        assert!(opts.device.is_none());
        assert!(opts.language.is_none());
        assert!(opts.diarize.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = WhisperCppOptions {
            model: WhisperModel::LargeV3,
            language: Some("en".into()),
            ..WhisperCppOptions::default()
        };
        assert_eq!(opts.model, WhisperModel::LargeV3);
        assert_eq!(opts.language.as_deref(), Some("en"));
        assert!(opts.device.is_none());
    }

    #[test]
    fn model_id_is_consistent() {
        let repo = "ggerganov/whisper.cpp";
        assert_eq!(WhisperModel::Tiny.as_model_id(), repo);
        assert_eq!(WhisperModel::Base.as_model_id(), repo);
        assert_eq!(WhisperModel::Small.as_model_id(), repo);
        assert_eq!(WhisperModel::Medium.as_model_id(), repo);
        assert_eq!(WhisperModel::LargeV3.as_model_id(), repo);
    }

    #[test]
    fn ggml_filenames_are_correct() {
        assert_eq!(WhisperModel::Tiny.as_ggml_filename(), "ggml-tiny.bin");
        assert_eq!(WhisperModel::Base.as_ggml_filename(), "ggml-base.bin");
        assert_eq!(WhisperModel::Small.as_ggml_filename(), "ggml-small.bin");
        assert_eq!(WhisperModel::Medium.as_ggml_filename(), "ggml-medium.bin");
        assert_eq!(
            WhisperModel::LargeV3.as_ggml_filename(),
            "ggml-large-v3.bin"
        );
    }

    #[test]
    fn display_impl() {
        assert_eq!(WhisperModel::Tiny.to_string(), "tiny");
        assert_eq!(WhisperModel::Base.to_string(), "base");
        assert_eq!(WhisperModel::Small.to_string(), "small");
        assert_eq!(WhisperModel::Medium.to_string(), "medium");
        assert_eq!(WhisperModel::LargeV3.to_string(), "large-v3");
    }

    #[test]
    fn serde_roundtrip_options() {
        let opts = WhisperCppOptions {
            model: WhisperModel::Medium,
            device: Some("cuda:0".into()),
            language: Some("ja".into()),
            diarize: Some(true),
            cache_dir: Some(PathBuf::from("/var/tmp/whisper-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: WhisperCppOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model, WhisperModel::Medium);
        assert_eq!(parsed.device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.language.as_deref(), Some("ja"));
        assert_eq!(parsed.diarize, Some(true));
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/var/tmp/whisper-cache"))
        );
    }

    #[test]
    fn serde_roundtrip_model_enum() {
        for model in [
            WhisperModel::Tiny,
            WhisperModel::Base,
            WhisperModel::Small,
            WhisperModel::Medium,
            WhisperModel::LargeV3,
        ] {
            let json = serde_json::to_string(&model).expect("serialize");
            let parsed: WhisperModel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, model);
        }
    }

    // ---- Backend constructor validation -------------------------------

    #[test]
    fn new_rejects_empty_device() {
        let opts = WhisperCppOptions {
            device: Some(String::new()),
            ..WhisperCppOptions::default()
        };
        let result = WhisperCppBackend::new(opts);
        assert!(matches!(result, Err(SttError::InvalidOptions(_))));
    }

    #[test]
    fn new_rejects_empty_language() {
        let opts = WhisperCppOptions {
            language: Some(String::new()),
            ..WhisperCppOptions::default()
        };
        let result = WhisperCppBackend::new(opts);
        assert!(matches!(result, Err(SttError::InvalidOptions(_))));
    }

    #[test]
    fn new_with_defaults_succeeds() {
        let backend = WhisperCppBackend::new(WhisperCppOptions::default())
            .expect("default options should validate");
        assert_eq!(backend.model(), WhisperModel::Small);
        assert_eq!(backend.model_id(), "ggerganov/whisper.cpp");
        assert_eq!(backend.model_filename(), "ggml-small.bin");
        assert_eq!(
            <WhisperCppBackend as AudioBackend>::id(&backend),
            "whispercpp:small"
        );
        assert_eq!(
            <WhisperCppBackend as AudioBackend>::provider_kind(&backend),
            "stt"
        );
    }

    #[test]
    fn new_with_large_v3() {
        let backend = WhisperCppBackend::new(WhisperCppOptions {
            model: WhisperModel::LargeV3,
            ..WhisperCppOptions::default()
        })
        .expect("LargeV3 options should validate");
        assert_eq!(backend.model(), WhisperModel::LargeV3);
        assert_eq!(backend.model_filename(), "ggml-large-v3.bin");
        assert_eq!(
            <WhisperCppBackend as AudioBackend>::id(&backend),
            "whispercpp:large-v3"
        );
    }

    #[test]
    fn new_accepts_valid_device() {
        let backend = WhisperCppBackend::new(WhisperCppOptions {
            device: Some("cuda:0".into()),
            ..WhisperCppOptions::default()
        })
        .expect("cuda:0 should validate");
        assert_eq!(backend.device_str(), Some("cuda:0"));
    }

    #[test]
    fn new_accepts_valid_language() {
        let backend = WhisperCppBackend::new(WhisperCppOptions {
            language: Some("en".into()),
            ..WhisperCppOptions::default()
        })
        .expect("en should validate");
        assert_eq!(backend.model(), WhisperModel::Small);
    }

    // ---- Engine-path tests (network + disk, ignored by default) ------
    //
    // Successful loads download a ~500MB model from HuggingFace, so mark
    // them `#[ignore]` and run explicitly via
    //   cargo test -p blazen-audio-stt --features whispercpp -- --ignored

    #[tokio::test]
    #[ignore = "downloads ~500MB whisper-small model from HuggingFace"]
    async fn engine_loads_small_model() {
        let backend = WhisperCppBackend::new(WhisperCppOptions::default())
            .expect("default options should validate");
        backend.load().await.expect("engine should load");
        assert!(backend.is_loaded().await);
    }

    // Live-models test: round-trip a 1-second sine wave at 16 kHz
    // through the real `ggerganov/whisper.cpp` tiny checkpoint.
    // Gated because it fetches ~75 MB of GGML weights on first run.
    //
    // Requires BOTH `whispercpp` (to compile the backend) and
    // `live-models` (to compile this test) features, e.g.
    //   cargo test -p blazen-audio-stt --features whispercpp,live-models \
    //       -- live_transcribe_tiny_sine_wave_16khz --nocapture
    #[cfg(feature = "live-models")]
    #[tokio::test]
    async fn live_transcribe_tiny_sine_wave_16khz() {
        let backend = WhisperCppBackend::new(WhisperCppOptions {
            model: WhisperModel::Tiny,
            language: Some("en".into()),
            ..WhisperCppOptions::default()
        })
        .expect("Tiny options should validate");

        backend.load().await.expect("engine should load");
        assert!(backend.is_loaded().await);

        // 1 second of a 440 Hz sine wave at 16 kHz, mono, amplitude 0.5.
        // The cast-precision-loss allow is fine here: `i` runs over
        // [0, 16_000) which is well within f32's exact-integer range
        // (<= 2^23 ≈ 1.6e7).
        let sample_rate: u32 = 16_000;
        let len = sample_rate as usize;
        let mut pcm: Vec<f32> = Vec::with_capacity(len);
        for i in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate as f32;
            pcm.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        // Encode as a minimal 44-byte-header RIFF/WAVE 16-bit PCM mono
        // file. `WhisperCppBackend::transcribe` takes a path on disk
        // (not raw PCM), so we materialize the WAV under repo-local
        // scratch (~/.cache/blazen/test-scratch/) — never /tmp.
        let mut wav: Vec<u8> = Vec::with_capacity(44 + len * 2);
        let data_len = u32::try_from(len * 2).expect("data_len fits u32");
        let riff_len = 36u32 + data_len;
        let byte_rate = sample_rate * 2; // mono * 2 bytes per sample

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&riff_len.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav.extend_from_slice(&1u16.to_le_bytes()); // 1 channel
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for sample in &pcm {
            let clamped = sample.clamp(-1.0, 1.0);
            #[allow(clippy::cast_possible_truncation)]
            let scaled = (clamped * f32::from(i16::MAX)) as i16;
            wav.extend_from_slice(&scaled.to_le_bytes());
        }

        let home = std::env::var("HOME").expect("HOME must be set for scratch dir");
        let scratch_dir = std::path::PathBuf::from(home)
            .join(".cache")
            .join("blazen")
            .join("test-scratch");
        tokio::fs::create_dir_all(&scratch_dir)
            .await
            .expect("create scratch dir");
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let wav_path = scratch_dir.join(format!("whispercpp-live-{pid}-{nanos}.wav"));
        tokio::fs::write(&wav_path, &wav)
            .await
            .expect("write scratch WAV");

        let result = backend.transcribe(&wav_path, Some("en")).await;

        // Best-effort cleanup before asserting so a failing assertion
        // doesn't leave the scratch WAV behind.
        let _ = tokio::fs::remove_file(&wav_path).await;

        let transcription = result.expect("transcribe should succeed");
        // Sine wave isn't real speech, so the model may emit empty
        // text or hallucinated tokens. Weak assertion: the call
        // returned Ok and didn't panic, and `.text` is a String we
        // can inspect (length is >= 0 trivially by type).
        let _: &String = &transcription.text;
        // If the model produced segments, each segment's text should
        // be valid UTF-8 (guaranteed by type) and have non-decreasing
        // timestamps — sanity-check ordering without asserting any
        // particular content.
        for seg in &transcription.segments {
            assert!(
                seg.end_ms >= seg.start_ms,
                "segment end_ms {} should be >= start_ms {}",
                seg.end_ms,
                seg.start_ms,
            );
        }
    }
}
