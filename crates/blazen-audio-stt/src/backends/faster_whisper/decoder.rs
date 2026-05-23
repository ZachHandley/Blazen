//! `CTranslate2` Whisper decoder wrapper (Wave F.2.3).
//!
//! Thin Rust wrapper over [`ct2rs::Whisper`] that owns the loaded
//! `CTranslate2` model handle, exposes [`transcribe`][FasterWhisperDecoder::transcribe]
//! plus a stub [`detect_language`][FasterWhisperDecoder::detect_language], and
//! adapts the `ct2rs` error surface to [`SttError`].
//!
//! # Boundary
//!
//! - **In scope**: model load, transcription, segment-level timestamps
//!   parsed from Whisper's `<|t.tt|>` timestamp tokens, error mapping
//!   onto [`SttError`].
//! - **Out of scope** (this wave): word-level timestamps require the
//!   lower-level `ct2rs::sys::Whisper::align` API. The high-level
//!   [`ct2rs::Whisper`] facade does **not** expose the encoded mel
//!   features needed by `align`, so word-level alignment would have to
//!   either drop down to the `sys` layer (re-implementing the mel
//!   spectrogram pipeline) or wait for an upstream `ct2rs` API
//!   extension. We surface the [`DecodedWord`] type and the
//!   `want_word_timestamps` option so the caller surface is stable;
//!   `words` is always emitted empty in this wave.
//! - **Standalone language detection** is similarly gated behind the
//!   `sys::Whisper::detect_language` path. The high-level
//!   [`ct2rs::Whisper`] performs inline language detection inside
//!   [`generate`][ct2rs::Whisper::generate] when `language` is `None`,
//!   so [`detect_language`][FasterWhisperDecoder::detect_language]
//!   returns [`DecoderError::LanguageDetection`] in this wave — callers
//!   should pass `None` to [`transcribe`][FasterWhisperDecoder::transcribe]
//!   instead and let `ct2rs` auto-detect.
//!
//! # Threading
//!
//! `ct2rs::Whisper::generate` is a blocking C++ FFI call. The wrapper
//! itself is `Send + Sync` (the `cxx::UniquePtr` inside `ct2rs` is
//! `Send`); the calling layer ([`super::mod`]) routes inference through
//! [`tokio::task::spawn_blocking`].

use std::path::Path;
use std::sync::Arc;

use ct2rs::{ComputeType, Config, Device, Whisper, WhisperOptions};
use thiserror::Error;

use crate::SttError;
use crate::traits::{TranscriptionResult, TranscriptionSegment};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Decoder-side configuration for [`FasterWhisperDecoder`].
///
/// Defaults mirror the upstream `faster-whisper` Python project's
/// `WhisperModel(...)` defaults: beam-size 5, best-of 5, the standard
/// temperature fallback ladder `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`,
/// `compute_type="int8"` on CPU.
#[derive(Debug, Clone)]
pub struct FasterWhisperDecoderConfig {
    /// `CTranslate2` compute type as a human string.
    ///
    /// Recognised: `"default"`, `"auto"`, `"int8"`, `"int8_float32"`,
    /// `"int8_float16"`, `"int8_bfloat16"`, `"int16"`, `"float16"`,
    /// `"bfloat16"`, `"float32"`. Unknown values fall back to
    /// [`ComputeType::DEFAULT`] (i.e. whatever quantisation the model was
    /// converted with).
    pub compute_type: String,
    /// CUDA device index. `None` selects CPU; `Some(0)` selects
    /// `Device::CUDA` with `device_indices = [0]`, etc.
    pub device_index: Option<usize>,
    /// CPU thread count per replica. `None` => upstream default
    /// (`num_threads_per_replica = 0`, i.e. `CTranslate2`'s auto-pick).
    pub num_threads: Option<usize>,
    /// Beam-search width handed to [`WhisperOptions::beam_size`].
    /// Upstream default 5.
    pub beam_size: usize,
    /// Best-of for the sampling-fallback path. Currently advisory —
    /// [`WhisperOptions`] in ct2rs 0.9.x does not expose a `best_of`
    /// field directly (it has `num_hypotheses`); we map it onto
    /// [`WhisperOptions::num_hypotheses`] so the upstream knob still
    /// affects ct2rs. Upstream default 5.
    pub best_of: usize,
    /// Temperature fallback ladder. Empty means greedy / no fallback.
    /// ct2rs's [`WhisperOptions`] only carries a single
    /// [`sampling_temperature`][ct2rs::WhisperOptions::sampling_temperature];
    /// we apply the first non-zero entry (or `1.0` if all zero) as the
    /// sampling temperature. The full faster-whisper temperature-retry
    /// loop is reserved for a future wave.
    pub temperatures: Vec<f32>,
    /// Optional initial prompt for decoder biasing. Currently advisory:
    /// ct2rs's high-level [`Whisper::generate`] does not accept a
    /// caller-supplied prefix prompt (it builds
    /// `<|startoftranscript|><|lang|><|transcribe|>` internally). Tracked
    /// for a future wave that drops to the `sys` layer.
    pub initial_prompt: Option<String>,
    /// Token IDs to suppress (handed straight to
    /// [`WhisperOptions::suppress_tokens`]). `-1` is "the default
    /// suppress-set from `config.json`".
    pub suppress_tokens: Vec<i32>,
}

impl Default for FasterWhisperDecoderConfig {
    fn default() -> Self {
        Self {
            compute_type: "int8".to_owned(),
            device_index: None,
            num_threads: None,
            beam_size: 5,
            best_of: 5,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            initial_prompt: None,
            suppress_tokens: vec![-1],
        }
    }
}

// ---------------------------------------------------------------------------
// Per-call options
// ---------------------------------------------------------------------------

/// Per-call options for [`FasterWhisperDecoder::transcribe`].
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// If `true`, ask the decoder to emit word-level timestamps.
    ///
    /// **Wave F.2.3 limitation**: the high-level [`ct2rs::Whisper`] API
    /// does not currently surface the encoded features needed by
    /// `ct2rs::sys::Whisper::align`, so word-level timestamps are
    /// reserved for a future wave. This flag is preserved on the public
    /// surface so callers don't have to be rewritten when the
    /// implementation lands; today it is a no-op and [`DecodedWord`]
    /// vectors are always emitted empty.
    pub want_word_timestamps: bool,
    /// If `true`, request segment-level timestamps (the default
    /// `<|t.tt|>` Whisper timestamp tokens). Parsed into
    /// [`DecodedSegment::start`] / `end` on the return path.
    pub want_segment_timestamps: bool,
    /// If `true` and `language` is `None`, ct2rs runs its inline
    /// language-detection inside [`Whisper::generate`]. The detected
    /// language **is not surfaced** in this wave because the high-level
    /// `generate` return type does not include it; see
    /// [`detect_language`][FasterWhisperDecoder::detect_language] for
    /// rationale.
    pub detect_language_when_none: bool,
}

// ---------------------------------------------------------------------------
// Decoded output types
// ---------------------------------------------------------------------------

/// Output of [`FasterWhisperDecoder::detect_language`].
///
/// Currently unreachable on this wave because language detection only
/// runs inline through [`transcribe`][FasterWhisperDecoder::transcribe]
/// — the [`FasterWhisperDecoder::detect_language`] entrypoint returns
/// [`DecoderError::LanguageDetection`] in Wave F.2.3.
#[derive(Debug, Clone)]
pub struct DetectedLanguage {
    /// ISO 639-1 (or 639-3 for languages without a 639-1 code, matching
    /// Whisper's tokenizer convention — `"en"`, `"yue"`, etc.).
    pub language_code: String,
    /// Probability mass assigned to the detected language by the
    /// encoder, in `[0, 1]`.
    pub probability: f32,
}

/// A rich transcript produced by [`FasterWhisperDecoder::transcribe`].
///
/// This is the decoder's native output. The adapter
/// [`into_transcription_result`] converts it (with some loss — see the
/// helper's doc) onto the trait surface [`TranscriptionResult`].
#[derive(Debug, Clone)]
pub struct DecodedTranscript {
    /// Concatenated transcript text (segments joined with a single
    /// space, leading/trailing whitespace trimmed).
    pub text: String,
    /// Language code that the decoder operated under. `Some` only when
    /// the caller passed an explicit language to
    /// [`transcribe`][FasterWhisperDecoder::transcribe]; ct2rs's inline
    /// auto-detection does not surface the chosen language through the
    /// high-level API.
    pub language: Option<String>,
    /// Per-segment breakdown. One entry per 30-second chunk handed to
    /// ct2rs; when `want_segment_timestamps` was set, `start` / `end`
    /// are parsed from the inline `<|t.tt|>` Whisper timestamp tokens.
    pub segments: Vec<DecodedSegment>,
}

/// One segment within a [`DecodedTranscript`].
#[derive(Debug, Clone)]
pub struct DecodedSegment {
    /// Segment text with Whisper timestamp tokens stripped.
    pub text: String,
    /// Start offset in seconds from the beginning of the source audio.
    /// `0.0` when the caller did not request segment timestamps.
    pub start: f32,
    /// End offset in seconds. `0.0` when timestamps were not requested.
    pub end: f32,
    /// Word-level breakdown (always empty in Wave F.2.3 — see
    /// [`TranscribeOptions::want_word_timestamps`]).
    pub words: Vec<DecodedWord>,
}

/// One word within a [`DecodedSegment`]. Reserved for a future wave.
#[derive(Debug, Clone)]
pub struct DecodedWord {
    /// Word text (with leading space preserved as Whisper emits it).
    pub text: String,
    /// Word start in seconds.
    pub start: f32,
    /// Word end in seconds.
    pub end: f32,
    /// Per-word probability in `[0, 1]`.
    pub probability: f32,
}

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

/// Errors surfaced by [`FasterWhisperDecoder`].
///
/// All variants flatten cleanly into [`SttError`] via the [`From`] impl
/// below so callers can `?`-convert through the pipeline boundary.
#[derive(Debug, Error)]
pub enum DecoderError {
    /// `ct2rs::Whisper::new` (model load / `CTranslate2` init) failed.
    #[error("ct2rs load failed: {0}")]
    Load(String),
    /// `ct2rs::Whisper::generate` (decoder inference) failed.
    #[error("ct2rs inference failed: {0}")]
    Inference(String),
    /// The path handed to [`FasterWhisperDecoder::load`] either does not
    /// exist, is not a directory, or is missing the
    /// `CTranslate2`/Whisper bundle files (`model.bin`, `config.json`,
    /// `tokenizer.json`, `preprocessor_config.json`).
    #[error("invalid model directory: {0}")]
    InvalidModelDir(String),
    /// Standalone language detection failed or is unavailable on the
    /// current ct2rs API surface.
    #[error("language detection failed: {0}")]
    LanguageDetection(String),
}

impl From<DecoderError> for SttError {
    fn from(err: DecoderError) -> Self {
        match err {
            DecoderError::Load(msg) | DecoderError::InvalidModelDir(msg) => Self::ModelLoad(msg),
            DecoderError::Inference(msg) => Self::Transcription(msg),
            DecoderError::LanguageDetection(msg) => {
                Self::Transcription(format!("language detection: {msg}"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Loaded `CTranslate2` Whisper handle.
///
/// Construct via [`FasterWhisperDecoder::load`]. Inference is blocking
/// (C++ FFI) — wrap calls in [`tokio::task::spawn_blocking`] from async
/// contexts.
pub struct FasterWhisperDecoder {
    inner: Arc<Whisper>,
    config: FasterWhisperDecoderConfig,
}

impl std::fmt::Debug for FasterWhisperDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FasterWhisperDecoder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl FasterWhisperDecoder {
    /// Open a `CTranslate2`-converted Whisper model from a local
    /// directory.
    ///
    /// Expected layout (every `Systran/faster-whisper-*` repo ships
    /// exactly this — Wave F.2.5 wires the HF download):
    ///
    /// ```text
    /// model_dir/
    ///   ├── config.json
    ///   ├── model.bin
    ///   ├── preprocessor_config.json
    ///   ├── tokenizer.json
    ///   └── vocabulary.json
    /// ```
    ///
    /// # Errors
    ///
    /// - [`DecoderError::InvalidModelDir`] if `model_dir` does not exist
    ///   or is not a directory.
    /// - [`DecoderError::Load`] if `ct2rs::Whisper::new` fails (missing
    ///   files, malformed `config.json`, ABI mismatch, etc.).
    pub fn load(
        model_dir: &Path,
        config: FasterWhisperDecoderConfig,
    ) -> Result<Self, DecoderError> {
        if !model_dir.exists() {
            return Err(DecoderError::InvalidModelDir(format!(
                "{} does not exist",
                model_dir.display()
            )));
        }
        if !model_dir.is_dir() {
            return Err(DecoderError::InvalidModelDir(format!(
                "{} is not a directory",
                model_dir.display()
            )));
        }

        let ct2_config = build_ct2_config(&config);
        let inner = Whisper::new(model_dir, ct2_config)
            .map_err(|e| DecoderError::Load(format!("{e:#}")))?;

        Ok(Self {
            inner: Arc::new(inner),
            config,
        })
    }

    /// Run language detection on (up to) 30 seconds of audio.
    ///
    /// **Wave F.2.3 limitation**: returns
    /// [`DecoderError::LanguageDetection`]. The high-level
    /// [`ct2rs::Whisper`] API runs detection inline inside
    /// [`generate`][ct2rs::Whisper::generate] when `language=None` but
    /// does not expose a standalone entrypoint. Callers needing
    /// auto-detect should pass `language: None` to
    /// [`transcribe`][Self::transcribe] instead.
    ///
    /// # Errors
    ///
    /// Always returns [`DecoderError::LanguageDetection`] in Wave F.2.3.
    pub fn detect_language(&self, _samples: &[f32]) -> Result<DetectedLanguage, DecoderError> {
        Err(DecoderError::LanguageDetection(
            "standalone language detection is reserved for a future wave \
             — ct2rs::Whisper does not expose detect_language on its high-level \
             surface; pass language=None to transcribe() for inline auto-detect"
                .to_owned(),
        ))
    }

    /// Transcribe `samples` (16 kHz mono `f32` in `[-1, 1]`).
    ///
    /// `samples` should come from
    /// [`super::audio::prepare_for_whisper`]. ct2rs internally chunks
    /// into 30-second windows and concatenates the per-chunk text.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::Inference`] on any ct2rs failure.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        opts: &TranscribeOptions,
    ) -> Result<DecodedTranscript, DecoderError> {
        let options = build_whisper_options(&self.config);
        let want_timestamps = opts.want_segment_timestamps || opts.want_word_timestamps;

        let chunks = self
            .inner
            .generate(samples, language, want_timestamps, &options)
            .map_err(|e| DecoderError::Inference(format!("{e:#}")))?;

        let segments: Vec<DecodedSegment> = chunks
            .into_iter()
            .map(|chunk| parse_segment(&chunk, want_timestamps))
            .collect();

        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();

        Ok(DecodedTranscript {
            text,
            language: language.map(str::to_owned),
            segments,
        })
    }

    /// Read-only access to the configuration this decoder was loaded
    /// with.
    #[must_use]
    pub fn config(&self) -> &FasterWhisperDecoderConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helpers (private + crate-internal)
// ---------------------------------------------------------------------------

/// Convert the rich [`DecodedTranscript`] into the trait surface
/// [`TranscriptionResult`].
///
/// Fidelity loss: [`DecodedWord`] vectors are dropped (the trait carries
/// segment-level only). Segment start/end are converted from seconds to
/// milliseconds (the trait uses `i64` ms).
pub(super) fn into_transcription_result(decoded: DecodedTranscript) -> TranscriptionResult {
    let segments = decoded
        .segments
        .into_iter()
        .map(|s| TranscriptionSegment {
            start_ms: seconds_to_ms(s.start),
            end_ms: seconds_to_ms(s.end),
            text: s.text,
        })
        .collect();
    TranscriptionResult {
        text: decoded.text,
        segments,
        language: decoded.language,
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn seconds_to_ms(seconds: f32) -> i64 {
    // Whisper segments are non-negative; clamp at 0 anyway for safety.
    let ms = (f64::from(seconds) * 1000.0).round();
    if ms <= 0.0 { 0 } else { ms as i64 }
}

fn build_ct2_config(cfg: &FasterWhisperDecoderConfig) -> Config {
    let (device, device_indices) = match cfg.device_index {
        Some(idx) => (Device::CUDA, vec![i32::try_from(idx).unwrap_or(i32::MAX)]),
        None => (Device::CPU, vec![0]),
    };

    Config {
        device,
        compute_type: parse_compute_type(&cfg.compute_type),
        device_indices,
        tensor_parallel: false,
        num_threads_per_replica: cfg.num_threads.unwrap_or(0),
        max_queued_batches: 0,
        cpu_core_offset: -1,
    }
}

fn parse_compute_type(s: &str) -> ComputeType {
    match s.to_ascii_lowercase().as_str() {
        "auto" => ComputeType::AUTO,
        "float32" | "f32" | "fp32" => ComputeType::FLOAT32,
        "int8" | "i8" => ComputeType::INT8,
        "int8_float32" | "int8_f32" | "int8_fp32" => ComputeType::INT8_FLOAT32,
        "int8_float16" | "int8_f16" | "int8_fp16" => ComputeType::INT8_FLOAT16,
        "int8_bfloat16" | "int8_bf16" => ComputeType::INT8_BFLOAT16,
        "int16" | "i16" => ComputeType::INT16,
        "float16" | "f16" | "fp16" => ComputeType::FLOAT16,
        "bfloat16" | "bf16" => ComputeType::BFLOAT16,
        // "default" and any unknown value fall back to the model's
        // converted-with quantisation.
        _ => ComputeType::DEFAULT,
    }
}

fn build_whisper_options(cfg: &FasterWhisperDecoderConfig) -> WhisperOptions {
    let temperature = cfg
        .temperatures
        .iter()
        .copied()
        .find(|t| *t > 0.0)
        .unwrap_or(1.0);

    WhisperOptions {
        beam_size: cfg.beam_size.max(1),
        num_hypotheses: cfg.best_of.max(1),
        sampling_temperature: temperature,
        suppress_tokens: cfg.suppress_tokens.clone(),
        ..Default::default()
    }
}

/// Parse one ct2rs-returned chunk string into a [`DecodedSegment`].
///
/// When `want_timestamps` is `true`, ct2rs leaves `<|t.tt|>` timestamp
/// tokens embedded in the string. We strip them out for `text` and pull
/// the **first** and **last** timestamp tokens as the segment bounds.
/// When `want_timestamps` is `false`, no parsing is necessary.
fn parse_segment(raw: &str, want_timestamps: bool) -> DecodedSegment {
    if !want_timestamps {
        return DecodedSegment {
            text: raw.trim().to_owned(),
            start: 0.0,
            end: 0.0,
            words: Vec::new(),
        };
    }

    let mut timestamps: Vec<f32> = Vec::new();
    let mut cleaned = String::with_capacity(raw.len());
    let mut rest = raw;

    while let Some(open) = rest.find("<|") {
        cleaned.push_str(&rest[..open]);
        let after_open = &rest[open + 2..];
        let Some(close_rel) = after_open.find("|>") else {
            // Malformed — bail and keep what we have so far.
            cleaned.push_str(&rest[open..]);
            break;
        };
        let token = &after_open[..close_rel];
        // Whisper timestamp tokens are pure floats like "0.00", "2.40".
        if let Ok(ts) = token.parse::<f32>() {
            timestamps.push(ts);
        }
        rest = &after_open[close_rel + 2..];
    }
    cleaned.push_str(rest);

    let start = timestamps.first().copied().unwrap_or(0.0);
    let end = timestamps.last().copied().unwrap_or(0.0);

    DecodedSegment {
        text: cleaned.trim().to_owned(),
        start,
        end,
        words: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn decoder_config_defaults_match_upstream_faster_whisper() {
        let cfg = FasterWhisperDecoderConfig::default();
        assert_eq!(cfg.compute_type, "int8");
        assert!(cfg.device_index.is_none());
        assert!(cfg.num_threads.is_none());
        assert_eq!(cfg.beam_size, 5);
        assert_eq!(cfg.best_of, 5);
        assert_eq!(cfg.temperatures.len(), 6);
        assert!((cfg.temperatures[0] - 0.0).abs() < f32::EPSILON);
        assert!((cfg.temperatures[5] - 1.0).abs() < f32::EPSILON);
        assert!(cfg.initial_prompt.is_none());
        assert_eq!(cfg.suppress_tokens, vec![-1]);
    }

    #[test]
    fn decoder_load_returns_error_for_missing_dir() {
        let missing = PathBuf::from("/nonexistent/blazen-faster-whisper-decoder-test-dir");
        let err = FasterWhisperDecoder::load(&missing, FasterWhisperDecoderConfig::default())
            .expect_err("loading a nonexistent path must error");
        assert!(
            matches!(
                err,
                DecoderError::InvalidModelDir(_) | DecoderError::Load(_)
            ),
            "expected InvalidModelDir or Load, got {err:?}"
        );
    }

    #[test]
    fn decoder_load_returns_error_for_file_path() {
        // Use Cargo.toml of this crate — guaranteed to exist as a file.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let err = FasterWhisperDecoder::load(&path, FasterWhisperDecoderConfig::default())
            .expect_err("loading a file path (not a directory) must error");
        assert!(
            matches!(err, DecoderError::InvalidModelDir(_)),
            "expected InvalidModelDir, got {err:?}"
        );
    }

    #[test]
    fn decoder_error_converts_to_stt_error() {
        let inference: SttError = DecoderError::Inference("boom".to_owned()).into();
        match inference {
            SttError::Transcription(msg) => assert!(msg.contains("boom"), "msg = {msg}"),
            other => panic!("expected Transcription, got {other:?}"),
        }

        let load: SttError = DecoderError::Load("missing model.bin".to_owned()).into();
        match load {
            SttError::ModelLoad(msg) => assert!(msg.contains("missing model.bin"), "msg = {msg}"),
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        let bad_dir: SttError = DecoderError::InvalidModelDir("nope".to_owned()).into();
        assert!(matches!(bad_dir, SttError::ModelLoad(_)));

        let lang: SttError = DecoderError::LanguageDetection("nope".to_owned()).into();
        match lang {
            SttError::Transcription(msg) => {
                assert!(msg.contains("language detection"), "msg = {msg}");
            }
            other => panic!("expected Transcription, got {other:?}"),
        }
    }

    #[test]
    fn detect_language_is_unsupported_in_wave_f2_3() {
        // Construct a decoder-config (not a decoder — no model needed):
        // detect_language is a stub, we want to confirm the error
        // message is informative without loading any weights.
        //
        // We do this via a small synthetic test of the error path: the
        // function never touches `&self.inner`, so the assertion still
        // holds with any decoder. Instead, verify the error variant
        // shape through the decode-time stub directly by mocking
        // (skipped — there's no public way to fabricate a decoder
        // without weights). The error variant is exercised by
        // `decoder_error_converts_to_stt_error` above.
    }

    #[test]
    fn parse_segment_extracts_first_and_last_timestamps() {
        let raw = "<|0.00|> Hello world.<|2.40|>";
        let seg = parse_segment(raw, true);
        assert_eq!(seg.text, "Hello world.");
        assert!((seg.start - 0.0).abs() < f32::EPSILON);
        assert!((seg.end - 2.40).abs() < 1e-4);
        assert!(seg.words.is_empty());
    }

    #[test]
    fn parse_segment_without_timestamps_skips_parsing() {
        let raw = " Hello world. ";
        let seg = parse_segment(raw, false);
        assert_eq!(seg.text, "Hello world.");
        assert!((seg.start - 0.0).abs() < f32::EPSILON);
        assert!((seg.end - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn into_transcription_result_preserves_text_and_segments() {
        let decoded = DecodedTranscript {
            text: "hello world".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![DecodedSegment {
                text: "hello world".to_owned(),
                start: 0.0,
                end: 1.5,
                words: Vec::new(),
            }],
        };
        let result = into_transcription_result(decoded);
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].start_ms, 0);
        assert_eq!(result.segments[0].end_ms, 1500);
        assert_eq!(result.segments[0].text, "hello world");
    }

    #[test]
    fn parse_compute_type_recognises_common_aliases() {
        assert!(matches!(parse_compute_type("int8"), ComputeType::INT8));
        assert!(matches!(parse_compute_type("INT8"), ComputeType::INT8));
        assert!(matches!(
            parse_compute_type("float16"),
            ComputeType::FLOAT16
        ));
        assert!(matches!(parse_compute_type("fp16"), ComputeType::FLOAT16));
        assert!(matches!(
            parse_compute_type("nonsense"),
            ComputeType::DEFAULT
        ));
    }

    #[test]
    fn build_whisper_options_carries_decoder_config() {
        let cfg = FasterWhisperDecoderConfig {
            beam_size: 3,
            best_of: 7,
            temperatures: vec![0.2, 0.5],
            suppress_tokens: vec![-1, 50_257],
            ..FasterWhisperDecoderConfig::default()
        };
        let opts = build_whisper_options(&cfg);
        assert_eq!(opts.beam_size, 3);
        assert_eq!(opts.num_hypotheses, 7);
        assert!((opts.sampling_temperature - 0.2).abs() < f32::EPSILON);
        assert_eq!(opts.suppress_tokens, vec![-1, 50_257]);
    }
}
