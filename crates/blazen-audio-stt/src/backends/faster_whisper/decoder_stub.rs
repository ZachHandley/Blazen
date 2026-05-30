//! Stub `CTranslate2` Whisper decoder for targets where `ct2rs` is
//! unavailable.
//!
//! As of `ct2rs` 0.9.18 the crate vendors and compiles `CTranslate2`'s C++
//! source, and that build feeds `-mavx512f` / `-mavx` / `-mfma` to the
//! compiler for the configured target triple. Apple clang rejects those
//! options for `x86_64-apple-macosx`
//! (`error: unsupported option '-mavx512f' for target 'x86_64-apple-macosx'`),
//! so `ct2rs` cannot be cross-compiled to `x86_64-apple-darwin` from an
//! Apple-silicon host.
//!
//! On that target the `ct2rs` dependency is gated out in `Cargo.toml` (see
//! the `[target.'cfg(not(all(target_arch = "x86_64", target_os =
//! "macos")))'.dependencies]` block) and this module compiles in place of
//! the real [`decoder`](super::decoder). It mirrors the real decoder's
//! **entire public surface** byte-for-byte — same type names, same fields,
//! same `FasterWhisperDecoder` constructor / method signatures, same
//! `into_transcription_result` adapter — so the sibling modules
//! ([`mod`](super), [`pipeline`](super::pipeline)) and the downstream
//! `FasterWhisperBackend` / `FasterWhisperProvider` symbol surface compile
//! unchanged.
//!
//! The only behavioural difference: [`FasterWhisperDecoder::load`] returns
//! [`DecoderError::Load`] with an "unsupported target" message at runtime
//! instead of initialising `CTranslate2`. The provider therefore surfaces
//! [`SttError::ModelLoad`] when a caller tries to transcribe on
//! `x86_64-apple-darwin` — exactly the `blazen-embed-fastembed`
//! `UnsupportedTarget` pattern for ORT.

use std::path::Path;

use thiserror::Error;

use crate::SttError;
use crate::traits::{TranscriptionResult, TranscriptionSegment};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Decoder-side configuration for [`FasterWhisperDecoder`].
///
/// Surface-identical to the real [`decoder::FasterWhisperDecoderConfig`].
#[derive(Debug, Clone)]
pub struct FasterWhisperDecoderConfig {
    /// `CTranslate2` compute type as a human string.
    pub compute_type: String,
    /// CUDA device index. `None` selects CPU.
    pub device_index: Option<usize>,
    /// CPU thread count per replica. `None` => `CTranslate2`'s auto-pick.
    pub num_threads: Option<usize>,
    /// Beam-search width. Upstream default 5.
    pub beam_size: usize,
    /// Best-of for the sampling-fallback path. Upstream default 5.
    pub best_of: usize,
    /// Temperature fallback ladder. Empty means greedy / no fallback.
    pub temperatures: Vec<f32>,
    /// Optional initial prompt for decoder biasing.
    pub initial_prompt: Option<String>,
    /// Token IDs to suppress. `-1` is "the default suppress-set".
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
///
/// Surface-identical to the real [`decoder::TranscribeOptions`].
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// If `true`, ask the decoder to emit word-level timestamps.
    pub want_word_timestamps: bool,
    /// If `true`, request segment-level timestamps.
    pub want_segment_timestamps: bool,
    /// If `true` and `language` is `None`, run inline language detection.
    pub detect_language_when_none: bool,
}

// ---------------------------------------------------------------------------
// Decoded output types
// ---------------------------------------------------------------------------

/// Output of [`FasterWhisperDecoder::detect_language`].
#[derive(Debug, Clone)]
pub struct DetectedLanguage {
    /// ISO 639-1 (or 639-3) language code.
    pub language_code: String,
    /// Probability mass assigned to the detected language, in `[0, 1]`.
    pub probability: f32,
}

/// A rich transcript produced by [`FasterWhisperDecoder::transcribe`].
#[derive(Debug, Clone)]
pub struct DecodedTranscript {
    /// Concatenated transcript text.
    pub text: String,
    /// Language code that the decoder operated under.
    pub language: Option<String>,
    /// Per-segment breakdown.
    pub segments: Vec<DecodedSegment>,
}

/// One segment within a [`DecodedTranscript`].
#[derive(Debug, Clone)]
pub struct DecodedSegment {
    /// Segment text with Whisper timestamp tokens stripped.
    pub text: String,
    /// Start offset in seconds from the beginning of the source audio.
    pub start: f32,
    /// End offset in seconds.
    pub end: f32,
    /// Word-level breakdown.
    pub words: Vec<DecodedWord>,
}

/// One word within a [`DecodedSegment`]. Reserved for a future wave.
#[derive(Debug, Clone)]
pub struct DecodedWord {
    /// Word text.
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
/// Surface-identical to the real [`decoder::DecoderError`]; all variants
/// flatten into [`SttError`] via the [`From`] impl below.
#[derive(Debug, Error)]
pub enum DecoderError {
    /// `ct2rs::Whisper::new` (model load / `CTranslate2` init) failed. On
    /// the stub target this carries the "unsupported target" message.
    #[error("ct2rs load failed: {0}")]
    Load(String),
    /// `ct2rs::Whisper::generate` (decoder inference) failed.
    #[error("ct2rs inference failed: {0}")]
    Inference(String),
    /// The model directory does not exist, is not a directory, or is
    /// missing the `CTranslate2`/Whisper bundle files.
    #[error("invalid model directory: {0}")]
    InvalidModelDir(String),
    /// Standalone language detection failed or is unavailable.
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
// Decoder (stub)
// ---------------------------------------------------------------------------

/// Stub `CTranslate2` Whisper handle.
///
/// On `x86_64-apple-darwin` the `ct2rs` dependency is gated out, so this
/// type carries no model handle. [`FasterWhisperDecoder::load`] always
/// fails with an "unsupported target" [`DecoderError::Load`]; every other
/// method is therefore unreachable on a constructed value.
pub struct FasterWhisperDecoder {
    /// A constructed `FasterWhisperDecoder` is unreachable on the stub
    /// target — `load` is the only constructor and it always errors — so
    /// the only field is an uninhabited witness.
    never: std::convert::Infallible,
}

impl std::fmt::Debug for FasterWhisperDecoder {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.never {}
    }
}

impl FasterWhisperDecoder {
    /// Open a `CTranslate2`-converted Whisper model from a local
    /// directory.
    ///
    /// **Stub target (`x86_64-apple-darwin`)**: `ct2rs`/`CTranslate2` is
    /// unavailable here (it can't cross-compile — see the module doc), so
    /// this always returns [`DecoderError::Load`] with an "unsupported
    /// target" message.
    ///
    /// # Errors
    ///
    /// Always returns [`DecoderError::Load`] on the stub target.
    pub fn load(
        _model_dir: &Path,
        _config: FasterWhisperDecoderConfig,
    ) -> Result<Self, DecoderError> {
        tracing::warn!(
            "FasterWhisperDecoder::load called on a target without ct2rs / \
             CTranslate2 (x86_64-apple-darwin) — returning Unsupported. The \
             faster-whisper backend cannot run on this target; use the \
             whisper.cpp or candle STT backend instead."
        );
        Err(DecoderError::Load(
            "faster-whisper backend unavailable on this target: ct2rs / \
             CTranslate2 cannot be cross-compiled to x86_64-apple-darwin \
             (Apple clang rejects the -mavx512f/-mavx/-mfma flags the \
             CTranslate2 build emits). Use the whisper.cpp or candle STT \
             backend on this target."
                .to_owned(),
        ))
    }

    /// Run language detection on (up to) 30 seconds of audio.
    ///
    /// Unreachable on the stub target ([`load`](Self::load) never returns a
    /// value), so the body never executes.
    ///
    /// # Errors
    ///
    /// Unreachable — no `FasterWhisperDecoder` value can be constructed on
    /// the stub target.
    pub fn detect_language(&self, _samples: &[f32]) -> Result<DetectedLanguage, DecoderError> {
        match self.never {}
    }

    /// Transcribe `samples` (16 kHz mono `f32` in `[-1, 1]`).
    ///
    /// Unreachable on the stub target ([`load`](Self::load) never returns a
    /// value), so the body never executes.
    ///
    /// # Errors
    ///
    /// Unreachable — no `FasterWhisperDecoder` value can be constructed on
    /// the stub target.
    pub fn transcribe(
        &self,
        _samples: &[f32],
        _language: Option<&str>,
        _opts: &TranscribeOptions,
    ) -> Result<DecodedTranscript, DecoderError> {
        match self.never {}
    }

    /// Read-only access to the configuration this decoder was loaded with.
    ///
    /// Unreachable on the stub target.
    #[must_use]
    pub fn config(&self) -> &FasterWhisperDecoderConfig {
        match self.never {}
    }
}

// ---------------------------------------------------------------------------
// Helpers (crate-internal)
// ---------------------------------------------------------------------------

/// Convert the rich [`DecodedTranscript`] into the trait surface
/// [`TranscriptionResult`].
///
/// Identical to the real decoder's adapter — kept so the sibling
/// [`mod`](super) `transcribe` path compiles unchanged. (Unreachable in
/// practice on the stub target because no `DecodedTranscript` is ever
/// produced.)
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
    let ms = (f64::from(seconds) * 1000.0).round();
    if ms <= 0.0 { 0 } else { ms as i64 }
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
    fn decoder_load_returns_unsupported_on_stub_target() {
        let any = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let err = FasterWhisperDecoder::load(&any, FasterWhisperDecoderConfig::default())
            .expect_err("ct2rs is unavailable on the stub target — load must error");
        match err {
            DecoderError::Load(msg) => {
                assert!(
                    msg.contains("unavailable on this target"),
                    "expected an unsupported-target message, got {msg}"
                );
            }
            other => panic!("expected DecoderError::Load, got {other:?}"),
        }
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
}
