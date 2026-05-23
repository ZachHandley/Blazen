//! faster-whisper backend — a Rust port of the upstream
//! `SYSTRAN/faster-whisper` Python project, wrapping `CTranslate2`'s
//! INT8-quantised Whisper inference via the `ct2rs` Rust binding.
//!
//! # Architecture
//!
//! 1. **Audio frontend** ([`audio`]) — resamples input to 16 kHz mono
//!    and computes the 80-band log-mel spectrogram Whisper expects.
//! 2. **Tokenizer** ([`tokenizer`]) — Whisper BPE tokenization is
//!    owned internally by `ct2rs::Whisper` (loads `tokenizer.json`
//!    from the model directory, encodes the prompt, decodes the output
//!    sequence). No Rust-side tokenizer is required; see the module
//!    doc for rationale.
//! 3. **Decoder** ([`decoder`]) — thin wrapper over `ct2rs::Whisper`
//!    that owns the `CTranslate2` model handle and performs the
//!    encoder/decoder inference.
//! 4. **Weights** ([`weights`]) — Hugging Face Hub download for the
//!    `Systran/faster-whisper-*` repos (which ship the `CTranslate2`
//!    `model.bin` + `config.json` + tokenizer files).
//! 5. **Pipeline** ([`pipeline`]) — orchestrates audio → mel →
//!    decoder → segments and surfaces the
//!    [`SttBackend::transcribe`] / [`SttBackend::stream`] entrypoints.
//!
//! # Wave plan
//!
//! - **Wave F.0** (this commit): scaffolding only.
//!   [`FasterWhisperBackend`] and [`FasterWhisperConfig`] exist;
//!   every [`SttBackend`] method returns [`SttError::Unsupported`].
//!   The five sub-modules carry one-line doc-comments and no code.
//! - **Wave F.1** (this commit): add the `ct2rs` dependency (gated
//!   behind `faster-whisper`) plus a link-probe smoke test
//!   ([`tests::link_probe_ct2rs_loads_successfully`]) that confirms
//!   the Rust binding loads the `CTranslate2` shared library.
//! - **Waves F.2.1, F.2.3 – F.2.5**: real [`audio`], [`decoder`],
//!   [`pipeline`], and [`weights`] implementations land, wiring the
//!   live `transcribe` path. Wave **F.2.2** (Whisper BPE tokenization)
//!   is absorbed by `ct2rs::Whisper`'s internal tokenizer and
//!   contributes no Rust surface — see [`tokenizer`] for details.
//! - **Wave F.2.7**: streaming override for [`SttBackend::stream`]
//!   built on top of the file-based pipeline.
//!
//! # License
//!
//! Upstream `SYSTRAN/faster-whisper` ships MIT source over MIT
//! `CTranslate2` over MIT Whisper. The Rust binding `ct2rs` is MIT.
//! The whole stack is commercial-safe — unlike Spark-TTS, `MaskGCT`,
//! and `AudioLDM`, no non-commercial weights restriction applies.

#![cfg(feature = "faster-whisper")]

mod audio;
mod decoder;
mod pipeline;
mod tokenizer;
mod weights;

// Re-export the Wave F.2.1 audio preprocessing surface so it is reachable
// from sibling modules and integration tests ahead of Waves F.2.4 / F.2.7
// wiring it into the live transcribe path.
pub use audio::{AudioError, AudioInput, TARGET_SAMPLE_RATE, prepare_for_whisper};
pub use decoder::{
    DecodedSegment, DecodedTranscript, DecodedWord, DecoderError, DetectedLanguage,
    FasterWhisperDecoder, FasterWhisperDecoderConfig, TranscribeOptions,
};

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use futures_core::Stream;
use tokio::sync::OnceCell;

use crate::SttError;
use crate::traits::{StreamingTranscript, SttBackend, TranscriptionResult};

/// Stable backend-id prefix surfaced via [`AudioBackend::id`].
pub const FASTER_WHISPER_BACKEND_ID_PREFIX: &str = "faster-whisper";

/// Configuration knobs for [`FasterWhisperBackend`].
///
/// Defaults target the cheapest first-light validation target:
/// `Systran/faster-whisper-tiny` (39 M params, ~75 MB INT8 weights).
/// Larger checkpoints (`Systran/faster-whisper-base`,
/// `Systran/faster-whisper-small`, `…-medium`, `…-large-v3`) are
/// drop-in replacements once Wave F.2.5 wires the HF download.
#[derive(Debug, Clone)]
pub struct FasterWhisperConfig {
    /// Hugging Face repo id for the `CTranslate2` Whisper bundle.
    /// Default `"Systran/faster-whisper-tiny"`.
    pub model_id: String,
    /// Local filesystem path to a pre-downloaded `CTranslate2` Whisper
    /// bundle directory (the layout documented on
    /// [`FasterWhisperDecoder::load`]).
    ///
    /// `None` (the default) means "wait for Wave F.2.5's HF download
    /// helper" — until that wave lands, every
    /// [`SttBackend::transcribe`] call returns [`SttError::Unsupported`]
    /// unless the caller fills this field explicitly.
    pub model_dir: Option<PathBuf>,
    /// Decoder knobs (beam size, compute type, etc.). See
    /// [`FasterWhisperDecoderConfig`].
    pub decoder: FasterWhisperDecoderConfig,
}

impl Default for FasterWhisperConfig {
    fn default() -> Self {
        Self {
            model_id: "Systran/faster-whisper-tiny".to_owned(),
            model_dir: None,
            decoder: FasterWhisperDecoderConfig::default(),
        }
    }
}

/// faster-whisper backend handle.
///
/// Construct via [`FasterWhisperBackend::new`]. Weights load lazily
/// on the first [`SttBackend::transcribe`] call once Wave F.2 lands.
/// Until then every [`SttBackend`] method returns
/// [`SttError::Unsupported`].
#[derive(Clone)]
pub struct FasterWhisperBackend {
    id: String,
    config: FasterWhisperConfig,
    /// Lazily-initialised ct2rs decoder. `OnceCell` keeps `load()` and
    /// the first `transcribe()` race-safely converging on a single weight
    /// load — same pattern used by [`crate::backends::candle`].
    inner: Arc<OnceCell<Arc<FasterWhisperDecoder>>>,
}

impl std::fmt::Debug for FasterWhisperBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FasterWhisperBackend")
            .field("id", &self.id)
            .field("model_id", &self.config.model_id)
            .field("model_dir", &self.config.model_dir)
            .field("loaded", &self.inner.initialized())
            .finish_non_exhaustive()
    }
}

impl FasterWhisperBackend {
    /// Build a new faster-whisper backend with the given configuration.
    ///
    /// No weights are downloaded or loaded at construction time — the
    /// ct2rs decoder is materialised lazily on the first transcription
    /// call when [`FasterWhisperConfig::model_dir`] is set. With
    /// `model_dir: None` (the default), [`SttBackend::transcribe`]
    /// returns [`SttError::Unsupported`] until Wave F.2.5 wires the HF
    /// download helper.
    #[must_use]
    pub fn new(config: FasterWhisperConfig) -> Self {
        let id = format!("{FASTER_WHISPER_BACKEND_ID_PREFIX}:{}", config.model_id);
        Self {
            id,
            config,
            inner: Arc::new(OnceCell::new()),
        }
    }

    /// Ensure the ct2rs decoder is loaded; returns a cheap `Arc` clone
    /// of the cached handle.
    ///
    /// # Errors
    ///
    /// - [`SttError::Unsupported`] when [`FasterWhisperConfig::model_dir`]
    ///   is `None`. The message mentions Wave F.2.5 so callers know how
    ///   to unblock themselves manually in the meantime.
    /// - [`SttError::ModelLoad`] when the bundle is malformed or ct2rs
    ///   refuses to initialise.
    async fn ensure_loaded(&self) -> Result<Arc<FasterWhisperDecoder>, SttError> {
        let model_dir = self.config.model_dir.clone().ok_or_else(|| {
            SttError::Unsupported(
                "faster-whisper backend requires FasterWhisperConfig::model_dir until Wave F.2.5 \
                 wires automatic Hugging Face download — set model_dir manually to a \
                 CTranslate2 Whisper bundle for now"
                    .to_owned(),
            )
        })?;
        let decoder_cfg = self.config.decoder.clone();
        self.inner
            .get_or_try_init(|| async move {
                tokio::task::spawn_blocking(move || {
                    FasterWhisperDecoder::load(&model_dir, decoder_cfg)
                        .map(Arc::new)
                        .map_err(SttError::from)
                })
                .await
                .map_err(|e| SttError::ModelLoad(format!("join error: {e}")))?
            })
            .await
            .map(Arc::clone)
    }

    /// The resolved model id this backend was configured with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    /// Access the underlying [`FasterWhisperConfig`] (read-only).
    #[must_use]
    pub fn config(&self) -> &FasterWhisperConfig {
        &self.config
    }
}

impl Default for FasterWhisperBackend {
    fn default() -> Self {
        Self::new(FasterWhisperConfig::default())
    }
}

#[async_trait]
impl AudioBackend for FasterWhisperBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "stt"
    }

    async fn is_loaded(&self) -> bool {
        self.inner.initialized()
    }
}

#[async_trait]
impl SttBackend for FasterWhisperBackend {
    async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        let handle = self.ensure_loaded().await?;

        let samples = prepare_for_whisper(AudioInput::Path(audio_path)).map_err(SttError::from)?;

        let lang_owned = language.map(str::to_owned);
        let decoded = tokio::task::spawn_blocking(move || {
            handle.transcribe(
                &samples,
                lang_owned.as_deref(),
                &TranscribeOptions {
                    want_segment_timestamps: true,
                    ..TranscribeOptions::default()
                },
            )
        })
        .await
        .map_err(|e| SttError::Transcription(format!("join error: {e}")))?
        .map_err(SttError::from)?;

        Ok(decoder::into_transcription_result(decoded))
    }

    async fn stream(
        &self,
        _audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        _language: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>>, SttError>
    {
        Err(SttError::Unsupported(
            "faster-whisper streaming override lands in Wave F.2.7".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faster_whisper_config_defaults_match_first_light_target() {
        let cfg = FasterWhisperConfig::default();
        assert_eq!(cfg.model_id, "Systran/faster-whisper-tiny");
        assert!(
            cfg.model_dir.is_none(),
            "model_dir defaults to None pending Wave F.2.5"
        );
        assert_eq!(cfg.decoder.beam_size, 5);
        assert_eq!(cfg.decoder.compute_type, "int8");
    }

    #[test]
    fn faster_whisper_backend_id_includes_model() {
        let backend = FasterWhisperBackend::new(FasterWhisperConfig::default());
        assert_eq!(backend.id(), "faster-whisper:Systran/faster-whisper-tiny");
        assert_eq!(backend.model_id(), "Systran/faster-whisper-tiny");
        assert_eq!(backend.provider_kind(), "stt");
    }

    #[tokio::test]
    async fn is_loaded_starts_false() {
        let backend = FasterWhisperBackend::default();
        assert!(!backend.is_loaded().await);
    }

    #[tokio::test]
    async fn transcribe_without_model_dir_returns_unsupported() {
        // With model_dir: None (the default), transcribe must short-circuit
        // before touching audio I/O with an Unsupported error that points
        // at Wave F.2.5 (the wave that will land the HF download helper).
        let backend = FasterWhisperBackend::default();
        let err = backend
            .transcribe(Path::new("/nonexistent.wav"), None)
            .await
            .expect_err("missing model_dir must surface Unsupported");
        match err {
            SttError::Unsupported(msg) => {
                assert!(msg.contains("model_dir"), "msg = {msg}");
                assert!(msg.contains("F.2.5"), "msg = {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn transcribe_with_invalid_model_dir_returns_model_load_error() {
        // With a model_dir that points nowhere, we should bubble up
        // through the decoder's InvalidModelDir -> SttError::ModelLoad
        // mapping — not panic, not silently succeed.
        let cfg = FasterWhisperConfig {
            model_dir: Some(PathBuf::from(
                "/nonexistent/blazen-faster-whisper-mod-test-dir",
            )),
            ..FasterWhisperConfig::default()
        };
        let backend = FasterWhisperBackend::new(cfg);
        let err = backend
            .transcribe(Path::new("/nonexistent.wav"), None)
            .await
            .expect_err("invalid model_dir must surface an error");
        assert!(
            matches!(err, SttError::ModelLoad(_)),
            "expected ModelLoad, got {err:?}"
        );
    }

    #[tokio::test]
    async fn stream_returns_wave_f0_unsupported() {
        use futures_util::stream;

        let backend = FasterWhisperBackend::default();
        let audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>> = Box::pin(stream::empty());
        let result = backend.stream(audio, None).await;
        match result {
            Err(SttError::Unsupported(msg)) => {
                assert!(msg.contains("Wave F."), "msg = {msg}");
                assert!(msg.contains("streaming"), "msg = {msg}");
            }
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("scaffold must surface Unsupported"),
        }
    }

    /// Wave F.1 link probe — confirms the `ct2rs` crate (and the
    /// `CTranslate2` C++ runtime it links against) is reachable from
    /// this crate. Roundtrips `LogLevel` through `set_log_level` /
    /// `get_log_level`, both of which dispatch into the `CTranslate2`
    /// shared library via `cxx`. If the link is broken (missing
    /// symbol, ABI mismatch, build failure), this test refuses to
    /// link and the failure surfaces at compile / load time rather
    /// than buried in the first transcription call.
    #[test]
    fn link_probe_ct2rs_loads_successfully() {
        use ct2rs::sys::get_log_level;
        use ct2rs::{LogLevel, set_log_level};

        // Snapshot the current level so the test is idempotent across
        // re-runs in the same process (nextest forks per-test, but
        // belt-and-suspenders since the C++ runtime is process-global).
        let original = get_log_level();

        // Round-trip a non-default level through the FFI boundary.
        set_log_level(LogLevel::Error);
        assert_eq!(
            get_log_level(),
            LogLevel::Error,
            "ct2rs ↔ CTranslate2 round-trip failed — link probe broken"
        );

        set_log_level(LogLevel::Warning);
        assert_eq!(get_log_level(), LogLevel::Warning);

        // Restore.
        set_log_level(original);
    }
}
