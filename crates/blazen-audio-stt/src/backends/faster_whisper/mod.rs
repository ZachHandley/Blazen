//! faster-whisper backend — a Rust port of the upstream
//! `SYSTRAN/faster-whisper` Python project, wrapping `CTranslate2`'s
//! INT8-quantised Whisper inference via the `ct2rs` Rust binding.
//!
//! # Architecture
//!
//! 1. **Audio frontend** ([`audio`]) — resamples input to 16 kHz mono
//!    and computes the 80-band log-mel spectrogram Whisper expects.
//! 2. **Tokenizer** ([`tokenizer`]) — Whisper BPE tokenizer (loaded
//!    from `tokenizer.json` alongside the CT2 weights).
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
//! - **Waves F.2.1 – F.2.5**: real [`audio`], [`tokenizer`],
//!   [`decoder`], [`pipeline`], and [`weights`] implementations land,
//!   wiring the live `transcribe` path.
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

use std::path::Path;
use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use futures_core::Stream;

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
}

impl Default for FasterWhisperConfig {
    fn default() -> Self {
        Self {
            model_id: "Systran/faster-whisper-tiny".to_owned(),
        }
    }
}

/// faster-whisper backend handle.
///
/// Construct via [`FasterWhisperBackend::new`]. Weights load lazily
/// on the first [`SttBackend::transcribe`] call once Wave F.2 lands.
/// Until then every [`SttBackend`] method returns
/// [`SttError::Unsupported`].
#[derive(Debug, Clone)]
pub struct FasterWhisperBackend {
    id: String,
    config: FasterWhisperConfig,
}

impl FasterWhisperBackend {
    /// Build a new faster-whisper backend with the given configuration.
    ///
    /// No weights are downloaded at construction time — the underlying
    /// pipeline is materialised on the first transcription call once
    /// Wave F.2 lands. Until then every [`SttBackend`] method returns
    /// [`SttError::Unsupported`].
    #[must_use]
    pub fn new(config: FasterWhisperConfig) -> Self {
        let id = format!("{FASTER_WHISPER_BACKEND_ID_PREFIX}:{}", config.model_id);
        Self { id, config }
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
        false
    }
}

#[async_trait]
impl SttBackend for FasterWhisperBackend {
    async fn transcribe(
        &self,
        _audio_path: &Path,
        _language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        Err(SttError::Unsupported(
            "faster-whisper Wave F.0 scaffolding — ct2rs integration lands in Wave F.1+".into(),
        ))
    }

    async fn stream(
        &self,
        _audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        _language: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>>, SttError>
    {
        Err(SttError::Unsupported(
            "faster-whisper Wave F.0 scaffolding — streaming override lands in Wave F.2.7".into(),
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
    async fn transcribe_returns_wave_f0_unsupported() {
        let backend = FasterWhisperBackend::default();
        let err = backend
            .transcribe(Path::new("/nonexistent.wav"), None)
            .await
            .expect_err("scaffold must surface Unsupported");
        match err {
            SttError::Unsupported(msg) => {
                assert!(msg.contains("Wave F."), "msg = {msg}");
                assert!(msg.contains("ct2rs"), "msg = {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
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
