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
mod pipeline;
mod tokenizer;
mod weights;

// `decoder` wraps `ct2rs` (CTranslate2). On `x86_64-apple-darwin` and
// `x86_64-pc-windows-msvc` `ct2rs` is gated out (it can't build there —
// the CTranslate2 build emits `-mavx512f`/`-mavx`/`-mfma` which Apple clang
// rejects for `x86_64-apple-macosx`, and `sentencepiece-sys`'s cmake/MSVC
// build fails on windows), so we compile a surface-identical runtime stub
// (`decoder_stub`) in its place — mirroring the `blazen-embed-fastembed`
// ORT precedent. Both arms expose the same `decoder` module path so the
// re-export and the sibling modules (`pipeline`, this `mod`) are unchanged.
// Keep this predicate in lockstep with the `ct2rs` target-gate in Cargo.toml.
#[cfg(not(any(
    all(target_arch = "x86_64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "windows"),
    target_env = "musl"
)))]
mod decoder;
#[cfg(any(
    all(target_arch = "x86_64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "windows"),
    target_env = "musl"
))]
mod decoder_stub;
#[cfg(any(
    all(target_arch = "x86_64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "windows"),
    target_env = "musl"
))]
use decoder_stub as decoder;

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
    /// `None` (the default) routes through
    /// [`weights::ensure_downloaded`] on the first transcription call to
    /// fetch and cache the [`model_id`][Self::model_id] bundle from
    /// Hugging Face Hub. Set this explicitly to point at a manually
    /// populated bundle directory and skip the HF download entirely.
    pub model_dir: Option<PathBuf>,
    /// Optional Hugging Face Hub revision pin (branch, tag, or commit
    /// SHA). `None` (the default) follows the upstream `main` branch,
    /// matching the `faster-whisper` Python project's default behavior.
    /// Passing an explicit revision causes the cache to be keyed by
    /// `{model_id}@{revision}` so multiple revisions of the same repo
    /// can coexist on disk.
    pub revision: Option<String>,
    /// Decoder knobs (beam size, compute type, etc.). See
    /// [`FasterWhisperDecoderConfig`].
    pub decoder: FasterWhisperDecoderConfig,
}

impl Default for FasterWhisperConfig {
    fn default() -> Self {
        Self {
            model_id: "Systran/faster-whisper-tiny".to_owned(),
            model_dir: None,
            revision: None,
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
    /// When [`FasterWhisperConfig::model_dir`] is `Some`, that directory
    /// is used as-is. When it is `None`, the bundle is fetched on demand
    /// from Hugging Face Hub via [`weights::ensure_downloaded`] using
    /// [`FasterWhisperConfig::model_id`] (and the optional
    /// [`FasterWhisperConfig::revision`]) and cached under the shared
    /// `blazen_model_cache` root.
    ///
    /// # Errors
    ///
    /// - [`SttError::ModelLoad`] when the Hugging Face download fails,
    ///   the bundle is malformed, or ct2rs refuses to initialise.
    /// - [`SttError::Io`] when a filesystem operation fails while
    ///   resolving the cache directory.
    async fn ensure_loaded(&self) -> Result<Arc<FasterWhisperDecoder>, SttError> {
        let cfg_model_dir = self.config.model_dir.clone();
        let model_id = self.config.model_id.clone();
        let revision = self.config.revision.clone();
        let decoder_cfg = self.config.decoder.clone();
        self.inner
            .get_or_try_init(|| async move {
                let model_dir = match cfg_model_dir {
                    Some(dir) => dir,
                    None => weights::ensure_downloaded(&model_id, revision.as_deref())
                        .await
                        .map_err(SttError::from)?,
                };
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

    /// Window-based streaming transcription (Wave F.2.4).
    ///
    /// `audio` must yield 16 kHz mono f32 PCM (the
    /// `prepare_for_whisper` resampler is **not** applied to streaming
    /// input — callers are responsible for upstream resampling, matching
    /// the `whisper-streaming` backend's contract). The pipeline buffers
    /// incoming samples until it has a full 30-second window (480 000
    /// samples), then dispatches one [`FasterWhisperDecoder::transcribe`]
    /// per window on the blocking pool. Each window's transcript is
    /// emitted as a single [`StreamingTranscript`] with
    /// `is_final = true` and `latency_seconds` carrying the window's
    /// start-offset (in seconds) from the stream's first sample. The
    /// final partial window (if non-empty) flushes on EOS.
    ///
    /// See [`pipeline`] for the rationale on window-based dispatch
    /// (ct2rs exposes no native streaming/callback hook).
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] if the lazy [`ensure_loaded`] call
    /// fails (HF download or ct2rs init); per-window inference failures
    /// surface as `Err` items on the returned stream.
    async fn stream(
        &self,
        audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        language: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>>, SttError>
    {
        let decoder = self.ensure_loaded().await?;
        let lang = language.map(str::to_owned);
        Ok(pipeline::streaming_transcribe(decoder, audio, lang))
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
    async fn transcribe_with_invalid_repo_id_surfaces_model_load_error() {
        // With model_dir: None (the default) we route through
        // weights::ensure_downloaded. An empty model_id is rejected
        // before any network I/O and surfaces as ModelLoad (the
        // `WeightsError -> SttError` mapping flattens InvalidRepoId).
        let cfg = FasterWhisperConfig {
            model_id: String::new(),
            ..FasterWhisperConfig::default()
        };
        let backend = FasterWhisperBackend::new(cfg);
        let err = backend
            .transcribe(Path::new("/nonexistent.wav"), None)
            .await
            .expect_err("empty model_id must surface an error");
        assert!(
            matches!(err, SttError::ModelLoad(_)),
            "expected ModelLoad, got {err:?}"
        );
    }

    #[test]
    fn config_default_revision_is_none() {
        // Sanity check that the new `revision` field defaults to None,
        // matching upstream `faster-whisper`'s "follow main" behavior.
        let cfg = FasterWhisperConfig::default();
        assert!(cfg.revision.is_none());
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

    /// Streaming-surface live test (Wave F.2.4) — replaces the
    /// obsolete `stream_returns_wave_f0_unsupported` from the F.0
    /// scaffold. Feeds 90 s of silence (split into 90 × 16 000-sample
    /// chunks) through [`SttBackend::stream`] and asserts the pipeline
    /// emits exactly 3 windows (no partial trailing window — 90 s is an
    /// exact multiple of the 30 s window). Gated on
    /// `BLAZEN_TEST_FASTER_WHISPER=1` because it pulls the
    /// `Systran/faster-whisper-tiny` weights on first run.
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_FASTER_WHISPER=1 and pulls ~75 MB of CTranslate2 Whisper weights from HF Hub"]
    async fn stream_yields_one_transcript_per_30s_window() {
        use futures_util::{StreamExt, stream};

        if std::env::var("BLAZEN_TEST_FASTER_WHISPER").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_FASTER_WHISPER != 1");
            return;
        }

        // 90 chunks × 16 000 samples each = 90 s of silence at 16 kHz =
        // 3 full 30 s windows, no partial tail.
        let chunks: Vec<Vec<f32>> = (0..90).map(|_| vec![0.0_f32; 16_000]).collect();
        let audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>> = Box::pin(stream::iter(chunks));

        let backend = FasterWhisperBackend::default();
        let mut out = backend
            .stream(audio, Some("en"))
            .await
            .expect("stream construction must succeed");

        let mut emissions: Vec<StreamingTranscript> = Vec::new();
        while let Some(item) = out.next().await {
            emissions.push(item.expect("no per-window inference errors on silence"));
        }
        assert_eq!(
            emissions.len(),
            3,
            "90 s of input must produce exactly 3 × 30 s windows"
        );
        // Every emission is committed (window-based dispatch never
        // produces interim guesses).
        assert!(emissions.iter().all(|t| t.is_final));
        // Window start offsets must be 0, 30, 60 seconds.
        let starts: Vec<f32> = emissions
            .iter()
            .map(|t| t.latency_seconds.unwrap_or(f32::NAN))
            .collect();
        assert!((starts[0] - 0.0).abs() < 1e-3, "starts = {starts:?}");
        assert!((starts[1] - 30.0).abs() < 1e-3, "starts = {starts:?}");
        assert!((starts[2] - 60.0).abs() < 1e-3, "starts = {starts:?}");
    }

    /// Streaming-surface live test — confirms the public contract for
    /// an empty input stream: [`SttBackend::stream`] returns `Ok(_)`
    /// (the lazy weight load still has to succeed) and the resulting
    /// stream terminates immediately with no emissions. Gated on
    /// `BLAZEN_TEST_FASTER_WHISPER=1` for the same reason as
    /// [`stream_yields_one_transcript_per_30s_window`] — the
    /// `ensure_loaded` call still pulls weights.
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_FASTER_WHISPER=1 and pulls ~75 MB of CTranslate2 Whisper weights from HF Hub"]
    async fn stream_returns_ok_for_empty_input() {
        use futures_util::{StreamExt, stream};

        if std::env::var("BLAZEN_TEST_FASTER_WHISPER").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_FASTER_WHISPER != 1");
            return;
        }

        let backend = FasterWhisperBackend::default();
        let empty: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>> = Box::pin(stream::empty());
        let mut out = backend
            .stream(empty, None)
            .await
            .expect("stream construction must succeed even on empty input");
        let count = out.by_ref().count().await;
        assert_eq!(count, 0, "empty input must yield zero emissions");
    }

    /// End-to-end Wave F.2.5 sanity test — constructs a default
    /// [`FasterWhisperBackend`] (no `model_dir`), writes a tiny
    /// 1-second silent 16 kHz mono WAV to a repo-local scratch dir,
    /// calls [`SttBackend::transcribe`] on it, and asserts that the HF
    /// download → ct2rs load → transcribe path returns `Ok(_)`.
    ///
    /// Gated by `BLAZEN_TEST_FASTER_WHISPER=1` (matches the
    /// `BLAZEN_TEST_BARK` pattern) because it downloads the ~75 MB
    /// `Systran/faster-whisper-tiny` `CTranslate2` bundle from HF Hub on
    /// first run.
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_FASTER_WHISPER=1 and pulls ~75 MB of CTranslate2 Whisper weights from HF Hub"]
    async fn transcribe_uses_hf_download_when_model_dir_none() {
        if std::env::var("BLAZEN_TEST_FASTER_WHISPER").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_FASTER_WHISPER != 1");
            return;
        }

        // Honor the "no /tmp scratch" project rule: root the tempdir
        // under the user's cache directory ($HOME/.cache on Linux), or
        // fall back to the CWD if HOME is unset.
        let cache_root = std::env::var_os("HOME")
            .map_or_else(
                || std::env::current_dir().expect("cwd"),
                |h| PathBuf::from(h).join(".cache"),
            )
            .join("blazen-faster-whisper-tests");
        std::fs::create_dir_all(&cache_root).expect("create cache root");
        let tmp = tempfile::Builder::new()
            .prefix("transcribe-")
            .tempdir_in(&cache_root)
            .expect("tempdir");

        // 1-second 16 kHz mono silence as PCM16 LE.
        let wav_path = tmp.path().join("silence.wav");
        let pcm_samples: Vec<i16> = vec![0; 16_000];
        write_pcm16_wav(&wav_path, 16_000, &pcm_samples).expect("write wav");

        let backend = FasterWhisperBackend::default();
        let result = backend
            .transcribe(&wav_path, Some("en"))
            .await
            .expect("transcribe succeeds end-to-end with HF download");

        // Silence may yield zero segments, but the call must complete
        // successfully — i.e. the download + load + decode pipeline
        // produced a TranscriptionResult, not an error.
        let _ = result.segments;
    }

    /// Write a minimal PCM16 LE WAV (RIFF/WAVE) — 44-byte header + samples.
    /// Used by the live transcribe test; kept inside `cfg(test)` so it
    /// doesn't bloat the public surface.
    #[cfg(test)]
    fn write_pcm16_wav(path: &Path, sample_rate: u32, samples: &[i16]) -> std::io::Result<()> {
        use std::io::Write;
        let n_channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * u32::from(n_channels) * u32::from(bits_per_sample) / 8;
        let block_align = n_channels * bits_per_sample / 8;
        let data_size = u32::try_from(samples.len() * 2).unwrap_or(u32::MAX);
        let riff_size = 36u32.saturating_add(data_size);

        let mut f = std::fs::File::create(path)?;
        f.write_all(b"RIFF")?;
        f.write_all(&riff_size.to_le_bytes())?;
        f.write_all(b"WAVE")?;
        f.write_all(b"fmt ")?;
        f.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
        f.write_all(&1u16.to_le_bytes())?; // PCM format
        f.write_all(&n_channels.to_le_bytes())?;
        f.write_all(&sample_rate.to_le_bytes())?;
        f.write_all(&byte_rate.to_le_bytes())?;
        f.write_all(&block_align.to_le_bytes())?;
        f.write_all(&bits_per_sample.to_le_bytes())?;
        f.write_all(b"data")?;
        f.write_all(&data_size.to_le_bytes())?;
        for s in samples {
            f.write_all(&s.to_le_bytes())?;
        }
        Ok(())
    }

    /// Wave F.1 link probe — confirms the `ct2rs` crate (and the
    /// `CTranslate2` C++ runtime it links against) is reachable from
    /// this crate. Roundtrips `LogLevel` through `set_log_level` /
    /// `get_log_level`, both of which dispatch into the `CTranslate2`
    /// shared library via `cxx`. If the link is broken (missing
    /// symbol, ABI mismatch, build failure), this test refuses to
    /// link and the failure surfaces at compile / load time rather
    /// than buried in the first transcription call.
    // `ct2rs` is gated out on `x86_64-apple-darwin`,
    // `x86_64-pc-windows-msvc`, and all musl targets (the decoder compiles to
    // a stub there — see the `decoder` module cfg above), so the link probe
    // only exists on targets that actually link CTranslate2.
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
        target_env = "musl"
    )))]
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
