//! Whisper-streaming: low-latency real-time speech-to-text via Silero
//! VAD-gated chunking and a sliding-window candle Whisper KV-cache.
//!
//! # Architecture
//!
//! - **Wave A.2** — Silero VAD wrapper in [`vad`] and the chunked
//!   candle Whisper decode loop in [`decoder`].
//! - **Wave A.3** — weight-loading plumbing. The backend lazily
//!   materialises both the VAD and decoder on first use via
//!   [`WhisperStreamingBackend::vad_owned_mut`] and
//!   [`WhisperStreamingBackend::decoder_owned_mut`]; the VAD weights
//!   come from the `deepghs/silero-vad-onnx` HF mirror (see [`vad`]
//!   docs).
//! - **Wave A.4 (this wave)** — pipeline. [`pipeline::spawn_pipeline`]
//!   orchestrates VAD-gated chunking + decode and is plumbed into
//!   [`SttBackend::stream`] below.
//!
//! The lazy `vad`/`decoder` slots are `Arc<Mutex<Option<…>>>` so that
//! [`SttBackend::stream`] can acquire an [`OwnedMutexGuard`] and move
//! it into the spawned pipeline task (which has a `'static` lifetime
//! bound from the boxed `Stream` return type).

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use futures_core::Stream;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::error::SttError;
use crate::traits::{StreamingTranscript, SttBackend, TranscriptionResult};

mod decoder;
mod pipeline;
mod vad;

pub use decoder::{ChunkedWhisperConfig, ChunkedWhisperDecoder, DecodedChunk};
pub use vad::{SileroVad, SileroVadConfig, VadFrame};

/// Whisper-streaming backend handle.
///
/// Constructed via [`WhisperStreamingBackend::new`]; passes
/// configuration through to the pipeline (added in Wave A.4).
pub struct WhisperStreamingBackend {
    id: String,
    config: WhisperStreamingConfig,
    /// Lazy Silero VAD. Materialised on the first call to
    /// [`Self::vad_owned_mut`] (Wave A.3 plumbing); kept in a
    /// `tokio::sync::Mutex` because the initialisation path is async
    /// (HF download) and the lock must be held across the `.await` so
    /// that two concurrent callers don't double-download. Wrapped in
    /// [`Arc`] so [`SttBackend::stream`] (Wave A.4) can call
    /// [`Mutex::lock_owned`] and move the resulting
    /// [`OwnedMutexGuard`] into a `'static` spawned task.
    vad: Arc<Mutex<Option<SileroVad>>>,
    /// Lazy chunked decoder. Same lifecycle / locking story as `vad` —
    /// the underlying candle backend's first inference triggers the
    /// weight download internally, but constructing
    /// [`ChunkedWhisperDecoder`] itself is cheap and synchronous.
    decoder: Arc<Mutex<Option<ChunkedWhisperDecoder>>>,
}

/// Static configuration for the streaming backend.
#[derive(Debug, Clone)]
pub struct WhisperStreamingConfig {
    /// Model identifier (HF repo). Reuses the candle backend's loader.
    pub model_id: String,
    /// Sliding-window chunk length in seconds. Default 30s
    /// (Whisper's native window).
    pub chunk_seconds: f32,
    /// Overlap between successive chunks in seconds. Default 5s.
    pub chunk_overlap_seconds: f32,
}

impl Default for WhisperStreamingConfig {
    fn default() -> Self {
        Self {
            model_id: "openai/whisper-base".into(),
            chunk_seconds: 30.0,
            chunk_overlap_seconds: 5.0,
        }
    }
}

impl WhisperStreamingBackend {
    /// Construct a new streaming backend from `config`.
    ///
    /// Cheap and synchronous: no weights are loaded. The Silero VAD is
    /// built from the embedded model on the first call to
    /// [`Self::vad_owned_mut`] (no I/O); the candle Whisper weights
    /// download on the first inference through the decoder surfaced by
    /// [`Self::decoder_owned_mut`].
    #[must_use]
    pub fn new(config: WhisperStreamingConfig) -> Self {
        let id = format!("whisper-streaming:{}", config.model_id);
        Self {
            id,
            config,
            vad: Arc::new(Mutex::new(None)),
            decoder: Arc::new(Mutex::new(None)),
        }
    }

    /// Borrow the configuration this backend was constructed with.
    #[must_use]
    pub const fn config(&self) -> &WhisperStreamingConfig {
        &self.config
    }

    /// Lazily materialise the Silero VAD and return an *owned* guard
    /// holding the populated slot.
    ///
    /// The Silero VAD is built from the model embedded in the binary
    /// ([`vad`] module docs) — no I/O or network. Subsequent calls return
    /// the existing instance. The returned [`OwnedMutexGuard`] can be moved
    /// into a spawned task — this is how [`SttBackend::stream`] keeps the
    /// VAD locked for the entire duration of the streaming pipeline.
    ///
    /// # Errors
    ///
    /// Propagates any [`SttError::ModelLoad`] from building the embedded
    /// model (effectively never on a healthy build).
    pub(crate) async fn vad_owned_mut(
        &self,
    ) -> Result<OwnedMutexGuard<Option<SileroVad>>, SttError> {
        let mut guard = Arc::clone(&self.vad).lock_owned().await;
        if guard.is_none() {
            *guard = Some(SileroVad::new(vad::SileroVadConfig::default())?);
        }
        Ok(guard)
    }

    /// Lazily materialise the [`ChunkedWhisperDecoder`] and return an
    /// *owned* guard holding the populated slot.
    ///
    /// Construction is synchronous and cheap: the underlying candle
    /// Whisper backend defers its weight download to the first
    /// `decode_chunk` call. The decoder honours
    /// [`WhisperStreamingConfig::model_id`],
    /// [`WhisperStreamingConfig::chunk_seconds`] and
    /// [`WhisperStreamingConfig::chunk_overlap_seconds`]; device
    /// selection mirrors the candle backend default
    /// ([`candle_core::Device::Cpu`] — binary crates that need CUDA or
    /// Metal acceleration should construct their own
    /// [`ChunkedWhisperConfig`] and call into the decoder directly).
    ///
    /// The owned guard form lets [`SttBackend::stream`] move the lock
    /// into the spawned pipeline task.
    ///
    /// # Errors
    ///
    /// Propagates any [`SttError::InvalidOptions`] from
    /// [`ChunkedWhisperDecoder::new`] (bad chunk geometry / unknown
    /// model id).
    pub(crate) async fn decoder_owned_mut(
        &self,
    ) -> Result<OwnedMutexGuard<Option<ChunkedWhisperDecoder>>, SttError> {
        let mut guard = Arc::clone(&self.decoder).lock_owned().await;
        if guard.is_none() {
            let decoder_cfg = decoder::ChunkedWhisperConfig {
                model_id: self.config.model_id.clone(),
                chunk_seconds: self.config.chunk_seconds,
                chunk_overlap_seconds: self.config.chunk_overlap_seconds,
                ..decoder::ChunkedWhisperConfig::default()
            };
            let decoder = ChunkedWhisperDecoder::new(decoder_cfg)?;
            *guard = Some(decoder);
        }
        Ok(guard)
    }
}

#[async_trait]
impl AudioBackend for WhisperStreamingBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "stt"
    }
}

#[async_trait]
impl SttBackend for WhisperStreamingBackend {
    async fn transcribe(
        &self,
        _audio_path: &Path,
        _language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        Err(SttError::Unsupported(
            "use stream() for whisper-streaming; route to candle backend for blocking transcription"
                .into(),
        ))
    }

    /// Stream whisper-streaming transcription.
    ///
    /// Audio chunks are assumed to be **16 kHz mono f32 PCM in
    /// `[-1.0, 1.0]`**; resampling must happen upstream. Chunk lengths
    /// in the input stream may be arbitrary — the pipeline re-cuts them
    /// into 512-sample VAD frames before feeding Silero VAD.
    ///
    /// The `language` per-call hint is currently a no-op: the underlying
    /// [`ChunkedWhisperDecoder`] only honours a config-level language
    /// set at construction time (see [`WhisperStreamingConfig`]). To
    /// stream with a specific language, construct a new backend with
    /// the desired language in its config; the per-call hint here is
    /// reserved for a future API extension that threads it through the
    /// decoder.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] when the embedded Silero VAD cannot
    /// be built, or when [`ChunkedWhisperDecoder::new`] rejects the config.
    /// Pipeline-time failures (decoder inference, malformed frames)
    /// propagate as `Err` items inside the returned stream.
    async fn stream(
        &self,
        audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        _language: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>>, SttError>
    {
        let vad_guard = self.vad_owned_mut().await?;
        let decoder_guard = self.decoder_owned_mut().await?;
        Ok(pipeline::spawn_pipeline(
            audio,
            vad_guard,
            decoder_guard,
            self.config.chunk_seconds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_whisper_base() {
        let cfg = WhisperStreamingConfig::default();
        assert_eq!(cfg.model_id, "openai/whisper-base");
        assert!((cfg.chunk_seconds - 30.0).abs() < f32::EPSILON);
        assert!((cfg.chunk_overlap_seconds - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn backend_id_includes_model() {
        let be = WhisperStreamingBackend::new(WhisperStreamingConfig::default());
        assert_eq!(
            <WhisperStreamingBackend as AudioBackend>::id(&be),
            "whisper-streaming:openai/whisper-base"
        );
        assert_eq!(
            <WhisperStreamingBackend as AudioBackend>::provider_kind(&be),
            "stt"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn backend_starts_with_empty_vad_and_decoder_slots() {
        let be = WhisperStreamingBackend::new(WhisperStreamingConfig::default());
        assert!(be.vad.lock().await.is_none());
        assert!(be.decoder.lock().await.is_none());
        // Arc wrapping is structural; confirm the strong count is 1
        // until something else clones it (e.g. `lock_owned`).
        assert_eq!(Arc::strong_count(&be.vad), 1);
        assert_eq!(Arc::strong_count(&be.decoder), 1);
    }

    #[test]
    fn config_accessor_returns_constructor_args() {
        let cfg = WhisperStreamingConfig {
            model_id: "openai/whisper-tiny".into(),
            chunk_seconds: 20.0,
            chunk_overlap_seconds: 3.0,
        };
        let be = WhisperStreamingBackend::new(cfg);
        assert_eq!(be.config().model_id, "openai/whisper-tiny");
        assert!((be.config().chunk_seconds - 20.0).abs() < f32::EPSILON);
        assert!((be.config().chunk_overlap_seconds - 3.0).abs() < f32::EPSILON);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn transcribe_returns_unsupported() {
        let be = WhisperStreamingBackend::new(WhisperStreamingConfig::default());
        let err = be
            .transcribe(Path::new("/dev/null"), None)
            .await
            .expect_err("scaffolding wave must not implement transcribe");
        assert!(matches!(err, SttError::Unsupported(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn decoder_owned_mut_lazily_materializes_and_caches() {
        // `decoder_owned_mut` is synchronous internally (no HF
        // download) because `ChunkedWhisperDecoder::new` defers weight
        // loading to the first inference; this exercises the
        // slot-caching branch without touching the network.
        let be = WhisperStreamingBackend::new(WhisperStreamingConfig::default());
        {
            let guard = be.decoder_owned_mut().await.expect("first call");
            assert!(guard.is_some(), "first call must populate the slot");
        }
        // Second call must reuse, not recreate. We can't directly
        // observe identity through the guard, but we can at least
        // confirm the slot stays populated.
        {
            let guard = be.decoder_owned_mut().await.expect("second call");
            assert!(guard.is_some());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn vad_owned_mut_loads_embedded_model() {
        // The Silero VAD is built from the embedded ONNX — no network, no
        // path. First call populates the slot; the second reuses it.
        let be = WhisperStreamingBackend::new(WhisperStreamingConfig::default());
        {
            let guard = be.vad_owned_mut().await.expect("build embedded silero-vad");
            assert!(guard.is_some(), "vad slot must be populated after call");
        }
        let guard = be.vad_owned_mut().await.expect("reuse");
        assert!(guard.is_some());
    }

    /// End-to-end live test: build the embedded Silero VAD, load candle
    /// Whisper, stream a synthetic 5-second 440 Hz tone through
    /// `SttBackend::stream`, and assert at least one final emission
    /// arrives (its `text` may be empty — Whisper rarely transcribes
    /// pure sine waves to anything meaningful, and Silero v5 doesn't flag a
    /// pure tone as speech — but the pipeline must reach the `finalize`
    /// path and emit the terminal record).
    ///
    /// Skipped by default (`#[ignore]`) because it downloads candle
    /// whisper-base weights. Unlock with:
    ///
    /// ```sh
    /// cargo nextest run -p blazen-audio-stt --features whisper-streaming \
    ///     stream_emits_at_least_one_chunk_for_synthetic_tone -- --ignored
    /// ```
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "downloads candle whisper-base weights"]
    async fn stream_emits_at_least_one_chunk_for_synthetic_tone() {
        use futures_util::StreamExt;
        use futures_util::stream;

        let cfg = WhisperStreamingConfig {
            // Shorter chunks → less audio needed before a flush fires.
            chunk_seconds: 5.0,
            chunk_overlap_seconds: 1.0,
            ..WhisperStreamingConfig::default()
        };
        let be = WhisperStreamingBackend::new(cfg);

        // 5 s of a 440 Hz tone at amplitude 0.5 — chunked into 1600-
        // sample (100 ms) input items to exercise the leftover-buffer
        // logic.
        let sample_rate = 16_000_usize;
        let total_samples = sample_rate * 5;
        let chunk = 1600_usize;
        let mut samples = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        let items: Vec<Vec<f32>> = samples
            .chunks(chunk)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let input: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>> = Box::pin(stream::iter(items));

        let mut out = be
            .stream(input, None)
            .await
            .expect("stream construction must succeed");
        let mut saw_final = false;
        while let Some(item) = out.next().await {
            let t = item.expect("no pipeline errors on synthetic tone");
            if t.is_final {
                saw_final = true;
            }
        }
        assert!(
            saw_final,
            "expected at least one final emission from `finalize` at EOS"
        );
    }
}
