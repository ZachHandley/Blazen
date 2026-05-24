//! Spark-TTS backend (`SparkAudio/Spark-TTS`) — `BiCodec` neural audio
//! codec + Qwen2.5-0.5B autoregressive decoder for zero-shot
//! voice-cloning text-to-speech.
//!
//! # Architecture
//!
//! 1. **Text tokenizer** ([`tokenizer`]) — Spark's text frontend
//!    converts input text into the token vocabulary consumed by the
//!    Qwen2.5 backbone.
//! 2. **Qwen2.5-0.5B autoregressive decoder** ([`decoder`]) —
//!    transformer LM that emits `BiCodec` semantic + global tokens
//!    conditioned on text and an optional reference-audio prompt.
//! 3. **`BiCodec` neural codec** ([`bicodec`]) — novel codec design
//!    that decodes Spark's semantic + global tokens into a 16 kHz
//!    waveform without a separate vocoder.
//! 4. **Pipeline** ([`pipeline`]) — orchestrates tokenizer → decoder →
//!    `BiCodec` and surfaces the [`TtsBackend::synthesize`] /
//!    [`TtsBackend::clone_voice`] entrypoints.
//!
//! # Wave plan
//!
//! - **Wave S.1**: scaffolding only.
//! - **Wave S.2.1–S.2.3**: [`bicodec`], [`decoder`], [`tokenizer`]
//!   land with full unit coverage but no live wiring.
//! - **Wave S.2.4** (this commit): [`weights`] + [`pipeline`] glue the
//!   sub-modules into a working text-to-speech path. The
//!   [`TtsBackend::synthesize`] entrypoint now performs the real
//!   download → tokenizer → LLM → `BiCodec` → WAV pipeline.
//! - **Wave S.2.5+**: zero-shot voice cloning (wav2vec2-XLS-R port +
//!   reference-audio `BiCodec::tokenize` integration) and streaming
//!   synthesis remain `Unsupported` pending future waves.
//!
//! # License
//!
//! Upstream `SparkAudio/Spark-TTS` ships Apache-2.0 source with
//! CC-BY-NC-SA-4.0 weights. The split is preserved here: the Rust
//! port is compatible with Apache-2.0, but downstream users of the
//! published weights must honour the non-commercial weights license.
//! The backend emits a single [`blazen_audio::warn_nc_once`] the first
//! time it materialises a pipeline.

#![cfg(feature = "spark-tts")]

mod bicodec;
mod decoder;
mod pipeline;
mod tokenizer;
mod weights;

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{
    AudioBackend, AudioFormat, CloneVoiceRequest, GeneratedAudio, VoiceHandle, warn_nc_once,
};
use candle_core::Device;
use futures_core::Stream;
use tokio::sync::OnceCell;

use self::pipeline::{SPARK_TTS_SAMPLE_RATE_HZ, SparkPipeline};
use crate::traits::{StreamingAudioChunk, TtsBackend};
use crate::{TtsError, TtsOptions};

/// Stable backend-id prefix surfaced via [`AudioBackend::id`].
pub const SPARK_BACKEND_ID_PREFIX: &str = "spark-tts";

/// License string surfaced via [`warn_nc_once`] on the first
/// [`SparkPipeline`] materialisation. Spark-TTS weights are
/// `CC-BY-NC-SA-4.0` per
/// <https://huggingface.co/SparkAudio/Spark-TTS-0.5B>.
const SPARK_WEIGHTS_LICENSE: &str = "CC-BY-NC-SA-4.0";

/// Configuration knobs for [`SparkTtsBackend`].
///
/// Defaults match upstream `SparkAudio/Spark-TTS`: the
/// `SparkAudio/Spark-TTS-0.5B` checkpoint on Hugging Face Hub, which
/// bundles the Qwen2.5-0.5B decoder, the `BiCodec` weights, and the
/// shared text tokenizer.
#[derive(Debug, Clone)]
pub struct SparkTtsConfig {
    /// Hugging Face repo id for the Spark-TTS bundle. Default
    /// `"SparkAudio/Spark-TTS-0.5B"`.
    pub model_id: String,
    /// Optional pre-resolved bundle directory. When `Some(_)` the
    /// backend skips the [`weights::ensure_downloaded`] step entirely
    /// and loads from the supplied path (containing `LLM/` +
    /// `BiCodec/`). When `None` (default) the bundle is downloaded +
    /// cached under [`blazen_model_cache::ModelCache`].
    pub model_dir: Option<PathBuf>,
    /// Optional revision (branch / tag / commit SHA) to pin against.
    /// Forwarded to [`weights::ensure_downloaded`]. `None` resolves to
    /// `main`.
    pub revision: Option<String>,
}

impl Default for SparkTtsConfig {
    fn default() -> Self {
        Self {
            model_id: "SparkAudio/Spark-TTS-0.5B".to_owned(),
            model_dir: None,
            revision: None,
        }
    }
}

/// Spark-TTS backend handle.
///
/// Construct via [`SparkTtsBackend::new`]. The underlying
/// [`SparkPipeline`] is materialised lazily on the first
/// [`TtsBackend::synthesize`] call (matching the lazy-load convention
/// used by [`super::bark`] and [`super::f5`]).
///
/// `Debug` is hand-written to redact the lazily-loaded pipeline
/// (`SparkPipeline` wraps `Qwen2Model` + `BiCodec` which don't
/// implement `Debug`); we surface only the human-meaningful fields plus
/// a load-state flag so debug-printing the backend stays cheap.
#[derive(Clone)]
pub struct SparkTtsBackend {
    id: String,
    config: SparkTtsConfig,
    inner: Arc<OnceCell<Arc<SparkPipeline>>>,
}

impl std::fmt::Debug for SparkTtsBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparkTtsBackend")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("loaded", &self.inner.initialized())
            .finish()
    }
}

impl SparkTtsBackend {
    /// Build a new Spark-TTS backend with the given configuration.
    ///
    /// No weights are downloaded at construction time — the underlying
    /// pipeline is materialised on the first synthesis call.
    #[must_use]
    pub fn new(config: SparkTtsConfig) -> Self {
        let id = format!("{SPARK_BACKEND_ID_PREFIX}:{}", config.model_id);
        Self {
            id,
            config,
            inner: Arc::new(OnceCell::new()),
        }
    }

    /// The resolved model id this backend was configured with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    /// Access the underlying [`SparkTtsConfig`] (read-only).
    #[must_use]
    pub fn config(&self) -> &SparkTtsConfig {
        &self.config
    }

    /// Materialise (or fetch the cached) [`SparkPipeline`] for this
    /// backend's config. Subsequent calls reuse the cached pipeline.
    async fn ensure_loaded(&self) -> Result<Arc<SparkPipeline>, TtsError> {
        let cfg = self.config.clone();
        let id = self.id.clone();
        let pipeline = self
            .inner
            .get_or_try_init(|| async move {
                // Resolve bundle directory: explicit override > HF download.
                let bundle_dir = match cfg.model_dir.clone() {
                    Some(dir) => dir,
                    None => weights::ensure_downloaded(&cfg.model_id, cfg.revision.as_deref())
                        .await
                        .map_err(TtsError::from)?,
                };

                // Device selection mirrors the F5 + Bark backends: CPU
                // by default. GPU selection is deferred to a future
                // wave (it requires plumbing a `device` knob onto
                // SparkTtsConfig + matching CUDA / Metal feature
                // gating, which is out of scope for S.2.4).
                let device = Device::Cpu;

                let pipeline = SparkPipeline::load(&bundle_dir, device)
                    .await
                    .map_err(TtsError::from)?;

                // CC-BY-NC-SA-4.0 weights: emit a one-shot warn once
                // the first inference is about to run. Apache-2.0 code
                // is unconstrained, but downstream consumers of the
                // weights need to honour the NC license.
                warn_nc_once(&id, &cfg.model_id, SPARK_WEIGHTS_LICENSE);

                Ok::<Arc<SparkPipeline>, TtsError>(Arc::new(pipeline))
            })
            .await?;
        Ok(Arc::clone(pipeline))
    }
}

impl Default for SparkTtsBackend {
    fn default() -> Self {
        Self::new(SparkTtsConfig::default())
    }
}

#[async_trait]
impl AudioBackend for SparkTtsBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "tts"
    }

    async fn is_loaded(&self) -> bool {
        self.inner.initialized()
    }
}

#[async_trait]
impl TtsBackend for SparkTtsBackend {
    async fn synthesize(
        &self,
        text: &str,
        _options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        let pipeline = self.ensure_loaded().await?;
        let wav_bytes = pipeline
            .synthesize_wav(text)
            .await
            .map_err(TtsError::from)?;
        Ok(GeneratedAudio {
            bytes: wav_bytes,
            format: AudioFormat::Wav,
            sample_rate: SPARK_TTS_SAMPLE_RATE_HZ,
            channels: 1,
            // Duration is left unset here: the WAV header carries the
            // canonical sample count, and recomputing it would require
            // re-decoding the bytes we just encoded. Callers that need
            // it can parse the RIFF `data` chunk size directly.
            duration_seconds: None,
        })
    }

    async fn clone_voice(&self, _request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(
            "voice cloning requires wav2vec2-XLS-R; pending separate port".into(),
        ))
    }

    async fn stream_synthesize(
        &self,
        _text: &str,
        _voice: Option<&str>,
        _options: TtsOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingAudioChunk, TtsError>> + Send>>, TtsError>
    {
        Err(TtsError::Unsupported(
            "Spark-TTS Wave S.2 scaffolding — stream_synthesize will land once BiCodec + Qwen2.5 decoder ship".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spark_config_defaults_match_upstream() {
        let cfg = SparkTtsConfig::default();
        assert_eq!(cfg.model_id, "SparkAudio/Spark-TTS-0.5B");
        assert!(cfg.model_dir.is_none());
        assert!(cfg.revision.is_none());
    }

    #[test]
    fn spark_backend_id_includes_model() {
        let backend = SparkTtsBackend::new(SparkTtsConfig::default());
        assert_eq!(backend.id(), "spark-tts:SparkAudio/Spark-TTS-0.5B");
        assert_eq!(backend.model_id(), "SparkAudio/Spark-TTS-0.5B");
        assert_eq!(backend.provider_kind(), "tts");
    }

    #[tokio::test]
    async fn is_loaded_starts_false() {
        let backend = SparkTtsBackend::default();
        assert!(!backend.is_loaded().await);
    }

    #[tokio::test]
    async fn synthesize_with_invalid_model_dir_surfaces_model_load_error() {
        // Force the lazy loader to take the explicit `model_dir` branch
        // and try to read `LLM/config.json` from a nonexistent path. No
        // network access is performed.
        let backend = SparkTtsBackend::new(SparkTtsConfig {
            model_id: "SparkAudio/Spark-TTS-0.5B".into(),
            model_dir: Some(PathBuf::from("/nonexistent/spark-tts-mod-test")),
            revision: None,
        });
        let err = backend
            .synthesize("hello", &TtsOptions::default())
            .await
            .expect_err("nonexistent model_dir must error");
        match err {
            TtsError::ModelLoad(msg) => {
                assert!(
                    msg.contains("/nonexistent/spark-tts-mod-test"),
                    "expected path in error, got: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
        // After a failed init, the cell is still uninitialised so
        // subsequent calls re-try (matches tokio's OnceCell semantics).
        assert!(!backend.is_loaded().await);
    }

    #[tokio::test]
    async fn stream_synthesize_returns_clear_pending_error() {
        let backend = SparkTtsBackend::default();
        let result = backend
            .stream_synthesize("hello", None, TtsOptions::default())
            .await;
        match result {
            Err(TtsError::Unsupported(msg)) => {
                assert!(msg.contains("Wave S.2"), "msg = {msg}");
                assert!(msg.contains("BiCodec"), "msg = {msg}");
            }
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("scaffold must surface Unsupported"),
        }
    }

    #[tokio::test]
    async fn clone_voice_returns_pending_wav2vec2_unsupported() {
        let backend = SparkTtsBackend::default();
        let req = CloneVoiceRequest {
            name: "alice".to_owned(),
            audio_bytes: vec![1, 2, 3, 4],
            transcript: Some("hello".to_owned()),
        };
        let err = backend
            .clone_voice(req)
            .await
            .expect_err("voice cloning must surface Unsupported");
        match err {
            TtsError::Unsupported(msg) => {
                assert!(
                    msg.contains("wav2vec2"),
                    "expected wav2vec2 hint, got: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
