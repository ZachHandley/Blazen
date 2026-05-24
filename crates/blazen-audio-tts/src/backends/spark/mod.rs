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
//! - **Wave S.1** (this commit): scaffolding only.
//!   [`SparkTtsBackend`] and [`SparkTtsConfig`] exist; every
//!   [`TtsBackend`] method returns [`TtsError::Unsupported`]. The
//!   five sub-modules carry one-line doc-comments and no code.
//! - **Wave S.2**: real [`bicodec`], [`decoder`], [`tokenizer`],
//!   [`weights`], and [`pipeline`] implementations land, wiring the
//!   live `synthesize` / `clone_voice` paths.
//!
//! # License
//!
//! Upstream `SparkAudio/Spark-TTS` ships Apache-2.0 source with
//! CC-BY-NC-SA-4.0 weights. The split is preserved here: the Rust
//! port is compatible with Apache-2.0, but downstream users of the
//! published weights must honour the non-commercial weights license.

#![cfg(feature = "spark-tts")]

mod bicodec;
mod decoder;
mod pipeline;
mod tokenizer;
mod weights;

use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, CloneVoiceRequest, GeneratedAudio, VoiceHandle};
use futures_core::Stream;

use crate::traits::{StreamingAudioChunk, TtsBackend};
use crate::{TtsError, TtsOptions};

/// Stable backend-id prefix surfaced via [`AudioBackend::id`].
pub const SPARK_BACKEND_ID_PREFIX: &str = "spark-tts";

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
}

impl Default for SparkTtsConfig {
    fn default() -> Self {
        Self {
            model_id: "SparkAudio/Spark-TTS-0.5B".to_owned(),
        }
    }
}

/// Spark-TTS backend handle.
///
/// Construct via [`SparkTtsBackend::new`]. Weights load lazily on the
/// first [`TtsBackend::synthesize`] / [`TtsBackend::clone_voice`] call
/// (matching the lazy-load convention used by [`super::bark`] and
/// [`super::f5`]).
#[derive(Debug, Clone)]
pub struct SparkTtsBackend {
    id: String,
    config: SparkTtsConfig,
}

impl SparkTtsBackend {
    /// Build a new Spark-TTS backend with the given configuration.
    ///
    /// No weights are downloaded at construction time — the underlying
    /// pipeline is materialised on the first synthesis call once Wave
    /// S.2 lands. Until then every [`TtsBackend`] method returns
    /// [`TtsError::Unsupported`].
    #[must_use]
    pub fn new(config: SparkTtsConfig) -> Self {
        let id = format!("{SPARK_BACKEND_ID_PREFIX}:{}", config.model_id);
        Self { id, config }
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
        false
    }
}

#[async_trait]
impl TtsBackend for SparkTtsBackend {
    async fn synthesize(
        &self,
        _text: &str,
        _options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        Err(TtsError::Unsupported(
            "Spark-TTS Wave S.1 scaffolding — Wave S.2 lands BiCodec + Qwen2.5 decoder".into(),
        ))
    }

    async fn clone_voice(&self, _request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(
            "Spark-TTS Wave S.1 scaffolding — Wave S.2 lands BiCodec + Qwen2.5 decoder".into(),
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
    async fn synthesize_returns_wave_s1_unsupported() {
        let backend = SparkTtsBackend::default();
        let err = backend
            .synthesize("hello", &TtsOptions::default())
            .await
            .expect_err("scaffold must surface Unsupported");
        match err {
            TtsError::Unsupported(msg) => {
                assert!(msg.contains("Wave S.2"), "msg = {msg}");
                assert!(msg.contains("BiCodec"), "msg = {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
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
    async fn clone_voice_returns_wave_s1_unsupported() {
        let backend = SparkTtsBackend::default();
        let req = CloneVoiceRequest {
            name: "alice".to_owned(),
            audio_bytes: vec![1, 2, 3, 4],
            transcript: Some("hello".to_owned()),
        };
        let err = backend
            .clone_voice(req)
            .await
            .expect_err("scaffold must surface Unsupported");
        assert!(matches!(err, TtsError::Unsupported(_)));
    }
}
