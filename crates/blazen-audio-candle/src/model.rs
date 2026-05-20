//! [`AudioModel`] trait and the [`CandleAudioProvider`] wrapper.
//!
//! `blazen-audio-candle` exposes a tiny capability trait — `AudioModel` —
//! that every concrete model in this crate (EnCodec, MusicGen-when-it-lands,
//! AudioGen-when-it-lands, ...) implements. The [`CandleAudioProvider`]
//! holds an `Arc<dyn AudioModel>` and is what the `blazen-llm` bridge
//! actually wraps.
//!
//! ## Why a trait at all?
//!
//! The crate ships EnCodec **now** (functional codec, no autoregressive
//! head) and MusicGen / AudioGen scaffolds that return
//! [`CandleAudioError::NotYetImplemented`] until the upstream
//! `candle-transformers` adds the autoregressive decoder. By unifying
//! everything behind one trait, the `blazen-llm` backend bridge stays
//! identical regardless of which model the caller picks — only the trait
//! impl moves from "Unsupported" to "real" as upstream coverage grows.
//!
//! See `/home/zach/.cache/blazen-pr6-research/PR6_PLAN.md` §3c and §4c for
//! the full rationale.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;

/// A candle-backed audio generator.
///
/// All methods are async to leave room for blocking weight loads and
/// background-thread inference dispatch even though the simplest
/// implementations are inherently CPU-bound.
#[async_trait]
pub trait AudioModel: Send + Sync + std::any::Any {
    /// Human-readable model identifier (e.g. `"musicgen-small"`).
    fn name(&self) -> &str;

    /// Native sample rate (Hz) the model produces. EnCodec is 24 kHz by
    /// default; MusicGen is 32 kHz; AudioGen is 16 kHz.
    fn sample_rate(&self) -> u32;

    /// Generate `duration_seconds` of mono PCM samples (range `[-1.0, 1.0]`)
    /// for the given text prompt.
    ///
    /// Returns [`crate::error::CandleAudioError::NotYetImplemented`] for
    /// models whose autoregressive head is not yet ported to candle.
    async fn generate(&self, prompt: &str, duration_seconds: f32) -> Result<Vec<f32>>;

    /// Upcast to [`std::any::Any`] so downstream bridges (e.g. the
    /// `candle_audio` backend bridge in `blazen-llm`) can downcast to a
    /// concrete model type when they need codec-specific methods that
    /// don't fit the generative trait surface.
    ///
    /// Implementations are usually trivial: `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The public provider type wired into `blazen-llm` via the
/// `backends::candle_audio` bridge.
///
/// Holds the actual model behind an `Arc<dyn AudioModel>` so users can swap
/// implementations (e.g. EnCodec-as-passthrough for tests, MusicGen for
/// real music generation) without changing the surrounding glue.
#[derive(Clone)]
pub struct CandleAudioProvider {
    model: Arc<dyn AudioModel>,
}

impl CandleAudioProvider {
    /// Wrap an existing [`AudioModel`] implementation.
    #[must_use]
    pub fn new(model: Arc<dyn AudioModel>) -> Self {
        Self { model }
    }

    /// Borrow the underlying model.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn AudioModel> {
        &self.model
    }

    /// Forward to [`AudioModel::name`].
    #[must_use]
    pub fn name(&self) -> &str {
        self.model.name()
    }

    /// Forward to [`AudioModel::sample_rate`].
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.model.sample_rate()
    }

    /// Forward to [`AudioModel::generate`].
    ///
    /// # Errors
    ///
    /// Propagates [`crate::error::CandleAudioError`] from the underlying
    /// model — most commonly `NotYetImplemented` for MusicGen/AudioGen, or
    /// `EngineNotAvailable` if the crate was built without `--features engine`.
    pub async fn generate(&self, prompt: &str, duration_seconds: f32) -> Result<Vec<f32>> {
        self.model.generate(prompt, duration_seconds).await
    }
}

impl std::fmt::Debug for CandleAudioProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleAudioProvider")
            .field("model", &self.model.name())
            .field("sample_rate", &self.model.sample_rate())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeModel;

    #[async_trait]
    impl AudioModel for FakeModel {
        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "fake"
        }
        fn sample_rate(&self) -> u32 {
            16_000
        }
        async fn generate(&self, _prompt: &str, _duration: f32) -> Result<Vec<f32>> {
            Ok(vec![0.0; 16])
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn provider_forwards_to_model() {
        let provider = CandleAudioProvider::new(Arc::new(FakeModel));
        assert_eq!(provider.name(), "fake");
        assert_eq!(provider.sample_rate(), 16_000);
        let samples = provider.generate("anything", 0.001).await.unwrap();
        assert_eq!(samples.len(), 16);
    }

    #[test]
    fn debug_includes_name_and_sample_rate() {
        let provider = CandleAudioProvider::new(Arc::new(FakeModel));
        let dbg = format!("{provider:?}");
        assert!(dbg.contains("fake"));
        assert!(dbg.contains("16000"));
    }
}
