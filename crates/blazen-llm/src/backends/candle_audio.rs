//! Bridge between [`blazen_audio_candle::CandleAudioProvider`] and the
//! [`AudioGeneration`](crate::compute::AudioGeneration) trait.
//!
//! ## Current shape
// Mentions of EnCodec / MusicGen / AudioGen / blazen-* crates frequently
// fire `clippy::doc_markdown` even though they're product names, not Rust
// identifiers. Allowed locally to keep the docs readable.
#![allow(clippy::doc_markdown)]
//!
//! Per `PR6_PLAN.md` §3c the upstream `candle-transformers` 0.10.2 ships
//! EnCodec but does **not** ship the MusicGen / AudioGen autoregressive
//! decoder. This bridge therefore:
//!
//! - Surfaces `text_to_speech` as
//!   [`BlazenError::Unsupported`](crate::error::BlazenError) — `blazen-audio-candle`
//!   is for music / SFX, not speech (use `blazen-audio-tts` for TTS).
//! - Surfaces `generate_music` and `generate_sfx` as `Unsupported` until
//!   the upstream MusicGen autoregressive head lands; the upstream gap is
//!   propagated 1:1 from `CandleAudioError::NotYetImplemented`.
//! - Exposes inherent `encode_audio` / `decode_audio` helpers on
//!   [`CandleAudioProvider`] for callers who want the EnCodec codec
//!   directly. These do not fit the `ComputeProvider` job-API mould.
//!
//! The bridge is `#[cfg(feature = "candle-audio")]` in the parent module so
//! it stays out of builds that don't link `blazen-audio-candle`.

use async_trait::async_trait;
use blazen_audio_candle::{CandleAudioError, CandleAudioProvider, EncodecModel};

use crate::compute::{
    AudioGeneration, AudioResult, ComputeProvider, ComputeRequest, ComputeResult, JobHandle,
    JobStatus, MusicRequest, SpeechRequest,
};
use crate::error::BlazenError;

// ---------------------------------------------------------------------------
// Error translation
// ---------------------------------------------------------------------------

/// Convert a [`CandleAudioError`] into the public [`BlazenError`] shape.
///
/// `NotYetImplemented` and `EngineNotAvailable` both translate to
/// [`BlazenError::Unsupported`] — they're permanent (within the current
/// build) and the caller should pick a different provider.
fn to_blazen(err: CandleAudioError) -> BlazenError {
    match err {
        CandleAudioError::EngineNotAvailable | CandleAudioError::NotYetImplemented(_) => {
            BlazenError::unsupported(err.to_string())
        }
        CandleAudioError::InvalidInput(msg) => BlazenError::provider("candle-audio", msg),
        other => BlazenError::provider("candle-audio", other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for CandleAudioProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider_id(&self) -> &str {
        "candle-audio"
    }

    async fn submit(&self, _request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        Err(BlazenError::unsupported(
            "candle-audio runs locally and does not use the ComputeRequest \
             job API; call `AudioGeneration::generate_music` / `generate_sfx` \
             directly instead",
        ))
    }

    async fn status(&self, _job: &JobHandle) -> Result<JobStatus, BlazenError> {
        Err(BlazenError::unsupported(
            "candle-audio does not expose a job queue -- generation is synchronous",
        ))
    }

    async fn result(&self, _job: JobHandle) -> Result<ComputeResult, BlazenError> {
        Err(BlazenError::unsupported(
            "candle-audio does not expose a job queue -- generation is synchronous",
        ))
    }

    async fn cancel(&self, _job: &JobHandle) -> Result<(), BlazenError> {
        Err(BlazenError::unsupported(
            "candle-audio generation is synchronous and cannot be cancelled",
        ))
    }
}

// ---------------------------------------------------------------------------
// AudioGeneration
// ---------------------------------------------------------------------------

#[async_trait]
impl AudioGeneration for CandleAudioProvider {
    async fn text_to_speech(&self, _request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        Err(BlazenError::unsupported(
            "candle-audio is for music and sound-effect generation; \
             for text-to-speech use blazen-audio-piper (or blazen-audio-tts \
             when it lands)",
        ))
    }

    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        // Forward to the underlying model. Until the autoregressive head
        // lands, this returns `NotYetImplemented` which we surface as
        // `BlazenError::Unsupported` with the full long-form message.
        let duration = request.duration_seconds.unwrap_or(10.0);
        let _ = self
            .generate(&request.prompt, duration)
            .await
            .map_err(to_blazen)?;
        // Unreachable while the upstream model gap remains, but kept for
        // when it lands: once `generate` returns real PCM, wire it into an
        // `AudioResult` here.
        Err(BlazenError::unsupported(
            "candle-audio: generate_music returned PCM but the bridge has \
             not yet been wired to wrap it into an AudioResult -- update \
             crates/blazen-llm/src/backends/candle_audio.rs",
        ))
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        // Same story as generate_music — AudioGen scaffold lives in the
        // same world as MusicGen and is gated on the same upstream port.
        let duration = request.duration_seconds.unwrap_or(5.0);
        let _ = self
            .generate(&request.prompt, duration)
            .await
            .map_err(to_blazen)?;
        Err(BlazenError::unsupported(
            "candle-audio: generate_sfx returned PCM but the bridge has \
             not yet been wired to wrap it into an AudioResult -- update \
             crates/blazen-llm/src/backends/candle_audio.rs",
        ))
    }
}

// ---------------------------------------------------------------------------
// Codec helpers (inherent, not exposed via a trait)
// ---------------------------------------------------------------------------

/// Inherent codec helpers on top of [`CandleAudioProvider`].
///
/// The EnCodec encode/decode API does not fit any `ComputeProvider`
/// capability trait — it's a deterministic codec round-trip, not a
/// generative job — so it lives as inherent methods on the bridge struct.
///
/// Both methods require the provider to wrap an
/// [`blazen_audio_candle::EncodecModel`] specifically; callers pass that
/// directly to [`CandleAudioProvider::new`] then `use` this trait to get
/// the codec methods in scope.
#[allow(dead_code)] // Public-API trait — consumers import + impl-by-default.
pub trait CandleAudioCodec {
    /// Encode mono PCM samples (`f32` in `[-1.0, 1.0]`) into EnCodec tokens.
    ///
    /// # Errors
    ///
    /// Surfaces [`BlazenError::Provider`] when the inner model is not an
    /// [`EncodecModel`], when the sample rate disagrees with the model's
    /// native rate, or when candle inference fails.
    fn encode_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> impl std::future::Future<Output = Result<Vec<u32>, BlazenError>> + Send;

    /// Decode EnCodec tokens back into mono PCM samples.
    ///
    /// # Errors
    ///
    /// Surfaces [`BlazenError::Provider`] when the inner model is not an
    /// [`EncodecModel`] or when candle inference fails.
    fn decode_audio(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, BlazenError>> + Send;
}

impl CandleAudioCodec for CandleAudioProvider {
    async fn encode_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u32>, BlazenError> {
        let encodec = downcast_encodec(self)?;
        encodec
            .encode_pcm(samples, sample_rate)
            .await
            .map_err(to_blazen)
    }

    async fn decode_audio(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> Result<Vec<f32>, BlazenError> {
        let encodec = downcast_encodec(self)?;
        encodec
            .decode_tokens(tokens, num_codebooks)
            .await
            .map_err(to_blazen)
    }
}

/// Borrow the inner [`EncodecModel`] from a [`CandleAudioProvider`], or
/// raise a clear error if the provider wraps a different model.
#[allow(dead_code)] // Helper for the `CandleAudioCodec` public-API trait.
fn downcast_encodec(provider: &CandleAudioProvider) -> Result<&EncodecModel, BlazenError> {
    let any_ref = provider.model().as_ref().as_any();
    any_ref.downcast_ref::<EncodecModel>().ok_or_else(|| {
        BlazenError::provider(
            "candle-audio",
            format!(
                "encode_audio/decode_audio require an EncodecModel, but this \
                 provider wraps `{}`",
                provider.name()
            ),
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_audio_candle::MusicgenModel;
    use std::sync::Arc;

    fn musicgen_provider() -> CandleAudioProvider {
        CandleAudioProvider::new(Arc::new(MusicgenModel::default()))
    }

    #[tokio::test]
    async fn provider_id_is_candle_audio() {
        let provider = musicgen_provider();
        assert_eq!(ComputeProvider::provider_id(&provider), "candle-audio");
    }

    #[tokio::test]
    async fn tts_is_unsupported() {
        let provider = musicgen_provider();
        let err = AudioGeneration::text_to_speech(&provider, SpeechRequest::new("hello"))
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn generate_music_propagates_not_yet_implemented() {
        let provider = musicgen_provider();
        let request = MusicRequest::new("upbeat jazz");
        let err = AudioGeneration::generate_music(&provider, request)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, BlazenError::Unsupported { .. }),
            "expected Unsupported, got {err:?}"
        );
        assert!(
            msg.contains("MusicGen") || msg.contains("autoregressive"),
            "error should mention the MusicGen gap, got: {msg}"
        );
    }

    #[tokio::test]
    async fn generate_sfx_propagates_not_yet_implemented() {
        let provider = musicgen_provider();
        let request = MusicRequest::new("rain on a tin roof");
        let err = AudioGeneration::generate_sfx(&provider, request)
            .await
            .unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn submit_is_unsupported() {
        let provider = musicgen_provider();
        let request = ComputeRequest {
            model: "musicgen-small".into(),
            input: serde_json::Value::Null,
            webhook: None,
        };
        let err = provider.submit(request).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn cancel_is_unsupported() {
        let provider = musicgen_provider();
        let handle = JobHandle {
            id: "fake".into(),
            provider: "candle-audio".into(),
            model: "musicgen-small".into(),
            submitted_at: chrono::Utc::now(),
        };
        let err = provider.cancel(&handle).await.unwrap_err();
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn encode_on_musicgen_provider_errors_clearly() {
        let provider = musicgen_provider();
        let err = provider
            .encode_audio(&[0.0_f32; 16], 24_000)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("EncodecModel"),
            "expected error to mention EncodecModel, got: {msg}"
        );
    }
}
