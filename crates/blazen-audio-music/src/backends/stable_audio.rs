//! Stable Audio (Stability AI) generation backend — placeholder.
//!
//! # Status (May 2026)
//!
//! Stable Audio (the diffusion-based text-to-audio model from Stability AI,
//! released in two public variants — Stable Audio Open 1.0 and Stable Audio
//! 2.0) does **not** have a candle / candle-transformers port today. Unlike
//! MusicGen/AudioGen (which share an EnCodec + autoregressive-transformer
//! architecture and only differ in checkpoints), Stable Audio uses a latent
//! diffusion stack with a custom VAE — porting it is a much larger lift and
//! is intentionally out of scope for the PR-AUDIO restructure.
//!
//! Wave 28 of PR-AUDIO will add `docs/UNSUPPORTED_AUDIO.md` with an entry
//! pointing at the upstream Python reference implementations and the cloud
//! AudioGeneration providers in `blazen-llm` as the recommended path.
//!
//! For now this module ships a no-op [`StableAudioBackend`] that returns
//! [`MusicError::NotYetImplemented`] from both entry points so the type
//! exists in the binding surface and downstream code can route to a clear,
//! actionable error rather than `no such type`.

use async_trait::async_trait;
use blazen_audio::{AudioBackend, GeneratedAudio};

use crate::error::MusicError;
use crate::traits::MusicBackend;

/// Long-form message returned by every blocked Stable Audio entry point.
pub(crate) const STABLE_AUDIO_NOT_IMPLEMENTED: &str = "Stable Audio is not supported in this crate. \
     Stability AI's Stable Audio models use a latent-diffusion architecture \
     (custom VAE + diffusion transformer) that has no candle / \
     candle-transformers port today. PR-AUDIO Wave 28 documents this gap \
     in docs/UNSUPPORTED_AUDIO.md and points at the upstream Python \
     reference implementation and the cloud AudioGeneration providers in \
     blazen-llm as the recommended path.";

/// Stable Audio backend — placeholder scaffold (see module docs).
#[derive(Debug, Clone, Default)]
pub struct StableAudioBackend {
    _private: (),
}

impl StableAudioBackend {
    /// Construct a Stable Audio backend handle.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

#[async_trait]
impl AudioBackend for StableAudioBackend {
    fn id(&self) -> &'static str {
        "stable-audio"
    }

    fn provider_kind(&self) -> &'static str {
        "music"
    }
}

#[async_trait]
impl MusicBackend for StableAudioBackend {
    async fn generate_music(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        Err(MusicError::not_yet_implemented(
            STABLE_AUDIO_NOT_IMPLEMENTED,
        ))
    }

    async fn generate_sfx(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        Err(MusicError::not_yet_implemented(
            STABLE_AUDIO_NOT_IMPLEMENTED,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stable_audio_music_returns_not_yet_implemented() {
        let backend = StableAudioBackend::new();
        assert_eq!(backend.id(), "stable-audio");
        assert_eq!(backend.provider_kind(), "music");
        let err = backend
            .generate_music("orchestral build", 8.0)
            .await
            .unwrap_err();
        match err {
            MusicError::NotYetImplemented(msg) => {
                assert!(msg.contains("Stable Audio"));
            }
            other => panic!("expected NotYetImplemented, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stable_audio_sfx_returns_not_yet_implemented() {
        let backend = StableAudioBackend::new();
        let err = backend.generate_sfx("ocean waves", 4.0).await.unwrap_err();
        assert!(matches!(err, MusicError::NotYetImplemented(_)));
    }
}
