//! Stable Audio (Stability AI) generation backend.
//!
//! # Status (May 2026)
//!
//! Wave 3.5 of the PR-AUDIO restructure landed a native candle port of
//! the full Stable Audio Open stack (Oobleck VAE + DiT + T5 conditioner +
//! DPM-Solver++ / distilled samplers + safetensors key remap). The port
//! lives behind the `stable-audio` cargo feature.
//!
//! # Feature surface
//!
//! - **`stable-audio` ON** — [`StableAudioBackend`] is the real backend
//!   defined in [`pipeline`]. It loads `stabilityai/stable-audio-open-small`
//!   (default) or `stabilityai/stable-audio-open-1.0`, runs the diffusion
//!   loop on CPU / CUDA / Metal, and decodes through the Oobleck VAE to
//!   `(B, 2, T)` stereo waveforms at 44.1 kHz.
//! - **`stable-audio` OFF** — [`StableAudioBackend`] is a stub that
//!   constructs in `const`-time but surfaces
//!   [`MusicError::NotYetImplemented`] from every `generate_*` entry
//!   point, with a message pointing at the `stable-audio` feature flag
//!   so downstream callers see a clear, actionable error.
//!
//! The type name is stable across both feature modes — callers never
//! need to `cfg` their code on `stable-audio`.

#[cfg(feature = "stable-audio")]
mod conditioner;
#[cfg(feature = "stable-audio")]
mod dit;
#[cfg(feature = "stable-audio")]
mod oobleck;
#[cfg(feature = "stable-audio")]
pub mod pipeline;
#[cfg(feature = "stable-audio")]
mod sampler;
#[cfg(feature = "stable-audio")]
mod weights;

#[cfg(feature = "stable-audio")]
pub use pipeline::{
    StableAudioBackend, StableAudioConfig, StableAudioPipeline, StableAudioVariant,
};

#[cfg(not(feature = "stable-audio"))]
pub use stub::StableAudioBackend;

/// Stub fallback used when the `stable-audio` cargo feature is OFF.
///
/// Every entry point returns [`MusicError::NotYetImplemented`] with a
/// message that names the feature flag, so downstream callers get a
/// clear, actionable error instead of `no such type`.
#[cfg(not(feature = "stable-audio"))]
mod stub {
    use async_trait::async_trait;
    use blazen_audio::{AudioBackend, GeneratedAudio};

    use crate::error::MusicError;
    use crate::traits::MusicBackend;

    /// Long-form message returned by every blocked Stable Audio entry
    /// point when the `stable-audio` feature is OFF. Points the caller
    /// at the feature flag they need to enable to get a real backend.
    pub(crate) const STABLE_AUDIO_NOT_IMPLEMENTED: &str = "Stable Audio support requires the `stable-audio` cargo feature. \
         Rebuild blazen-audio-music with `--features stable-audio` (or one \
         of its supersets) to enable the native candle port of \
         stabilityai/stable-audio-open. The feature pulls in candle-core, \
         candle-nn, candle-transformers, tokenizers, and hf-hub.";

    /// Stable Audio backend — stub fallback (see module docs).
    #[derive(Debug, Clone, Default)]
    pub struct StableAudioBackend {
        _private: (),
    }

    impl StableAudioBackend {
        /// Construct a Stable Audio backend handle. With the
        /// `stable-audio` feature OFF, the returned handle's
        /// `generate_*` calls all surface
        /// [`MusicError::NotYetImplemented`].
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
                    assert!(msg.contains("stable-audio"));
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
}
