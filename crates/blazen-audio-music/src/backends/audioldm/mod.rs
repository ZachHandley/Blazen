//! AudioLDM2 (text-to-audio latent diffusion) generation backend.
//!
//! # Status (May 2026)
//!
//! Wave H.1 is **scaffolding only**: this module exposes
//! [`AudioLdmBackend`] with the correct trait surface but every
//! [`MusicBackend`] entry point returns
//! [`MusicError::NotYetImplemented`]. Wave H.2 lands the real UNet 2D
//! denoiser, DDIM sampler, T5 / FLAN-T5 text encoder, mel-spectrogram
//! VAE, HiFi-GAN vocoder, safetensors weight remap, and end-to-end
//! pipeline.
//!
//! Unlike Stable Audio (which uses a DiT in the shared
//! `blazen-audio-core` primitives), AudioLDM2 uses a UNet 2D denoiser
//! operating on mel-spectrogram latents, so the AudioLDM stack is
//! self-contained inside this module.
//!
//! The backend is gated on the `audioldm` cargo feature, which is OFF
//! by default. The feature is currently empty (no extra dependencies)
//! because the scaffolding placeholders do not reference any model
//! crates yet -- the real candle/tokenizers/hf-hub wiring lands in
//! Wave H.2.

mod pipeline;
mod sampler;
mod text_encoder;
mod unet;
mod vae;
mod vocoder;
mod weights;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, GeneratedAudio};

use crate::error::MusicError;
use crate::traits::MusicBackend;

/// Long-form message returned by every blocked AudioLDM entry point
/// during Wave H.1 scaffolding. Wave H.2 replaces this with the real
/// UNet 2D denoiser + DDIM sampler implementation.
pub(crate) const AUDIOLDM_SCAFFOLDING: &str =
    "AudioLDM is still scaffolding -- wave H.2 lands the UNet 2D denoiser and DDIM sampler";

/// AudioLDM2 text-to-audio latent-diffusion backend (Wave H.1
/// scaffolding -- see module docs).
#[derive(Debug, Clone, Default)]
pub struct AudioLdmBackend {
    _private: (),
}

impl AudioLdmBackend {
    /// Construct an AudioLDM backend handle. During Wave H.1 every
    /// `generate_*` call surfaces [`MusicError::NotYetImplemented`].
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

#[async_trait]
impl AudioBackend for AudioLdmBackend {
    fn id(&self) -> &'static str {
        "audioldm"
    }

    fn provider_kind(&self) -> &'static str {
        "music"
    }
}

#[async_trait]
impl MusicBackend for AudioLdmBackend {
    async fn generate_music(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        Err(MusicError::not_yet_implemented(AUDIOLDM_SCAFFOLDING))
    }

    async fn generate_sfx(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        Err(MusicError::not_yet_implemented(AUDIOLDM_SCAFFOLDING))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn audioldm_music_returns_not_yet_implemented() {
        let backend = AudioLdmBackend::new();
        assert_eq!(backend.id(), "audioldm");
        assert_eq!(backend.provider_kind(), "music");
        let err = backend
            .generate_music("ambient pad", 8.0)
            .await
            .unwrap_err();
        match err {
            MusicError::NotYetImplemented(msg) => {
                assert!(msg.contains("scaffolding"));
            }
            other => panic!("expected NotYetImplemented, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn audioldm_sfx_returns_not_yet_implemented() {
        let backend = AudioLdmBackend::new();
        let err = backend
            .generate_sfx("rain on tin roof", 4.0)
            .await
            .unwrap_err();
        assert!(matches!(err, MusicError::NotYetImplemented(_)));
    }
}
