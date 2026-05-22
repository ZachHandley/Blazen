//! `MaskGCT` text-to-speech backend (Amphion) — masked-generative
//! transformer pipeline producing acoustic codec tokens, with zero-shot
//! voice cloning via prompt-conditioned non-autoregressive decoding.
//!
//! Upstream: <https://github.com/open-mmlab/Amphion> (MIT-licensed).
//! Canonical HF checkpoint: `amphion/MaskGCT`. Paper:
//! <https://arxiv.org/abs/2409.00750>.
//!
//! # Architecture
//!
//! `MaskGCT` synthesises speech in two masked-generative stages plus a
//! neural codec + vocoder:
//!
//! 1. **Text → semantic** ([`t2s`]): `Llama-NAR` non-autoregressive
//!    masked transformer maps text tokens to semantic tokens.
//! 2. **Semantic → acoustic** ([`s2a`]): `SoundStorm` masked transformer
//!    maps semantic tokens to multi-codebook acoustic tokens.
//! 3. **Codec** ([`codec`]): `RepCodec` / DAC encode-decode of acoustic
//!    tokens.
//! 4. **Vocoder**: `Vocos` decodes acoustic features to a waveform (wired
//!    in via the codec layer).
//!
//! # Wave plan
//!
//! - **Wave M.1** (this commit): scaffolding only. The backend struct
//!   exists, the [`TtsBackend`] impl returns [`TtsError::Unsupported`]
//!   from every method until the real pipeline lands.
//! - **Wave M.2**: real `Llama-NAR` T2S + `SoundStorm` S2A masked
//!   transformers in [`t2s`] / [`s2a`], wired together by [`pipeline`]
//!   via [`codec`] + [`weights`].

#![cfg(feature = "maskgct")]

mod codec;
mod pipeline;
mod s2a;
mod t2s;
mod weights;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, CloneVoiceRequest, GeneratedAudio, VoiceHandle};

use crate::traits::TtsBackend;
use crate::{TtsError, TtsOptions};

/// Stable backend identifier, exposed at runtime via
/// [`AudioBackend::id`]. The trailing colon mirrors the other backends
/// (`bark:suno/bark-small`, `f5:SWivid/F5-TTS`).
pub const MASKGCT_BACKEND_ID_PREFIX: &str = "maskgct";

/// `MaskGCT` text-to-speech backend.
///
/// Construct via [`MaskGctBackend::new`]. All synthesis / voice methods
/// currently return [`TtsError::Unsupported`] — Wave M.2 lands the real
/// `Llama-NAR` T2S and `SoundStorm` S2A masked transformers.
#[derive(Debug, Clone)]
pub struct MaskGctBackend {
    id: String,
    config: MaskGctConfig,
}

/// Configuration knobs for [`MaskGctBackend`].
#[derive(Debug, Clone)]
pub struct MaskGctConfig {
    /// Hugging Face repo id for the `MaskGCT` checkpoint bundle. Defaults
    /// to `"amphion/MaskGCT"`.
    pub model_id: String,
}

impl Default for MaskGctConfig {
    fn default() -> Self {
        Self {
            model_id: "amphion/MaskGCT".to_owned(),
        }
    }
}

impl MaskGctBackend {
    /// Construct a new `MaskGCT` backend with the given configuration.
    #[must_use]
    pub fn new(config: MaskGctConfig) -> Self {
        let id = format!("{MASKGCT_BACKEND_ID_PREFIX}:{}", config.model_id);
        Self { id, config }
    }

    /// The resolved model id this backend was configured with.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }
}

impl Default for MaskGctBackend {
    fn default() -> Self {
        Self::new(MaskGctConfig::default())
    }
}

#[async_trait]
impl AudioBackend for MaskGctBackend {
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
impl TtsBackend for MaskGctBackend {
    async fn synthesize(
        &self,
        _text: &str,
        _options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        Err(TtsError::Unsupported(
            "MaskGCT Wave M.1 scaffolding — Wave M.2 lands T2S + S2A masked transformers".into(),
        ))
    }

    async fn clone_voice(&self, _request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(
            "MaskGCT Wave M.1 scaffolding — Wave M.2 lands T2S + S2A masked transformers".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_amphion_reference() {
        let cfg = MaskGctConfig::default();
        assert_eq!(cfg.model_id, "amphion/MaskGCT");
    }

    #[test]
    fn new_sets_id_from_model_id() {
        let backend = MaskGctBackend::new(MaskGctConfig::default());
        assert_eq!(backend.id(), "maskgct:amphion/MaskGCT");
        assert_eq!(backend.model_id(), "amphion/MaskGCT");
        assert_eq!(backend.provider_kind(), "tts");
    }

    #[tokio::test]
    async fn synthesize_returns_unsupported_during_scaffolding() {
        let backend = MaskGctBackend::default();
        let err = backend
            .synthesize("hello", &TtsOptions::default())
            .await
            .expect_err("Wave M.1 must return Unsupported");
        assert!(matches!(err, TtsError::Unsupported(_)));
    }

    #[tokio::test]
    async fn clone_voice_returns_unsupported_during_scaffolding() {
        let backend = MaskGctBackend::default();
        let req = CloneVoiceRequest {
            name: "test".to_owned(),
            audio_bytes: Vec::new(),
            transcript: None,
        };
        let err = backend
            .clone_voice(req)
            .await
            .expect_err("Wave M.1 must return Unsupported");
        assert!(matches!(err, TtsError::Unsupported(_)));
    }

    #[tokio::test]
    async fn is_loaded_is_false_during_scaffolding() {
        let backend = MaskGctBackend::default();
        assert!(!backend.is_loaded().await);
    }
}
