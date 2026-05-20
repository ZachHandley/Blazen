//! MusicGen scaffold (NOT yet functional).
//!
//! # Why this module exists as a stub
//!
//! Meta's MusicGen (the text-conditioned music model from the
//! [audiocraft](https://github.com/facebookresearch/audiocraft) repo) is
//! a transformer decoder that autoregressively predicts EnCodec codebook
//! tokens conditioned on a T5-encoded text prompt. The pieces are:
//!
//! 1. T5 text encoder — present in `candle-transformers` (`models::t5`).
//! 2. EnCodec codec — present in `candle-transformers` (`models::encodec`)
//!    and wrapped by [`crate::encodec`] in this crate.
//! 3. MusicGen autoregressive decoder + delay-pattern token interleaver +
//!    classifier-free-guidance sampler — **MISSING** from
//!    `candle-transformers` 0.10.2 as of May 2026.
//!
//! The reference implementation in the `huggingface/candle` repo's
//! `candle-examples/examples/musicgen/` directory is a partial port that
//! loads the model graph but **does not implement the generation loop**;
//! see the upstream `TODO` in `musicgen_model.rs`. There is no published
//! `candle-transformers::models::musicgen` module.
//!
//! Filing an upstream tracking issue against `huggingface/candle` is on
//! the PR6 follow-up list; until then, this module ships:
//!
//! - The public type ([`MusicgenModel`]) that downstream callers can
//!   construct so their bindings (Python, Node, etc.) compile and surface
//!   a *clear* "not yet implemented" error instead of "no such type".
//! - The [`MusicgenConfig`] sized for the canonical
//!   `facebook/musicgen-small` (~300M params) and `musicgen-medium`
//!   (~1.5B params) checkpoints.
//! - An [`AudioModel`](crate::AudioModel) impl whose `generate` returns
//!   [`CandleAudioError::NotYetImplemented`] with a long-form message
//!   explaining the gap, what's needed to close it, and the
//!   Python-audiocraft fallback.
//!
//! When the autoregressive head lands upstream (or someone in-house
//! ports it), this file is the **only** place that needs to change to
//! light up music generation end-to-end across every binding.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{CandleAudioError, Result};
use crate::model::AudioModel;

/// Available MusicGen checkpoints on Hugging Face Hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MusicgenVariant {
    /// `facebook/musicgen-small` — ~300M params, 32 kHz mono output.
    Small,
    /// `facebook/musicgen-medium` — ~1.5B params, 32 kHz mono output.
    Medium,
    /// `facebook/musicgen-large` — ~3.3B params, 32 kHz mono output.
    Large,
}

impl MusicgenVariant {
    /// The Hugging Face Hub repo identifier for this variant.
    #[must_use]
    pub const fn hf_repo(self) -> &'static str {
        match self {
            Self::Small => "facebook/musicgen-small",
            Self::Medium => "facebook/musicgen-medium",
            Self::Large => "facebook/musicgen-large",
        }
    }

    /// Native sample rate (always 32 kHz for MusicGen, regardless of size).
    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        32_000
    }
}

/// Configuration for a [`MusicgenModel`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicgenConfig {
    /// Which checkpoint to use.
    pub variant: MusicgenVariant,
    /// Optional override for the HF cache directory.
    pub cache_dir: Option<std::path::PathBuf>,
    /// Run on CPU even when GPU acceleration is available.
    pub cpu_only: bool,
}

impl Default for MusicgenConfig {
    fn default() -> Self {
        Self {
            variant: MusicgenVariant::Small,
            cache_dir: None,
            cpu_only: false,
        }
    }
}

/// MusicGen text-to-music model (scaffold; see module docs).
///
/// Construction is cheap (no model load) — the inevitable
/// [`CandleAudioError::NotYetImplemented`] surfaces on the first
/// [`AudioModel::generate`] call.
#[derive(Debug, Clone)]
pub struct MusicgenModel {
    config: MusicgenConfig,
    name: String,
}

impl MusicgenModel {
    /// Construct a MusicGen model handle from the given config.
    #[must_use]
    pub fn new(config: MusicgenConfig) -> Self {
        let name = match config.variant {
            MusicgenVariant::Small => "musicgen-small",
            MusicgenVariant::Medium => "musicgen-medium",
            MusicgenVariant::Large => "musicgen-large",
        }
        .to_string();
        Self { config, name }
    }

    /// Borrow the model config.
    #[must_use]
    pub fn config(&self) -> &MusicgenConfig {
        &self.config
    }
}

impl Default for MusicgenModel {
    fn default() -> Self {
        Self::new(MusicgenConfig::default())
    }
}

/// Long-form message returned by every blocked MusicGen entry point.
///
/// Centralized as a `const fn` (well, `const &str`) so the wording stays
/// consistent across error sites and any future re-export.
pub(crate) const MUSICGEN_NOT_IMPLEMENTED: &str = "MusicGen text-to-music is not yet implemented in this crate. \
     candle-transformers 0.10.2 ships EnCodec (the audio codec) and T5 \
     (the text encoder) but does NOT ship the MusicGen autoregressive \
     decoder, delay-pattern token interleaver, or classifier-free-guidance \
     sampler -- those are the missing ~600-1000 LOC needed to predict \
     EnCodec tokens from a text prompt. The huggingface/candle repo has a \
     partial musicgen example that loads the model graph but stops before \
     the generation loop. Until the upstream Rust port lands, use the \
     Python `audiocraft` library (or the cloud AudioGeneration providers \
     in blazen-llm). See blazen/crates/blazen-audio-candle/src/musicgen.rs \
     for the scaffold and PR6_PLAN.md section 3c for the full audit.";

#[async_trait]
impl AudioModel for MusicgenModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample_rate(&self) -> u32 {
        self.config.variant.sample_rate()
    }

    async fn generate(&self, _prompt: &str, _duration_seconds: f32) -> Result<Vec<f32>> {
        Err(CandleAudioError::not_yet_implemented(
            MUSICGEN_NOT_IMPLEMENTED,
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variant_is_small() {
        let cfg = MusicgenConfig::default();
        assert_eq!(cfg.variant, MusicgenVariant::Small);
        assert_eq!(cfg.variant.hf_repo(), "facebook/musicgen-small");
    }

    #[test]
    fn sample_rate_is_32k_for_all_variants() {
        assert_eq!(MusicgenVariant::Small.sample_rate(), 32_000);
        assert_eq!(MusicgenVariant::Medium.sample_rate(), 32_000);
        assert_eq!(MusicgenVariant::Large.sample_rate(), 32_000);
    }

    #[test]
    fn name_reflects_variant() {
        let model = MusicgenModel::new(MusicgenConfig {
            variant: MusicgenVariant::Medium,
            ..MusicgenConfig::default()
        });
        assert_eq!(model.name(), "musicgen-medium");
    }

    #[tokio::test]
    async fn generate_returns_not_yet_implemented() {
        let model = MusicgenModel::default();
        let err = model.generate("upbeat jazz", 5.0).await.unwrap_err();
        match err {
            CandleAudioError::NotYetImplemented(msg) => {
                assert!(msg.contains("MusicGen"));
                assert!(msg.contains("autoregressive"));
            }
            other => panic!("expected NotYetImplemented, got {other:?}"),
        }
    }
}
