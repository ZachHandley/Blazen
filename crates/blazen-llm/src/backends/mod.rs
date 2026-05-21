//! Local inference backend bridges.
//!
//! Each sub-module gates behind a feature flag and implements the appropriate
//! `blazen-llm` trait (e.g. [`EmbeddingModel`](crate::EmbeddingModel)) for
//! the backing crate's model type.

#[cfg(feature = "candle-embed")]
pub mod candle_embed;

#[cfg(feature = "candle-llm")]
pub mod candle_llm;

#[cfg(feature = "diffusion")]
pub mod diffusion;

#[cfg(feature = "embed")]
pub mod embed;

#[cfg(feature = "mistralrs")]
pub mod mistralrs;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "vllm")]
pub mod vllm;

// ---------------------------------------------------------------------------
// Audio bridges (new restructure — see PR_AUDIO_PLAN.md)
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-tts")]
pub mod tts;

#[cfg(feature = "audio-stt")]
pub mod audio_stt;

#[cfg(feature = "audio-music")]
pub mod audio_music;

#[cfg(feature = "audio-codec")]
pub mod audio_codec;
