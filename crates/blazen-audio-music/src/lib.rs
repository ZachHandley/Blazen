//! # blazen-audio-music
//!
//! Music and sound-effect (SFX) generation backends for Blazen's audio
//! capability surface.
//!
//! This crate provides the [`MusicBackend`] capability trait, the
//! [`MusicProvider`] generic wrapper, and a [`DynMusicProvider`] type-erased
//! handle suitable for dynamic dispatch. Concrete engines live in the
//! [`backends`] module:
//!
//! - [`backends::musicgen`] — Meta's MusicGen text-to-music (real,
//!   end-to-end implementation: T5 prompt encoder + decoder LM with
//!   delay-pattern interleaver + Classifier-Free Guidance + EnCodec
//!   audio decoder). Gated behind the `musicgen` cargo feature.
//! - [`backends::audiogen`] — Meta's AudioGen text-to-SFX (real,
//!   end-to-end implementation reusing the MusicGen autoregressive
//!   pipeline with AudioGen-specific config: 16 kHz EnCodec +
//!   1.5B-param decoder LM + `facebook/audiogen-medium` weights).
//!   Gated behind the `audiogen` cargo feature (alias for `musicgen`).
//! - [`backends::stable_audio`] — Stability AI's Stable Audio (placeholder;
//!   no candle port currently exists upstream).
//!
//! ## Feature flags
//!
//! - `default` — no engines linked; every `generate_*` call surfaces
//!   [`MusicError::NotYetImplemented`] or [`MusicError::EngineNotAvailable`].
//! - `musicgen` — links `candle-core`, `candle-nn`, `candle-transformers`,
//!   `tokenizers`, `hf-hub`, and `blazen-audio-codec/encodec`. Required
//!   to construct a real MusicGen backend.
//! - `audiogen` — alias that activates the `musicgen` dependency set and
//!   additionally compiles in the [`backends::audiogen`] wrapper.
//! - `live-models` — opt-in integration tests that pull real MusicGen /
//!   AudioGen weights (multi-GB) from the Hugging Face Hub.

#![deny(missing_docs)]
// Crate-level docs mention product names (MusicGen, AudioGen, Stable Audio,
// EnCodec, Hugging Face, ...) frequently. Wrapping every mention in
// backticks produces noisy code reviews without aiding readability — match
// the convention used by sibling `blazen-audio-candle`.
#![allow(clippy::doc_markdown)]

pub mod backends;
pub mod error;
pub mod provider;
pub mod traits;

pub use error::MusicError;
pub use provider::{DynMusicProvider, MusicProvider};
pub use traits::{MusicBackend, MusicChunk};

#[cfg(feature = "audiogen")]
pub use backends::audiogen::{
    AUDIOGEN_FRAME_RATE, AUDIOGEN_MAX_DURATION_HARD_LIMIT, AUDIOGEN_SAMPLE_RATE, AudioGenBackend,
    AudioGenConfig,
};
