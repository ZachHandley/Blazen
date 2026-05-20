//! Candle-backed local audio generation (music & SFX).
//!
//! This crate is the local-inference counterpart to the cloud
//! `AudioGeneration` providers in `blazen-llm`. It exposes a single
//! capability trait [`AudioModel`] plus the [`CandleAudioProvider`] wrapper
//! used by the `backends::candle_audio` bridge in `blazen-llm`.
//!
//! ## Status (May 2026)
//!
//! `candle-transformers` 0.10.2 ships EnCodec (the neural audio codec) but
//! **does not ship the MusicGen / AudioGen autoregressive transformer
//! heads** that consume EnCodec tokens to produce music or sound effects.
//! See `/home/zach/.cache/blazen-pr6-research/PR6_PLAN.md` §3c for the full
//! audit.
//!
//! Consequences for this crate:
//!
//! | Component | State |
//! |---|---|
//! | [`encodec::EncodecModel`] | **Functional** — encode PCM into discrete codebook tokens and decode tokens back to PCM. Loads weights from `facebook/encodec_24khz`. |
//! | [`musicgen::MusicgenModel`] | **Scaffold only** — `generate` returns [`CandleAudioError::NotYetImplemented`] with a clear pointer to the upstream gap. |
//!
//! AudioGen lives in the same world as MusicGen and is intentionally not
//! scaffolded yet — once the MusicGen autoregressive head ships, AudioGen
//! is a 1-day port of the same machinery against a different checkpoint.
//!
//! ## Feature flags
//!
//! - `default` — no-op. The crate compiles, but every `generate_*` call
//!   surfaces [`CandleAudioError::EngineNotAvailable`].
//! - `engine` — links `candle-core`, `candle-nn`, `candle-transformers`,
//!   `hf-hub`. Required for any real inference.
//! - `cuda` — implies `engine`; enables the CUDA backend in candle. Requires
//!   CUDA toolkit + an Ampere-or-newer GPU.
//! - `metal` — implies `engine`. Apple Silicon Metal backend (candle-core only).

#![cfg_attr(docsrs, feature(doc_cfg))]
// The crate's docs mention product names (EnCodec, MusicGen, AudioGen,
// Hugging Face, ...) frequently. Wrapping every mention in backticks
// produces noisy code reviews without aiding readability — pedantic's
// doc_markdown lint flags the prose. We opt out at the crate level.
#![allow(clippy::doc_markdown)]

pub mod error;
pub mod model;
pub mod musicgen;

#[cfg(feature = "engine")]
pub mod encodec;

pub use error::{CandleAudioError, Result};
pub use model::{AudioModel, CandleAudioProvider};
pub use musicgen::MusicgenModel;

#[cfg(feature = "engine")]
pub use encodec::EncodecModel;
