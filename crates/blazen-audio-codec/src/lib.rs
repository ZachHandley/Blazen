//! # blazen-audio-codec
//!
//! Neural-audio-codec backends for Blazen. Codecs translate raw PCM into
//! discrete codebook tokens (and back) so generative models can operate in
//! a low-rate, GPU-friendly token space instead of 24-48 kHz waveform
//! samples.
//!
//! This crate plays the same role for *codecs* that
//! [`blazen-audio-tts`](../blazen_audio_tts/index.html) plays for TTS and
//! [`blazen-audio-music`](../blazen_audio_music/index.html) plays for music:
//! a single capability-typed [`CodecBackend`] trait plus a
//! monomorphizable [`CodecProvider<B>`] and an erased
//! [`DynCodecProvider`] for binding layers.
//!
//! ## Backends
//!
//! | Backend | Feature flag | Status |
//! |---|---|---|
//! | [`backends::encodec`] | `encodec` | **Functional** — Meta's EnCodec 24 kHz / 4-codebook neural codec via `candle-transformers`. |
//! | [`backends::dac`] | `dac` | **Functional decode** — Descript Audio Codec (`descript/dac_44khz`, 9 codebooks at 8 kbps) via `candle-transformers`. Encode short-circuits until candle exposes a public RVQ encode path. |
//! | [`backends::snac`] | `snac` | **Functional** — Multi-Scale Neural Audio Codec (`hubertsiuzdak/snac_24khz`, 3 multi-scale codebooks @ vq_strides `[4, 2, 1]`, 24 kHz, ~3 kbps) via `candle-transformers`. Both encode and decode are wired. |
//!
//! ## Why a dedicated codec trait?
//!
//! Codecs are pure functions over PCM and tokens — no prompts, no
//! sampling temperature, no voices. Trying to thread them through the
//! generative [`AudioBackend`](blazen_audio::AudioBackend) surface would
//! force every codec to invent prompt semantics it doesn't have, so the
//! [`CodecBackend`] trait adds **only** `encode_pcm` / `decode_tokens`
//! and inherits the lifecycle methods (`load` / `unload` /
//! `is_loaded`) from [`AudioBackend`].
//!
//! See `PR_AUDIO_PLAN.md` §3 + §5 W5 for the full restructure rationale.

#![cfg_attr(docsrs, feature(doc_cfg))]
// Crate prose mentions product names (EnCodec, MusicGen, AudioGen, DAC,
// SNAC, Hugging Face, ...) frequently. Backticking every mention is
// noisy; opt out at the crate level like `blazen-audio-candle` does.
#![allow(clippy::doc_markdown)]

pub mod backends;
pub mod error;
pub mod provider;
pub mod traits;

pub use error::CodecError;
pub use provider::{CodecProvider, DynCodecProvider};
pub use traits::CodecBackend;
