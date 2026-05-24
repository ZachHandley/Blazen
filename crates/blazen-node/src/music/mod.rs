// Product names (MusicGen, AudioGen, Stable Audio, EnCodec, Hugging Face,
// ...) are referenced frequently in these binding docs. Wrapping every
// mention in backticks produces noisy code reviews without aiding
// readability — match the convention used by the sibling
// `blazen-audio-music` crate.
#![allow(clippy::doc_markdown)]

//! Music + SFX generation bindings (MusicGen, AudioGen, Stable Audio).
//!
//! Exposes one `#[napi]` class per backend
//! ([`musicgen::JsMusicgenBackend`], [`audiogen::JsAudioGenBackend`],
//! [`stable_audio::JsStableAudioBackend`]) plus a unified
//! [`model::JsMusicModel`] aggregator with `.musicgen()` / `.audioGen()` /
//! `.stableAudio()` factories. Each class exposes identical
//! `generateMusic` / `generateSfx` / `streamGenerateMusic` /
//! `streamGenerateSfx` methods so callers can swap backends without
//! changing call sites.
//!
//! Chunked streaming returns plain `Float32Array` slices of 32-bit float
//! PCM samples in `[-1, 1]` at the backend's native sample rate; the
//! non-streaming `generate*` methods return a [`chunk::JsMusicResult`]
//! carrying the encoded bytes plus container/sample-rate/channel metadata.

use napi::Unknown;
use napi::threadsafe_function::ThreadsafeFunction;

pub mod audiogen;
pub mod chunk;
pub mod model;
pub mod musicgen;
pub mod stable_audio;

pub use audiogen::{JsAudioGenBackend, JsAudioGenOptions};
pub use chunk::{JsMusicChunk, JsMusicResult};
pub use model::JsMusicModel;
pub use musicgen::{JsMusicgenBackend, JsMusicgenOptions, JsMusicgenVariant};
pub use stable_audio::{JsStableAudioBackend, JsStableAudioOptions, JsStableAudioVariant};

/// `ThreadsafeFunction` alias for the music streaming callback.
///
/// Each invocation receives a [`JsMusicChunk`] carrying a `Float32Array`
/// of PCM samples plus an `isFinal` flag and optional measured latency.
/// `CalleeHandled = false` (no error-first callback convention);
/// `Weak = true` so the registered handler does not prevent Node.js from
/// exiting.
pub(crate) type StreamMusicChunkCallbackTsfn =
    ThreadsafeFunction<JsMusicChunk, Unknown<'static>, JsMusicChunk, napi::Status, false, true>;
