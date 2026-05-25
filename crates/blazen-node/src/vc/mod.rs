// Product names (RVC, Retrieval-based Voice Conversion, HuBERT,
// NSF-HiFi-GAN, ...) are referenced frequently in these binding docs.
// Wrapping every mention in backticks produces noisy code reviews
// without aiding readability — match the convention used by the sibling
// `blazen-audio-vc` crate.
#![allow(clippy::doc_markdown)]

//! Voice-conversion bindings (RVC + future engines).
//!
//! Exposes one `#[napi]` class per backend ([`rvc::JsRvcBackend`]) plus a
//! unified [`model::JsVcModel`] aggregator with a `.rvc()` factory. Each
//! class exposes identical `convertVoice` / `streamConvertPcm` /
//! `listTargetVoices` / `registerTargetVoice` methods so callers can
//! swap backends without changing call sites.
//!
//! Chunked streaming returns plain `Float32Array` slices of 32-bit float
//! PCM samples at the target voice's native sample rate (typically
//! 32 kHz or 40 kHz for RVC-family backends); the non-streaming
//! `convertVoice` method returns a [`chunk::JsVcResult`] carrying the
//! encoded WAV bytes plus sample-rate / duration metadata.

use napi::Unknown;
use napi::threadsafe_function::ThreadsafeFunction;

pub mod chunk;
pub mod model;
pub mod rvc;

pub use chunk::{JsTargetVoice, JsVcChunk, JsVcResult};
pub use model::JsVcModel;
pub use rvc::{JsRvcBackend, JsRvcOptions};

/// `ThreadsafeFunction` alias for the voice-conversion streaming
/// callback.
///
/// Each invocation receives a [`JsVcChunk`] carrying a `Float32Array`
/// of converted PCM samples plus an `isFinal` flag and optional
/// measured latency. `CalleeHandled = false` (no error-first callback
/// convention); `Weak = true` so the registered handler does not
/// prevent Node.js from exiting.
pub(crate) type StreamVcChunkCallbackTsfn =
    ThreadsafeFunction<JsVcChunk, Unknown<'static>, JsVcChunk, napi::Status, false, true>;
