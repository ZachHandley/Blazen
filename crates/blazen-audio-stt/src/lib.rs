//! # blazen-audio-stt
//!
//! Multi-backend speech-to-text engine crate for Blazen. Sibling to
//! `blazen-audio-tts`, `blazen-audio-music`, and `blazen-audio-codec`;
//! all four share the capability-agnostic vocabulary defined in
//! [`blazen_audio`].
//!
//! ## Surface shape
//!
//! - [`SttBackend`]: capability trait extending [`blazen_audio::AudioBackend`]
//!   that every STT engine implements.
//! - [`SttProvider<B>`]: typed wrapper for Rust callers; monomorphizes
//!   on the concrete backend.
//! - [`DynSttProvider`]: erased wrapper (`Box<dyn SttBackend>`) for FFI
//!   / language-binding boundaries that cannot carry generics. See
//!   `Appendix B` of the PR-AUDIO plan for the dual-shape rationale.
//! - [`SttOptions`]: cross-backend options (model id, language hint,
//!   sample rate, device, diarization toggle).
//! - [`SttError`]: capability-agnostic error type; flattens engine-native
//!   failures into one of a small set of variants and implements
//!   `From<SttError> for blazen_audio::AudioError`.
//! - [`TranscriptionResult`] / [`TranscriptionSegment`]: result types.
//! - [`StreamingTranscript`]: per-chunk emission from the streaming
//!   `SttBackend::stream` surface.
//!
//! ## Backends
//!
//! Each backend lives in [`backends`] under its own feature gate:
//!
//! | Backend     | Feature       | Notes                                          |
//! |-------------|---------------|------------------------------------------------|
//! | whisper.cpp | `whispercpp`  | Local CPU/GPU via the `whisper-rs` bindings.   |
//! | candle      | `candle`      | Pure-Rust Whisper via `candle-transformers`.   |
//!
//! Platform-specific acceleration for the whisper.cpp backend (`cuda`,
//! `metal`, `coreml`) is exposed as opt-in no-op alias features —
//! consumers wanting GPU acceleration must add `whisper-rs` as a direct
//! dependency in their binary crate. See this crate's `Cargo.toml`
//! comments for the rationale.

#![deny(missing_docs)]

pub mod backends;
pub mod error;
pub mod options;
pub mod provider;
pub mod traits;

pub use error::SttError;
pub use options::SttOptions;
pub use provider::{DynSttProvider, SttProvider};
pub use traits::{StreamingTranscript, SttBackend, TranscriptionResult, TranscriptionSegment};

#[cfg(feature = "candle")]
pub use backends::candle::{
    CandleWhisperBackend, CandleWhisperConfig, WhisperModel as CandleWhisperModel, WhisperTask,
};

#[cfg(feature = "whisper-streaming")]
pub use backends::whisper_streaming::{WhisperStreamingBackend, WhisperStreamingConfig};

#[cfg(feature = "faster-whisper")]
pub use backends::faster_whisper::{FasterWhisperBackend, FasterWhisperConfig};
