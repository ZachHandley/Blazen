//! Local speech-to-text backend for Blazen using [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp).
//!
//! This crate wraps the `whisper-rs` Rust bindings to provide fully local,
//! offline speech-to-text transcription with no API keys required.
//!
//! When used through `blazen-llm` with the `whispercpp` feature flag, this
//! crate's [`WhisperCppProvider`] will implement `blazen_llm::Transcription`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                      |
//! |----------|--------------------------------------------------|
//! | `engine` | Links the actual `whisper-rs` runtime (CPU)      |
//! | `cuda`   | NVIDIA CUDA GPU acceleration                     |
//! | `metal`  | Apple Silicon GPU acceleration (Metal)            |
//! | `coreml` | Apple `CoreML` acceleration                        |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run transcription. This keeps workspace builds
//! fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::{WhisperModel, WhisperOptions};
pub use provider::{TranscriptionResult, TranscriptionSegment, WhisperCppProvider, WhisperError};
