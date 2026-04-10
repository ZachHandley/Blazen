//! Local text-to-speech backend for Blazen using [`Piper`](https://github.com/rhasspy/piper).
//!
//! This crate provides fully local, offline TTS synthesis via Piper voice
//! models running on ONNX Runtime, with no API keys required.
//!
//! When used through `blazen-llm` with the `piper` feature flag, this
//! crate's [`PiperProvider`] will implement `blazen_llm::AudioGeneration`.
//!
//! # Feature flags
//!
//! | Feature  | Description                                      |
//! |----------|--------------------------------------------------|
//! | `engine` | Links the ONNX Runtime backend for Piper         |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run synthesis. This keeps workspace builds
//! fast when the heavy native dependencies are not needed.

mod options;
mod provider;

pub use options::PiperOptions;
pub use provider::{PiperError, PiperProvider};
