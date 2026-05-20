//! Local text-to-speech backend for Blazen, powered by
//! [`any-tts`](https://crates.io/crates/any-tts).
//!
//! This crate provides fully local, offline TTS synthesis with multiple
//! pluggable model backends (Kokoro-82M, `VibeVoice`, Qwen3-TTS), no API
//! keys required. When used through `blazen-llm` with the `tts` feature
//! flag this crate's [`TtsProvider`] is bridged onto the
//! `blazen_llm::AudioGeneration::text_to_speech` trait method.
//!
//! # Feature flags
//!
//! | Feature  | Description                                                 |
//! |----------|-------------------------------------------------------------|
//! | `engine` | Links the `any-tts` crate. Without this the provider stubs. |
//!
//! Without the `engine` feature the crate compiles (options struct + stub
//! provider) but cannot actually run synthesis. This keeps workspace
//! builds fast when the heavy native model code is not needed.

mod options;
mod provider;

pub use options::{TtsModel, TtsOptions};
pub use provider::{SynthesizedAudio, TtsError, TtsProvider};
