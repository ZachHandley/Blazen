//! # blazen-audio
//!
//! Capability-agnostic shared types and traits for Blazen's audio engine
//! crates (TTS, STT, music, codec, voice cloning, etc.).
//!
//! This crate mirrors the role that `blazen-llm` plays for the LLM engine
//! crates: it provides the small, dependency-free vocabulary every audio
//! engine speaks, so per-engine crates (`blazen-audio-tts`,
//! `blazen-audio-whispercpp`, `blazen-audio-candle`, `blazen-audio-piper`,
//! ...) do not have to import `blazen-llm` just to share a few request and
//! result shapes.
//!
//! Bridging between these capability-agnostic types and the existing
//! `blazen-llm` `compute::{requests, results}` types lives in
//! `blazen-llm/src/backends/*.rs` and is added incrementally in later PR
//! waves — this crate intentionally has **zero engine dependencies**.
//!
//! ## Modules
//!
//! - [`backend`]: the base [`AudioBackend`] trait every engine implements
//! - [`types`]: the shared audio payload types ([`AudioFormat`],
//!   [`SampleFormat`], [`GeneratedAudio`], [`AudioMetadata`])
//! - [`voice`]: voice-management requests / responses (list, clone, design)
//! - [`error`]: the capability-agnostic [`AudioError`] type

#![deny(missing_docs)]

pub mod backend;
pub mod error;
pub mod nc_license;
pub mod types;
pub mod voice;

pub use backend::AudioBackend;
pub use error::AudioError;
pub use nc_license::warn_nc_once;
pub use types::{AudioFormat, AudioMetadata, GeneratedAudio, SampleFormat};
pub use voice::{
    CloneVoiceRequest, DesignVoiceRequest, ListVoicesRequest, ListVoicesResponse, VoiceDto,
    VoiceHandle, VoiceKind,
};
