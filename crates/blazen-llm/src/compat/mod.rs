//! Backwards-compatibility shims for crate names that were dissolved by
//! the PR-AUDIO restructure.
//!
//! Each sub-module wraps the canonical type from the new multi-backend
//! crate (`blazen-audio-stt`, `blazen-audio-tts`, `blazen-audio-music`,
//! `blazen-audio-codec`) in a thin struct that re-exposes the inherent
//! methods the bindings called on the previous, single-engine type.
//!
//! These shims are slated for removal once every binding has been
//! migrated to construct the new types directly.

#[cfg(feature = "audio-stt-whispercpp")]
pub mod whisper;
