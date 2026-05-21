//! The [`MusicBackend`] capability trait.
//!
//! Engines implement this trait on top of [`AudioBackend`](blazen_audio::AudioBackend)
//! to participate in music + SFX routing in the manager / pipeline layer.

use async_trait::async_trait;
use blazen_audio::{AudioBackend, GeneratedAudio};

use crate::error::MusicError;

/// Capability trait for music + sound-effect generation backends.
///
/// Backends accept a free-form text prompt plus a target duration in
/// seconds and return a fully-rendered [`GeneratedAudio`] payload. The two
/// methods are split so callers (and routing logic in the manager) can tell
/// long-form music generation apart from short SFX synthesis even though
/// both are conditioned on text.
///
/// Backends that only support one of the two should return
/// [`MusicError::not_yet_implemented`] (with an explanatory message) from
/// the unsupported entry point.
#[async_trait]
pub trait MusicBackend: AudioBackend {
    /// Generate `duration_seconds` of music conditioned on `prompt`.
    ///
    /// # Errors
    ///
    /// Returns [`MusicError::InvalidInput`] for malformed inputs and any of
    /// the backend / engine variants for runtime failures.
    async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError>;

    /// Generate `duration_seconds` of sound-effect audio conditioned on
    /// `prompt`.
    ///
    /// # Errors
    ///
    /// Returns [`MusicError::InvalidInput`] for malformed inputs and any of
    /// the backend / engine variants for runtime failures.
    async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError>;
}
