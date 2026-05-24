//! The [`MusicBackend`] capability trait.
//!
//! Engines implement this trait on top of [`AudioBackend`](blazen_audio::AudioBackend)
//! to participate in music + SFX routing in the manager / pipeline layer.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, GeneratedAudio};
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::error::MusicError;

/// One emission from a streaming music backend.
///
/// Each chunk carries a slice of 32-bit float PCM samples at the
/// backend's expected output sample rate, an `is_final` flag, and an
/// optional measured per-chunk latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicChunk {
    /// 32-bit float PCM samples in [-1, 1] at the backend's expected
    /// output sample rate (matches the `sample_rate` field on the
    /// [`GeneratedAudio`] returned by the non-streaming
    /// [`MusicBackend::generate_music`] call).
    pub samples: Vec<f32>,
    /// `true` when this is the final chunk emitted for the
    /// generation call; `false` for intermediate chunks.
    pub is_final: bool,
    /// Optional latency-from-call-start in seconds for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_seconds: Option<f32>,
}

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

    /// Stream-based music generation for low-latency / progressive playback.
    ///
    /// Yields a stream of audio chunks as the backend produces them, rather
    /// than waiting for the full track. The total concatenated audio is
    /// equivalent to a single [`MusicBackend::generate_music`] call.
    ///
    /// Default impl returns [`MusicError::not_yet_implemented`]. Backends
    /// that support streaming override this.
    ///
    /// # Errors
    ///
    /// Returns [`MusicError::not_yet_implemented`] from the default impl.
    /// Backend implementations may also return any of the
    /// [`MusicError`] variants for inference-time failures (either on this
    /// call or on items yielded from the returned stream).
    async fn stream_generate_music(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        Err(MusicError::not_yet_implemented(format!(
            "stream_generate_music not supported by backend `{}`",
            self.id()
        )))
    }

    /// Stream-based SFX generation for low-latency / progressive playback.
    ///
    /// Yields a stream of audio chunks as the backend produces them, rather
    /// than waiting for the full clip. The total concatenated audio is
    /// equivalent to a single [`MusicBackend::generate_sfx`] call.
    ///
    /// Default impl returns [`MusicError::not_yet_implemented`]. Backends
    /// that support streaming override this.
    ///
    /// # Errors
    ///
    /// Returns [`MusicError::not_yet_implemented`] from the default impl.
    /// Backend implementations may also return any of the
    /// [`MusicError`] variants for inference-time failures (either on this
    /// call or on items yielded from the returned stream).
    async fn stream_generate_sfx(
        &self,
        _prompt: &str,
        _duration_seconds: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<MusicChunk, MusicError>> + Send>>, MusicError>
    {
        Err(MusicError::not_yet_implemented(format!(
            "stream_generate_sfx not supported by backend `{}`",
            self.id()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn music_chunk_roundtrips_json() {
        let chunk = MusicChunk {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            is_final: true,
            latency_seconds: Some(0.125),
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        let back: MusicChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.samples, chunk.samples);
        assert_eq!(back.is_final, chunk.is_final);
        assert_eq!(back.latency_seconds, chunk.latency_seconds);
    }

    #[test]
    fn music_chunk_omits_none_latency() {
        let chunk = MusicChunk {
            samples: vec![1.0],
            is_final: false,
            latency_seconds: None,
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        assert!(!json.contains("latency_seconds"), "json = {json}");
    }
}
