//! The [`VoiceConversionBackend`] trait — capability extension of
//! [`blazen_audio::AudioBackend`] for speaker-transform engines (RVC and
//! similar architectures that take a source utterance and re-render it
//! in a target speaker's voice).

use std::path::Path;
use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::error::VcError;

/// A registered target speaker that a [`VoiceConversionBackend`] can
/// render source audio into.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetVoice {
    /// Backend-scoped identifier for this voice. Passed to
    /// [`VoiceConversionBackend::convert_voice`] and
    /// [`VoiceConversionBackend::stream_convert`].
    pub id: String,
    /// Optional human-readable label (display name) for UIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Native sample rate the backend renders this voice at, in Hz.
    /// Callers can resample the WAV bytes returned by
    /// [`VoiceConversionBackend::convert_voice`] if a different rate is
    /// required downstream.
    pub sample_rate_hz: u32,
}

/// Capability extension of [`AudioBackend`] for voice-conversion
/// engines.
///
/// Implementations take a source audio file (the utterance to convert)
/// plus a target-voice identifier and produce audio rendered in the
/// target speaker's voice. Both file-based and streaming variants are
/// covered; backends that only implement one fall back to the default
/// [`VcError::Unsupported`] impl for the other.
///
/// Implementors should be cheap to clone (typically wrap any expensive
/// state in `Arc` internally) and safe to share across tokio tasks.
#[async_trait]
pub trait VoiceConversionBackend: AudioBackend {
    /// Convert a source utterance to the voice of a registered target
    /// speaker, returning the rendered audio as a self-describing WAV
    /// byte buffer.
    ///
    /// # Audio format
    ///
    /// The returned bytes form a complete WAV (RIFF/`fmt `/`data`)
    /// container holding 16-bit signed little-endian PCM samples in
    /// either mono or stereo at the backend's native sample rate (see
    /// [`TargetVoice::sample_rate_hz`] for the per-voice rate; typical
    /// values are 32 kHz or 40 kHz for RVC-family backends and 24 kHz
    /// for diffusion-based voice converters).
    ///
    /// # Arguments
    ///
    /// * `input_audio_path` — path to the source utterance on disk in a
    ///   format the backend can decode (typically 16-bit PCM WAV at
    ///   16 kHz or 24 kHz).
    /// * `target_voice_id` — identifier of a previously-registered
    ///   target voice (see [`Self::list_target_voices`] and
    ///   [`Self::register_target_voice`]).
    ///
    /// # Errors
    ///
    /// Returns [`VcError::EngineNotAvailable`] when the backend was
    /// built without the required engine feature;
    /// [`VcError::VoiceNotFound`] when `target_voice_id` is not
    /// registered; [`VcError::Io`] on file-read failures;
    /// [`VcError::ModelLoad`] on weight-load failures; and
    /// [`VcError::Conversion`] on inference-time failures.
    async fn convert_voice(
        &self,
        input_audio_path: &Path,
        target_voice_id: &str,
    ) -> Result<Vec<u8>, VcError>;

    /// List the target voices that this backend can currently render.
    ///
    /// Default impl returns [`VcError::Unsupported`]; backends with a
    /// fixed or discoverable voice catalogue override this.
    ///
    /// # Errors
    ///
    /// Returns [`VcError::Unsupported`] from the default impl. Backend
    /// implementations may also return [`VcError::Io`] when probing a
    /// voice directory or [`VcError::ModelLoad`] when reading voice
    /// metadata files.
    async fn list_target_voices(&self) -> Result<Vec<TargetVoice>, VcError> {
        Err(VcError::Unsupported("listing target voices".into()))
    }

    /// Register a new target voice with this backend, using
    /// `reference_audio_path` as the speaker-embedding source.
    ///
    /// Default impl returns [`VcError::Unsupported`]; backends that
    /// support runtime voice registration (e.g. RVC with on-the-fly
    /// embedding extraction) override this.
    ///
    /// # Arguments
    ///
    /// * `voice_id` — backend-scoped identifier the caller will pass to
    ///   [`Self::convert_voice`] and [`Self::stream_convert`] going
    ///   forward.
    /// * `reference_audio_path` — path to a clean reference utterance
    ///   from the target speaker (typically 5–30 seconds, mono, 16 kHz
    ///   or 24 kHz PCM WAV).
    ///
    /// # Errors
    ///
    /// Returns [`VcError::Unsupported`] from the default impl. Backend
    /// implementations may also return [`VcError::Io`] on reference-
    /// audio read failures and [`VcError::ModelLoad`] when embedding
    /// extraction fails.
    async fn register_target_voice(
        &self,
        _voice_id: &str,
        _reference_audio_path: &Path,
    ) -> Result<(), VcError> {
        Err(VcError::Unsupported("registering target voices".into()))
    }

    /// Stream-based voice conversion for low-latency real-time use.
    ///
    /// Consumes a stream of 32-bit float PCM sample chunks at the
    /// backend's expected source sample rate (typically 16 kHz mono)
    /// and yields a stream of converted PCM sample chunks at the
    /// target voice's native sample rate (see
    /// [`TargetVoice::sample_rate_hz`]).
    ///
    /// Default impl returns [`VcError::Unsupported`]; backends that
    /// support streaming override this.
    ///
    /// # Arguments
    ///
    /// * `audio` — boxed stream of source PCM sample chunks.
    /// * `target_voice_id` — identifier of a previously-registered
    ///   target voice.
    ///
    /// # Errors
    ///
    /// Returns [`VcError::Unsupported`] from the default impl. Backend
    /// implementations may also return [`VcError::VoiceNotFound`] when
    /// `target_voice_id` is not registered, [`VcError::ModelLoad`] on
    /// weight-init failures, and [`VcError::Conversion`] on inference-
    /// time failures (either on this call or on items yielded from the
    /// returned stream).
    async fn stream_convert(
        &self,
        _audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        _target_voice_id: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<f32>, VcError>> + Send>>, VcError> {
        Err(VcError::Unsupported("streaming voice conversion".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_voice_roundtrips_json() {
        let voice = TargetVoice {
            id: "speaker-01".into(),
            label: Some("Alice".into()),
            sample_rate_hz: 40_000,
        };
        let json = serde_json::to_string(&voice).expect("serialize");
        let back: TargetVoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, "speaker-01");
        assert_eq!(back.label.as_deref(), Some("Alice"));
        assert_eq!(back.sample_rate_hz, 40_000);
    }

    #[test]
    fn target_voice_omits_missing_label() {
        let voice = TargetVoice {
            id: "speaker-02".into(),
            label: None,
            sample_rate_hz: 32_000,
        };
        let json = serde_json::to_string(&voice).expect("serialize");
        assert!(!json.contains("label"));
    }
}
