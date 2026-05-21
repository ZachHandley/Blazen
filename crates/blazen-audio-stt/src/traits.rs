//! The [`SttBackend`] trait — capability extension of
//! [`blazen_audio::AudioBackend`] for speech-to-text engines.

use std::path::Path;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use serde::{Deserialize, Serialize};

use crate::error::SttError;

/// A time-aligned segment within a [`TranscriptionResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Start time of the segment in milliseconds from the start of the audio.
    pub start_ms: i64,
    /// End time of the segment in milliseconds from the start of the audio.
    pub end_ms: i64,
    /// The transcribed text for this segment.
    pub text: String,
}

/// The result of an [`SttBackend::transcribe`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Full transcribed text (concatenation of all segment texts, trimmed).
    pub text: String,
    /// Time-aligned segments produced by the backend.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segments: Vec<TranscriptionSegment>,
    /// The detected or specified language of the transcribed audio
    /// (ISO 639-1 code), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Capability extension of [`AudioBackend`] for speech-to-text engines.
///
/// Implementations cover both file-based and (eventually) streaming
/// transcription. The base trait covers file-based transcription only;
/// streaming will be added as an optional method on a follow-up PR.
///
/// Implementors should be cheap to clone (typically wrap any expensive
/// state in `Arc` internally) and safe to share across tokio tasks.
#[async_trait]
pub trait SttBackend: AudioBackend {
    /// Transcribe an audio file from disk.
    ///
    /// # Arguments
    ///
    /// * `audio_path` — path to an audio file on disk in a format the
    ///   backend can decode (typically 16 kHz mono WAV for Whisper-family
    ///   backends).
    /// * `language` — optional ISO 639-1 language hint for this single
    ///   call. When `None`, the backend uses its configured default (which
    ///   may itself be auto-detect).
    ///
    /// # Errors
    ///
    /// Returns [`SttError::EngineNotAvailable`] when the backend was built
    /// without the required engine feature; [`SttError::Io`] on file-read
    /// failures; [`SttError::ModelLoad`] on weight-load failures; and
    /// [`SttError::Transcription`] on inference-time failures.
    async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_roundtrips_json() {
        let seg = TranscriptionSegment {
            start_ms: 0,
            end_ms: 1_500,
            text: "hello".into(),
        };
        let json = serde_json::to_string(&seg).expect("serialize");
        let back: TranscriptionSegment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.start_ms, 0);
        assert_eq!(back.end_ms, 1_500);
        assert_eq!(back.text, "hello");
    }

    #[test]
    fn result_omits_empty_segments_and_unknown_language() {
        let res = TranscriptionResult {
            text: "hi".into(),
            segments: vec![],
            language: None,
        };
        let json = serde_json::to_string(&res).expect("serialize");
        assert!(!json.contains("segments"));
        assert!(!json.contains("language"));
    }
}
