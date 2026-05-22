//! The [`SttBackend`] trait — capability extension of
//! [`blazen_audio::AudioBackend`] for speech-to-text engines.

use std::path::Path;
use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::AudioBackend;
use futures_core::Stream;
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

/// One emission from a streaming STT backend.
///
/// Partial transcripts are interim guesses that may be revised in
/// subsequent emissions; final transcripts are committed and will
/// not be revised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingTranscript {
    /// Text emitted for this chunk (partial or final).
    pub text: String,
    /// `true` when this emission is committed and will not be revised
    /// in a later emission; `false` for interim guesses.
    pub is_final: bool,
    /// Optional confidence in `[0, 1]` if the backend produces one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Optional latency-from-utterance-start in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_seconds: Option<f32>,
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

    /// Stream-based transcription for low-latency real-time STT.
    ///
    /// Consumes a stream of audio sample chunks (32-bit float PCM at the
    /// backend's expected sample rate — typically 16 kHz mono for
    /// Whisper-family backends) and yields a stream of
    /// [`StreamingTranscript`] emissions. Each emission carries either a
    /// partial (interim) or final (committed) transcript for the audio
    /// consumed so far.
    ///
    /// Default impl returns [`SttError::Unsupported`]. Backends that
    /// support streaming (e.g. `whisper-streaming`) override this.
    ///
    /// # Arguments
    ///
    /// * `audio` — boxed stream of PCM sample chunks.
    /// * `language` — optional ISO 639-1 language hint for this single
    ///   call. When `None`, the backend uses its configured default.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::Unsupported`] from the default impl. Backend
    /// implementations may also return [`SttError::ModelLoad`] on weight
    /// init failures and [`SttError::Transcription`] on inference-time
    /// failures (either on this call or on items yielded from the
    /// returned stream).
    async fn stream(
        &self,
        _audio: Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>,
        _language: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingTranscript, SttError>> + Send>>, SttError>
    {
        Err(SttError::Unsupported("streaming transcription".into()))
    }
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
