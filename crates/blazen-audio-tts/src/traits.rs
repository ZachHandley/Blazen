//! The [`TtsBackend`] trait that every concrete TTS engine implements.
//!
//! Extends [`AudioBackend`] from `blazen-audio` with the TTS-specific
//! surface: synthesis, voice listing, voice cloning, and voice design.
//! Default implementations return [`TtsError::Unsupported`] for the
//! optional voice-management methods so backends that only do plain
//! synthesis (Kokoro, Piper, `VibeVoice`, …) don't have to override them.

use std::pin::Pin;

use async_trait::async_trait;
use blazen_audio::{
    AudioBackend, CloneVoiceRequest, DesignVoiceRequest, GeneratedAudio, ListVoicesRequest,
    ListVoicesResponse, VoiceHandle,
};
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::{TtsError, TtsOptions};

/// One emission from a streaming TTS backend.
///
/// Each chunk carries a slice of 32-bit float PCM samples at the
/// backend's expected output sample rate, an `is_final` flag, and an
/// optional measured per-chunk latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingAudioChunk {
    /// 32-bit float PCM samples in [-1, 1] at the backend's expected
    /// output sample rate (matches the `sample_rate` field on the
    /// [`GeneratedAudio`] returned by the non-streaming `synthesize`
    /// call).
    pub samples: Vec<f32>,
    /// `true` when this is the final chunk emitted for the
    /// synthesis call; `false` for intermediate chunks.
    pub is_final: bool,
    /// Optional latency-from-call-start in seconds for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_seconds: Option<f32>,
}

/// Capability trait for any TTS engine in the Blazen ecosystem.
///
/// Implementors automatically inherit the lifecycle methods (`load`,
/// `unload`, `is_loaded`, `id`, `provider_kind`) from [`AudioBackend`].
/// Concrete backends live under
/// [`crate::backends`](crate::backends) — `anytts`, `openai`, and the
/// reserved `piper` slot.
#[async_trait]
pub trait TtsBackend: AudioBackend {
    /// Synthesize speech for `text` honoring the `options` overrides
    /// (voice, language, sample rate, …). Returns a fully-formed
    /// [`GeneratedAudio`] payload with container, sample rate, channels,
    /// and (when known) duration.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    async fn synthesize(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError>;

    /// List voices known to this backend. Backends without a voice
    /// catalog (Piper local, single-voice servers) MAY return an empty
    /// response; backends that genuinely cannot enumerate voices SHOULD
    /// return [`TtsError::Unsupported`].
    ///
    /// Default implementation surfaces `Unsupported`.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    async fn list_voices(
        &self,
        _request: &ListVoicesRequest,
    ) -> Result<ListVoicesResponse, TtsError> {
        Err(TtsError::Unsupported(format!(
            "list_voices not supported by backend `{}`",
            self.id()
        )))
    }

    /// Clone a voice from a reference audio clip.
    ///
    /// Default implementation surfaces `Unsupported`.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    async fn clone_voice(&self, _request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(format!(
            "clone_voice not supported by backend `{}`",
            self.id()
        )))
    }

    /// Design a brand-new voice from a free-form text description.
    ///
    /// Default implementation surfaces `Unsupported`.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    async fn design_voice(&self, _request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(format!(
            "design_voice not supported by backend `{}`",
            self.id()
        )))
    }

    /// Remove a previously-cloned or -designed voice.
    ///
    /// Default implementation surfaces `Unsupported`.
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    async fn delete_voice(&self, _voice_id: &str) -> Result<(), TtsError> {
        Err(TtsError::Unsupported(format!(
            "delete_voice not supported by backend `{}`",
            self.id()
        )))
    }

    /// Stream-based synthesis for low-latency real-time TTS.
    ///
    /// Yields a stream of audio chunks as the backend produces them, rather
    /// than waiting for the full utterance. The total concatenated audio is
    /// equivalent to a single `synthesize` call.
    ///
    /// Default impl returns [`TtsError::Unsupported`]. Backends that support
    /// streaming (Bark, F5-TTS, Spark-TTS, `MaskGCT`) override this.
    ///
    /// # Arguments
    ///
    /// * `text` — the text to synthesize.
    /// * `voice` — optional voice id; `None` uses the voice configured in
    ///   `options` (or the backend's default).
    /// * `options` — same options struct as the non-streaming `synthesize`.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Unsupported`] from the default impl. Backend
    /// implementations may also return [`TtsError::ModelLoad`] on weight
    /// init failures and [`TtsError::Synthesis`] on inference-time failures
    /// (either on this call or on items yielded from the returned stream).
    async fn stream_synthesize(
        &self,
        _text: &str,
        _voice: Option<&str>,
        _options: TtsOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingAudioChunk, TtsError>> + Send>>, TtsError>
    {
        Err(TtsError::Unsupported("streaming synthesis".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_chunk_roundtrips_json() {
        let chunk = StreamingAudioChunk {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            is_final: true,
            latency_seconds: Some(0.125),
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        let back: StreamingAudioChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.samples, vec![0.0, 0.5, -0.5, 1.0]);
        assert!(back.is_final);
        assert_eq!(back.latency_seconds, Some(0.125));
    }

    #[test]
    fn streaming_chunk_omits_missing_latency() {
        let chunk = StreamingAudioChunk {
            samples: vec![0.0],
            is_final: false,
            latency_seconds: None,
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        assert!(!json.contains("latency"));
    }
}
