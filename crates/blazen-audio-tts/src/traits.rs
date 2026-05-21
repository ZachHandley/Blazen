//! The [`TtsBackend`] trait that every concrete TTS engine implements.
//!
//! Extends [`AudioBackend`] from `blazen-audio` with the TTS-specific
//! surface: synthesis, voice listing, voice cloning, and voice design.
//! Default implementations return [`TtsError::Unsupported`] for the
//! optional voice-management methods so backends that only do plain
//! synthesis (Kokoro, Piper, `VibeVoice`, …) don't have to override them.

use async_trait::async_trait;
use blazen_audio::{
    AudioBackend, CloneVoiceRequest, DesignVoiceRequest, GeneratedAudio, ListVoicesRequest,
    ListVoicesResponse, VoiceHandle,
};

use crate::{TtsError, TtsOptions};

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
}
