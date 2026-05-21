//! Voice-management requests / responses shared by TTS backends with
//! cloning or voice-design capabilities (e.g. `ElevenLabs`, `Cartesia`,
//! `zonos`-style local TTS).

use serde::{Deserialize, Serialize};

/// Whether a voice is a vendor preset, designed via prompt, or cloned from
/// reference audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceKind {
    /// A built-in vendor preset voice.
    Preset,
    /// A voice the caller designed via text description / parameters.
    Designed,
    /// A voice cloned from reference audio.
    Cloned,
}

/// Filter parameters for [`AudioBackend`](crate::AudioBackend)-style voice
/// listing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListVoicesRequest {
    /// Restrict to voices for a given BCP-47 language tag (e.g. `"en"`,
    /// `"ja"`, `"fr-CA"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Restrict to voices of a given [`VoiceKind`]. Serialized as a
    /// lowercase string so callers from dynamic-typed bindings can pass
    /// `"preset"` / `"designed"` / `"cloned"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<VoiceKind>,
}

/// Response from a voice-listing operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListVoicesResponse {
    /// Voices matching the filter, in vendor-defined order.
    pub voices: Vec<VoiceDto>,
}

/// Capability-agnostic voice description suitable for cross-binding
/// transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceDto {
    /// Vendor-assigned voice identifier (opaque, pass back via
    /// [`VoiceHandle`]).
    pub id: String,
    /// Human-readable voice name.
    pub name: String,
    /// BCP-47 language tag, if the voice is language-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Whether this voice is a preset, designed, or cloned voice.
    pub kind: VoiceKind,
}

/// Request to clone a voice from one or more reference audio samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloneVoiceRequest {
    /// Human-readable name for the new voice.
    pub name: String,
    /// Raw bytes of a reference audio clip. The encoding is engine-defined
    /// — most engines accept WAV/MP3/FLAC; pass via
    /// [`AudioMetadata`](crate::AudioMetadata) at the call site if needed.
    pub audio_bytes: Vec<u8>,
    /// Optional verbatim transcript of the reference clip, used by some
    /// engines (e.g. Tortoise, XTTS) to improve clone fidelity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

/// Request to synthesize a brand-new voice from a text description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignVoiceRequest {
    /// Human-readable name for the new voice.
    pub name: String,
    /// Free-form description (e.g. `"warm, middle-aged narrator with a
    /// slight British accent"`).
    pub description: String,
    /// Optional BCP-47 language hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Engine-specific parameters (e.g. pitch, age, gender, style) as a
    /// freeform JSON object. Backends MUST treat unknown keys as
    /// ignored-but-warned, not as errors.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// Opaque persistent handle returned after [`CloneVoiceRequest`] or
/// [`DesignVoiceRequest`] succeeds. Pass back as a `voice` identifier on
/// subsequent speech requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceHandle {
    /// Vendor-assigned voice identifier.
    pub id: String,
    /// Which engine / provider owns this voice (e.g. `"elevenlabs"`,
    /// `"piper"`, `"candle-orpheus"`).
    pub provider: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_kind_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&VoiceKind::Preset).expect("serialize VoiceKind"),
            "\"preset\""
        );
        assert_eq!(
            serde_json::to_string(&VoiceKind::Designed).expect("serialize VoiceKind"),
            "\"designed\""
        );
        assert_eq!(
            serde_json::to_string(&VoiceKind::Cloned).expect("serialize VoiceKind"),
            "\"cloned\""
        );
    }

    #[test]
    fn list_voices_request_defaults_are_empty() {
        let req = ListVoicesRequest::default();
        let json = serde_json::to_string(&req).expect("serialize ListVoicesRequest");
        assert_eq!(json, "{}");
    }

    #[test]
    fn voice_dto_roundtrips_json() {
        let dto = VoiceDto {
            id: "vx_42".into(),
            name: "Aria".into(),
            language: Some("en".into()),
            kind: VoiceKind::Preset,
        };
        let json = serde_json::to_string(&dto).expect("serialize VoiceDto");
        let decoded: VoiceDto = serde_json::from_str(&json).expect("deserialize VoiceDto");
        assert_eq!(decoded.id, "vx_42");
        assert_eq!(decoded.name, "Aria");
        assert_eq!(decoded.language.as_deref(), Some("en"));
        assert_eq!(decoded.kind, VoiceKind::Preset);
    }

    #[test]
    fn design_voice_request_parameters_default_to_null() {
        let req = DesignVoiceRequest {
            name: "Storyteller".into(),
            description: "warm narrator".into(),
            language: None,
            parameters: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&req).expect("serialize DesignVoiceRequest");
        let decoded: DesignVoiceRequest =
            serde_json::from_str(&json).expect("deserialize DesignVoiceRequest");
        assert_eq!(decoded.parameters, serde_json::Value::Null);
    }
}
