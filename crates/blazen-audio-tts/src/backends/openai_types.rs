//! Wire-format DTOs for the `OpenAI`-compatible TTS HTTP surface.
//!
//! These types model:
//! - `POST /v1/audio/speech` — the standard `OpenAI` text-to-speech endpoint
//!   (also accepted by every `OpenAI`-compat TTS service: Groq, Together, the
//!   in-house `OpenAI`-compat TTS deployment, etc.).
//! - `GET  /v1/voices` — the de-facto voice-listing endpoint used by
//!   most `OpenAI`-compat TTS deployments. Not part of the official `OpenAI`
//!   spec; the response shape varies across vendors so we keep the DTO
//!   permissive (`#[serde(default)]` everywhere) and surface unknown
//!   fields via the `metadata` catch-all.
//! - `POST /v1/voices/clone` — multipart upload of a reference clip +
//!   a name and optional transcript. Returns a `VoiceDto`.
//! - `POST /v1/voices/design` — text-only voice designer (creates a voice
//!   from a natural-language description without a reference clip).
//! - `DELETE /v1/voices/{id}` — remove a previously-cloned or designed voice.
//!
//! No field reaches public API directly; the `OpenAI` backend in
//! [`crate::backends::openai`] maps these to/from the capability-agnostic
//! `blazen_audio::{VoiceDto, VoiceHandle, ListVoicesResponse}` shapes.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// /v1/audio/speech
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/audio/speech`.
///
/// Mirrors the `OpenAI` public spec. The `speed` and `stream` fields are
/// `Option` so they only appear in the JSON when set, which keeps the
/// request bytes byte-identical to what `curl` examples on the `OpenAI`
/// docs site emit.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct SpeechRequestDto<'a> {
    pub model: &'a str,
    pub input: &'a str,
    pub voice: &'a str,
    pub response_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// /v1/voices (list)
// ---------------------------------------------------------------------------

/// `GET /v1/voices` response.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ListVoicesResponse {
    /// All voices known to the server (presets + cloned + designed).
    #[serde(default)]
    pub voices: Vec<VoiceDto>,
}

/// A voice as returned by `/v1/voices` or `/v1/voices/clone`.
///
/// Field set is intentionally permissive — vendors disagree on the
/// presence of `language`, `description`, and `metadata`.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct VoiceDto {
    /// Stable voice identifier — what callers pass back as the `voice`
    /// override on a synthesis request to synthesize with this voice.
    pub id: String,
    /// Human-readable display name.
    #[serde(default)]
    pub name: String,
    /// Optional BCP-47 language tag (e.g. `"en"`, `"ja"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional free-form description of the voice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Catch-all for vendor-specific extras (preview URL, gender hint,
    /// embedding handle, etc.).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// /v1/voices/clone (multipart)
// ---------------------------------------------------------------------------

/// Response from `POST /v1/voices/clone`.
///
/// Some services return the voice object directly; others wrap it under
/// a `voice` field. We accept both shapes via the untagged enum.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CloneVoiceResponse {
    /// Voice object wrapped under a `voice` key.
    Wrapped {
        /// The newly-cloned voice.
        voice: VoiceDto,
    },
    /// Voice object returned at the top level.
    Direct(VoiceDto),
}

impl CloneVoiceResponse {
    /// Unwrap to the inner `VoiceDto` regardless of which shape the server returned.
    #[must_use]
    pub fn into_voice(self) -> VoiceDto {
        match self {
            Self::Wrapped { voice } | Self::Direct(voice) => voice,
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/voices/design (JSON)
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/voices/design`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DesignVoiceRequestDto<'a> {
    pub name: &'a str,
    pub description: &'a str,
}

/// Response from `POST /v1/voices/design`. Same shape as
/// `CloneVoiceResponse` — accept both wrapped and direct forms.
pub type DesignVoiceResponse = CloneVoiceResponse;

// ---------------------------------------------------------------------------
// Error envelope (OpenAI-spec)
// ---------------------------------------------------------------------------

/// Standard OpenAI-style error envelope.
///
/// Both real `OpenAI` and most compat services return
/// `{ "error": { "message": "...", "type": "...", ... } }` on non-2xx
/// responses. We try to deserialize this first and fall back to the raw
/// body if it does not match.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorBody,
}

/// Inner error object on the `OpenAI` envelope.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ErrorBody {
    pub message: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub r#type: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_request_dto_omits_none_fields() {
        let dto = SpeechRequestDto {
            model: "tts-1",
            input: "hello",
            voice: "alloy",
            response_format: "mp3",
            speed: None,
            stream: None,
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert!(json.get("speed").is_none());
        assert!(json.get("stream").is_none());
        assert_eq!(json["model"], "tts-1");
        assert_eq!(json["input"], "hello");
        assert_eq!(json["voice"], "alloy");
        assert_eq!(json["response_format"], "mp3");
    }

    #[test]
    fn speech_request_dto_serializes_speed_and_stream() {
        let dto = SpeechRequestDto {
            model: "tts-1-hd",
            input: "x",
            voice: "nova",
            response_format: "wav",
            speed: Some(1.25),
            stream: Some(true),
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["speed"], 1.25);
        assert_eq!(json["stream"], true);
    }

    #[test]
    fn list_voices_response_deserializes_minimum() {
        let raw = r#"{"voices":[{"id":"alloy","name":"Alloy"}]}"#;
        let parsed: ListVoicesResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.voices.len(), 1);
        assert_eq!(parsed.voices[0].id, "alloy");
        assert_eq!(parsed.voices[0].name, "Alloy");
        assert!(parsed.voices[0].language.is_none());
    }

    #[test]
    fn list_voices_response_preserves_unknown_metadata() {
        let raw =
            r#"{"voices":[{"id":"v1","name":"V1","metadata":{"preview_url":"https://x/y"}}]}"#;
        let parsed: ListVoicesResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.voices[0].metadata["preview_url"], "https://x/y");
    }

    #[test]
    fn clone_voice_response_accepts_wrapped_form() {
        let raw = r#"{"voice":{"id":"cloned","name":"Cloned"}}"#;
        let parsed: CloneVoiceResponse = serde_json::from_str(raw).unwrap();
        let v = parsed.into_voice();
        assert_eq!(v.id, "cloned");
        assert_eq!(v.name, "Cloned");
    }

    #[test]
    fn clone_voice_response_accepts_direct_form() {
        let raw = r#"{"id":"direct","name":"Direct"}"#;
        let parsed: CloneVoiceResponse = serde_json::from_str(raw).unwrap();
        let v = parsed.into_voice();
        assert_eq!(v.id, "direct");
        assert_eq!(v.name, "Direct");
    }

    #[test]
    fn error_envelope_round_trips_minimum() {
        let raw = r#"{"error":{"message":"invalid voice"}}"#;
        let parsed: ErrorEnvelope = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.error.message, "invalid voice");
    }
}
