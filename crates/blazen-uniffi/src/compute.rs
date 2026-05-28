//! Non-LLM compute wire-format records for the UniFFI bindings.
//!
//! The central capability-erased `TtsModel` / `SttModel` / `ImageGenModel` /
//! `ThreeDModel` opaque handles and their per-backend factories
//! (`new_local_tts_model`, `new_piper_tts_model`, `new_spark_tts_model`,
//! `new_whisper_stt_model`, `new_faster_whisper_stt_model`,
//! `new_diffusion_model`, `new_triposr_3d_model`, `new_fal_*_model`) have
//! moved to the per-engine concrete provider classes under
//! [`crate::concrete`] (e.g. `PiperProvider`, `WhisperCppProvider`,
//! `TripoSrProvider`, `FalProvider`). This module retains only the
//! wire-format records ([`TtsResult`], [`SttResult`], [`ImageGenResult`],
//! [`ThreeDGenerateResult`]) that the concrete providers return so foreign
//! code continues to see a single stable result shape per modality.
//!
//! ## Wire-format shape
//!
//! The upstream [`AudioGeneration`](blazen_llm::compute::AudioGeneration) /
//! [`Transcription`](blazen_llm::compute::Transcription) /
//! [`ImageGeneration`](blazen_llm::compute::ImageGeneration) traits carry
//! rich request and response types. UniFFI's UDL grammar collapses
//! poorly under that level of nesting, so this module flattens the
//! response shapes to:
//!
//! - **TTS output** ([`TtsResult`]): `audio_base64` (empty when the
//!   provider only returned a URL), `mime_type`, `duration_ms`.
//! - **STT output** ([`SttResult`]): `transcript`, `language` (empty when
//!   the provider didn't report one), `duration_ms`.
//! - **Image output** ([`ImageGenResult`]): `Vec<Media>` reusing
//!   [`crate::llm::Media`]. URL-only outputs surface with `data_base64`
//!   set to the URL string and `mime_type` populated from the upstream
//!   [`MediaType`](blazen_llm::MediaType).
//! - **3D output** ([`ThreeDGenerateResult`]): `model_bytes` (typically a
//!   GLB container) plus the IANA `mime_type` (`"model/gltf-binary"`).

use crate::llm::Media;

// ---------------------------------------------------------------------------
// Wire-format records
// ---------------------------------------------------------------------------

/// The result of a text-to-speech synthesis call.
///
/// `audio_base64` is the empty string when the upstream provider returned a
/// URL only (the URL travels in the `data_base64` slot of a downstream
/// [`Media`] when callers route through [`crate::llm::Media`]; pure TTS
/// callers should detect the empty `audio_base64` and fall back to fetching
/// the URL themselves). `mime_type` reflects the upstream
/// [`MediaType`](blazen_llm::MediaType); `duration_ms` is zero when the
/// provider didn't report timing.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TtsResult {
    pub audio_base64: String,
    pub mime_type: String,
    pub duration_ms: u64,
}

/// The result of a speech-to-text transcription call.
///
/// `language` is the empty string when the provider didn't report a
/// detected language. `duration_ms` reflects the upstream
/// [`RequestTiming::total_ms`](blazen_llm::RequestTiming) — zero when the
/// backend didn't measure it.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SttResult {
    pub transcript: String,
    pub language: String,
    pub duration_ms: u64,
}

/// The result of an image-generation call.
///
/// `images[i].kind` is always `"image"`. `data_base64` contains either the
/// raw base64 bytes (when the upstream `MediaOutput.base64` field is
/// populated) or the URL string (when only `MediaOutput.url` is set);
/// callers must inspect `mime_type` and treat the field accordingly.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ImageGenResult {
    pub images: Vec<Media>,
}

/// Result of a single-image-to-3D generation call.
///
/// Carries the rendered model as bytes (typically GLB / glTF-binary at
/// `model/gltf-binary`) plus the IANA MIME type so foreign callers can
/// dispatch on the format without sniffing the buffer.
#[cfg(feature = "triposr")]
#[derive(Debug, Clone, uniffi::Record)]
pub struct ThreeDGenerateResult {
    /// Encoded 3D model bytes (GLB container with embedded vertices /
    /// indices / vertex colors).
    pub model_bytes: Vec<u8>,
    /// IANA MIME type of `model_bytes`. Typically `"model/gltf-binary"`.
    pub mime_type: String,
}
