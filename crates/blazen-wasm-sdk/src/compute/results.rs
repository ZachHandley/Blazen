//! Typed compute result types -- re-exported from `blazen_llm::compute`.
//!
//! Each result type derives [`tsify_next::Tsify`] upstream with
//! `into_wasm_abi` + `from_wasm_abi`, so the `pub use` is sufficient to
//! surface them as TypeScript interfaces in `blazen_wasm_sdk.d.ts`.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;

pub use blazen_llm::compute::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, TranscriptionSegment, VideoResult,
    VoiceHandle,
};

/// TS-facing copy of [`blazen_audio_music::MusicChunk`].
///
/// One emission from a streaming music backend: a slice of 32-bit float
/// PCM samples at the backend's expected output sample rate, an
/// `is_final` flag marking the last chunk of the generation call, and an
/// optional measured per-chunk latency in seconds.
///
/// The native [`blazen_audio_music::MusicChunk`] derives `Serialize` /
/// `Deserialize` but not `Tsify`, so this thin wrapper exists purely to
/// publish a TypeScript interface alongside the rest of the SDK's
/// compute result types.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmMusicChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the backend's output
    /// sample rate.
    pub samples: Vec<f32>,
    /// `true` when this is the final chunk emitted for the generation
    /// call; `false` for intermediate chunks.
    pub is_final: bool,
    /// Optional latency-from-call-start in seconds for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_seconds: Option<f32>,
}

impl From<blazen_audio_music::MusicChunk> for WasmMusicChunk {
    fn from(c: blazen_audio_music::MusicChunk) -> Self {
        Self {
            samples: c.samples,
            is_final: c.is_final,
            latency_seconds: c.latency_seconds,
        }
    }
}

/// TS-facing chunk emitted by a streaming voice-conversion backend.
///
/// One emission carries a slice of 32-bit float PCM samples produced by
/// the RVC pipeline, an `is_final` flag marking the last chunk of a
/// conversion call, an optional per-chunk sample rate override, and an
/// optional measured per-chunk latency in seconds.
///
/// `blazen-audio-vc` itself does not derive [`tsify_next::Tsify`] on its
/// types (it is a native-only surface in default-feature builds — only
/// the trait, [`blazen_audio_vc::TargetVoice`] and
/// [`blazen_audio_vc::VcError`] compile to wasm32), so this thin
/// wrapper publishes a TypeScript interface for the JS-implementable
/// [`crate::capability_providers::WasmVcProvider`] surface.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmVcChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the backend's output
    /// sample rate.
    pub samples: Vec<f32>,
    /// `true` when this is the final chunk emitted for the conversion
    /// call; `false` for intermediate chunks.
    pub is_final: bool,
    /// Optional per-chunk sample-rate override (Hz). Backends that emit
    /// at a fixed rate may omit this field; callers should fall back to
    /// the [`WasmTargetVoice::sample_rate_hz`] of the active target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Optional latency-from-call-start in seconds for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_seconds: Option<f32>,
}

/// TS-facing copy of [`blazen_audio_vc::TargetVoice`].
///
/// Describes a target voice known to a voice-conversion backend: its
/// stable identifier, optional human-readable label, and the native
/// sample rate at which the backend emits PCM for this voice.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmTargetVoice {
    /// Stable identifier (e.g. `"alice"`, `"speaker-007"`).
    pub id: String,
    /// Optional human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Native sample rate for this voice, in Hz.
    pub sample_rate_hz: u32,
}

impl From<blazen_audio_vc::TargetVoice> for WasmTargetVoice {
    fn from(t: blazen_audio_vc::TargetVoice) -> Self {
        Self {
            id: t.id,
            label: t.label,
            sample_rate_hz: t.sample_rate_hz,
        }
    }
}

/// TS-facing result of a non-streaming voice-conversion call.
///
/// Carries the rendered WAV audio bytes, the sample rate at which the
/// backend emitted PCM, and the identifier of the target voice the
/// source audio was converted to.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WasmVcResult {
    /// Rendered audio as WAV bytes.
    pub wav_bytes: Vec<u8>,
    /// Sample rate at which the backend emitted PCM, in Hz.
    pub sample_rate_hz: u32,
    /// Identifier of the target voice the source audio was converted to.
    pub target_voice_id: String,
}
