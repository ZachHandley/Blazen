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
