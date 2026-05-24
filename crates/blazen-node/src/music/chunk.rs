// Product names referenced in docs (MusicGen, AudioGen, Stable Audio,
// ...) match the convention used by the sibling `blazen-audio-music`
// crate, which also allows this lint crate-wide.
#![allow(clippy::doc_markdown)]

//! Streaming + non-streaming result types for the music binding.

use blazen_audio::{AudioFormat, GeneratedAudio};
use blazen_audio_music::MusicChunk;
use napi::bindgen_prelude::{Float32Array, Uint8Array};
use napi_derive::napi;

/// One emission from a streaming music backend.
///
/// Carries a `Float32Array` slice of 32-bit float PCM samples in `[-1, 1]`
/// at the backend's native output sample rate (32 kHz for MusicGen,
/// 16 kHz for AudioGen, 44.1 kHz stereo for Stable Audio), an `isFinal`
/// flag, and an optional measured per-chunk latency in seconds.
#[napi(object)]
pub struct JsMusicChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the backend's native
    /// output sample rate (interleaved for multi-channel outputs).
    pub samples: Float32Array,
    /// `true` when this is the final chunk emitted for the generation
    /// call; `false` for intermediate chunks.
    #[napi(js_name = "isFinal")]
    pub is_final: bool,
    /// Optional measured latency-from-call-start for this chunk, in
    /// seconds. `null` when the backend does not surface a timestamp.
    #[napi(js_name = "latencySeconds")]
    pub latency_seconds: Option<f64>,
}

/// Fully-rendered music + SFX result returned by the non-streaming
/// `generateMusic` / `generateSfx` entry points.
///
/// `bytes` carries the encoded clip — typically a WAV container holding
/// PCM samples; `format` distinguishes the container so callers can route
/// directly to a player without re-sniffing the payload.
#[napi(object)]
pub struct JsMusicResult {
    /// Encoded audio bytes (typically WAV for MusicGen / AudioGen /
    /// Stable Audio).
    pub bytes: Uint8Array,
    /// Container format: one of `"wav"`, `"mp3"`, `"flac"`, `"opus"`,
    /// or `"pcm"`.
    pub format: String,
    /// Sample rate in hertz.
    #[napi(js_name = "sampleRate")]
    pub sample_rate: u32,
    /// Channel count (mono = 1, stereo = 2).
    pub channels: u32,
    /// Duration of the clip in seconds, if known.
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
}

/// Convert a [`MusicChunk`] from the upstream music crate into a
/// [`JsMusicChunk`] suitable for crossing the napi boundary. Copies the
/// sample buffer once into a JS-owned `Float32Array`.
#[must_use]
pub fn build_music_chunk(chunk: MusicChunk) -> JsMusicChunk {
    JsMusicChunk {
        samples: Float32Array::with_data_copied(chunk.samples),
        is_final: chunk.is_final,
        latency_seconds: chunk.latency_seconds.map(f64::from),
    }
}

/// Convert a [`GeneratedAudio`] payload from the upstream audio crate
/// into a [`JsMusicResult`] suitable for crossing the napi boundary.
/// Copies the encoded byte buffer once into a JS-owned `Uint8Array`.
#[must_use]
pub fn build_music_result(generated: GeneratedAudio) -> JsMusicResult {
    let format = match generated.format {
        AudioFormat::Wav => "wav",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Flac => "flac",
        AudioFormat::Opus => "opus",
        AudioFormat::Pcm => "pcm",
    };
    JsMusicResult {
        bytes: Uint8Array::with_data_copied(generated.bytes),
        format: format.to_string(),
        sample_rate: generated.sample_rate,
        channels: u32::from(generated.channels),
        duration_seconds: generated.duration_seconds.map(f64::from),
    }
}
