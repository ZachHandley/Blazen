// Product names referenced in docs (RVC, HuBERT, NSF-HiFi-GAN, ...) match
// the convention used by the sibling `blazen-audio-vc` crate, which also
// allows this lint crate-wide.
#![allow(clippy::doc_markdown)]

//! Streaming + non-streaming result types for the voice-conversion
//! binding.

use napi::bindgen_prelude::{Float32Array, Uint8Array};
use napi_derive::napi;

#[cfg(feature = "audio-vc")]
use blazen_audio_vc::TargetVoice;

/// One emission from a streaming voice-conversion backend.
///
/// Carries a `Float32Array` slice of 32-bit float PCM samples in `[-1, 1]`
/// at the target voice's native sample rate (typically 32 kHz or 40 kHz
/// for RVC-family backends), an `isFinal` flag, and an optional measured
/// per-chunk latency in seconds.
#[napi(object)]
pub struct JsVcChunk {
    /// 32-bit float PCM samples in `[-1, 1]` at the target voice's
    /// native sample rate (mono).
    pub samples: Float32Array,
    /// `true` when this is the final chunk emitted for the conversion
    /// call; `false` for intermediate chunks.
    #[napi(js_name = "isFinal")]
    pub is_final: bool,
    /// Optional measured latency-from-call-start for this chunk, in
    /// seconds. `null` when the backend does not surface a timestamp
    /// (RVC backends today do not).
    #[napi(js_name = "latencySeconds")]
    pub latency_seconds: Option<f64>,
}

/// Fully-rendered voice-conversion result returned by the non-streaming
/// `convertVoice` entry point.
///
/// `bytes` carries a self-describing WAV (RIFF/`fmt `/`data`) container
/// holding 16-bit signed little-endian PCM samples at the target voice's
/// native sample rate. `sampleRate` and `durationSeconds` are parsed out
/// of the WAV header so callers don't need to re-sniff the payload to
/// route it to a player.
#[napi(object)]
pub struct JsVcResult {
    /// Encoded WAV bytes (16-bit signed little-endian PCM).
    pub bytes: Uint8Array,
    /// Sample rate in hertz, parsed from the WAV `fmt ` chunk.
    #[napi(js_name = "sampleRate")]
    pub sample_rate: u32,
    /// Duration of the clip in seconds, derived from the WAV `data`
    /// chunk size + frame stride. `null` if the WAV header could not be
    /// parsed (in which case `sampleRate` falls back to `0`).
    #[napi(js_name = "durationSeconds")]
    pub duration_seconds: Option<f64>,
}

/// A registered target voice descriptor.
///
/// Returned by [`crate::vc::JsRvcBackend::list_target_voices`] /
/// [`crate::vc::JsVcModel::list_target_voices`] and accepted by the
/// matching `convertVoice` / `streamConvertPcm` calls (the `id` field
/// is the lookup key).
#[napi(object)]
pub struct JsTargetVoice {
    /// Backend-scoped identifier for this voice. Passed to
    /// `convertVoice` / `streamConvertPcm`.
    pub id: String,
    /// Optional human-readable display name for UIs.
    pub label: Option<String>,
    /// Native sample rate the backend renders this voice at, in Hz.
    #[napi(js_name = "sampleRateHz")]
    pub sample_rate_hz: u32,
}

/// Convert a backend-yielded `Vec<f32>` PCM buffer into a [`JsVcChunk`]
/// suitable for crossing the napi boundary. Copies the sample buffer
/// once into a JS-owned `Float32Array`.
#[must_use]
pub fn build_vc_chunk(samples: Vec<f32>, is_final: bool) -> JsVcChunk {
    JsVcChunk {
        samples: Float32Array::with_data_copied(samples),
        is_final,
        latency_seconds: None,
    }
}

/// Convert WAV bytes returned by
/// [`blazen_audio_vc::VoiceConversionBackend::convert_voice`] into a
/// [`JsVcResult`] suitable for crossing the napi boundary. Parses the
/// RIFF/`fmt `/`data` header to extract the sample rate and clip
/// duration; copies the encoded byte buffer once into a JS-owned
/// `Uint8Array`.
#[must_use]
pub fn build_vc_result(bytes: Vec<u8>) -> JsVcResult {
    let (sample_rate, duration_seconds) =
        parse_wav_metadata(&bytes).map_or((0, None), |(sr, dur)| (sr, Some(dur)));
    JsVcResult {
        bytes: Uint8Array::with_data_copied(bytes),
        sample_rate,
        duration_seconds,
    }
}

/// Convert an upstream [`TargetVoice`] into a [`JsTargetVoice`].
#[cfg(feature = "audio-vc")]
#[must_use]
pub fn build_target_voice(voice: TargetVoice) -> JsTargetVoice {
    JsTargetVoice {
        id: voice.id,
        label: voice.label,
        sample_rate_hz: voice.sample_rate_hz,
    }
}

/// Parse the `sample_rate` (Hz) and `duration` (seconds) out of a WAV
/// (RIFF/WAVE) byte buffer.
///
/// Walks the RIFF chunk table looking for `fmt ` (for the sample-rate,
/// channel-count, and bits-per-sample triple) and `data` (for the byte
/// length). Returns `None` if the buffer is too small, the magic numbers
/// don't match, or either required chunk is missing.
fn parse_wav_metadata(raw: &[u8]) -> Option<(u32, f64)> {
    if raw.len() < 44 {
        return None;
    }
    if &raw[..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        return None;
    }

    let mut cursor = 12_usize;
    let mut channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut data_len: Option<usize> = None;
    while cursor + 8 <= raw.len() {
        let id = &raw[cursor..cursor + 4];
        let size = u32::from_le_bytes([
            raw[cursor + 4],
            raw[cursor + 5],
            raw[cursor + 6],
            raw[cursor + 7],
        ]) as usize;
        let body_start = cursor + 8;
        let body_end = body_start.saturating_add(size).min(raw.len());
        let body = &raw[body_start..body_end];
        match id {
            b"fmt " if body.len() >= 16 => {
                channels = Some(u16::from_le_bytes([body[2], body[3]]));
                sample_rate = Some(u32::from_le_bytes([body[4], body[5], body[6], body[7]]));
                bits_per_sample = Some(u16::from_le_bytes([body[14], body[15]]));
            }
            b"data" => {
                data_len = Some(body.len());
            }
            _ => {}
        }
        // RIFF chunks are 2-byte aligned.
        cursor = body_end + (body_end & 1);
        if data_len.is_some() && sample_rate.is_some() {
            break;
        }
    }

    let sr = sample_rate?;
    let ch = u32::from(channels?);
    let bps = u32::from(bits_per_sample?);
    let data = data_len?;
    if ch == 0 || bps == 0 {
        return None;
    }
    let bytes_per_frame = ch * (bps / 8);
    if bytes_per_frame == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let frames = (data as u64) / u64::from(bytes_per_frame);
    #[allow(clippy::cast_precision_loss)]
    let duration_seconds = frames as f64 / f64::from(sr);
    Some((sr, duration_seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_wav(sample_rate: u32, channels: u16, bits_per_sample: u16, frames: usize) -> Vec<u8> {
        let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
        let block_align = channels * (bits_per_sample / 8);
        let data_len = frames * usize::from(block_align);
        let mut buf = Vec::with_capacity(44 + data_len);
        buf.extend_from_slice(b"RIFF");
        #[allow(clippy::cast_possible_truncation)]
        buf.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes());
        buf.extend_from_slice(&1_u16.to_le_bytes()); // audio_format = PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        #[allow(clippy::cast_possible_truncation)]
        buf.extend_from_slice(&(data_len as u32).to_le_bytes());
        buf.extend(std::iter::repeat_n(0_u8, data_len));
        buf
    }

    #[test]
    fn parse_wav_metadata_extracts_sample_rate_and_duration() {
        let raw = synth_wav(40_000, 1, 16, 80_000); // 2 seconds @ 40 kHz mono
        let (sr, dur) = parse_wav_metadata(&raw).expect("parse");
        assert_eq!(sr, 40_000);
        assert!((dur - 2.0).abs() < 1e-6, "got {dur}");
    }

    #[test]
    fn parse_wav_metadata_rejects_short_buffers() {
        assert!(parse_wav_metadata(&[0_u8; 16]).is_none());
    }

    #[test]
    fn parse_wav_metadata_rejects_non_riff() {
        let mut raw = synth_wav(16_000, 1, 16, 1_600);
        raw[0] = b'X';
        assert!(parse_wav_metadata(&raw).is_none());
    }
}
