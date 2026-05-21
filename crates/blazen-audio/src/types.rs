//! Capability-agnostic audio payload types shared by every audio backend.

use serde::{Deserialize, Serialize};

/// Container format for an encoded audio clip.
///
/// `Pcm` denotes raw interleaved samples with no container — interpret
/// alongside [`GeneratedAudio::sample_rate`], [`GeneratedAudio::channels`],
/// and (for PCM) the originating [`SampleFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// WAVE / RIFF container — typically PCM payload.
    Wav,
    /// MPEG-1 Audio Layer III.
    Mp3,
    /// Free Lossless Audio Codec.
    Flac,
    /// Opus in an Ogg container.
    Opus,
    /// Raw interleaved PCM samples (no container).
    Pcm,
}

/// Sample format for raw PCM payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleFormat {
    /// 16-bit signed little-endian integers.
    I16,
    /// 32-bit IEEE-754 little-endian floats in `[-1.0, 1.0]`.
    F32,
}

/// A single generated audio clip returned by a TTS / music / codec backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedAudio {
    /// Encoded audio bytes.
    pub bytes: Vec<u8>,
    /// Container format (see [`AudioFormat`]).
    pub format: AudioFormat,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of audio channels (mono = 1, stereo = 2).
    pub channels: u16,
    /// Duration of the clip in seconds, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
}

/// Side-channel metadata about an audio payload — handy for transcription
/// callers that want to know the source format without re-decoding the
/// bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Container format of the source bytes.
    pub format: AudioFormat,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Raw-PCM sample format, if [`format`](Self::format) is
    /// [`AudioFormat::Pcm`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_format: Option<SampleFormat>,
    /// Duration in seconds, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_audio_roundtrips_json() {
        let payload = GeneratedAudio {
            bytes: vec![0x52, 0x49, 0x46, 0x46],
            format: AudioFormat::Wav,
            sample_rate: 24_000,
            channels: 1,
            duration_seconds: Some(1.25),
        };
        let json = serde_json::to_string(&payload).expect("serialize GeneratedAudio");
        let decoded: GeneratedAudio =
            serde_json::from_str(&json).expect("deserialize GeneratedAudio");
        assert_eq!(decoded.bytes, payload.bytes);
        assert_eq!(decoded.format, AudioFormat::Wav);
        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.duration_seconds, Some(1.25));
    }

    #[test]
    fn audio_metadata_omits_optional_fields() {
        let meta = AudioMetadata {
            format: AudioFormat::Mp3,
            sample_rate: 44_100,
            channels: 2,
            sample_format: None,
            duration_seconds: None,
        };
        let json = serde_json::to_string(&meta).expect("serialize AudioMetadata");
        assert!(!json.contains("sample_format"));
        assert!(!json.contains("duration_seconds"));
    }

    #[test]
    fn sample_format_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&SampleFormat::I16).expect("serialize SampleFormat"),
            "\"i16\""
        );
        assert_eq!(
            serde_json::to_string(&SampleFormat::F32).expect("serialize SampleFormat"),
            "\"f32\""
        );
    }
}
