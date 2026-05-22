//! Shared 16-bit PCM → WAV container writer used by every backend that
//! emits a final waveform.
//!
//! Lives outside [`super::musicgen`] (and [`super::stable_audio`]) so any
//! backend can call it regardless of which engine feature is enabled.
//! [`super::musicgen`] re-exports this symbol as `pcm_to_wav` for
//! backward compatibility with callers that already imported it from
//! there.

#![cfg(any(feature = "musicgen", feature = "stable-audio"))]

/// Pack `f32` PCM into a 16-bit-PCM WAV byte vector.
///
/// `samples` is interleaved across `channels` (so stereo input is
/// `[L0, R0, L1, R1, …]`). The output is a standard RIFF/WAVE container
/// with a single `fmt ` chunk + `data` chunk and no metadata extensions.
///
/// Sample values outside `[-1.0, 1.0]` are clamped before the
/// `i16`-scaling step to avoid wrap-around distortion.
#[must_use]
pub fn pcm_to_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "WAV `data` chunk size is a u32 by spec; sample buffers \
                  above 2^32 bytes are impossible in practice"
    )]
    let data_size = (samples.len() * usize::from(bits_per_sample / 8)) as u32;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1_u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Clamped to [-1.0, 1.0] above; multiplying by i16::MAX \
                      keeps the result inside i16 range by construction"
        )]
        let i = (clamped * f32::from(i16::MAX)) as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}
