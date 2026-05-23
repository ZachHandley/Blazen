//! Audio preprocessing for the faster-whisper backend (Wave F.2.1).
//!
//! `ct2rs::Whisper::generate` consumes raw `f32` PCM samples in `[-1.0, 1.0]`
//! sampled at the model's `sampling_rate` (16 kHz for every Whisper
//! checkpoint shipped under `Systran/faster-whisper-*`). The
//! `CTranslate2` runtime computes the log-mel spectrogram **internally**
//! via the `mel_spec` crate vendored by `ct2rs` (see
//! [`ct2rs::Whisper::generate`](https://docs.rs/ct2rs/0.9.18/ct2rs/struct.Whisper.html#method.generate)
//! — confirmed by inspecting `ct2rs-0.9.18/src/whisper.rs`).
//!
//! That bounds this module's responsibility:
//!
//! 1. Accept caller input as either a path to a 16 kHz mono 16-bit PCM
//!    WAV (matching the contract the existing whisper.cpp and Candle
//!    backends already advertise) or already-decoded raw PCM with
//!    arbitrary sample rate and channel count.
//! 2. If the input is stereo (or higher), downmix to mono via the
//!    arithmetic mean of per-channel samples.
//! 3. If the input is not at 16 kHz, resample with a simple linear
//!    interpolator. Whisper's log-mel pipeline is highly tolerant of
//!    resampling quality — a multi-tap polyphase filter is overkill for
//!    a model that quantises everything down to 80- or 128-bin mels
//!    every 10 ms.
//! 4. Emit a fresh `Vec<f32>` ready to hand to
//!    [`ct2rs::Whisper::generate`][`super::decoder`].
//!
//! Mel computation, FFT, window function, and 30-second chunking are
//! **not** this module's concern — see
//! [`ct2rs::Whisper::generate`](https://docs.rs/ct2rs/0.9.18/ct2rs/struct.Whisper.html#method.generate)
//! for the internal pipeline.
//!
//! ## File-decoding scope
//!
//! For the [`AudioInput::Path`] arm we mirror the minimal "skip the
//! 44-byte RIFF/WAVE header, read raw `i16` LE samples" decoder used by
//! [`crate::backends::whispercpp`] and [`crate::backends::candle`] —
//! deliberately matching their contract so callers can move audio
//! between backends without rewriting their I/O. Richer container
//! support (FLAC, OGG, MP3) belongs upstream in caller code (e.g. via
//! `symphonia`) and intentionally does not pull a new dependency into
//! this crate.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::SttError;

/// Whisper's fixed input sample rate, in hertz.
///
/// Every `Systran/faster-whisper-*` checkpoint ships a
/// `preprocessor_config.json` with `sampling_rate: 16000`, so we hard-code
/// this rather than reading it back from the model.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Errors surfaced by [`prepare_for_whisper`].
///
/// Variants flatten cleanly into [`SttError`] via the [`From`] impl
/// below, so callers can `?`-convert through the pipeline boundary.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The supplied audio file could not be opened or read.
    #[error("audio file `{path}`: {source}")]
    Io {
        /// Offending path (preserved for caller diagnostics).
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The supplied bytes do not look like a valid 16 kHz 16-bit PCM
    /// WAV (missing `RIFF`/`WAVE` magic or sub-header truncation).
    #[error("audio file is not a supported WAV: {0}")]
    InvalidWav(String),

    /// The caller passed a zero-channel buffer or an unsupported sample
    /// rate that cannot be resampled (e.g. `sample_rate == 0`).
    #[error("invalid audio parameters: {0}")]
    InvalidParameters(String),
}

impl From<AudioError> for SttError {
    fn from(err: AudioError) -> Self {
        match err {
            AudioError::Io { path, source } => Self::Io(std::io::Error::new(
                source.kind(),
                format!("audio file `{}`: {source}", path.display()),
            )),
            AudioError::InvalidWav(msg) => Self::Transcription(format!("audio decode: {msg}")),
            AudioError::InvalidParameters(msg) => Self::InvalidOptions(msg),
        }
    }
}

/// User-supplied audio input handed to [`prepare_for_whisper`].
///
/// `Path` is decoded as a 16 kHz mono 16-bit PCM WAV (the format every
/// other Blazen STT backend already accepts on its file-based entry
/// point). `RawPcm` is for pipelines that have already done their own
/// decoding (microphone capture, codec output, synthesised tests) and
/// just need the resample / downmix to 16 kHz mono.
#[derive(Debug, Clone, Copy)]
pub enum AudioInput<'a> {
    /// Filesystem path to a 16 kHz mono 16-bit PCM WAV.
    ///
    /// Container support beyond minimal WAV is intentionally out of
    /// scope — wire `symphonia` or `hound` upstream and feed
    /// [`AudioInput::RawPcm`] instead.
    Path(&'a Path),
    /// Already-decoded raw PCM samples plus their sample rate and
    /// channel count.
    ///
    /// `samples` is interleaved by channel
    /// (`[L0, R0, L1, R1, ...]` for stereo).
    RawPcm {
        /// Interleaved PCM samples.
        samples: &'a [f32],
        /// Source sample rate in hertz.
        sample_rate: u32,
        /// Channel count (1 = mono, 2 = stereo, …).
        channels: u16,
    },
}

/// Load `input` and emit a fresh `Vec<f32>` of 16 kHz mono `f32` samples
/// in `[-1.0, 1.0]` suitable for [`ct2rs::Whisper::generate`].
///
/// # Errors
///
/// Returns [`AudioError::Io`] if a `Path` input cannot be read,
/// [`AudioError::InvalidWav`] if the file's header is missing or
/// malformed, and [`AudioError::InvalidParameters`] if `channels == 0`
/// or `sample_rate == 0` on a `RawPcm` input.
pub fn prepare_for_whisper(input: AudioInput<'_>) -> Result<Vec<f32>, AudioError> {
    let (mut samples, sample_rate, channels) = match input {
        AudioInput::Path(path) => {
            let raw_bytes = std::fs::read(path).map_err(|source| AudioError::Io {
                path: path.to_path_buf(),
                source,
            })?;
            let pcm = decode_wav_16k_mono(&raw_bytes)?;
            // The minimal WAV decoder hard-codes 16 kHz mono — matches
            // the long-standing contract of [`crate::backends::candle`]
            // and [`crate::backends::whispercpp`]. Downstream resample /
            // downmix calls are no-ops on this branch.
            (pcm, TARGET_SAMPLE_RATE, 1u16)
        }
        AudioInput::RawPcm {
            samples,
            sample_rate,
            channels,
        } => {
            if channels == 0 {
                return Err(AudioError::InvalidParameters(
                    "channels must be >= 1".into(),
                ));
            }
            if sample_rate == 0 {
                return Err(AudioError::InvalidParameters(
                    "sample_rate must be > 0".into(),
                ));
            }
            (samples.to_vec(), sample_rate, channels)
        }
    };

    if channels > 1 {
        samples = downmix_to_mono(&samples, channels);
    }

    if sample_rate != TARGET_SAMPLE_RATE {
        samples = resample_linear(&samples, sample_rate, TARGET_SAMPLE_RATE);
    }

    Ok(samples)
}

/// Average the interleaved channel samples down to a single mono track.
///
/// Whisper trained on mono audio; the official `OpenAI` reference
/// implementation downmixes by arithmetic mean, and the faster-whisper
/// Python project (which we are porting) does the same via
/// `librosa.to_mono`.
fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let channels = usize::from(channels);
    debug_assert!(channels > 1, "downmix called with mono input");
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    #[allow(clippy::cast_precision_loss)]
    let inv_channels = 1.0_f32 / channels as f32;
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        mono.push(sum * inv_channels);
    }
    mono
}

/// Linear-interpolation resampler.
///
/// Whisper's mel pipeline is intentionally low-fidelity (80- or 128-bin
/// mels at 100 frames per second), so a multi-tap polyphase filter
/// would be wasted work — linear interpolation across `f32` samples is
/// well below the quantisation floor of the downstream feature
/// extractor for the common 44.1 kHz / 48 kHz → 16 kHz cases.
///
/// Returns an empty `Vec` for empty input (rather than dividing by
/// zero on the output-length calculation).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }
    let src_len = input.len();
    // Round-half-up rather than floor: `(src_len * dst_rate + src_rate/2) / src_rate`.
    let dst_len_u64 =
        (src_len as u64 * u64::from(dst_rate) + u64::from(src_rate) / 2) / u64::from(src_rate);
    let dst_len = dst_len_u64.max(1) as usize;
    let mut out = Vec::with_capacity(dst_len);
    let ratio = f64::from(src_rate) / f64::from(dst_rate);
    for i in 0..dst_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = input[idx.min(src_len - 1)];
        let b = input[(idx + 1).min(src_len - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Decode a 16 kHz mono 16-bit PCM WAV into `f32` samples in `[-1.0, 1.0]`.
///
/// Mirrors the byte-for-byte format the `whispercpp` and `candle`
/// backends already accept so the three backends share one file
/// contract. Anything more exotic should be decoded upstream and
/// handed in as [`AudioInput::RawPcm`].
fn decode_wav_16k_mono(raw_bytes: &[u8]) -> Result<Vec<f32>, AudioError> {
    if raw_bytes.len() < 44 {
        return Err(AudioError::InvalidWav(
            "file too small to be a valid WAV".into(),
        ));
    }
    if &raw_bytes[..4] != b"RIFF" || &raw_bytes[8..12] != b"WAVE" {
        return Err(AudioError::InvalidWav("missing RIFF/WAVE header".into()));
    }
    let pcm_data = &raw_bytes[44..];
    let mut samples = Vec::with_capacity(pcm_data.len() / 2);
    for chunk in pcm_data.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(f32::from(s) / 32_768.0);
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_passes_through_when_already_target_format() {
        // 16 kHz mono f32 — should round-trip bit-for-bit.
        let samples: Vec<f32> = (0..1_600)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f32;
                (x / 1_600.0).sin()
            })
            .collect();
        let out = prepare_for_whisper(AudioInput::RawPcm {
            samples: &samples,
            sample_rate: TARGET_SAMPLE_RATE,
            channels: 1,
        })
        .expect("prepare");
        assert_eq!(out.len(), samples.len());
        // Bit-for-bit — no resample, no downmix, no quantisation.
        for (a, b) in out.iter().zip(samples.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn prepare_resamples_44100_to_16000() {
        // 1 second of a 440 Hz sine at 44.1 kHz, mono.
        let src_rate = 44_100u32;
        let n = src_rate as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / src_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        let out = prepare_for_whisper(AudioInput::RawPcm {
            samples: &samples,
            sample_rate: src_rate,
            channels: 1,
        })
        .expect("prepare");
        // Length should be ~ n * 16000 / 44100 = 16000. Allow ±2 for
        // rounding (round-half-up on the dst-len calculation).
        let expected = 16_000usize;
        let diff = out.len().abs_diff(expected);
        assert!(
            diff <= 2,
            "expected ~{expected} samples, got {} (diff {diff})",
            out.len()
        );
    }

    #[test]
    fn prepare_downmixes_stereo_to_mono() {
        // 16 kHz stereo, channel A = +1.0, channel B = -1.0.
        // Interleaved: [+1, -1, +1, -1, ...].
        let n_frames = 800usize;
        let mut interleaved = Vec::with_capacity(n_frames * 2);
        for _ in 0..n_frames {
            interleaved.push(1.0_f32);
            interleaved.push(-1.0_f32);
        }
        let out = prepare_for_whisper(AudioInput::RawPcm {
            samples: &interleaved,
            sample_rate: TARGET_SAMPLE_RATE,
            channels: 2,
        })
        .expect("prepare");
        assert_eq!(out.len(), n_frames);
        for s in out {
            assert!(s.abs() < 1e-6, "expected ~0, got {s}");
        }
    }

    #[test]
    fn prepare_rejects_zero_channels() {
        let result = prepare_for_whisper(AudioInput::RawPcm {
            samples: &[0.0; 16],
            sample_rate: TARGET_SAMPLE_RATE,
            channels: 0,
        });
        assert!(matches!(result, Err(AudioError::InvalidParameters(_))));
    }

    #[test]
    fn prepare_rejects_zero_sample_rate() {
        let result = prepare_for_whisper(AudioInput::RawPcm {
            samples: &[0.0; 16],
            sample_rate: 0,
            channels: 1,
        });
        assert!(matches!(result, Err(AudioError::InvalidParameters(_))));
    }

    #[test]
    fn audio_error_flattens_into_stt_error() {
        let err: SttError = AudioError::InvalidParameters("oops".into()).into();
        assert!(matches!(err, SttError::InvalidOptions(_)));
        let err: SttError = AudioError::InvalidWav("bad header".into()).into();
        assert!(matches!(err, SttError::Transcription(_)));
    }

    #[test]
    fn resample_passthrough_when_rates_match() {
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let out = resample_linear(&input, 16_000, 16_000);
        assert_eq!(out, input);
    }

    #[test]
    fn resample_handles_empty_input() {
        let out = resample_linear(&[], 44_100, 16_000);
        assert!(out.is_empty());
    }

    #[test]
    fn downmix_preserves_mono_amplitude() {
        // Two identical channels should downmix to the same waveform.
        let interleaved = vec![0.5_f32, 0.5, -0.25, -0.25, 0.75, 0.75];
        let mono = downmix_to_mono(&interleaved, 2);
        assert_eq!(mono, vec![0.5, -0.25, 0.75]);
    }

    #[test]
    fn decode_wav_rejects_truncated_input() {
        let result = decode_wav_16k_mono(&[0u8; 10]);
        assert!(matches!(result, Err(AudioError::InvalidWav(_))));
    }

    #[test]
    fn decode_wav_rejects_non_riff_header() {
        let mut buf = vec![0u8; 64];
        buf[..4].copy_from_slice(b"OGGS");
        let result = decode_wav_16k_mono(&buf);
        assert!(matches!(result, Err(AudioError::InvalidWav(_))));
    }
}
