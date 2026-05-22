//! F0 (fundamental-frequency / pitch) extractor for the RVC voice-conversion
//! backend.
//!
//! RVC drives its generator with a per-frame pitch trajectory in addition to
//! the content embeddings from [`super::content`]. The canonical extractor in
//! the upstream RVC repo is [RMVPE] -- a tiny CRNN that emits a 360-bin
//! softmax over cents-spaced pitch classes for each 10 ms frame of 16 kHz
//! audio. This module wraps the RMVPE ONNX checkpoint through
//! [`tract-onnx`], following the same pattern that
//! `crates/blazen-audio-stt/src/backends/whisper_streaming/vad.rs` uses for
//! the Silero VAD graph.
//!
//! [RMVPE]: https://github.com/Dream-High/RMVPE
//!
//! # Model assumptions
//!
//! This wrapper targets the ONNX export of RMVPE shipped by the upstream
//! RVC project (`rmvpe.onnx`, ~~ 180 MB). The graph has a single dynamic
//! audio input (`1 x N` f32 at 16 kHz) and a single logits output of shape
//! `1 x n_frames x 360`. Per the upstream Python at
//! <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/lib/rmvpe.py>
//! frames are 10 ms apart (`hop = 160` samples at 16 kHz) and each bin
//! covers 20 cents starting at 32.7 Hz (C1, MIDI 24). Decoding is:
//!
//! ```text
//! cents = bin * 20 + 1997.379408437619
//! hz    = 10 * 2 ** ((cents - 1884.0) / 1200)
//! ```
//!
//! which is equivalent to the more compact form used by the RVC Python:
//! `hz = 10 * 2 ** ((cents) / 1200)` where `cents` is rebased to
//! `bin * 20 + 1997.379...`. Since both terms are constants we collapse
//! them in [`bin_to_hz`] below.
//!
//! Unvoiced frames are signalled by a peak softmax value below
//! [`RmvpeF0Config::voiced_threshold`] (default 0.03 to match the
//! upstream `infer_from_audio` default), in which case [`RmvpeF0::extract`]
//! emits `0.0` for that frame.
//!
//! # `pitch_to_coarse`
//!
//! RVC's generator does not consume Hz directly. It takes a `pitchf` Hz
//! contour (for the f0-conditioned NSF path) **and** a `pitch` integer
//! contour where each frame is quantised onto a mel-spaced grid spanning
//! [`F0_MIN`, `F0_MAX`] with [`PITCH_COARSE_BINS`] - 2 internal bins (the
//! first and last bin are reserved for "below min" / "above max" /
//! "unvoiced"). The mel formula RVC uses is the HTK / Slaney "1127" mel:
//!
//! ```text
//! mel(f) = 1127 * ln(1 + f / 700)
//! ```
//!
//! and the quantisation is
//!
//! ```text
//! bin = round( (mel(f) - mel_min) / (mel_max - mel_min) * (n_bins - 2) ) + 1
//! ```
//!
//! clamped to `[1, n_bins - 1]`. Unvoiced (`hz == 0.0`) maps to bin `0`.
//! This is a deterministic pure function and is unit-tested below.

use std::path::Path;

use candle_core::Device;
use thiserror::Error;
use tract_onnx::prelude::*;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Sample rate RMVPE expects on its audio input.
pub const RMVPE_SAMPLE_RATE: usize = 16_000;

/// Hop size (samples) between successive RMVPE output frames.
///
/// 160 samples at 16 kHz is exactly 10 ms.
pub const RMVPE_HOP: usize = 160;

/// Number of pitch classes in the RMVPE softmax output.
pub const RMVPE_PITCH_BINS: usize = 360;

/// Cents per pitch bin in the RMVPE output.
pub const RMVPE_CENTS_PER_BIN: f32 = 20.0;

/// Cents offset of bin 0 in the RMVPE output. Corresponds to
/// `12 * log2(32.7 / 10) * 100` rounded to RVC's constant. This is the
/// value used by the upstream Python reference.
pub const RMVPE_CENTS_BASE: f32 = 1997.3794;

/// Default lower bound for the mel-spaced coarse pitch grid (Hz).
pub const F0_MIN: f32 = 50.0;

/// Default upper bound for the mel-spaced coarse pitch grid (Hz).
pub const F0_MAX: f32 = 1100.0;

/// Default number of coarse pitch bins. Bin 0 is reserved for
/// unvoiced / out-of-range frames.
pub const PITCH_COARSE_BINS: usize = 256;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`RmvpeF0`].
///
/// Wave D.3's `pipeline.rs` flattens these into
/// [`crate::error::VcError`] via a hand-written conversion alongside the
/// other rvc submodule errors.
#[derive(Debug, Error)]
pub enum F0Error {
    /// Loading the RMVPE ONNX file (file open, format parse, graph
    /// optimise, IO-shape inference) failed.
    #[error("rmvpe load: {0}")]
    ModelLoad(String),

    /// Inference (tract `run`) failed at runtime, or reshaping its
    /// outputs into the expected `(frames, bins)` layout failed.
    #[error("rmvpe inference: {0}")]
    Inference(String),

    /// Caller passed an audio buffer that violates the wrapper's
    /// contract (e.g. empty input).
    #[error("rmvpe invalid input: {0}")]
    InvalidInput(String),
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Compiled tract plan over the optimised RMVPE graph. Aliased to dodge
/// the clippy `type_complexity` lint on every field that mentions it,
/// mirroring `SileroPlan` in the STT crate.
type RmvpePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Static configuration for [`RmvpeF0`].
#[derive(Debug, Clone)]
pub struct RmvpeF0Config {
    /// Frames whose peak softmax (after the implicit sigmoid -> argmax)
    /// is below this value are treated as unvoiced and emit `0.0`.
    /// Defaults to `0.03` to match the upstream
    /// `RMVPE.infer_from_audio(thred=0.03)` default.
    pub voiced_threshold: f32,
}

impl Default for RmvpeF0Config {
    fn default() -> Self {
        Self {
            voiced_threshold: 0.03,
        }
    }
}

// ---------------------------------------------------------------------------
// RmvpeF0
// ---------------------------------------------------------------------------

/// RMVPE-based pitch extractor.
///
/// Loads the RMVPE ONNX graph once and runs it per [`extract`](Self::extract)
/// call. The wrapper itself is stateless across calls -- RMVPE is a
/// fully-feed-forward CRNN over the whole utterance (no LSTM state to
/// thread). The struct stores the compiled plan plus the resolved
/// input/output indices so we don't have to walk the graph metadata on
/// every call.
pub struct RmvpeF0 {
    model: RmvpePlan,
    config: RmvpeF0Config,
    /// Index of the audio input in the model's input list. We resolve
    /// this once at load by name (`"input"` or the first input if the
    /// name doesn't match) to be robust against re-exports that rename
    /// the input port.
    input_audio_ix: usize,
    /// Index of the logits output (`(1, n_frames, 360)`) in the model's
    /// output list.
    output_logits_ix: usize,
}

impl std::fmt::Debug for RmvpeF0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RmvpeF0")
            .field("config", &self.config)
            .field("input_audio_ix", &self.input_audio_ix)
            .field("output_logits_ix", &self.output_logits_ix)
            .finish_non_exhaustive()
    }
}

impl RmvpeF0 {
    /// Load an RMVPE ONNX model from disk.
    ///
    /// `device` is accepted for API uniformity with the sibling
    /// [`super::content::ContentEncoder`] (which lives on a candle
    /// [`Device`]) but currently ignored: tract-onnx is CPU-only in the
    /// 0.22 release. The argument is taken by reference and immediately
    /// forgotten; we keep it in the signature so the
    /// [`super::pipeline`] driver in Wave D.3 can pass a single
    /// `&Device` to every sub-component without special-casing F0.
    ///
    /// # Errors
    ///
    /// Returns [`F0Error::ModelLoad`] when the file can't be read, when
    /// `tract-onnx` fails to optimise the graph, or when neither the
    /// expected named ports nor a positional fallback can be resolved.
    pub fn load(model_path: &Path, device: &Device) -> Result<Self, F0Error> {
        Self::load_with_config(model_path, device, RmvpeF0Config::default())
    }

    /// [`Self::load`] with an explicit [`RmvpeF0Config`].
    ///
    /// # Errors
    ///
    /// See [`Self::load`].
    pub fn load_with_config(
        model_path: &Path,
        _device: &Device,
        config: RmvpeF0Config,
    ) -> Result<Self, F0Error> {
        let inference = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| F0Error::ModelLoad(format!("open {}: {e}", model_path.display())))?;

        // Try to locate by canonical port names first, then fall back
        // to positional indices. Different exporters call the audio
        // input "input", "waveform", "audio", "x", etc. -- we don't
        // want to hard-fail just because an alternate exporter is used.
        let input_audio_ix =
            find_input_index(&inference, &["input", "waveform", "audio", "x"]).unwrap_or(0);
        let output_logits_ix =
            find_output_index(&inference, &["output", "logits", "salience", "y"]).unwrap_or(0);

        if input_audio_ix >= inference.inputs.len() {
            return Err(F0Error::ModelLoad(format!(
                "resolved audio input index {input_audio_ix} >= input count {}",
                inference.inputs.len()
            )));
        }
        if output_logits_ix >= inference.outputs.len() {
            return Err(F0Error::ModelLoad(format!(
                "resolved logits output index {output_logits_ix} >= output count {}",
                inference.outputs.len()
            )));
        }

        let optimized = inference
            .into_optimized()
            .map_err(|e| F0Error::ModelLoad(format!("optimize: {e}")))?;
        let model = optimized
            .into_runnable()
            .map_err(|e| F0Error::ModelLoad(format!("into_runnable: {e}")))?;

        Ok(Self {
            model,
            config,
            input_audio_ix,
            output_logits_ix,
        })
    }

    /// Borrow the wrapper's config.
    #[must_use]
    pub const fn config(&self) -> &RmvpeF0Config {
        &self.config
    }

    /// Extract a per-frame pitch contour (Hz) from a mono 16 kHz buffer.
    ///
    /// Returns one value per 10 ms frame. The output length is
    /// `samples_16khz.len().div_ceil(RMVPE_HOP)` -- if the model's own
    /// frame count differs (which happens when the exporter rounds
    /// differently), the result is right-padded with zeros or trimmed
    /// to match this contract. Frames whose peak softmax falls below
    /// [`RmvpeF0Config::voiced_threshold`] emit `0.0` (unvoiced).
    ///
    /// # Errors
    ///
    /// Returns [`F0Error::InvalidInput`] for an empty buffer, or
    /// [`F0Error::Inference`] when the ONNX graph fails to run or when
    /// its output cannot be reshaped into the expected
    /// `(n_frames, RMVPE_PITCH_BINS)` layout.
    pub fn extract(&self, samples_16khz: &[f32]) -> Result<Vec<f32>, F0Error> {
        if samples_16khz.is_empty() {
            return Err(F0Error::InvalidInput(
                "empty audio buffer; need at least one sample".into(),
            ));
        }

        let n_samples = samples_16khz.len();
        let expected_frames = n_samples.div_ceil(RMVPE_HOP);

        // (1, N) f32 audio tensor.
        let audio = tract_ndarray::Array2::from_shape_vec((1, n_samples), samples_16khz.to_vec())
            .map_err(|e| F0Error::Inference(format!("reshape audio (1, {n_samples}): {e}")))?
            .into_tensor();

        let n_inputs = self.model.model().inputs.len();
        let mut ordered: Vec<Option<TValue>> = vec![None; n_inputs];
        ordered[self.input_audio_ix] = Some(TValue::from_const(audio.into()));
        let ordered_inputs: TVec<TValue> = ordered
            .into_iter()
            .map(|v| v.ok_or_else(|| F0Error::Inference("rmvpe missing input slot".into())))
            .collect::<Result<_, _>>()?;

        let outputs = self
            .model
            .run(ordered_inputs)
            .map_err(|e| F0Error::Inference(format!("tract run: {e}")))?;

        let logits = outputs
            .get(self.output_logits_ix)
            .ok_or_else(|| F0Error::Inference("rmvpe missing logits output".into()))?;
        let slice = logits
            .as_slice::<f32>()
            .map_err(|e| F0Error::Inference(format!("logits as_slice: {e}")))?;

        // Expected layout: (1, n_frames, 360) row-major. Verify the
        // total element count is a multiple of RMVPE_PITCH_BINS and
        // infer n_frames from it.
        if !slice.len().is_multiple_of(RMVPE_PITCH_BINS) {
            return Err(F0Error::Inference(format!(
                "logits length {} not a multiple of {}",
                slice.len(),
                RMVPE_PITCH_BINS
            )));
        }
        let model_frames = slice.len() / RMVPE_PITCH_BINS;

        // Decode each frame: softmax-argmax across the 360 bins, gate
        // on the peak softmax value, then convert bin -> Hz.
        let mut hz = Vec::with_capacity(model_frames);
        for f in 0..model_frames {
            let row = &slice[f * RMVPE_PITCH_BINS..(f + 1) * RMVPE_PITCH_BINS];
            let (best_bin, best_prob) = softmax_argmax(row);
            if best_prob < self.config.voiced_threshold {
                hz.push(0.0);
            } else {
                hz.push(bin_to_hz(best_bin));
            }
        }

        // Right-pad with zeros (unvoiced) or trim to match the contract
        // length (one frame per 10 ms of input).
        if hz.len() < expected_frames {
            hz.resize(expected_frames, 0.0);
        } else if hz.len() > expected_frames {
            hz.truncate(expected_frames);
        }

        Ok(hz)
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Numerically stable softmax + argmax over a single frame's logits.
/// Returns `(best_bin, best_softmax_value)`.
fn softmax_argmax(logits: &[f32]) -> (usize, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }
    let mut max_logit = f32::NEG_INFINITY;
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > max_logit {
            max_logit = l;
            best = i;
        }
    }
    // Compute the softmax denominator and the numerator at `best`.
    let mut sum_exp = 0.0_f32;
    for &l in logits {
        sum_exp += (l - max_logit).exp();
    }
    let prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
    (best, prob)
}

/// Convert an RMVPE bin index to Hz.
///
/// Derivation (matching the upstream RVC Python):
///
/// ```text
/// cents = bin * RMVPE_CENTS_PER_BIN + RMVPE_CENTS_BASE
/// hz    = 10 * 2 ** ((cents - 1884.0) / 1200)
/// ```
///
/// `(cents - 1884.0) / 1200 = (bin * 20 + 113.379_4) / 1200`. We could
/// fold the constants further but keeping them explicit makes the
/// derivation auditable against the reference.
#[must_use]
pub fn bin_to_hz(bin: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)] // bin <= 360, exact f32
    let bin_f = bin as f32;
    let cents = bin_f * RMVPE_CENTS_PER_BIN + RMVPE_CENTS_BASE;
    10.0 * 2.0_f32.powf((cents - 1884.0) / 1200.0)
}

/// Mel scale used by RVC for coarse pitch quantisation: HTK 1127 mel.
#[must_use]
pub fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Quantise a per-frame Hz contour onto the mel-spaced coarse-pitch
/// grid RVC's generator consumes.
///
/// - `pitch_hz`: one value per frame; `0.0` means unvoiced.
/// - `f0_min` / `f0_max`: Hz bounds for the grid. RVC uses
///   ([`F0_MIN`], [`F0_MAX`]) = (50, 1100).
/// - `n_bins`: total bin count, including the reserved bin 0 for
///   unvoiced. RVC uses [`PITCH_COARSE_BINS`] = 256.
///
/// Returns a `Vec<u32>` of the same length as `pitch_hz`. Bin 0 is
/// reserved for unvoiced; voiced frames land in `[1, n_bins - 1]`. If
/// `n_bins < 2` every frame collapses to 0.
#[must_use]
pub fn pitch_to_coarse(pitch_hz: &[f32], f0_min: f32, f0_max: f32, n_bins: usize) -> Vec<u32> {
    if n_bins < 2 {
        return vec![0; pitch_hz.len()];
    }
    let mel_min = hz_to_mel(f0_min);
    let mel_max = hz_to_mel(f0_max);
    // `(n_bins - 2)` because bin 0 is reserved for unvoiced and bin
    // `n_bins - 1` is the inclusive top of the voiced range. The
    // `+ 1` in the formula shifts the voiced grid up so bin 0 stays
    // free for "no pitch".
    #[allow(clippy::cast_precision_loss)] // n_bins typically 256, exact f32
    let span = (n_bins - 2) as f32;
    let mel_range = mel_max - mel_min;

    pitch_hz
        .iter()
        .map(|&hz| {
            if hz <= 0.0 {
                return 0_u32;
            }
            // Fast path: avoid `ln` when the input is already known to
            // be out of range. Saves a transcendental per unvoiced /
            // clipped frame.
            if hz <= f0_min {
                return 1;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            if hz >= f0_max {
                return (n_bins - 1) as u32;
            }
            let mel = hz_to_mel(hz);
            // Normalise into [0, 1], scale to the voiced grid span,
            // round, then shift past the reserved bin 0.
            let frac = ((mel - mel_min) / mel_range).clamp(0.0, 1.0);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let bin = (frac * span).round() as i64 + 1;
            // Clamp to [1, n_bins - 1].
            #[allow(clippy::cast_possible_wrap)]
            let upper = (n_bins - 1) as i64;
            let clamped = bin.clamp(1, upper);
            // clamped is in [1, n_bins - 1]; n_bins is `usize` and we
            // never call this with `n_bins >= u32::MAX`, so the cast is
            // lossless. Allowed truncation/sign-loss reflects that.
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            {
                clamped as u32
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Resolve an input outlet by trying each candidate name in order.
fn find_input_index(model: &InferenceModel, candidates: &[&str]) -> Option<usize> {
    for name in candidates {
        for (ix, outlet) in model.inputs.iter().enumerate() {
            if model.nodes()[outlet.node].name == *name {
                return Some(ix);
            }
        }
    }
    None
}

/// Resolve an output outlet by trying each candidate name in order.
fn find_output_index(model: &InferenceModel, candidates: &[&str]) -> Option<usize> {
    for name in candidates {
        for (ix, outlet) in model.outputs.iter().enumerate() {
            if model.nodes()[outlet.node].name == *name {
                return Some(ix);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_upstream_rmvpe_default() {
        let cfg = RmvpeF0Config::default();
        assert!((cfg.voiced_threshold - 0.03).abs() < f32::EPSILON);
    }

    #[test]
    fn constants_match_rmvpe_paper() {
        assert_eq!(RMVPE_SAMPLE_RATE, 16_000);
        assert_eq!(RMVPE_HOP, 160);
        assert_eq!(RMVPE_PITCH_BINS, 360);
        assert!((RMVPE_CENTS_PER_BIN - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn hz_to_mel_round_trip_zero() {
        // mel(0) = 0 by construction.
        assert!(hz_to_mel(0.0).abs() < 1e-6);
    }

    #[test]
    fn hz_to_mel_monotonic() {
        // The mel scale is strictly increasing for Hz >= 0.
        let mut prev = hz_to_mel(0.0);
        for hz in [50.0_f32, 100.0, 220.0, 440.0, 880.0, 1100.0] {
            let m = hz_to_mel(hz);
            assert!(
                m > prev,
                "hz_to_mel not monotonic at {hz}: prev={prev}, m={m}"
            );
            prev = m;
        }
    }

    #[test]
    fn pitch_to_coarse_unvoiced_maps_to_bin_zero() {
        let coarse = pitch_to_coarse(&[0.0, 0.0, 0.0], F0_MIN, F0_MAX, PITCH_COARSE_BINS);
        assert_eq!(coarse, vec![0, 0, 0]);
    }

    #[test]
    fn pitch_to_coarse_negative_treated_as_unvoiced() {
        // Defensive: some upstream extractors emit small negative
        // numbers when unvoiced. We treat anything <= 0 as unvoiced.
        let coarse = pitch_to_coarse(&[-1.0, -0.0], F0_MIN, F0_MAX, PITCH_COARSE_BINS);
        assert_eq!(coarse, vec![0, 0]);
    }

    #[test]
    fn pitch_to_coarse_220hz_lands_strictly_inside_voiced_range() {
        // 220 Hz (A3) sits well inside [50, 1100]. On the HTK-1127 mel
        // scale, [50, 1100] Hz spans mel approx [77.9, 1064.0] and
        // 220 Hz maps to mel approx 308.0, so the normalised position
        // is approx 0.23 -- well inside the voiced range, never on a
        // boundary. We assert it lands strictly inside `(1, n_bins-1)`
        // rather than a single bin to stay robust to small rounding
        // tweaks in the formula. We also pin it to within the lower
        // half because the mel compression genuinely pushes 220 Hz
        // below the geometric midpoint -- this is the upstream RVC
        // behaviour, not a bug.
        let coarse = pitch_to_coarse(&[220.0], F0_MIN, F0_MAX, PITCH_COARSE_BINS);
        assert_eq!(coarse.len(), 1);
        let bin = coarse[0];
        let last = u32::try_from(PITCH_COARSE_BINS - 1).expect("bin count fits in u32");
        let half = u32::try_from(PITCH_COARSE_BINS / 2).expect("bin count fits in u32");
        assert!(
            (2..last).contains(&bin),
            "220 Hz should be a strictly interior voiced bin in [2, {last}); got {bin}"
        );
        assert!(
            bin < half,
            "220 Hz should be below the mel midpoint of [50, 1100] Hz; got bin {bin} >= {half}"
        );
    }

    #[test]
    fn pitch_to_coarse_above_max_clamps_to_last_bin() {
        let coarse = pitch_to_coarse(&[2_000.0, 5_000.0], F0_MIN, F0_MAX, PITCH_COARSE_BINS);
        let last = u32::try_from(PITCH_COARSE_BINS - 1).expect("bin count fits in u32");
        assert_eq!(coarse, vec![last, last]);
    }

    #[test]
    fn pitch_to_coarse_at_or_below_min_clamps_to_first_voiced_bin() {
        let coarse = pitch_to_coarse(&[10.0, 50.0], F0_MIN, F0_MAX, PITCH_COARSE_BINS);
        assert_eq!(coarse, vec![1, 1]);
    }

    #[test]
    fn pitch_to_coarse_is_monotonic_in_voiced_range() {
        // Bin index should be (weakly) non-decreasing as Hz rises
        // through the voiced grid.
        let xs: Vec<f32> = (1_u8..=20).map(|i| 50.0 + f32::from(i) * 50.0).collect();
        let coarse = pitch_to_coarse(&xs, F0_MIN, F0_MAX, PITCH_COARSE_BINS);
        for w in coarse.windows(2) {
            assert!(
                w[0] <= w[1],
                "coarse not monotonic: {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn pitch_to_coarse_degenerate_nbins_returns_all_zero() {
        let coarse = pitch_to_coarse(&[100.0, 200.0, 300.0], F0_MIN, F0_MAX, 1);
        assert_eq!(coarse, vec![0, 0, 0]);
    }

    #[test]
    fn bin_to_hz_is_monotonic() {
        // RMVPE bin -> Hz is strictly increasing in bin.
        let mut prev = bin_to_hz(0);
        for b in 1..RMVPE_PITCH_BINS {
            let cur = bin_to_hz(b);
            assert!(cur > prev, "bin_to_hz not monotonic at bin {b}");
            prev = cur;
        }
    }

    #[test]
    fn softmax_argmax_handles_uniform_logits() {
        // Uniform logits -> probability 1/N at every bin; argmax picks
        // the first by convention.
        let logits = [1.0_f32; 8];
        let (bin, prob) = softmax_argmax(&logits);
        assert_eq!(bin, 0);
        assert!((prob - 1.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_argmax_handles_one_hot() {
        // A single very large logit -> probability ~ 1.0 at that bin.
        let mut logits = vec![0.0_f32; RMVPE_PITCH_BINS];
        logits[123] = 50.0;
        let (bin, prob) = softmax_argmax(&logits);
        assert_eq!(bin, 123);
        assert!(prob > 0.9999);
    }

    #[test]
    fn softmax_argmax_empty_returns_zero() {
        let (bin, prob) = softmax_argmax(&[]);
        assert_eq!(bin, 0);
        assert!(prob.abs() < f32::EPSILON);
    }
}
