//! Silero VAD wrapper.
//!
//! Loads `silero_vad.onnx` via `tract-onnx` and exposes a streaming
//! API: feed 512-sample 16 kHz f32 chunks, receive voice-activity
//! flags + utterance boundaries.
//!
//! # Model assumptions
//!
//! This wrapper targets the **Silero VAD v4** ONNX graph published at
//! <https://github.com/snakers4/silero-vad> (file
//! `files/silero_vad.onnx`, ~~ February 2024 release). That graph has
//! three inputs (`input`, `h`, `c`) and three outputs (`output`, `hn`,
//! `cn`). The audio frame must be exactly **512 samples at 16 kHz** (32
//! ms); other sizes change the model's receptive field and produce
//! undefined behaviour. The newer v5 graph (Jul 2024) collapsed `h`/`c`
//! into a single `state` input and added an `sr` scalar — running v5
//! weights through this wrapper without adjusting the IO names will
//! fail at load time.
//!
//! # `HuggingFace` source
//!
//! [`SileroVad::from_hf`] downloads the v4 ONNX from the
//! [`deepghs/silero-vad-onnx`](https://huggingface.co/deepghs/silero-vad-onnx)
//! HF mirror (file `silero_vad.onnx`, 2.33 MB — byte-identical to the
//! file shipped at `src/silero_vad/data/silero_vad.onnx` in the upstream
//! GitHub repo), pinned to revision
//! `8547eb3c577a6f712c1ed1a554c21c5d9137867d` (the "Upload 2 files"
//! commit from 2024-08-31, which uploaded the v4 weights themselves —
//! we deliberately skip the later README-update commit so the pin
//! tracks the binary content).
//!
//! The other obvious HF mirror, `onnx-community/silero-vad/onnx/model.onnx`
//! (2.24 MB), is the **v5** graph (smaller; different IO names) and is
//! deliberately *not* used here — it would fail this wrapper's
//! `input`/`h`/`c` lookup at load time.
//!
//! The LSTM hidden states have shape `[2, 1, 64]` for both `h` and `c`.
//! `feed` threads them across forward calls so the model can integrate
//! evidence over the audio stream.
//!
//! # Utterance-boundary hysteresis
//!
//! `feed` applies a two-counter hysteresis on top of the raw speech
//! probability:
//!
//! - `active_frames` counts consecutive frames whose probability
//!   exceeds [`SileroVadConfig::speech_threshold`]; crossing
//!   [`SileroVadConfig::utterance_start_frames`] emits
//!   [`VadFrame::utterance_start`] once.
//! - `inactive_frames` counts the converse; crossing
//!   [`SileroVadConfig::utterance_end_frames`] emits
//!   [`VadFrame::utterance_end`].
//!
//! This matches the upstream Python reference's `VADIterator` semantics
//! at default settings.

#![cfg(feature = "whisper-streaming")]

use std::path::Path;

use tract_onnx::prelude::*;

use crate::error::SttError;

/// Compiled Silero VAD plan. Hides the verbose nested generics of
/// `tract`'s [`SimplePlan`] generic over a typed graph so clippy's
/// `type_complexity` lint doesn't fire on every field that mentions it.
type SileroPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Sample rate the Silero VAD v4 graph expects.
pub(crate) const VAD_SAMPLE_RATE: usize = 16_000;

/// Frame size (samples) the Silero VAD v4 graph expects.
pub(crate) const VAD_FRAME_SIZE: usize = 512;

/// `HuggingFace` repo hosting the v4 ONNX weights (see crate-level
/// docs in this module for why we pin this specific mirror).
pub(crate) const VAD_HF_REPO: &str = "deepghs/silero-vad-onnx";

/// File name inside [`VAD_HF_REPO`] for the v4 weights.
pub(crate) const VAD_HF_FILE: &str = "silero_vad.onnx";

/// Commit SHA the v4 ONNX is pinned to. See the module-level
/// "`HuggingFace` source" doc for the choice rationale.
pub(crate) const VAD_HF_REVISION: &str = "8547eb3c577a6f712c1ed1a554c21c5d9137867d";

/// Shape of the LSTM hidden states (`h` and `c`).
const VAD_STATE_SHAPE: [usize; 3] = [2, 1, 64];

/// One emission from [`SileroVad::feed`].
#[derive(Debug, Clone)]
pub struct VadFrame {
    /// Voice-activity probability in `[0, 1]`.
    pub speech_prob: f32,
    /// Whether this frame transitioned the wrapper into an active
    /// utterance (rises exactly once per utterance, on the frame that
    /// crossed [`SileroVadConfig::utterance_start_frames`]).
    pub utterance_start: bool,
    /// Whether this frame transitioned the wrapper out of an active
    /// utterance (rises exactly once per utterance, on the frame that
    /// crossed [`SileroVadConfig::utterance_end_frames`]).
    pub utterance_end: bool,
}

/// Static configuration for [`SileroVad`].
#[derive(Debug, Clone)]
pub struct SileroVadConfig {
    /// Speech-probability threshold for considering a frame "active".
    /// Defaults to `0.5` (the upstream `VADIterator` default).
    pub speech_threshold: f32,
    /// Minimum consecutive active frames to declare an utterance start.
    /// Defaults to `8` (~256 ms at 32 ms/frame).
    pub utterance_start_frames: usize,
    /// Minimum consecutive inactive frames to declare an utterance end.
    /// Defaults to `16` (~512 ms at 32 ms/frame).
    pub utterance_end_frames: usize,
}

impl Default for SileroVadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            utterance_start_frames: 8,
            utterance_end_frames: 16,
        }
    }
}

/// Streaming Silero VAD wrapper.
///
/// Single-stream and not `Sync`-safe by itself: each independent audio
/// stream needs its own [`SileroVad`] (or a call to [`SileroVad::reset`]
/// between streams). The wrapper owns the LSTM hidden state and the
/// hysteresis counters.
pub struct SileroVad {
    model: SileroPlan,
    config: SileroVadConfig,
    /// LSTM hidden state `h` of shape `[2, 1, 64]`. Threaded across
    /// `feed` calls.
    h: Tensor,
    /// LSTM cell state `c` of shape `[2, 1, 64]`. Threaded across
    /// `feed` calls.
    c: Tensor,
    /// Index of the input named `input` in the model's input list.
    input_audio_ix: usize,
    /// Index of the input named `h`.
    input_h_ix: usize,
    /// Index of the input named `c`.
    input_cell_ix: usize,
    /// Index of the output named `output` (speech probability).
    output_prob_ix: usize,
    /// Index of the output named `hn` (new h state).
    output_hn_ix: usize,
    /// Index of the output named `cn` (new c state).
    output_cell_ix: usize,
    /// Consecutive frames at or above `speech_threshold`.
    active_frames: usize,
    /// Consecutive frames below `speech_threshold`.
    inactive_frames: usize,
    /// Whether the wrapper currently considers itself inside an
    /// utterance (toggled by the hysteresis rules above).
    is_in_utterance: bool,
}

impl std::fmt::Debug for SileroVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SileroVad")
            .field("config", &self.config)
            .field("active_frames", &self.active_frames)
            .field("inactive_frames", &self.inactive_frames)
            .field("is_in_utterance", &self.is_in_utterance)
            .finish_non_exhaustive()
    }
}

impl SileroVad {
    /// Load a Silero VAD model from disk.
    ///
    /// `onnx_path` must point at a Silero VAD **v4** ONNX file. Other
    /// versions use different input/output names and will fail with
    /// [`SttError::ModelLoad`].
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] when the file can't be read, the
    /// graph can't be optimised, or any of the expected IO names
    /// (`input`/`h`/`c` → `output`/`hn`/`cn`) are missing.
    pub fn from_path(onnx_path: &Path, config: SileroVadConfig) -> Result<Self, SttError> {
        // Fix the input shape so optimisation can specialise the LSTM
        // for our single-batch single-frame contract. `tract-onnx`
        // requires a concrete fact when calling `into_optimized`.
        let input_fact = InferenceFact::dt_shape(f32::datum_type(), [1usize, VAD_FRAME_SIZE]);
        let state_fact = InferenceFact::dt_shape(f32::datum_type(), VAD_STATE_SHAPE);

        let mut inference = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .map_err(|e| SttError::ModelLoad(format!("silero-vad load: {e}")))?;

        // Locate the named inputs/outputs so we can pass them in the
        // order tract expects (positional, not keyed by name).
        let input_audio_ix = find_input_index(&inference, "input")?;
        let input_h_ix = find_input_index(&inference, "h")?;
        let input_cell_ix = find_input_index(&inference, "c")?;
        let output_prob_ix = find_output_index(&inference, "output")?;
        let output_hn_ix = find_output_index(&inference, "hn")?;
        let output_cell_ix = find_output_index(&inference, "cn")?;

        inference
            .set_input_fact(input_audio_ix, input_fact)
            .map_err(|e| SttError::ModelLoad(format!("silero-vad set audio fact: {e}")))?;
        inference
            .set_input_fact(input_h_ix, state_fact.clone())
            .map_err(|e| SttError::ModelLoad(format!("silero-vad set h fact: {e}")))?;
        inference
            .set_input_fact(input_cell_ix, state_fact)
            .map_err(|e| SttError::ModelLoad(format!("silero-vad set c fact: {e}")))?;

        let optimized = inference
            .into_optimized()
            .map_err(|e| SttError::ModelLoad(format!("silero-vad optimize: {e}")))?;
        let model = optimized
            .into_runnable()
            .map_err(|e| SttError::ModelLoad(format!("silero-vad runnable: {e}")))?;

        let (h, c) = zero_states()?;

        Ok(Self {
            model,
            config,
            h,
            c,
            input_audio_ix,
            input_h_ix,
            input_cell_ix,
            output_prob_ix,
            output_hn_ix,
            output_cell_ix,
            active_frames: 0,
            inactive_frames: 0,
            is_in_utterance: false,
        })
    }

    /// Download the Silero VAD v4 ONNX from `HuggingFace` and load it.
    ///
    /// Uses `hf-hub` (already a `blazen-audio-stt` dep via the
    /// `whisper-streaming` feature → `candle`) to fetch
    /// [`VAD_HF_FILE`] from [`VAD_HF_REPO`] at [`VAD_HF_REVISION`] into
    /// the HF cache directory; on cache hit no network is touched. The
    /// model file is ~2.3 MB.
    ///
    /// Pinned to a specific commit SHA for reproducibility — Silero VAD
    /// publishes new model versions periodically and the v5 release
    /// renamed the ONNX inputs in a way that would silently break this
    /// wrapper's IO lookup.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] when the HF download fails (no
    /// network, repo gone, file moved, revision rotated, etc.) or when
    /// the downloaded ONNX fails to load (see [`Self::from_path`]).
    pub async fn from_hf(config: SileroVadConfig) -> Result<Self, SttError> {
        // `hf-hub` is only available behind the `ureq` feature on this
        // crate (see `blazen-audio-stt/Cargo.toml`). `ureq::Api` is
        // blocking, so we dispatch the download itself on
        // `spawn_blocking` — same shape as `candle.rs`'s `load_blocking`.
        let onnx_path = tokio::task::spawn_blocking(|| {
            let api = hf_hub::api::sync::ApiBuilder::new()
                .build()
                .map_err(|e| SttError::ModelLoad(format!("silero-vad hf-hub api: {e}")))?;
            let repo = api.repo(hf_hub::Repo::with_revision(
                VAD_HF_REPO.to_string(),
                hf_hub::RepoType::Model,
                VAD_HF_REVISION.to_string(),
            ));
            repo.get(VAD_HF_FILE).map_err(|e| {
                SttError::ModelLoad(format!(
                    "silero-vad hf-hub fetch {VAD_HF_REPO}@{VAD_HF_REVISION}/{VAD_HF_FILE}: {e}"
                ))
            })
        })
        .await
        .map_err(|e| SttError::ModelLoad(format!("silero-vad spawn_blocking: {e}")))??;

        Self::from_path(&onnx_path, config)
    }

    /// Feed a 512-sample 16 kHz f32 frame; return a [`VadFrame`].
    ///
    /// # Errors
    ///
    /// Returns [`SttError::InvalidOptions`] when `audio_frame.len() !=
    /// 512`, or [`SttError::Transcription`] when ONNX inference fails.
    pub fn feed(&mut self, audio_frame: &[f32]) -> Result<VadFrame, SttError> {
        if audio_frame.len() != VAD_FRAME_SIZE {
            return Err(SttError::InvalidOptions(format!(
                "silero-vad expects {VAD_FRAME_SIZE}-sample 16 kHz frames; got {}",
                audio_frame.len()
            )));
        }

        let audio_tensor =
            tract_ndarray::Array2::from_shape_vec((1, VAD_FRAME_SIZE), audio_frame.to_vec())
                .map_err(|e| SttError::Transcription(format!("silero-vad audio reshape: {e}")))?
                .into_tensor();

        // Build positional inputs in the order tract expects.
        let mut inputs: TVec<TValue> = tvec!(
            TValue::from_const(audio_tensor.into()),
            TValue::from_const(self.h.clone().into()),
            TValue::from_const(self.c.clone().into()),
        );
        // Reorder so the slot at `input_audio_ix` holds the audio
        // tensor, etc. We initialised them in slot order [audio, h, c]
        // above; rotate into the model's actual ordering.
        let mut ordered: Vec<Option<TValue>> = vec![None; 3];
        ordered[self.input_audio_ix] = Some(inputs.remove(0));
        ordered[self.input_h_ix] = Some(inputs.remove(0));
        ordered[self.input_cell_ix] = Some(inputs.remove(0));
        let ordered_inputs: TVec<TValue> = ordered
            .into_iter()
            .map(|v| {
                v.ok_or_else(|| SttError::Transcription("silero-vad missing input slot".into()))
            })
            .collect::<Result<_, _>>()?;

        let outputs = self
            .model
            .run(ordered_inputs)
            .map_err(|e| SttError::Transcription(format!("silero-vad run: {e}")))?;

        let prob_tensor = outputs
            .get(self.output_prob_ix)
            .ok_or_else(|| SttError::Transcription("silero-vad missing prob output".into()))?;
        let probs = prob_tensor
            .as_slice::<f32>()
            .map_err(|e| SttError::Transcription(format!("silero-vad prob slice: {e}")))?;
        let speech_prob = probs
            .first()
            .copied()
            .ok_or_else(|| SttError::Transcription("silero-vad prob tensor empty".into()))?;

        // Threaded states.
        let hn = outputs
            .get(self.output_hn_ix)
            .ok_or_else(|| SttError::Transcription("silero-vad missing hn output".into()))?;
        let cn = outputs
            .get(self.output_cell_ix)
            .ok_or_else(|| SttError::Transcription("silero-vad missing cn output".into()))?;
        self.h = hn.clone().into_tensor();
        self.c = cn.clone().into_tensor();

        Ok(self.apply_hysteresis(speech_prob))
    }

    /// Reset the LSTM hidden state and hysteresis counters (e.g.
    /// between independent audio streams).
    ///
    /// # Errors
    ///
    /// Returns [`SttError::Transcription`] when the zero-state tensors
    /// cannot be allocated (effectively never on a healthy system).
    pub fn reset(&mut self) -> Result<(), SttError> {
        let (h, c) = zero_states()?;
        self.h = h;
        self.c = c;
        self.active_frames = 0;
        self.inactive_frames = 0;
        self.is_in_utterance = false;
        Ok(())
    }

    /// Number of samples the model expects per frame (always 512).
    #[must_use]
    pub const fn frame_size() -> usize {
        VAD_FRAME_SIZE
    }

    /// Sample rate the model expects (always 16 000).
    #[must_use]
    pub const fn sample_rate() -> usize {
        VAD_SAMPLE_RATE
    }

    /// Borrow the wrapper's config.
    #[must_use]
    pub const fn config(&self) -> &SileroVadConfig {
        &self.config
    }

    /// Test-only constructor that exercises the hysteresis logic without
    /// loading an ONNX file. Used by `cargo test` to validate boundary
    /// detection on synthetic probability streams.
    #[cfg(test)]
    fn dummy(config: SileroVadConfig) -> Self {
        Self {
            model: dummy_model(),
            config,
            h: Tensor::zero::<f32>(&VAD_STATE_SHAPE).expect("zero h"),
            c: Tensor::zero::<f32>(&VAD_STATE_SHAPE).expect("zero c"),
            input_audio_ix: 0,
            input_h_ix: 1,
            input_cell_ix: 2,
            output_prob_ix: 0,
            output_hn_ix: 1,
            output_cell_ix: 2,
            active_frames: 0,
            inactive_frames: 0,
            is_in_utterance: false,
        }
    }

    /// Apply the active/inactive-frame hysteresis to a raw probability.
    /// Pure logic — extracted so unit tests can drive it without ONNX.
    fn apply_hysteresis(&mut self, speech_prob: f32) -> VadFrame {
        let active = speech_prob >= self.config.speech_threshold;
        let mut utterance_start = false;
        let mut utterance_end = false;

        if active {
            self.active_frames = self.active_frames.saturating_add(1);
            self.inactive_frames = 0;
            if !self.is_in_utterance && self.active_frames >= self.config.utterance_start_frames {
                self.is_in_utterance = true;
                utterance_start = true;
            }
        } else {
            self.inactive_frames = self.inactive_frames.saturating_add(1);
            self.active_frames = 0;
            if self.is_in_utterance && self.inactive_frames >= self.config.utterance_end_frames {
                self.is_in_utterance = false;
                utterance_end = true;
            }
        }

        VadFrame {
            speech_prob,
            utterance_start,
            utterance_end,
        }
    }
}

fn zero_states() -> Result<(Tensor, Tensor), SttError> {
    let h = Tensor::zero::<f32>(&VAD_STATE_SHAPE)
        .map_err(|e| SttError::ModelLoad(format!("silero-vad h alloc: {e}")))?;
    let c = Tensor::zero::<f32>(&VAD_STATE_SHAPE)
        .map_err(|e| SttError::ModelLoad(format!("silero-vad c alloc: {e}")))?;
    Ok((h, c))
}

fn find_input_index(model: &InferenceModel, name: &str) -> Result<usize, SttError> {
    for (ix, outlet) in model.inputs.iter().enumerate() {
        if model.nodes()[outlet.node].name == name {
            return Ok(ix);
        }
    }
    Err(SttError::ModelLoad(format!(
        "silero-vad: input named `{name}` not found (graph version mismatch? this wrapper expects v4)"
    )))
}

fn find_output_index(model: &InferenceModel, name: &str) -> Result<usize, SttError> {
    for (ix, outlet) in model.outputs.iter().enumerate() {
        if model.nodes()[outlet.node].name == name {
            return Ok(ix);
        }
    }
    Err(SttError::ModelLoad(format!(
        "silero-vad: output named `{name}` not found (graph version mismatch? this wrapper expects v4)"
    )))
}

/// A trivial single-node model used only by the test-only
/// [`SileroVad::dummy`] constructor. It is never executed — the tests
/// that use it drive [`SileroVad::apply_hysteresis`] directly.
#[cfg(test)]
fn dummy_model() -> SileroPlan {
    let mut model = TypedModel::default();
    let _ = model
        .add_source("input", f32::fact([1usize, VAD_FRAME_SIZE]))
        .expect("dummy source");
    model.into_runnable().expect("dummy into_runnable")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_upstream_vaditerator() {
        let cfg = SileroVadConfig::default();
        assert!((cfg.speech_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.utterance_start_frames, 8);
        assert_eq!(cfg.utterance_end_frames, 16);
    }

    #[test]
    fn frame_size_and_sample_rate_are_v4_constants() {
        assert_eq!(SileroVad::frame_size(), 512);
        assert_eq!(SileroVad::sample_rate(), 16_000);
    }

    #[test]
    fn hysteresis_emits_start_after_threshold_frames() {
        let mut vad = SileroVad::dummy(SileroVadConfig {
            utterance_start_frames: 3,
            ..SileroVadConfig::default()
        });
        // Two active frames are not enough.
        for _ in 0..2 {
            let f = vad.apply_hysteresis(0.9);
            assert!(!f.utterance_start);
        }
        // Third active frame crosses the threshold.
        let third = vad.apply_hysteresis(0.9);
        assert!(third.utterance_start);
        assert!(!third.utterance_end);
        // Further active frames don't re-emit start.
        let fourth = vad.apply_hysteresis(0.9);
        assert!(!fourth.utterance_start);
    }

    #[test]
    fn hysteresis_emits_end_after_threshold_silence() {
        let mut vad = SileroVad::dummy(SileroVadConfig {
            utterance_start_frames: 1,
            utterance_end_frames: 2,
            ..SileroVadConfig::default()
        });
        // Enter utterance.
        let start = vad.apply_hysteresis(0.9);
        assert!(start.utterance_start);
        // First silent frame doesn't end yet.
        let s1 = vad.apply_hysteresis(0.1);
        assert!(!s1.utterance_end);
        // Second silent frame crosses end threshold.
        let s2 = vad.apply_hysteresis(0.1);
        assert!(s2.utterance_end);
    }

    #[test]
    fn hysteresis_only_emits_end_once_per_utterance() {
        let mut vad = SileroVad::dummy(SileroVadConfig {
            utterance_start_frames: 1,
            utterance_end_frames: 1,
            ..SileroVadConfig::default()
        });
        let _ = vad.apply_hysteresis(0.9); // start
        let end = vad.apply_hysteresis(0.1);
        assert!(end.utterance_end);
        let still_silent = vad.apply_hysteresis(0.1);
        assert!(!still_silent.utterance_end);
    }

    #[test]
    fn hysteresis_resets_counters_on_state_flip() {
        let mut vad = SileroVad::dummy(SileroVadConfig {
            utterance_start_frames: 3,
            utterance_end_frames: 3,
            ..SileroVadConfig::default()
        });
        // Two active frames — close to start, but not yet.
        let _ = vad.apply_hysteresis(0.9);
        let _ = vad.apply_hysteresis(0.9);
        // A silent frame in the middle should reset the active counter.
        let _ = vad.apply_hysteresis(0.1);
        // Two more active frames — still shouldn't be enough.
        let _ = vad.apply_hysteresis(0.9);
        let f = vad.apply_hysteresis(0.9);
        assert!(!f.utterance_start);
    }

    #[test]
    fn feed_rejects_wrong_frame_size() {
        let mut vad = SileroVad::dummy(SileroVadConfig::default());
        let too_short = vec![0.0_f32; 256];
        let err = vad
            .feed(&too_short)
            .expect_err("must reject 256-sample frame");
        assert!(matches!(err, SttError::InvalidOptions(_)));
    }

    // Live ONNX integration test. Synthesises 32 ms of silence + ~1 s
    // of a 440 Hz tone (loud enough that Silero registers it as
    // speech-like) + 32 ms of silence, then asserts that an
    // utterance-start fires somewhere inside the tone region.
    //
    // Skipped by default (`#[ignore]`) because it needs the Silero VAD
    // v4 ONNX weights. Unlock by setting `BLAZEN_SILERO_VAD_ONNX_PATH`
    // to a local `silero_vad.onnx` and dropping `--ignored` into the
    // nextest invocation:
    //
    // ```sh
    // BLAZEN_SILERO_VAD_ONNX_PATH=/path/to/silero_vad.onnx \
    //     cargo nextest run -p blazen-audio-stt --features whisper-streaming \
    //         live_vad_detects_synthetic_tone -- --ignored
    // ```
    #[test]
    #[ignore = "needs BLAZEN_SILERO_VAD_ONNX_PATH pointing at silero_vad.onnx"]
    fn live_vad_detects_synthetic_tone() {
        let path = std::env::var("BLAZEN_SILERO_VAD_ONNX_PATH")
            .expect("BLAZEN_SILERO_VAD_ONNX_PATH must be set for this test");
        let mut vad = SileroVad::from_path(std::path::Path::new(&path), SileroVadConfig::default())
            .expect("load silero-vad");

        // 32 ms of silence, then 1 s of a 440 Hz tone at 0.5 amplitude,
        // then 32 ms of silence. 16 kHz mono.
        let mut samples = Vec::with_capacity(VAD_FRAME_SIZE * 33);
        samples.extend(std::iter::repeat_n(0.0_f32, VAD_FRAME_SIZE));
        for i in 0..VAD_SAMPLE_RATE {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / VAD_SAMPLE_RATE as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        samples.extend(std::iter::repeat_n(0.0_f32, VAD_FRAME_SIZE));

        let mut saw_start = false;
        for frame in samples.chunks_exact(VAD_FRAME_SIZE) {
            let r = vad.feed(frame).expect("vad feed");
            if r.utterance_start {
                saw_start = true;
            }
        }
        assert!(
            saw_start,
            "expected utterance_start somewhere inside the synthetic tone window"
        );
    }
}
