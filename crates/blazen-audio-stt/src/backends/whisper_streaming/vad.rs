//! Silero VAD wrapper.
//!
//! Runs an embedded, control-flow-free **Silero VAD v5 (16 kHz)** ONNX and
//! exposes a streaming API: feed 512-sample 16 kHz f32 chunks, receive
//! voice-activity flags + utterance boundaries.
//!
//! # Dual runtime backend
//!
//! The same embedded model (`assets/silero_vad_16k.onnx`) is executed through
//! one of two runtimes, chosen by Cargo feature:
//!
//! - **`vad-ort`** — ONNX Runtime (`ort`), for native targets (macOS,
//!   Linux-gnu, Windows; CPU + GPU).
//! - **`vad-tract`** — pure-Rust `tract`, for `musl` / `wasm` targets where
//!   ONNX Runtime doesn't ship.
//!
//! These are **mutually-exclusive** engine selectors — a real artifact enables
//! exactly one (native → `vad-ort`; musl/wasm → `vad-tract`). The only config
//! that turns on both is a CI convenience like `--all-features`. There,
//! **`vad-tract` takes precedence**: it is pure-Rust and links no native
//! runtime, so it cannot collide at link/run time with the other native ML
//! libraries that `--all-features` also pulls in (ONNX Runtime via `ort`,
//! `CTranslate2` via `ct2rs`, whisper.cpp) — a combination that otherwise
//! SIGSEGVs when an `ort` session is created in the same process. A
//! `compile_error!` fires if neither backend is enabled under
//! `whisper-streaming`.
//!
//! # Why an embedded, re-exported model
//!
//! Every stock Silero ONNX export (v4 and v5) contains nested ONNX `If` ops
//! (the `sr == 16000` STFT branch plus decoder shape-guards) that `tract`
//! cannot statically analyse — see snakers4/silero-vad#728. We ship a
//! re-exported, fixed-16 kHz, control-flow-free graph instead
//! (`scripts/export_silero_vad_16k.py`); both runtimes load it identically.
//! Embedding it (`include_bytes!`) also means the VAD needs no network and
//! works fully offline, including on `musl`.
//!
//! # Input contract
//!
//! The model takes `input` `[1, 576]` (a 512-sample frame prefixed with the
//! 64-sample tail of the previous frame — the STFT lookback "context") and a
//! recurrent `state` `[2, 1, 128]`, and returns `output` `[1, 1]` (speech
//! probability) and `stateN` `[2, 1, 128]`. [`SileroVad::feed`] owns the
//! 64-sample context bookkeeping and threads `state` across calls.
//!
//! # Utterance-boundary hysteresis
//!
//! `feed` applies a two-counter hysteresis on top of the raw speech
//! probability:
//!
//! - `active_frames` counts consecutive frames whose probability exceeds
//!   [`SileroVadConfig::speech_threshold`]; crossing
//!   [`SileroVadConfig::utterance_start_frames`] emits
//!   [`VadFrame::utterance_start`] once.
//! - `inactive_frames` counts the converse; crossing
//!   [`SileroVadConfig::utterance_end_frames`] emits
//!   [`VadFrame::utterance_end`].
//!
//! This matches the upstream Python reference's `VADIterator` semantics at
//! default settings.

#![cfg(feature = "whisper-streaming")]

#[cfg(not(any(feature = "vad-ort", feature = "vad-tract")))]
compile_error!(
    "the `whisper-streaming` feature needs a VAD runtime backend: enable \
     `vad-ort` (ONNX Runtime; native) or `vad-tract` (pure-Rust tract; musl/wasm)."
);

use crate::error::SttError;

/// The embedded control-flow-free Silero VAD v5 (16 kHz) ONNX. Regenerate
/// with `scripts/export_silero_vad_16k.py`.
const MODEL_BYTES: &[u8] = include_bytes!("../../../assets/silero_vad_16k.onnx");

/// Sample rate the model expects.
pub(crate) const VAD_SAMPLE_RATE: usize = 16_000;

/// Frame size (samples) the caller feeds per [`SileroVad::feed`] call.
pub(crate) const VAD_FRAME_SIZE: usize = 512;

/// STFT lookback context (samples) prefixed to each frame to form the
/// model's `[1, 576]` input.
const VAD_CONTEXT: usize = 64;

/// Model input length (`VAD_CONTEXT + VAD_FRAME_SIZE`).
const VAD_INPUT_LEN: usize = VAD_CONTEXT + VAD_FRAME_SIZE;

/// Element count of the recurrent `state` tensor (`2 * 1 * 128`).
const VAD_STATE_LEN: usize = 256;

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

// ---------------------------------------------------------------------------
// Runtime backends — same embedded model, different engine. Exactly one is
// compiled (`ort` wins when both features are on).
// ---------------------------------------------------------------------------

// `vad-tract` wins when both backends are enabled (see module docs): the
// pure-Rust engine avoids the native-library coexistence SIGSEGV under
// `--all-features`. A real single-backend artifact is unaffected.
//
// `x86_64-apple-darwin` is also excluded here even under `vad-ort`: `ort`
// (ONNX Runtime) has no Intel-mac prebuilt as of rc.12 and is target-gated
// out of the crate (see Cargo.toml), so that triple uses the `tract` engine
// below instead.
#[cfg(all(
    feature = "vad-ort",
    not(feature = "vad-tract"),
    not(all(target_arch = "x86_64", target_os = "macos"))
))]
mod engine {
    use super::{MODEL_BYTES, SttError, VAD_INPUT_LEN};
    use ort::session::Session;
    use ort::value::Tensor;

    /// ONNX Runtime VAD engine.
    pub(super) struct Engine {
        session: Session,
    }

    impl Engine {
        pub(super) fn load() -> Result<Self, SttError> {
            let session = Session::builder()
                .map_err(|e| SttError::ModelLoad(format!("silero-vad ort builder: {e}")))?
                .commit_from_memory(MODEL_BYTES)
                .map_err(|e| SttError::ModelLoad(format!("silero-vad ort load: {e}")))?;
            Ok(Self { session })
        }

        /// Run one frame: `input` is `[VAD_INPUT_LEN]`, `state` is the flat
        /// `[2*1*128]` recurrent state. Returns `(speech_prob, new_state)`.
        pub(super) fn run(
            &mut self,
            input: &[f32],
            state: &[f32],
        ) -> Result<(f32, Vec<f32>), SttError> {
            #[allow(clippy::cast_possible_wrap)]
            let inp = Tensor::from_array((vec![1_i64, VAD_INPUT_LEN as i64], input.to_vec()))
                .map_err(|e| SttError::Transcription(format!("silero-vad input tensor: {e}")))?;
            let st = Tensor::from_array((vec![2_i64, 1, 128], state.to_vec()))
                .map_err(|e| SttError::Transcription(format!("silero-vad state tensor: {e}")))?;

            let outputs = self
                .session
                .run(ort::inputs!["input" => inp, "state" => st])
                .map_err(|e| SttError::Transcription(format!("silero-vad ort run: {e}")))?;

            let prob = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(|e| SttError::Transcription(format!("silero-vad prob extract: {e}")))?
                .1
                .first()
                .copied()
                .ok_or_else(|| SttError::Transcription("silero-vad empty prob output".into()))?;
            let new_state = outputs["stateN"]
                .try_extract_tensor::<f32>()
                .map_err(|e| SttError::Transcription(format!("silero-vad stateN extract: {e}")))?
                .1
                .to_vec();
            Ok((prob, new_state))
        }
    }
}

// Pure-Rust `tract` engine: selected by the explicit `vad-tract` feature
// (musl / wasm), and ALSO on `x86_64-apple-darwin` under `vad-ort` (where ORT
// has no prebuilt — see the Cargo.toml target-gate + the ort arm above).
#[cfg(any(
    feature = "vad-tract",
    all(feature = "vad-ort", target_arch = "x86_64", target_os = "macos")
))]
mod engine {
    use super::{MODEL_BYTES, SttError, VAD_INPUT_LEN};
    use tract_onnx::prelude::*;

    type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

    /// Pure-Rust `tract` VAD engine. Positional IO (tract runs by input
    /// order), so we record the model's input/output slot indices by name.
    pub(super) struct Engine {
        model: Plan,
        in_input: usize,
        in_state: usize,
        out_prob: usize,
        out_state: usize,
    }

    impl Engine {
        pub(super) fn load() -> Result<Self, SttError> {
            let mut inference = tract_onnx::onnx()
                .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))
                .map_err(|e| SttError::ModelLoad(format!("silero-vad tract load: {e}")))?;

            let in_input = find_input(&inference, "input")?;
            let in_state = find_input(&inference, "state")?;
            let out_prob = find_output(&inference, "output")?;
            let out_state = find_output(&inference, "stateN")?;

            inference
                .set_input_fact(
                    in_input,
                    InferenceFact::dt_shape(f32::datum_type(), [1usize, VAD_INPUT_LEN]),
                )
                .map_err(|e| SttError::ModelLoad(format!("silero-vad input fact: {e}")))?;
            inference
                .set_input_fact(
                    in_state,
                    InferenceFact::dt_shape(f32::datum_type(), [2usize, 1, 128]),
                )
                .map_err(|e| SttError::ModelLoad(format!("silero-vad state fact: {e}")))?;

            let model = inference
                .into_optimized()
                .map_err(|e| SttError::ModelLoad(format!("silero-vad optimize: {e}")))?
                .into_runnable()
                .map_err(|e| SttError::ModelLoad(format!("silero-vad runnable: {e}")))?;

            Ok(Self {
                model,
                in_input,
                in_state,
                out_prob,
                out_state,
            })
        }

        pub(super) fn run(
            &mut self,
            input: &[f32],
            state: &[f32],
        ) -> Result<(f32, Vec<f32>), SttError> {
            let input_t = tract_ndarray::Array2::from_shape_vec((1, VAD_INPUT_LEN), input.to_vec())
                .map_err(|e| SttError::Transcription(format!("silero-vad input reshape: {e}")))?
                .into_tensor();
            let state_t = tract_ndarray::Array3::from_shape_vec((2, 1, 128), state.to_vec())
                .map_err(|e| SttError::Transcription(format!("silero-vad state reshape: {e}")))?
                .into_tensor();

            let mut ordered: Vec<Option<TValue>> = vec![None; 2];
            ordered[self.in_input] = Some(TValue::from_const(input_t.into()));
            ordered[self.in_state] = Some(TValue::from_const(state_t.into()));
            let inputs: TVec<TValue> = ordered
                .into_iter()
                .map(|v| {
                    v.ok_or_else(|| SttError::Transcription("silero-vad missing input slot".into()))
                })
                .collect::<Result<_, _>>()?;

            let outputs = self
                .model
                .run(inputs)
                .map_err(|e| SttError::Transcription(format!("silero-vad tract run: {e}")))?;

            let prob = outputs[self.out_prob]
                .as_slice::<f32>()
                .map_err(|e| SttError::Transcription(format!("silero-vad prob slice: {e}")))?
                .first()
                .copied()
                .ok_or_else(|| SttError::Transcription("silero-vad empty prob output".into()))?;
            let new_state = outputs[self.out_state]
                .as_slice::<f32>()
                .map_err(|e| SttError::Transcription(format!("silero-vad stateN slice: {e}")))?
                .to_vec();
            Ok((prob, new_state))
        }
    }

    fn find_input(model: &InferenceModel, name: &str) -> Result<usize, SttError> {
        for (ix, outlet) in model.inputs.iter().enumerate() {
            if model.nodes()[outlet.node].name == name {
                return Ok(ix);
            }
        }
        Err(SttError::ModelLoad(format!(
            "silero-vad: input `{name}` not found in embedded model"
        )))
    }

    fn find_output(model: &InferenceModel, name: &str) -> Result<usize, SttError> {
        for (ix, outlet) in model.outputs.iter().enumerate() {
            if model.outlet_label(*outlet) == Some(name) || model.nodes()[outlet.node].name == name
            {
                return Ok(ix);
            }
        }
        Err(SttError::ModelLoad(format!(
            "silero-vad: output `{name}` not found in embedded model"
        )))
    }
}

// ---------------------------------------------------------------------------
// Wrapper
// ---------------------------------------------------------------------------

/// Streaming Silero VAD wrapper.
///
/// Single-stream: each independent audio stream needs its own [`SileroVad`]
/// (or a call to [`SileroVad::reset`] between streams). Owns the runtime
/// engine, the recurrent state, the 64-sample STFT context, and the
/// hysteresis counters.
pub struct SileroVad {
    /// Runtime engine over the embedded model. `None` only for the
    /// test-only [`SileroVad::dummy`] constructor (which never runs
    /// inference).
    engine: Option<engine::Engine>,
    config: SileroVadConfig,
    /// Recurrent `state` (`[2, 1, 128]`, flat) threaded across `feed` calls.
    state: Vec<f32>,
    /// 64-sample STFT lookback (tail of the previous frame), prefixed to the
    /// next frame to form the `[1, 576]` model input.
    context: Vec<f32>,
    /// Consecutive frames at or above `speech_threshold`.
    active_frames: usize,
    /// Consecutive frames below `speech_threshold`.
    inactive_frames: usize,
    /// Whether the wrapper currently considers itself inside an utterance.
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
    /// Construct from the embedded model. No I/O or network — the model is
    /// compiled into the binary.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] if the runtime engine fails to build
    /// the embedded ONNX (effectively never on a healthy build).
    pub fn new(config: SileroVadConfig) -> Result<Self, SttError> {
        let engine = engine::Engine::load()?;
        Ok(Self {
            engine: Some(engine),
            config,
            state: vec![0.0; VAD_STATE_LEN],
            context: vec![0.0; VAD_CONTEXT],
            active_frames: 0,
            inactive_frames: 0,
            is_in_utterance: false,
        })
    }

    /// Feed a 512-sample 16 kHz f32 frame; return a [`VadFrame`].
    ///
    /// # Errors
    ///
    /// Returns [`SttError::InvalidOptions`] when `audio_frame.len() != 512`,
    /// or [`SttError::Transcription`] when ONNX inference fails.
    pub fn feed(&mut self, audio_frame: &[f32]) -> Result<VadFrame, SttError> {
        if audio_frame.len() != VAD_FRAME_SIZE {
            return Err(SttError::InvalidOptions(format!(
                "silero-vad expects {VAD_FRAME_SIZE}-sample 16 kHz frames; got {}",
                audio_frame.len()
            )));
        }

        // Prefix the 64-sample lookback context to form the [1, 576] input.
        let mut input = Vec::with_capacity(VAD_INPUT_LEN);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(audio_frame);

        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| SttError::Transcription("silero-vad: engine not loaded".into()))?;
        let (speech_prob, new_state) = engine.run(&input, &self.state)?;
        self.state = new_state;

        // Next frame's context is this frame's trailing 64 samples.
        self.context
            .copy_from_slice(&audio_frame[VAD_FRAME_SIZE - VAD_CONTEXT..]);

        Ok(self.apply_hysteresis(speech_prob))
    }

    /// Reset the recurrent state, STFT context, and hysteresis counters
    /// (e.g. between independent audio streams).
    pub fn reset(&mut self) {
        self.state = vec![0.0; VAD_STATE_LEN];
        self.context = vec![0.0; VAD_CONTEXT];
        self.active_frames = 0;
        self.inactive_frames = 0;
        self.is_in_utterance = false;
    }

    /// Number of samples the caller feeds per frame (always 512).
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
    /// loading the model. The engine is `None`; never call `feed` on it.
    #[cfg(test)]
    fn dummy(config: SileroVadConfig) -> Self {
        Self {
            engine: None,
            config,
            state: vec![0.0; VAD_STATE_LEN],
            context: vec![0.0; VAD_CONTEXT],
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
    fn frame_size_and_sample_rate_are_constants() {
        assert_eq!(SileroVad::frame_size(), 512);
        assert_eq!(SileroVad::sample_rate(), 16_000);
    }

    #[test]
    fn hysteresis_emits_start_after_threshold_frames() {
        let mut vad = SileroVad::dummy(SileroVadConfig {
            utterance_start_frames: 3,
            ..SileroVadConfig::default()
        });
        for _ in 0..2 {
            let f = vad.apply_hysteresis(0.9);
            assert!(!f.utterance_start);
        }
        let third = vad.apply_hysteresis(0.9);
        assert!(third.utterance_start);
        assert!(!third.utterance_end);
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
        let start = vad.apply_hysteresis(0.9);
        assert!(start.utterance_start);
        let s1 = vad.apply_hysteresis(0.1);
        assert!(!s1.utterance_end);
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
        let _ = vad.apply_hysteresis(0.9);
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
        let _ = vad.apply_hysteresis(0.9);
        let _ = vad.apply_hysteresis(0.9);
        let _ = vad.apply_hysteresis(0.1);
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

    /// End-to-end: build the active runtime engine from the embedded model,
    /// feed 32 ms of silence + ~1 s of a 440 Hz tone + 32 ms of silence, and
    /// verify the engine runs (finite probs in [0, 1]) and responds to the
    /// tone (loudest tone frame scores above the leading silence). Hermetic:
    /// the model is embedded, so this runs on both `vad-ort` and `vad-tract`
    /// with no network. We do NOT assert the tone crosses the speech
    /// threshold — Silero v5 is trained on human speech and a pure sine is
    /// not speech, so it (correctly) stays below 0.5.
    #[test]
    fn embedded_model_runs_and_discriminates_tone() {
        let mut vad = SileroVad::new(SileroVadConfig::default()).expect("load embedded silero-vad");

        let mut samples = Vec::with_capacity(VAD_FRAME_SIZE * 33);
        samples.extend(std::iter::repeat_n(0.0_f32, VAD_FRAME_SIZE));
        for i in 0..VAD_SAMPLE_RATE {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / VAD_SAMPLE_RATE as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        samples.extend(std::iter::repeat_n(0.0_f32, VAD_FRAME_SIZE));

        let probs: Vec<f32> = samples
            .chunks_exact(VAD_FRAME_SIZE)
            .map(|frame| vad.feed(frame).expect("vad feed").speech_prob)
            .collect();

        assert!(
            probs
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "all VAD probabilities must be finite and in [0, 1]: {probs:?}"
        );
        let silence_prob = probs[0];
        let tone_max = probs[1..].iter().copied().fold(f32::MIN, f32::max);
        assert!(
            tone_max > silence_prob,
            "tone window (max {tone_max}) should score above leading silence ({silence_prob})"
        );
    }
}
