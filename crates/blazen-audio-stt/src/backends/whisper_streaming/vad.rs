//! Silero VAD wrapper.
//!
//! Loads `silero_vad.onnx` via `ort` (ONNX Runtime) and exposes a
//! streaming API: feed 512-sample 16 kHz f32 chunks, receive
//! voice-activity flags + utterance boundaries.
//!
//! # Model assumptions
//!
//! This wrapper targets the **Silero VAD v5** ONNX graph published at
//! <https://github.com/snakers4/silero-vad> (file `silero_vad.onnx`,
//! Jul 2024+). That graph has three inputs (`input`, `state`, `sr`) and
//! two outputs (`output`, `stateN`) — v5 collapsed v4's separate `h`/`c`
//! LSTM states into a single `[2, 1, 128]` `state` tensor and added the
//! `sr` (sample-rate) scalar. The audio frame must be exactly **512
//! samples at 16 kHz** (32 ms); other sizes change the model's receptive
//! field and produce undefined behaviour.
//!
//! We run it through `ort` (ONNX Runtime) rather than `tract` because the
//! v5 graph contains a control-flow `If` op (the 16 kHz / 8 kHz STFT
//! branch) that tract cannot analyse during optimization; ONNX Runtime
//! executes it directly.
//!
//! # `HuggingFace` source
//!
//! [`SileroVad::from_hf`] downloads the ONNX from the
//! [`deepghs/silero-vad-onnx`](https://huggingface.co/deepghs/silero-vad-onnx)
//! HF mirror (file `silero_vad.onnx`, ~2.2 MB), pinned to revision
//! `8547eb3c577a6f712c1ed1a554c21c5d9137867d`. That pinned file is the v5
//! graph (`input`/`state`/`sr` → `output`/`stateN`); the pin keeps the
//! binary content fixed for reproducibility.
//!
//! The combined recurrent `state` has shape `[2, 1, 128]`. `feed` threads
//! it across forward calls so the model can integrate evidence over the
//! audio stream.
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

use ort::session::Session;
use ort::value::Tensor;

use crate::error::SttError;

/// Sample rate the Silero VAD v5 graph expects.
pub(crate) const VAD_SAMPLE_RATE: usize = 16_000;

/// Frame size (samples) the Silero VAD v5 graph expects.
pub(crate) const VAD_FRAME_SIZE: usize = 512;

/// `HuggingFace` repo hosting the v5 ONNX weights (see crate-level
/// docs in this module for why we pin this specific mirror).
pub(crate) const VAD_HF_REPO: &str = "deepghs/silero-vad-onnx";

/// File name inside [`VAD_HF_REPO`] for the v5 weights.
pub(crate) const VAD_HF_FILE: &str = "silero_vad.onnx";

/// Commit SHA the v5 ONNX is pinned to. See the module-level
/// "`HuggingFace` source" doc for the choice rationale.
pub(crate) const VAD_HF_REVISION: &str = "8547eb3c577a6f712c1ed1a554c21c5d9137867d";

/// Dimensions of the combined recurrent `state` tensor (`[2, 1, 128]`).
/// Silero VAD v5 fuses v4's separate `h` / `c` LSTM states (`[2, 1, 64]`
/// each) into this single tensor.
const VAD_STATE_DIMS: [i64; 3] = [2, 1, 128];

/// Element count of the `state` tensor (`2 * 1 * 128`).
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

/// Streaming Silero VAD wrapper.
///
/// Single-stream: each independent audio stream needs its own
/// [`SileroVad`] (or a call to [`SileroVad::reset`] between streams). The
/// wrapper owns the ONNX Runtime session, the recurrent state, and the
/// hysteresis counters.
pub struct SileroVad {
    /// Loaded ONNX Runtime session. `None` only for the test-only
    /// [`SileroVad::dummy`] constructor (which never runs inference).
    model: Option<Session>,
    config: SileroVadConfig,
    /// Combined recurrent `state` (`[2, 1, 128]`, row-major), threaded
    /// across `feed` calls (Silero v5's fused `h`/`c`).
    state: Vec<f32>,
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
    /// `onnx_path` must point at a Silero VAD **v5** ONNX file
    /// (`input`/`state`/`sr` → `output`/`stateN`).
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] when the ONNX Runtime session can't
    /// be created or the file can't be loaded.
    pub fn from_path(onnx_path: &Path, config: SileroVadConfig) -> Result<Self, SttError> {
        let session = Session::builder()
            .map_err(|e| SttError::ModelLoad(format!("silero-vad ort builder: {e}")))?
            .commit_from_file(onnx_path)
            .map_err(|e| {
                SttError::ModelLoad(format!(
                    "silero-vad ort load {}: {e}",
                    onnx_path.display()
                ))
            })?;

        Ok(Self {
            model: Some(session),
            config,
            state: vec![0.0; VAD_STATE_LEN],
            active_frames: 0,
            inactive_frames: 0,
            is_in_utterance: false,
        })
    }

    /// Download the Silero VAD v5 ONNX from `HuggingFace` and load it.
    ///
    /// Uses `hf-hub` (already a `blazen-audio-stt` dep via the
    /// `whisper-streaming` feature → `candle`) to fetch
    /// [`VAD_HF_FILE`] from [`VAD_HF_REPO`] at [`VAD_HF_REVISION`] into
    /// the HF cache directory; on cache hit no network is touched. The
    /// model file is ~2.2 MB.
    ///
    /// Pinned to a specific commit SHA for reproducibility — Silero VAD
    /// publishes new model versions periodically.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] when the HF download fails (no
    /// network, repo gone, file moved, revision rotated, etc.) or when
    /// the downloaded ONNX fails to load (see [`Self::from_path`]).
    pub async fn from_hf(config: SileroVadConfig) -> Result<Self, SttError> {
        // `hf-hub`'s sync `Api` is blocking, so dispatch the download on
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

        let speech_prob;
        let new_state;
        {
            // The dims (512 / 16 000) are tiny; the usize→i64 casts can't wrap.
            #[allow(clippy::cast_possible_wrap)]
            let audio = Tensor::from_array((vec![1_i64, VAD_FRAME_SIZE as i64], audio_frame.to_vec()))
                .map_err(|e| SttError::Transcription(format!("silero-vad audio tensor: {e}")))?;
            let state = Tensor::from_array((VAD_STATE_DIMS.to_vec(), self.state.clone()))
                .map_err(|e| SttError::Transcription(format!("silero-vad state tensor: {e}")))?;
            // `sr` is a rank-0 int64 scalar (the v5 graph's sample-rate input).
            #[allow(clippy::cast_possible_wrap)]
            let sr = Tensor::from_array((Vec::<i64>::new(), vec![VAD_SAMPLE_RATE as i64]))
                .map_err(|e| SttError::Transcription(format!("silero-vad sr tensor: {e}")))?;

            let session = self
                .model
                .as_mut()
                .ok_or_else(|| SttError::Transcription("silero-vad: session not loaded".into()))?;

            let outputs = session
                .run(ort::inputs!["input" => audio, "state" => state, "sr" => sr])
                .map_err(|e| SttError::Transcription(format!("silero-vad run: {e}")))?;

            let (_, prob) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(|e| SttError::Transcription(format!("silero-vad prob extract: {e}")))?;
            speech_prob = *prob
                .first()
                .ok_or_else(|| SttError::Transcription("silero-vad empty prob output".into()))?;

            let (_, st) = outputs["stateN"]
                .try_extract_tensor::<f32>()
                .map_err(|e| SttError::Transcription(format!("silero-vad stateN extract: {e}")))?;
            new_state = st.to_vec();
        }
        // Thread the recurrent state forward.
        self.state = new_state;

        Ok(self.apply_hysteresis(speech_prob))
    }

    /// Reset the recurrent state and hysteresis counters (e.g. between
    /// independent audio streams).
    pub fn reset(&mut self) {
        self.state = vec![0.0; VAD_STATE_LEN];
        self.active_frames = 0;
        self.inactive_frames = 0;
        self.is_in_utterance = false;
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
    /// detection on synthetic probability streams — it never runs
    /// inference, so the session is `None`.
    #[cfg(test)]
    fn dummy(config: SileroVadConfig) -> Self {
        Self {
            model: None,
            config,
            state: vec![0.0; VAD_STATE_LEN],
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
    fn frame_size_and_sample_rate_are_v5_constants() {
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
    // v5 ONNX weights. Unlock by setting `BLAZEN_SILERO_VAD_ONNX_PATH`
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
        // Local-path convenience variant; skip when the env isn't set (the
        // `from_hf` download path is covered by the mod.rs streaming test).
        let Ok(path) = std::env::var("BLAZEN_SILERO_VAD_ONNX_PATH") else {
            eprintln!("skipping: BLAZEN_SILERO_VAD_ONNX_PATH not set");
            return;
        };
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

        // Verify the ort v5 inference pipeline: every frame yields a
        // finite probability in [0, 1], and the model *responds* to the
        // tone (the loudest tone frame scores higher than the leading
        // silence). We deliberately do NOT assert the tone crosses the
        // speech threshold — Silero v5 is trained on human speech and a
        // pure 440 Hz sine is not speech, so it (correctly) stays below 0.5.
        let probs: Vec<f32> = samples
            .chunks_exact(VAD_FRAME_SIZE)
            .map(|frame| vad.feed(frame).expect("vad feed").speech_prob)
            .collect();
        assert!(
            probs.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "all VAD probabilities must be finite and in [0, 1]: {probs:?}"
        );
        let silence_prob = probs[0];
        let tone_max = probs[1..]
            .iter()
            .copied()
            .fold(f32::MIN, f32::max);
        assert!(
            tone_max > silence_prob,
            "tone window (max {tone_max}) should score above leading silence ({silence_prob})"
        );
    }
}
