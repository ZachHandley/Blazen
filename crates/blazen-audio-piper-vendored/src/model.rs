//! ONNX Runtime-backed inference for Piper voice models.
//!
//! Matches upstream `piper-rs::model` (which also runs on `ort`). The graph
//! IO contract: VITS-family Piper voices take
//! `[input_ids, input_lengths, scales[, sid]]` and emit a single
//! `[batch, 1, samples]` float audio tensor.
//!
//! The engine is `ort` and not the workspace's `tract` because tract cannot
//! statically analyse Piper VITS graphs at all — its ONNX `Pad` rule rejects
//! the 2-input form Piper exports (tract 0.22), and its symbolic-dim algebra
//! cannot prove the attention layers' Reshape volume equality
//! `2b(2p²+p−1) == 2b(p+1)(2p−1)` (tract 0.23). Both fail during
//! `into_typed()` analysis, so no tract chain (optimized or decluttered) can
//! even load these graphs.

use std::collections::HashMap;

use serde::Deserialize;

/// Beginning-of-sequence phoneme marker (per upstream Piper convention).
pub const BOS: char = '^';
/// End-of-sequence phoneme marker.
pub const EOS: char = '$';
/// Padding phoneme marker (inserted between every emitted id, also upstream).
pub const PAD: char = '_';

/// Audio block of a Piper voice config.
#[derive(Debug, Deserialize)]
pub struct AudioConfig {
    /// Sample rate the voice emits in Hz.
    pub sample_rate: u32,
}

/// eSpeak block of a Piper voice config — names the phonemizer voice
/// (e.g. `"en-us"`, `"ar"`) used to translate text to IPA.
#[derive(Debug, Deserialize)]
pub struct ESpeakConfig {
    /// Voice / language code passed to `espeak-ng --voice=...`.
    pub voice: String,
}

/// Inference block — VITS-family stochasticity / pacing controls.
#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    /// Stochasticity scale applied during sampling.
    pub noise_scale: f32,
    /// Pacing scale — `>1.0` slower, `<1.0` faster.
    pub length_scale: f32,
    /// Stochasticity scale applied to the duration predictor.
    pub noise_w: f32,
}

/// Top-level Piper voice config (`<voice>.onnx.json`).
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Audio config block.
    pub audio: AudioConfig,
    /// eSpeak block (phonemizer language).
    pub espeak: ESpeakConfig,
    /// Inference defaults.
    pub inference: InferenceConfig,
    /// Number of distinct speakers baked into the voice (1 = single-speaker).
    pub num_speakers: u32,
    /// Optional name→id map for multi-speaker voices.
    #[serde(default)]
    pub speaker_id_map: HashMap<String, i64>,
    /// Required IPA phoneme → token id map for the embedded text encoder.
    pub phoneme_id_map: HashMap<char, Vec<i64>>,
}

/// Translate an IPA phoneme string to the input-id sequence Piper voices expect.
///
/// Convention from upstream Piper: prepend BOS, then for each known
/// phoneme emit `id, pad_id`, then append EOS. Unknown chars are skipped.
pub fn phonemes_to_ids(config: &ModelConfig, phonemes: &str) -> Vec<i64> {
    let map = &config.phoneme_id_map;
    let pad_id = *map.get(&PAD).and_then(|v| v.first()).unwrap_or(&0);
    let bos_id = *map.get(&BOS).and_then(|v| v.first()).unwrap_or(&0);
    let eos_id = *map.get(&EOS).and_then(|v| v.first()).unwrap_or(&0);

    let mut ids = Vec::with_capacity((phonemes.len() + 1) * 2);
    ids.push(bos_id);
    for ch in phonemes.chars() {
        if let Some(id) = map.get(&ch).and_then(|v| v.first()) {
            ids.push(*id);
            ids.push(pad_id);
        }
    }
    ids.push(eos_id);
    ids
}

// ---------------------------------------------------------------------------
// Inference engine — ONNX Runtime everywhere it ships a prebuilt, an
// `Unsupported` stub on `x86_64-apple-darwin` (see Cargo.toml target-gate).
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
mod engine_impl {
    use std::path::Path;
    use std::sync::Mutex;

    use ort::session::Session;
    use ort::value::Tensor;

    use super::{ModelConfig, phonemes_to_ids};
    use crate::{PiperError, PiperResult};

    /// ONNX Runtime Piper engine. `Session::run` takes `&mut self`, so the
    /// session sits behind a `Mutex` to keep [`crate::Piper::create`]
    /// callable through `&self`.
    pub struct Engine {
        session: Mutex<Session>,
    }

    impl Engine {
        pub(crate) fn load(model_path: &Path) -> PiperResult<Self> {
            let session = Session::builder()
                .map_err(|e| PiperError::InferenceError(format!("ort session builder: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| {
                    PiperError::FailedToLoadResource(format!(
                        "ort onnx load failed for `{}`: {e}",
                        model_path.display()
                    ))
                })?;
            Ok(Self {
                session: Mutex::new(session),
            })
        }

        pub(crate) fn from_session(session: Session) -> Self {
            Self {
                session: Mutex::new(session),
            }
        }

        /// Run one forward pass of a Piper VITS graph and return raw f32
        /// PCM samples (first output tensor, flattened).
        pub(crate) fn infer(
            &self,
            config: &ModelConfig,
            phonemes: &str,
            noise_scale: f32,
            length_scale: f32,
            noise_w: f32,
            speaker_id: i64,
        ) -> PiperResult<Vec<f32>> {
            let ids = phonemes_to_ids(config, phonemes);
            let input_len = i64::try_from(ids.len())
                .map_err(|e| PiperError::InferenceError(format!("input_len overflow: {e}")))?;

            let input_t = Tensor::from_array((vec![1_i64, input_len], ids))
                .map_err(|e| PiperError::InferenceError(format!("input tensor: {e}")))?;
            let lengths_t = Tensor::from_array((vec![1_i64], vec![input_len]))
                .map_err(|e| PiperError::InferenceError(format!("lengths tensor: {e}")))?;
            let scales_t =
                Tensor::from_array((vec![3_i64], vec![noise_scale, length_scale, noise_w]))
                    .map_err(|e| PiperError::InferenceError(format!("scales tensor: {e}")))?;

            let mut session = self
                .session
                .lock()
                .map_err(|e| PiperError::InferenceError(format!("session mutex poisoned: {e}")))?;

            // Positional inputs, same order as upstream piper-rs:
            // [input, input_lengths, scales[, sid]].
            let outputs = if config.num_speakers > 1 {
                let sid_t = Tensor::from_array((vec![1_i64], vec![speaker_id]))
                    .map_err(|e| PiperError::InferenceError(format!("sid tensor: {e}")))?;
                session.run(ort::inputs![input_t, lengths_t, scales_t, sid_t])
            } else {
                session.run(ort::inputs![input_t, lengths_t, scales_t])
            }
            .map_err(|e| PiperError::InferenceError(format!("ort run failed: {e}")))?;

            let (_, audio) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| PiperError::InferenceError(format!("output extract failed: {e}")))?;
            Ok(audio.to_vec())
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
mod engine_impl {
    use std::path::Path;

    use super::ModelConfig;
    use crate::{PiperError, PiperResult};

    const UNSUPPORTED: &str = "piper TTS has no inference engine on x86_64-apple-darwin: \
         ONNX Runtime ships no Intel-mac prebuilt (ort-sys 2.0.0-rc.12) and tract \
         cannot statically analyse Piper VITS graphs";

    /// Stub engine for `x86_64-apple-darwin` — construction always fails,
    /// so [`crate::Piper`] can never be built on this triple.
    pub struct Engine;

    impl Engine {
        pub(crate) fn load(_model_path: &Path) -> PiperResult<Self> {
            Err(PiperError::Unsupported(UNSUPPORTED.into()))
        }

        pub(crate) fn infer(
            &self,
            _config: &ModelConfig,
            _phonemes: &str,
            _noise_scale: f32,
            _length_scale: f32,
            _noise_w: f32,
            _speaker_id: i64,
        ) -> PiperResult<Vec<f32>> {
            Err(PiperError::Unsupported(UNSUPPORTED.into()))
        }
    }
}

pub(crate) use engine_impl::Engine;
