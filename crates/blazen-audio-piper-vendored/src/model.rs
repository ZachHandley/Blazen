//! Tract-backed inference for Piper voice models.
//!
//! Patched from upstream `piper-rs::model` to swap `ort` for `tract-onnx`.
//! The graph IO contract is unchanged: VITS-family Piper voices take
//! `[input_ids, input_lengths, scales[, sid]]` and emit a single
//! `[batch, 1, samples]` float audio tensor.

use std::collections::HashMap;

use serde::Deserialize;
use tract_onnx::prelude::{
    IntoTValue, IntoTensor, SimplePlan, Tensor, TypedFact, TypedModel, TypedOp, tract_ndarray, tvec,
};

use crate::{PiperError, PiperResult};

/// Beginning-of-sequence phoneme marker (per upstream Piper convention).
pub const BOS: char = '^';
/// End-of-sequence phoneme marker.
pub const EOS: char = '$';
/// Padding phoneme marker (inserted between every emitted id, also upstream).
pub const PAD: char = '_';

/// Type alias for a fully-built, runnable tract plan over a Piper graph.
///
/// Exposed so callers that want to load the ONNX graph through their own
/// IO can hand the result to [`crate::Piper::from_model`].
pub type TractPiperModel = SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>;

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

/// Run one forward pass of a Piper VITS graph and return raw f32 PCM samples.
///
/// `model` is a tract plan already built from the voice's `.onnx` file.
/// Returns the contents of the first output tensor (audio) flattened to `Vec<f32>`.
///
/// # Errors
///
/// [`PiperError::InferenceError`] on any tract failure (shape, dtype, run).
#[allow(clippy::too_many_arguments)]
pub fn infer(
    model: &mut TractPiperModel,
    config: &ModelConfig,
    phonemes: &str,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids = phonemes_to_ids(config, phonemes);
    let input_len = ids.len();

    // Build tensors. tract uses ndarray shapes — `[1, input_len]` for
    // ids, `[1]` for lengths, `[3]` for the scales triple, `[1]` for sid.
    let ids_tensor = i64_tensor_2d(1, input_len, ids)?;
    let lengths_tensor =
        i64_tensor_1d(vec![i64::try_from(input_len).map_err(|e| {
            PiperError::InferenceError(format!("input_len overflow: {e}"))
        })?])?;
    let scales_tensor = f32_tensor_1d(vec![noise_scale, length_scale, noise_w])?;

    let inputs = if config.num_speakers > 1 {
        let sid_tensor = i64_tensor_1d(vec![speaker_id])?;
        tvec![
            ids_tensor.into_tvalue(),
            lengths_tensor.into_tvalue(),
            scales_tensor.into_tvalue(),
            sid_tensor.into_tvalue(),
        ]
    } else {
        tvec![
            ids_tensor.into_tvalue(),
            lengths_tensor.into_tvalue(),
            scales_tensor.into_tvalue(),
        ]
    };

    let outputs = model
        .run(inputs)
        .map_err(|e| PiperError::InferenceError(format!("tract run failed: {e}")))?;

    let audio = outputs
        .first()
        .ok_or_else(|| PiperError::InferenceError("no outputs from graph".into()))?;
    let view = audio
        .to_array_view::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("output view failed: {e}")))?;

    Ok(view.iter().copied().collect())
}

fn i64_tensor_2d(rows: usize, cols: usize, data: Vec<i64>) -> PiperResult<Tensor> {
    let arr = tract_ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PiperError::InferenceError(format!("i64 reshape ({rows}x{cols}): {e}")))?;
    Ok(arr.into_tensor())
}

fn i64_tensor_1d(data: Vec<i64>) -> PiperResult<Tensor> {
    let arr = tract_ndarray::Array1::from(data);
    Ok(arr.into_tensor())
}

fn f32_tensor_1d(data: Vec<f32>) -> PiperResult<Tensor> {
    let arr = tract_ndarray::Array1::from(data);
    Ok(arr.into_tensor())
}
