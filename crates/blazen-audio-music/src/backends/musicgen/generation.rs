//! MusicGen autoregressive generation loop.
//!
//! Wires together [`super::model::MusicgenForConditionalGeneration`],
//! [`super::delay_pattern`], and [`super::sampler::cfg_combine`] into a
//! single `generate_tokens` entry point that returns the post-undelay
//! EnCodec code grid.
//!
//! The loop is the structural twin of
//! `candle_transformers::models::parler_tts::Model::generate` with two
//! additions:
//!
//! 1. Classifier-Free Guidance: the decoder is invoked twice per step
//!    (cond + uncond), the per-codebook logits are blended per
//!    [`super::sampler::cfg_combine`], then sampled.
//! 2. MusicGen-style delay pattern: codebook `cb` only begins sampling
//!    at step `cb`; before that it stays pinned to the BOS sentinel.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use super::delay_pattern;
use super::model::MusicgenForConditionalGeneration;
use super::sampler;

/// Per-call generation hyper-parameters.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Total decoder steps. The post-undelay frame count is
    /// `max_steps - (num_codebooks - 1)`. EnCodec runs at 50 Hz so
    /// `seconds * 50 + num_codebooks - 1` is the right value for a
    /// duration target.
    pub max_steps: usize,
    /// Sampling temperature (`None` → argmax).
    pub temperature: Option<f64>,
    /// Top-p nucleus threshold (`None` disables top-p).
    pub top_p: Option<f64>,
    /// Top-k limit (`None` disables top-k).
    pub top_k: Option<usize>,
    /// Classifier-Free Guidance scale (`3.0` is the canonical default).
    pub cfg_scale: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_steps: 0,
            temperature: Some(1.0),
            top_p: Some(0.9),
            top_k: Some(250),
            cfg_scale: 3.0,
            seed: 0,
        }
    }
}

impl GenerationParams {
    fn sampling(&self) -> Sampling {
        // Match candle's `LogitsProcessor::new` ordering: argmax > top-k > top-p > all.
        let temperature = self
            .temperature
            .and_then(|v| if v < 1e-7 { None } else { Some(v) });
        match temperature {
            None => Sampling::ArgMax,
            Some(t) => match (self.top_k, self.top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: t,
                },
                (Some(k), None) => Sampling::TopK { k, temperature: t },
                (None, Some(p)) => Sampling::TopP { p, temperature: t },
                (None, None) => Sampling::All { temperature: t },
            },
        }
    }
}

/// Result of [`generate_tokens`]: row-major `[num_codebooks, t]` plus
/// the effective number of EnCodec frames `t`.
#[derive(Debug, Clone)]
pub struct GeneratedTokens {
    /// Flattened tokens, length `num_codebooks * frames`.
    pub flat: Vec<u32>,
    /// Number of EnCodec frames after undelay.
    pub frames: usize,
    /// Codebooks (always 4 for MusicGen).
    pub num_codebooks: usize,
}

/// Drive the MusicGen decoder for `params.max_steps` autoregressive
/// steps and return the undelayed EnCodec codes.
///
/// `prompt_tokens` is `[1, prompt_len]` (T5 input ids on `device`).
/// `device` selects where the inputs / intermediates live.
///
/// # Errors
///
/// Propagates candle errors from the encoder, decoder, and EnCodec stack
/// (mostly tensor-shape, OOM, or device-dispatch failures).
pub fn generate_tokens(
    model: &mut MusicgenForConditionalGeneration,
    prompt_tokens: &Tensor,
    params: &GenerationParams,
    device: &Device,
) -> Result<GeneratedTokens> {
    let num_codebooks = model.decoder.num_codebooks();
    let bos =
        u32::try_from(model.config().musicgen.bos_token_id).map_err(candle_core::Error::wrap)?;

    // Reset T5 KV cache (the encoder caches relative position biases internally).
    model.text_encoder.clear_kv_cache();

    // Run the T5 encoder once on the real prompt and once on an "empty"
    // prompt (a single pad token). Cache both for the entire generation.
    let cond_encoded = model.text_encoder.forward(prompt_tokens)?;
    let cond_encoded = match model.enc_to_dec_proj.as_ref() {
        None => cond_encoded,
        Some(proj) => cond_encoded.apply(proj)?,
    };

    model.text_encoder.clear_kv_cache();
    // Empty prompt = single pad token (same convention audiocraft uses).
    let pad_id = model.cfg.t5.pad_token_id;
    let empty = Tensor::from_slice(
        &[u32::try_from(pad_id).map_err(candle_core::Error::wrap)?],
        (1, 1),
        device,
    )?;
    let uncond_encoded = model.text_encoder.forward(&empty)?;
    let uncond_encoded = match model.enc_to_dec_proj.as_ref() {
        None => uncond_encoded,
        Some(proj) => uncond_encoded.apply(proj)?,
    };

    let mut lp = LogitsProcessor::from_sampling(params.seed, params.sampling());

    // Token grid being filled in. Outer Vec indexed by codebook.
    let mut delayed: Vec<Vec<u32>> = delay_pattern::bos_prefix(num_codebooks, bos);

    // Each step: feed the *full* history so far (no KV cache yet --
    // matches the upstream `musicgen_model.rs` behaviour). Build an
    // `[1, num_codebooks, history_len + 1]` tensor where the last column
    // is whatever each codebook produced *previously*; the very first
    // step uses BOS for every codebook.
    for step in 0..params.max_steps {
        // Build the input matrix. For step 0 the matrix is `[1, cb, 1]`
        // of all-BOS. For step > 0 it is `[1, cb, step + 1]` containing
        // the full history (BOS column at index 0, then one column per
        // sampled step).
        let input = build_input_matrix(&delayed, num_codebooks, step, bos, device)?;
        let cond_logits = model.decoder.forward(&input, &cond_encoded, 0)?;
        let uncond_logits = model.decoder.forward(&input, &uncond_encoded, 0)?;
        // We only need the *last* timestep of each codebook's logits.
        let last_idx = input.dim(2)? - 1;
        let cond_last: Vec<Tensor> = cond_logits
            .iter()
            .map(|t| t.i((0, last_idx, ..)))
            .collect::<Result<_>>()?;
        let uncond_last: Vec<Tensor> = uncond_logits
            .iter()
            .map(|t| t.i((0, last_idx, ..)))
            .collect::<Result<_>>()?;

        let active: Vec<bool> = (0..num_codebooks)
            .map(|cb| delay_pattern::codebook_active(step, cb))
            .collect();
        let tokens = sampler::sample_codebooks(
            &cond_last,
            &uncond_last,
            &active,
            bos,
            &mut lp,
            params.cfg_scale,
        )?;
        for (cb, tok) in tokens.into_iter().enumerate() {
            // Inactive codebooks already have BOS in their prefix; only
            // push for active ones to keep the row lengths aligned with
            // the delay pattern.
            if active[cb] {
                delayed[cb].push(tok);
            } else {
                // BOS prefix already populated by `bos_prefix`; the next
                // active step will push the first real token.
            }
        }
    }

    let (flat, frames) = delay_pattern::undelay(&delayed)
        .ok_or_else(|| candle_core::Error::Msg("delay-pattern undelay failed".into()))?;
    Ok(GeneratedTokens {
        flat,
        frames,
        num_codebooks,
    })
}

/// Build the `[1, num_codebooks, history_len]` input id matrix for step
/// `step`.
///
/// At step 0 the matrix is `[1, cb, 1]` of all-BOS. At step `s > 0`
/// the matrix contains the BOS column followed by `s` sampled columns;
/// inactive codebooks contribute BOS for any column in their delay
/// region.
fn build_input_matrix(
    delayed: &[Vec<u32>],
    num_codebooks: usize,
    step: usize,
    bos: u32,
    device: &Device,
) -> Result<Tensor> {
    let history_len = step + 1; // +1 for the BOS column
    let mut data = Vec::with_capacity(num_codebooks * history_len);
    for (cb, row) in delayed.iter().enumerate().take(num_codebooks) {
        // Column 0 is BOS.
        data.push(bos);
        // Columns 1..history_len are the previously sampled tokens
        // (or BOS for delay-pattern dead zones).
        for prev_step in 0..step {
            let token = if delay_pattern::codebook_active(prev_step, cb) {
                // `row` = [bos prefix..., t0, t1, ...]. The first real
                // token for codebook `cb` was sampled at step `cb`, so
                // the index into the *real* portion is `prev_step - cb`.
                let global_idx = cb + (prev_step - cb);
                row.get(global_idx).copied().unwrap_or(bos)
            } else {
                bos
            };
            data.push(token);
        }
    }
    let t =
        Tensor::from_vec(data, (1, num_codebooks, history_len), device)?.to_dtype(DType::U32)?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_picks_argmax_when_temperature_zero() {
        let p = GenerationParams {
            max_steps: 10,
            temperature: None,
            top_p: None,
            top_k: None,
            cfg_scale: 3.0,
            seed: 0,
        };
        match p.sampling() {
            Sampling::ArgMax => {}
            other => panic!("expected ArgMax got {other:?}"),
        }
    }

    #[test]
    fn sampling_picks_topk_then_topp_when_both_set() {
        let p = GenerationParams {
            max_steps: 10,
            temperature: Some(1.0),
            top_p: Some(0.9),
            top_k: Some(50),
            cfg_scale: 3.0,
            seed: 0,
        };
        match p.sampling() {
            Sampling::TopKThenTopP { k, p, .. } => {
                assert_eq!(k, 50);
                assert!((p - 0.9).abs() < 1e-9);
            }
            other => panic!("expected TopKThenTopP got {other:?}"),
        }
    }
}
