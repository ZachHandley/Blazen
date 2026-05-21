//! Classifier-Free-Guidance (CFG) logits combiner + per-codebook sampler.
//!
//! MusicGen is trained with conditioning dropout, which means the decoder
//! can be invoked twice per step: once with the real prompt and once with
//! the empty / null prompt. CFG then combines the two logit vectors as:
//!
//! ```text
//! logits_cfg = uncond + cfg_scale * (cond - uncond)
//! ```
//!
//! (This matches the audiocraft + HuggingFace transformers convention
//! where `cfg_scale` of `1.0` recovers pure conditional sampling and
//! `3.0` is the canonical default.)
//!
//! The combined logits are then fed through
//! [`candle_transformers::generation::LogitsProcessor`] to draw the next
//! codebook index.

use candle_core::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

/// Combine conditional + unconditional logits per the CFG formula.
///
/// Both inputs must broadcast to the same shape (typically
/// `[vocab_size]`). Returns a fresh tensor on the same device.
///
/// # Errors
///
/// Propagates the first candle error raised by the subtract / scale /
/// add chain (shape mismatch or device-dispatch failures).
pub fn cfg_combine(cond: &Tensor, uncond: &Tensor, cfg_scale: f64) -> Result<Tensor> {
    // uncond + cfg_scale * (cond - uncond)
    let delta = (cond - uncond)?;
    let scaled = (delta * cfg_scale)?;
    uncond + scaled
}

/// Sample one token per active codebook.
///
/// `cond_logits` and `uncond_logits` have len `num_codebooks`; each entry
/// is `[vocab_size]`. `active_mask[cb] = true` means codebook `cb` is
/// allowed to advance this step (per the delay pattern); inactive ones
/// receive the supplied `bos_token`.
///
/// # Errors
///
/// Bubbles candle errors from the CFG combine + sampler.
///
/// # Panics
///
/// Panics in debug builds if `cond_logits`, `uncond_logits`, and
/// `active_mask` do not all have the same length.
pub fn sample_codebooks(
    cond_logits: &[Tensor],
    uncond_logits: &[Tensor],
    active_mask: &[bool],
    bos_token: u32,
    lp: &mut LogitsProcessor,
    cfg_scale: f64,
) -> Result<Vec<u32>> {
    debug_assert_eq!(cond_logits.len(), uncond_logits.len());
    debug_assert_eq!(cond_logits.len(), active_mask.len());

    let mut tokens = Vec::with_capacity(cond_logits.len());
    for ((cond, uncond), active) in cond_logits
        .iter()
        .zip(uncond_logits.iter())
        .zip(active_mask.iter())
    {
        if !*active {
            tokens.push(bos_token);
            continue;
        }
        let merged = cfg_combine(cond, uncond, cfg_scale)?;
        let token = lp.sample(&merged)?;
        tokens.push(token);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_transformers::generation::{LogitsProcessor, Sampling};

    #[test]
    fn cfg_combine_is_uncond_when_scale_is_zero() {
        let dev = Device::Cpu;
        let cond = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], 3, &dev).unwrap();
        let uncond = Tensor::from_slice(&[0.5_f32, 0.5, 0.5], 3, &dev).unwrap();
        let merged = cfg_combine(&cond, &uncond, 0.0).unwrap();
        let v: Vec<f32> = merged.to_vec1().unwrap();
        assert_eq!(v, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn cfg_combine_is_cond_when_scale_is_one() {
        let dev = Device::Cpu;
        let cond = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], 3, &dev).unwrap();
        let uncond = Tensor::from_slice(&[0.5_f32, 0.5, 0.5], 3, &dev).unwrap();
        let merged = cfg_combine(&cond, &uncond, 1.0).unwrap();
        let v: Vec<f32> = merged.to_vec1().unwrap();
        // uncond + 1.0 * (cond - uncond) == cond
        for (got, want) in v.iter().zip([1.0_f32, 2.0, 3.0].iter()) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }

    #[test]
    fn cfg_combine_amplifies_difference_at_scale_three() {
        let dev = Device::Cpu;
        let cond = Tensor::from_slice(&[2.0_f32, 4.0, 6.0], 3, &dev).unwrap();
        let uncond = Tensor::from_slice(&[1.0_f32, 1.0, 1.0], 3, &dev).unwrap();
        let merged = cfg_combine(&cond, &uncond, 3.0).unwrap();
        let v: Vec<f32> = merged.to_vec1().unwrap();
        // 1 + 3*(2-1)=4, 1+3*(4-1)=10, 1+3*(6-1)=16
        let want = [4.0_f32, 10.0, 16.0];
        for (got, w) in v.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-5, "got {got} want {w}");
        }
    }

    #[test]
    fn sample_codebooks_returns_bos_for_inactive_streams() {
        let dev = Device::Cpu;
        let make = |peak: usize| -> Tensor {
            let mut v = vec![0.0_f32; 8];
            v[peak] = 100.0;
            Tensor::from_vec(v, 8, &dev).unwrap()
        };
        let cond = vec![make(3), make(5), make(7), make(1)];
        let uncond = vec![make(3), make(5), make(7), make(1)];
        let active = [true, true, false, false];
        let mut lp = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
        let toks = sample_codebooks(&cond, &uncond, &active, 2048, &mut lp, 3.0).expect("sample");
        assert_eq!(toks, vec![3, 5, 2048, 2048]);
    }
}
