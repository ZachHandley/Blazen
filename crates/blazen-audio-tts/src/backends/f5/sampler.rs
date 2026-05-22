//! Flow-matching Euler ODE sampler for F5-TTS.
//!
//! F5-TTS is built on conditional flow matching (Lipman et al., 2023 —
//! arxiv:2210.02747). The `DiT` predicts a velocity field
//! `v_theta(x_t, t, text, reference)` that transports an initial
//! Gaussian sample `x_0 ~ N(0, I)` along a straight-line probability
//! path to the data sample `x_1` (target mel-spectrogram). At inference
//! time we integrate the ODE
//!
//! ```text
//!     dx/dt = v_theta(x_t, t, text, reference)
//! ```
//!
//! with a first-order Euler solver, classifier-free guidance, and
//! `n_steps = 32` (configurable) function evaluations.
//!
//! # Schedule
//!
//! Upstream `SWivid/F5-TTS/src/f5_tts/model/cfm.py::CFM::sample`
//! constructs the time grid as `torch.linspace(0, 1, steps + 1)` —
//! `steps + 1` knots, `steps` (uniform) intervals — and then optionally
//! warps it with the "sway sampling" perturbation
//! `t' = t + s * (cos(pi/2 * t) - 1 + t)` (default `s = -1.0`,
//! cfm.py:216). Both schedules are exposed here so Wave F.4 can match
//! upstream behavior exactly. We follow upstream and do *not* assume
//! `dt = 1/n_steps` once sway is enabled — the per-step `dt` comes from
//! the schedule's successive differences.
//!
//! # Classifier-free guidance
//!
//! Upstream uses the
//!
//! ```text
//!     v = v_cond + (v_cond - v_uncond) * cfg_strength
//! ```
//!
//! variant (cfm.py:191) — equivalent to the more common
//! `v_uncond + (1 + cfg)*(v_cond - v_uncond)` reformulation. Default
//! `cfg_strength = 2.0` (`utils_infer.py:62`). When `cfg_strength` is
//! within `1e-5` of zero upstream skips the unconditional pass
//! entirely (cfm.py:167); we keep that fast path and also add a
//! `cfg_strength == 0` shortcut for the cond-only / classifier-free-off
//! mode that callers may want.
//!
//! Reference: `SWivid/F5-TTS/src/f5_tts/model/cfm.py`, sample method
//! at lines 84–229.

#![cfg(feature = "f5-tts")]

use candle_core::{Device, Tensor};

use crate::error::TtsError;

/// Sampler configuration.
///
/// Defaults match upstream F5-TTS inference defaults
/// (`utils_infer.py:61-63`): 32 Euler steps, CFG strength 2.0, sway
/// coefficient -1.0.
#[derive(Debug, Clone, Copy)]
pub struct F5Sampling {
    /// Number of Euler steps (function evaluations).
    pub n_steps: usize,
    /// Classifier-free guidance strength. `pred + (pred - null) * cfg`.
    /// When `< 1e-5`, the unconditional pass is skipped entirely.
    pub cfg_strength: f32,
    /// Sway-sampling coefficient. `Some(-1.0)` is upstream default.
    /// `None` keeps a uniform `linspace(0, 1, n+1)` grid.
    pub sway_sampling_coef: Option<f32>,
    /// Optional RNG seed for the initial noise.
    ///
    /// **Currently a no-op for CPU**: candle's CPU `Tensor::randn`
    /// path doesn't honour a device-level seed (`Device::set_seed`
    /// is CUDA-only and CPU randn pulls from a thread-local default
    /// RNG that ignores any seed we'd set). The field is retained so
    /// callers can keep wiring a seed through the public API without
    /// the call site needing to know about the gap.
    ///
    /// TODO: wire to a `rand::Rng`-seeded path that generates the
    /// initial noise on the host and uploads via `Tensor::from_vec`
    /// when full CPU reproducibility is required. CUDA support can
    /// then route through `candle_core::cuda::CudaDevice::set_seed`.
    pub seed: Option<u64>,
}

impl Default for F5Sampling {
    fn default() -> Self {
        Self {
            n_steps: 32,
            cfg_strength: 2.0,
            sway_sampling_coef: Some(-1.0),
            seed: None,
        }
    }
}

/// Threshold below which `cfg_strength` is treated as "off" and the
/// unconditional forward pass is skipped (matches `cfm.py:167`).
const CFG_OFF_EPSILON: f32 = 1e-5;

/// Build the time-step schedule on the host.
///
/// Returns `n_steps + 1` knots in `[0, 1]`. With `sway = None` this is
/// `linspace(0, 1, n_steps + 1)`; with `sway = Some(s)` the schedule
/// is warped per upstream cfm.py:215-216.
fn build_schedule(n_steps: usize, sway: Option<f32>) -> Vec<f32> {
    // n_steps == 0 is degenerate (no work) but we still want at least
    // one knot so callers see the noise tensor unchanged.
    let n_knots = n_steps + 1;
    let mut t: Vec<f32> = (0..n_knots)
        .map(|i| {
            // linspace(0, 1, n_knots)
            if n_steps == 0 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let num = i as f32;
                #[allow(clippy::cast_precision_loss)]
                let den = n_steps as f32;
                num / den
            }
        })
        .collect();
    if let Some(s) = sway {
        for v in &mut t {
            // t' = t + s * (cos(pi/2 * t) - 1 + t)
            let warp = (std::f32::consts::FRAC_PI_2 * *v).cos() - 1.0 + *v;
            *v += s * warp;
        }
    }
    t
}

/// Signature of the velocity-field forward pass.
///
/// Matches the (anticipated) `F5Dit::forward(mel, text_ids, timestep)`
/// signature from Wave 2.A. We deliberately take a closure here so
/// (a) tests can mock the `DiT` without an `unsafe` weight-load path and
/// (b) the sampler stays decoupled from the wrapper's concrete type
/// while sibling Wave 2.A is in flight. Wave F.4 wires
/// `dit.forward(...)` into this closure.
///
/// Arguments:
/// - `x`: `[B, T_mel, mel_dim]` current latent state.
/// - `text_ids`: `[B, T_text]` text token IDs (or a null-batch when
///   the caller is requesting the unconditional pass).
/// - `t`: `[B]` scalar timestep tensor broadcastable to each sample.
///
/// Returns: predicted velocity tensor with the same shape as `x`.
pub(super) type F5Forward<'a> = &'a dyn Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor, TtsError>;

/// Run the Euler flow-matching sampler.
///
/// Arguments:
/// - `forward`: the `DiT` velocity-field evaluator (see [`F5Forward`]).
/// - `text_ids`: `[B, T_text]` conditional text tokens.
/// - `null_text_ids`: `[B, T_text]` unconditional (null) text tokens
///   used for classifier-free guidance. Ignored when
///   `cfg_strength.abs() < 1e-5`.
/// - `mel_shape`: target `(B, T_mel, mel_dim)`.
/// - `sampling`: [`F5Sampling`] config.
/// - `device`: candle device for the noise tensor + the timestep
///   scalar.
///
/// Returns: `[B, T_mel, mel_dim]` predicted mel-spectrogram.
#[allow(dead_code)] // Wired by Wave F.4; kept pub(super) until then.
pub(super) fn sample(
    forward: F5Forward<'_>,
    text_ids: &Tensor,
    null_text_ids: &Tensor,
    mel_shape: (usize, usize, usize),
    sampling: &F5Sampling,
    device: &Device,
) -> Result<Tensor, TtsError> {
    let (b, t_mel, mel_dim) = mel_shape;
    let shape = [b, t_mel, mel_dim];

    // 1. Sample noise: x_0 ~ N(0, I).
    //    `sampling.seed` is currently a no-op on CPU (candle's CPU
    //    randn pulls from a thread-local default RNG that ignores any
    //    seed we'd set, and `Device::set_seed` is CUDA-only — calling
    //    it on CPU returns an error). We deliberately drop the seed
    //    here so the sampler stays callable on CPU; see the
    //    `F5Sampling::seed` doc-comment for the planned CPU-side
    //    seeded-RNG path.
    let _ = sampling.seed;
    let mut x = Tensor::randn(0.0_f32, 1.0_f32, &shape, device)
        .map_err(|e| TtsError::Synthesis(format!("f5-tts: sample noise: {e}")))?;

    // 2. Build schedule (host-side); compute per-step dt as the
    //    forward difference (sway sampling makes dt non-uniform).
    let t_knots = build_schedule(sampling.n_steps, sampling.sway_sampling_coef);

    // 3. Euler integration.
    let use_cfg = sampling.cfg_strength.abs() >= CFG_OFF_EPSILON;
    #[allow(clippy::cast_possible_truncation)]
    let cfg = f64::from(sampling.cfg_strength);

    for step in 0..sampling.n_steps {
        let t = t_knots[step];
        let dt = f64::from(t_knots[step + 1] - t_knots[step]);

        // Per-batch scalar timestep tensor (shape `[B]`). Repeating
        // the scalar matches upstream: every sample in the batch
        // shares the same integrator time.
        let t_tensor = Tensor::from_slice(&vec![t; b], (b,), device)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: timestep tensor: {e}")))?;

        let v_cond = forward(&x, text_ids, &t_tensor).map_err(|e| match e {
            TtsError::Synthesis(msg) => {
                TtsError::Synthesis(format!("f5-tts: dit conditional: {msg}"))
            }
            other => other,
        })?;

        let v = if use_cfg {
            let v_uncond = forward(&x, null_text_ids, &t_tensor).map_err(|e| match e {
                TtsError::Synthesis(msg) => {
                    TtsError::Synthesis(format!("f5-tts: dit unconditional: {msg}"))
                }
                other => other,
            })?;
            // Upstream formula (cfm.py:191):
            //   v = v_cond + (v_cond - v_uncond) * cfg_strength
            let diff = v_cond
                .sub(&v_uncond)
                .map_err(|e| TtsError::Synthesis(format!("f5-tts: cfg diff: {e}")))?;
            let scaled =
                (diff * cfg).map_err(|e| TtsError::Synthesis(format!("f5-tts: cfg scale: {e}")))?;
            v_cond
                .add(&scaled)
                .map_err(|e| TtsError::Synthesis(format!("f5-tts: cfg combine: {e}")))?
        } else {
            v_cond
        };

        let dx = (v * dt).map_err(|e| TtsError::Synthesis(format!("f5-tts: euler dx: {e}")))?;
        x = x
            .add(&dx)
            .map_err(|e| TtsError::Synthesis(format!("f5-tts: euler step: {e}")))?;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::cell::Cell;

    /// Counts conditional / unconditional forward calls. We use the
    /// presence of `null_text_ids` (vs `text_ids`) to disambiguate
    /// which branch was taken: a mock dit just echoes the input back
    /// unchanged so we can assert shape preservation while still
    /// observing call counts via a closure-captured `Cell`.
    fn mock_identity_forward<'a>(
        cond_calls: &'a Cell<usize>,
        uncond_calls: &'a Cell<usize>,
        cond_id: i64,
        uncond_id: i64,
    ) -> impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor, TtsError> + 'a {
        move |x: &Tensor, text_ids: &Tensor, _t: &Tensor| -> Result<Tensor, TtsError> {
            // Inspect the first text id to classify the call.
            let first = text_ids
                .flatten_all()
                .map_err(|e| TtsError::Synthesis(format!("mock flatten: {e}")))?
                .get(0)
                .map_err(|e| TtsError::Synthesis(format!("mock index: {e}")))?
                .to_scalar::<i64>()
                .map_err(|e| TtsError::Synthesis(format!("mock scalar: {e}")))?;
            if first == cond_id {
                cond_calls.set(cond_calls.get() + 1);
            } else if first == uncond_id {
                uncond_calls.set(uncond_calls.get() + 1);
            }
            // Return a zero velocity field so x stays unchanged. This
            // lets `sample_returns_correct_shape` assert shape without
            // worrying about NaNs from compounding random updates.
            x.zeros_like()
                .map_err(|e| TtsError::Synthesis(format!("mock zeros: {e}")))
        }
    }

    #[test]
    fn f5_sampling_defaults_match_upstream() {
        // Cross-checked against
        // SWivid/F5-TTS/src/f5_tts/infer/utils_infer.py:61-63
        // (nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0).
        let s = F5Sampling::default();
        assert_eq!(s.n_steps, 32);
        assert!((s.cfg_strength - 2.0).abs() < f32::EPSILON);
        assert_eq!(s.sway_sampling_coef, Some(-1.0));
        assert!(s.seed.is_none());
    }

    #[test]
    fn euler_step_size_matches_inverse_n_steps() {
        // Without sway sampling, the schedule is linspace(0, 1, n+1)
        // so every successive dt == 1/n_steps. With sway, the per-step
        // dt varies — but the total `sum(dt_i)` still equals 1.0
        // because the warp fixes the endpoints (t=0 and t=1 are
        // invariant under `cos(pi/2 * t) - 1 + t`).
        let knots_uniform = build_schedule(32, None);
        assert_eq!(knots_uniform.len(), 33);
        let dt_uniform = knots_uniform[1] - knots_uniform[0];
        assert!((dt_uniform - 1.0 / 32.0).abs() < 1e-6);
        for w in knots_uniform.windows(2) {
            assert!((w[1] - w[0] - dt_uniform).abs() < 1e-5);
        }

        let knots_sway = build_schedule(32, Some(-1.0));
        assert!((knots_sway[0] - 0.0).abs() < 1e-5);
        assert!((knots_sway[knots_sway.len() - 1] - 1.0).abs() < 1e-5);
        let total: f32 = knots_sway.windows(2).map(|w| w[1] - w[0]).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cfg_off_skips_unconditional_call() {
        let device = Device::Cpu;
        let cond_calls = Cell::new(0_usize);
        let uncond_calls = Cell::new(0_usize);
        let cond_id: i64 = 7;
        let uncond_id: i64 = 0;
        let fwd = mock_identity_forward(&cond_calls, &uncond_calls, cond_id, uncond_id);

        let text_ids = Tensor::from_slice(&[cond_id, cond_id], (1, 2), &device).expect("text_ids");
        let null_ids =
            Tensor::from_slice(&[uncond_id, uncond_id], (1, 2), &device).expect("null_ids");
        let sampling = F5Sampling {
            n_steps: 4,
            cfg_strength: 0.0, // off — unconditional pass must be skipped
            sway_sampling_coef: None,
            seed: Some(0xF5),
        };
        let _ = sample(&fwd, &text_ids, &null_ids, (1, 3, 5), &sampling, &device)
            .expect("sample with cfg=0");
        assert_eq!(cond_calls.get(), 4);
        assert_eq!(uncond_calls.get(), 0);
    }

    #[test]
    fn cfg_on_runs_both_passes() {
        let device = Device::Cpu;
        let cond_calls = Cell::new(0_usize);
        let uncond_calls = Cell::new(0_usize);
        let cond_id: i64 = 7;
        let uncond_id: i64 = 0;
        let fwd = mock_identity_forward(&cond_calls, &uncond_calls, cond_id, uncond_id);

        let text_ids = Tensor::from_slice(&[cond_id, cond_id], (1, 2), &device).expect("text_ids");
        let null_ids =
            Tensor::from_slice(&[uncond_id, uncond_id], (1, 2), &device).expect("null_ids");
        let sampling = F5Sampling {
            n_steps: 4,
            cfg_strength: 2.0,
            sway_sampling_coef: Some(-1.0),
            seed: Some(0xF5),
        };
        let _ = sample(&fwd, &text_ids, &null_ids, (1, 3, 5), &sampling, &device)
            .expect("sample with cfg=2");
        assert_eq!(cond_calls.get(), 4);
        assert_eq!(uncond_calls.get(), 4);
    }

    #[test]
    fn sample_returns_correct_shape() {
        let device = Device::Cpu;
        let cond_calls = Cell::new(0_usize);
        let uncond_calls = Cell::new(0_usize);
        let fwd = mock_identity_forward(&cond_calls, &uncond_calls, 7, 0);

        let text_ids = Tensor::from_slice(&[7_i64, 7], (1, 2), &device).expect("text_ids");
        let null_ids = Tensor::from_slice(&[0_i64, 0], (1, 2), &device).expect("null_ids");
        let sampling = F5Sampling {
            n_steps: 8,
            cfg_strength: 2.0,
            sway_sampling_coef: Some(-1.0),
            seed: Some(0xF5),
        };

        let shape = (2, 16, 100); // B=2, T_mel=16, mel_dim=100
        let out =
            sample(&fwd, &text_ids, &null_ids, shape, &sampling, &device).expect("sample shape");
        assert_eq!(out.dims(), &[shape.0, shape.1, shape.2]);
    }
}
