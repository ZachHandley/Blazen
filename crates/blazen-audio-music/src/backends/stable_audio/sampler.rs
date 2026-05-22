//! Diffusion samplers for the Stable Audio Open native candle port.
//!
//! Two samplers live here:
//!
//! * [`DpmSolverPlusPlus`] — DPM-Solver++ (2nd-order multistep, Karras-style
//!   sigma schedule). Used by `stable-audio-open-1.0` (the 1.21 B variant).
//!   100 steps by default.
//! * [`DistilledSolver`] — fixed-schedule few-step ODE solver. Used by
//!   `stable-audio-open-small` (the 341 M distilled variant). 8 steps.
//!
//! Both samplers consume a closure that produces the DiT's velocity
//! prediction `v` for a given (latent, timestep) pair, perform the
//! v-objective → x0/eps conversion internally, and return the final
//! denoised latent of shape `(B, 64, T)`.
//!
//! Determinism: initial Gaussian noise is sampled from a seeded
//! `xoshiro256**` PRNG fed through Box-Muller. candle's [`Tensor::randn`]
//! API is not directly seedable, so we materialise the noise as
//! `Vec<f32>` and lift it into a tensor via [`Tensor::from_vec`].

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Sampler trait
// ---------------------------------------------------------------------------

/// Common interface for Stable Audio diffusion samplers.
#[allow(
    dead_code,
    reason = "`num_steps` and `schedule` are surfaced for diagnostics and \
              the per-step progress callback wave; the pipeline today \
              only invokes `sample`."
)]
pub trait Sampler: Send + Sync {
    /// Number of inference steps this sampler performs.
    fn num_steps(&self) -> usize;

    /// Per-step `(timestep, sigma)` pairs ordered from high noise to low
    /// noise (i.e. ascending denoising progress). `sigma` is the noise
    /// level; `timestep` is the value handed to the DiT's continuous
    /// timestep input (we use `sigma / (1 + sigma)` for the v-objective).
    fn schedule(&self, device: &Device) -> Result<Vec<(f32, f32)>>;

    /// Run the full sampling loop.
    ///
    /// `denoise_fn(latent, timestep)` must return the DiT's velocity
    /// prediction `v` of the same shape as `latent`. `latent_shape` is
    /// `(B, 64, T)`. Initial noise is sampled deterministically from
    /// `seed`.
    fn sample(
        &self,
        latent_shape: &[usize],
        seed: u64,
        device: &Device,
        denoise_fn: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor>;
}

// ---------------------------------------------------------------------------
// Deterministic Gaussian noise (xoshiro256** + Box-Muller)
// ---------------------------------------------------------------------------

/// Tiny xoshiro256** PRNG so we can sample reproducible Gaussian noise
/// without taking a `rand` dependency. State is seeded from a u64 via
/// SplitMix64 (the construction xoshiro authors recommend).
#[derive(Debug, Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn from_seed(seed: u64) -> Self {
        // SplitMix64 to expand a u64 into four non-zero 64-bit lanes.
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut splitmix = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        let s = [splitmix(), splitmix(), splitmix(), splitmix()];
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f32 in `(0, 1)` — open interval avoids `ln(0)` in
    /// Box-Muller. 24 random bits fit exactly in an f32 mantissa.
    #[allow(clippy::cast_possible_truncation)]
    fn next_u01(&mut self) -> f32 {
        let bits = u32::try_from(self.next_u64() >> 40).unwrap_or(0);
        let unit = (f64::from(bits) + 0.5) * (1.0_f64 / f64::from(1u32 << 24));
        unit as f32
    }
}

/// Draw `n` i.i.d. samples from `Normal(0, 1)` via Box-Muller.
fn gaussian_vec(rng: &mut Xoshiro256, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() + 2 <= n {
        let u1 = rng.next_u01();
        let u2 = rng.next_u01();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        out.push(r * theta.cos());
        out.push(r * theta.sin());
    }
    if out.len() < n {
        let u1 = rng.next_u01();
        let u2 = rng.next_u01();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        out.push(r * (std::f32::consts::TAU * u2).cos());
    }
    out
}

/// Allocate the initial `x_T = sigma_max * eps` noise tensor.
fn seeded_init_noise(
    shape: &[usize],
    sigma_max: f32,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = Xoshiro256::from_seed(seed);
    let mut samples = gaussian_vec(&mut rng, n);
    for v in &mut samples {
        *v *= sigma_max;
    }
    Tensor::from_vec(samples, shape, device)?.to_dtype(DType::F32)
}

// ---------------------------------------------------------------------------
// V-objective conversion helpers
// ---------------------------------------------------------------------------

/// Convert a DiT velocity prediction `v` at noise level `sigma` into the
/// predicted denoised latent `x0`. With the EDM parameterisation:
///
/// ```text
/// alpha_t       = 1 / sqrt(1 + sigma^2)
/// sigma_alpha_t = sigma / sqrt(1 + sigma^2)
/// x0            = alpha_t * x_t - sigma_alpha_t * v
/// ```
fn v_to_x0(x_t: &Tensor, v: &Tensor, sigma: f32) -> Result<Tensor> {
    let denom = (1.0_f32 + sigma * sigma).sqrt();
    let alpha = 1.0_f32 / denom;
    let sigma_alpha = sigma / denom;
    let lhs = (x_t * f64::from(alpha))?;
    let rhs = (v * f64::from(sigma_alpha))?;
    lhs - rhs
}

/// Map a sigma to the DiT's continuous timestep input. Stable Audio
/// uses `t = sigma / (1 + sigma)` so `t ∈ [0, 1)` regardless of how
/// large sigma_max is.
#[inline]
fn sigma_to_timestep(sigma: f32) -> f32 {
    sigma / (1.0 + sigma)
}

// ---------------------------------------------------------------------------
// DPM-Solver++ (multistep, 2nd order)
// ---------------------------------------------------------------------------

/// DPM-Solver++ (2M) for the Stable Audio Open 1.0 variant.
///
/// Karras-style sigma schedule with `rho = 7`:
///
/// ```text
/// sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
/// ```
///
/// Followed by an appended `sigma = 0` terminal step. The multistep
/// update in ODE-Karras coordinates is:
///
/// ```text
/// x_{i+1} = (sigma_{i+1} / sigma_i) * x_i - (sigma_{i+1} - sigma_i) * x0_i
///           - 0.5 * (sigma_{i+1} - sigma_i)
///             * (x0_i - x0_{i-1}) / (sigma_i - sigma_{i-1})
/// ```
///
/// The leading step degenerates to first-order Euler (no history).
#[derive(Debug, Clone)]
pub struct DpmSolverPlusPlus {
    num_steps: usize,
    sigma_min: f32,
    sigma_max: f32,
}

impl DpmSolverPlusPlus {
    /// Build a DPM++ sampler with explicit step count and sigma range.
    #[must_use]
    pub fn new(num_steps: usize, sigma_min: f32, sigma_max: f32) -> Self {
        assert!(num_steps >= 2, "DPM++ needs at least 2 steps");
        assert!(sigma_min > 0.0 && sigma_max > sigma_min);
        Self {
            num_steps,
            sigma_min,
            sigma_max,
        }
    }

    /// Defaults for `stable-audio-open-1.0`: 100 steps, sigma_min=0.002,
    /// sigma_max=80 (matches the reference `stable-audio-tools` config).
    #[must_use]
    pub fn stable_audio_1_0() -> Self {
        Self::new(100, 0.002, 80.0)
    }

    /// Karras sigma schedule (descending; terminal 0 appended in
    /// [`Sampler::sample`]).
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn karras_sigmas(&self) -> Vec<f32> {
        let rho = 7.0_f64;
        let inv_rho = 1.0 / rho;
        let max_inv = f64::from(self.sigma_max).powf(inv_rho);
        let min_inv = f64::from(self.sigma_min).powf(inv_rho);
        let denom = (self.num_steps - 1) as f64;
        (0..self.num_steps)
            .map(|i| {
                let t = i as f64 / denom;
                let base = max_inv + t * (min_inv - max_inv);
                base.powf(rho) as f32
            })
            .collect()
    }
}

impl Sampler for DpmSolverPlusPlus {
    fn num_steps(&self) -> usize {
        self.num_steps
    }

    fn schedule(&self, _device: &Device) -> Result<Vec<(f32, f32)>> {
        Ok(self
            .karras_sigmas()
            .into_iter()
            .map(|s| (sigma_to_timestep(s), s))
            .collect())
    }

    fn sample(
        &self,
        latent_shape: &[usize],
        seed: u64,
        device: &Device,
        denoise_fn: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let mut sigmas = self.karras_sigmas();
        sigmas.push(0.0); // terminal step

        let mut x = seeded_init_noise(latent_shape, self.sigma_max, seed, device)?;
        let mut prev_x0: Option<Tensor> = None;
        let mut prev_sigma: f32 = sigmas[0];

        for i in 0..self.num_steps {
            let sigma_cur = sigmas[i];
            let sigma_next = sigmas[i + 1];
            let t_cur = sigma_to_timestep(sigma_cur);

            let v = denoise_fn(&x, t_cur)?;
            let x0 = v_to_x0(&x, &v, sigma_cur)?;

            let ratio = f64::from(if sigma_cur > 0.0 {
                sigma_next / sigma_cur
            } else {
                0.0
            });
            let lin = f64::from(sigma_next - sigma_cur);

            // First-order Euler step.
            let mut x_next = ((&x * ratio)? - (&x0 * lin)?)?;

            // 2nd-order correction once we have a history term.
            if let Some(prev) = &prev_x0 {
                let dsigma_prev = sigma_cur - prev_sigma;
                if dsigma_prev.abs() > f32::EPSILON {
                    let weight = -0.5 * lin / f64::from(dsigma_prev);
                    let diff = (&x0 - prev)?;
                    x_next = (x_next + (diff * weight)?)?;
                }
            }

            x = x_next;
            prev_x0 = Some(x0);
            prev_sigma = sigma_cur;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Distilled few-step solver (small variant)
// ---------------------------------------------------------------------------

/// Hand-tuned 8-step schedule for `stable-audio-open-small`.
///
/// **TODO(stable-audio-open-small):** these floats are placeholders modelled
/// after the public ARM `audiogen` distilled-schedule pattern (geometric
/// interpolation in log-sigma between `sigma_min=0.03` and `sigma_max=80`).
/// They must be replaced with the *exact* sigma values from
/// `stable-audio-tools`' `distilled_small_inference_config.json` before
/// shipping audible output, otherwise the model will run but produce
/// off-distribution samples.
const DISTILLED_SMALL_SIGMAS: [f32; 8] =
    [80.000, 32.000, 12.800, 5.120, 2.048, 0.819, 0.328, 0.030];

/// Distilled few-step ODE sampler used by `stable-audio-open-small`.
///
/// Unlike [`DpmSolverPlusPlus`] this is a *fixed* schedule with no
/// 2nd-order correction — the distillation procedure absorbs the
/// higher-order trajectory into the network weights, so per-step Euler
/// matches the teacher trajectory within distillation noise.
#[derive(Debug, Clone)]
pub struct DistilledSolver {
    timesteps: Vec<f32>,
}

impl DistilledSolver {
    /// Build a distilled sampler from an explicit sigma schedule
    /// (descending, terminal value approaches `sigma_min`).
    #[must_use]
    pub fn new(timesteps: Vec<f32>) -> Self {
        assert!(timesteps.len() >= 2, "distilled solver needs ≥2 sigmas");
        Self { timesteps }
    }

    /// The default 8-step schedule for `stable-audio-open-small`.
    ///
    /// See the constant `DISTILLED_SMALL_SIGMAS` above — values are
    /// placeholders pending transcription from
    /// `stable-audio-tools`. The schedule shape (descending, 8 steps,
    /// covering `sigma ∈ [~0.03, 80]`) is correct.
    #[must_use]
    pub fn stable_audio_small() -> Self {
        Self::new(DISTILLED_SMALL_SIGMAS.to_vec())
    }
}

impl Sampler for DistilledSolver {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn schedule(&self, _device: &Device) -> Result<Vec<(f32, f32)>> {
        Ok(self
            .timesteps
            .iter()
            .map(|&s| (sigma_to_timestep(s), s))
            .collect())
    }

    fn sample(
        &self,
        latent_shape: &[usize],
        seed: u64,
        device: &Device,
        denoise_fn: &dyn Fn(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let mut sigmas = self.timesteps.clone();
        sigmas.push(0.0);

        let sigma_max = self.timesteps[0];
        let mut x = seeded_init_noise(latent_shape, sigma_max, seed, device)?;

        for i in 0..self.timesteps.len() {
            let sigma_cur = sigmas[i];
            let sigma_next = sigmas[i + 1];
            let t_cur = sigma_to_timestep(sigma_cur);

            let v = denoise_fn(&x, t_cur)?;
            let x0 = v_to_x0(&x, &v, sigma_cur)?;

            // Euler-style step in Karras-ODE coords.
            let ratio = f64::from(if sigma_cur > 0.0 {
                sigma_next / sigma_cur
            } else {
                0.0
            });
            let lin = f64::from(sigma_next - sigma_cur);
            x = ((&x * ratio)? - (&x0 * lin)?)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn dpmpp_schedule_monotonic() {
        let s = DpmSolverPlusPlus::stable_audio_1_0();
        let pairs = s.schedule(&cpu()).unwrap();
        assert_eq!(pairs.len(), 100);
        // sigmas strictly decreasing.
        for w in pairs.windows(2) {
            assert!(
                w[0].1 > w[1].1,
                "sigma must decrease: {} !> {}",
                w[0].1,
                w[1].1
            );
        }
        // first sigma == sigma_max, last sigma == sigma_min.
        assert!((pairs[0].1 - 80.0).abs() < 1e-3);
        assert!((pairs[99].1 - 0.002).abs() < 1e-3);
    }

    #[test]
    fn distilled_schedule_has_eight_steps() {
        let s = DistilledSolver::stable_audio_small();
        assert_eq!(s.num_steps(), 8);
        let pairs = s.schedule(&cpu()).unwrap();
        assert_eq!(pairs.len(), 8);
        // every (timestep, sigma) is finite and sigma ∈ (0, 100).
        for (t, sigma) in pairs {
            assert!(t.is_finite() && sigma.is_finite());
            assert!(sigma > 0.0 && sigma < 100.0);
        }
    }

    fn zero_denoise(latent: &Tensor, _t: f32) -> Result<Tensor> {
        // Returns a zero tensor of the same shape as the latent.
        latent.zeros_like()
    }

    #[test]
    fn sample_with_zero_denoise_returns_finite_distilled() {
        let s = DistilledSolver::stable_audio_small();
        let shape = [1usize, 64, 16];
        let out = s.sample(&shape, 42, &cpu(), &zero_denoise).unwrap();
        assert_eq!(out.dims(), &shape);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|x| x.is_finite()),
            "NaN/Inf in distilled out"
        );
    }

    #[test]
    fn sample_with_zero_denoise_returns_finite_dpmpp() {
        // Use a smaller step count to keep the test cheap.
        let s = DpmSolverPlusPlus::new(8, 0.002, 80.0);
        let shape = [1usize, 64, 16];
        let out = s.sample(&shape, 7, &cpu(), &zero_denoise).unwrap();
        assert_eq!(out.dims(), &shape);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|x| x.is_finite()), "NaN/Inf in DPM++ out");
    }

    #[test]
    fn sample_is_deterministic_with_seed() {
        let s = DistilledSolver::stable_audio_small();
        let shape = [1usize, 64, 8];
        let a = s.sample(&shape, 12345, &cpu(), &zero_denoise).unwrap();
        let b = s.sample(&shape, 12345, &cpu(), &zero_denoise).unwrap();
        let c = s.sample(&shape, 67890, &cpu(), &zero_denoise).unwrap();

        let va: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let vb: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        let vc: Vec<f32> = c.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(va, vb, "same seed must produce byte-identical output");
        assert_ne!(va, vc, "different seed must produce different output");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn xoshiro_box_muller_roughly_unit_normal() {
        let mut rng = Xoshiro256::from_seed(0xdead_beef);
        let n = 4096_usize;
        let samples = gaussian_vec(&mut rng, n);
        let nf = n as f32;
        let mean: f32 = samples.iter().sum::<f32>() / nf;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / nf;
        assert!(mean.abs() < 0.1, "mean drift: {mean}");
        assert!((var - 1.0).abs() < 0.15, "variance off: {var}");
    }
}
