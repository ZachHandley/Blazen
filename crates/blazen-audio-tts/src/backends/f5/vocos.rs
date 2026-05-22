//! Vocos mel-spectrogram vocoder for F5-TTS.
//!
//! Converts the `DiT`'s predicted 24 kHz mel-spectrogram into a raw
//! waveform via a `ConvNeXt`-V1 backbone followed by an `iSTFT` head.
//! Canonical Hugging Face checkpoint is `charactr/vocos-mel-24khz`.
//!
//! Architecture (per Vocos paper, Siuzdak 2023, arxiv:2306.00814,
//! upstream `gemelo-ai/vocos`):
//!
//! 1. **Input embedding** — 1-D conv `mel_bins -> hidden_dim`,
//!    kernel 7, padding 3 (matches `vocos/models.py:VocosBackbone`).
//! 2. **N `ConvNeXt`-V1 blocks** (`vocos/modules.py:ConvNeXtBlock`):
//!    - Depthwise `Conv1d(dim, dim, kernel_size=7, padding=3,
//!      groups=dim)`.
//!    - `LayerNorm` (channels-last after a `(B,C,T) -> (B,T,C)`
//!      transpose).
//!    - Pointwise `Linear(dim -> intermediate_dim)`.
//!    - `GELU`.
//!    - Pointwise `Linear(intermediate_dim -> dim)`.
//!    - Per-channel `gamma` `LayerScale` (init `1 / num_layers`).
//!    - Transpose back to `(B,C,T)`, residual add.
//! 3. **Final `LayerNorm`** on channels-last.
//! 4. **iSTFT head** (`vocos/heads.py:ISTFTHead` +
//!    `vocos/spectral_ops.py:ISTFT`):
//!    - Linear projection `hidden_dim -> n_fft + 2`.
//!    - Split into magnitude (first half) and phase (second half).
//!    - `mag = clip(exp(mag), max=1e2)`, `S = mag * (cos(p) + j sin(p))`.
//!    - `irfft` per frame, Hann-window weighting, overlap-add fold,
//!      divide by the per-sample squared-window envelope.
//!
//! # FFT backend
//!
//! `candle-core` has no native real-FFT op (cudnn `Fft` enum hits but
//! no `Tensor::rfft`/`irfft`). The crate's other audio paths
//! (`blazen-audio-music`, `blazen-audio-codec`) do not surface an FFT
//! either. Wave F.2 therefore drives the inverse FFT via [`rustfft`]
//! on CPU — the iSTFT runs once per synthesis at the end of the
//! pipeline, so the host-roundtrip cost is amortised against the
//! preceding `DiT` sampling loop. `rustfft` is gated behind the
//! `f5-tts` cargo feature; the rest of the crate is unaffected.

#![cfg(feature = "f5-tts")]
// `VarBuilder` is the candle convention for module construction —
// each `.pp(...)` returns a fresh child builder, so the outer caller
// happily passes it by value (matches `blazen_audio_core::dit`'s
// module-level allow).
#![allow(clippy::needless_pass_by_value)]
// Hann-window math, FFT scaling, and LayerScale-init all live in
// floating-point land where the small precision loss from
// `usize as f32/f64` is expected and benign.
#![allow(clippy::cast_precision_loss)]

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder, conv1d, layer_norm, linear};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

/// Configuration knobs for the Vocos vocoder.
///
/// Defaults map to the canonical `charactr/vocos-mel-24khz`
/// checkpoint (24 kHz output, 100-bin mel, 8 `ConvNeXt` layers,
/// hidden dim 512, intermediate dim 1536, `n_fft` 1024, `hop_length`
/// 256, `win_length` 1024).
#[derive(Debug, Clone)]
pub struct VocosConfig {
    /// Input mel-spectrogram dimensionality (number of mel bins).
    pub mel_dim: usize,
    /// `ConvNeXt` hidden / embedding dimension.
    pub hidden_dim: usize,
    /// Number of stacked `ConvNeXt` blocks in the backbone.
    pub n_layers: usize,
    /// Pointwise-expansion dimensionality inside each `ConvNeXt`
    /// block. Upstream uses `4 * hidden_dim`.
    pub intermediate_dim: usize,
    /// FFT window size for the iSTFT head.
    pub n_fft: usize,
    /// STFT hop length (stride between consecutive frames).
    pub hop_length: usize,
    /// STFT analysis-window length (typically equal to `n_fft`).
    pub win_length: usize,
    /// Output sample rate. Informational — the iSTFT itself does
    /// not depend on it.
    pub sample_rate: u32,
}

impl VocosConfig {
    /// Default `charactr/vocos-mel-24khz` configuration.
    #[must_use]
    pub fn vocos_24khz() -> Self {
        Self {
            mel_dim: 100,
            hidden_dim: 512,
            n_layers: 8,
            intermediate_dim: 1536,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            sample_rate: 24_000,
        }
    }

    /// Initial value for the per-block `LayerScale` `gamma` (per
    /// `vocos/modules.py:ConvNeXtBlock`: `1 / num_layers`).
    fn layer_scale_init(&self) -> f64 {
        1.0 / self.n_layers.max(1) as f64
    }

    /// FFT bin count (`n_fft / 2 + 1`).
    fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// One `ConvNeXt`-V1 block in the Vocos backbone.
///
/// Mirrors `vocos/modules.py:ConvNeXtBlock` — depthwise conv,
/// channels-last `LayerNorm`, two pointwise linears with `GELU`,
/// a learnable per-channel `gamma` (`LayerScale`), residual add.
pub(super) struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    /// Construct a block from a `VarBuilder`. Weight paths follow
    /// upstream Vocos checkpoint naming (`dwconv`, `norm`, `pwconv1`,
    /// `pwconv2`, `gamma`).
    pub fn load(vb: VarBuilder, cfg: &VocosConfig) -> Result<Self> {
        let dwconv = conv1d(
            cfg.hidden_dim,
            cfg.hidden_dim,
            7,
            Conv1dConfig {
                padding: 3,
                groups: cfg.hidden_dim,
                ..Default::default()
            },
            vb.pp("dwconv"),
        )?;
        let norm = layer_norm(cfg.hidden_dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = linear(cfg.hidden_dim, cfg.intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(cfg.intermediate_dim, cfg.hidden_dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(cfg.hidden_dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Forward pass.
    ///
    /// `x`: `[B, hidden_dim, T]`. Returns `[B, hidden_dim, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        // Depthwise conv in NCT layout (groups == hidden_dim).
        let h = self.dwconv.forward(x)?;
        // (B,C,T) -> (B,T,C) for channels-last LayerNorm + pointwise
        // Linears.
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.pwconv2.forward(&h)?;
        // LayerScale: per-channel gamma multiply.
        // gamma: [C] broadcast against h: [B,T,C].
        let h = h.broadcast_mul(&self.gamma)?;
        // (B,T,C) -> (B,C,T) and residual add.
        let h = h.transpose(1, 2)?;
        residual + h
    }
}

/// Vocos `ConvNeXt` backbone + iSTFT-head projection.
///
/// Forward returns the raw `[B, n_fft + 2, T]` magnitude/phase
/// spectrum — pass it through [`istft`] to obtain the waveform.
pub struct VocosBackbone {
    cfg: VocosConfig,
    embed: Conv1d,
    in_norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
    head: Linear,
}

impl VocosBackbone {
    /// Construct the backbone from a `VarBuilder` rooted at the
    /// model's top-level path. Layout mirrors
    /// `vocos.feature_extractor`/`vocos.backbone`/`vocos.head`
    /// in the upstream checkpoint:
    ///
    /// - `embed`: `Conv1d(mel_dim, hidden_dim, k=7, p=3)`.
    /// - `norm`: input `LayerNorm(hidden_dim)`.
    /// - `convnext.{i}.{dwconv,norm,pwconv1,pwconv2,gamma}`.
    /// - `final_layer_norm`: `LayerNorm(hidden_dim)`.
    /// - `head.out`: `Linear(hidden_dim, n_fft + 2)`.
    pub fn load(vb: VarBuilder, cfg: VocosConfig) -> Result<Self> {
        let embed = conv1d(
            cfg.mel_dim,
            cfg.hidden_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("embed"),
        )?;
        let in_norm = layer_norm(cfg.hidden_dim, 1e-6, vb.pp("norm"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        let block_vb = vb.pp("convnext");
        for i in 0..cfg.n_layers {
            blocks.push(ConvNeXtBlock::load(block_vb.pp(i.to_string()), &cfg)?);
        }
        let final_norm = layer_norm(cfg.hidden_dim, 1e-6, vb.pp("final_layer_norm"))?;
        let n_out = cfg.n_fft + 2;
        let head = linear(cfg.hidden_dim, n_out, vb.pp("head").pp("out"))?;
        Ok(Self {
            cfg,
            embed,
            in_norm,
            blocks,
            final_norm,
            head,
        })
    }

    /// Read-only access to the underlying [`VocosConfig`].
    #[must_use]
    pub fn config(&self) -> &VocosConfig {
        &self.cfg
    }

    /// Forward pass.
    ///
    /// `mel`: `[B, mel_dim, T]`. Returns `[B, n_fft + 2, T]`,
    /// suitable for feeding to [`istft`].
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let h = self.embed.forward(mel)?;
        // Input LayerNorm runs channels-last (per
        // vocos/models.py:VocosBackbone.forward).
        let h = h.transpose(1, 2)?;
        let h = self.in_norm.forward(&h)?;
        let mut h = h.transpose(1, 2)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = h.transpose(1, 2)?;
        let h = self.final_norm.forward(&h)?;
        // Head projection (channels-last), transpose back to NCT.
        let h = self.head.forward(&h)?;
        h.transpose(1, 2)
    }
}

/// iSTFT — inverse Short-Time Fourier Transform with Hann-window
/// overlap-add.
///
/// Mirrors `vocos/spectral_ops.py:ISTFT` (`padding="same"` branch):
///
/// 1. Split `spec: [B, n_fft + 2, T]` into magnitude (first
///    `n_fft / 2 + 1` channels) and phase (second half).
/// 2. `mag = clip(exp(mag), max=1e2)`,
///    `S = mag * (cos(p) + j sin(p))`.
/// 3. Per-frame `irfft` → real signal of length `n_fft`.
/// 4. Multiply by Hann window, fold via overlap-add into a buffer of
///    length `(T - 1) * hop + win_length`.
/// 5. Compute the squared-window envelope by overlap-adding
///    `window^2` over the same frame layout, then divide pointwise.
/// 6. Trim `pad = (win_length - hop_length) / 2` samples from each
///    end so the output matches the centred-STFT length.
///
/// Returns `[B, n_samples]` (`f32` waveform).
pub fn istft(spec: &Tensor, cfg: &VocosConfig) -> Result<Tensor> {
    let (b, c, t_frames) = spec.dims3()?;
    let n_bins = cfg.n_bins();
    let expected_c = 2 * n_bins;
    if c != expected_c {
        candle_core::bail!("vocos iSTFT: spec channel dim {c} != 2 * (n_fft/2 + 1) = {expected_c}");
    }
    if cfg.win_length != cfg.n_fft {
        // Upstream Vocos always uses win_length == n_fft. The fold
        // path below assumes this so we don't need an asymmetric
        // pre-pad.
        candle_core::bail!(
            "vocos iSTFT: win_length {} must equal n_fft {}",
            cfg.win_length,
            cfg.n_fft
        );
    }
    if cfg.hop_length == 0 || cfg.hop_length > cfg.win_length {
        candle_core::bail!(
            "vocos iSTFT: hop_length {} must be in 1..=win_length {}",
            cfg.hop_length,
            cfg.win_length
        );
    }
    let pad = (cfg.win_length - cfg.hop_length) / 2;
    let full_len = (t_frames - 1) * cfg.hop_length + cfg.win_length;
    if full_len <= 2 * pad {
        candle_core::bail!(
            "vocos iSTFT: full output length {full_len} <= 2 * pad {pad_doubled} — \
             input has too few frames ({t_frames}) for win/hop {win}/{hop}",
            pad_doubled = 2 * pad,
            win = cfg.win_length,
            hop = cfg.hop_length,
        );
    }
    let out_len = full_len - 2 * pad;

    // Magnitude/phase split + clip(exp(mag), max=1e2).
    // `clip` floor is 0.0 because exp() is non-negative — only the
    // upper bound matters for numerical stability against runaway
    // amplitudes.
    let mag = spec.i((.., 0..n_bins, ..))?;
    let phase = spec.i((.., n_bins..2 * n_bins, ..))?;
    let mag = mag.exp()?.clamp(0f32, 1e2f32)?;
    let real = (&mag * phase.cos()?)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let imag = (&mag * phase.sin()?)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;

    // Pre-compute the Hann window and the per-sample squared-window
    // envelope once per call. These only depend on (T, win, hop), not
    // on batch contents.
    let window = hann_window(cfg.win_length);
    let envelope = window_envelope(&window, t_frames, cfg.hop_length, full_len);

    // Per-frame inverse real FFT via rustfft.
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(cfg.n_fft);
    let mut scratch = vec![Complex32::new(0.0, 0.0); ifft.get_inplace_scratch_len()];

    let mut out = vec![0f32; b * out_len];
    for bi in 0..b {
        let mut frame_buf = vec![Complex32::new(0.0, 0.0); cfg.n_fft];
        let mut acc = vec![0f32; full_len];
        for ti in 0..t_frames {
            // Build the full (Hermitian) complex spectrum: the
            // negative-frequency bins are the conjugate of the
            // positive-frequency bins (excluding DC and Nyquist).
            for k in 0..n_bins {
                frame_buf[k] = Complex32::new(real[bi][k][ti], imag[bi][k][ti]);
            }
            for k in 1..=(cfg.n_fft - n_bins) {
                let mirror = cfg.n_fft - k;
                frame_buf[mirror] = frame_buf[k].conj();
            }
            ifft.process_with_scratch(&mut frame_buf, &mut scratch);
            // rustfft's inverse FFT is unscaled, so divide by n_fft
            // to recover the convention torch.fft.irfft uses (the
            // `norm="backward"` setting in upstream Vocos).
            let scale = 1.0 / cfg.n_fft as f32;
            let frame_start = ti * cfg.hop_length;
            for s in 0..cfg.win_length {
                acc[frame_start + s] += frame_buf[s].re * scale * window[s];
            }
        }
        // Normalise by the squared-window envelope and trim to the
        // centred-STFT region.
        for s in 0..out_len {
            let env = envelope[s + pad];
            if env > 1e-11 {
                out[bi * out_len + s] = acc[s + pad] / env;
            }
        }
    }

    Tensor::from_vec(out, (b, out_len), spec.device())
}

/// Hann window of length `n` matching `torch.hann_window(n,
/// periodic=True)` (the default upstream Vocos uses). Periodic Hann
/// has `w[k] = 0.5 * (1 - cos(2 * pi * k / n))`.
fn hann_window(n: usize) -> Vec<f32> {
    let mut w = vec![0f32; n];
    if n <= 1 {
        if n == 1 {
            w[0] = 1.0;
        }
        return w;
    }
    let denom = n as f32;
    for (k, slot) in w.iter_mut().enumerate() {
        *slot = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / denom).cos());
    }
    w
}

/// Overlap-add the squared Hann window across `t_frames` frames at
/// `hop_length` stride into a buffer of length `full_len`. This is
/// what `vocos/spectral_ops.py:ISTFT` uses to normalise the output —
/// matches the COLA-compliance trick noted in the upstream code.
fn window_envelope(
    window: &[f32],
    t_frames: usize,
    hop_length: usize,
    full_len: usize,
) -> Vec<f32> {
    let win_len = window.len();
    let mut env = vec![0f32; full_len];
    for ti in 0..t_frames {
        let frame_start = ti * hop_length;
        for s in 0..win_len {
            env[frame_start + s] += window[s] * window[s];
        }
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_cfg() -> VocosConfig {
        // A miniature config that exercises every code path without
        // melting the test runtime.
        VocosConfig {
            mel_dim: 4,
            hidden_dim: 8,
            n_layers: 2,
            intermediate_dim: 16,
            n_fft: 16,
            hop_length: 4,
            win_length: 16,
            sample_rate: 24_000,
        }
    }

    fn make_vb<'a>(varmap: &'a VarMap, device: &'a Device) -> VarBuilder<'a> {
        VarBuilder::from_varmap(varmap, DType::F32, device)
    }

    #[test]
    fn vocos_24khz_config_matches_upstream() {
        let cfg = VocosConfig::vocos_24khz();
        assert_eq!(cfg.mel_dim, 100);
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.n_layers, 8);
        assert_eq!(cfg.intermediate_dim, 1536);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.hop_length, 256);
        assert_eq!(cfg.win_length, 1024);
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.n_bins(), 513);
        // 1 / num_layers — matches vocos/modules.py.
        assert!((cfg.layer_scale_init() - 0.125).abs() < 1e-9);
    }

    #[test]
    fn convnext_block_forward_shape_correct() {
        let device = cpu();
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = make_vb(&varmap, &device);
        let block = ConvNeXtBlock::load(vb.pp("block0"), &cfg).expect("load block");

        let b = 2;
        let t = 5;
        let x = Tensor::randn(0f32, 1.0f32, (b, cfg.hidden_dim, t), &device).expect("randn");
        let y = block.forward(&x).expect("block forward");
        assert_eq!(y.dims(), &[b, cfg.hidden_dim, t]);
    }

    #[test]
    fn vocos_backbone_forward_shape_correct() {
        let device = cpu();
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = make_vb(&varmap, &device);
        let backbone = VocosBackbone::load(vb, cfg.clone()).expect("load backbone");

        let b = 1;
        let t = 6;
        let mel = Tensor::randn(0f32, 1.0f32, (b, cfg.mel_dim, t), &device).expect("randn");
        let out = backbone.forward(&mel).expect("backbone forward");
        assert_eq!(out.dims(), &[b, cfg.n_fft + 2, t]);
    }

    #[test]
    fn istft_round_trips_silence() {
        let device = cpu();
        let cfg = tiny_cfg();
        let n_bins = cfg.n_bins();
        let t = 5;
        // Magnitudes (first half) are log-scale: `exp(very_negative)`
        // ≈ 0, so the resulting complex spectrum (and waveform) is
        // numerically near-zero everywhere.
        let mut data = vec![0f32; 2 * n_bins * t];
        for v in data.iter_mut().take(n_bins * t) {
            *v = -20.0;
        }
        let spec = Tensor::from_vec(data, (1, 2 * n_bins, t), &device).expect("spec");
        let wav = istft(&spec, &cfg).expect("istft");
        let (b, n) = wav.dims2().expect("dims2");
        assert_eq!(b, 1);
        let expected =
            (t - 1) * cfg.hop_length + cfg.win_length - 2 * ((cfg.win_length - cfg.hop_length) / 2);
        assert_eq!(n, expected);
        let v = wav.to_vec2::<f32>().expect("to_vec2");
        for sample in &v[0] {
            assert!(sample.abs() < 1e-4, "expected silence, got {sample}");
        }
    }

    #[test]
    fn istft_overlap_add_window_normalization() {
        // Feed a constant-magnitude, zero-phase spectrum and confirm
        // the iSTFT recovers a flat, non-zero signal — i.e. the
        // window-envelope normalisation cancels the Hann tapering
        // (this is the COLA property Vocos relies on for clean
        // reconstruction).
        let device = cpu();
        let cfg = tiny_cfg();
        let n_bins = cfg.n_bins();
        let t = 5;

        // mag = log(1) = 0 -> exp(0) = 1 across every bin; phase = 0.
        // Concretely the time-domain frame is a unit impulse at s=0
        // (the inverse DFT of an all-ones spectrum). After Hann
        // weighting + overlap-add, dividing by the squared-window
        // envelope produces the same impulse pattern at each hop —
        // but crucially the *non-zero* output samples are stable
        // (not NaN/inf) and finite.
        let data = vec![0f32; 2 * n_bins * t];
        let spec = Tensor::from_vec(data, (1, 2 * n_bins, t), &device).expect("spec");
        let wav = istft(&spec, &cfg).expect("istft");
        let v = wav.to_vec2::<f32>().expect("to_vec2");
        // Every sample must be finite (no division-by-zero leakage
        // through the envelope normalisation).
        for sample in &v[0] {
            assert!(sample.is_finite(), "non-finite sample: {sample}");
        }

        // Also sanity-check the envelope helper directly: for a
        // periodic Hann window with hop = win/4 (here 4/16) the
        // overlap-add of `window^2` is constant over the central
        // region. Pick a sample well away from the edges to avoid
        // ramp-up/ramp-down frames.
        let window = hann_window(cfg.win_length);
        let env = window_envelope(
            &window,
            t,
            cfg.hop_length,
            (t - 1) * cfg.hop_length + cfg.win_length,
        );
        let mid = env.len() / 2;
        // Periodic-Hann COLA at hop = win/4: in the steady-state
        // region each output sample is covered by exactly four
        // overlapping windows, and the mean of `w^2` for a periodic
        // Hann window is `3/8`. That gives `4 * 3/8 = 1.5` for the
        // squared-window envelope sum — a finite, well-defined
        // constant used as the iSTFT normaliser.
        assert!(
            (env[mid] - 1.5).abs() < 1e-5,
            "envelope mid = {} (want ~1.5)",
            env[mid],
        );
    }
}
