//! NSF-HiFi-GAN generator for the RVC voice-conversion backend.
//!
//! RVC v2 drives synthesis with a HiFi-GAN-style generator augmented with
//! a [neural source-filter] (NSF) front-end. The source module turns the
//! per-frame Hz contour from [`super::f0`] into a harmonic + noise
//! excitation at the output sample rate; the HiFi-GAN backbone then
//! upsamples the per-frame content embedding from [`super::content`],
//! conditioned on a speaker embedding, while injecting downsampled
//! copies of the source signal at every upsampling stage. The final
//! tanh-bounded 1-channel convolution yields mono PCM in `[-1, 1]`.
//!
//! Reference: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/lib/infer_pack/models.py>
//! (classes `SineGen`, `SourceModuleHnNSF`, `GeneratorNSF`).
//!
//! # Default config (`default_v2_40khz`)
//!
//! Matches the canonical RVC v2 40 kHz checkpoint:
//!
//! - `hidden_dim = 768` (HuBERT-large content width)
//! - `n_speakers = 109` (single-speaker checkpoints still allocate the
//!   full 109-slot embedding table for ABI parity with multi-speaker
//!   training runs).
//! - `pitch_n_bins = 256` (matches [`super::f0::PITCH_COARSE_BINS`])
//! - `output_sample_rate_hz = 40_000`
//! - `upsample_rates = [10, 10, 2, 2]` → total upsample 400 (= the
//!   40 kHz hop length at the 100 Hz frame rate the content encoder
//!   emits).
//! - `upsample_kernel_sizes = [16, 16, 4, 4]`
//! - `upsample_initial_channel = 512`
//! - `resblock_kernel_sizes = [3, 7, 11]`
//! - `resblock_dilation_sizes = [[1, 3, 5]; 3]`
//!
//! # Weight loading
//!
//! Weights are pulled from a [`candle_nn::VarBuilder`] rooted at the
//! generator's namespace. Path layout matches the upstream `PyTorch`
//! parameter names so a converted safetensors export drops straight in:
//!
//! ```text
//! emb_pitch                 (pitch_n_bins, hidden_dim)
//! emb_g                     (n_speakers, gin_channels=256)
//! conv_pre                  Conv1d(hidden_dim -> upsample_initial_channel, k=7, p=3)
//! m_source.l_linear         Linear(harmonic_num + 1 -> 1)
//! noise_convs.{i}           Conv1d(1 -> c_out_i, k=stride, stride=stride)
//! cond.{i}                  Conv1d(gin_channels -> c_out_i, k=1)
//! ups.{i}                   ConvTranspose1d(c_in_i -> c_out_i, k=ksz_i, stride=rate_i)
//! resblocks.{j}.{convs1,convs2}.{k}  (j over MRF resblocks, k over dilations)
//! conv_post                 Conv1d(last_c_out -> 1, k=7, p=3)
//! ```
//!
//! Weight-norm wrappers are baked into the released RVC checkpoints
//! (weight + bias are already the post-normalisation tensors) so no
//! runtime renormalisation is needed.
//!
//! [neural source-filter]: https://arxiv.org/abs/1904.12088

#![cfg(feature = "rvc")]
// candle's `VarBuilder` is the recommended construction pattern -- every
// `.pp(...)` returns a fresh child, so passing the outer builder by
// value matches the rest of the crate.
#![allow(clippy::needless_pass_by_value)]
// `SineGen` / window / pitch math live in floating-point land where
// the tiny precision loss from `usize as f32` is expected.
#![allow(clippy::cast_precision_loss)]
// Per-stage book-keeping in `synthesize` / `load_from_var_builder`
// stays clearer with short names (`x`, `g`, `b`, etc.) that match
// upstream notation.
#![allow(clippy::similar_names, clippy::many_single_char_names)]
// The forward / loader read top-to-bottom and splitting them up just
// to chase the heuristic line cap hurts readability.
#![allow(clippy::too_many_lines)]

use candle_core::{D, Device, IndexOp, Module, Result as CResult, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, VarBuilder,
    conv_transpose1d, conv1d, embedding, ops::leaky_relu,
};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Speaker-embedding (`gin_channels`) width used by RVC v2. Always 256
/// regardless of how many speakers the checkpoint was trained on; the
/// per-speaker variation lives in [`NsfHifiGanConfig::n_speakers`].
pub const GIN_CHANNELS: usize = 256;

/// Number of harmonics the `SineGen` emits in addition to the fundamental.
/// RVC v2 hard-codes this to 8 (`harmonic_num = 8`), so the linear
/// source mixer takes `harmonic_num + 1 = 9` input channels.
pub const HARMONIC_NUM: usize = 8;

/// `LeakyReLU` negative slope shared across the generator. Upstream RVC
/// uses 0.1 everywhere (`LRELU_SLOPE = 0.1`).
const LRELU_SLOPE: f64 = 0.1;

/// Source-noise amplitude on unvoiced frames. Matches
/// `SineGen.noise_std` in the upstream reference.
const SINE_NOISE_STD: f32 = 0.003;

/// Sine amplitude on voiced frames. Matches `SineGen.sine_amp` upstream.
const SINE_AMP: f32 = 0.1;

/// Voiced threshold (Hz) -- frames with `pitch_hz <= VOICED_THRESHOLD`
/// are treated as unvoiced (sine harmonics zeroed; noise still
/// injected).
const VOICED_THRESHOLD: f32 = 10.0;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`NsfHifiGan`].
#[derive(Debug, Error)]
pub enum GeneratorError {
    /// Weight load / [`VarBuilder`] resolution failed.
    #[error("nsf-hifigan load: {0}")]
    ModelLoad(String),

    /// Forward pass failed (shape mismatch, etc.).
    #[error("nsf-hifigan inference: {0}")]
    Inference(String),

    /// Wrapper for IO errors that may bubble up from caller-side
    /// helpers (kept on the surface so the pipeline can collapse all
    /// generator-side failures into one variant).
    #[error("nsf-hifigan io: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapper for raw candle errors -- keeps the `?` ergonomics tidy
    /// for callers that want to forward a candle failure verbatim.
    #[error("nsf-hifigan candle: {0}")]
    Candle(#[from] candle_core::Error),
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`NsfHifiGan`].
#[derive(Debug, Clone)]
pub struct NsfHifiGanConfig {
    /// Content-embedding width arriving from
    /// [`super::content::ContentEncoder`]. RVC v2 emits 768-d `HuBERT`
    /// features.
    pub hidden_dim: usize,
    /// Number of speaker slots in the embedding table. RVC v2 trains
    /// with 109 slots even for single-speaker checkpoints.
    pub n_speakers: usize,
    /// Number of coarse pitch bins. Mirrors
    /// [`super::f0::PITCH_COARSE_BINS`].
    pub pitch_n_bins: usize,
    /// Output sample rate in Hz.
    pub output_sample_rate_hz: u32,
    /// Per-stage upsample strides. Product equals the total upsample
    /// factor between frame-rate features and audio-rate samples.
    pub upsample_rates: Vec<usize>,
    /// Per-stage `ConvTranspose1d` kernel sizes.
    pub upsample_kernel_sizes: Vec<usize>,
    /// Channel width entering the first upsample stage (= channels
    /// emitted by `conv_pre`). Each subsequent stage halves the width.
    pub upsample_initial_channel: usize,
    /// Kernel sizes of the parallel MRF resblocks (one resblock per
    /// kernel). Default `[3, 7, 11]`.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilation schedule for each MRF resblock.
    /// `resblock_dilation_sizes[j]` is consumed by resblock `j`; each
    /// entry is a list of dilations applied in series within that
    /// resblock.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
}

impl NsfHifiGanConfig {
    /// Defaults matching the upstream RVC v2 40 kHz checkpoint.
    #[must_use]
    pub fn default_v2_40khz() -> Self {
        Self {
            hidden_dim: 768,
            n_speakers: 109,
            pitch_n_bins: 256,
            output_sample_rate_hz: 40_000,
            upsample_rates: vec![10, 10, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        }
    }

    /// Total upsample factor (= product of `upsample_rates`).
    #[must_use]
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    /// Sanity-check the invariants used by the forward pass.
    fn validate(&self) -> Result<(), GeneratorError> {
        if self.upsample_rates.is_empty() {
            return Err(GeneratorError::ModelLoad(
                "upsample_rates must be non-empty".into(),
            ));
        }
        if self.upsample_rates.len() != self.upsample_kernel_sizes.len() {
            return Err(GeneratorError::ModelLoad(format!(
                "upsample_rates len {} != upsample_kernel_sizes len {}",
                self.upsample_rates.len(),
                self.upsample_kernel_sizes.len(),
            )));
        }
        if self.resblock_kernel_sizes.is_empty() {
            return Err(GeneratorError::ModelLoad(
                "resblock_kernel_sizes must be non-empty".into(),
            ));
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(GeneratorError::ModelLoad(format!(
                "resblock_kernel_sizes len {} != resblock_dilation_sizes len {}",
                self.resblock_kernel_sizes.len(),
                self.resblock_dilation_sizes.len(),
            )));
        }
        if self.upsample_initial_channel == 0 {
            return Err(GeneratorError::ModelLoad(
                "upsample_initial_channel must be > 0".into(),
            ));
        }
        // Each upsample stage halves the channel width; make sure we
        // never bottom out below 1 channel.
        let final_ch = self.stage_out_channels(self.upsample_rates.len() - 1);
        if final_ch == 0 {
            return Err(GeneratorError::ModelLoad(format!(
                "upsample_initial_channel {} is too small for {} stages \
                 (would reach 0 channels)",
                self.upsample_initial_channel,
                self.upsample_rates.len()
            )));
        }
        Ok(())
    }

    /// Channel width entering the `i`-th upsample stage.
    fn stage_in_channels(&self, i: usize) -> usize {
        let shift = u32::try_from(i).unwrap_or(u32::MAX);
        self.upsample_initial_channel
            .checked_shr(shift)
            .unwrap_or(0)
    }

    /// Channel width exiting the `i`-th upsample stage.
    fn stage_out_channels(&self, i: usize) -> usize {
        let shift = u32::try_from(i + 1).unwrap_or(u32::MAX);
        self.upsample_initial_channel
            .checked_shr(shift)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// SineGen
// ---------------------------------------------------------------------------

/// Sine-harmonic + noise excitation generator.
///
/// Given a per-frame Hz contour, [`SineGen::forward`] emits a
/// `(B, HARMONIC_NUM + 1, n_samples)` tensor where channel `k` is a
/// sine wave at `(k + 1) * f0` Hz on voiced frames and Gaussian noise
/// on unvoiced frames. The harmonic phase is built by cumulative-sum
/// over the upsampled (frame -> audio) pitch trace, which is the same
/// trick the upstream Python uses (modulo the unwrap/wrap to
/// `[-pi, pi]`, which we skip because `sin` is `2*pi`-periodic).
#[derive(Debug, Clone)]
struct SineGen {
    sample_rate: u32,
    harmonic_num: usize,
    total_upsample: usize,
}

impl SineGen {
    const fn new(sample_rate: u32, harmonic_num: usize, total_upsample: usize) -> Self {
        Self {
            sample_rate,
            harmonic_num,
            total_upsample,
        }
    }

    /// Forward pass.
    ///
    /// `pitch_hz`: `(B, n_frames)` f32. Returns
    /// `(B, harmonic_num + 1, n_frames * total_upsample)` f32 on the
    /// same device as `pitch_hz`.
    fn forward(&self, pitch_hz: &Tensor) -> CResult<Tensor> {
        let (b, n_frames) = pitch_hz.dims2()?;
        let device = pitch_hz.device();
        let dtype = pitch_hz.dtype();
        let total_up = self.total_upsample;
        let n_samples = n_frames * total_up;

        // Upsample the frame-rate pitch to audio rate via
        // nearest-neighbour interpolation. interpolate1d expects
        // (N, C, L); we treat the batch dim as N and add a singleton
        // channel.
        let f0_audio = pitch_hz
            .unsqueeze(1)? // (B, 1, n_frames)
            .interpolate1d(n_samples)? // (B, 1, n_samples)
            .squeeze(1)?; // (B, n_samples)

        // Voiced mask (1.0 voiced, 0.0 unvoiced) at audio rate.
        let voiced_mask = f0_audio.gt(VOICED_THRESHOLD)?.to_dtype(dtype)?;

        // For each harmonic k in 0..=harmonic_num compute the
        // instantaneous frequency f0 * (k+1) / sample_rate, cumsum,
        // then `sin(2*pi * phase)`.
        let sr = self.sample_rate.max(1) as f32;
        let two_pi = 2.0_f32 * std::f32::consts::PI;

        let mut channels: Vec<Tensor> = Vec::with_capacity(self.harmonic_num + 1);
        for k in 0..=self.harmonic_num {
            let factor = ((k as f32) + 1.0) / sr;
            let inst_freq = (&f0_audio * f64::from(factor))?; // (B, n_samples)
            let phase = inst_freq.cumsum(D::Minus1)?; // (B, n_samples)
            let arg = (&phase * f64::from(two_pi))?;
            let sine = arg.sin()?;
            // Apply voiced mask + sine amp.
            let sine = sine.broadcast_mul(&voiced_mask)?;
            let sine = (&sine * f64::from(SINE_AMP))?;
            channels.push(sine.unsqueeze(1)?); // (B, 1, n_samples)
        }
        let harmonics = Tensor::cat(&channels, 1)?; // (B, H+1, n_samples)

        // Noise on unvoiced regions: scale a standard normal by
        // SINE_NOISE_STD where voiced_mask == 0, and by SINE_AMP/3
        // where voiced_mask == 1 (matches upstream -- `sine_amp / 3`
        // for the voiced noise floor).
        let noise = Tensor::randn(
            0_f32,
            1.0_f32,
            (b, self.harmonic_num + 1, n_samples),
            device,
        )?
        .to_dtype(dtype)?;
        let ones = Tensor::ones_like(&voiced_mask)?;
        let unvoiced_mask = (&ones - &voiced_mask)?; // (B, n_samples)
        let unvoiced_mask_b = unvoiced_mask.unsqueeze(1)?; // (B, 1, n_samples)
        let voiced_mask_b = voiced_mask.unsqueeze(1)?; // (B, 1, n_samples)
        let unvoiced_amp = (&unvoiced_mask_b * f64::from(SINE_NOISE_STD))?;
        let voiced_amp = (&voiced_mask_b * f64::from(SINE_AMP / 3.0))?;
        let amp = (&unvoiced_amp + &voiced_amp)?; // (B, 1, n_samples)
        let noise_scaled = noise.broadcast_mul(&amp)?;

        // On unvoiced regions the harmonics are already zeroed by the
        // voiced mask, so adding noise gives the final excitation.
        &harmonics + &noise_scaled
    }
}

// ---------------------------------------------------------------------------
// MRF / ResBlock1
// ---------------------------------------------------------------------------

/// HiFi-GAN "`ResBlock1`" -- two parallel conv stacks per dilation, each
/// with a leaky-relu pre-activation and a residual add.
///
/// Layout (matching upstream `models.py:ResBlock1`):
///
/// ```text
/// for d in dilations:
///     h1 = leaky_relu(x)
///     h1 = convs1[d](h1)   # Conv1d(C, C, k, dilation=d, padding=(k-1)*d/2)
///     h2 = leaky_relu(h1)
///     h2 = convs2[d](h2)   # Conv1d(C, C, k, dilation=1, padding=(k-1)/2)
///     x  = x + h2
/// ```
#[derive(Debug, Clone)]
struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
    ) -> CResult<Self> {
        let half_k = kernel_size.saturating_sub(1) / 2;
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            // convs1: dilated, padding = (k-1) * d / 2 (per upstream).
            let padding1 = kernel_size.saturating_sub(1) * d / 2;
            let cfg1 = Conv1dConfig {
                padding: padding1,
                stride: 1,
                dilation: d,
                groups: 1,
                ..Default::default()
            };
            convs1.push(conv1d(
                channels,
                channels,
                kernel_size,
                cfg1,
                vb.pp("convs1").pp(i.to_string()),
            )?);
            // convs2: dilation 1, padding = (k-1)/2.
            let cfg2 = Conv1dConfig {
                padding: half_k,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            convs2.push(conv1d(
                channels,
                channels,
                kernel_size,
                cfg2,
                vb.pp("convs2").pp(i.to_string()),
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut h = x.clone();
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let a = leaky_relu(&h, LRELU_SLOPE)?;
            let a = c1.forward(&a)?;
            let a = leaky_relu(&a, LRELU_SLOPE)?;
            let a = c2.forward(&a)?;
            h = (&h + &a)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// NSF-HiFi-GAN generator. See module docs for the high-level pass.
pub struct NsfHifiGan {
    cfg: NsfHifiGanConfig,
    device: Device,

    // Conditioning embeddings.
    emb_pitch: Embedding,
    emb_g: Embedding,

    // Input projection.
    conv_pre: Conv1d,

    // Source module: SineGen + 1x1 mixer.
    source: SineGen,
    // 1x1 Conv1d so the mixer can run in channels-first layout
    // alongside the rest of the generator.
    source_linear: Conv1d,

    // Per-stage upsampling.
    ups: Vec<ConvTranspose1d>,
    noise_convs: Vec<Conv1d>,
    cond_convs: Vec<Conv1d>,

    // MRF: one Vec<ResBlock1> per upsample stage; each Vec has
    // `resblock_kernel_sizes.len()` resblocks.
    resblocks: Vec<Vec<ResBlock1>>,

    // Final 1-channel projection.
    conv_post: Conv1d,
}

impl std::fmt::Debug for NsfHifiGan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NsfHifiGan")
            .field("output_sample_rate_hz", &self.cfg.output_sample_rate_hz)
            .field("total_upsample", &self.cfg.total_upsample())
            .field("n_stages", &self.ups.len())
            .finish_non_exhaustive()
    }
}

impl NsfHifiGan {
    /// Construct the generator from a [`VarBuilder`] rooted at the
    /// generator namespace.
    ///
    /// # Errors
    ///
    /// Returns [`GeneratorError::ModelLoad`] if the config is internally
    /// inconsistent or if any weight tensor is missing / mis-shaped, and
    /// [`GeneratorError::Candle`] for raw candle failures bubbled up
    /// from layer constructors.
    pub fn load_from_var_builder(
        vb: VarBuilder,
        device: &Device,
        config: NsfHifiGanConfig,
    ) -> Result<Self, GeneratorError> {
        config.validate()?;

        let emb_pitch = embedding(config.pitch_n_bins, config.hidden_dim, vb.pp("emb_pitch"))?;
        let emb_g = embedding(config.n_speakers, GIN_CHANNELS, vb.pp("emb_g"))?;

        // conv_pre: hidden_dim -> upsample_initial_channel, k=7, p=3.
        let conv_pre = conv1d(
            config.hidden_dim,
            config.upsample_initial_channel,
            7,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;

        // SineGen + 1x1 linear mixer over the (HARMONIC_NUM + 1) channels.
        let source = SineGen::new(
            config.output_sample_rate_hz,
            HARMONIC_NUM,
            config.total_upsample(),
        );
        let source_linear = conv1d(
            HARMONIC_NUM + 1,
            1,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("m_source").pp("l_linear"),
        )?;

        // Per-stage upsample / noise-conv / speaker-cond / MRF.
        let n_stages = config.upsample_rates.len();
        let mut ups = Vec::with_capacity(n_stages);
        let mut noise_convs = Vec::with_capacity(n_stages);
        let mut cond_convs = Vec::with_capacity(n_stages);
        let mut resblocks: Vec<Vec<ResBlock1>> = Vec::with_capacity(n_stages);

        // `noise_stride_remaining` tracks how much downsampling the
        // source signal still needs to match the *current* stage's
        // audio rate. After stage `i`, the source has been downsampled
        // by `prod(upsample_rates[i+1..])`.
        let mut noise_stride_remaining: usize = config.total_upsample();
        for i in 0..n_stages {
            let c_in = config.stage_in_channels(i);
            let c_out = config.stage_out_channels(i);
            let rate = config.upsample_rates[i];
            let ksz = config.upsample_kernel_sizes[i];
            // ConvTranspose1d padding follows the upstream convention
            // `(k - rate) / 2`. Falls back to 0 when k < rate (the
            // default config keeps k >= rate everywhere).
            let padding = ksz.saturating_sub(rate) / 2;
            let up_cfg = ConvTranspose1dConfig {
                padding,
                output_padding: 0,
                stride: rate,
                dilation: 1,
                groups: 1,
            };
            ups.push(conv_transpose1d(
                c_in,
                c_out,
                ksz,
                up_cfg,
                vb.pp("ups").pp(i.to_string()),
            )?);

            // After applying `ups[i]`, the source still needs to be
            // downsampled by `prod(upsample_rates[i+1..])` to match
            // the post-upsample audio rate. Update before consuming.
            noise_stride_remaining /= rate;
            // Source -> stage rate: Conv1d(1 -> c_out, k=stride,
            // stride=stride). When the remaining stride is 1 (final
            // stage), use a 1x1 identity-style conv so the source
            // feeds in at full rate.
            let (nk, ns, np) = if noise_stride_remaining > 1 {
                (
                    noise_stride_remaining * 2,
                    noise_stride_remaining,
                    noise_stride_remaining / 2,
                )
            } else {
                (1, 1, 0)
            };
            let nc_cfg = Conv1dConfig {
                padding: np,
                stride: ns,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            noise_convs.push(conv1d(
                1,
                c_out,
                nk,
                nc_cfg,
                vb.pp("noise_convs").pp(i.to_string()),
            )?);

            // Speaker conditioning: 1x1 Conv1d(GIN_CHANNELS -> c_out).
            cond_convs.push(conv1d(
                GIN_CHANNELS,
                c_out,
                1,
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    ..Default::default()
                },
                vb.pp("cond").pp(i.to_string()),
            )?);

            // MRF: one ResBlock1 per (kernel, dilation-list) pair.
            // Indexed flat in the upstream checkpoint as
            // `resblocks.{i * n_kernels + j}` so the index unrolls
            // across both axes.
            let mut stage_blocks = Vec::with_capacity(config.resblock_kernel_sizes.len());
            for (j, (&ksize, dils)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let flat_ix = i * config.resblock_kernel_sizes.len() + j;
                stage_blocks.push(ResBlock1::load(
                    vb.pp("resblocks").pp(flat_ix.to_string()),
                    c_out,
                    ksize,
                    dils,
                )?);
            }
            resblocks.push(stage_blocks);
        }

        // conv_post: last_c_out -> 1, k=7, p=3.
        let last_c_out = config.stage_out_channels(n_stages - 1);
        let conv_post = conv1d(
            last_c_out,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            cfg: config,
            device: device.clone(),
            emb_pitch,
            emb_g,
            conv_pre,
            source,
            source_linear,
            ups,
            noise_convs,
            cond_convs,
            resblocks,
            conv_post,
        })
    }

    /// Output sample rate in Hz.
    #[must_use]
    pub const fn output_sample_rate_hz(&self) -> u32 {
        self.cfg.output_sample_rate_hz
    }

    /// Read-only access to the config.
    #[must_use]
    pub const fn config(&self) -> &NsfHifiGanConfig {
        &self.cfg
    }

    /// Borrow the device the generator was built on. Useful for the
    /// pipeline driver, which threads the same device through every
    /// component.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Run the full forward pass and return mono PCM `f32` samples in
    /// `[-1, 1]`.
    ///
    /// # Arguments
    ///
    /// - `content`: `(1, hidden_dim, n_frames)` f32 channels-first.
    /// - `pitch_coarse`: `(1, n_frames)` i64 coarse pitch bins.
    /// - `pitch_hz`: `(1, n_frames)` f32 Hz contour (0.0 = unvoiced).
    /// - `speaker_id`: `(1,)` i64.
    ///
    /// Output length is approximately `n_frames * total_upsample`.
    ///
    /// # Errors
    ///
    /// Returns [`GeneratorError::Inference`] on shape mismatches and
    /// [`GeneratorError::Candle`] on raw candle failures.
    pub fn synthesize(
        &self,
        content: &Tensor,
        pitch_coarse: &Tensor,
        pitch_hz: &Tensor,
        speaker_id: &Tensor,
    ) -> Result<Vec<f32>, GeneratorError> {
        // ---- shape sanity ----
        let (b_c, hid, n_frames) = content.dims3()?;
        if hid != self.cfg.hidden_dim {
            return Err(GeneratorError::Inference(format!(
                "content hidden dim {hid} != configured {}",
                self.cfg.hidden_dim
            )));
        }
        let (b_pc, n_frames_pc) = pitch_coarse.dims2()?;
        let (b_ph, n_frames_ph) = pitch_hz.dims2()?;
        if b_c != b_pc || b_c != b_ph {
            return Err(GeneratorError::Inference(format!(
                "batch mismatch: content {b_c}, pitch_coarse {b_pc}, pitch_hz {b_ph}"
            )));
        }
        if n_frames != n_frames_pc || n_frames != n_frames_ph {
            return Err(GeneratorError::Inference(format!(
                "frame-count mismatch: content {n_frames}, pitch_coarse {n_frames_pc}, \
                 pitch_hz {n_frames_ph}"
            )));
        }

        // ---- pitch embed + content fuse ----
        // emb_pitch: (B, T) i64 -> (B, T, hidden_dim) -> (B, hidden_dim, T).
        let pitch_emb = self.emb_pitch.forward(pitch_coarse)?; // (B, T, H)
        let pitch_emb = pitch_emb.transpose(1, 2)?; // (B, H, T)
        let x = (content + &pitch_emb)?;

        // ---- conv_pre ----
        let mut x = self.conv_pre.forward(&x)?; // (B, C0, T)

        // ---- speaker embed ----
        // emb_g: (B,) i64 -> (B, GIN_CHANNELS) -> (B, GIN_CHANNELS, 1).
        let g = self.emb_g.forward(speaker_id)?; // (B, GIN)
        let g = g.unsqueeze(2)?; // (B, GIN, 1)

        // ---- source excitation ----
        let harmonics = self.source.forward(pitch_hz)?; // (B, H+1, n_samples)
        // 1x1 mix down to (B, 1, n_samples) so per-stage `noise_convs`
        // can consume a single-channel source.
        let source_wave = self.source_linear.forward(&harmonics)?;
        let source_wave = source_wave.tanh()?;

        // ---- upsampling stages ----
        for i in 0..self.ups.len() {
            // pre-activation + upsample.
            x = leaky_relu(&x, LRELU_SLOPE)?;
            x = self.ups[i].forward(&x)?;
            // Source -> stage rate (Conv1d w/ stride matches the
            // remaining downsample factor for this stage).
            let src_i = self.noise_convs[i].forward(&source_wave)?; // (B, c_out, T_i')
            // Align the source temporal length to `x`'s (the
            // ConvTranspose1d + the source-side stride conv can
            // disagree by a few samples on the edges; we trim to the
            // shorter length so the broadcast add is well-defined).
            let (_, _, t_x) = x.dims3()?;
            let (_, _, t_s) = src_i.dims3()?;
            let t = t_x.min(t_s);
            let x_trim = x.narrow(2, 0, t)?;
            let src_trim = src_i.narrow(2, 0, t)?;
            x = (x_trim + src_trim)?;
            // Speaker conditioning (broadcast over time).
            let cond = self.cond_convs[i].forward(&g)?; // (B, c_out, 1)
            x = x.broadcast_add(&cond)?;
            // MRF: sum the parallel resblock outputs and divide by
            // their count (upstream uses arithmetic mean across the
            // parallel kernels).
            let stage_blocks = &self.resblocks[i];
            let mut acc: Option<Tensor> = None;
            for rb in stage_blocks {
                let y = rb.forward(&x)?;
                acc = Some(match acc {
                    Some(a) => (a + y)?,
                    None => y,
                });
            }
            let summed = acc.ok_or_else(|| {
                GeneratorError::Inference("MRF stage produced no resblocks".into())
            })?;
            let denom = stage_blocks.len() as f64;
            x = (summed / denom)?;
        }

        // ---- final activation + conv_post + tanh ----
        x = leaky_relu(&x, LRELU_SLOPE)?;
        let x = self.conv_post.forward(&x)?; // (B, 1, T)
        let x = x.tanh()?; // bound to [-1, 1]

        // (B, 1, T) -> Vec<f32> (B == 1 by construction).
        let (_, _, t) = x.dims3()?;
        let samples = x
            .i((0, 0, ..))?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        if samples.len() != t {
            return Err(GeneratorError::Inference(format!(
                "output sample count {} != tensor time dim {}",
                samples.len(),
                t
            )));
        }
        Ok(samples)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use candle_nn::VarMap;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn config_default_v2_40khz_upsample_math() {
        let cfg = NsfHifiGanConfig::default_v2_40khz();
        assert_eq!(cfg.upsample_rates, vec![10, 10, 2, 2]);
        // 10 * 10 * 2 * 2 = 400 -- the canonical hop length matching a
        // 40 kHz output rate at the content encoder's 100 Hz frame
        // rate.
        assert_eq!(cfg.total_upsample(), 400);
        assert_eq!(cfg.output_sample_rate_hz, 40_000);
        assert_eq!(cfg.upsample_initial_channel, 512);
        // Each stage halves channels: 512 -> 256 -> 128 -> 64 -> 32.
        assert_eq!(cfg.stage_in_channels(0), 512);
        assert_eq!(cfg.stage_out_channels(0), 256);
        assert_eq!(cfg.stage_out_channels(1), 128);
        assert_eq!(cfg.stage_out_channels(2), 64);
        assert_eq!(cfg.stage_out_channels(3), 32);
        // Validation should accept the default.
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validation_catches_mismatched_lengths() {
        let mut cfg = NsfHifiGanConfig::default_v2_40khz();
        cfg.upsample_kernel_sizes.pop();
        let err = cfg.validate().expect_err("should fail");
        match err {
            GeneratorError::ModelLoad(_) => {}
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn sine_gen_shape() {
        let device = cpu();
        // Total upsample 4 keeps the test fast; we only care that the
        // SineGen forward emits the right shape on the right device.
        let total_up = 4_usize;
        let sg = SineGen::new(16_000, HARMONIC_NUM, total_up);
        let n_frames = 5_usize;
        // Mixed voiced/unvoiced pitch to exercise both code paths.
        let pitch = Tensor::from_vec(
            vec![0.0_f32, 110.0, 220.0, 0.0, 440.0],
            (1, n_frames),
            &device,
        )
        .expect("pitch tensor");
        let out = sg.forward(&pitch).expect("sine gen forward");
        assert_eq!(out.dims(), &[1, HARMONIC_NUM + 1, n_frames * total_up]);
        // All samples should be finite (no NaN/inf leakage from the
        // cumsum-based phase computation).
        let flat = out
            .reshape(((HARMONIC_NUM + 1) * n_frames * total_up,))
            .and_then(|t| t.to_vec1::<f32>())
            .expect("flatten");
        for v in flat {
            assert!(v.is_finite(), "non-finite SineGen sample: {v}");
        }
    }

    #[test]
    fn resblock1_forward_shape_preserved() {
        let device = cpu();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let rb = ResBlock1::load(vb.pp("rb"), 8, 3, &[1, 3, 5]).expect("resblock load");
        let x = Tensor::randn(0_f32, 1.0_f32, (1, 8, 16), &device).expect("randn");
        let y = rb.forward(&x).expect("resblock forward");
        assert_eq!(y.dims(), &[1, 8, 16]);
    }

    /// Build a miniature generator config small enough to instantiate
    /// in tests but still exercising every code path (multi-stage
    /// upsample, MRF, source-side conditioning, speaker conditioning).
    fn tiny_cfg() -> NsfHifiGanConfig {
        NsfHifiGanConfig {
            hidden_dim: 16,
            n_speakers: 4,
            pitch_n_bins: 8,
            output_sample_rate_hz: 8_000,
            upsample_rates: vec![4, 2],
            upsample_kernel_sizes: vec![8, 4],
            upsample_initial_channel: 16, // -> 16, 8, 4 (final c_out = 4)
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
        }
    }

    #[test]
    fn nsf_hifigan_synthesize_shape_and_range() {
        let device = cpu();
        let cfg = tiny_cfg();
        let total_up = cfg.total_upsample();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model =
            NsfHifiGan::load_from_var_builder(vb, &device, cfg.clone()).expect("load generator");

        let n_frames = 3;
        let content =
            Tensor::randn(0_f32, 1.0_f32, (1, cfg.hidden_dim, n_frames), &device).expect("content");
        let pitch_coarse =
            Tensor::from_vec(vec![1_i64, 3, 5], (1, n_frames), &device).expect("pitch_coarse");
        let pitch_hz = Tensor::from_vec(vec![110.0_f32, 220.0, 0.0], (1, n_frames), &device)
            .expect("pitch_hz");
        let speaker_id = Tensor::from_vec(vec![0_i64], (1,), &device).expect("speaker_id");

        let samples = model
            .synthesize(&content, &pitch_coarse, &pitch_hz, &speaker_id)
            .expect("synthesize");
        // The trim-to-min step in synthesize() can shave a few samples
        // off the nominal `n_frames * total_up` length depending on the
        // ConvTranspose1d's output formula; assert we're within one
        // upsample window of the target.
        let nominal = n_frames * total_up;
        assert!(
            samples.len() <= nominal + total_up && samples.len() + total_up >= nominal,
            "output length {} not within ±{total_up} of nominal {nominal}",
            samples.len()
        );
        // tanh-bounded output.
        for s in &samples {
            assert!(s.is_finite(), "non-finite sample: {s}");
            assert!((-1.0..=1.0).contains(s), "out-of-range sample: {s}");
        }
        assert_eq!(model.output_sample_rate_hz(), cfg.output_sample_rate_hz);
    }

    #[test]
    fn nsf_hifigan_rejects_mismatched_frame_counts() {
        let device = cpu();
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model =
            NsfHifiGan::load_from_var_builder(vb, &device, cfg.clone()).expect("load generator");

        let content = Tensor::randn(0_f32, 1.0_f32, (1, cfg.hidden_dim, 4), &device).expect("c");
        let pitch_coarse = Tensor::from_vec(vec![1_i64; 3], (1, 3), &device).expect("pitch_coarse");
        let pitch_hz = Tensor::from_vec(vec![100.0_f32; 4], (1, 4), &device).expect("pitch_hz");
        let speaker_id = Tensor::from_vec(vec![0_i64], (1,), &device).expect("speaker_id");
        let err = model
            .synthesize(&content, &pitch_coarse, &pitch_hz, &speaker_id)
            .expect_err("should fail on mismatched frame counts");
        match err {
            GeneratorError::Inference(msg) => {
                assert!(
                    msg.contains("frame-count mismatch"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected Inference, got {other:?}"),
        }
    }
}
