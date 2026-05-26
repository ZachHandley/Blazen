//! Oobleck 1D residual VAE — the autoencoder used by Stability AI's
//! Stable Audio Open family.
//!
//! Architecture (Stable Audio Open Small, the variant we target first):
//!
//! ```text
//!   stereo waveform  (B, 2, T)                        T = 44_100 · seconds
//!         │
//!         ▼
//!   pre-conv:    Conv1d(2 → C0, kernel=7, pad=3)      C0 = channels[0] = 128
//!         │
//!         ▼   (one EncoderBlock per stride in strides=[2,4,8,8])
//!   EncoderBlock_i:
//!     ┌────────────────────────────────────────────────────────────────┐
//!     │ ResidualUnit(dilation=1)                                       │
//!     │ ResidualUnit(dilation=3)                                       │
//!     │ ResidualUnit(dilation=9)                                       │
//!     │ snake → Conv1d(C_i → C_{i+1}, kernel=2·stride, stride=stride)   │
//!     └────────────────────────────────────────────────────────────────┘
//!         │
//!         ▼
//!   post-block:   ResidualUnit(dilation=1)
//!   snake →  Conv1d(C_last → 2·latent_dim, kernel=3, pad=1)
//!         │
//!         ▼
//!   bottleneck split → (mean, raw_logvar)
//!   logvar = raw_logvar.tanh() · LOGVAR_SCALE
//!         │
//!         ▼
//!   latent (B, 64, T/512)                              86 Hz @ 44_100 Hz input
//! ```
//!
//! The decoder mirrors the encoder: a `Conv1d(latent_dim → C_last, k=7, p=3)`,
//! a `ResidualUnit(d=1)`, four `DecoderBlock`s that each upsample by their
//! stride via a `ConvTranspose1d` followed by three `ResidualUnit`s with
//! dilations `[1, 3, 9]`, and a final `snake → Conv1d(C0 → 2, k=7, p=3) → tanh`
//! that bounds the waveform to `[-1, 1]`.
//!
//! Each snake activation owns a learnable per-channel `alpha` of shape
//! `[1, C, 1]` (broadcastable over batch and time). Computed as
//! `x + (1/alpha) · sin(alpha · x)²`. The `alpha` parameter is loaded from
//! the safetensors checkpoint under the `alpha` key inside each snake block.
//!
//! Param-mapping note: this module ONLY defines the candle architecture and
//! consumes weights through the supplied [`VarBuilder`]. The HF safetensors
//! → candle key remap lives in `weights.rs` (built in a later wave); this
//! file is intentionally key-agnostic and treats the `VarBuilder` paths as
//! the source of truth.

// NOTE: This module is feature-gated on `stable-audio` (Wave 3.5+); the
// feature activates the same candle / candle-nn dep set MusicGen uses.
// a dedicated `stable-audio` feature and migrate the gate; until then the
// file compiles inside any build that already pulls candle in.
//
// `needless_pass_by_value` is allowed because `VarBuilder` is the canonical
// constructor input across all of candle (`conv1d`, `Linear::new`,
// EnCodec, MusicGen, StableDiffusion, …) — taking it by value matches the
// upstream convention. `cast_precision_loss` / `cast_possible_wrap` /
// `cast_lossless` come up in test assertions where the input values are
// tiny and statically bounded.
#![allow(
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]

use candle_core::{Module, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder, conv_transpose1d,
    conv1d,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Oobleck VAE. Matches the `model_config.json`
/// schema shipped with `stabilityai/stable-audio-open-{small,1.0}` for the
/// `model.pretransform.config.config` block, with field names normalized
/// to Rust conventions.
#[derive(Debug, Clone, PartialEq)]
pub struct OobleckConfig {
    /// Output channel count for the pre-conv and channel widths after each
    /// encoder block. Length must equal `strides.len() + 1` — `channels[0]`
    /// is the post-pre-conv width and `channels[i+1]` is the output of the
    /// `i`-th encoder block.
    pub channels: Vec<usize>,
    /// Downsampling factor for each encoder block. The decoder mirrors this
    /// in reverse. Product = total downsample ratio (`512` for Small).
    pub strides: Vec<usize>,
    /// Dilations used by the three residual units inside each block.
    pub dilations: Vec<usize>,
    /// Latent channel count (the bottleneck width). `64` for both Stable
    /// Audio Open Small and 1.0.
    pub latent_dim: usize,
    /// Native sample rate of the waveform domain (`44_100` for both
    /// Stable Audio Open variants).
    pub sample_rate: u32,
    /// Number of input/output audio channels. `2` (stereo) for both
    /// Stable Audio Open variants.
    pub audio_channels: usize,
    /// Soft-clipping scale applied to the post-tanh log-variance. The
    /// Stability AI reference uses a wide range (`[-30, 20]`) but the
    /// open-source autoencoder fixes this to `LOGVAR_SCALE`; expose it
    /// so 1.0 configs can override.
    pub logvar_scale: f32,
}

/// Default scale for the bottleneck log-variance tanh; matches the
/// `tanh_bottleneck` factor in `stable-audio-tools/models/bottleneck.py`.
const LOGVAR_SCALE: f32 = 9.0;

impl OobleckConfig {
    /// Config for `stabilityai/stable-audio-open-small`.
    ///
    /// `channels=[128, 256, 512, 1024, 1024]`, `strides=[2, 4, 8, 8]`,
    /// `dilations=[1, 3, 9]`, `latent_dim=64`, sample_rate=44_100, stereo.
    ///
    /// Downsample ratio = `2 · 4 · 8 · 8 = 512`. Latent frame rate at
    /// 44_100 Hz input = `44_100 / 512 ≈ 86.13 Hz`.
    #[must_use]
    pub fn stable_audio_small() -> Self {
        Self {
            channels: vec![128, 256, 512, 1024, 1024],
            strides: vec![2, 4, 8, 8],
            dilations: vec![1, 3, 9],
            latent_dim: 64,
            sample_rate: 44_100,
            audio_channels: 2,
            logvar_scale: LOGVAR_SCALE,
        }
    }

    /// Total temporal downsample ratio applied by the encoder
    /// (= product of `strides`).
    #[must_use]
    pub fn downsample_ratio(&self) -> usize {
        self.strides.iter().product()
    }

    /// Sanity-check the invariants between `channels`, `strides`, and
    /// `dilations`. Returns a candle error so callers can `?` it.
    fn validate(&self) -> Result<()> {
        if self.channels.len() != self.strides.len() + 1 {
            candle_core::bail!(
                "OobleckConfig: channels.len() ({}) must equal strides.len() ({}) + 1",
                self.channels.len(),
                self.strides.len(),
            );
        }
        if self.dilations.is_empty() {
            candle_core::bail!("OobleckConfig: dilations must be non-empty");
        }
        if self.latent_dim == 0 {
            candle_core::bail!("OobleckConfig: latent_dim must be > 0");
        }
        if self.audio_channels == 0 {
            candle_core::bail!("OobleckConfig: audio_channels must be > 0");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Snake activation
// ---------------------------------------------------------------------------

/// Apply the snake activation `x + (1/alpha) · sin(alpha · x)²` element-wise.
///
/// `alpha` is the per-channel learnable parameter, shaped `[1, C, 1]`
/// so it broadcasts cleanly across the batch and time dimensions of an
/// `(B, C, T)` activation.
///
/// Numerical safety: `alpha` is reciprocated, so any channel whose loaded
/// value is exactly zero would blow up. The Stability checkpoint never
/// initializes `alpha` to zero (init is `1.0` with a per-channel learnable
/// scalar drift), but we still clamp via `+ 1e-9` before reciprocation to
/// stay safe under random `VarBuilder` initializers used in tests.
pub fn snake_activation(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let alpha = alpha.to_dtype(dtype)?;
    // Stability-safe reciprocal: alpha + 1e-9 to avoid div-by-zero with
    // randomly-initialized VarBuilders in unit tests.
    let eps = Tensor::new(1e-9_f32, x.device())?.to_dtype(dtype)?;
    let alpha_safe = alpha.broadcast_add(&eps)?;
    let inv_alpha = alpha_safe.recip()?;
    let scaled = x.broadcast_mul(&alpha)?;
    let sin_sq = scaled.sin()?.sqr()?;
    let bump = sin_sq.broadcast_mul(&inv_alpha)?;
    x + bump
}

/// Snake activation parameter container — a single `[1, channels, 1]`
/// learnable tensor. Lives as its own struct so the safetensors loader
/// can route the `alpha` key cleanly.
#[derive(Debug, Clone)]
pub struct Snake {
    alpha: Tensor,
}

impl Snake {
    /// Construct a Snake activation by loading its `alpha` tensor under
    /// `vb / "alpha"`.
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    /// The learnable per-channel alpha tensor, shaped `[1, C, 1]`.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Surfaced for parity tests + future Snake variant wiring; \
                  the forward path reads self.alpha directly."
    )]
    pub fn alpha(&self) -> &Tensor {
        &self.alpha
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        snake_activation(x, &self.alpha)
    }
}

// ---------------------------------------------------------------------------
// Residual unit
// ---------------------------------------------------------------------------

/// Padding for a 1D dilated conv of odd kernel size that preserves the
/// time axis: `same`-padding for `kernel * dilation`.
#[inline]
fn dilated_same_padding(kernel: usize, dilation: usize) -> usize {
    ((kernel - 1) * dilation) / 2
}

/// 3-conv residual unit used inside every encoder and decoder block.
///
/// Structure (per `stable-audio-tools` `OobleckResidualUnit`):
///
/// ```text
///   snake_in → dilated Conv1d(C, C, kernel=7, dilation=d, padding=same)
///   snake_mid → Conv1d(C, C, kernel=1)
///   + residual
/// ```
///
/// `dilation` cycles through the configured `dilations` list (typically
/// `[1, 3, 9]`).
#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake_in: Snake,
    conv_dilated: Conv1d,
    snake_mid: Snake,
    conv_point: Conv1d,
}

impl ResidualUnit {
    /// Build a residual unit with the given `channels` and `dilation`.
    /// Loads weights under the supplied `VarBuilder` using the canonical
    /// `stable-audio-tools` sub-keys:
    ///
    /// * `snake1.alpha`
    /// * `conv1.{weight,bias}`     — dilated 7-kernel conv
    /// * `snake2.alpha`
    /// * `conv2.{weight,bias}`     — 1-kernel pointwise conv
    pub fn new(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        const KERNEL: usize = 7;
        let snake_in = Snake::new(channels, vb.pp("snake1"))?;
        let conv_dilated = conv1d(
            channels,
            channels,
            KERNEL,
            Conv1dConfig {
                padding: dilated_same_padding(KERNEL, dilation),
                stride: 1,
                dilation,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv1"),
        )?;
        let snake_mid = Snake::new(channels, vb.pp("snake2"))?;
        let conv_point = conv1d(
            channels,
            channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv2"),
        )?;
        Ok(Self {
            snake_in,
            conv_dilated,
            snake_mid,
            conv_point,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake_in.forward(x)?;
        let h = self.conv_dilated.forward(&h)?;
        let h = self.snake_mid.forward(&h)?;
        let h = self.conv_point.forward(&h)?;
        x + h
    }
}

// ---------------------------------------------------------------------------
// Encoder block
// ---------------------------------------------------------------------------

/// One encoder block: three dilated residual units followed by a strided
/// downsample conv with a pre-activation snake.
#[derive(Debug, Clone)]
pub struct EncoderBlock {
    residuals: Vec<ResidualUnit>,
    snake_down: Snake,
    conv_down: Conv1d,
}

impl EncoderBlock {
    /// Build an encoder block that maps `in_channels → out_channels` and
    /// downsamples the time axis by `stride`. Residual units operate at
    /// `in_channels` and the downsample conv changes width.
    ///
    /// Weight layout under `vb`:
    ///
    /// * `res1`, `res2`, `res3` — three [`ResidualUnit`]s, one per dilation
    /// * `snake.alpha`           — pre-downsample snake
    /// * `conv.{weight,bias}`    — downsample conv, kernel=`2·stride`, stride=`stride`
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        if dilations.is_empty() {
            candle_core::bail!("EncoderBlock: dilations must be non-empty");
        }
        let residuals = dilations
            .iter()
            .enumerate()
            .map(|(i, &d)| ResidualUnit::new(in_channels, d, vb.pp(format!("res{}", i + 1))))
            .collect::<Result<Vec<_>>>()?;

        let snake_down = Snake::new(in_channels, vb.pp("snake"))?;
        // The downsample conv uses kernel = 2 * stride and "same"-style
        // padding equal to stride / 2 (rounded down). This is the
        // convention used by `stable-audio-tools` and matches the
        // reference output shape: ceil(T / stride).
        let kernel = 2 * stride;
        let padding = stride.div_ceil(2);
        let conv_down = conv1d(
            in_channels,
            out_channels,
            kernel,
            Conv1dConfig {
                padding,
                stride,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv"),
        )?;

        Ok(Self {
            residuals,
            snake_down,
            conv_down,
        })
    }
}

impl Module for EncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for unit in &self.residuals {
            h = unit.forward(&h)?;
        }
        let h = self.snake_down.forward(&h)?;
        self.conv_down.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Decoder block
// ---------------------------------------------------------------------------

/// One decoder block: pre-upsample snake → strided `ConvTranspose1d` →
/// three dilated residual units. Mirrors [`EncoderBlock`].
#[derive(Debug, Clone)]
pub struct DecoderBlock {
    snake_up: Snake,
    conv_up: ConvTranspose1d,
    residuals: Vec<ResidualUnit>,
}

impl DecoderBlock {
    /// Build a decoder block that maps `in_channels → out_channels` and
    /// upsamples the time axis by `stride`. Residual units operate at
    /// `out_channels` (post-upsample width).
    ///
    /// Weight layout under `vb`:
    ///
    /// * `snake.alpha`             — pre-upsample snake (over `in_channels`)
    /// * `conv.{weight,bias}`      — transpose conv, kernel=`2·stride`, stride=`stride`
    /// * `res1`, `res2`, `res3`    — three [`ResidualUnit`]s, one per dilation
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        if dilations.is_empty() {
            candle_core::bail!("DecoderBlock: dilations must be non-empty");
        }
        let snake_up = Snake::new(in_channels, vb.pp("snake"))?;
        let kernel = 2 * stride;
        let padding = stride.div_ceil(2);
        let conv_up = conv_transpose1d(
            in_channels,
            out_channels,
            kernel,
            ConvTranspose1dConfig {
                padding,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv"),
        )?;
        let residuals = dilations
            .iter()
            .enumerate()
            .map(|(i, &d)| ResidualUnit::new(out_channels, d, vb.pp(format!("res{}", i + 1))))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            snake_up,
            conv_up,
            residuals,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake_up.forward(x)?;
        let mut h = self.conv_up.forward(&h)?;
        for unit in &self.residuals {
            h = unit.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Full Oobleck encoder: pre-conv → N encoder blocks → post residual unit →
/// final conv that doubles the channel count to `2 · latent_dim` (mean +
/// raw_logvar split downstream).
///
/// The encoder is built + weight-loaded as part of the VAE but is not
/// exercised on the diffusion-only inference path (which only needs
/// [`OobleckDecoder`]). Wave 4+ will wire it up for audio-to-audio
/// (inpainting / variation / encode-then-resynth) flows.
#[allow(
    dead_code,
    reason = "Encoder fields are loaded from the safetensors weights for \
              round-trip parity tests and audio-to-audio flows added in \
              later waves; the diffusion-only inference path only calls \
              the decoder."
)]
#[derive(Debug, Clone)]
pub struct OobleckEncoder {
    pre_conv: Conv1d,
    blocks: Vec<EncoderBlock>,
    post_block: ResidualUnit,
    post_snake: Snake,
    final_conv: Conv1d,
    logvar_scale: f32,
    latent_dim: usize,
}

impl OobleckEncoder {
    /// Construct the encoder from `config` using the supplied `VarBuilder`.
    /// Weight layout under `vb`:
    ///
    /// * `pre_conv.{weight,bias}`
    /// * `blocks.{0..N-1}.…`       — see [`EncoderBlock::new`]
    /// * `post_block.…`             — a single [`ResidualUnit`] at `channels[last]`
    /// * `post_snake.alpha`
    /// * `final_conv.{weight,bias}` — kernel=3, stride=1, padding=1
    pub fn new(config: &OobleckConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let pre_conv = conv1d(
            config.audio_channels,
            config.channels[0],
            7,
            Conv1dConfig {
                padding: 3,
                ..Conv1dConfig::default()
            },
            vb.pp("pre_conv"),
        )?;

        let blocks = config
            .strides
            .iter()
            .enumerate()
            .map(|(i, &stride)| {
                EncoderBlock::new(
                    config.channels[i],
                    config.channels[i + 1],
                    stride,
                    &config.dilations,
                    vb.pp(format!("blocks.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let last_channels = *config.channels.last().expect("validated non-empty");
        let post_block = ResidualUnit::new(last_channels, 1, vb.pp("post_block"))?;
        let post_snake = Snake::new(last_channels, vb.pp("post_snake"))?;
        let final_conv = conv1d(
            last_channels,
            2 * config.latent_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Conv1dConfig::default()
            },
            vb.pp("final_conv"),
        )?;

        Ok(Self {
            pre_conv,
            blocks,
            post_block,
            post_snake,
            final_conv,
            logvar_scale: config.logvar_scale,
            latent_dim: config.latent_dim,
        })
    }

    /// Encode a waveform `(B, audio_channels, T)` to `(mean, logvar)`,
    /// each shaped `(B, latent_dim, T / downsample_ratio)`.
    #[allow(
        dead_code,
        reason = "Encoder forward is reserved for audio-to-audio flows; \
                  the diffusion-only inference path only calls the decoder."
    )]
    pub fn forward(&self, audio: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.pre_conv.forward(audio)?;
        let mut h = h;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = self.post_block.forward(&h)?;
        let h = self.post_snake.forward(&h)?;
        let h = self.final_conv.forward(&h)?;

        // Split channel dim into (mean, raw_logvar) — first `latent_dim`
        // channels are the mean, second `latent_dim` are the raw logvar.
        let mean = h.narrow(1, 0, self.latent_dim)?;
        let raw_logvar = h.narrow(1, self.latent_dim, self.latent_dim)?;
        let scale = Tensor::new(self.logvar_scale, h.device())?.to_dtype(raw_logvar.dtype())?;
        let logvar = raw_logvar.tanh()?.broadcast_mul(&scale)?;
        Ok((mean, logvar))
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Full Oobleck decoder: input conv → pre residual unit → N decoder blocks →
/// snake → output conv → tanh (bounds waveform to `[-1, 1]`).
#[derive(Debug, Clone)]
pub struct OobleckDecoder {
    pre_conv: Conv1d,
    pre_block: ResidualUnit,
    blocks: Vec<DecoderBlock>,
    post_snake: Snake,
    final_conv: Conv1d,
}

impl OobleckDecoder {
    /// Construct the decoder from `config`. Weight layout under `vb`:
    ///
    /// * `pre_conv.{weight,bias}`   — `latent_dim → channels[last]`, kernel=7, padding=3
    /// * `pre_block.…`               — a single [`ResidualUnit`] at `channels[last]`
    /// * `blocks.{0..N-1}.…`         — see [`DecoderBlock::new`], iterated in REVERSE order vs encoder
    /// * `post_snake.alpha`
    /// * `final_conv.{weight,bias}` — `channels[0] → audio_channels`, kernel=7, padding=3
    pub fn new(config: &OobleckConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let last_channels = *config.channels.last().expect("validated non-empty");

        let pre_conv = conv1d(
            config.latent_dim,
            last_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Conv1dConfig::default()
            },
            vb.pp("pre_conv"),
        )?;
        let pre_block = ResidualUnit::new(last_channels, 1, vb.pp("pre_block"))?;

        // Decoder blocks walk the channel/stride pairs in REVERSE: the
        // first decoder block maps `channels[last] → channels[last-1]`
        // with `strides[last]`.
        let n = config.strides.len();
        let blocks = (0..n)
            .map(|i| {
                let in_c = config.channels[n - i];
                let out_c = config.channels[n - i - 1];
                let stride = config.strides[n - i - 1];
                DecoderBlock::new(
                    in_c,
                    out_c,
                    stride,
                    &config.dilations,
                    vb.pp(format!("blocks.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let post_snake = Snake::new(config.channels[0], vb.pp("post_snake"))?;
        let final_conv = conv1d(
            config.channels[0],
            config.audio_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Conv1dConfig::default()
            },
            vb.pp("final_conv"),
        )?;

        Ok(Self {
            pre_conv,
            pre_block,
            blocks,
            post_snake,
            final_conv,
        })
    }

    /// Decode a latent `(B, latent_dim, T_latent)` to a stereo waveform
    /// `(B, audio_channels, T_latent · downsample_ratio)`.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let h = self.pre_conv.forward(latent)?;
        let mut h = self.pre_block.forward(&h)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = self.post_snake.forward(&h)?;
        let h = self.final_conv.forward(&h)?;
        h.tanh()
    }
}

// ---------------------------------------------------------------------------
// VAE wrapper
// ---------------------------------------------------------------------------

/// Oobleck variational autoencoder: encoder + decoder + reparameterization.
#[derive(Debug, Clone)]
pub struct OobleckVAE {
    /// Encoder half. Held for audio-to-audio flows added in later waves;
    /// the diffusion-only inference path only touches the decoder.
    #[allow(
        dead_code,
        reason = "Held for audio-to-audio flows added in later waves; \
                  the diffusion-only inference path only touches the \
                  decoder."
    )]
    encoder: OobleckEncoder,
    decoder: OobleckDecoder,
    /// Resolved config. Surfaced via [`Self::config`] for callers that
    /// need to introspect the latent frame rate / channel layout.
    #[allow(
        dead_code,
        reason = "Surfaced via config()/latent_frame_rate(); the encode + \
                  decode paths use values cached on the encoder/decoder \
                  sub-modules."
    )]
    config: OobleckConfig,
}

impl OobleckVAE {
    /// Build the VAE from `config`, loading encoder weights under
    /// `vb / "encoder"` and decoder weights under `vb / "decoder"`.
    pub fn new(vb: VarBuilder, config: &OobleckConfig) -> Result<Self> {
        let encoder = OobleckEncoder::new(config, vb.pp("encoder"))?;
        let decoder = OobleckDecoder::new(config, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            config: config.clone(),
        })
    }

    /// Encode a waveform to `(mean, logvar)`. See [`OobleckEncoder::forward`].
    #[allow(
        dead_code,
        reason = "Reserved for audio-to-audio flows added in later waves; \
                  the diffusion-only inference path only calls decode()."
    )]
    pub fn encode(&self, audio: &Tensor) -> Result<(Tensor, Tensor)> {
        self.encoder.forward(audio)
    }

    /// Decode a latent to a waveform. See [`OobleckDecoder::forward`].
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.decoder.forward(latent)
    }

    /// Deterministic eval shortcut: take the posterior mode (== mean) as
    /// the latent without sampling. Use this for diffusion-time inference
    /// where stochasticity comes from the diffusion sampler, not the VAE.
    #[allow(
        clippy::unused_self,
        dead_code,
        reason = "Reserved for audio-to-audio flows; not called on the \
                  diffusion-only inference path."
    )]
    pub fn mode(&self, mean: &Tensor) -> Tensor {
        mean.clone()
    }

    /// Reparameterized sample: `latent = mean + std · ε` where
    /// `std = exp(0.5 · logvar)` and `ε ~ N(0, 1)` matches `mean`'s
    /// dtype/device/shape. Used at training time (not for inference;
    /// inference should call [`Self::mode`]).
    #[allow(
        clippy::unused_self,
        dead_code,
        reason = "Training-time sampler kept for the encode-then-resample \
                  flow added in later waves."
    )]
    pub fn sample(&self, mean: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        let half = Tensor::new(0.5_f32, mean.device())?.to_dtype(mean.dtype())?;
        let std = logvar.broadcast_mul(&half)?.exp()?;
        // randn_like wants (mean_value, stddev_value) — sample a unit
        // Gaussian by passing (0, 1) so the broadcast_mul below applies
        // our learned std.
        let eps = mean.randn_like(0.0, 1.0)?;
        let noise = std.mul(&eps)?;
        mean.add(&noise)
    }

    /// Borrow the resolved config.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Public accessor surfaced for callers that need to \
                  introspect the active hyperparameter pack."
    )]
    pub fn config(&self) -> &OobleckConfig {
        &self.config
    }

    /// Borrow the encoder sub-module.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Public accessor surfaced for audio-to-audio flows; not \
                  called on the diffusion-only inference path."
    )]
    pub fn encoder(&self) -> &OobleckEncoder {
        &self.encoder
    }

    /// Borrow the decoder sub-module.
    #[must_use]
    #[allow(
        dead_code,
        reason = "Public accessor surfaced for parity tests; the inference \
                  path calls Self::decode() directly."
    )]
    pub fn decoder(&self) -> &OobleckDecoder {
        &self.decoder
    }

    /// Latent frame rate at the configured sample rate
    /// (`sample_rate / downsample_ratio`).
    #[must_use]
    #[allow(
        dead_code,
        reason = "Surfaced for diagnostics + the pipeline orchestration \
                  layer; the t_latent math in pipeline.rs uses an \
                  integer-86 Hz constant for bit-exact runs."
    )]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample_rate fits in i24 (44_100); the f32 cast is exact."
    )]
    pub fn latent_frame_rate(&self) -> f32 {
        self.config.sample_rate as f32 / self.config.downsample_ratio() as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// Build a VAE backed by a fresh randomly-initialized `VarMap`. The
    /// `VarMap` is leaked (`Box::leak`) so the resulting `VarBuilder` can
    /// safely outlive the constructor call — this is a test helper and
    /// the leaked memory is freed at process exit.
    fn fresh_vae(device: &Device, dtype: DType, cfg: &OobleckConfig) -> Result<OobleckVAE> {
        let varmap: &'static VarMap = Box::leak(Box::new(VarMap::new()));
        let vb = VarBuilder::from_varmap(varmap, dtype, device);
        OobleckVAE::new(vb, cfg)
    }

    #[test]
    fn small_config_invariants() {
        let cfg = OobleckConfig::stable_audio_small();
        assert_eq!(cfg.downsample_ratio(), 512);
        assert_eq!(cfg.latent_dim, 64);
        assert_eq!(cfg.audio_channels, 2);
        assert_eq!(cfg.sample_rate, 44_100);
        assert_eq!(cfg.channels.len(), cfg.strides.len() + 1);
        // 86.13 Hz latent frame rate
        let frame_rate = cfg.sample_rate as f32 / cfg.downsample_ratio() as f32;
        assert!((frame_rate - 86.13).abs() < 0.1);
    }

    #[test]
    fn snake_activation_zero_alpha_is_safe() -> Result<()> {
        // alpha exactly zero: the +1e-9 guard should keep recip() finite
        // and produce a finite output (the sin² term goes to zero too).
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0_f32, 2.0, 3.0]], &device)?
            .unsqueeze(0)?
            .transpose(1, 2)?; // (1, 3, 1) — 3 channels of length 1
        let alpha = Tensor::zeros((1, 3, 1), DType::F32, &device)?;
        let out = snake_activation(&x, &alpha)?;
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        for v in flat {
            assert!(v.is_finite(), "snake output {v} is non-finite");
        }
        Ok(())
    }

    #[test]
    fn snake_activation_matches_formula() -> Result<()> {
        // alpha = 1.0 → snake(x) = x + sin(x)²
        let device = Device::Cpu;
        let xs = [0.5_f32, -0.5, 1.5];
        // Layout (B=1, C=3, T=1)
        let x = Tensor::new(&xs, &device)?.reshape((1, 3, 1))?;
        let alpha = Tensor::ones((1, 3, 1), DType::F32, &device)?;
        let out = snake_activation(&x, &alpha)?;
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        for (got, &x_in) in flat.iter().zip(xs.iter()) {
            let expected = x_in + x_in.sin().powi(2);
            assert!(
                (got - expected).abs() < 1e-5,
                "snake(x={x_in}): got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn encoder_shape_10_seconds() -> Result<()> {
        let device = Device::Cpu;
        let cfg = OobleckConfig::stable_audio_small();
        let vae = fresh_vae(&device, DType::F32, &cfg)?;

        // 10 s @ 44.1 kHz, stereo. 10 · 44_100 = 441_000 samples.
        let audio = Tensor::zeros((1, 2, 441_000), DType::F32, &device)?;
        let (mean, logvar) = vae.encode(&audio)?;
        // Expect (1, 64, 441_000 / 512) ≈ (1, 64, 861)
        assert_eq!(mean.dims()[0], 1);
        assert_eq!(mean.dims()[1], 64);
        assert!(
            (mean.dims()[2] as i64 - 861).abs() <= 4,
            "unexpected mean time dim {}",
            mean.dims()[2]
        );
        assert_eq!(mean.shape(), logvar.shape());
        // logvar must be bounded by ±logvar_scale (tanh output ∈ [-1, 1]).
        let bound = cfg.logvar_scale + 1e-5;
        let lv = logvar.flatten_all()?.to_vec1::<f32>()?;
        for v in lv {
            assert!(v.abs() <= bound, "logvar {v} outside ±{bound}");
        }
        Ok(())
    }

    #[test]
    fn decoder_shape_10_seconds() -> Result<()> {
        let device = Device::Cpu;
        let cfg = OobleckConfig::stable_audio_small();
        let vae = fresh_vae(&device, DType::F32, &cfg)?;

        // 10 s of latents @ 86 Hz ≈ 860 frames.
        let latent = Tensor::zeros((1, 64, 860), DType::F32, &device)?;
        let audio = vae.decode(&latent)?;
        assert_eq!(audio.dims()[0], 1);
        assert_eq!(audio.dims()[1], 2);
        // 860 · 512 = 440_320 samples — within rounding of 441_000.
        let expected = 860 * 512;
        let got = audio.dims()[2] as i64;
        assert!(
            (got - expected as i64).abs() <= 1024,
            "decoder time dim {got} not close to {expected}"
        );
        // Output is tanh-bounded.
        let flat = audio.flatten_all()?.to_vec1::<f32>()?;
        for v in flat.iter().take(4096) {
            assert!(v.abs() <= 1.0 + 1e-5, "decoded sample {v} outside [-1, 1]");
        }
        Ok(())
    }

    #[test]
    fn mode_returns_mean_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let cfg = OobleckConfig::stable_audio_small();
        let vae = fresh_vae(&device, DType::F32, &cfg)?;
        let mean = Tensor::randn(0.0_f32, 1.0, (1, 64, 16), &device)?;
        let got = vae.mode(&mean);
        let a = mean.flatten_all()?.to_vec1::<f32>()?;
        let b = got.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    fn sample_shape_matches_mean() -> Result<()> {
        let device = Device::Cpu;
        let cfg = OobleckConfig::stable_audio_small();
        let vae = fresh_vae(&device, DType::F32, &cfg)?;
        let mean = Tensor::zeros((2, 64, 32), DType::F32, &device)?;
        let logvar = Tensor::zeros((2, 64, 32), DType::F32, &device)?;
        let sample = vae.sample(&mean, &logvar)?;
        assert_eq!(sample.shape(), mean.shape());
        Ok(())
    }

    /// Numerical parity vs the Python tensor-dump harness.
    ///
    /// `#[ignore]` because the dumps at `~/.cache/blazen-stableaudio-research/dumps/`
    /// require running `tests/python/stable_audio_dump.py` first. CI runs
    /// this manually under the `live-models` feature.
    ///
    /// Expected dump keys (produced by the Python harness):
    /// * `oobleck_encoder_input`   — (1, 2, 220_500) reference waveform
    /// * `oobleck_encoder_mean`    — (1, 64, ~430)
    /// * `oobleck_encoder_logvar`  — (1, 64, ~430)
    /// * `oobleck_decoder_output`  — (1, 2, ~220_500)
    #[test]
    #[ignore = "requires Python dump harness; see crate-level docs"]
    fn parity_vs_python_dump() {
        // Intentionally minimal: this is a scaffold for when wave 3.x wires
        // the dump-loader helper from `tests/stable_audio_block_compare.rs`
        // into the crate. The helper lives in an integration test target,
        // not in this module's tree, so this body stays a no-op marker
        // (the `#[ignore]` plus the docstring above are the contract).
    }
}
