//! F5-TTS DiT wrapper.
//!
//! Composes [`blazen_audio_core::dit`] / [`blazen_audio_core::rope`]
//! primitives into the F5-specific DiT recipe:
//!
//! - Self-attention only (no cross-attention). Text conditioning enters
//!   via **early feature-axis concat** with the noisy mel and the
//!   reference mel (`InputEmbedding.proj`), not via cross-attn.
//! - Per-block AdaLN modulation driven by a `TimestepEmbedding(dim)`
//!   stack (sinusoidal positional embedding → 2-layer MLP). Each block
//!   produces six chunks `(shift_msa, scale_msa, gate_msa, shift_mlp,
//!   scale_mlp, gate_mlp)` — note F5's chunk **order** puts shift before
//!   scale, unlike Stable Audio's `(scale, shift, gate)` triples — and
//!   uses **plain `gate` multiplication** (not `sigmoid(1 - gate)`).
//! - Mel-spectrogram I/O (100-bin) rather than audio latent. Input is
//!   the concatenation of noisy mel, reference mel, and text embedding
//!   along the feature axis, projected to `dim` and added to a
//!   depthwise-Conv1d positional embedding (`ConvPositionEmbedding`).
//! - RoPE on self-attn, applied over `dim_head` components (full rotary
//!   in F5 vs Stable Audio's partial rotary — head_dim = 64 here so the
//!   tables are sized to 64).
//! - Final `AdaLayerNorm_Final` (2-chunk shift/scale modulation by time)
//!   followed by a `proj_out` linear back to `mel_dim`.
//! - Optional `long_skip_connection` (off by default in
//!   F5TTS_Base.yaml; the field is plumbed for parity).
//!
//! # What lives in this file vs siblings
//!
//! - **dit_wrapper.rs** (this file): DiT proper — input embed (proj +
//!   conv pos), transformer blocks, final modulated norm, output
//!   projection. Expects the caller to pass a precomputed
//!   `text_embed: (B, T, text_dim)` tensor.
//! - **text encoder + ConvNeXtV2 blocks**: deferred to the Wave F.3
//!   `tokenizer.rs` / `weights.rs` step. F5-TTS's text encoder is
//!   non-trivial (sinusoidal positional → masked embedding →
//!   ConvNeXtV2 stack → optional duration-aware upsampling), and
//!   keeping it out of `dit_wrapper.rs` lets the transformer stack be
//!   reviewed and weight-validated independently. This matches the
//!   upstream split between `dit.py::DiT` and `modules.py::TextEmbedding`.
//! - **vocos.rs / sampler.rs**: owned by parallel sibling agents; do
//!   not touch.
//!
//! Upstream reference: `SWivid/F5-TTS/src/f5_tts/model/backbones/dit.py`
//! (DiT class, InputEmbedding, AdaLayerNorm_Final) and
//! `.../model/modules.py` (DiTBlock, AdaLayerNorm, TimestepEmbedding,
//! SinusPositionEmbedding, ConvPositionEmbedding). F5TTS_Base config:
//! `src/f5_tts/configs/F5TTS_Base.yaml` — `dim=1024, depth=22,
//! heads=16, dim_head=64, ff_mult=2, text_dim=512, conv_layers=4,
//! mel_dim=100`. `text_num_embeds=2545` is the published
//! `F5TTS_Base/vocab.txt` line count (English ASCII + Mandarin
//! pinyin), not from the YAML config.

#![cfg(feature = "f5-tts")]
#![allow(clippy::similar_names)] // shift_msa / scale_msa / gate_msa
#![allow(clippy::module_name_repetitions)]
// Architecture docstrings reference upstream Python identifiers
// (DiTBlock, AdaLayerNorm_Final, F5TTS_Base, etc.) — backticking every
// PascalCase reference would bury the prose. Allow naked identifiers in
// docs throughout this file.
#![allow(clippy::doc_markdown)]
// VarBuilder is the candle convention for module construction.
#![allow(clippy::needless_pass_by_value)]
// b, t, c, h, d are canonical PyTorch port names; renaming hurts
// cross-referencing against `dit.py`.
#![allow(clippy::many_single_char_names)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
// Wave F.2 lands the standalone DiT wrapper; Wave F.4 wires it into the
// pipeline. Until then the public surface is exercised by the in-tree
// tests only.
#![allow(dead_code)]

use blazen_audio_core::dit::Attention;
use blazen_audio_core::rope::precompute_rope_freqs;
use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder, conv1d, linear, linear_no_bias};

/// Configuration for the F5-TTS DiT.
///
/// Defaults match `src/f5_tts/configs/F5TTS_Base.yaml`:
/// `dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, text_dim=512,
/// conv_layers=4, mel_dim=100`. `text_num_embeds=2545` comes from
/// the published `F5TTS_Base/vocab.txt` line count.
#[derive(Debug, Clone)]
pub struct F5DitConfig {
    /// Mel-spectrogram bin count. 100 in the upstream `F5TTS_Base.yaml`
    /// (also `n_mel_channels: 100` in the Vocos config).
    pub mel_dim: usize,
    /// Transformer hidden width. 1024 for Base.
    pub hidden_dim: usize,
    /// Transformer block count. 22 for Base.
    pub depth: usize,
    /// Number of attention heads. 16 for Base.
    pub heads: usize,
    /// Per-head dimension. 64 for Base (`heads * dim_head = 1024`).
    pub head_dim: usize,
    /// Feed-forward expansion multiplier. **2** for Base (NOT 4 —
    /// upstream `F5TTS_Base.yaml` sets `ff_mult: 2`).
    pub ff_mult: usize,
    /// Text embedding dimension. 512 for Base.
    pub text_dim: usize,
    /// Number of text-encoder ConvNeXtV2 blocks. 4 for Base. Plumbed
    /// here for the F.3 text-encoder wave; the DiT stack itself doesn't
    /// touch these (text_embed arrives pre-encoded).
    pub conv_layers: usize,
    /// Text vocabulary size. `2545` for Base (English ASCII + Mandarin
    /// pinyin per upstream `F5TTS_Base/vocab.txt`).
    pub text_num_embeds: usize,
    /// Width of the sinusoidal timestep embedding before the time MLP.
    /// 256 in `TimestepEmbedding(dim, freq_embed_dim=256)`.
    pub freq_embed_dim: usize,
    /// `ConvPositionEmbedding` kernel size. 31 in the reference.
    pub conv_pos_kernel: usize,
    /// `ConvPositionEmbedding` groups. 16 in the reference (depthwise).
    pub conv_pos_groups: usize,
    /// Max sequence length used to precompute RoPE tables. 4096 covers
    /// the 30-second @ 24 kHz / hop-256 ceiling with headroom.
    pub max_seq_len: usize,
    /// Whether to enable the optional long skip connection from
    /// pre-block to post-block. Off by default in F5TTS_Base.
    pub long_skip_connection: bool,
    /// Whether to apply `qk_norm` (L2 normalize Q and K). Off by
    /// default in F5TTS_Base (`qk_norm: null`).
    pub qk_norm: bool,
}

impl F5DitConfig {
    /// Hyperparameters for `F5TTS_Base` — the published flagship config.
    /// Cites `src/f5_tts/configs/F5TTS_Base.yaml`:
    /// - L? `dim: 1024`
    /// - L? `depth: 22`
    /// - L? `heads: 16`
    /// - L? `ff_mult: 2`
    /// - L? `text_dim: 512`
    /// - L? `conv_layers: 4`
    /// - mel-spectrogram config: `n_mel_channels: 100`.
    #[must_use]
    pub fn f5_base() -> Self {
        Self {
            mel_dim: 100,
            hidden_dim: 1024,
            depth: 22,
            heads: 16,
            head_dim: 64,
            ff_mult: 2,
            text_dim: 512,
            conv_layers: 4,
            // Matches F5TTS_Base/vocab.txt line count (English ASCII +
            // Mandarin pinyin = 2545 entries).
            text_num_embeds: 2545,
            freq_embed_dim: 256,
            conv_pos_kernel: 31,
            conv_pos_groups: 16,
            max_seq_len: 4096,
            long_skip_connection: false,
            qk_norm: false,
        }
    }

    /// FFN hidden width: `hidden_dim * ff_mult`.
    #[must_use]
    pub fn ff_inner(&self) -> usize {
        self.hidden_dim * self.ff_mult
    }
}

/// `Mish` activation: `x * tanh(softplus(x))`. Used by
/// `ConvPositionEmbedding` in the reference. Implemented inline because
/// candle's prelude does not expose a Mish op as of this writing.
fn mish(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = ln(1 + exp(x)); use the stable formulation
    // `max(x, 0) + ln(1 + exp(-|x|))` to avoid overflow for large x.
    let zero = Tensor::zeros_like(x)?;
    let pos = x.maximum(&zero)?;
    let neg_abs = x.abs()?.neg()?;
    let softplus = (pos + neg_abs.exp()?.affine(1.0, 1.0)?.log()?)?;
    let tanh_sp = softplus.tanh()?;
    x.mul(&tanh_sp)
}

/// `SinusPositionEmbedding(dim)` — sinusoidal embedding of a `(B,)`
/// continuous scalar to `(B, dim)` via the canonical
/// `sin/cos(scale * x * exp(-arange(half) * log(10000) / (half-1)))`
/// recipe. `scale=1000` in `F5-TTS` (vs `1.0` in `nanoGPT`).
#[derive(Debug)]
struct SinusPositionEmbedding {
    inv_freq: Tensor, // (half_dim,)
    scale: f64,
}

impl SinusPositionEmbedding {
    fn new(dim: usize, device: &candle_core::Device) -> Result<Self> {
        assert!(dim.is_multiple_of(2), "sinus dim must be even (got {dim})");
        let half = dim / 2;
        let log_10000 = 10_000f32.ln();
        let denom = (half - 1) as f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (-(i as f32) * log_10000 / denom).exp())
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half, device)?;
        Ok(Self {
            inv_freq,
            scale: 1000.0,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B,) f32 → (B, dim)
        let x = x.to_dtype(DType::F32)?;
        let inv_freq = self
            .inv_freq
            .to_device(x.device())?
            .unsqueeze(0)?
            .to_dtype(x.dtype())?; // (1, half)
        let x = x.unsqueeze(1)?; // (B, 1)
        let emb = (x.broadcast_mul(&inv_freq)? * self.scale)?;
        let s = emb.sin()?;
        let c = emb.cos()?;
        Tensor::cat(&[&s, &c], D::Minus1)
    }
}

/// `TimestepEmbedding(dim, freq_embed_dim=256)`: sinusoidal embed →
/// `Linear → SiLU → Linear`. Output `(B, hidden_dim)` is the per-batch
/// time conditioning vector consumed by every AdaLN-Zero block.
#[derive(Debug)]
struct TimestepEmbedding {
    sinus: SinusPositionEmbedding,
    mlp1: Linear,
    mlp2: Linear,
}

impl TimestepEmbedding {
    fn new(vb: VarBuilder, hidden_dim: usize, freq_embed_dim: usize) -> Result<Self> {
        let sinus = SinusPositionEmbedding::new(freq_embed_dim, vb.device())?;
        // PyTorch `nn.Sequential` indexing — `[0]` and `[2]` are the
        // two Linears, `[1]` is the SiLU activation.
        let mlp1 = linear(freq_embed_dim, hidden_dim, vb.pp("time_mlp.0"))?;
        let mlp2 = linear(hidden_dim, hidden_dim, vb.pp("time_mlp.2"))?;
        Ok(Self { sinus, mlp1, mlp2 })
    }

    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let h = self.sinus.forward(timestep)?;
        let h = self.mlp1.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.mlp2.forward(&h)
    }
}

/// `ConvPositionEmbedding(dim, kernel_size=31, groups=16)` — two
/// depthwise (well, grouped) Conv1d layers with Mish in between,
/// applied along the time axis and added to the input as a residual.
#[derive(Debug)]
struct ConvPositionEmbedding {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ConvPositionEmbedding {
    fn new(vb: VarBuilder, dim: usize, kernel_size: usize, groups: usize) -> Result<Self> {
        assert!(
            kernel_size % 2 == 1,
            "conv pos kernel must be odd (got {kernel_size})"
        );
        let cfg = Conv1dConfig {
            padding: kernel_size / 2,
            stride: 1,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        };
        // Reference uses `nn.Sequential(Conv1d, Mish, Conv1d, Mish)`;
        // sub-keys `[0]` and `[2]` are the two Conv1ds.
        let conv1 = conv1d(dim, dim, kernel_size, cfg, vb.pp("conv1d.0"))?;
        let conv2 = conv1d(dim, dim, kernel_size, cfg, vb.pp("conv1d.2"))?;
        Ok(Self { conv1, conv2 })
    }

    /// `x`: `(B, T, D)` — transposed to `(B, D, T)` for the conv,
    /// transposed back at the end.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bdt = x.transpose(1, 2)?.contiguous()?; // (B, D, T)
        let h = self.conv1.forward(&bdt)?;
        let h = mish(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = mish(&h)?;
        h.transpose(1, 2)?.contiguous()
    }
}

/// `InputEmbedding(mel_dim, text_dim, out_dim)`: project
/// `cat([x, cond, text_embed], dim=-1)` from `(2*mel_dim + text_dim)`
/// down to `out_dim`, then add the `ConvPositionEmbedding` as a
/// residual.
#[derive(Debug)]
struct InputEmbedding {
    proj: Linear,
    conv_pos_embed: ConvPositionEmbedding,
}

impl InputEmbedding {
    fn new(vb: VarBuilder, cfg: &F5DitConfig) -> Result<Self> {
        let proj = linear(
            cfg.mel_dim * 2 + cfg.text_dim,
            cfg.hidden_dim,
            vb.pp("proj"),
        )?;
        let conv_pos_embed = ConvPositionEmbedding::new(
            vb.pp("conv_pos_embed"),
            cfg.hidden_dim,
            cfg.conv_pos_kernel,
            cfg.conv_pos_groups,
        )?;
        Ok(Self {
            proj,
            conv_pos_embed,
        })
    }

    /// `x`: noisy mel `(B, T, mel_dim)`
    /// `cond`: reference mel `(B, T, mel_dim)` (zeroed when audio cond
    /// is dropped for CFG)
    /// `text_embed`: precomputed text features `(B, T, text_dim)`
    fn forward(&self, x: &Tensor, cond: &Tensor, text_embed: &Tensor) -> Result<Tensor> {
        let cat = Tensor::cat(&[x, cond, text_embed], D::Minus1)?;
        let projected = self.proj.forward(&cat)?;
        let pos = self.conv_pos_embed.forward(&projected)?;
        projected + pos
    }
}

/// F5's per-block `AdaLayerNorm` — produces six chunks
/// `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`
/// from a SiLU-activated linear of the time embedding.
///
/// **NOT** the same as `blazen_audio_core::adaln::AdaLNModulation`:
/// Stable Audio's modulation returns `(scale, shift, gate)` per path
/// in that order; F5's linear is laid out as `(shift, scale, gate)`
/// per path. We model the F5 layout directly so safetensors keys load
/// cleanly without a chunk-shuffle in `weights.rs`.
#[derive(Debug)]
pub(super) struct F5AdaLayerNorm {
    linear: Linear,
    norm: candle_nn::LayerNorm,
    embed_dim: usize,
}

impl F5AdaLayerNorm {
    fn new(vb: VarBuilder, embed_dim: usize) -> Result<Self> {
        // `nn.LayerNorm(dim, elementwise_affine=False)` → no gamma/beta
        // tensor on disk. We synthesize a unit gamma / zero beta to
        // satisfy candle's `LayerNorm` constructor which requires both.
        let device = vb.device();
        let gamma = Tensor::ones(embed_dim, DType::F32, device)?;
        let beta = Tensor::zeros(embed_dim, DType::F32, device)?;
        let norm = candle_nn::LayerNorm::new(gamma, beta, 1e-6);
        let linear = linear(embed_dim, embed_dim * 6, vb.pp("linear"))?;
        Ok(Self {
            linear,
            norm,
            embed_dim,
        })
    }

    /// `(x, t) → (x_normed_modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    ///
    /// Matches the reference exactly:
    /// ```python
    /// emb = linear(silu(t))
    /// shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp \
    ///     = chunk(emb, 6, dim=1)
    /// x = norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    /// ```
    fn forward(&self, x: &Tensor, t: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(&candle_nn::ops::silu(t)?)?; // (B, 6*D)
        let d = self.embed_dim;
        let shift_msa = emb.narrow(D::Minus1, 0, d)?.unsqueeze(1)?; // (B, 1, D)
        let scale_msa = emb.narrow(D::Minus1, d, d)?.unsqueeze(1)?;
        let gate_msa = emb.narrow(D::Minus1, 2 * d, d)?.unsqueeze(1)?;
        let shift_mlp = emb.narrow(D::Minus1, 3 * d, d)?.unsqueeze(1)?;
        let scale_mlp = emb.narrow(D::Minus1, 4 * d, d)?.unsqueeze(1)?;
        let gate_mlp = emb.narrow(D::Minus1, 5 * d, d)?.unsqueeze(1)?;
        let normed = self.norm.forward(x)?;
        let one = Tensor::ones_like(&scale_msa)?;
        let modulated = normed
            .broadcast_mul(&(scale_msa + one)?)?
            .broadcast_add(&shift_msa)?;
        Ok((modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

/// `AdaLayerNorm_Final(dim)` — output-side modulation by a 2-chunk
/// `(scale, shift)` linear of the time embedding. Reference:
/// ```python
/// emb = linear(silu(emb))
/// scale, shift = chunk(emb, 2, dim=1)
/// x = norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
/// ```
#[derive(Debug)]
pub(super) struct F5AdaLayerNormFinal {
    linear: Linear,
    norm: candle_nn::LayerNorm,
    embed_dim: usize,
}

impl F5AdaLayerNormFinal {
    fn new(vb: VarBuilder, embed_dim: usize) -> Result<Self> {
        let device = vb.device();
        let gamma = Tensor::ones(embed_dim, DType::F32, device)?;
        let beta = Tensor::zeros(embed_dim, DType::F32, device)?;
        let norm = candle_nn::LayerNorm::new(gamma, beta, 1e-6);
        let linear = linear(embed_dim, embed_dim * 2, vb.pp("linear"))?;
        Ok(Self {
            linear,
            norm,
            embed_dim,
        })
    }

    fn forward(&self, x: &Tensor, t: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&candle_nn::ops::silu(t)?)?;
        let d = self.embed_dim;
        let scale = emb.narrow(D::Minus1, 0, d)?.unsqueeze(1)?;
        let shift = emb.narrow(D::Minus1, d, d)?.unsqueeze(1)?;
        let normed = self.norm.forward(x)?;
        let one = Tensor::ones_like(&scale)?;
        normed.broadcast_mul(&(scale + one)?)?.broadcast_add(&shift)
    }
}

/// One F5 transformer block: AdaLN-modulated self-attention with RoPE,
/// then plain LayerNorm + AdaLN-modulated SwiGLU FFN. Both paths gate
/// the residual contribution with **plain** `gate * out` (not
/// `sigmoid(1 - gate)`).
///
/// Reference: `modules.py::DiTBlock.forward`:
/// ```python
/// norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = attn_norm(x, emb=t)
/// attn_output = attn(x=norm, mask=mask, rope=rope)
/// x = x + gate_msa.unsqueeze(1) * attn_output
/// norm = ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
/// ff_output = ff(norm)
/// x = x + gate_mlp.unsqueeze(1) * ff_output
/// ```
pub(super) struct F5DitBlock {
    attn_norm: F5AdaLayerNorm,
    attn: Attention,
    ff_norm: candle_nn::LayerNorm,
    ff: F5FeedForward,
}

impl std::fmt::Debug for F5DitBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F5DitBlock").finish_non_exhaustive()
    }
}

impl F5DitBlock {
    pub(super) fn new(vb: VarBuilder, cfg: &F5DitConfig) -> Result<Self> {
        let attn_norm = F5AdaLayerNorm::new(vb.pp("attn_norm"), cfg.hidden_dim)?;
        // F5-TTS uses split Q/K/V projections with bias and a `to_out.0`
        // output linear (diffusers `Attention` naming) — not Stable Audio's
        // fused bias-less `to_qkv`.
        let attn = Attention::new_self_split(
            cfg.hidden_dim,
            cfg.heads,
            cfg.head_dim,
            cfg.qk_norm,
            vb.pp("attn"),
        )?;
        // `ff_norm` is bias-less and weight-less in F5
        // (`elementwise_affine=False`). We use the bias-less variant
        // with a synthesized unit gamma (no safetensors entry needed).
        let device = vb.device();
        let gamma = Tensor::ones(cfg.hidden_dim, DType::F32, device)?;
        let ff_norm = candle_nn::LayerNorm::new_no_bias(gamma, 1e-6);
        let ff = F5FeedForward::new(vb.pp("ff"), cfg.hidden_dim, cfg.ff_inner())?;
        Ok(Self {
            attn_norm,
            attn,
            ff_norm,
            ff,
        })
    }

    /// Forward pass.
    ///
    /// `xs`: `(B, T, hidden_dim)`
    /// `t`: `(B, hidden_dim)` time-conditioning vector
    /// `rope_cos`, `rope_sin`: precomputed RoPE tables
    pub(super) fn forward(
        &self,
        xs: &Tensor,
        t: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let (normed, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.attn_norm.forward(xs, t)?;

        // Self-attention with RoPE.
        let attn_out = self
            .attn
            .forward(&normed, None, Some((rope_cos, rope_sin)))?;
        let xs = xs.add(&attn_out.broadcast_mul(&gate_msa)?)?;

        // Feed-forward path with AdaLN modulation on the pre-norm.
        let ff_in = self.ff_norm.forward(&xs)?;
        let one = Tensor::ones_like(&scale_mlp)?;
        let modulated = ff_in
            .broadcast_mul(&(scale_mlp + one)?)?
            .broadcast_add(&shift_mlp)?;
        let ff_out = self.ff.forward(&modulated)?;
        xs.add(&ff_out.broadcast_mul(&gate_mlp)?)
    }
}

/// F5-TTS feed-forward: `Linear(dim, inner) → GELU(tanh) → Linear(inner,
/// dim)`, both linears with bias.
///
/// This is the diffusers / F5 `FeedForward` layout — distinct from Stable
/// Audio's gated SwiGLU ([`blazen_audio_core::dit::FeedForward`]). The
/// inner `nn.Sequential` lives at `ff`; the project-in `Linear` is at
/// `ff.0.0` (followed by a paramless GELU at `ff.0.1`) and the project-out
/// `Linear` at `ff.2`. F5TTS_Base builds the GELU with `approximate="tanh"`,
/// so we use candle's tanh-approximate [`Tensor::gelu`] (not `gelu_erf`).
struct F5FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl F5FeedForward {
    fn new(vb: VarBuilder, hidden_dim: usize, inner_dim: usize) -> Result<Self> {
        let inner = vb.pp("ff");
        let proj_in = linear(hidden_dim, inner_dim, inner.pp("0").pp("0"))?;
        let proj_out = linear(inner_dim, hidden_dim, inner.pp("2"))?;
        Ok(Self { proj_in, proj_out })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.proj_in.forward(xs)?;
        let xs = xs.gelu()?;
        self.proj_out.forward(&xs)
    }
}

/// Full F5-TTS DiT.
///
/// Input/output: mel-spectrograms shaped `(B, T, mel_dim)`. The model
/// predicts the flow-matching velocity `dx/dt` at the queried timestep
/// `t \in [0, 1]`, i.e. the output has the same shape as the input.
pub struct F5Dit {
    cfg: F5DitConfig,
    time_embed: TimestepEmbedding,
    input_embed: InputEmbedding,
    blocks: Vec<F5DitBlock>,
    long_skip: Option<Linear>,
    norm_out: F5AdaLayerNormFinal,
    proj_out: Linear,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl std::fmt::Debug for F5Dit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F5Dit")
            .field("cfg", &self.cfg)
            .field("depth", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl F5Dit {
    /// Construct from a `VarBuilder` rooted at the DiT's safetensors
    /// prefix. The expected sub-key layout mirrors
    /// `f5_tts.model.backbones.dit.DiT`:
    /// - `time_embed.time_mlp.{0,2}.{weight,bias}`
    /// - `input_embed.proj.{weight,bias}`
    /// - `input_embed.conv_pos_embed.conv1d.{0,2}.{weight,bias}`
    /// - `transformer_blocks.<i>.{attn_norm,attn,ff_norm,ff}`
    /// - `long_skip_connection.weight` (only if `long_skip_connection`)
    /// - `norm_out.linear.{weight,bias}`
    /// - `proj_out.{weight,bias}`
    pub fn new(vb: VarBuilder, cfg: F5DitConfig) -> Result<Self> {
        assert_eq!(
            cfg.hidden_dim,
            cfg.heads * cfg.head_dim,
            "F5 hidden_dim ({}) must equal heads * head_dim ({} * {})",
            cfg.hidden_dim,
            cfg.heads,
            cfg.head_dim
        );
        let device = vb.device().clone();

        let time_embed =
            TimestepEmbedding::new(vb.pp("time_embed"), cfg.hidden_dim, cfg.freq_embed_dim)?;
        let input_embed = InputEmbedding::new(vb.pp("input_embed"), &cfg)?;

        let blocks_vb = vb.pp("transformer_blocks");
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(F5DitBlock::new(blocks_vb.pp(i), &cfg)?);
        }

        let long_skip = if cfg.long_skip_connection {
            Some(linear_no_bias(
                cfg.hidden_dim * 2,
                cfg.hidden_dim,
                vb.pp("long_skip_connection"),
            )?)
        } else {
            None
        };

        let norm_out = F5AdaLayerNormFinal::new(vb.pp("norm_out"), cfg.hidden_dim)?;
        let proj_out = linear(cfg.hidden_dim, cfg.mel_dim, vb.pp("proj_out"))?;

        // Full rotary over `head_dim` (no partial rotary in F5).
        let (rope_cos, rope_sin) = precompute_rope_freqs(cfg.max_seq_len, cfg.head_dim, &device)?;

        Ok(Self {
            cfg,
            time_embed,
            input_embed,
            blocks,
            long_skip,
            norm_out,
            proj_out,
            rope_cos,
            rope_sin,
        })
    }

    /// Read-only access to the active config.
    #[must_use]
    pub fn config(&self) -> &F5DitConfig {
        &self.cfg
    }

    /// Forward pass — predicts velocity `dx/dt` for flow-matching.
    ///
    /// - `x`: noisy mel `(B, T, mel_dim)` at the current sampler step.
    /// - `cond`: reference (or zeroed) mel `(B, T, mel_dim)`.
    /// - `text_embed`: precomputed text features `(B, T, text_dim)`
    ///   (already broadcast/upsampled to the audio time axis by the
    ///   Wave F.3 text encoder; this DiT only handles the transformer
    ///   stack).
    /// - `timestep`: `(B,)` flow-matching scalar in `[0, 1]`.
    ///
    /// Returns: `(B, T, mel_dim)` — predicted velocity (same shape as
    /// `x`).
    pub fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
        text_embed: &Tensor,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let (b, t_audio, mel) = x.dims3()?;
        debug_assert_eq!(mel, self.cfg.mel_dim);
        debug_assert_eq!(cond.dims3()?, (b, t_audio, mel));
        debug_assert_eq!(text_embed.dims3()?.0, b);
        debug_assert_eq!(text_embed.dims3()?.1, t_audio);
        debug_assert_eq!(text_embed.dims3()?.2, self.cfg.text_dim);

        // Time conditioning vector (B, hidden_dim).
        let t_emb = self.time_embed.forward(timestep)?;

        // Input embedding: concat-along-feature + project + conv_pos.
        let h = self.input_embed.forward(x, cond, text_embed)?;

        let residual = if self.long_skip.is_some() {
            Some(h.clone())
        } else {
            None
        };

        let mut h = h;
        for block in &self.blocks {
            h = block.forward(&h, &t_emb, &self.rope_cos, &self.rope_sin)?;
        }

        if let (Some(skip), Some(res)) = (self.long_skip.as_ref(), residual.as_ref()) {
            let cat = Tensor::cat(&[&h, res], D::Minus1)?;
            h = skip.forward(&cat)?;
        }

        let h = self.norm_out.forward(&h, &t_emb)?;
        let out = self.proj_out.forward(&h)?;
        debug_assert_eq!(out.dims3()?, (b, t_audio, self.cfg.mel_dim));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    /// Small config for fast shape tests (depth=2, hidden=32).
    fn tiny_cfg() -> F5DitConfig {
        F5DitConfig {
            mel_dim: 8,
            hidden_dim: 32,
            depth: 2,
            heads: 4,
            head_dim: 8,
            ff_mult: 2,
            text_dim: 16,
            conv_layers: 0,
            text_num_embeds: 256,
            freq_embed_dim: 16,
            conv_pos_kernel: 3,
            conv_pos_groups: 4,
            max_seq_len: 64,
            long_skip_connection: false,
            qk_norm: false,
        }
    }

    fn zeros_vb() -> (VarBuilder<'static>, Device) {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        (vb, device)
    }

    /// Asserts the F5TTS_Base defaults match the upstream config
    /// `src/f5_tts/configs/F5TTS_Base.yaml`:
    ///
    /// - `dim: 1024`
    /// - `depth: 22`
    /// - `heads: 16`
    /// - `ff_mult: 2`
    /// - `text_dim: 512`
    /// - `conv_layers: 4`
    /// - mel-spectrogram config `n_mel_channels: 100`.
    #[test]
    fn f5_base_config_matches_upstream() {
        let c = F5DitConfig::f5_base();
        assert_eq!(c.mel_dim, 100, "F5TTS_Base n_mel_channels");
        assert_eq!(c.hidden_dim, 1024, "F5TTS_Base.yaml dim");
        assert_eq!(c.depth, 22, "F5TTS_Base.yaml depth");
        assert_eq!(c.heads, 16, "F5TTS_Base.yaml heads");
        assert_eq!(c.head_dim, 64, "dim / heads = 1024 / 16");
        assert_eq!(c.hidden_dim, c.heads * c.head_dim);
        assert_eq!(c.ff_mult, 2, "F5TTS_Base.yaml ff_mult (NOT 4)");
        assert_eq!(c.text_dim, 512, "F5TTS_Base.yaml text_dim");
        assert_eq!(c.conv_layers, 4, "F5TTS_Base.yaml conv_layers");
        assert_eq!(
            c.text_num_embeds, 2545,
            "F5TTS_Base/vocab.txt line count (English ASCII + Mandarin pinyin)",
        );
        // FFN inner = dim * ff_mult = 1024 * 2 = 2048.
        assert_eq!(c.ff_inner(), 2048);
        // ConvPositionEmbedding defaults: kernel=31, groups=16.
        assert_eq!(c.conv_pos_kernel, 31);
        assert_eq!(c.conv_pos_groups, 16);
        // qk_norm null + long_skip off by default.
        assert!(!c.qk_norm);
        assert!(!c.long_skip_connection);
    }

    #[test]
    fn sinus_position_embedding_shape() -> Result<()> {
        let device = Device::Cpu;
        let sinus = SinusPositionEmbedding::new(16, &device)?;
        let t = Tensor::from_vec(vec![0.0f32, 0.25, 0.5, 0.75], 4, &device)?;
        let out = sinus.forward(&t)?;
        assert_eq!(out.dims(), &[4, 16]);
        // sin(0) = 0, cos(0) = 1 for the first row → first half zeros,
        // second half ones.
        let row0: Vec<f32> = out.narrow(0, 0, 1)?.squeeze(0)?.to_vec1()?;
        let (sin_half, cos_half) = row0.split_at(8);
        assert!(sin_half.iter().all(|v| v.abs() < 1e-5));
        assert!(cos_half.iter().all(|v| (v - 1.0).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn dit_block_forward_shape_correct() -> Result<()> {
        let (vb, device) = zeros_vb();
        let cfg = tiny_cfg();
        let block = F5DitBlock::new(vb.pp("block"), &cfg)?;
        let xs = Tensor::zeros((1, 6, cfg.hidden_dim), DType::F32, &device)?;
        let t_emb = Tensor::zeros((1, cfg.hidden_dim), DType::F32, &device)?;
        let (rope_cos, rope_sin) = precompute_rope_freqs(cfg.max_seq_len, cfg.head_dim, &device)?;
        let out = block.forward(&xs, &t_emb, &rope_cos, &rope_sin)?;
        assert_eq!(out.dims(), &[1, 6, cfg.hidden_dim]);
        Ok(())
    }

    #[test]
    fn dit_forward_shape_correct() -> Result<()> {
        let (vb, device) = zeros_vb();
        let cfg = tiny_cfg();
        let dit = F5Dit::new(vb, cfg.clone())?;
        let x = Tensor::zeros((2, 6, cfg.mel_dim), DType::F32, &device)?;
        let cond = Tensor::zeros((2, 6, cfg.mel_dim), DType::F32, &device)?;
        let text = Tensor::zeros((2, 6, cfg.text_dim), DType::F32, &device)?;
        let timestep = Tensor::zeros(2, DType::F32, &device)?;
        let out = dit.forward(&x, &cond, &text, &timestep)?;
        assert_eq!(out.dims(), &[2, 6, cfg.mel_dim]);
        Ok(())
    }

    #[test]
    fn velocity_prediction_shape_matches_input() -> Result<()> {
        let (vb, device) = zeros_vb();
        let cfg = tiny_cfg();
        let dit = F5Dit::new(vb, cfg.clone())?;
        for &t_audio in &[4usize, 8, 16] {
            let x = Tensor::zeros((1, t_audio, cfg.mel_dim), DType::F32, &device)?;
            let cond = Tensor::zeros((1, t_audio, cfg.mel_dim), DType::F32, &device)?;
            let text = Tensor::zeros((1, t_audio, cfg.text_dim), DType::F32, &device)?;
            let timestep = Tensor::zeros(1, DType::F32, &device)?;
            let out = dit.forward(&x, &cond, &text, &timestep)?;
            assert_eq!(
                out.dims(),
                x.dims(),
                "velocity dx/dt must have same shape as input mel (T_audio={t_audio})"
            );
        }
        Ok(())
    }

    #[test]
    fn dit_long_skip_changes_no_shape() -> Result<()> {
        // Sanity: enabling the long-skip path keeps the output shape.
        let (vb, device) = zeros_vb();
        let mut cfg = tiny_cfg();
        cfg.long_skip_connection = true;
        let dit = F5Dit::new(vb, cfg.clone())?;
        let x = Tensor::zeros((1, 4, cfg.mel_dim), DType::F32, &device)?;
        let cond = Tensor::zeros((1, 4, cfg.mel_dim), DType::F32, &device)?;
        let text = Tensor::zeros((1, 4, cfg.text_dim), DType::F32, &device)?;
        let timestep = Tensor::zeros(1, DType::F32, &device)?;
        let out = dit.forward(&x, &cond, &text, &timestep)?;
        assert_eq!(out.dims(), &[1, 4, cfg.mel_dim]);
        Ok(())
    }
}
