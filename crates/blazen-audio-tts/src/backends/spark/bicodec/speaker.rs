//! Speaker encoder tower used by `BiCodec` (sub-wave **S.2.1.e** of the
//! [`super`] `BiCodec` port).
//!
//! Mirrors three upstream files:
//!
//! * `sparktts/modules/speaker/ecapa_tdnn.py` — the ECAPA-TDNN-GLOB-c512
//!   front-end that turns a `(B, feat_dim, T_mel)` mel-spectrogram chunk
//!   into a `(B, embed_dim)` *x-vector* plus a `(B, 3·channels, T_mel)`
//!   pre-pool *latent*.
//! * `sparktts/modules/speaker/perceiver_encoder.py` — a
//!   `PerceiverResampler` that pools the variable-length ECAPA latent
//!   into a fixed-length `(B, num_latents, dim)` global representation.
//!   This is the (truncated) `naturalspeech2-pytorch` perceiver, ported
//!   in inference-only form.
//! * `sparktts/modules/speaker/speaker_encoder.py` — the wrapper that
//!   chains ECAPA → `PerceiverResampler` → [`super::quantizer::ResidualFsq`]
//!   → `Linear(num_latents * latent_dim → out_dim)` and exposes
//!   `tokenize` / `detokenize` for the `BiCodec` global stream.
//!
//! # Channel layout (diverges from upstream by one transpose)
//!
//! Upstream `ECAPA_TDNN.forward` starts with `x = x.permute(0, 2, 1)`,
//! treating its input as channels-last `(B, T_mel, feat_dim)`. The only
//! consumer ([`SpeakerEncoder`]) is itself called from
//! `BiCodec.tokenize` via `self.speaker_encoder.tokenize(mel.transpose(1,
//! 2))` — i.e. mels start channels-first `(B, feat_dim, T_mel)` and get
//! transposed to channels-last just so ECAPA can transpose them back.
//!
//! We hoist that round-trip out and accept channels-first
//! `(B, feat_dim, T_mel)` directly in [`EcapaTdnnGlobC512::forward`] and
//! [`SpeakerEncoder::tokenize`] / [`SpeakerEncoder::forward`]. End-to-end
//! semantics are unchanged because the two transposes cancel.
//!
//! # Inference-only port
//!
//! The upstream training-time machinery is dropped: no dropout (it's
//! identity at inference), no `BatchNorm` running-stats update path
//! (we always read `running_mean` / `running_var` via `forward_t(_,
//! false)`), no `init_weights` (we load from `state_dict`),
//! `quantize_dropout=False` on the inner `ResidualFSQ`.
//!
//! # State-dict key paths (Spark-TTS checkpoint)
//!
//! * `speaker_encoder` → [`EcapaTdnnGlobC512`]
//!   * `layer1.conv.weight` / `.bias`, `layer1.bn.{weight,bias,running_mean,running_var}`
//!   * `layer{2,3,4}.se_res2block.0.{conv,bn}.*` —
//!     [`Conv1dReluBn`] (1×1 head)
//!   * `layer{2,3,4}.se_res2block.1.convs.{i}.{weight,bias}` and
//!     `layer{2,3,4}.se_res2block.1.bns.{i}.*` — [`Res2Conv1dReluBn`]
//!   * `layer{2,3,4}.se_res2block.2.{conv,bn}.*` —
//!     [`Conv1dReluBn`] (1×1 tail)
//!   * `layer{2,3,4}.se_res2block.3.linear{1,2}.{weight,bias}` —
//!     [`SeConnect`]
//!   * `conv.{weight,bias}` — the 1×1 conv after `cat([out2, out3,
//!     out4])`
//!   * `pool.linear{1,2}.{weight,bias}` — [`Astp`]
//!   * `bn.{weight,bias,running_mean,running_var}` — post-pool
//!     `BatchNorm1d(3072)`
//!   * `linear.{weight,bias}` — `Linear(3072 → embed_dim)`
//!   * `bn2.*` — `BatchNorm1d(embed_dim)` (only present when
//!     `emb_bn=True`; upstream Spark-TTS leaves it at `False` so this
//!     key prefix is **absent** and we use `nn.Identity()` semantics).
//! * `perceiver_sampler` → [`PerceiverResampler`]
//!   * `proj_context.{weight,bias}` — only present when
//!     `dim_context != dim` (Spark-TTS: `1536 ≠ 128` → present)
//!   * `latents` — the learned latent token bank, shape `(32, 128)`
//!   * `layers.{i}.0.to_q.weight`, `layers.{i}.0.to_kv.weight`,
//!     `layers.{i}.0.to_out.weight` (no biases — `nn.Linear(..., bias=False)`)
//!   * `layers.{i}.1.0.weight,bias` — `LayerNorm` heading the
//!     `FeedForward`
//!   * `layers.{i}.1.1.weight,bias` — `Linear(dim, dim_inner * 2)`
//!   * `layers.{i}.1.3.weight,bias` — `Linear(dim_inner, dim)` (index
//!     3 because index 2 is `GEGLU`, which carries no parameters; the
//!     optional `CausalConv1d` slot is `None` for the resampler so the
//!     `Sequential` collapses `[LayerNorm, Linear, GEGLU, Linear]`).
//!   * `norm.gamma` — final [`RmsNorm`]
//! * `quantizer` → [`super::quantizer::ResidualFsq`]
//!   * `project_in.{weight,bias}` — `Linear(128 → 6)`
//!   * `project_out.{weight,bias}` — `Linear(6 → 128)`
//!   * `layers.0.…` — the lone inner FSQ has no projections (it sees
//!     `dim == codebook_dim == 6`)
//! * `project.{weight,bias}` — `Linear(num_latents * latent_dim → out_dim)`,
//!   i.e. `Linear(32 * 128 = 4096 → 1024)` for the Spark-TTS checkpoint
//!   (`out_dim=1024`, `latent_dim=128`, `token_num=32`).
//!
//! # `RMSNorm` formulation
//!
//! Upstream's `RMSNorm` is
//! `F.normalize(x, dim=-1) * sqrt(dim) * gamma`. Algebraically that's
//! `x / sqrt(sum(x^2)) * sqrt(dim) * gamma = x / sqrt(mean(x^2)) *
//! gamma`, i.e. the standard `RMSNorm` (no eps inside the rsqrt, just
//! `F.normalize`'s built-in `eps=1e-12`). We mirror this form verbatim
//! so we don't have to chase a `gamma` rename — the state-dict key is
//! `norm.gamma` rather than `norm.weight`.
//!
//! # `ASTP` variance numerics
//!
//! Upstream uses `var = sum(alpha * x^2) - mean^2`, the algebraically
//! convenient but numerically unstable form (subtraction of two
//! near-equal quantities). We replicate it bit-for-bit rather than
//! swapping in the stabler `sum(alpha * (x - mean)^2)` — the goal is
//! checkpoint-faithful inference, not improved numerics.

// `VarBuilder` is the canonical "consume by value" handle in candle.
// See primitives.rs for the same lint waiver and rationale.
#![allow(clippy::needless_pass_by_value)]
// The parent `bicodec/mod.rs` already applies `#[allow(dead_code)]` to
// `pub(super) mod speaker;` because the surface is consumed by the
// top-level BiCodec wiring in S.2.1.g — duplicating the attribute here
// trips `clippy::duplicated_attributes`. Other lints we need to silence
// at file scope:
//
// * `too_many_arguments` — three `load` constructors take 8 args (vb +
//   a handful of conv-layer shape knobs). Splitting them into config
//   structs would obscure parity with the upstream Python signatures.
// * `struct_field_names` — `SpeakerEncoder::speaker_encoder` matches
//   the upstream Python field name (`self.speaker_encoder`); renaming
//   would diverge from the state-dict key path.
// * `similar_names` — `layer_vb` vs `layers_vb` is the canonical
//   `pp(...)` chain pattern across this crate.
// * `many_single_char_names` — `Attention::forward(b, nq, _d)` mirrors
//   the upstream Python einsum naming directly.
#![allow(
    clippy::too_many_arguments,
    clippy::struct_field_names,
    clippy::similar_names,
    clippy::many_single_char_names,
    reason = "see file-scope comment above — preserve upstream Python parity"
)]

use candle_core::{D, Module, ModuleT, Result, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder, batch_norm,
    conv1d, layer_norm, linear, linear_no_bias, ops::softmax,
};

use super::quantizer::ResidualFsq;

// ---------------------------------------------------------------------------
// Conv1dReluBn
// ---------------------------------------------------------------------------

/// `Conv1d → ReLU → BatchNorm1d`. Mirrors upstream
/// `ecapa_tdnn.py::Conv1dReluBn`. State-dict children: `conv.{weight,
/// bias}` + `bn.{weight, bias, running_mean, running_var}`.
#[derive(Debug, Clone)]
struct Conv1dReluBn {
    conv: Conv1d,
    bn: BatchNorm,
}

impl Conv1dReluBn {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
            ..Conv1dConfig::default()
        };
        let conv = conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let bn = batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?.relu()?;
        // Always eval mode (`train=false`) — no running-stats update at
        // inference.
        self.bn.forward_t(&y, false)
    }
}

// ---------------------------------------------------------------------------
// Res2Conv1dReluBn
// ---------------------------------------------------------------------------

/// Res2Net-style `Conv1d → ReLU → BatchNorm` chain over `scale - 1`
/// chunks of the channel dim (the final chunk is passed through
/// untouched). Mirrors upstream `ecapa_tdnn.py::Res2Conv1dReluBn`.
///
/// `torch.split(x, self.width, 1)` splits dim 1 into chunks of `width`
/// each — for `channels % scale == 0` this yields exactly `scale`
/// chunks of width `width = channels / scale`. We reproduce that with
/// `Tensor::chunk(scale, 1)` because every Spark-TTS configuration uses
/// `channels = 512` and `scale = 8` (`512 % 8 == 0`).
///
/// State-dict children: `convs.{i}.{weight, bias}` and
/// `bns.{i}.{weight, bias, running_mean, running_var}` for `i in 0 ..
/// nums` where `nums = scale - 1`.
#[derive(Debug, Clone)]
struct Res2Conv1dReluBn {
    convs: Vec<Conv1d>,
    bns: Vec<BatchNorm>,
    scale: usize,
    nums: usize,
    width: usize,
}

impl Res2Conv1dReluBn {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        scale: usize,
    ) -> Result<Self> {
        if !channels.is_multiple_of(scale) {
            candle_core::bail!(
                "Res2Conv1dReluBn::load: channels {channels} must be divisible by scale {scale}"
            );
        }
        let width = channels / scale;
        let nums = if scale == 1 { 1 } else { scale - 1 };

        let convs_vb = vb.pp("convs");
        let bns_vb = vb.pp("bns");
        let mut convs = Vec::with_capacity(nums);
        let mut bns = Vec::with_capacity(nums);
        for i in 0..nums {
            let cfg = Conv1dConfig {
                padding,
                stride,
                dilation,
                groups: 1,
                ..Conv1dConfig::default()
            };
            convs.push(conv1d(width, width, kernel_size, cfg, convs_vb.pp(i))?);
            bns.push(batch_norm(width, BatchNormConfig::default(), bns_vb.pp(i))?);
        }
        Ok(Self {
            convs,
            bns,
            scale,
            nums,
            width,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `Tensor::chunk(scale, 1)` mirrors `torch.split(x, self.width,
        // 1)` exactly when `channels % scale == 0` (which we enforce in
        // `load`). All Spark-TTS use sites satisfy that constraint.
        let spx = x.chunk(self.scale, 1)?;
        let mut out: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut sp = spx[0].clone();
        for (i, (conv, bn)) in self.convs.iter().zip(self.bns.iter()).enumerate() {
            if i >= 1 {
                sp = (sp + &spx[i])?;
            }
            sp = conv.forward(&sp)?;
            sp = bn.forward_t(&sp.relu()?, false)?;
            out.push(sp.clone());
        }
        if self.scale != 1 {
            out.push(spx[self.nums].clone());
        }
        Tensor::cat(&out, 1)
    }
}

// ---------------------------------------------------------------------------
// SeConnect
// ---------------------------------------------------------------------------

/// Squeeze-excitation block: temporal-average pool → bottleneck MLP →
/// sigmoid gate → per-channel re-scale of `x`. Mirrors upstream
/// `ecapa_tdnn.py::SE_Connect`.
///
/// State-dict children: `linear1.{weight, bias}`, `linear2.{weight,
/// bias}`.
#[derive(Debug, Clone)]
struct SeConnect {
    linear1: Linear,
    linear2: Linear,
}

impl SeConnect {
    fn load(vb: VarBuilder, channels: usize, bottleneck: usize) -> Result<Self> {
        let linear1 = linear(channels, bottleneck, vb.pp("linear1"))?;
        let linear2 = linear(bottleneck, channels, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T)
        let pooled = x.mean(D::Minus1)?; // (B, C)
        let h = self.linear1.forward(&pooled)?.relu()?;
        let gate = candle_nn::ops::sigmoid(&self.linear2.forward(&h)?)?; // (B, C)
        // Broadcast (B, C) → (B, C, 1) → (B, C, T).
        let gate = gate.unsqueeze(D::Minus1)?;
        x.broadcast_mul(&gate)
    }
}

// ---------------------------------------------------------------------------
// SeRes2Block
// ---------------------------------------------------------------------------

/// `x + Sequential(Conv1dReluBn, Res2Conv1dReluBn, Conv1dReluBn,
/// SE_Connect)(x)`. Mirrors upstream `ecapa_tdnn.py::SE_Res2Block`.
///
/// State-dict children live under `se_res2block.{0..3}`, matching
/// upstream's `nn.Sequential` ordering:
///
/// * `se_res2block.0` → [`Conv1dReluBn`] (1×1 head)
/// * `se_res2block.1` → [`Res2Conv1dReluBn`]
/// * `se_res2block.2` → [`Conv1dReluBn`] (1×1 tail)
/// * `se_res2block.3` → [`SeConnect`]
#[derive(Debug, Clone)]
struct SeRes2Block {
    head: Conv1dReluBn,
    res2: Res2Conv1dReluBn,
    tail: Conv1dReluBn,
    se: SeConnect,
}

impl SeRes2Block {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        scale: usize,
        se_bottleneck: usize,
    ) -> Result<Self> {
        let inner = vb.pp("se_res2block");
        let head = Conv1dReluBn::load(inner.pp("0"), channels, channels, 1, 1, 0, 1, 1)?;
        let res2 = Res2Conv1dReluBn::load(
            inner.pp("1"),
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            scale,
        )?;
        let tail = Conv1dReluBn::load(inner.pp("2"), channels, channels, 1, 1, 0, 1, 1)?;
        let se = SeConnect::load(inner.pp("3"), channels, se_bottleneck)?;
        Ok(Self {
            head,
            res2,
            tail,
            se,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.head.forward(x)?;
        let y = self.res2.forward(&y)?;
        let y = self.tail.forward(&y)?;
        let y = self.se.forward(&y)?;
        x + y
    }
}

// ---------------------------------------------------------------------------
// ASTP — attentive statistics pooling
// ---------------------------------------------------------------------------

/// Attentive statistics pooling with optional global context.
/// Mirrors upstream `pooling_layers.py::ASTP`.
///
/// Output dimension is `2 * in_dim` — the concatenation of an
/// attention-weighted mean and standard deviation along the time axis.
///
/// State-dict children: `linear1.{weight, bias}` (`Conv1d` 1×1,
/// `in_dim * 3 → bottleneck` when `global_context_att=True`),
/// `linear2.{weight, bias}` (`Conv1d` 1×1, `bottleneck → in_dim`).
///
/// # Variance numerics
///
/// Upstream computes `var = sum(alpha * x^2) - mean^2`. This is the
/// algebraically convenient but numerically unstable form — for
/// near-constant inputs the subtraction can underflow into negatives
/// (which the `clamp(min=1e-7)` mitigates). We mirror the upstream form
/// verbatim to preserve bit-exact parity with the Python reference,
/// rather than swapping in the more stable `sum(alpha * (x - mean)^2)`.
#[derive(Debug, Clone)]
struct Astp {
    linear1: Conv1d,
    linear2: Conv1d,
    global_context_att: bool,
    in_dim: usize,
}

impl Astp {
    fn load(
        vb: VarBuilder,
        in_dim: usize,
        bottleneck: usize,
        global_context_att: bool,
    ) -> Result<Self> {
        let linear1_in = if global_context_att {
            in_dim * 3
        } else {
            in_dim
        };
        let cfg = Conv1dConfig::default(); // kernel=1, no pad, stride=1
        let linear1 = conv1d(linear1_in, bottleneck, 1, cfg, vb.pp("linear1"))?;
        let linear2 = conv1d(bottleneck, in_dim, 1, cfg, vb.pp("linear2"))?;
        Ok(Self {
            linear1,
            linear2,
            global_context_att,
            in_dim,
        })
    }

    fn out_dim(&self) -> usize {
        2 * self.in_dim
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, in_dim, T).
        let x_in = if self.global_context_att {
            // mean / std over time, then expand back to (B, in_dim, T)
            // and concat along channels → (B, 3*in_dim, T).
            let t = x.dim(D::Minus1)?;
            let context_mean = x.mean_keepdim(D::Minus1)?; // (B, in_dim, 1)
            let context_var = x.var_keepdim(D::Minus1)?; // (B, in_dim, 1) — unbiased (matches torch.var default).
            let context_std = (context_var + 1e-7_f64)?.sqrt()?;
            let context_mean = context_mean.broadcast_as(x.shape())?;
            let context_std = context_std.broadcast_as(x.shape())?;
            let _ = t;
            Tensor::cat(&[x, &context_mean, &context_std], 1)?
        } else {
            x.clone()
        };

        let alpha = self.linear1.forward(&x_in)?.tanh()?;
        let alpha = softmax(&self.linear2.forward(&alpha)?, 2)?; // (B, in_dim, T)

        let weighted = (&alpha * x)?;
        let mean = weighted.sum(D::Minus1)?; // (B, in_dim)
        let weighted_sq = (&alpha * x.sqr()?)?;
        let mean_sq = mean.sqr()?;
        let raw_var = (weighted_sq.sum(D::Minus1)? - &mean_sq)?;
        // clamp(min=1e-7) — upstream uses 1e-7 explicitly.
        let var = raw_var.maximum(1e-7_f64)?;
        let std = var.sqrt()?;
        Tensor::cat(&[&mean, &std], 1)
    }
}

// ---------------------------------------------------------------------------
// ECAPA-TDNN-GLOB-c512
// ---------------------------------------------------------------------------

/// ECAPA-TDNN with the `GLOB_c512` configuration: `channels = 512`,
/// `global_context_att = True`. Mirrors upstream
/// `ecapa_tdnn.py::ECAPA_TDNN_GLOB_c512`.
///
/// # Channel convention
///
/// Upstream's `forward` starts with `x = x.permute(0, 2, 1)`, treating
/// its input as channels-last `(B, T, F)`. We accept channels-first
/// `(B, F, T)` directly — the upstream transpose is just there to undo
/// `BiCodec.tokenize`'s own `mel.transpose(1, 2)`. See the module
/// docstring for the full rationale.
///
/// # Returned latent
///
/// Note this is the *pre-pool* ECAPA latent
/// (`F.relu(self.conv(cat([out2, out3, out4])))`) — shape
/// `(B, 3 * channels, T_mel)`. It feeds the [`PerceiverResampler`].
/// Upstream calls this `latent` and returns it via the
/// `return_latent=True` arm of `ECAPA_TDNN.forward`.
#[derive(Debug, Clone)]
pub(super) struct EcapaTdnnGlobC512 {
    layer1: Conv1dReluBn,
    layer2: SeRes2Block,
    layer3: SeRes2Block,
    layer4: SeRes2Block,
    conv: Conv1d, // 1×1, 3*channels → 3*channels
    pool: Astp,
    bn: BatchNorm, // post-pool BN over 2 * 3 * channels
    linear: Linear,
    channels: usize,
}

impl EcapaTdnnGlobC512 {
    /// Load an ECAPA-TDNN-GLOB-c512. Pass `vb` rooted at the
    /// `speaker_encoder` child of the `SpeakerEncoder` state dict.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any of the child loaders.
    pub(super) fn load(
        vb: VarBuilder,
        feat_dim: usize,
        embed_dim: usize,
        channels: usize,
    ) -> Result<Self> {
        let layer1 = Conv1dReluBn::load(vb.pp("layer1"), feat_dim, channels, 5, 1, 2, 1, 1)?;
        let layer2 = SeRes2Block::load(vb.pp("layer2"), channels, 3, 1, 2, 2, 8, 128)?;
        let layer3 = SeRes2Block::load(vb.pp("layer3"), channels, 3, 1, 3, 3, 8, 128)?;
        let layer4 = SeRes2Block::load(vb.pp("layer4"), channels, 3, 1, 4, 4, 8, 128)?;

        let cat_channels = channels * 3;
        let out_channels = 512 * 3; // upstream hard-codes 512*3 here regardless of `channels`.
        let conv = conv1d(
            cat_channels,
            out_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv"),
        )?;
        let pool = Astp::load(vb.pp("pool"), out_channels, 128, true)?;
        let bn = batch_norm(pool.out_dim(), BatchNormConfig::default(), vb.pp("bn"))?;
        let linear = linear(pool.out_dim(), embed_dim, vb.pp("linear"))?;
        Ok(Self {
            layer1,
            layer2,
            layer3,
            layer4,
            conv,
            pool,
            bn,
            linear,
            channels,
        })
    }

    /// Channel count of the trunk (the `c512` in `ECAPA_TDNN_GLOB_c512`).
    pub(super) fn channels(&self) -> usize {
        self.channels
    }

    /// Forward `(B, feat_dim, T_mel) → (x_vector: (B, embed_dim),
    /// latent: (B, 3 * 512, T_mel))`.
    ///
    /// The latent's channel count is hard-coded at `3 * 512 = 1536`
    /// rather than `3 * channels` because upstream's `nn.Conv1d` output
    /// dim is literally `512 * 3` (line 177 of `ecapa_tdnn.py`),
    /// independent of the constructor's `channels` argument. The
    /// shipping Spark-TTS checkpoint always uses `channels = 512` so
    /// the two coincide.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any of the four trunk
    /// layers, the 1×1 `conv`, the pooling, the post-pool BN, or the
    /// final `linear`.
    pub(super) fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: (B, feat_dim, T_mel), channels-first — see module docs.
        let out1 = self.layer1.forward(x)?;
        let out2 = self.layer2.forward(&out1)?;
        let out3 = self.layer3.forward(&out2)?;
        let out4 = self.layer4.forward(&out3)?;

        let cat = Tensor::cat(&[&out2, &out3, &out4], 1)?;
        let latent = self.conv.forward(&cat)?.relu()?; // (B, 3*512, T_mel)

        let pooled = self.pool.forward(&latent)?; // (B, 2 * 3 * 512)
        // BN expects channels in dim 1 — pooled is (B, C), so we
        // unsqueeze a fake time axis, run BN, squeeze.
        let bn_in = pooled.unsqueeze(D::Minus1)?;
        let bn_out = self.bn.forward_t(&bn_in, false)?;
        let bn_out = bn_out.squeeze(D::Minus1)?;
        let x_vector = self.linear.forward(&bn_out)?;
        Ok((x_vector, latent))
    }
}

// ---------------------------------------------------------------------------
// PerceiverResampler primitives — RMSNorm, GEGLU FF, cross-attention
// ---------------------------------------------------------------------------

/// `F.normalize(x, dim=-1) * sqrt(dim) * gamma` — the upstream
/// formulation of `RMSNorm` from `perceiver_encoder.py::RMSNorm`. We
/// preserve the `gamma` field name verbatim so the state-dict key
/// matches (`norm.gamma`, not `norm.weight`).
#[derive(Debug, Clone)]
struct RmsNorm {
    gamma: Tensor,
    scale: f64,
    dim: usize,
}

impl RmsNorm {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        #[allow(
            clippy::cast_precision_loss,
            reason = "dim is the latent feature size (Spark-TTS: 128); \
                      exact in f64."
        )]
        let scale = (dim as f64).sqrt();
        Ok(Self { gamma, scale, dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // F.normalize(x, dim=-1, eps=1e-12) = x / max(||x||_2, eps)
        let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let denom = (norm + 1e-12_f64)?;
        let normalized = x.broadcast_div(&denom)?;
        // multiply by sqrt(dim) * gamma (broadcast on last dim)
        let scaled = (normalized * self.scale)?;
        scaled.broadcast_mul(&self.gamma)
    }
}

/// `GEGLU` activation as `F.gelu(gate) * x` where `gate, x` are the
/// two halves of the last dim. Mirrors upstream
/// `perceiver_encoder.py::GEGLU`.
fn geglu(x: &Tensor) -> Result<Tensor> {
    let parts = x.chunk(2, D::Minus1)?;
    if parts.len() != 2 {
        candle_core::bail!(
            "geglu: expected last dim to split into exactly 2 chunks, got {}",
            parts.len()
        );
    }
    let value = &parts[0];
    let gate = &parts[1];
    // F.gelu defaults to the erf form; `gelu_erf` is the matching candle
    // op (`Tensor::gelu` is the tanh approximation).
    let gated = gate.gelu_erf()?;
    value.broadcast_mul(&gated)
}

/// `Sequential(Linear(dim, dim_inner * 2), GEGLU, Linear(dim_inner,
/// dim))` — the upstream `FeedForward` from
/// `perceiver_encoder.py::FeedForward` with `causal_conv=False` (the
/// only path the `PerceiverResampler` ever takes).
///
/// `dim_inner = int(dim * mult * 2 / 3)` matches upstream verbatim
/// (line 239) — e.g. `dim=128, mult=4` → `dim_inner = 341`.
///
/// # Pre-norm wrapping
///
/// Upstream `PerceiverResampler` wraps every attention/FF block in a
/// residual but does **not** prepend a `LayerNorm` — the FF itself
/// carries no explicit pre-norm in `perceiver_encoder.py`. State-dict
/// layout under `layers.{i}.1`:
///
/// * `layers.{i}.1.0.{weight, bias}` — `Linear(dim, dim_inner * 2)`
/// * `layers.{i}.1.2.{weight, bias}` — `Linear(dim_inner, dim)` (index
///   2 because index 1 is the `GEGLU` carrying no parameters)
///
/// Note: the upstream `Sequential` helper filters out `None`, so when
/// `causal_conv=False` the layout collapses to just two `Linear` slots
/// separated by the `GEGLU` placeholder.
#[derive(Debug, Clone)]
struct PerceiverFeedForward {
    inner_in: Linear,
    inner_out: Linear,
}

impl PerceiverFeedForward {
    fn load(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        // dim_inner = int(dim * mult * 2 / 3) — integer truncation
        // matches Python's `int(...)`.
        let dim_inner = (dim * mult * 2) / 3;
        let inner_in = linear(dim, dim_inner * 2, vb.pp("0"))?;
        let inner_out = linear(dim_inner, dim, vb.pp("2"))?;
        Ok(Self {
            inner_in,
            inner_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.inner_in.forward(x)?;
        let y = geglu(&y)?;
        self.inner_out.forward(&y)
    }
}

/// Multi-head cross-attention with the `cross_attn_include_queries=True`
/// flag set, matching the upstream `PerceiverResampler` configuration
/// (`perceiver_encoder.py::Attention`).
///
/// Forward: `to_q(x)` on the latents, `to_kv(cat([x, context], dim=-2))`
/// on the *queries-prepended* context, no causal mask, no dropout, no
/// pre-norm. All three projections are bias-free Linears.
///
/// State-dict children: `to_q.weight`, `to_kv.weight`, `to_out.weight`
/// (no biases).
#[derive(Debug, Clone)]
struct CrossAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
    scale: f64,
}

impl CrossAttention {
    fn load(
        vb: VarBuilder,
        dim: usize,
        dim_context: usize,
        heads: usize,
        dim_head: usize,
    ) -> Result<Self> {
        let dim_inner = dim_head * heads;
        let to_q = linear_no_bias(dim, dim_inner, vb.pp("to_q"))?;
        let to_kv = linear_no_bias(dim_context, dim_inner * 2, vb.pp("to_kv"))?;
        let to_out = linear_no_bias(dim_inner, dim, vb.pp("to_out"))?;
        #[allow(
            clippy::cast_precision_loss,
            reason = "dim_head is the per-head feature size (Spark-TTS: 64); \
                      exact in f64."
        )]
        let scale = 1.0_f64 / (dim_head as f64).sqrt();
        Ok(Self {
            to_q,
            to_kv,
            to_out,
            heads,
            dim_head,
            scale,
        })
    }

    /// `x`: `(B, N_q, dim)`. `context`: `(B, N_ctx, dim_context)`.
    /// Returns `(B, N_q, dim)`.
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, nq, _d) = x.dims3()?;
        // cross_attn_include_queries=True → prepend `x` to context.
        let context = Tensor::cat(&[x, context], 1)?;

        let q = self.to_q.forward(x)?; // (B, N_q, dim_inner)
        let kv = self.to_kv.forward(&context)?; // (B, N_ctx', dim_inner * 2)
        let kv_parts = kv.chunk(2, D::Minus1)?;
        let k = &kv_parts[0];
        let v = &kv_parts[1];

        // Reshape to (B, heads, N, dim_head).
        let nk = k.dim(1)?;
        let q = q
            .reshape((b, nq, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, nk, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, nk, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention: (B, H, N_q, N_k) = (Q @ K^T) * scale.
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * self.scale)?;
        let attn = softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // (B, H, N_q, dim_head)

        // (B, H, N_q, dim_head) → (B, N_q, H * dim_head)
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, nq, self.heads * self.dim_head))?;
        self.to_out.forward(&out)
    }
}

/// One (`CrossAttention`, `FeedForward`) residual pair from the
/// `PerceiverResampler` stack.
#[derive(Debug, Clone)]
struct PerceiverLayer {
    attn: CrossAttention,
    ff: PerceiverFeedForward,
}

// ---------------------------------------------------------------------------
// PerceiverResampler
// ---------------------------------------------------------------------------

/// Naturalspeech2-style perceiver resampler. Mirrors upstream
/// `perceiver_encoder.py::PerceiverResampler` in its
/// `use_flash_attn=False, mask=None` configuration (the only path
/// `SpeakerEncoder` invokes).
///
/// # Channels-last input
///
/// The resampler is intrinsically channels-last (sequence-first):
/// inputs are `(B, T_ctx, dim_context)`. The upstream `SpeakerEncoder`
/// transposes the ECAPA latent `(B, 1536, T_mel) → (B, T_mel, 1536)`
/// before calling this. We mirror that — see [`SpeakerEncoder::forward`].
///
/// State-dict children:
///
/// * `proj_context.{weight, bias}` — only present when
///   `dim_context != dim`.
/// * `latents` — shape `(num_latents, dim)`.
/// * `layers.{i}.0.…` — [`CrossAttention`]
/// * `layers.{i}.1.0.…` and `layers.{i}.1.2.…` —
///   [`PerceiverFeedForward`]
/// * `norm.gamma` — final [`RmsNorm`]
#[derive(Debug, Clone)]
pub(super) struct PerceiverResampler {
    proj_context: Option<Linear>,
    latents: Tensor, // (num_latents, dim)
    layers: Vec<PerceiverLayer>,
    norm: RmsNorm,
    dim: usize,
    num_latents: usize,
}

impl PerceiverResampler {
    /// Load a `PerceiverResampler`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the child loaders, or
    /// from a missing `latents` parameter / `proj_context` arm.
    pub(super) fn load(
        vb: VarBuilder,
        dim: usize,
        depth: usize,
        dim_context: usize,
        num_latents: usize,
        heads: usize,
        dim_head: usize,
        ff_mult: usize,
    ) -> Result<Self> {
        let proj_context = if dim_context == dim {
            None
        } else {
            Some(linear(dim_context, dim, vb.pp("proj_context"))?)
        };

        let latents = vb.get((num_latents, dim), "latents")?;

        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let layer_vb = layers_vb.pp(i);
            // Inside `nn.ModuleList([Attention, FeedForward])` the
            // children are addressed by integer index: ".0" / ".1".
            let attn = CrossAttention::load(layer_vb.pp("0"), dim, dim, heads, dim_head)?;
            let ff = PerceiverFeedForward::load(layer_vb.pp("1"), dim, ff_mult)?;
            layers.push(PerceiverLayer { attn, ff });
        }
        let norm = RmsNorm::load(vb.pp("norm"), dim)?;
        Ok(Self {
            proj_context,
            latents,
            layers,
            norm,
            dim,
            num_latents,
        })
    }

    /// Outer feature dim of the latents.
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// Number of learned latent tokens (= sequence length of the
    /// resampler output).
    pub(super) fn num_latents(&self) -> usize {
        self.num_latents
    }

    /// Forward `(B, T_ctx, dim_context) → (B, num_latents, dim)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the optional
    /// `proj_context`, the per-layer attention/FF, or the final
    /// `RmsNorm`.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _t, _dc) = x.dims3()?;

        let context = if let Some(pc) = &self.proj_context {
            pc.forward(x)?
        } else {
            x.clone()
        };

        // repeat(latents, "n d -> b n d", b=batch) — unsqueeze then
        // expand the batch axis. Broadcasted to a contiguous tensor so
        // downstream matmuls work in place.
        let latents = self
            .latents
            .unsqueeze(0)?
            .broadcast_as((b, self.num_latents, self.dim))?
            .contiguous()?;

        let mut latents = latents;
        for layer in &self.layers {
            latents = (layer.attn.forward(&latents, &context)? + &latents)?;
            latents = (layer.ff.forward(&latents)? + &latents)?;
        }
        self.norm.forward(&latents)
    }
}

// ---------------------------------------------------------------------------
// SpeakerEncoder
// ---------------------------------------------------------------------------

/// Configuration for [`SpeakerEncoder::load`]. Field names match
/// upstream `SpeakerEncoder.__init__` kwargs verbatim.
#[derive(Debug, Clone)]
pub(super) struct SpeakerEncoderConfig {
    /// Mel feature dim (upstream `input_dim`). Spark-TTS: 128.
    pub input_dim: usize,
    /// D-vector / x-vector output dim (upstream `out_dim`). Spark-TTS:
    /// 1024.
    pub out_dim: usize,
    /// Perceiver latent feature width (upstream `latent_dim`). Spark-TTS: 128.
    pub latent_dim: usize,
    /// Number of perceiver latents (upstream `token_num`). Spark-TTS: 32.
    pub token_num: usize,
    /// Per-axis FSQ levels (upstream `fsq_levels`). Spark-TTS:
    /// `[4, 4, 4, 4, 4, 4]`.
    pub fsq_levels: Vec<u32>,
    /// Residual-FSQ layer count (upstream `fsq_num_quantizers`).
    /// Spark-TTS: 1.
    pub fsq_num_quantizers: usize,
    /// ECAPA-TDNN trunk channels. Spark-TTS: 512.
    pub ecapa_channels: usize,
    /// Perceiver attention head count (upstream default `heads=8`).
    pub heads: usize,
    /// Perceiver per-head feature dim (upstream default `dim_head=64`).
    pub dim_head: usize,
    /// Perceiver feed-forward expansion (upstream default `ff_mult=4`).
    pub ff_mult: usize,
    /// Perceiver depth (upstream default `depth=2`).
    pub depth: usize,
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            out_dim: 1024,
            latent_dim: 128,
            token_num: 32,
            fsq_levels: vec![4, 4, 4, 4, 4, 4],
            fsq_num_quantizers: 1,
            ecapa_channels: 512,
            heads: 8,
            dim_head: 64,
            ff_mult: 4,
            depth: 2,
        }
    }
}

/// Speaker encoder tower — the global-stream tokenizer / detokenizer
/// pair for `BiCodec`. Mirrors upstream
/// `speaker_encoder.py::SpeakerEncoder`.
///
/// # Pipeline
///
/// * [`SpeakerEncoder::tokenize`]: `(B, input_dim, T_mel) → (B,
///   token_num, fsq_num_quantizers)` — `u32` indices into the
///   `ResidualFsq` codebook.
///   1. `speaker_encoder.forward(mels)` → ECAPA latent
///      `(B, 1536, T_mel)`.
///   2. `perceiver_sampler.forward(latent.transpose(1, 2))` →
///      `(B, token_num, latent_dim)`.
///   3. `quantizer.tokenize(_.transpose(1, 2))` →
///      `(B, token_num, fsq_num_quantizers)`.
/// * [`SpeakerEncoder::detokenize`]: `(B, token_num,
///   fsq_num_quantizers) → (B, out_dim)` — the *d-vector* used by the
///   `BiCodec` postnet.
///   1. `quantizer.detokenize(indices)` → `(B, latent_dim, token_num)`.
///   2. Transpose, reshape to `(B, token_num * latent_dim)`.
///   3. `project.forward(_)` → `(B, out_dim)`.
///
/// The `project` Linear is what bridges from the *quantized perceiver
/// codes* to the d-vector handed to the decoder. Its shape is
/// `Linear(token_num * latent_dim → out_dim)`, i.e.
/// `Linear(32 * 128 = 4096 → 1024)` in the shipping checkpoint —
/// upstream constructs it via `nn.Linear(latent_dim * token_num,
/// out_dim)` at line 69 of `speaker_encoder.py`.
#[derive(Debug, Clone)]
pub(super) struct SpeakerEncoder {
    speaker_encoder: EcapaTdnnGlobC512,
    perceiver_sampler: PerceiverResampler,
    quantizer: ResidualFsq,
    project: Linear,
    cfg: SpeakerEncoderConfig,
}

impl SpeakerEncoder {
    /// Load a `SpeakerEncoder`. `vb` should be rooted at the
    /// `SpeakerEncoder`'s state-dict subtree — children expected:
    /// `speaker_encoder`, `perceiver_sampler`, `quantizer`, `project`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any of the four child
    /// loaders.
    pub(super) fn load(vb: VarBuilder, cfg: SpeakerEncoderConfig) -> Result<Self> {
        let speaker_encoder = EcapaTdnnGlobC512::load(
            vb.pp("speaker_encoder"),
            cfg.input_dim,
            cfg.out_dim,
            cfg.ecapa_channels,
        )?;
        // dim_context = 512 * 3 = 1536 — the ECAPA pre-pool latent
        // channel count, hard-coded in upstream's `PerceiverResampler`
        // construction (line 59 of `speaker_encoder.py`).
        let perceiver_sampler = PerceiverResampler::load(
            vb.pp("perceiver_sampler"),
            cfg.latent_dim,
            cfg.depth,
            512 * 3,
            cfg.token_num,
            cfg.heads,
            cfg.dim_head,
            cfg.ff_mult,
        )?;
        let quantizer = ResidualFsq::load(
            vb.pp("quantizer"),
            cfg.latent_dim,
            &cfg.fsq_levels,
            cfg.fsq_num_quantizers,
        )?;
        let project = linear(
            cfg.latent_dim * cfg.token_num,
            cfg.out_dim,
            vb.pp("project"),
        )?;
        Ok(Self {
            speaker_encoder,
            perceiver_sampler,
            quantizer,
            project,
            cfg,
        })
    }

    /// Number of global tokens per utterance (= perceiver `num_latents`).
    pub(super) fn token_num(&self) -> usize {
        self.cfg.token_num
    }

    /// Number of residual FSQ quantizers (Spark-TTS: 1).
    pub(super) fn num_quantizers(&self) -> usize {
        self.cfg.fsq_num_quantizers
    }

    /// D-vector / x-vector output dim. Spark-TTS: 1024.
    pub(super) fn out_dim(&self) -> usize {
        self.cfg.out_dim
    }

    /// `tokenize`: `(B, input_dim, T_mel) → (B, token_num,
    /// num_quantizers)` `u32` global indices.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any stage of the
    /// `ECAPA → perceiver → ResidualFsq` pipeline.
    pub(super) fn tokenize(&self, mels: &Tensor) -> Result<Tensor> {
        let (_x_vec, latent) = self.speaker_encoder.forward(mels)?;
        // (B, 1536, T_mel) → (B, T_mel, 1536) for the perceiver.
        let latent_btf = latent.transpose(1, 2)?.contiguous()?;
        // (B, token_num, latent_dim) ← perceiver
        let perceived = self.perceiver_sampler.forward(&latent_btf)?;
        // ResidualFsq is channels-first: (B, latent_dim, token_num).
        let perceived_cf = perceived.transpose(1, 2)?.contiguous()?;
        self.quantizer.tokenize(&perceived_cf)
    }

    /// `detokenize`: `(B, token_num, num_quantizers) → (B, out_dim)`
    /// d-vector. Equivalent to upstream's `SpeakerEncoder.detokenize`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the inner `ResidualFsq`
    /// lookup or the final `project` Linear.
    pub(super) fn detokenize(&self, indices: &Tensor) -> Result<Tensor> {
        // (B, latent_dim, token_num) ← ResidualFsq (channels-first).
        let zq = self.quantizer.detokenize(indices)?;
        let (b, latent_dim, token_num) = zq.dims3()?;
        if latent_dim != self.cfg.latent_dim || token_num != self.cfg.token_num {
            candle_core::bail!(
                "SpeakerEncoder::detokenize: ResidualFsq returned shape \
                 (B, {latent_dim}, {token_num}); expected \
                 (B, latent_dim={}, token_num={})",
                self.cfg.latent_dim,
                self.cfg.token_num
            );
        }
        // Match upstream `zq.transpose(1, 2).reshape(B, -1)`:
        //   1. (B, latent_dim, token_num) → (B, token_num, latent_dim)
        //   2. (B, token_num, latent_dim) → (B, token_num * latent_dim)
        // The reshape happens AFTER the transpose, so the flattened
        // axis order is `[token_0_dim_0, token_0_dim_1, …,
        // token_{T-1}_dim_{D-1}]`. Skipping the transpose would
        // produce a different permutation and silently corrupt the
        // `project` input.
        let flat = zq
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, token_num * latent_dim))?;
        // Skip the type annotation on `linear` and let the helper
        // wrap into `Linear`'s implicit cast.
        self.project.forward(&flat)
    }
}

// Workspace-level `linear` re-export shenanigans: `LayerNorm` is
// imported but the parent-mod docs reference it only to remind callers
// the FF carries no explicit pre-norm — silence the unused import.
#[allow(
    dead_code,
    reason = "Re-imported only so the module-level docs can link to it; \
              the FF stack carries no explicit LayerNorm."
)]
type _DocOnlyLayerNorm = LayerNorm;
#[allow(
    dead_code,
    reason = "Same — keep `layer_norm` reachable for parity edits."
)]
const _DOC_ONLY_LAYER_NORM: fn(usize, f64, VarBuilder<'_>) -> Result<LayerNorm> = layer_norm;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "Tests construct deterministic vectors from tiny usize indices \
              (< 2^23) so f32 representation is exact."
)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn vb_from(map: HashMap<String, Tensor>, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_tensors(map, DType::F32, dev)
    }

    /// Insert a Conv1d state-dict entry (weight + optional bias).
    fn insert_conv1d(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_c: usize,
        in_c: usize,
        k: usize,
        dev: &Device,
    ) -> Result<()> {
        // Small deterministic weights: w[o, i, t] = 1.0 / (out_c * in_c * k).
        let total = out_c * in_c * k;
        let scale = 1.0_f32 / (total as f32);
        let v: Vec<f32> = (0..total).map(|i| scale * (i as f32 + 1.0)).collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(v, (out_c, in_c, k), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_c, DType::F32, dev)?,
        );
        Ok(())
    }

    fn insert_linear(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_f: usize,
        in_f: usize,
        dev: &Device,
    ) -> Result<()> {
        let total = out_f * in_f;
        let scale = 1.0_f32 / (in_f as f32);
        let v: Vec<f32> = (0..total)
            .map(|i| scale * ((i as f32 % 7.0) - 3.0))
            .collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(v, (out_f, in_f), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_f, DType::F32, dev)?,
        );
        Ok(())
    }

    fn insert_linear_no_bias(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_f: usize,
        in_f: usize,
        dev: &Device,
    ) -> Result<()> {
        let total = out_f * in_f;
        let scale = 1.0_f32 / (in_f as f32);
        let v: Vec<f32> = (0..total)
            .map(|i| scale * ((i as f32 % 5.0) - 2.0))
            .collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(v, (out_f, in_f), dev)?,
        );
        Ok(())
    }

    /// Insert a `BatchNorm1d` state-dict entry initialized to identity
    /// (`running_mean=0`, `running_var=1`, `weight=1`, `bias=0`) so the
    /// eval-mode BN is a no-op. Lets us assert pure conv/linear
    /// behaviour without BN-folding noise.
    fn insert_bn_identity(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        c: usize,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.running_mean"),
            Tensor::zeros(c, DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.running_var"),
            Tensor::ones(c, DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.weight"),
            Tensor::ones(c, DType::F32, dev)?,
        );
        map.insert(format!("{prefix}.bias"), Tensor::zeros(c, DType::F32, dev)?);
        Ok(())
    }

    #[test]
    fn conv1d_relu_bn_preserves_time_dim_with_padding() -> Result<()> {
        let dev = Device::Cpu;
        let in_c = 4;
        let out_c = 8;
        let k = 5;
        let pad = 2;
        let mut map = HashMap::new();
        insert_conv1d(&mut map, "conv", out_c, in_c, k, &dev)?;
        insert_bn_identity(&mut map, "bn", out_c, &dev)?;
        let vb = vb_from(map, &dev);

        let layer = Conv1dReluBn::load(vb, in_c, out_c, k, 1, pad, 1, 1)?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, in_c, 16), &dev)?;
        let y = layer.forward(&x)?;
        assert_eq!(y.dims(), &[2, out_c, 16]);
        Ok(())
    }

    #[test]
    fn res2_conv1d_relu_bn_concatenates_scale_chunks() -> Result<()> {
        let dev = Device::Cpu;
        let channels = 16;
        let scale = 8;
        let width = channels / scale;
        let k = 3;
        let pad = 2;
        let dil = 2;

        let mut map = HashMap::new();
        for i in 0..(scale - 1) {
            insert_conv1d(&mut map, &format!("convs.{i}"), width, width, k, &dev)?;
            insert_bn_identity(&mut map, &format!("bns.{i}"), width, &dev)?;
        }
        let vb = vb_from(map, &dev);
        let layer = Res2Conv1dReluBn::load(vb, channels, k, 1, pad, dil, scale)?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, channels, 12), &dev)?;
        let y = layer.forward(&x)?;
        assert_eq!(y.dims(), &[2, channels, 12]);
        Ok(())
    }

    #[test]
    fn se_connect_squeeze_excitation_modulates_per_channel() -> Result<()> {
        // Setup: build SeConnect with weights that yield a known gate
        // and verify per-channel scaling.
        let dev = Device::Cpu;
        let channels = 4;
        let bottleneck = 4;

        // linear1: identity-like (so pooled input passes through).
        // linear2: identity-like (so sigmoid(pooled_mean) is the gate).
        let mut map = HashMap::new();
        let mut w1 = vec![0.0_f32; bottleneck * channels];
        for i in 0..channels.min(bottleneck) {
            w1[i * channels + i] = 1.0;
        }
        map.insert(
            "linear1.weight".to_owned(),
            Tensor::from_vec(w1, (bottleneck, channels), &dev)?,
        );
        map.insert(
            "linear1.bias".to_owned(),
            Tensor::zeros(bottleneck, DType::F32, &dev)?,
        );
        let mut w2 = vec![0.0_f32; channels * bottleneck];
        for i in 0..channels.min(bottleneck) {
            w2[i * bottleneck + i] = 1.0;
        }
        map.insert(
            "linear2.weight".to_owned(),
            Tensor::from_vec(w2, (channels, bottleneck), &dev)?,
        );
        map.insert(
            "linear2.bias".to_owned(),
            Tensor::zeros(channels, DType::F32, &dev)?,
        );
        let vb = vb_from(map, &dev);
        let se = SeConnect::load(vb, channels, bottleneck)?;

        // x: channel `c` is constant value `c+1` across time. Pooled
        // mean per channel = c+1 → through ReLU(identity) and
        // sigmoid(identity) → sigmoid(c+1). Output[c,t] = x[c,t] *
        // sigmoid(c+1) → 0 along channel 0... wait, channel 0 has
        // value 1, so output = 1 * sigmoid(1). Just check that the
        // ratio output[c,t] / x[c,t] is constant per channel.
        let b = 1;
        let t = 3;
        let mut xv = vec![0.0_f32; b * channels * t];
        for c in 0..channels {
            for ti in 0..t {
                xv[c * t + ti] = (c as f32) + 1.0;
            }
        }
        let x = Tensor::from_vec(xv, (b, channels, t), &dev)?;
        let y = se.forward(&x)?;
        let y_v = y.flatten_all()?.to_vec1::<f32>()?;
        for c in 0..channels {
            let x_c = (c as f32) + 1.0;
            let expected_gate = 1.0_f32 / (1.0 + (-x_c).exp());
            let expected_y = x_c * expected_gate;
            for ti in 0..t {
                let got = y_v[c * t + ti];
                assert!(
                    (got - expected_y).abs() < 1e-5,
                    "SE gate mismatch at c={c}, t={ti}: got {got}, expected {expected_y}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn se_res2_block_residual_shape_preserved() -> Result<()> {
        let dev = Device::Cpu;
        let channels = 16;
        let scale = 8;
        let width = channels / scale;
        let bottleneck = 8;

        let mut map = HashMap::new();
        // se_res2block.0: Conv1dReluBn(channels, channels, 1, 1, 0)
        insert_conv1d(&mut map, "se_res2block.0.conv", channels, channels, 1, &dev)?;
        insert_bn_identity(&mut map, "se_res2block.0.bn", channels, &dev)?;
        // se_res2block.1: Res2Conv1dReluBn(channels, k=3, s=1, p=2, d=2, scale=8)
        for i in 0..(scale - 1) {
            insert_conv1d(
                &mut map,
                &format!("se_res2block.1.convs.{i}"),
                width,
                width,
                3,
                &dev,
            )?;
            insert_bn_identity(&mut map, &format!("se_res2block.1.bns.{i}"), width, &dev)?;
        }
        // se_res2block.2: Conv1dReluBn(channels, channels, 1, 1, 0)
        insert_conv1d(&mut map, "se_res2block.2.conv", channels, channels, 1, &dev)?;
        insert_bn_identity(&mut map, "se_res2block.2.bn", channels, &dev)?;
        // se_res2block.3: SeConnect(channels, 8)
        insert_linear(
            &mut map,
            "se_res2block.3.linear1",
            bottleneck,
            channels,
            &dev,
        )?;
        insert_linear(
            &mut map,
            "se_res2block.3.linear2",
            channels,
            bottleneck,
            &dev,
        )?;
        let vb = vb_from(map, &dev);

        let block = SeRes2Block::load(vb, channels, 3, 1, 2, 2, scale, bottleneck)?;
        let x = Tensor::randn(0.0_f32, 0.1, (2, channels, 10), &dev)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims(), &[2, channels, 10]);
        Ok(())
    }

    #[test]
    fn astp_output_dim_is_2x_input() -> Result<()> {
        let dev = Device::Cpu;
        let in_dim = 12;
        let bottleneck = 4;
        let mut map = HashMap::new();
        insert_conv1d(&mut map, "linear1", bottleneck, in_dim * 3, 1, &dev)?;
        insert_conv1d(&mut map, "linear2", in_dim, bottleneck, 1, &dev)?;
        let vb = vb_from(map, &dev);
        let pool = Astp::load(vb, in_dim, bottleneck, true)?;
        assert_eq!(pool.out_dim(), 2 * in_dim);

        let x = Tensor::randn(0.0_f32, 1.0, (2, in_dim, 7), &dev)?;
        let y = pool.forward(&x)?;
        assert_eq!(y.dims(), &[2, 2 * in_dim]);
        Ok(())
    }

    /// Build a full ECAPA-TDNN-GLOB-c512 state-dict with the
    /// scaled-down `channels=16` configuration suitable for fast
    /// CPU tests (the conv output dim is still hard-coded to
    /// `512 * 3 = 1536` upstream, so we mirror that exactly).
    fn build_ecapa_state_dict(
        feat_dim: usize,
        embed_dim: usize,
        channels: usize,
        dev: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut map = HashMap::new();
        // layer1: Conv1dReluBn(feat_dim, channels, k=5, pad=2)
        insert_conv1d(&mut map, "layer1.conv", channels, feat_dim, 5, dev)?;
        insert_bn_identity(&mut map, "layer1.bn", channels, dev)?;

        // layers 2/3/4: SE_Res2Block(channels, k=3, s=1, pad, dil, scale=8)
        let scale = 8;
        let width = channels / scale;
        let bottleneck = 128;
        for layer_i in 2..=4 {
            // 0: head Conv1dReluBn(channels, channels, 1)
            insert_conv1d(
                &mut map,
                &format!("layer{layer_i}.se_res2block.0.conv"),
                channels,
                channels,
                1,
                dev,
            )?;
            insert_bn_identity(
                &mut map,
                &format!("layer{layer_i}.se_res2block.0.bn"),
                channels,
                dev,
            )?;
            // 1: Res2Conv1dReluBn(channels, k=3, ..., scale=8)
            for i in 0..(scale - 1) {
                insert_conv1d(
                    &mut map,
                    &format!("layer{layer_i}.se_res2block.1.convs.{i}"),
                    width,
                    width,
                    3,
                    dev,
                )?;
                insert_bn_identity(
                    &mut map,
                    &format!("layer{layer_i}.se_res2block.1.bns.{i}"),
                    width,
                    dev,
                )?;
            }
            // 2: tail Conv1dReluBn(channels, channels, 1)
            insert_conv1d(
                &mut map,
                &format!("layer{layer_i}.se_res2block.2.conv"),
                channels,
                channels,
                1,
                dev,
            )?;
            insert_bn_identity(
                &mut map,
                &format!("layer{layer_i}.se_res2block.2.bn"),
                channels,
                dev,
            )?;
            // 3: SeConnect(channels, 128)
            insert_linear(
                &mut map,
                &format!("layer{layer_i}.se_res2block.3.linear1"),
                bottleneck,
                channels,
                dev,
            )?;
            insert_linear(
                &mut map,
                &format!("layer{layer_i}.se_res2block.3.linear2"),
                channels,
                bottleneck,
                dev,
            )?;
        }

        // 1×1 conv: cat_channels=3*channels → out_channels=3*512.
        let cat_channels = channels * 3;
        let out_channels = 512 * 3;
        insert_conv1d(&mut map, "conv", out_channels, cat_channels, 1, dev)?;

        // pool: Astp(out_channels, bottleneck=128, global_context_att=True)
        insert_conv1d(&mut map, "pool.linear1", 128, out_channels * 3, 1, dev)?;
        insert_conv1d(&mut map, "pool.linear2", out_channels, 128, 1, dev)?;

        // post-pool BN over 2 * out_channels
        insert_bn_identity(&mut map, "bn", 2 * out_channels, dev)?;

        // linear: Linear(2 * out_channels → embed_dim)
        insert_linear(&mut map, "linear", embed_dim, 2 * out_channels, dev)?;

        Ok(map)
    }

    #[test]
    fn ecapa_forward_returns_xvector_and_latent() -> Result<()> {
        let dev = Device::Cpu;
        let feat_dim = 16;
        let embed_dim = 24;
        let channels = 16;
        let map = build_ecapa_state_dict(feat_dim, embed_dim, channels, &dev)?;
        let vb = vb_from(map, &dev);
        let ecapa = EcapaTdnnGlobC512::load(vb, feat_dim, embed_dim, channels)?;

        let t_mel = 8;
        let x = Tensor::randn(0.0_f32, 1.0, (2, feat_dim, t_mel), &dev)?;
        let (x_vec, latent) = ecapa.forward(&x)?;
        // x_vector: (B, embed_dim)
        assert_eq!(x_vec.dims(), &[2, embed_dim]);
        // latent: (B, 3 * 512, T_mel) — note `3 * 512` even though
        // channels=16, matching the upstream hard-coding.
        assert_eq!(latent.dims(), &[2, 512 * 3, t_mel]);
        Ok(())
    }

    fn build_perceiver_state_dict(
        dim: usize,
        dim_context: usize,
        num_latents: usize,
        depth: usize,
        heads: usize,
        dim_head: usize,
        ff_mult: usize,
        dev: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut map = HashMap::new();
        if dim_context != dim {
            insert_linear(&mut map, "proj_context", dim, dim_context, dev)?;
        }
        // latents: (num_latents, dim) — random small values.
        let total = num_latents * dim;
        let v: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.013).sin()).collect();
        map.insert(
            "latents".to_owned(),
            Tensor::from_vec(v, (num_latents, dim), dev)?,
        );
        let dim_inner_head = dim_head * heads;
        let ff_inner = (dim * ff_mult * 2) / 3;
        for i in 0..depth {
            // attention: to_q (no bias), to_kv, to_out
            insert_linear_no_bias(
                &mut map,
                &format!("layers.{i}.0.to_q"),
                dim_inner_head,
                dim,
                dev,
            )?;
            insert_linear_no_bias(
                &mut map,
                &format!("layers.{i}.0.to_kv"),
                dim_inner_head * 2,
                dim,
                dev,
            )?;
            insert_linear_no_bias(
                &mut map,
                &format!("layers.{i}.0.to_out"),
                dim,
                dim_inner_head,
                dev,
            )?;
            // feedforward: layers.{i}.1.0 (Linear dim → ff_inner*2),
            // layers.{i}.1.2 (Linear ff_inner → dim). Index 1 is GEGLU
            // (no params).
            insert_linear(&mut map, &format!("layers.{i}.1.0"), ff_inner * 2, dim, dev)?;
            insert_linear(&mut map, &format!("layers.{i}.1.2"), dim, ff_inner, dev)?;
        }
        // norm.gamma
        map.insert("norm.gamma".to_owned(), Tensor::ones(dim, DType::F32, dev)?);
        Ok(map)
    }

    #[test]
    fn perceiver_resampler_output_shape_matches_num_latents() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 16;
        let dim_context = 24;
        let num_latents = 8;
        let depth = 2;
        let heads = 2;
        let dim_head = 8;
        let ff_mult = 4;
        let map = build_perceiver_state_dict(
            dim,
            dim_context,
            num_latents,
            depth,
            heads,
            dim_head,
            ff_mult,
            &dev,
        )?;
        let vb = vb_from(map, &dev);
        let pr = PerceiverResampler::load(
            vb,
            dim,
            depth,
            dim_context,
            num_latents,
            heads,
            dim_head,
            ff_mult,
        )?;
        let t_ctx = 12;
        let x = Tensor::randn(0.0_f32, 0.5, (2, t_ctx, dim_context), &dev)?;
        let y = pr.forward(&x)?;
        assert_eq!(y.dims(), &[2, num_latents, dim]);
        Ok(())
    }

    #[test]
    fn perceiver_resampler_different_contexts_produce_different_outputs() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 16;
        let dim_context = 24;
        let num_latents = 4;
        let depth = 2;
        let heads = 2;
        let dim_head = 8;
        let ff_mult = 4;
        let map = build_perceiver_state_dict(
            dim,
            dim_context,
            num_latents,
            depth,
            heads,
            dim_head,
            ff_mult,
            &dev,
        )?;
        let vb = vb_from(map, &dev);
        let pr = PerceiverResampler::load(
            vb,
            dim,
            depth,
            dim_context,
            num_latents,
            heads,
            dim_head,
            ff_mult,
        )?;
        let t_ctx = 10;
        let x_a = Tensor::randn(0.0_f32, 0.7, (1, t_ctx, dim_context), &dev)?;
        let x_b = Tensor::randn(2.0_f32, 0.7, (1, t_ctx, dim_context), &dev)?;
        let y_a = pr.forward(&x_a)?;
        let y_b = pr.forward(&x_b)?;
        let diff = (&y_a - &y_b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-4,
            "Perceiver output identical for differing contexts — conditioning broken (diff={diff})"
        );
        Ok(())
    }

    fn build_speaker_encoder_state_dict(
        cfg: &SpeakerEncoderConfig,
        dev: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut map = HashMap::new();
        // speaker_encoder/* — ECAPA
        let ecapa_map =
            build_ecapa_state_dict(cfg.input_dim, cfg.out_dim, cfg.ecapa_channels, dev)?;
        for (k, v) in ecapa_map {
            map.insert(format!("speaker_encoder.{k}"), v);
        }
        // perceiver_sampler/* — note dim_context is hard-coded 512*3
        let perc_map = build_perceiver_state_dict(
            cfg.latent_dim,
            512 * 3,
            cfg.token_num,
            cfg.depth,
            cfg.heads,
            cfg.dim_head,
            cfg.ff_mult,
            dev,
        )?;
        for (k, v) in perc_map {
            map.insert(format!("perceiver_sampler.{k}"), v);
        }
        // quantizer/* — ResidualFsq with project_in / project_out.
        let codebook_dim = cfg.fsq_levels.len();
        insert_linear(
            &mut map,
            "quantizer.project_in",
            codebook_dim,
            cfg.latent_dim,
            dev,
        )?;
        insert_linear(
            &mut map,
            "quantizer.project_out",
            cfg.latent_dim,
            codebook_dim,
            dev,
        )?;
        // ResidualFsq inner layers carry no projections (dim ==
        // codebook_dim inside). For Spark-TTS num_quantizers=1, that's
        // all we need.
        // project: Linear(latent_dim * token_num → out_dim)
        insert_linear(
            &mut map,
            "project",
            cfg.out_dim,
            cfg.latent_dim * cfg.token_num,
            dev,
        )?;
        Ok(map)
    }

    /// Tiny config that keeps the test fast on CPU.
    fn tiny_cfg() -> SpeakerEncoderConfig {
        SpeakerEncoderConfig {
            input_dim: 16,
            out_dim: 24,
            latent_dim: 16,
            token_num: 8,
            fsq_levels: vec![4, 4, 4, 4, 4, 4],
            fsq_num_quantizers: 1,
            ecapa_channels: 16,
            heads: 2,
            dim_head: 8,
            ff_mult: 4,
            depth: 2,
        }
    }

    #[test]
    fn speaker_encoder_tokenize_returns_correct_index_shape() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let map = build_speaker_encoder_state_dict(&cfg, &dev)?;
        let vb = vb_from(map, &dev);
        let se = SpeakerEncoder::load(vb, cfg.clone())?;

        let t_mel = 10;
        let mels = Tensor::randn(0.0_f32, 0.5, (2, cfg.input_dim, t_mel), &dev)?;
        let indices = se.tokenize(&mels)?;
        assert_eq!(
            indices.dims(),
            &[2, cfg.token_num, cfg.fsq_num_quantizers],
            "tokenize index shape mismatch"
        );
        assert_eq!(indices.dtype(), DType::U32);
        Ok(())
    }

    #[test]
    fn speaker_encoder_detokenize_returns_dvector_shape() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let map = build_speaker_encoder_state_dict(&cfg, &dev)?;
        let vb = vb_from(map, &dev);
        let se = SpeakerEncoder::load(vb, cfg.clone())?;

        // Valid indices in [0, codebook_size).
        let cb = se.quantizer_codebook_size_for_test();
        let n = 2 * cfg.token_num * cfg.fsq_num_quantizers;
        let n_u32 = u32::try_from(n).expect("test index count fits in u32");
        let raw: Vec<u32> = (0..n_u32).map(|i| i % cb).collect();
        let indices = Tensor::from_vec(raw, (2, cfg.token_num, cfg.fsq_num_quantizers), &dev)?;
        let dvec = se.detokenize(&indices)?;
        assert_eq!(dvec.dims(), &[2, cfg.out_dim]);
        Ok(())
    }

    impl SpeakerEncoder {
        /// Test-only helper: bubble up the inner `ResidualFsq` codebook size
        /// so the detokenize test can generate valid indices without
        /// re-deriving `prod(levels)`.
        pub(super) fn quantizer_codebook_size_for_test(&self) -> u32 {
            self.quantizer.codebook_size()
        }
    }
}
