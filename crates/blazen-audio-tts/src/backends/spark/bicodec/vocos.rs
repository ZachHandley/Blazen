//! Vocos-style `ConvNeXt` backbone used as the trunk of `BiCodec`'s
//! `Encoder`, prenet, and postnet (sub-wave **S.2.1.b** of the [`super`]
//! `BiCodec` port).
//!
//! Mirrors `sparktts/modules/blocks/vocos.py::{ConvNeXtBlock,
//! VocosBackbone}`. The `VocosResNetBackbone` variant is intentionally
//! skipped — Spark-TTS doesn't ship a checkpoint that uses it.
//!
//! # Channels-first I/O convention
//!
//! Upstream `VocosBackbone.forward` takes `(B, C, T)` channels-first
//! input, runs the entry `Conv1d`, then permutes to channels-last for
//! the `LayerNorm`s / pointwise `Linear`s inside each `ConvNeXtBlock`,
//! and permutes back inside each block so the I/O between blocks is
//! channels-first. The *final* `LayerNorm` of the backbone is applied
//! to the channels-last permuted tensor and the return value is left in
//! `(B, T, dim)` layout in upstream Python.
//!
//! We deliberately diverge from that one detail to make the surface
//! easier for downstream `BiCodec` waves to consume: our
//! [`VocosBackbone::forward`] returns channels-first `(B, dim, T)`
//! (final permute is *not* applied). Every consumer in this codec
//! immediately re-permutes back to channels-first anyway (sampler,
//! quantizer pre-conv, etc.), so doing the transpose here avoids a
//! redundant pair of permutes at every call site. The per-block I/O
//! semantics are unchanged and bit-identical to upstream.
//!
//! # `AdaLN` substitution
//!
//! Upstream replaces *only* the entry `LayerNorm` (and the per-block
//! `LayerNorm`) with [`AdaLayerNorm`] when `condition_dim is not None`.
//! The backbone's `final_layer_norm` always stays a plain affine
//! `LayerNorm` regardless of conditioning. We mirror that exactly.

// `VarBuilder` is the canonical "consume by value" handle in candle.
// See primitives.rs for the same lint waiver and rationale.
#![allow(clippy::needless_pass_by_value)]

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};

use super::primitives::AdaLayerNorm;

/// `AdaLN` eps — matches upstream `nn.LayerNorm(dim, eps=1e-6)` /
/// `AdaLayerNorm(condition_dim, dim, eps=1e-6)`.
const NORM_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// NormVariant
// ---------------------------------------------------------------------------

/// `LayerNorm` flavour used inside [`ConvNeXtBlock`] and as the entry
/// norm of [`VocosBackbone`]. Plain when `condition_dim is None`,
/// adaptive otherwise. The backbone's *final* norm is always plain
/// regardless of conditioning, so we don't use this enum for that one.
pub(super) enum NormVariant {
    /// Affine `LayerNorm` — loaded via [`candle_nn::layer_norm`].
    Plain(LayerNorm),
    /// Adaptive `LayerNorm` — loaded via
    /// [`AdaLayerNorm::load`][super::primitives::AdaLayerNorm::load].
    Ada(AdaLayerNorm),
}

impl NormVariant {
    /// Load either a `LayerNorm` (when `condition_dim` is `None`) or an
    /// [`AdaLayerNorm`] (when `Some`) under the given `VarBuilder`. The
    /// `VarBuilder` should already be `pp("norm")`-scoped by the caller
    /// — we treat the two flavours as a drop-in replacement for the
    /// same Python attribute name.
    fn load_block_norm(vb: VarBuilder, dim: usize, condition_dim: Option<usize>) -> Result<Self> {
        match condition_dim {
            None => Ok(Self::Plain(layer_norm(dim, NORM_EPS, vb)?)),
            Some(cd) => Ok(Self::Ada(AdaLayerNorm::load(vb, cd, dim, NORM_EPS)?)),
        }
    }

    /// Forward. `x` is `(B, T, C)` channels-last. `condition` must be
    /// `Some((B, condition_dim))` iff this variant is [`Self::Ada`] —
    /// passing a condition to a `Plain` variant is silently ignored
    /// (matches upstream which just calls `self.norm(x)` on the
    /// unconditional branch).
    fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        match self {
            Self::Plain(ln) => ln.forward(x),
            Self::Ada(ada) => {
                let cond = condition.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "NormVariant::Ada requires a condition tensor, got None".into(),
                    )
                })?;
                ada.forward(x, cond)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConvNeXtBlock
// ---------------------------------------------------------------------------

/// One `ConvNeXt` block adapted from
/// <https://github.com/facebookresearch/ConvNeXt> for 1-D audio:
///
/// ```text
///   residual = x                                  // (B, C, T)
///   x = dwconv(x)                                 // depthwise k=7, p=3
///   x = x.transpose(1, 2)                         // (B, T, C)
///   x = norm(x[, condition])                      // LN or AdaLN
///   x = pwconv1(x)                                // (B, T, intermediate)
///   x = gelu(x)
///   x = pwconv2(x)                                // (B, T, C)
///   if gamma is not None: x = gamma * x           // LayerScale
///   x = x.transpose(1, 2)                         // (B, C, T)
///   return residual + x
/// ```
///
/// The depthwise conv uses `groups = dim` so each channel is convolved
/// independently. The two pointwise convs are implemented as `Linear`
/// over the channel axis (cheaper to load and equivalent for 1×1 convs).
pub(super) struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: NormVariant,
    pwconv1: Linear,
    pwconv2: Linear,
    /// Optional `LayerScale` parameter of shape `(dim,)`. When `Some`,
    /// elementwise-multiplies the post-MLP activation in channels-last
    /// layout before the residual add. `None` is the "no scaling" case
    /// (`layer_scale_init_value` not supplied by the caller).
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    /// Load one block. State-dict keys under `vb`:
    ///
    /// * `dwconv.{weight,bias}` — depthwise `Conv1d(dim, dim, k=7)`.
    /// * `norm.{weight,bias}` (plain) **or** `norm.{scale,shift}.{weight,bias}`
    ///   (Ada) — see [`NormVariant`].
    /// * `pwconv1.{weight,bias}` — `Linear(dim, intermediate_dim)`.
    /// * `pwconv2.{weight,bias}` — `Linear(intermediate_dim, dim)`.
    /// * `gamma` — present iff `layer_scale_init_value is Some`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-load.
    pub(super) fn load(
        vb: VarBuilder,
        dim: usize,
        intermediate_dim: usize,
        layer_scale_init_value: Option<f64>,
        condition_dim: Option<usize>,
    ) -> Result<Self> {
        let dw_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: dim,
            ..Conv1dConfig::default()
        };
        // candle_nn::conv1d expects the standard
        // `weight: (out, in/groups, k)`, `bias: (out,)` PyTorch layout —
        // which is exactly what `nn.Conv1d` stores after instantiation.
        let dwconv = candle_nn::conv1d(dim, dim, 7, dw_cfg, vb.pp("dwconv"))?;
        let norm = NormVariant::load_block_norm(vb.pp("norm"), dim, condition_dim)?;
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;
        let gamma = if layer_scale_init_value.is_some() {
            Some(vb.get(dim, "gamma")?)
        } else {
            None
        };
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Forward. `x` is channels-first `(B, dim, T)`. `condition` must
    /// be `Some((B, condition_dim))` iff this block was loaded with
    /// `condition_dim is Some`. Output is `(B, dim, T)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the depthwise conv, the
    /// norm, the pointwise convs, or the residual add.
    pub(super) fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let h = self.dwconv.forward(x)?;
        // (B, C, T) → (B, T, C) for LayerNorm + pointwise Linears.
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = self.norm.forward(&h, condition)?;
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.pwconv2.forward(&h)?;
        let h = if let Some(gamma) = &self.gamma {
            // gamma: (dim,) broadcasts over (B, T, dim).
            h.broadcast_mul(gamma)?
        } else {
            h
        };
        // (B, T, C) → (B, C, T) before adding to residual.
        let h = h.transpose(1, 2)?.contiguous()?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// VocosBackbone
// ---------------------------------------------------------------------------

/// Vocos backbone built from a stack of [`ConvNeXtBlock`]s:
///
/// ```text
///   x = embed(x)            // Conv1d(input_channels -> dim, k=7, p=3)
///   x = norm_entry(x_T, condition?)  // LN or AdaLN, transposed to (B, T, C)
///   x_back_to_BCT
///   for block in blocks: x = block(x, condition?)
///   x = norm_final(x_T)     // plain LN — always, even when conditioned
///   return x_to_BCT         // (we transpose back, upstream does not)
/// ```
///
/// See the module docstring for why we return channels-first `(B, dim,
/// T)` rather than upstream's channels-last `(B, T, dim)`.
pub(super) struct VocosBackbone {
    /// Entry Conv1d, `input_channels -> dim`, kernel 7, padding 3.
    embed: Conv1d,
    /// Entry norm — affine `LayerNorm` when unconditional, `AdaLN` when
    /// `condition_dim is Some`. Mirrors upstream's `self.norm`.
    norm_entry: NormVariant,
    /// The stack of `ConvNeXt` blocks. Number of blocks = `num_layers`.
    blocks: Vec<ConvNeXtBlock>,
    /// Final norm — *always* plain affine `LayerNorm`. Upstream's
    /// `self.final_layer_norm` is hardcoded as `nn.LayerNorm`,
    /// independent of conditioning.
    norm_final: LayerNorm,
}

impl VocosBackbone {
    /// Load the backbone. State-dict keys under `vb`:
    ///
    /// * `embed.{weight,bias}` — entry `Conv1d`.
    /// * `norm.{weight,bias}` (plain) or `norm.{scale,shift}.{weight,bias}`
    ///   (Ada) — entry norm.
    /// * `convnext.{i}.…` for `i in 0..num_layers` — see
    ///   [`ConvNeXtBlock::load`].
    /// * `final_layer_norm.{weight,bias}` — always plain `LayerNorm`.
    ///
    /// `layer_scale_init_value` defaults upstream to `1.0 / num_layers`
    /// when not given; the caller passes the resolved value here (or
    /// `None` for no `LayerScale`).
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any sub-load.
    pub(super) fn load(
        vb: VarBuilder,
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        layer_scale_init_value: Option<f64>,
        condition_dim: Option<usize>,
    ) -> Result<Self> {
        let embed_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Conv1dConfig::default()
        };
        let embed = candle_nn::conv1d(input_channels, dim, 7, embed_cfg, vb.pp("embed"))?;
        let norm_entry = NormVariant::load_block_norm(vb.pp("norm"), dim, condition_dim)?;

        let convnext_vb = vb.pp("convnext");
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(ConvNeXtBlock::load(
                convnext_vb.pp(i.to_string()),
                dim,
                intermediate_dim,
                layer_scale_init_value,
                condition_dim,
            )?);
        }

        let norm_final = layer_norm(dim, NORM_EPS, vb.pp("final_layer_norm"))?;

        Ok(Self {
            embed,
            norm_entry,
            blocks,
            norm_final,
        })
    }

    /// Forward. `x` is `(B, input_channels, T)`. `condition` must be
    /// `Some((B, condition_dim))` iff this backbone was loaded with
    /// `condition_dim is Some`. Output is channels-first `(B, dim, T)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the entry conv, any norm,
    /// or any `ConvNeXtBlock` forward.
    pub(super) fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let h = self.embed.forward(x)?;
        // Entry norm runs in channels-last layout.
        let h_tc = h.transpose(1, 2)?.contiguous()?;
        let h_tc = self.norm_entry.forward(&h_tc, condition)?;
        // Back to channels-first for the block stack.
        let mut h = h_tc.transpose(1, 2)?.contiguous()?;
        for block in &self.blocks {
            h = block.forward(&h, condition)?;
        }
        // Final plain LayerNorm — channels-last.
        let h_tc = h.transpose(1, 2)?.contiguous()?;
        let h_tc = self.norm_final.forward(&h_tc)?;
        // Re-transpose to channels-first for downstream consumers.
        h_tc.transpose(1, 2)?.contiguous()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "tests build small deterministic vectors via `usize as f32` \
              indices; ranges are tiny (< 2^23) so precision is exact."
)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    /// Build a `VarBuilder` from a tensor map.
    fn vb_from(map: HashMap<String, Tensor>, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_tensors(map, DType::F32, dev)
    }

    /// Populate `map` with the keys for one plain `LayerNorm(dim)`
    /// under `prefix` (e.g. `"norm"`, `"final_layer_norm"`).
    fn put_plain_ln(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.weight"),
            Tensor::ones(dim, DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// Populate `map` with the keys for one `AdaLayerNorm(condition_dim, dim)`
    /// under `prefix`. Initialised so that condition=zeros yields
    /// `scale=ones, shift=zeros` (identity affine after LN).
    fn put_ada_ln(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        condition_dim: usize,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        map.insert(
            format!("{prefix}.scale.weight"),
            Tensor::zeros((dim, condition_dim), DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.scale.bias"),
            Tensor::ones(dim, DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.shift.weight"),
            Tensor::zeros((dim, condition_dim), DType::F32, dev)?,
        );
        map.insert(
            format!("{prefix}.shift.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// Populate `map` with depthwise conv weights for `(dim, dim, k=7)`
    /// with `groups=dim`. `bias` is filled with zeros.
    fn put_dwconv(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        dev: &Device,
    ) -> Result<()> {
        // groups=dim → weight shape (dim, dim/groups=1, k=7).
        let w_vec: Vec<f32> = (0..(dim * 7)).map(|i| (i as f32) * 0.005).collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(w_vec, (dim, 1, 7), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// Populate `map` with a `Linear(in_dim, out_dim)` under `prefix`.
    /// Weights are arbitrary deterministic floats; bias is zeros.
    fn put_linear(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        dev: &Device,
    ) -> Result<()> {
        let w_vec: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| ((i as f32) * 0.011).sin() * 0.1)
            .collect();
        map.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(w_vec, (out_dim, in_dim), dev)?,
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_dim, DType::F32, dev)?,
        );
        Ok(())
    }

    /// Populate `map` with all keys for one full `ConvNeXtBlock`. If
    /// `gamma` is `Some`, also writes a `gamma` tensor of that value.
    fn put_convnext_block(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        dim: usize,
        intermediate_dim: usize,
        condition_dim: Option<usize>,
        gamma: Option<f32>,
        dev: &Device,
    ) -> Result<()> {
        put_dwconv(map, &format!("{prefix}.dwconv"), dim, dev)?;
        match condition_dim {
            None => put_plain_ln(map, &format!("{prefix}.norm"), dim, dev)?,
            Some(cd) => put_ada_ln(map, &format!("{prefix}.norm"), cd, dim, dev)?,
        }
        put_linear(
            map,
            &format!("{prefix}.pwconv1"),
            dim,
            intermediate_dim,
            dev,
        )?;
        put_linear(
            map,
            &format!("{prefix}.pwconv2"),
            intermediate_dim,
            dim,
            dev,
        )?;
        if let Some(g) = gamma {
            let gamma_vec = vec![g; dim];
            map.insert(
                format!("{prefix}.gamma"),
                Tensor::from_vec(gamma_vec, dim, dev)?,
            );
        }
        Ok(())
    }

    #[test]
    fn convnext_block_zero_gamma_preserves_residual() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8;
        let intermediate_dim = 16;

        let mut map = HashMap::new();
        put_convnext_block(
            &mut map,
            "",
            dim,
            intermediate_dim,
            /* condition_dim */ None,
            /* gamma */ Some(0.0),
            &dev,
        )?;
        // `put_convnext_block("", …)` writes keys like ".dwconv.weight"
        // because of the leading separator. Strip those down to the
        // unprefixed form expected by `VarBuilder::root()`.
        let map: HashMap<String, Tensor> = map
            .into_iter()
            .map(|(k, v)| (k.trim_start_matches('.').to_owned(), v))
            .collect();
        let vb = vb_from(map, &dev);
        let block = ConvNeXtBlock::load(
            vb,
            dim,
            intermediate_dim,
            Some(1e-6),
            /* condition_dim */ None,
        )?;

        // Arbitrary deterministic input.
        let n = 2 * dim * 12;
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 0.5).collect();
        let x = Tensor::from_vec(xs, (2, dim, 12), &dev)?;
        let y = block.forward(&x, None)?;
        assert_eq!(y.dims(), &[2, dim, 12]);

        // With gamma=0 the MLP path is zeroed out and the block reduces
        // to the identity residual.
        let diff = (&y - &x)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "ConvNeXtBlock with gamma=0 should preserve input, got max diff {diff}"
        );
        Ok(())
    }

    #[test]
    fn convnext_block_forward_shape_preserved() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 384;
        let intermediate_dim = 1152;

        let mut map = HashMap::new();
        put_convnext_block(
            &mut map,
            "",
            dim,
            intermediate_dim,
            None,
            Some(1.0 / 12.0),
            &dev,
        )?;
        let map: HashMap<String, Tensor> = map
            .into_iter()
            .map(|(k, v)| (k.trim_start_matches('.').to_owned(), v))
            .collect();
        let vb = vb_from(map, &dev);
        let block = ConvNeXtBlock::load(vb, dim, intermediate_dim, Some(1.0 / 12.0), None)?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, dim, 48), &dev)?;
        let y = block.forward(&x, None)?;
        assert_eq!(y.dims(), &[2, dim, 48]);
        Ok(())
    }

    #[test]
    fn vocos_backbone_forward_shape_to_dim() -> Result<()> {
        let dev = Device::Cpu;
        let input_channels = 1024;
        let dim = 384;
        let intermediate_dim = 1152;
        let num_layers = 2_usize;
        let lsi_f32 = 1.0_f32 / (num_layers as f32);
        let lsi = Some(f64::from(lsi_f32));

        let mut map = HashMap::new();
        // embed: Conv1d(input_channels, dim, k=7), groups=1 →
        // weight shape (dim, input_channels, 7).
        let embed_w_vec: Vec<f32> = (0..(dim * input_channels * 7))
            .map(|i| ((i as f32) * 0.0007).cos() * 0.05)
            .collect();
        map.insert(
            "embed.weight".to_owned(),
            Tensor::from_vec(embed_w_vec, (dim, input_channels, 7), &dev)?,
        );
        map.insert(
            "embed.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        put_plain_ln(&mut map, "norm", dim, &dev)?;
        for i in 0..num_layers {
            put_convnext_block(
                &mut map,
                &format!("convnext.{i}"),
                dim,
                intermediate_dim,
                None,
                Some(lsi_f32),
                &dev,
            )?;
        }
        put_plain_ln(&mut map, "final_layer_norm", dim, &dev)?;

        let vb = vb_from(map, &dev);
        let backbone = VocosBackbone::load(
            vb,
            input_channels,
            dim,
            intermediate_dim,
            num_layers,
            lsi,
            None,
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 48), &dev)?;
        let y = backbone.forward(&x, None)?;
        assert_eq!(y.dims(), &[2, dim, 48]);
        Ok(())
    }

    #[test]
    fn vocos_backbone_conditioned_uses_ada_layernorm() -> Result<()> {
        let dev = Device::Cpu;
        let input_channels = 32;
        let dim = 16;
        let intermediate_dim = 24;
        let num_layers = 2_usize;
        let condition_dim = 1024;
        let lsi_f32 = 1.0_f32 / (num_layers as f32);
        let lsi = Some(f64::from(lsi_f32));

        let mut map = HashMap::new();
        let embed_w_vec: Vec<f32> = (0..(dim * input_channels * 7))
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        map.insert(
            "embed.weight".to_owned(),
            Tensor::from_vec(embed_w_vec, (dim, input_channels, 7), &dev)?,
        );
        map.insert(
            "embed.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        // Entry norm is AdaLN. Use non-identity weights so that two
        // different conditions yield different outputs.
        let scale_w: Vec<f32> = (0..(dim * condition_dim))
            .map(|i| ((i as f32) * 0.001).sin() * 0.02)
            .collect();
        map.insert(
            "norm.scale.weight".to_owned(),
            Tensor::from_vec(scale_w, (dim, condition_dim), &dev)?,
        );
        map.insert(
            "norm.scale.bias".to_owned(),
            Tensor::ones(dim, DType::F32, &dev)?,
        );
        let shift_w: Vec<f32> = (0..(dim * condition_dim))
            .map(|i| ((i as f32) * 0.001).cos() * 0.02)
            .collect();
        map.insert(
            "norm.shift.weight".to_owned(),
            Tensor::from_vec(shift_w, (dim, condition_dim), &dev)?,
        );
        map.insert(
            "norm.shift.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        // Per-block AdaLN — same shape pattern. Use identity-ish init
        // (scale=ones at zero-cond) for blocks so the test isolates the
        // entry norm's contribution.
        for i in 0..num_layers {
            put_dwconv(&mut map, &format!("convnext.{i}.dwconv"), dim, &dev)?;
            put_ada_ln(
                &mut map,
                &format!("convnext.{i}.norm"),
                condition_dim,
                dim,
                &dev,
            )?;
            put_linear(
                &mut map,
                &format!("convnext.{i}.pwconv1"),
                dim,
                intermediate_dim,
                &dev,
            )?;
            put_linear(
                &mut map,
                &format!("convnext.{i}.pwconv2"),
                intermediate_dim,
                dim,
                &dev,
            )?;
            // Small non-zero gamma so the block actually contributes,
            // but stays close to the identity residual.
            let gamma_vec = vec![lsi_f32; dim];
            map.insert(
                format!("convnext.{i}.gamma"),
                Tensor::from_vec(gamma_vec, dim, &dev)?,
            );
        }
        // Final norm is always plain LN.
        put_plain_ln(&mut map, "final_layer_norm", dim, &dev)?;

        let vb = vb_from(map, &dev);
        let backbone = VocosBackbone::load(
            vb,
            input_channels,
            dim,
            intermediate_dim,
            num_layers,
            lsi,
            Some(condition_dim),
        )?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, input_channels, 24), &dev)?;
        let cond_a = Tensor::randn(0.0_f32, 1.0, (2, condition_dim), &dev)?;
        let cond_b = Tensor::randn(0.0_f32, 1.0, (2, condition_dim), &dev)?;

        let y_a = backbone.forward(&x, Some(&cond_a))?;
        let y_b = backbone.forward(&x, Some(&cond_b))?;
        assert_eq!(y_a.dims(), &[2, dim, 24]);
        assert_eq!(y_b.dims(), &[2, dim, 24]);

        // Different conditions must produce different outputs — proves
        // conditioning actually flows through the entry norm + every
        // ConvNeXtBlock norm.
        let max_diff = (&y_a - &y_b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max_diff > 1e-4,
            "expected conditioned forward to differ across conditions, max_diff={max_diff}"
        );
        Ok(())
    }

    #[test]
    fn convnext_block_layer_scale_loaded_when_init_value_some() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 4;
        let intermediate_dim = 8;

        // Build the block twice — once with gamma, once without — and
        // verify (a) load succeeds in both cases and (b) the gamma=0.5
        // path scales the MLP contribution exactly by 0.5 versus the
        // gamma=1.0 baseline.
        let build = |gamma: Option<f32>| -> Result<(ConvNeXtBlock, Tensor)> {
            let mut map = HashMap::new();
            put_convnext_block(&mut map, "", dim, intermediate_dim, None, gamma, &dev)?;
            let map: HashMap<String, Tensor> = map
                .into_iter()
                .map(|(k, v)| (k.trim_start_matches('.').to_owned(), v))
                .collect();
            let vb = vb_from(map, &dev);
            let block = ConvNeXtBlock::load(vb, dim, intermediate_dim, gamma.map(f64::from), None)?;
            let xs: Vec<f32> = (0..(dim * 6)).map(|i| (i as f32) * 0.05).collect();
            let x = Tensor::from_vec(xs, (1, dim, 6), &dev)?;
            let y = block.forward(&x, None)?;
            // Return the MLP contribution: y - x.
            let diff = (&y - &x)?;
            Ok((block, diff))
        };

        let (_, diff_one) = build(Some(1.0))?;
        let (_, diff_half) = build(Some(0.5))?;

        // (y_half - x) should equal 0.5 * (y_one - x) within tolerance,
        // because gamma scales the MLP path linearly.
        let scaled = (&diff_one * 0.5_f64)?;
        let max_abs = (&diff_half - &scaled)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max_abs < 1e-5,
            "LayerScale didn't apply linearly: max_abs={max_abs}"
        );
        Ok(())
    }
}
