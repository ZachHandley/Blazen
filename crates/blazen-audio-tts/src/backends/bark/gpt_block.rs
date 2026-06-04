//! Shared GPT-style transformer block used by all three Bark stages
//! (semantic, coarse, fine). Wave B.2 landed three duplicate copies; this
//! module factors them out so the weight loader (Wave B.3) has a single
//! [`VarBuilder`] convention to honor across stages.
//!
//! Parameter naming follows the **HF transformers** Bark layout
//! (`att_proj`, `out_proj`, `layernorm_1`, `layernorm_2`, `in_proj`,
//! `out_proj`) — that is the canonical state-dict prefix in
//! `transformers/models/bark/modeling_bark.py::BarkBlock` and matches the
//! parameter paths inside `suno/bark-small`'s `pytorch_model.bin`. The
//! upstream `suno-ai/bark/bark/model.py` code uses different names
//! (`c_attn`, `c_proj`, `ln_1`, `ln_2`, `c_fc`); we map to the HF
//! transformers convention because that is what the safetensors / pth
//! state-dict actually contains in the published HF mirror.
//!
//! The single `is_causal` flag lets the fine stage instantiate the same
//! block with non-causal (bidirectional) attention; semantic + coarse
//! stages pass `is_causal=true`.

#![cfg(feature = "bark")]
// `VarBuilder` is consumed by value — the candle convention. Its inner
// state is `Arc<Inner>`-shaped so the move is cheap, and the upstream
// candle-transformers crate uses the same idiom.
#![allow(clippy::needless_pass_by_value)]
// Transformer math is full of (b, t, c) / (q, k, v) tensor names; the
// short single-character bindings match upstream and the math.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{
    LayerNorm, Linear, VarBuilder, layer_norm, layer_norm_no_bias, linear, linear_no_bias,
    ops::softmax_last_dim,
};

/// `LayerNorm` epsilon. Upstream Bark uses `PyTorch`'s default `1e-5`.
const LAYER_NORM_EPS: f64 = 1e-5;

/// Configuration for one [`Block`] / [`CausalSelfAttention`] / [`Mlp`]
/// triplet. Each Bark stage carries its own outer config and projects
/// the relevant fields into this struct.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlockConfig {
    /// Hidden / residual width (must be divisible by `n_head`).
    pub n_embd: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Maximum sequence length supported (used for the optional pre-built
    /// causal mask). Pass the outer stage's `block_size` here.
    pub block_size: usize,
    /// Whether `Linear` / `LayerNorm` projections carry a bias.
    pub bias: bool,
    /// Whether the attention is causal (semantic + coarse) or
    /// bidirectional (fine).
    pub is_causal: bool,
}

/// Combined-QKV multi-head self-attention with an optional causal mask.
///
/// Mirrors `transformers/models/bark/modeling_bark.py::BarkSelfAttention`.
/// The fine stage instantiates this with `is_causal=false`; the semantic
/// and coarse stages pass `is_causal=true`.
pub(super) struct CausalSelfAttention {
    /// Combined Q/K/V projection (HF transformers calls this `att_proj`).
    pub att_proj: Linear,
    /// Attention output projection (HF transformers calls this `out_proj`).
    pub out_proj: Linear,
    pub n_head: usize,
    pub n_embd: usize,
    pub head_dim: usize,
    pub is_causal: bool,
    /// Scale `1 / sqrt(head_dim)` applied to the QK dot product.
    pub scale: f64,
}

impl CausalSelfAttention {
    /// Load the attention sub-module from the supplied `VarBuilder`. The
    /// builder must already be rooted at the attention block's parent
    /// path (HF transformers nests attention under
    /// `layers.{i}.attn`, so callers typically pass `vb.pp("attn")`).
    pub(super) fn load(vb: VarBuilder, cfg: BlockConfig) -> Result<Self> {
        assert!(
            cfg.n_embd.is_multiple_of(cfg.n_head),
            "n_embd ({}) must be divisible by n_head ({})",
            cfg.n_embd,
            cfg.n_head
        );
        let head_dim = cfg.n_embd / cfg.n_head;
        let att_proj = if cfg.bias {
            linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("att_proj"))?
        } else {
            linear_no_bias(cfg.n_embd, 3 * cfg.n_embd, vb.pp("att_proj"))?
        };
        let out_proj = if cfg.bias {
            linear(cfg.n_embd, cfg.n_embd, vb.pp("out_proj"))?
        } else {
            linear_no_bias(cfg.n_embd, cfg.n_embd, vb.pp("out_proj"))?
        };
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            att_proj,
            out_proj,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
            head_dim,
            is_causal: cfg.is_causal,
            scale,
        })
    }

    /// Forward `xs` (`[B, T, n_embd]`) through the attention block.
    ///
    /// When `self.is_causal` the lower-triangular mask is constructed
    /// on the fly from `xs.device()` so the same block can serve
    /// arbitrary prefix lengths (Bark does not yet have a KV cache; that
    /// arrives with Wave B.4's incremental decoding).
    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, t, c) = xs.dims3()?;
        debug_assert_eq!(c, self.n_embd);

        let qkv = self.att_proj.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, c)?;
        let k = qkv.narrow(D::Minus1, c, c)?;
        let v = qkv.narrow(D::Minus1, 2 * c, c)?;

        // [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        let reshape_heads = |t_in: Tensor| -> Result<Tensor> {
            t_in.reshape((b_sz, t, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = reshape_heads(q)?;
        let k = reshape_heads(k)?;
        let v = reshape_heads(v)?;

        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let scores = if self.is_causal {
            let mask = causal_mask(t, xs.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };
        let attn = softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, t, self.n_embd))?;
        self.out_proj.forward(&out)
    }
}

/// Feed-forward MLP — `Linear(n_embd, 4*n_embd) → GELU → Linear(4*n_embd, n_embd)`.
///
/// HF transformers calls these `in_proj` and `out_proj`; the upstream
/// `suno-ai/bark` code calls them `c_fc` and `c_proj`. We honor the HF
/// names because that is what the state-dict ships with.
pub(super) struct Mlp {
    pub in_proj: Linear,
    pub out_proj: Linear,
}

impl Mlp {
    pub(super) fn load(vb: VarBuilder, n_embd: usize, bias: bool) -> Result<Self> {
        let inner = 4 * n_embd;
        let (in_proj, out_proj) = if bias {
            (
                linear(n_embd, inner, vb.pp("in_proj"))?,
                linear(inner, n_embd, vb.pp("out_proj"))?,
            )
        } else {
            (
                linear_no_bias(n_embd, inner, vb.pp("in_proj"))?,
                linear_no_bias(inner, n_embd, vb.pp("out_proj"))?,
            )
        };
        Ok(Self { in_proj, out_proj })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.in_proj.forward(xs)?;
        let xs = xs.gelu()?;
        self.out_proj.forward(&xs)
    }
}

/// One pre-LayerNorm GPT block:
/// `x = x + attn(layernorm_1(x)); x = x + mlp(layernorm_2(x))`.
///
/// Path layout under `vb` (matches HF transformers' `BarkBlock`):
/// `layernorm_1`, `layernorm_2`, `attn.{att_proj,out_proj}`,
/// `mlp.{in_proj,out_proj}`.
pub(super) struct Block {
    pub layernorm_1: LayerNorm,
    pub attn: CausalSelfAttention,
    pub layernorm_2: LayerNorm,
    pub mlp: Mlp,
}

impl Block {
    pub(super) fn load(vb: VarBuilder, cfg: BlockConfig) -> Result<Self> {
        // HF Bark's `BarkLayerNorm` carries a bias only when `config.bias`
        // is set; the published checkpoints use `bias=false`, so the layernorm
        // tensors ship weight-only. Honor the flag both for the linears
        // (above) and the layernorms here.
        let layernorm_1 = load_layer_norm(cfg.n_embd, cfg.bias, vb.pp("layernorm_1"))?;
        let attn = CausalSelfAttention::load(vb.pp("attn"), cfg)?;
        let layernorm_2 = load_layer_norm(cfg.n_embd, cfg.bias, vb.pp("layernorm_2"))?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg.n_embd, cfg.bias)?;
        Ok(Self {
            layernorm_1,
            attn,
            layernorm_2,
            mlp,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.layernorm_1.forward(xs)?;
        let h = self.attn.forward(&h)?;
        let xs = (xs + h)?;
        let h = self.layernorm_2.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        &xs + h
    }
}

/// Load a `LayerNorm` honoring Bark's optional-bias convention.
///
/// HF transformers' `BarkLayerNorm` only allocates a bias parameter when
/// `config.bias` is true; the published `suno/bark*` checkpoints set
/// `bias=false`, so their layernorm tensors are weight-only. candle's plain
/// [`layer_norm`] always expects a `bias` tensor, so we route to
/// [`layer_norm_no_bias`] (weight-only, still mean-removing) when `bias` is
/// false. Shared by [`Block::load`] and each stage's `layernorm_final`.
pub(super) fn load_layer_norm(size: usize, bias: bool, vb: VarBuilder) -> Result<LayerNorm> {
    if bias {
        layer_norm(size, LAYER_NORM_EPS, vb)
    } else {
        layer_norm_no_bias(size, LAYER_NORM_EPS, vb)
    }
}

/// Build an additive `[1, 1, t, t]` causal mask: `0` on/below the
/// diagonal, `-inf` above. Pattern matches the upstream tril mask used
/// by `BarkSelfAttention.forward` in HF transformers.
pub(super) fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            data.push(if j <= i { 0.0_f32 } else { f32::NEG_INFINITY });
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)?.reshape((1, 1, seq_len, seq_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    fn tiny_cfg(is_causal: bool) -> BlockConfig {
        BlockConfig {
            n_embd: 16,
            n_head: 2,
            block_size: 32,
            bias: true,
            is_causal,
        }
    }

    #[test]
    fn causal_block_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg(true);
        let block = Block::load(vb, cfg)?;
        let xs = Tensor::zeros((1, 8, cfg.n_embd), DType::F32, &device)?;
        let out = block.forward(&xs)?;
        assert_eq!(out.dims(), &[1, 8, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn non_causal_block_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg(false);
        let block = Block::load(vb, cfg)?;
        let xs = Tensor::zeros((2, 4, cfg.n_embd), DType::F32, &device)?;
        let out = block.forward(&xs)?;
        assert_eq!(out.dims(), &[2, 4, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn mlp_forward_shape_and_bias() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = Mlp::load(vb, 16, true)?;
        let xs = Tensor::zeros((1, 4, 16), DType::F32, &device)?;
        let out = mlp.forward(&xs)?;
        assert_eq!(out.dims(), &[1, 4, 16]);
        Ok(())
    }

    #[test]
    fn mlp_forward_no_bias() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = Mlp::load(vb, 8, false)?;
        let xs = Tensor::zeros((1, 2, 8), DType::F32, &device)?;
        let out = mlp.forward(&xs)?;
        assert_eq!(out.dims(), &[1, 2, 8]);
        Ok(())
    }

    #[test]
    fn attention_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = tiny_cfg(true);
        let attn = CausalSelfAttention::load(vb, cfg)?;
        let xs = Tensor::zeros((2, 5, cfg.n_embd), DType::F32, &device)?;
        let out = attn.forward(&xs)?;
        assert_eq!(out.dims(), &[2, 5, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn causal_mask_lower_triangular() -> Result<()> {
        let device = Device::Cpu;
        let m = causal_mask(4, &device)?;
        let flat: Vec<f32> = m.flatten_all()?.to_vec1()?;
        assert_eq!(flat.len(), 16);
        for i in 0..4 {
            for j in 0..4 {
                let v = flat[i * 4 + j];
                if j <= i {
                    assert!(v.abs() < f32::EPSILON, "expected 0 at ({i},{j}), got {v}");
                } else {
                    assert!(v.is_infinite() && v.is_sign_negative());
                }
            }
        }
        Ok(())
    }
}
