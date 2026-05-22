//! Triplane-output transformer decoder (Wave T.2).
//!
//! Mirrors the decoder transformer in `TripoSR`
//! (<https://github.com/VAST-AI-Research/TripoSR/blob/main/tsr/models/transformer/transformer_1d.py>).
//! The decoder consumes `DINOv2` image tokens (cross-attention K/V) and a
//! learned set of triplane query tokens (`3 * res * res` tokens at
//! `hidden_dim`) and emits a [`Triplane`] with per-plane shape
//! `(1, triplane_channels, res, res)`.
//!
//! Each block applies, in order, with residual additions between
//! sub-blocks:
//!
//! 1. `LN -> self-attention(triplane_tokens)`
//! 2. `LN -> cross-attention(queries=triplane_tokens, kv=image_tokens)`
//! 3. `LN -> MLP(hidden_dim -> hidden_dim*mlp_ratio -> hidden_dim, GELU)`
//!
//! After the final block: `LN -> Linear(hidden_dim, triplane_channels)`,
//! then a reshape from `(1, 3 * res * res, channels)` into three planes
//! of shape `(1, channels, res, res)` ordered `[xy, yz, xz]` to match
//! [`Triplane::new`].
//!
//! The cross-attention building blocks come from
//! [`blazen_audio_core::dit::Attention`], whose `kv_proj` expects K/V to
//! share the query embed dim. We therefore project the raw `DINOv2`
//! image tokens from `image_token_dim` to `hidden_dim` via a learned
//! linear (`image_kv_proj`) before feeding them into each block's
//! cross-attention.

// Wave T.2 lands each TripoSR sub-module ahead of the consuming
// `pipeline.rs` rewrite (Wave T.3). Until pipeline.rs picks these types
// up, the public surface here looks dead to rustc / clippy.
#![allow(dead_code)]
// `VarBuilder` is the candle convention for module construction --
// every existing backend takes it by value and the move is a cheap
// `Arc` clone. Matches `blazen_audio_core::dit` and the candle
// reference style.
#![allow(clippy::needless_pass_by_value)]
// `dim`, `res`, `ch`, etc. are baked into the surrounding 3D
// vocabulary; the `usize -> f32` casts here are bounded by config
// values that never approach `2^23`.
#![allow(clippy::cast_precision_loss)]

use blazen_3d_core::triplane::{Triplane, TriplaneError};
use blazen_audio_core::dit::Attention;
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use thiserror::Error;

/// Hyperparameters for the `TripoSR` triplane-output transformer decoder.
///
/// The defaults returned by [`TripoSrTransformerConfig::base_default`]
/// match the upstream `TripoSR`-base release.
#[derive(Debug, Clone, Copy)]
pub struct TripoSrTransformerConfig {
    /// Hidden size of the incoming image tokens (`DINOv2` base = 768).
    pub image_token_dim: usize,
    /// Edge length of each square output triplane (in feature pixels).
    pub triplane_resolution: usize,
    /// Output channels stored per plane.
    pub triplane_channels: usize,
    /// Transformer hidden / query-token width.
    pub hidden_dim: usize,
    /// Number of attention heads (must divide `hidden_dim`).
    pub num_attention_heads: usize,
    /// Per-head dim (must equal `hidden_dim / num_attention_heads`).
    pub head_dim: usize,
    /// Number of stacked transformer blocks.
    pub num_layers: usize,
    /// Multiplicative factor for the MLP inner dim.
    pub mlp_ratio: f32,
}

impl TripoSrTransformerConfig {
    /// `TripoSR`-base default hyperparameters.
    ///
    /// - `image_token_dim = 768` (DINOv2-base)
    /// - `triplane_resolution = 32`
    /// - `triplane_channels = 40`
    /// - `hidden_dim = 1024`
    /// - `num_attention_heads = 16`
    /// - `head_dim = 64`
    /// - `num_layers = 16`
    /// - `mlp_ratio = 4.0`
    #[must_use]
    pub const fn base_default() -> Self {
        Self {
            image_token_dim: 768,
            triplane_resolution: 32,
            triplane_channels: 40,
            hidden_dim: 1024,
            num_attention_heads: 16,
            head_dim: 64,
            num_layers: 16,
            mlp_ratio: 4.0,
        }
    }
}

/// Errors raised by [`TripoSrTransformer`] construction and forward
/// passes.
#[derive(Debug, Error)]
pub enum TripoSrTransformerError {
    /// A tensor had the wrong rank or a mismatched dimension.
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    /// A candle tensor / module operation failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Constructing the output [`Triplane`] failed (rank / shape /
    /// device mismatch).
    #[error("blazen-3d-core error: {0}")]
    Core3d(String),
}

impl From<TriplaneError> for TripoSrTransformerError {
    fn from(err: TriplaneError) -> Self {
        Self::Core3d(err.to_string())
    }
}

/// Affine `LayerNorm` with separately-loaded `weight` and `bias` (the
/// `PyTorch` default `LayerNorm` naming used by `TripoSR`'s upstream
/// safetensors keys).
fn layer_norm(dim: usize, vb: VarBuilder) -> Result<candle_nn::LayerNorm, candle_core::Error> {
    let weight = vb.get(dim, "weight")?;
    let bias = vb.get(dim, "bias")?;
    Ok(candle_nn::LayerNorm::new(weight, bias, 1e-5))
}

/// `GELU` MLP: `Linear(hidden, inner) -> GELU -> Linear(inner, hidden)`.
///
/// `TripoSR` uses a plain two-layer `GELU` MLP rather than the
/// `SwiGLU` recipe the audio `DiT` uses, so we don't reuse
/// [`blazen_audio_core::dit::FeedForward`] here.
#[derive(Debug)]
struct GeluMlp {
    fc1: Linear,
    fc2: Linear,
}

impl GeluMlp {
    fn new(hidden: usize, inner: usize, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let fc1 = linear_no_bias(hidden, inner, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(inner, hidden, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu()?;
        self.fc2.forward(&h)
    }
}

/// One transformer block: self-attn -> cross-attn -> MLP, each
/// preceded by its own `LayerNorm` and followed by a residual add.
#[derive(Debug)]
struct Block {
    norm_self: candle_nn::LayerNorm,
    self_attn: Attention,
    norm_cross: candle_nn::LayerNorm,
    cross_attn: Attention,
    norm_mlp: candle_nn::LayerNorm,
    mlp: GeluMlp,
}

impl Block {
    fn new(
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        mlp_inner: usize,
        vb: VarBuilder,
    ) -> Result<Self, candle_core::Error> {
        let norm_self = layer_norm(hidden_dim, vb.pp("norm_self"))?;
        let self_attn = Attention::new_self(
            hidden_dim,
            num_heads,
            head_dim,
            false, // no QK-norm
            vb.pp("self_attn"),
        )?;
        let norm_cross = layer_norm(hidden_dim, vb.pp("norm_cross"))?;
        // `kv_dim == hidden_dim` because image tokens are pre-projected
        // to `hidden_dim` by the outer transformer (see
        // `TripoSrTransformer::image_kv_proj`). This matches the audio
        // `Attention::new_cross` contract, whose `kv_proj` shape
        // `(kv_dim -> 2 * kv_dim)` requires `kv_dim == embed_dim`.
        let cross_attn = Attention::new_cross(
            hidden_dim,
            hidden_dim,
            num_heads,
            head_dim,
            false,
            vb.pp("cross_attn"),
        )?;
        let norm_mlp = layer_norm(hidden_dim, vb.pp("norm_mlp"))?;
        let mlp = GeluMlp::new(hidden_dim, mlp_inner, vb.pp("mlp"))?;
        Ok(Self {
            norm_self,
            self_attn,
            norm_cross,
            cross_attn,
            norm_mlp,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, image_kv: &Tensor) -> Result<Tensor, TripoSrTransformerError> {
        // Self-attention sub-block.
        let h = self.norm_self.forward(x)?;
        let h = self.self_attn.forward(&h, None, None)?;
        let x = (x + h)?;

        // Cross-attention sub-block (queries = triplane tokens, K/V =
        // pre-projected image tokens).
        let h = self.norm_cross.forward(&x)?;
        let h = self.cross_attn.forward(&h, Some(image_kv), None)?;
        let x = (x + h)?;

        // MLP sub-block.
        let h = self.norm_mlp.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let x = (x + h)?;
        Ok(x)
    }
}

/// Triplane-output transformer decoder for `TripoSR`.
///
/// See module docs for the block recipe and weight-key conventions.
#[derive(Debug)]
pub struct TripoSrTransformer {
    /// Learned query tokens of shape `(1, 3 * res * res, hidden_dim)`.
    query_tokens: Tensor,
    /// Projects raw `DINOv2` image tokens from `image_token_dim` to
    /// `hidden_dim` so the cross-attention's `kv_proj` (which expects
    /// `kv_dim == embed_dim`) can consume them.
    image_kv_proj: Linear,
    blocks: Vec<Block>,
    final_norm: candle_nn::LayerNorm,
    /// Projects per-token features from `hidden_dim` to
    /// `triplane_channels`.
    out_proj: Linear,
    config: TripoSrTransformerConfig,
}

impl TripoSrTransformer {
    /// Construct the transformer by reading every learned tensor from
    /// `vb`. Sub-paths used:
    ///
    /// - `query_tokens` -> `(1, 3 * res * res, hidden_dim)` learned
    ///   triplane queries.
    /// - `image_kv_proj` -> linear projection for the raw image
    ///   tokens.
    /// - `blocks.<i>.norm_self`, `blocks.<i>.self_attn`,
    ///   `blocks.<i>.norm_cross`, `blocks.<i>.cross_attn`,
    ///   `blocks.<i>.norm_mlp`, `blocks.<i>.mlp` for each of
    ///   `num_layers` blocks.
    /// - `final_norm` -> trailing `LayerNorm`.
    /// - `out_proj` -> final `hidden_dim -> triplane_channels` linear.
    ///
    /// # Errors
    ///
    /// Returns [`TripoSrTransformerError::Candle`] when a required
    /// tensor is missing or has the wrong shape.
    pub fn load_from_var_builder(
        vb: VarBuilder,
        config: TripoSrTransformerConfig,
    ) -> Result<Self, TripoSrTransformerError> {
        if config.hidden_dim != config.num_attention_heads * config.head_dim {
            return Err(TripoSrTransformerError::InvalidShape(format!(
                "hidden_dim ({}) must equal num_attention_heads ({}) * head_dim ({})",
                config.hidden_dim, config.num_attention_heads, config.head_dim
            )));
        }
        let num_query_tokens = 3 * config.triplane_resolution * config.triplane_resolution;
        let query_tokens = vb.get((1, num_query_tokens, config.hidden_dim), "query_tokens")?;
        let image_kv_proj = linear_no_bias(
            config.image_token_dim,
            config.hidden_dim,
            vb.pp("image_kv_proj"),
        )?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mlp_inner = (config.hidden_dim as f32 * config.mlp_ratio).round() as usize;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(Block::new(
                config.hidden_dim,
                config.num_attention_heads,
                config.head_dim,
                mlp_inner,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        let final_norm = layer_norm(config.hidden_dim, vb.pp("final_norm"))?;
        let out_proj = linear_no_bias(
            config.hidden_dim,
            config.triplane_channels,
            vb.pp("out_proj"),
        )?;
        Ok(Self {
            query_tokens,
            image_kv_proj,
            blocks,
            final_norm,
            out_proj,
            config,
        })
    }

    /// Forward pass.
    ///
    /// `image_tokens` must have shape
    /// `(1, num_image_tokens, image_token_dim)` — typically the `DINOv2`
    /// per-patch outputs (`num_image_tokens = 1370` for DINOv2-base at
    /// 224x224 input including the CLS token). The returned
    /// [`Triplane`] has three planes ordered `[xy, yz, xz]`, each of
    /// shape `(1, triplane_channels, triplane_resolution,
    /// triplane_resolution)`.
    ///
    /// # Errors
    ///
    /// Returns [`TripoSrTransformerError::InvalidShape`] when
    /// `image_tokens` does not match the configured shape, or
    /// [`TripoSrTransformerError::Candle`] on a tensor-op failure.
    pub fn forward(&self, image_tokens: &Tensor) -> Result<Triplane, TripoSrTransformerError> {
        let (b, _t, c) = image_tokens.dims3().map_err(|e| {
            TripoSrTransformerError::InvalidShape(format!(
                "image_tokens must be (1, T, image_token_dim); got {e}"
            ))
        })?;
        if b != 1 {
            return Err(TripoSrTransformerError::InvalidShape(format!(
                "image_tokens batch dim must be 1, got {b}"
            )));
        }
        if c != self.config.image_token_dim {
            return Err(TripoSrTransformerError::InvalidShape(format!(
                "image_tokens last dim must be {}, got {c}",
                self.config.image_token_dim
            )));
        }

        // Pre-project image tokens to `hidden_dim` for cross-attention.
        let image_kv = self.image_kv_proj.forward(image_tokens)?;

        // Triplane queries are a single learned tensor that does not
        // depend on the image; we just clone the handle (candle tensors
        // are shallow-cloneable) before running blocks.
        let mut x = self.query_tokens.clone();
        for block in &self.blocks {
            x = block.forward(&x, &image_kv)?;
        }
        let x = self.final_norm.forward(&x)?;
        // (1, 3 * res * res, hidden_dim) -> (1, 3 * res * res,
        // triplane_channels).
        let x = self.out_proj.forward(&x)?;

        let res = self.config.triplane_resolution;
        let ch = self.config.triplane_channels;
        // Reshape (1, 3, res, res, ch) and pull each plane out.
        let x = x.reshape((1, 3, res, res, ch))?;
        // Per-plane: select index along dim 1 -> (1, res, res, ch),
        // then permute to (1, ch, res, res) to match the Triplane
        // contract.
        let mut planes: Vec<Tensor> = Vec::with_capacity(3);
        for i in 0..3 {
            let plane = x
                .narrow(1, i, 1)? // (1, 1, res, res, ch)
                .squeeze(1)? // (1, res, res, ch)
                .permute((0, 3, 1, 2))? // (1, ch, res, res)
                .contiguous()?;
            planes.push(plane);
        }
        let planes_arr: [Tensor; 3] = planes.try_into().map_err(|_| {
            TripoSrTransformerError::InvalidShape("expected exactly 3 planes after reshape".into())
        })?;
        Triplane::new(planes_arr).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn config_base_default_hyperparameters_match_reference() {
        let cfg = TripoSrTransformerConfig::base_default();
        assert_eq!(cfg.image_token_dim, 768);
        assert_eq!(cfg.triplane_resolution, 32);
        assert_eq!(cfg.triplane_channels, 40);
        assert_eq!(cfg.hidden_dim, 1024);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_layers, 16);
        assert!((cfg.mlp_ratio - 4.0).abs() < f32::EPSILON);
        assert_eq!(
            cfg.hidden_dim,
            cfg.num_attention_heads * cfg.head_dim,
            "hidden_dim must factor cleanly into heads * head_dim"
        );
    }

    #[test]
    fn forward_shape_matches_triplane_contract() {
        let dev = Device::Cpu;
        let cfg = TripoSrTransformerConfig {
            image_token_dim: 64,
            triplane_resolution: 4,
            triplane_channels: 8,
            hidden_dim: 64,
            num_attention_heads: 4,
            head_dim: 16,
            num_layers: 2,
            mlp_ratio: 2.0,
        };
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let transformer = TripoSrTransformer::load_from_var_builder(vb, cfg)
            .expect("transformer build from zero-init VarBuilder");

        let image_tokens = Tensor::zeros((1, 5, cfg.image_token_dim), DType::F32, &dev).unwrap();
        let triplane = transformer
            .forward(&image_tokens)
            .expect("forward should succeed with matching shapes");

        assert_eq!(triplane.channels_per_plane(), cfg.triplane_channels);
        assert_eq!(triplane.resolution(), cfg.triplane_resolution);
    }
}
