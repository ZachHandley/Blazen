//! Vendored fork of `candle_transformers::models::dac` (v0.10.2) — Descript
//! Audio Codec (DAC).
//!
//! See [`VENDORED.md`](../VENDORED.md) for the full vendoring rationale.
//! In short: upstream's `VectorQuantizer` keeps its `in_proj` projection and
//! `codebook` embedding **private**, so external callers cannot run
//! nearest-neighbour quantisation against a loaded DAC checkpoint —
//! `Model::decode_codes` is the only public verb. This fork (a) exposes
//! those internal fields, (b) adds [`VectorQuantizer::forward`] which
//! mirrors the reference Python implementation
//! (`descript-audio-codec/dac/nn/quantize.py::VectorQuantize.forward`),
//! and (c) adds [`Model::encode_to_codes`] so the parent codec crate can
//! wire a complete encode + decode round-trip without patching upstream.
//!
//! Aside from the changes above, the source is a verbatim port of the
//! upstream module; please keep the diff minimal when re-syncing.
//!
//! ## License
//!
//! Upstream `candle-transformers` ships under MIT OR Apache-2.0 (see
//! `VENDORED.md`). This fork is redistributed under the Blazen workspace
//! license (MPL-2.0) per upstream's MIT permissions; the upstream
//! copyright notice is preserved in `VENDORED.md`.

// Pedantic lints fire heavily on the verbatim port (cast widths, fn arg
// counts, doc-related nits that came in from upstream). The vendored
// fork keeps the upstream coding style on purpose — silence the
// pedantic class crate-wide so re-syncs stay one-to-one with upstream.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::unreadable_literal,
    clippy::module_name_repetitions
)]

use candle_core::{D, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use candle_transformers::models::encodec;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub num_codebooks: usize,
    pub model_bitrate: u32,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub frame_rate: u32,
    pub sampling_rate: u32,
}

#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp(1))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = encodec::conv1d_weight_norm(dim, dim, 1, Conv1dConfig::default(), vb.pp(3))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;
        let pad = (xs.dim(D::Minus1)? - ys.dim(D::Minus1)?) / 2;
        if pad > 0 {
            &ys + xs.narrow(D::Minus1, pad, ys.dim(D::Minus1)?)
        } else {
            ys + xs
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncoderBlock {
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl EncoderBlock {
    pub fn new(dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let res1 = ResidualUnit::new(dim / 2, 1, vb.pp(0))?;
        let res2 = ResidualUnit::new(dim / 2, 3, vb.pp(1))?;
        let res3 = ResidualUnit::new(dim / 2, 9, vb.pp(2))?;
        let snake1 = Snake1d::new(dim / 2, vb.pp(3))?;
        let cfg1 = Conv1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim / 2, dim, 2 * stride, cfg1, vb.pp(4))?;
        Ok(Self {
            res1,
            res2,
            res3,
            snake1,
            conv1,
        })
    }
}

impl Module for EncoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)?
            .apply(&self.snake1)?
            .apply(&self.conv1)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in &self.blocks {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

impl Encoder {
    pub fn new(
        mut d_model: usize,
        strides: &[usize],
        d_latent: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(1, d_model, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (block_idx, stride) in strides.iter().enumerate() {
            d_model *= 2;
            let block = EncoderBlock::new(d_model, *stride, vb.pp(block_idx + 1))?;
            blocks.push(block);
        }
        let snake1 = Snake1d::new(d_model, vb.pp(strides.len() + 1))?;
        let cfg2 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv2 =
            encodec::conv1d_weight_norm(d_model, d_latent, 3, cfg2, vb.pp(strides.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: ConvTranspose1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv_tr1 = encodec::conv_transpose1d_weight_norm(
            in_dim,
            out_dim,
            2 * stride,
            true,
            cfg,
            vb.pp(1),
        )?;
        let res1 = ResidualUnit::new(out_dim, 1, vb.pp(2))?;
        let res2 = ResidualUnit::new(out_dim, 3, vb.pp(3))?;
        let res3 = ResidualUnit::new(out_dim, 9, vb.pp(4))?;
        Ok(Self {
            snake1,
            conv_tr1,
            res1,
            res2,
            res3,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_tr1)?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    pub fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(in_c, channels, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(rates.len());
        for (idx, stride) in rates.iter().enumerate() {
            let block = DecoderBlock::new(channels, channels / 2, *stride, vb.pp(idx + 1))?;
            channels /= 2;
            blocks.push(block);
        }
        let snake1 = Snake1d::new(channels, vb.pp(rates.len() + 1))?;
        let conv2 = encodec::conv1d_weight_norm(channels, d_out, 7, cfg1, vb.pp(rates.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in &self.blocks {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

/// Factorised vector quantiser. **Changed from upstream**: `in_proj` and
/// `codebook` are now `pub` so external callers can run nearest-neighbour
/// quantisation via [`VectorQuantizer::forward`].
#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    pub in_proj: Conv1d,
    pub out_proj: Conv1d,
    pub codebook: candle_nn::Embedding,
}

impl VectorQuantizer {
    pub fn new(in_dim: usize, cb_size: usize, cb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj = encodec::conv1d_weight_norm(
            in_dim,
            cb_dim,
            1,
            Conv1dConfig::default(),
            vb.pp("in_proj"),
        )?;
        let out_proj = encodec::conv1d_weight_norm(
            cb_dim,
            in_dim,
            1,
            Conv1dConfig::default(),
            vb.pp("out_proj"),
        )?;
        let codebook = candle_nn::embedding(cb_size, cb_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
        })
    }

    pub fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        embed_id.apply(&self.codebook)
    }

    pub fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }

    /// **New**: nearest-neighbour quantisation of a `[B, in_dim, T]`
    /// latent slab. Mirrors `descript-audio-codec/dac/nn/quantize.py`'s
    /// `VectorQuantize.forward` (see `decode_latents`):
    ///
    /// 1. `z_e = in_proj(z)` — project into the low-dim codebook space.
    /// 2. L2-normalise both `z_e` and the codebook (ViT-VQGAN trick).
    /// 3. Pick the codebook entry that minimises Euclidean distance per
    ///    `(B, T)` slot — equivalent to maximising cosine similarity
    ///    once both sides are unit-normed.
    /// 4. Look the chosen indices back up to recover the quantised
    ///    latent and apply `out_proj` to return to model dim.
    ///
    /// Returns `(z_q, indices)` where `z_q` has shape `[B, in_dim, T]`
    /// (i.e. the same shape as `z`) and `indices` has shape `[B, T]`
    /// holding `u32` codebook indices.
    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        // z: [B, in_dim, T] -> z_e: [B, cb_dim, T]
        let z_e = z.apply(&self.in_proj)?;

        let (b, d, t) = z_e.dims3()?;
        // [B, D, T] -> [B, T, D] -> [B*T, D]
        let z_e_btd = z_e.transpose(1, 2)?.contiguous()?;
        let enc = z_e_btd.reshape((b * t, d))?;

        // L2-normalise rows of `enc` ([B*T, D]) and rows of the codebook
        // ([N, D]) — `embedding.weight` is shape `[N, D]`.
        let enc_norm = l2_normalize(&enc)?;
        let codebook = self.codebook.embeddings(); // [N, D]
        let cb_norm = l2_normalize(codebook)?;

        // dist[i, n] = ||enc_norm[i]||^2 - 2 enc_norm[i].dot(cb_norm[n]) + ||cb_norm[n]||^2
        // After L2-norm both squared-norms are ~1, but we keep the full
        // expansion to match the upstream Python reference exactly.
        let enc_sq = enc_norm.sqr()?.sum_keepdim(D::Minus1)?; // [B*T, 1]
        let cb_sq = cb_norm.sqr()?.sum_keepdim(D::Minus1)?; // [N, 1]
        let cb_sq_row = cb_sq.transpose(0, 1)?; // [1, N]
        let cross = enc_norm.matmul(&cb_norm.transpose(0, 1)?.contiguous()?)?; // [B*T, N]
        let dist = enc_sq
            .broadcast_add(&cb_sq_row)?
            .broadcast_sub(&(cross * 2.0)?)?;

        // argmin over the N codebook axis. `argmax(-dist)` keeps numeric
        // parity with the Python reference (`(-dist).max(1)[1]`).
        let indices = dist.neg()?.argmax(D::Minus1)?; // [B*T], u32
        let indices = indices.reshape((b, t))?; // [B, T]

        // Look the selected codes back up and reshape to [B, cb_dim, T]
        // before projecting out to model dim.
        let z_q_low = self.decode_code(&indices)?; // [B, cb_dim, T]
        let z_q = z_q_low.apply(&self.out_proj)?; // [B, in_dim, T]

        Ok((z_q, indices))
    }
}

/// L2-normalise `xs` along the last dimension with a small epsilon to
/// match `torch.nn.functional.normalize`'s default behaviour
/// (eps = 1e-12).
fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    // Avoid division by zero on dead codebook entries; PyTorch clamps to
    // `eps` rather than adding it, but the numerical difference here is
    // negligible for our use case (codebook entries are never exactly
    // zero in a trained DAC checkpoint).
    let denom = (norm + 1e-12)?;
    xs.broadcast_div(&denom)
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    pub quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub fn new(
        input_dim: usize,
        n_codebooks: usize,
        cb_size: usize,
        cb_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = &vb.pp("quantizers");
        let quantizers = (0..n_codebooks)
            .map(|i| VectorQuantizer::new(input_dim, cb_size, cb_dim, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizers })
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            let z_p_i = quantizer.decode_code(&codes.i((.., idx))?)?;
            let z_q_i = z_p_i.apply(&quantizer.out_proj)?;
            let s = match sum {
                None => z_q_i,
                Some(s) => (s + z_q_i)?,
            };
            sum = Some(s);
        }
        match sum {
            Some(s) => Ok(s),
            None => candle_core::bail!("empty codebooks"),
        }
    }

    /// **New**: forward residual quantisation of a `[B, input_dim, T]`
    /// latent slab. Mirrors `ResidualVectorQuantize.forward` in
    /// `descript-audio-codec/dac/nn/quantize.py` in inference mode
    /// (no quantiser dropout): for each of the `n_codebooks` quantisers
    /// in turn, quantise the running residual, accumulate the
    /// reconstruction, and subtract the per-step contribution from the
    /// residual.
    ///
    /// Returns the stacked codebook indices with shape
    /// `[B, n_codebooks, T]` (u32), ready to feed straight back into
    /// [`Self::from_codes`] for verification or into
    /// [`Model::decode_codes`] for resynthesis.
    pub fn forward_codes(&self, z: &Tensor) -> Result<Tensor> {
        let mut residual = z.clone();
        let mut indices: Vec<Tensor> = Vec::with_capacity(self.quantizers.len());
        for quantizer in &self.quantizers {
            let (z_q_i, idx_i) = quantizer.forward(&residual)?;
            residual = (residual - z_q_i)?;
            indices.push(idx_i);
        }
        if indices.is_empty() {
            candle_core::bail!("empty codebooks");
        }
        // Stack along a new codebook axis: each `idx_i` is [B, T] -> stacked is
        // [B, n_codebooks, T].
        Tensor::stack(&indices, 1)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub encoder: Encoder,
    pub quantizer: ResidualVectorQuantizer,
    pub decoder: Decoder,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(64, &[2, 4, 8, 8], cfg.latent_dim, vb.pp("encoder"))?;
        let quantizer = ResidualVectorQuantizer::new(
            cfg.latent_dim,
            cfg.num_codebooks,
            cfg.codebook_size,
            8,
            vb.pp("quantizer"),
        )?;
        let decoder = Decoder::new(cfg.latent_dim, 1536, &[8, 8, 4, 2], 1, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            quantizer,
            decoder,
        })
    }

    pub fn decode_codes(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let audio_values = self.quantizer.from_codes(audio_codes)?;
        audio_values.apply(&self.decoder)
    }

    /// **New**: full encode pipeline — runs the audio through the
    /// encoder to produce a continuous latent then through the residual
    /// vector quantiser to produce discrete codebook indices.
    ///
    /// `audio` must be shape `[B, 1, T]` (mono PCM, batch-of-one is
    /// the typical use). Returns `[B, n_codebooks, T']` where `T'` is
    /// the down-sampled frame count after the encoder's stride stack
    /// (`[2, 4, 8, 8]` -> total stride 512 for the canonical
    /// `descript/dac_44khz` checkpoint).
    pub fn encode_to_codes(&self, audio: &Tensor) -> Result<Tensor> {
        let z = audio.apply(&self.encoder)?;
        self.quantizer.forward_codes(&z)
    }
}
