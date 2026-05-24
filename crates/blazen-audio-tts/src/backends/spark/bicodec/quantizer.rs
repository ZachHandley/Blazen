//! Vector + scalar quantizers used by `BiCodec` (sub-wave **S.2.1.d** of
//! the [`super`] `BiCodec` port).
//!
//! Three quantizers ship here:
//!
//! * [`FactorizedVectorQuantize`] — the *semantic* stream's codebook
//!   lookup. Mirrors `sparktts/modules/vq/factorized_vector_quantize.py`
//!   in inference mode (no EMA cluster updates, no commitment / codebook
//!   losses, no straight-through estimator). It projects `(B, input_dim,
//!   T)` into a low-dimensional cosine space, L2-normalises both the
//!   projected query and every codebook entry, picks the nearest
//!   codebook row, and projects back out. The `BiCodec` checkpoint uses
//!   `input_dim = 1024`, `codebook_size = 8192`, `codebook_dim = 8`, so
//!   both projections are real `WNConv1d` 1×1 convs (NOT Identity).
//! * [`Fsq`] — finite scalar quantizer
//!   (`sparktts/modules/fsq/finite_scalar_quantization.py`). Each of the
//!   `len(levels)` axes is independently snapped to one of `levels[d]`
//!   evenly-spaced grid points, producing a single integer index in
//!   `[0, prod(levels))`. For Spark-TTS the levels are `[4, 4, 4, 4, 4,
//!   4]` so the basis vector is `[1, 4, 16, 64, 256, 1024]` and the
//!   codebook size is `4^6 = 4096`.
//! * [`ResidualFsq`] — thin wrapper over a list of [`Fsq`] layers
//!   (`sparktts/modules/fsq/residual_fsq.py`). For Spark-TTS the speaker
//!   encoder always uses `num_quantizers = 1`, so this is effectively
//!   *one* FSQ plus a `Linear(dim ↔ codebook_dim)` pair (the upstream
//!   passes `dim=codebook_dim` into each inner FSQ — line 89 of
//!   `residual_fsq.py` — so the inner FSQ never has its own projections;
//!   they live on the wrapper).
//!
//! # Inference-only port
//!
//! The upstream Python carries training-only machinery (EMA cluster
//! tracking, `commit_loss`, `codebook_loss`, straight-through
//! estimator, quantize-dropout, `dist.all_reduce`). We drop all of it:
//! `BiCodec`'s inference path only needs `tokenize` / `detokenize` /
//! `get_output_from_indices`.
//!
//! # State-dict key paths
//!
//! * [`FactorizedVectorQuantize`]: `in_project.{weight_g, weight_v,
//!   bias}`, `out_project.{weight_g, weight_v, bias}`,
//!   `codebook.weight`. The upstream Python field names are
//!   `in_project` / `out_project` (matching
//!   `factorized_vector_quantize.py` lines 60–61) — NOT `input_proj` /
//!   `output_proj`.
//! * [`Fsq`]: `project_in.{weight, bias}` and `project_out.{weight,
//!   bias}` (only present when `dim != effective_codebook_dim` — see
//!   `has_projections` upstream).
//! * [`ResidualFsq`]: `project_in.{weight, bias}`,
//!   `project_out.{weight, bias}`, `layers.{i}.…` (one inner FSQ per
//!   residual layer; inner FSQ's `has_projections` is always `False`
//!   because upstream passes `dim=codebook_dim`).
//!
//! # Banker's rounding (round-half-to-even)
//!
//! `torch.round()` uses round-half-to-even (banker's rounding) — that's
//! the same rule IEEE-754 calls "to nearest, ties to even". Rust's
//! [`f32::round`] is round-half-away-from-zero, which would diverge
//! from `PyTorch` on any half-integer input. For bit-exact FSQ parity we
//! implement the banker's rule explicitly inside [`round_half_to_even`]
//! — without that, an exactly-`±0.5` activation lands on a different
//! codebook entry than the Python reference.

// `VarBuilder` is the canonical "consume by value" handle in candle.
// See primitives.rs for the same lint waiver and rationale.
#![allow(clippy::needless_pass_by_value)]

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear};

use super::primitives::WeightNormConv1d;

// ---------------------------------------------------------------------------
// FactorizedVectorQuantize
// ---------------------------------------------------------------------------

/// One of the two faces of the semantic-stream codebook lookup: a
/// `WNConv1d(input_dim → codebook_dim)` projection when the input and
/// codebook dimensions differ, or an identity pass-through when they
/// already match.
///
/// `BiCodec`'s `quantizer.in_project` / `quantizer.out_project` are real
/// `WNConv1d` 1×1 convs (`input_dim = 1024 ≠ codebook_dim = 8`), so the
/// `Identity` arm is never taken in production — it exists purely so
/// the module still compiles cleanly under a hypothetical config where
/// `input_dim == codebook_dim` (matching upstream's `nn.Identity()`
/// fallback at `factorized_vector_quantize.py` lines 64–65).
#[derive(Debug, Clone)]
enum Projection {
    WnConv(WeightNormConv1d),
    Identity,
}

impl Projection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::WnConv(c) => c.forward(x),
            Self::Identity => Ok(x.clone()),
        }
    }

    fn load(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        if in_channels == out_channels {
            return Ok(Self::Identity);
        }
        let conv = WeightNormConv1d::load(
            vb,
            in_channels,
            out_channels,
            /* kernel_size */ 1,
            /* stride */ 1,
            /* padding */ 0,
            /* dilation */ 1,
            /* groups */ 1,
            /* with_bias */ true,
        )?;
        Ok(Self::WnConv(conv))
    }
}

/// Factorised vector quantizer for the `BiCodec` semantic stream.
///
/// Inference-only port of `FactorizedVectorQuantize` from
/// `sparktts/modules/vq/factorized_vector_quantize.py`.
///
/// # Algorithm
///
/// 1. Project `z: (B, input_dim, T) → z_e: (B, codebook_dim, T)` via
///    [`Projection`] (`WNConv1d` 1×1 in the `BiCodec` checkpoint).
/// 2. Flatten to `(B·T, codebook_dim)` and L2-normalise along the
///    feature axis.
/// 3. L2-normalise the codebook rows the same way. The matching metric
///    is then cosine similarity, equivalent to Euclidean distance on
///    the unit hypersphere — this is the "factorized" trick.
/// 4. `indices = argmin_n ‖q - c_n‖² = argmax_n q·c_n`.
/// 5. For [`detokenize`](Self::detokenize), look the chosen rows back
///    up from the *unnormalised* codebook (matching upstream
///    `decode_code` → `F.embedding(indices, codebook.weight)`), then
///    transpose to `(B, codebook_dim, T)` and project back out via
///    `out_project`.
#[derive(Debug, Clone)]
pub(super) struct FactorizedVectorQuantize {
    in_project: Projection,
    out_project: Projection,
    codebook: Embedding,
    codebook_size: usize,
    codebook_dim: usize,
}

impl FactorizedVectorQuantize {
    /// Load the quantizer from `vb`. Expects state-dict children
    /// `in_project`, `out_project`, and `codebook` (see module docs).
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from missing tensors or shape
    /// mismatches in the underlying child loaders.
    pub(super) fn load(
        vb: VarBuilder,
        input_dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        let in_project = Projection::load(vb.pp("in_project"), input_dim, codebook_dim)?;
        let out_project = Projection::load(vb.pp("out_project"), codebook_dim, input_dim)?;
        let codebook = embedding(codebook_size, codebook_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_project,
            out_project,
            codebook,
            codebook_size,
            codebook_dim,
        })
    }

    /// Codebook size (number of entries).
    pub(super) fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Codebook entry dimensionality.
    pub(super) fn codebook_dim(&self) -> usize {
        self.codebook_dim
    }

    /// Encode `z: (B, input_dim, T) → indices: (B, T)` (`u32`).
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the projection, L2-norm
    /// computation, or matmul/argmax fallbacks.
    pub(super) fn tokenize(&self, z: &Tensor) -> Result<Tensor> {
        let z_e = self.in_project.forward(z)?;
        let (b, _d, t) = z_e.dims3()?;

        // Flatten (B, D, T) → (B·T, D) via transpose first.
        let enc = z_e
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, self.codebook_dim))?;
        let enc_norm = l2_normalize_last(&enc)?;

        let codebook = self.codebook.embeddings(); // (N, D)
        let cb_norm = l2_normalize_last(codebook)?;

        // dist[i, n] = ‖enc_norm[i]‖² - 2·enc_norm[i]·cb_norm[n] + ‖cb_norm[n]‖²
        // After L2-norm both squared-norms are ≈ 1, but we keep the
        // full expansion to mirror the upstream Python reference.
        let enc_sq = enc_norm.sqr()?.sum_keepdim(D::Minus1)?; // (B·T, 1)
        let cb_sq = cb_norm.sqr()?.sum_keepdim(D::Minus1)?; // (N, 1)
        let cb_sq_row = cb_sq.transpose(0, 1)?; // (1, N)
        let cross = enc_norm.matmul(&cb_norm.transpose(0, 1)?.contiguous()?)?; // (B·T, N)
        let dist = enc_sq
            .broadcast_add(&cb_sq_row)?
            .broadcast_sub(&(cross * 2.0)?)?;

        // `argmax(-dist)` matches upstream's `(-dist).max(1)[1]`.
        let indices = dist.neg()?.argmax(D::Minus1)?; // (B·T,) u32
        indices.reshape((b, t))
    }

    /// Decode `indices: (B, T) → z_q: (B, input_dim, T)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the embedding lookup,
    /// transpose, or out-projection.
    pub(super) fn detokenize(&self, indices: &Tensor) -> Result<Tensor> {
        // (B, T) → (B, T, D)
        let z_q = self.codebook.forward(indices)?;
        // (B, T, D) → (B, D, T)
        let z_q = z_q.transpose(1, 2)?.contiguous()?;
        self.out_project.forward(&z_q)
    }
}

/// L2-normalise along the last dim. Mirrors
/// `torch.nn.functional.normalize` (default `dim=-1`, `eps=1e-12`).
fn l2_normalize_last(xs: &Tensor) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let denom = (norm + 1e-12)?;
    xs.broadcast_div(&denom)
}

// ---------------------------------------------------------------------------
// FSQ (finite scalar quantizer)
// ---------------------------------------------------------------------------

/// Finite scalar quantizer (the lucidrains / Mentzer-et-al. 2023 design
/// from <https://arxiv.org/abs/2309.15505>).
///
/// Inference-only port of `FSQ` from
/// `sparktts/modules/fsq/finite_scalar_quantization.py`. Each of the
/// `len(levels)` quantization axes is bounded by a shifted-scaled
/// `tanh`, rounded to its nearest grid point, and the resulting integer
/// per-axis labels are packed into a single codebook index via a
/// mixed-radix basis (`cumprod([1] + levels[:-1])`).
///
/// # Spark-TTS configuration
///
/// `levels = [4, 4, 4, 4, 4, 4]`, `dim = 6`, `num_codebooks = 1` when
/// constructed by [`ResidualFsq`] (which passes `dim=codebook_dim`).
/// In that wiring `has_projections == False` and the inner FSQ is a
/// pure scalar-quantize stage; the projections live on the outer
/// [`ResidualFsq`].
///
/// # Channel layout
///
/// This module exposes [`tokenize`](Self::tokenize) /
/// [`detokenize`](Self::detokenize) as *channels-first* operations so
/// the surrounding `BiCodec` code (and [`ResidualFsq`]) doesn't need to
/// scatter `transpose(1, 2)` calls everywhere. The upstream Python
/// `FSQ.forward` is channels-last by default; we transpose at the
/// boundary.
#[derive(Debug, Clone)]
pub(super) struct Fsq {
    /// Per-axis quantization level count, length `codebook_dim`.
    levels: Vec<u32>,
    /// `cumprod([1] + levels[:-1])` — mixed-radix basis for
    /// `codes ↔ index`.
    basis: Vec<u32>,
    /// Outer-facing feature dim. Equals `effective_codebook_dim` when
    /// the FSQ is used from inside [`ResidualFsq`].
    dim: usize,
    /// `len(levels) * num_codebooks`. For Spark-TTS = 6.
    effective_codebook_dim: usize,
    /// Per-axis FSQ codebook width (= `len(levels)`).
    codebook_dim: usize,
    /// Number of independent FSQ codebooks packed into the feature dim.
    /// For Spark-TTS = 1.
    num_codebooks: usize,
    /// Cached `prod(levels)`. Spark-TTS = 4096.
    codebook_size: u32,
    project_in: Option<Linear>,
    project_out: Option<Linear>,
}

impl Fsq {
    /// Load an FSQ. Pass `vb` rooted at the FSQ's state-dict subtree.
    /// `project_in` / `project_out` are only created (and only loaded)
    /// when `dim != num_codebooks * len(levels)` — matching upstream's
    /// `has_projections` flag.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the optional
    /// `linear(…)` loaders. Returns an error if `levels` is empty.
    pub(super) fn load(
        vb: VarBuilder,
        levels: &[u32],
        dim: usize,
        num_codebooks: usize,
    ) -> Result<Self> {
        if levels.is_empty() {
            candle_core::bail!("Fsq::load: `levels` must be non-empty");
        }
        let codebook_dim = levels.len();
        let effective_codebook_dim = codebook_dim * num_codebooks;
        let has_projections = dim != effective_codebook_dim;
        let project_in = if has_projections {
            Some(linear(dim, effective_codebook_dim, vb.pp("project_in"))?)
        } else {
            None
        };
        let project_out = if has_projections {
            Some(linear(effective_codebook_dim, dim, vb.pp("project_out"))?)
        } else {
            None
        };

        let basis = compute_basis(levels);
        let codebook_size = levels.iter().product();

        Ok(Self {
            levels: levels.to_vec(),
            basis,
            dim,
            effective_codebook_dim,
            codebook_dim,
            num_codebooks,
            codebook_size,
            project_in,
            project_out,
        })
    }

    /// Per-axis level counts (length = `codebook_dim`).
    pub(super) fn levels(&self) -> &[u32] {
        &self.levels
    }

    /// Mixed-radix basis vector, `cumprod([1] + levels[:-1])`.
    pub(super) fn basis(&self) -> &[u32] {
        &self.basis
    }

    /// Total codebook size, `prod(levels)`. Spark-TTS: 4096.
    pub(super) fn codebook_size(&self) -> u32 {
        self.codebook_size
    }

    /// Per-FSQ codebook width (= `len(levels)`).
    pub(super) fn codebook_dim(&self) -> usize {
        self.codebook_dim
    }

    /// Outer-facing feature dim.
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stacked FSQ codebooks. Spark-TTS: 1.
    pub(super) fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Encode channels-first `z: (B, dim, T) → indices: (B, T,
    /// num_codebooks)` (`u32`).
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the projection, bound /
    /// round arithmetic, or the basis dot-product.
    pub(super) fn tokenize(&self, z: &Tensor) -> Result<Tensor> {
        let codes = self.quantize_codes(z)?; // (B, T, num_codebooks, codebook_dim)
        self.codes_to_indices(&codes)
    }

    /// Decode channels-first `indices: (B, T, num_codebooks) → codes:
    /// (B, dim, T)`. Applies `project_out` when present.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the per-axis modulo /
    /// rescale / projection.
    pub(super) fn detokenize(&self, indices: &Tensor) -> Result<Tensor> {
        // (B, T, num_codebooks, codebook_dim), grid-centered floats in
        // [-1, 1].
        let codes = self.indices_to_codes(indices)?;

        // Collapse the per-codebook dim: (B, T, num_cb, cb_dim) →
        // (B, T, effective_codebook_dim).
        let (b, t, _c, _d) = codes.dims4()?;
        let codes = codes.reshape((b, t, self.effective_codebook_dim))?;

        // Optional `project_out`: (B, T, effective) → (B, T, dim).
        let codes = if let Some(po) = &self.project_out {
            po.forward(&codes)?
        } else {
            codes
        };

        // Channels-last → channels-first.
        codes.transpose(1, 2)?.contiguous()
    }

    /// Encode `z: (B, dim, T)` to grid-centered codes in `[-1, 1]`,
    /// shape `(B, T, num_codebooks, codebook_dim)`. This is the
    /// `quantize ∘ project_in ∘ rearrange` pipeline upstream's
    /// `FSQ.forward` runs before computing indices.
    fn quantize_codes(&self, z: &Tensor) -> Result<Tensor> {
        // Channels-first → channels-last for project_in.
        let (b, d, t) = z.dims3()?;
        if d != self.dim {
            candle_core::bail!(
                "Fsq::quantize_codes: expected feature dim {}, got {d}",
                self.dim
            );
        }
        let z_btd = z.transpose(1, 2)?.contiguous()?; // (B, T, dim)

        let z_btd = if let Some(pi) = &self.project_in {
            pi.forward(&z_btd)? // (B, T, effective)
        } else {
            z_btd
        };

        // (B, T, effective) → (B, T, num_codebooks, codebook_dim)
        let z_packed = z_btd.reshape((b, t, self.num_codebooks, self.codebook_dim))?;

        // bound + round
        self.quantize_and_normalize(&z_packed)
    }

    /// Apply `bound` then `round_half_to_even` then divide by
    /// `levels // 2` so codes are grid-centered in `[-1, 1]`.
    /// Mirrors upstream's `FSQ.quantize` (lines 133-137).
    fn quantize_and_normalize(&self, z: &Tensor) -> Result<Tensor> {
        let bounded = self.bound(z)?;
        let rounded = round_half_to_even(&bounded)?;
        // half_width = levels // 2  ; renormalize to [-1, 1].
        let half_width = self.half_width_tensor(z.device(), z.dtype())?;
        rounded.broadcast_div(&half_width)
    }

    /// `bound(z) = tanh(z + shift) * half_l - offset` with `half_l =
    /// (level - 1) * (1 + eps) / 2` and `offset = 0.5` when `level` is
    /// even, else `0.0`. Mirrors upstream `FSQ.bound` (lines 126-131).
    fn bound(&self, z: &Tensor) -> Result<Tensor> {
        let dtype = z.dtype();
        let dev = z.device();
        let eps = 1e-3_f64;

        // Per-axis half_l = (level - 1) * (1 + eps) / 2, computed in f64
        // for precision then cast to f32. `levels` are small (≤ 4 in
        // Spark-TTS), so the cast is exact.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "half_l is < (max_level - 1) ≈ 3.003 — well within f32 \
                      range; FSQ levels are tiny (≤ 4 for Spark-TTS)."
        )]
        let half_l: Vec<f32> = self
            .levels
            .iter()
            .map(|&l| ((f64::from(l) - 1.0) * (1.0 + eps) / 2.0) as f32)
            .collect();
        let offset_vec: Vec<f32> = self
            .levels
            .iter()
            .map(|&l| if l % 2 == 0 { 0.5_f32 } else { 0.0 })
            .collect();
        // shift = atanh(offset / half_l); when offset == 0, shift = 0.
        let shift_vec: Vec<f32> = self
            .levels
            .iter()
            .zip(half_l.iter())
            .map(|(&l, &h)| {
                if l % 2 == 0 {
                    (0.5_f32 / h).atanh()
                } else {
                    0.0
                }
            })
            .collect();

        // Broadcast over (B, T, num_codebooks, codebook_dim) — only
        // the last axis varies.
        let half_l_t =
            Tensor::from_vec(half_l, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)?;
        let offset_t =
            Tensor::from_vec(offset_vec, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)?;
        let shift_t =
            Tensor::from_vec(shift_vec, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)?;

        // tanh(z + shift) * half_l - offset
        let shifted = z.broadcast_add(&shift_t)?;
        let t = shifted.tanh()?;
        let scaled = t.broadcast_mul(&half_l_t)?;
        scaled.broadcast_sub(&offset_t)
    }

    /// `half_width = (levels // 2)` broadcast to shape
    /// `(1, 1, 1, codebook_dim)`.
    fn half_width_tensor(&self, dev: &Device, dtype: DType) -> Result<Tensor> {
        // `levels` are tiny (≤ 4 in Spark-TTS) so `l / 2` fits exactly
        // in an f32 mantissa.
        #[allow(
            clippy::cast_precision_loss,
            reason = "levels are bounded by FSQ codebook width (Spark-TTS: 4); \
                      f32 represents [0, 2^23) exactly."
        )]
        let vec: Vec<f32> = self.levels.iter().map(|&l| (l / 2) as f32).collect();
        Tensor::from_vec(vec, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)
    }

    /// `codes_to_indices`: `(B, T, num_cb, cb_dim) → (B, T, num_cb)`.
    ///
    /// Inverts the `quantize`'s `/ half_width` normalization, then
    /// projects the per-axis integer labels onto the mixed-radix basis.
    fn codes_to_indices(&self, zhat: &Tensor) -> Result<Tensor> {
        let dtype = zhat.dtype();
        let dev = zhat.device();
        let half_width = self.half_width_tensor(dev, dtype)?;
        // _scale_and_shift: (zhat * half_width) + half_width  → integer
        // labels in [0, level).
        let scaled = zhat
            .broadcast_mul(&half_width)?
            .broadcast_add(&half_width)?;

        // basis is `cumprod([1] + levels[:-1])`. Spark-TTS max value
        // = 4^5 = 1024 — exactly representable in f32.
        #[allow(
            clippy::cast_precision_loss,
            reason = "basis bounded by prod(levels[:-1]) (Spark-TTS: 1024); \
                      f32 represents [0, 2^23) exactly."
        )]
        let basis_vec: Vec<f32> = self.basis.iter().map(|&b| b as f32).collect();
        let basis_t =
            Tensor::from_vec(basis_vec, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)?;

        // sum_{d} scaled[..., d] * basis[d]
        let weighted = scaled.broadcast_mul(&basis_t)?;
        let indices_f = weighted.sum(D::Minus1)?; // (B, T, num_cb)
        // Cast to integer. Round to nearest first to absorb any
        // accumulated floating-point error from the * half_width step.
        let rounded = round_half_to_even(&indices_f)?;
        rounded.to_dtype(DType::U32)
    }

    /// `indices_to_codes` (the per-FSQ piece): `(B, T, num_cb) →
    /// (B, T, num_cb, cb_dim)` of grid-centered floats in `[-1, 1]`.
    fn indices_to_codes(&self, indices: &Tensor) -> Result<Tensor> {
        let dtype = DType::F32;
        let dev = indices.device();
        let (b, t, ncb) = indices.dims3()?;
        if ncb != self.num_codebooks {
            candle_core::bail!(
                "Fsq::indices_to_codes: expected num_codebooks {}, got {ncb}",
                self.num_codebooks
            );
        }

        // (B, T, num_cb) → (B, T, num_cb, 1) so we can broadcast against
        // the per-axis basis / level vectors.
        let idx_f = indices.to_dtype(dtype)?.unsqueeze(D::Minus1)?;

        // Both basis and levels are tiny u32 values (≤ 1024) — exact
        // in f32.
        #[allow(
            clippy::cast_precision_loss,
            reason = "basis ≤ prod(levels[:-1]) (Spark-TTS: 1024), levels ≤ \
                      per-axis FSQ width (Spark-TTS: 4); both exact in f32."
        )]
        let basis_vec: Vec<f32> = self.basis.iter().map(|&b| b as f32).collect();
        let basis_t =
            Tensor::from_vec(basis_vec, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)?;
        #[allow(
            clippy::cast_precision_loss,
            reason = "levels bounded by per-axis FSQ width (Spark-TTS: 4); \
                      exact in f32."
        )]
        let levels_vec: Vec<f32> = self.levels.iter().map(|&l| l as f32).collect();
        let levels_t =
            Tensor::from_vec(levels_vec, (1, 1, 1, self.codebook_dim), dev)?.to_dtype(dtype)?;

        // codes_non_centered = (indices // basis) % levels
        // = floor(idx / basis) - floor(floor(idx / basis) / levels) * levels
        let div = idx_f.broadcast_div(&basis_t)?.floor()?;
        let inner = div.broadcast_div(&levels_t)?.floor()?;
        let level_indices = div.broadcast_sub(&inner.broadcast_mul(&levels_t)?)?;

        // _scale_and_shift_inverse: (level_indices - half_width) / half_width.
        let half_width = self.half_width_tensor(dev, dtype)?;
        let codes = level_indices
            .broadcast_sub(&half_width)?
            .broadcast_div(&half_width)?;

        // Cast result back to f32 (already is) and reshape — return
        // (B, T, num_cb, cb_dim).
        let _ = (b, t); // silence unused warning under some feature flags
        Ok(codes)
    }
}

/// `cumprod([1] + levels[:-1])`.
fn compute_basis(levels: &[u32]) -> Vec<u32> {
    let mut basis = Vec::with_capacity(levels.len());
    let mut acc: u32 = 1;
    for (i, &_l) in levels.iter().enumerate() {
        basis.push(acc);
        if i < levels.len() - 1 {
            acc = acc.checked_mul(levels[i]).expect("FSQ basis overflow");
        }
    }
    basis
}

/// Round-half-to-even (banker's rounding) over a tensor, matching
/// `torch.round`'s IEEE-754 "round to nearest, ties to even" rule.
///
/// Rust's `f32::round` is round-half-away-from-zero, which would land
/// `±0.5` on a different codebook entry than the Python reference. We
/// implement the tie-breaking rule explicitly:
///
/// ```text
///   floor_x = floor(x)
///   frac    = x - floor_x        (in [0, 1))
///   round_x = floor_x                       if frac < 0.5
///   round_x = floor_x + 1                   if frac > 0.5
///   round_x = floor_x + (floor_x mod 2)     if frac == 0.5
/// ```
///
/// The `floor_x mod 2` piece picks the even neighbour: `floor(0.5) = 0`
/// (even, no bump) → `0`; `floor(1.5) = 1` (odd, bump) → `2`;
/// `floor(2.5) = 2` (even, no bump) → `2`; etc.
fn round_half_to_even(x: &Tensor) -> Result<Tensor> {
    let floor_x = x.floor()?;
    let frac = x.sub(&floor_x)?;

    // half_mask: 1.0 where |frac - 0.5| <= tol, else 0.0. We use a
    // tolerance because `x` may not be exactly representable.
    let half = (frac.affine(1.0, -0.5)?).abs()?;
    let tol = 1e-7_f64;
    let half_mask = half.le(tol)?.to_dtype(floor_x.dtype())?;

    // gt_half_mask: 1.0 where frac > 0.5 (strict, excluding the tie).
    let gt_half = frac.gt(0.5_f64 + tol)?.to_dtype(floor_x.dtype())?;

    // even_bump: at ties, add `floor_x % 2` to break to even. We
    // compute `floor_x - 2 * floor(floor_x / 2)`.
    let two = Tensor::full(2.0_f32, floor_x.shape(), x.device())?.to_dtype(floor_x.dtype())?;
    let half_of_floor = floor_x.broadcast_div(&two)?.floor()?;
    let twice = half_of_floor.broadcast_mul(&two)?;
    let parity = floor_x.sub(&twice)?; // 0.0 if even, 1.0 if odd.

    // tie_bump = half_mask * parity
    let tie_bump = half_mask.mul(&parity)?;

    floor_x.add(&gt_half)?.add(&tie_bump)
}

// ---------------------------------------------------------------------------
// ResidualFSQ (Spark-TTS speaker-encoder wrapper)
// ---------------------------------------------------------------------------

/// Residual finite scalar quantizer wrapper. For Spark-TTS this only
/// ever wraps **one** [`Fsq`] layer (`num_quantizers = 1`), so the
/// residual-loop machinery degenerates to a single `quantized + residual
/// = x` step. We still implement the iteration so the surface matches
/// upstream's `ResidualFSQ.forward` and so a checkpoint with multiple
/// residual layers would just work.
///
/// # Channels-first I/O
///
/// Spark-TTS's only consumer is
/// `sparktts/modules/speaker/speaker_encoder.py::SpeakerEncoder`, which
/// instantiates this with `is_channel_first=True` and feeds in
/// `(B, latent_dim, T)`. We hard-code that convention rather than
/// taking an extra `channel_first: bool` knob, because we have no other
/// consumer.
///
/// # Projection layout
///
/// Upstream passes `dim=codebook_dim` into each inner FSQ, so every
/// inner FSQ has `has_projections == False`. The `dim ↔ codebook_dim`
/// projection (Linear, with bias) instead lives on the *outer*
/// `ResidualFSQ` as `project_in` / `project_out`. For Spark-TTS:
/// `project_in: Linear(128 → 6)`, `project_out: Linear(6 → 128)`.
#[derive(Debug, Clone)]
pub(super) struct ResidualFsq {
    project_in: Option<Linear>,
    project_out: Option<Linear>,
    layers: Vec<Fsq>,
    /// Per-layer scale factors: `scales[i] = (levels - 1) ** -i`. For
    /// `i=0` the scale is all-ones, which is the only case `BiCodec` hits.
    /// Stored as `(num_quantizers, codebook_dim)` `Vec<Vec<f32>>` so we
    /// can build per-step broadcast tensors without keeping a `Tensor`
    /// around (and incurring its `Send`-bound friction).
    scales: Vec<Vec<f32>>,
    dim: usize,
    codebook_dim: usize,
}

impl ResidualFsq {
    /// Load `num_quantizers` residual FSQ layers under
    /// `vb / "layers" / {i}`. The `dim ↔ codebook_dim` projections live
    /// under `vb / "project_in"` / `vb / "project_out"` and are only
    /// loaded when `dim != codebook_dim`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from any of the child loaders.
    /// Returns an error if `levels` is empty or `num_quantizers` is 0.
    pub(super) fn load(
        vb: VarBuilder,
        dim: usize,
        levels: &[u32],
        num_quantizers: usize,
    ) -> Result<Self> {
        if levels.is_empty() {
            candle_core::bail!("ResidualFsq::load: `levels` must be non-empty");
        }
        if num_quantizers == 0 {
            candle_core::bail!("ResidualFsq::load: `num_quantizers` must be >= 1");
        }
        let codebook_dim = levels.len();
        let requires_projection = codebook_dim != dim;
        let project_in = if requires_projection {
            Some(linear(dim, codebook_dim, vb.pp("project_in"))?)
        } else {
            None
        };
        let project_out = if requires_projection {
            Some(linear(codebook_dim, dim, vb.pp("project_out"))?)
        } else {
            None
        };

        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(num_quantizers);
        let mut scales = Vec::with_capacity(num_quantizers);
        for i in 0..num_quantizers {
            // Inner FSQ takes dim==codebook_dim so its has_projections is False.
            let fsq = Fsq::load(
                layers_vb.pp(i),
                levels,
                codebook_dim,
                /* num_codebooks */ 1,
            )?;
            layers.push(fsq);

            // scales[i] = (levels - 1) ** -i. For i=0 this is all-ones
            // by convention (0 ** 0 = 1 in Python's pow).
            #[allow(clippy::cast_precision_loss)]
            let i_f = i as f32;
            #[allow(
                clippy::cast_precision_loss,
                reason = "levels - 1 ≤ 3 for Spark-TTS; exact in f32."
            )]
            let scale: Vec<f32> = levels
                .iter()
                .map(|&l| {
                    if i == 0 {
                        1.0
                    } else {
                        f32::from(u16::try_from(l - 1).unwrap_or(u16::MAX)).powf(-i_f)
                    }
                })
                .collect();
            scales.push(scale);
        }

        Ok(Self {
            project_in,
            project_out,
            layers,
            scales,
            dim,
            codebook_dim,
        })
    }

    /// Outer feature dim (`Linear` in/out side).
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// Inner FSQ codebook dim (= `len(levels)`).
    pub(super) fn codebook_dim(&self) -> usize {
        self.codebook_dim
    }

    /// Number of residual FSQ layers. Spark-TTS: 1.
    pub(super) fn num_quantizers(&self) -> usize {
        self.layers.len()
    }

    /// Codebook size of a single residual layer (`prod(levels)`).
    pub(super) fn codebook_size(&self) -> u32 {
        self.layers[0].codebook_size()
    }

    /// Channels-first encode: `z: (B, dim, T) → indices: (B, T,
    /// num_quantizers)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the projection, residual
    /// loop, or any inner FSQ tokenize.
    pub(super) fn tokenize(&self, z: &Tensor) -> Result<Tensor> {
        let (_quantized, indices) = self.forward(z)?;
        Ok(indices)
    }

    /// Channels-first decode: `indices: (B, T, num_quantizers) → z_q:
    /// (B, dim, T)`. Equivalent to upstream's
    /// `get_output_from_indices(indices).transpose(1, 2)` in the
    /// `SpeakerEncoder` consumer.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the per-layer index ->
    /// code lookup or the outer `project_out`.
    pub(super) fn detokenize(&self, indices: &Tensor) -> Result<Tensor> {
        self.get_output_from_indices(indices)
    }

    /// Upstream's `get_output_from_indices`: per-layer
    /// `indices_to_codes`, scale by the per-layer `scales`, sum across
    /// layers, then `project_out`. Returns `(B, dim, T)`.
    ///
    /// # Errors
    ///
    /// Propagates [`candle_core::Error`] from the per-layer index ->
    /// code lookup or the outer `project_out`.
    pub(super) fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let (_b, _t, ncb) = indices.dims3()?;
        if ncb != self.layers.len() {
            candle_core::bail!(
                "ResidualFsq::get_output_from_indices: expected {} quantizers, got {ncb}",
                self.layers.len()
            );
        }

        let dev = indices.device();
        // Sum codes across layers, in (B, T, codebook_dim) channels-last
        // space, then optionally project out and transpose.
        let mut sum: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            // Slice (B, T, 1) for this layer.
            let layer_idx = indices.i((.., .., i..i + 1))?;
            // (B, T, 1, codebook_dim) grid-centered floats.
            let codes = layer.indices_to_codes(&layer_idx)?;
            // Collapse the num_codebooks=1 axis: (B, T, codebook_dim).
            let codes = codes.squeeze(2)?;

            // Scale by self.scales[i] (per-axis along the last dim).
            let scale_t = Tensor::from_vec(self.scales[i].clone(), (1, 1, self.codebook_dim), dev)?
                .to_dtype(codes.dtype())?;
            let scaled = codes.broadcast_mul(&scale_t)?;
            sum = Some(match sum {
                Some(s) => (s + scaled)?,
                None => scaled,
            });
        }
        let codes_sum = sum.expect("non-empty layers — checked in `load`");

        // (B, T, codebook_dim) → (B, T, dim) via project_out.
        let codes_sum = if let Some(po) = &self.project_out {
            po.forward(&codes_sum)?
        } else {
            codes_sum
        };

        // Channels-last → channels-first.
        codes_sum.transpose(1, 2)?.contiguous()
    }

    /// Full forward returning `(quantized: (B, dim, T), indices: (B, T,
    /// num_quantizers))`. This is the inference equivalent of
    /// upstream's `ResidualFSQ.forward` with `is_channel_first=True`
    /// and `quantize_dropout=False`.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, d, _t) = x.dims3()?;
        if d != self.dim {
            candle_core::bail!(
                "ResidualFsq::forward: expected feature dim {}, got {d}",
                self.dim
            );
        }

        // (B, dim, T) → (B, T, dim) → maybe project_in → (B, T,
        // codebook_dim).
        let x_btd = x.transpose(1, 2)?.contiguous()?;
        let x_btd = if let Some(pi) = &self.project_in {
            pi.forward(&x_btd)?
        } else {
            x_btd
        };

        // residual / quantized_out / all_indices loop, channels-first
        // back-to-back (we treat the (B, T, codebook_dim) tensor as
        // pseudo-channels-first for the inner FSQ by transposing
        // before each layer.tokenize / indices_to_codes call).
        let mut residual = x_btd; // (B, T, cb_dim)
        let mut quantized_sum: Option<Tensor> = None;
        let mut all_indices: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        let dev = residual.device().clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let scale_t =
                Tensor::from_vec(self.scales[i].clone(), (1, 1, self.codebook_dim), &dev)?
                    .to_dtype(residual.dtype())?;

            // residual / scale, then to channels-first for the inner FSQ.
            let scaled_residual = residual.broadcast_div(&scale_t)?;
            let scaled_cf = scaled_residual.transpose(1, 2)?.contiguous()?; // (B, cb_dim, T)
            let indices = layer.tokenize(&scaled_cf)?; // (B, T, 1)
            let codes = layer.indices_to_codes(&indices)?; // (B, T, 1, cb_dim)
            let codes = codes.squeeze(2)?; // (B, T, cb_dim)

            // quantized = codes * scale
            let quantized = codes.broadcast_mul(&scale_t)?;

            residual = residual.sub(&quantized)?;
            quantized_sum = Some(match quantized_sum {
                Some(q) => (q + &quantized)?,
                None => quantized,
            });
            all_indices.push(indices);
        }

        let quantized_out = quantized_sum.expect("non-empty layers — checked in `load`");
        // project_out + back to channels-first.
        let quantized_out = if let Some(po) = &self.project_out {
            po.forward(&quantized_out)?
        } else {
            quantized_out
        };
        let quantized_cf = quantized_out.transpose(1, 2)?.contiguous()?; // (B, dim, T)

        // Stack indices along last dim: each is (B, T, 1) so stacking
        // along axis -1 (and concatenating since they're already 3D)
        // gives (B, T, num_quantizers).
        let indices_stacked = if all_indices.len() == 1 {
            all_indices.into_iter().next().expect("len==1")
        } else {
            Tensor::cat(&all_indices, D::Minus1)?
        };

        Ok((quantized_cf, indices_stacked))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "tests construct small deterministic vectors via `usize as f32` \
              indices; ranges are tiny (< 2^23) so precision is exact."
)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn vb_from(map: HashMap<String, Tensor>, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_tensors(map, DType::F32, dev)
    }

    /// Build a `WeightNormConv1d`-shaped state-dict entry set under
    /// `prefix` from a dense kernel `w: (out, in, k)` and a `bias`. We
    /// store `weight_g = 1.0` and `weight_v = w` so that the folded
    /// kernel reduces to `w / ||w||` — which we account for in the test
    /// expectations rather than trying to invert.
    fn insert_wnconv1d_identity(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_c: usize,
        in_c: usize,
        k: usize,
        dev: &Device,
    ) {
        // weight_g = ones (out, 1, 1)
        map.insert(
            format!("{prefix}.weight_g"),
            Tensor::ones((out_c, 1, 1), DType::F32, dev).expect("ones"),
        );
        // weight_v: a permutation/identity-like kernel for kernel_size=1.
        let mut v = vec![0.0_f32; out_c * in_c * k];
        for o in 0..out_c {
            // Single-tap (k=1): diagonal so that v[o, o, 0] = 1, others
            // = 0 → folded weight = identity matrix / ||identity row|| =
            // identity (since ||(0,…,1,…,0)|| = 1).
            if o < in_c {
                v[o * in_c * k + o * k] = 1.0;
            }
        }
        map.insert(
            format!("{prefix}.weight_v"),
            Tensor::from_vec(v, (out_c, in_c, k), dev).expect("from_vec"),
        );
        map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(out_c, DType::F32, dev).expect("zeros"),
        );
    }

    #[test]
    fn factorized_vq_tokenize_returns_argmax_over_l2_normalized() -> Result<()> {
        let dev = Device::Cpu;
        let input_dim = 4;
        let codebook_size = 4;
        let codebook_dim = 4;

        // dim == codebook_dim → Identity projections, no `in_project` /
        // `out_project` weights needed in the state dict.
        let mut map = HashMap::new();
        // Codebook rows: one-hot along the first 4 axes.
        let mut cb = vec![0.0_f32; codebook_size * codebook_dim];
        for i in 0..codebook_size {
            cb[i * codebook_dim + i] = 1.0;
        }
        map.insert(
            "codebook.weight".to_owned(),
            Tensor::from_vec(cb, (codebook_size, codebook_dim), &dev)?,
        );
        let vb = vb_from(map, &dev);

        let fvq = FactorizedVectorQuantize::load(vb, input_dim, codebook_size, codebook_dim)?;

        // Input z: (1, 4, 1), one slot whose dominant dim is index 2.
        let z = Tensor::from_vec(vec![0.1_f32, 0.0, 0.95, 0.05], (1, 4, 1), &dev)?;
        let indices = fvq.tokenize(&z)?;
        assert_eq!(indices.dims(), &[1, 1]);
        let idx = indices.flatten_all()?.to_vec1::<u32>()?[0];
        assert_eq!(idx, 2, "expected nearest codebook row index 2, got {idx}");
        Ok(())
    }

    #[test]
    fn factorized_vq_detokenize_round_trip_keeps_shape_and_nonzero() -> Result<()> {
        let dev = Device::Cpu;
        let input_dim = 4;
        let codebook_size = 4;
        let codebook_dim = 4;
        let mut map = HashMap::new();
        let mut cb = vec![0.0_f32; codebook_size * codebook_dim];
        for i in 0..codebook_size {
            cb[i * codebook_dim + i] = (i as f32) + 1.0; // non-unit so detokenize ≠ 0.
        }
        map.insert(
            "codebook.weight".to_owned(),
            Tensor::from_vec(cb, (codebook_size, codebook_dim), &dev)?,
        );
        let vb = vb_from(map, &dev);
        let fvq = FactorizedVectorQuantize::load(vb, input_dim, codebook_size, codebook_dim)?;

        let z = Tensor::from_vec(
            vec![0.1_f32, 0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2],
            (1, 4, 2),
            &dev,
        )?;
        let indices = fvq.tokenize(&z)?;
        let z_q = fvq.detokenize(&indices)?;
        assert_eq!(z_q.dims(), &[1, input_dim, 2]);
        let max_abs = z_q.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max_abs > 0.0, "detokenize produced all-zero output");
        Ok(())
    }

    #[test]
    fn factorized_vq_with_projections_loads_and_runs() -> Result<()> {
        // Exercise the WNConv1d projection arms (matching the BiCodec
        // checkpoint shape, just smaller).
        let dev = Device::Cpu;
        let input_dim = 6;
        let codebook_size = 4;
        let codebook_dim = 4;

        let mut map = HashMap::new();
        insert_wnconv1d_identity(&mut map, "in_project", codebook_dim, input_dim, 1, &dev);
        insert_wnconv1d_identity(&mut map, "out_project", input_dim, codebook_dim, 1, &dev);
        // Codebook = one-hot for index 0.
        let mut cb = vec![0.0_f32; codebook_size * codebook_dim];
        for i in 0..codebook_size {
            cb[i * codebook_dim + i] = 1.0;
        }
        map.insert(
            "codebook.weight".to_owned(),
            Tensor::from_vec(cb, (codebook_size, codebook_dim), &dev)?,
        );
        let vb = vb_from(map, &dev);
        let fvq = FactorizedVectorQuantize::load(vb, input_dim, codebook_size, codebook_dim)?;
        assert_eq!(fvq.codebook_size(), codebook_size);
        assert_eq!(fvq.codebook_dim(), codebook_dim);

        let z = Tensor::randn(0.0_f32, 1.0, (2, input_dim, 3), &dev)?;
        let indices = fvq.tokenize(&z)?;
        assert_eq!(indices.dims(), &[2, 3]);
        let z_q = fvq.detokenize(&indices)?;
        assert_eq!(z_q.dims(), &[2, input_dim, 3]);
        Ok(())
    }

    #[test]
    fn fsq_basis_matches_spark_config() {
        let levels = vec![4_u32, 4, 4, 4, 4, 4];
        let basis = compute_basis(&levels);
        assert_eq!(basis, vec![1, 4, 16, 64, 256, 1024]);
        let codebook_size: u32 = levels.iter().product();
        assert_eq!(codebook_size, 4096);
    }

    #[test]
    fn fsq_quantize_then_codes_to_indices_in_range() -> Result<()> {
        let dev = Device::Cpu;
        let levels = vec![4_u32; 6];
        let codebook_dim = levels.len();
        // dim == effective_codebook_dim → no projections, empty vb.
        let map: HashMap<String, Tensor> = HashMap::new();
        let vb = vb_from(map, &dev);
        let fsq = Fsq::load(vb, &levels, codebook_dim, /* num_codebooks */ 1)?;

        let z = Tensor::randn(0.0_f32, 1.0, (2, codebook_dim, 5), &dev)?;
        let indices = fsq.tokenize(&z)?;
        assert_eq!(indices.dims(), &[2, 5, 1]);
        let max = indices.flatten_all()?.max(0)?.to_scalar::<u32>()?;
        let min = indices.flatten_all()?.min(0)?.to_scalar::<u32>()?;
        assert!(
            max < fsq.codebook_size(),
            "max index {max} >= codebook_size {}",
            fsq.codebook_size()
        );
        assert!(
            min < fsq.codebook_size(),
            "min index {min} >= codebook_size {}",
            fsq.codebook_size()
        );
        Ok(())
    }

    #[test]
    fn fsq_indices_to_codes_is_inverse_of_codes_to_indices() -> Result<()> {
        // The identity `codes_to_indices(indices_to_codes(i)) == i` holds
        // for ALL valid indices in `[0, codebook_size)`. Re-running the
        // full `tokenize` (which re-applies `bound`/`tanh`) would NOT
        // round-trip — `tanh` is not a left-inverse of itself on grid
        // points. The right test is to exercise the post-`quantize`
        // pieces in isolation.
        let dev = Device::Cpu;
        let levels = vec![4_u32; 6];
        let codebook_dim = levels.len();
        let map: HashMap<String, Tensor> = HashMap::new();
        let vb = vb_from(map, &dev);
        let fsq = Fsq::load(vb, &levels, codebook_dim, /* num_codebooks */ 1)?;

        // All 4096 valid indices.
        let cb_size = fsq.codebook_size();
        let raw: Vec<u32> = (0..cb_size).collect();
        let n = raw.len();
        // Shape (B=1, T=n, num_cb=1).
        let indices = Tensor::from_vec(raw, (1, n, 1), &dev)?;
        let codes = fsq.indices_to_codes(&indices)?; // (1, n, 1, cb_dim)
        let indices_round = fsq.codes_to_indices(&codes)?; // (1, n, 1)
        let i1_v = indices.flatten_all()?.to_vec1::<u32>()?;
        let i2_v = indices_round.flatten_all()?.to_vec1::<u32>()?;
        assert_eq!(
            i1_v, i2_v,
            "FSQ codes_to_indices ↔ indices_to_codes round trip diverged"
        );
        Ok(())
    }

    #[test]
    fn residual_fsq_load_with_num_quantizers_one_acts_as_fsq() -> Result<()> {
        // num_quantizers=1, dim == codebook_dim → no outer projections,
        // inner FSQ also has none. Should match a bare FSQ exactly.
        let dev = Device::Cpu;
        let levels = vec![4_u32; 6];
        let codebook_dim = levels.len();

        let map: HashMap<String, Tensor> = HashMap::new();
        let vb = vb_from(map, &dev);
        let rfsq = ResidualFsq::load(vb, codebook_dim, &levels, /* num_quantizers */ 1)?;

        let map_b: HashMap<String, Tensor> = HashMap::new();
        let vb_b = vb_from(map_b, &dev);
        let fsq = Fsq::load(vb_b, &levels, codebook_dim, /* num_codebooks */ 1)?;

        let z = Tensor::randn(0.0_f32, 1.0, (1, codebook_dim, 3), &dev)?;
        let r_idx = rfsq.tokenize(&z)?; // (1, 3, 1)
        let f_idx = fsq.tokenize(&z)?; // (1, 3, 1)
        assert_eq!(r_idx.dims(), f_idx.dims());
        assert_eq!(
            r_idx.flatten_all()?.to_vec1::<u32>()?,
            f_idx.flatten_all()?.to_vec1::<u32>()?
        );
        Ok(())
    }

    #[test]
    fn residual_fsq_get_output_from_indices_matches_detokenize() -> Result<()> {
        let dev = Device::Cpu;
        let levels = vec![4_u32; 6];
        let codebook_dim = levels.len();
        let map: HashMap<String, Tensor> = HashMap::new();
        let vb = vb_from(map, &dev);
        let rfsq = ResidualFsq::load(vb, codebook_dim, &levels, 1)?;

        // Random valid indices in [0, codebook_size).
        let cb_size = rfsq.codebook_size();
        let n = 4;
        let raw: Vec<u32> = (0..n).map(|i| (i * 37) % cb_size).collect();
        let indices = Tensor::from_vec(raw, (1, n as usize, 1), &dev)?;
        let a = rfsq.detokenize(&indices)?;
        let b = rfsq.get_output_from_indices(&indices)?;
        assert_eq!(a.dims(), b.dims());
        let diff = (&a - &b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-6,
            "detokenize / get_output_from_indices diverge: {diff}"
        );
        Ok(())
    }

    #[test]
    fn residual_fsq_with_projection_dim_128_to_6_loads() -> Result<()> {
        // Spark-TTS shape: dim=128, levels=[4]*6, num_quantizers=1.
        let dev = Device::Cpu;
        let levels = vec![4_u32; 6];
        let dim = 128_usize;
        let codebook_dim = levels.len();

        // Build random Linear weights for project_in / project_out.
        let mut map = HashMap::new();
        map.insert(
            "project_in.weight".to_owned(),
            Tensor::randn(0.0_f32, 0.05, (codebook_dim, dim), &dev)?,
        );
        map.insert(
            "project_in.bias".to_owned(),
            Tensor::zeros(codebook_dim, DType::F32, &dev)?,
        );
        map.insert(
            "project_out.weight".to_owned(),
            Tensor::randn(0.0_f32, 0.05, (dim, codebook_dim), &dev)?,
        );
        map.insert(
            "project_out.bias".to_owned(),
            Tensor::zeros(dim, DType::F32, &dev)?,
        );
        let vb = vb_from(map, &dev);
        let rfsq = ResidualFsq::load(vb, dim, &levels, 1)?;
        assert_eq!(rfsq.dim(), 128);
        assert_eq!(rfsq.codebook_dim(), 6);
        assert_eq!(rfsq.num_quantizers(), 1);
        assert_eq!(rfsq.codebook_size(), 4096);

        // Round-trip a random input through tokenize → detokenize.
        let z = Tensor::randn(0.0_f32, 1.0, (2, dim, 32), &dev)?;
        let idx = rfsq.tokenize(&z)?;
        assert_eq!(idx.dims(), &[2, 32, 1]);
        let z_q = rfsq.detokenize(&idx)?;
        assert_eq!(z_q.dims(), &[2, dim, 32]);
        Ok(())
    }

    #[test]
    fn round_half_to_even_handles_ties() -> Result<()> {
        let dev = Device::Cpu;
        // x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        // PyTorch round (banker's): [-2, -2, 0, 0, 2, 2, 4]
        let xs = vec![-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
        let t = Tensor::from_vec(xs, 7, &dev)?;
        let r = round_half_to_even(&t)?;
        let got = r.to_vec1::<f32>()?;
        let expected = vec![-2.0_f32, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0];
        assert_eq!(got, expected, "banker's rounding mismatch");
        Ok(())
    }
}
