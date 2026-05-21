//! Per-architecture trainable model wrappers.
//!
//! Each submodule mirrors `candle_transformers::models::<arch>`'s forward
//! pass but substitutes [`crate::lora::LoraLinear`] at every site listed
//! in [`crate::config::LoraConfig::target_modules`] (q/k/v/o by default,
//! optionally gate/up/down for MLP LoRA).
//!
//! Wave 2A populates `qwen2`, 2B populates `llama`, 2C populates `mistral`.
//! Wave 8a adds full fine-tune support via [`TrainMode::FullFineTune`].

pub mod llama;
pub mod mistral;
pub mod qwen2;

use candle_core::{Result, Var};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, VarMap, embedding, rms_norm};

use crate::config::QloraQuantDtype;

/// Selects whether a per-architecture wrapper trains LoRA adapters on top
/// of a frozen base, or every base parameter directly.
///
/// The forward pass is identical between modes — only the placement of
/// trainable [`Var`]s differs. In [`TrainMode::LoraOnly`] the base weights
/// live in a read-only [`VarBuilder`] (typically backed by an mmap'd
/// safetensors file) and only the LoRA A/B matrices land in the trainer's
/// [`VarMap`]. In [`TrainMode::FullFineTune`] every base weight is copied
/// into the trainer's [`VarMap`] as a fresh [`Var`] so it picks up
/// gradients during `loss.backward()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainMode {
    /// Base weights frozen (read-only `VarBuilder`); only LoRA adapters
    /// at [`crate::config::LoraConfig::target_modules`] receive gradients.
    LoraOnly,
    /// All base weights copied into the trainer's [`VarMap`] as trainable
    /// [`Var`]s. The arch wrapper ignores [`crate::config::LoraConfig::target_modules`]
    /// in this mode — no LoRA layers are constructed.
    FullFineTune,
    /// Quantized base + trainable LoRA adapters (QLoRA). At every target
    /// linear the base weight is quantized to the GGUF integer format
    /// carried in `quant` and wrapped with a [`crate::qlora::QLoraLinear`].
    /// Non-target linears stay as plain frozen [`candle_nn::Linear`]s (no
    /// quantization saving on them — they're typically a tiny fraction
    /// of the total parameter budget). Only the LoRA `A`/`B` matrices land
    /// in the trainer's [`VarMap`] and receive gradients.
    Qlora {
        /// GGUF integer format the base weights are packed into.
        quant: QloraQuantDtype,
    },
}

/// Copy a weight tensor read from `src_vb` (e.g., mmap'd safetensors) into
/// `train_varmap` as a fresh [`Var`] under `absolute_key`, then construct
/// a trainable [`Linear`] referencing those [`Var`]-backed tensors.
///
/// `with_bias` controls whether a `.bias` companion weight is also copied
/// — matching `candle_nn::linear` (bias) vs `linear_no_bias` semantics.
///
/// The returned `Linear` holds tensors that share storage with the
/// `Var`s in `train_varmap`, so `VarMap::all_vars()` (and thus the AdamW
/// optimizer parameter set) will include them automatically.
///
/// This helper is the FFT analogue of `candle_nn::linear` / `linear_no_bias`
/// and is shared between `qwen2`, `llama`, and `mistral` wrappers.
///
/// # Errors
///
/// Propagates any tensor read error from `src_vb` or `Var::from_tensor`
/// failure (e.g., on a non-writable device).
fn linear_into_varmap(
    in_dim: usize,
    out_dim: usize,
    src_vb: &VarBuilder,
    train_varmap: &VarMap,
    absolute_key: &str,
    with_bias: bool,
) -> Result<Linear> {
    // Why: read the base weight from whatever the caller pointed src_vb at
    // (typically MmapedSafetensors), then promote it to a Var in the
    // trainer's VarMap so AdamW sees it via VarMap::all_vars().
    let weight_src = src_vb.get((out_dim, in_dim), "weight")?;
    let weight_var = Var::from_tensor(&weight_src)?;
    // The Linear must reference the *same* Var's tensor — `as_tensor()`
    // returns a shallow handle whose autograd id matches the Var, so
    // gradients computed at this Linear flow back to the Var sitting in
    // the trainer's VarMap.
    let weight_tensor = weight_var.as_tensor().clone();

    let bias_var = if with_bias {
        let b_src = src_vb.get(out_dim, "bias")?;
        Some(Var::from_tensor(&b_src)?)
    } else {
        None
    };
    let bias_tensor = bias_var.as_ref().map(|v| v.as_tensor().clone());

    // Why: install Vars into the trainer's VarMap under the absolute key
    // the caller wants (so checkpoint save/load matches the source
    // safetensors naming convention). Direct insert via data().lock() is
    // the only API path — VarMap::get can't accept a pre-made Var.
    {
        let mut guard = train_varmap
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        guard.insert(format!("{absolute_key}.weight"), weight_var);
        if let Some(b_var) = bias_var {
            guard.insert(format!("{absolute_key}.bias"), b_var);
        }
    }

    Ok(Linear::new(weight_tensor, bias_tensor))
}

/// FFT analogue of `candle_nn::embedding` — reads the embedding table
/// from `src_vb` and inserts it as a trainable [`Var`] under
/// `{absolute_key}.weight` in `train_varmap`.
///
/// # Errors
///
/// Propagates any tensor read error from `src_vb` or `Var::from_tensor`
/// failure.
fn embedding_into_varmap(
    in_size: usize,
    out_size: usize,
    src_vb: &VarBuilder,
    train_varmap: &VarMap,
    absolute_key: &str,
) -> Result<Embedding> {
    let w_src = src_vb.get((in_size, out_size), "weight")?;
    let w_var = Var::from_tensor(&w_src)?;
    let w_tensor = w_var.as_tensor().clone();
    {
        let mut guard = train_varmap
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        guard.insert(format!("{absolute_key}.weight"), w_var);
    }
    Ok(Embedding::new(w_tensor, out_size))
}

/// FFT analogue of `candle_nn::rms_norm` — reads the affine weight
/// from `src_vb` and inserts it as a trainable [`Var`] under
/// `{absolute_key}.weight` in `train_varmap`.
///
/// `candle_nn::rms_norm` constructs a `RmsNorm` from a `LayerNorm` whose
/// `affine` flag is `false`, so it has only `weight` (no `bias`). The
/// public `RmsNorm::rms_norm(weight, eps)` constructor takes a
/// pre-loaded tensor — exactly what we need to wire in a `Var`-backed
/// trainable weight.
///
/// # Errors
///
/// Propagates any tensor read error from `src_vb` or `Var::from_tensor`
/// failure.
fn rms_norm_into_varmap(
    size: usize,
    eps: f64,
    src_vb: &VarBuilder,
    train_varmap: &VarMap,
    absolute_key: &str,
) -> Result<RmsNorm> {
    let w_src = src_vb.get(size, "weight")?;
    let w_var = Var::from_tensor(&w_src)?;
    let w_tensor = w_var.as_tensor().clone();
    {
        let mut guard = train_varmap
            .data()
            .lock()
            .expect("train varmap mutex poisoned by another thread");
        guard.insert(format!("{absolute_key}.weight"), w_var);
    }
    Ok(RmsNorm::new(w_tensor, eps))
}

/// Mode-aware wrapper around `candle_nn::embedding` that either reads
/// frozen from `src_vb` or routes through [`embedding_into_varmap`].
///
/// # Errors
///
/// Propagates errors from the underlying load path.
pub(crate) fn build_embedding(
    in_size: usize,
    out_size: usize,
    src_vb: VarBuilder,
    loader: &BaseLoader<'_>,
    abs_path: &str,
) -> Result<Embedding> {
    match loader.mode {
        // Why: QLoRA only quantizes the per-layer attention/MLP linears
        // listed in target_modules. Embeddings, lm_head, and layer norms
        // stay frozen-dense — same as LoraOnly mode — because they're a
        // tiny fraction of total params and quantizing them hurts
        // accuracy disproportionately.
        TrainMode::LoraOnly | TrainMode::Qlora { .. } => embedding(in_size, out_size, src_vb),
        TrainMode::FullFineTune => embedding_into_varmap(
            in_size,
            out_size,
            &src_vb,
            loader
                .train_varmap
                .expect("FullFineTune requires train_varmap"),
            abs_path,
        ),
    }
}

/// Mode-aware wrapper around `candle_nn::rms_norm` that either reads
/// frozen from `src_vb` or routes through [`rms_norm_into_varmap`].
///
/// # Errors
///
/// Propagates errors from the underlying load path.
pub(crate) fn build_rms_norm(
    size: usize,
    eps: f64,
    src_vb: VarBuilder,
    loader: &BaseLoader<'_>,
    abs_path: &str,
) -> Result<RmsNorm> {
    match loader.mode {
        // Why: same rationale as build_embedding — QLoRA leaves the per-
        // layer norms in dense form.
        TrainMode::LoraOnly | TrainMode::Qlora { .. } => rms_norm(size, eps, src_vb),
        TrainMode::FullFineTune => rms_norm_into_varmap(
            size,
            eps,
            &src_vb,
            loader
                .train_varmap
                .expect("FullFineTune requires train_varmap"),
            abs_path,
        ),
    }
}

/// Shared mode + train-varmap carrier used by every arch wrapper to
/// route base-weight construction through either the frozen
/// `candle_nn::linear` / `embedding` / `rms_norm` constructors or the
/// FFT `_into_varmap` analogues above.
pub(crate) struct BaseLoader<'a> {
    pub mode: TrainMode,
    pub train_varmap: Option<&'a VarMap>,
}

impl BaseLoader<'_> {
    /// FFT-aware analogue of `candle_nn::linear` (with bias).
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying load path.
    pub fn linear(
        &self,
        in_dim: usize,
        out_dim: usize,
        base_vb: VarBuilder,
        abs_path: &str,
    ) -> Result<Linear> {
        match self.mode {
            // Why: in QLoRA mode the per-target linears are wrapped by
            // QLoraLinear (which owns its own quantized base), not by a
            // plain Linear. This helper is only called for non-target or
            // bias-bearing linears (e.g. Qwen2 q/k/v carry bias) — for
            // those we read the dense base frozen, identical to LoraOnly.
            TrainMode::LoraOnly | TrainMode::Qlora { .. } => {
                candle_nn::linear(in_dim, out_dim, base_vb)
            }
            TrainMode::FullFineTune => linear_into_varmap(
                in_dim,
                out_dim,
                &base_vb,
                self.train_varmap
                    .expect("FullFineTune requires train_varmap"),
                abs_path,
                true,
            ),
        }
    }

    /// FFT-aware analogue of `candle_nn::linear_no_bias`.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying load path.
    pub fn linear_no_bias(
        &self,
        in_dim: usize,
        out_dim: usize,
        base_vb: VarBuilder,
        abs_path: &str,
    ) -> Result<Linear> {
        match self.mode {
            // Why: same rationale as `linear` — QLoRA wraps targets via
            // `maybe_lora_linear`'s QLoRA branch directly, so callers of
            // this helper get the dense frozen path.
            TrainMode::LoraOnly | TrainMode::Qlora { .. } => {
                candle_nn::linear_no_bias(in_dim, out_dim, base_vb)
            }
            TrainMode::FullFineTune => linear_into_varmap(
                in_dim,
                out_dim,
                &base_vb,
                self.train_varmap
                    .expect("FullFineTune requires train_varmap"),
                abs_path,
                false,
            ),
        }
    }
}
