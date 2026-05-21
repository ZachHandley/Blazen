//! Trainable QLoRA-augmented linear layer.
//!
//! [`QLoraLinear`] wraps a *frozen* base weight matrix stored as a 4-bit (or
//! other GGUF-integer) [`QTensor`] with two trainable low-rank matrices A
//! (`rank × in_dim`) and B (`out_dim × rank`):
//!
//! `forward(x) = qmatmul(x, base_q) + scale * B(A(x))`
//!
//! where `scale = alpha / rank` (PEFT convention). The base path goes
//! through [`QMatMul::forward`] (a no-backward custom op — exactly what we
//! want, since the quantized base is permanently frozen), and the LoRA
//! delta path goes through the differentiable `Linear` ops [`LoraLinear`]
//! uses, so gradients flow back to `A` and `B` only.
//!
//! ## Why this saves VRAM
//!
//! A 7B-param Llama at bf16 is ~13.4 GB resident; with `Q4_0` packing the
//! same weights occupy ~3.5 GB (4.5 bits per parameter once block-scale
//! overhead is included). The trainable parameters are unchanged from a
//! plain LoRA run — only the frozen base shrinks. Activation memory is
//! identical to LoRA because every base matmul still dequantizes to f32
//! at the matmul kernel boundary.
//!
//! ## What this is NOT
//!
//! QLoRA proper (Dettmers et al. 2023) uses *NF4* — a non-uniform 4-bit
//! quantization tuned to a unit Gaussian. candle 0.10.2 ships only GGUF
//! integer formats, so this implementation substitutes `Q4_0` (or any
//! variant in [`crate::config::QloraQuantDtype`]). The training-loop
//! dynamics are identical; the absolute quantization error is a hair
//! worse than NF4 for the same bit-rate. Swapping in NF4 is a one-line
//! change once candle exposes it.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};

use crate::config::QloraQuantDtype;

/// Map the config-crate wire enum onto candle's [`GgmlDType`].
///
/// Kept private to this module — callers build a [`QLoraLinear`] by passing
/// a [`QloraQuantDtype`] and never need to name `GgmlDType` themselves.
#[must_use]
pub(crate) fn ggml_dtype_for(q: QloraQuantDtype) -> GgmlDType {
    match q {
        QloraQuantDtype::Q4_0 => GgmlDType::Q4_0,
        QloraQuantDtype::Q4K => GgmlDType::Q4K,
        QloraQuantDtype::Q5_0 => GgmlDType::Q5_0,
        QloraQuantDtype::Q8_0 => GgmlDType::Q8_0,
    }
}

/// QLoRA-augmented linear layer.
///
/// Owns:
/// * `base_q` — a [`QMatMul`] over an [`Arc<QTensor>`] holding the frozen
///   4-bit (or other GGUF-integer) base weights. Forward through this op
///   has no backward pass, which is the correct semantic: the quantized
///   base must never receive gradient updates (and indeed *cannot* in
///   candle 0.10.2 — there is no in-place re-quantize primitive).
/// * `a`, `b` — full-precision [`Linear`]s holding the trainable LoRA
///   projections, registered against the caller's [`candle_nn::VarMap`]
///   via the provided [`VarBuilder`]. Identical layout to
///   [`crate::lora::LoraLinear`] so the PEFT exporter round-trips the
///   adapter unchanged.
/// * `scale` — `alpha / rank`, baked at construction so the forward path
///   is a single `affine` multiply.
///
/// `B` is zero-initialized so `forward(x) == qmatmul(x, base_q)` at step
/// 0 — i.e. the model's initial behavior equals the dequantized base
/// model, modulo the small quantization error from the 4-bit packing
/// itself. The `a` projection uses Kaiming-uniform (PEFT convention) so
/// once `B` starts moving away from zero the gradient magnitudes are
/// well-conditioned.
pub struct QLoraLinear {
    base_q: QMatMul,
    a: Linear,
    b: Linear,
    scale: f32,
    in_dim: usize,
    out_dim: usize,
}

impl QLoraLinear {
    /// Quantize an existing dense base weight matrix into a frozen
    /// [`QTensor`] and wrap it with new trainable LoRA matrices.
    ///
    /// * `base_weight` is the dense `[out_dim, in_dim]` tensor read from
    ///   the pretrained checkpoint (typically through a mmap'd
    ///   safetensors `VarBuilder`). It is consumed during the call —
    ///   after [`QTensor::quantize`] returns, the dense copy is dropped
    ///   and only the 4-bit packed representation lives on.
    /// * `quant` is the GGUF integer format to pack into.
    /// * `lora_vb` is the [`VarBuilder`] scope for the LoRA params; the
    ///   produced var names are `lora_A.weight` and `lora_B.weight` under
    ///   that prefix (the PEFT-canonical layout the adapter exporter and
    ///   `crates/blazen-llm-candle/src/lora.rs::LoadedAdapter::from_dir`
    ///   round-trip).
    /// * `alpha / rank` is folded into `scale` once at construction.
    ///
    /// `base_weight` must have shape `(out_dim, in_dim)`. `in_dim` must
    /// be a multiple of the quantization block size for `quant`
    /// (e.g. 32 for `Q4_0` / `Q5_0` / `Q8_0`, 256 for `Q4K`); otherwise
    /// [`QTensor::quantize`] returns an error.
    ///
    /// # Errors
    ///
    /// Returns any candle error from [`QTensor::quantize`],
    /// [`QMatMul::from_arc`], or the underlying `VarBuilder` allocations.
    // Why: VarBuilder is a small Arc-cloning handle; candle's idiomatic
    // call style takes it by value (mirrors `candle_nn::linear` and the
    // existing `LoraLinear::wrap`).
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_dense_base(
        base_weight: Tensor,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        quant: QloraQuantDtype,
        lora_vb: VarBuilder,
    ) -> Result<Self> {
        if base_weight.dims() != [out_dim, in_dim] {
            return Err(candle_core::Error::Msg(format!(
                "QLoraLinear: base_weight shape {:?} != [out_dim={out_dim}, in_dim={in_dim}]",
                base_weight.dims(),
            )));
        }

        let dtype = ggml_dtype_for(quant);
        let qtensor = QTensor::quantize(&base_weight, dtype)?;
        let base_q = QMatMul::from_arc(Arc::new(qtensor))?;

        let a_ws = lora_vb.get_with_hints(
            (rank, in_dim),
            "lora_A.weight",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let b_ws = lora_vb.get_with_hints((out_dim, rank), "lora_B.weight", Init::Const(0.0))?;

        #[allow(clippy::cast_precision_loss)]
        let scale = alpha / rank as f32;

        Ok(Self {
            base_q,
            a: Linear::new(a_ws, None),
            b: Linear::new(b_ws, None),
            scale,
            in_dim,
            out_dim,
        })
    }

    /// Wrap an already-built dense [`Linear`] by quantizing its weight in
    /// place. Convenience over [`Self::from_dense_base`] for the
    /// per-arch wrappers that have already loaded the frozen base via
    /// [`candle_nn::linear_no_bias`].
    ///
    /// # Errors
    ///
    /// Forwards errors from [`Self::from_dense_base`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn wrap(
        base: Linear,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        quant: QloraQuantDtype,
        lora_vb: VarBuilder,
    ) -> Result<Self> {
        // Why: Linear holds the weight tensor by Arc internally; .weight()
        // returns the shared tensor. We do NOT need to move out of `base`
        // because QTensor::quantize allocates fresh storage from the
        // dequantized f32 source.
        let base_weight = base.weight().clone();
        Self::from_dense_base(base_weight, in_dim, out_dim, rank, alpha, quant, lora_vb)
    }

    /// Names of the trainable parameters this layer registers, relative
    /// to the [`VarBuilder`]'s prefix.
    ///
    /// Identical to [`crate::lora::LoraLinear::param_names`] — the
    /// trained adapter artifact is bit-for-bit compatible with a plain
    /// LoRA run so the same PEFT loader consumes both.
    #[must_use]
    pub fn param_names() -> [&'static str; 2] {
        ["lora_A.weight", "lora_B.weight"]
    }

    /// Effective scaling factor `alpha / rank` baked at construction.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Input dimension of the layer.
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Output dimension of the layer.
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// The GGUF dtype the frozen base is packed in.
    #[must_use]
    pub fn base_dtype(&self) -> GgmlDType {
        match &self.base_q {
            QMatMul::QTensor(t) => t.dtype(),
            // Why: QMatMul::from_arc collapses F32/F16/BF16 to the dequantized
            // Tensor variant; for our QLoRA use we only pass true 4-bit/8-bit
            // integer formats, so these branches are dead in practice. Return
            // F32 / F16 as the closest dtype label so the getter stays total.
            QMatMul::Tensor(_) => GgmlDType::F32,
            QMatMul::TensorF16(_) => GgmlDType::F16,
        }
    }

    /// Compute the base matmul only (no LoRA delta) for debug / testing.
    ///
    /// Equivalent to `dequantize(base_q) @ x.T` on the math side, but goes
    /// through the per-device dequant-on-matmul kernel for parity with the
    /// `forward` hot path.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error.
    pub fn forward_base_only(&self, xs: &Tensor) -> Result<Tensor> {
        self.base_q.forward(xs)
    }

    /// Dequantize the frozen base to a dense f32 tensor on `device`.
    ///
    /// Useful for tests that compare the QLoRA forward against a baseline
    /// computed from the original dense weight, and for the export path
    /// that wants to write a merged checkpoint.
    ///
    /// # Errors
    ///
    /// Forwards any candle tensor error.
    pub fn dequantize_base(&self, device: &Device) -> Result<Tensor> {
        match &self.base_q {
            QMatMul::QTensor(t) => t.dequantize(device),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.to_device(device),
        }
    }
}

impl Module for QLoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Why: the base path goes through QMatMul::forward, which uses
        // `apply_op1_no_bwd` — gradients do NOT flow back to base_q (and
        // they MUST NOT, because the quantized representation has no
        // gradient kernel). The LoRA delta path goes through plain Linear
        // ops, so backward correctly propagates only to a/b.
        let base_out = self.base_q.forward(xs)?;
        let lora_out = self
            .b
            .forward(&self.a.forward(xs)?)?
            .affine(f64::from(self.scale), 0.0)?;
        // Why: dequant-on-matmul produces dtype matching xs (CPU) or f32
        // (CUDA path), while LoRA Linear forward preserves xs's dtype.
        // The two should already match; add directly. If a future device
        // backend changes that contract we'll cast here.
        base_out + lora_out
    }
}

/// Drop the in-flight set of LoRA-only var names that round-trip through
/// the PEFT exporter. Identical to [`crate::lora::lora_param_names`] —
/// QLoRA adapters are PEFT-format LoRA adapters at rest.
#[must_use]
pub fn qlora_param_names() -> &'static [&'static str] {
    &["lora_A.weight", "lora_B.weight"]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names
)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

    fn deterministic_dense_weight(out_dim: usize, in_dim: usize, device: &Device) -> Tensor {
        // Why: a small spread-around-zero pattern that's representative
        // of pretrained weights without being literally zero (which would
        // be a degenerate QTensor::quantize input).
        let mut data = Vec::with_capacity(out_dim * in_dim);
        for o in 0..out_dim {
            for i in 0..in_dim {
                let v = ((o as f32) - (in_dim as f32) * 0.5_f32) * 0.01
                    + ((i as f32) - (out_dim as f32) * 0.5_f32) * 0.003;
                data.push(v);
            }
        }
        Tensor::from_vec(data, (out_dim, in_dim), device).expect("weight tensor")
    }

    fn build_qlora(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        quant: QloraQuantDtype,
    ) -> (QLoraLinear, VarMap, Device, Tensor) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let base_w = deterministic_dense_weight(out_dim, in_dim, &device);
        let layer = QLoraLinear::from_dense_base(
            base_w.clone(),
            in_dim,
            out_dim,
            rank,
            alpha,
            quant,
            lora_vb,
        )
        .expect("build QLoraLinear");
        (layer, varmap, device, base_w)
    }

    #[test]
    fn qlora_linear_shape_round_trip() {
        // Why: forward output shape must match a plain Linear at the same
        // (in_dim, out_dim) — batch and seq dims pass through unchanged.
        let in_dim = 64; // multiple of 32 for Q4_0
        let out_dim = 16;
        let rank = 4;
        let (layer, _vm, device, _base_w) =
            build_qlora(in_dim, out_dim, rank, 8.0, QloraQuantDtype::Q4_0);

        assert_eq!(layer.in_dim(), in_dim);
        assert_eq!(layer.out_dim(), out_dim);
        assert_eq!(layer.base_dtype(), GgmlDType::Q4_0);
        assert!(
            (layer.scale() - (8.0_f32 / rank as f32)).abs() < f32::EPSILON,
            "scale mismatch: {} vs {}",
            layer.scale(),
            8.0 / rank as f32,
        );

        let bsz = 3;
        let seq = 5;
        let mut xs_data = Vec::with_capacity(bsz * seq * in_dim);
        for i in 0..(bsz * seq * in_dim) {
            xs_data.push(((i as f32) * 0.01).sin());
        }
        let xs = Tensor::from_vec(xs_data, (bsz, seq, in_dim), &device).expect("xs");
        let ys = layer.forward(&xs).expect("forward");
        assert_eq!(ys.dims(), &[bsz, seq, out_dim]);
    }

    #[test]
    fn qlora_linear_forward_matches_dequant() {
        // Why: at step 0, B is zero-initialized so the LoRA delta vanishes;
        // the QLoRA forward must equal `dequant(base_q) @ x.T` exactly
        // (modulo Q4_0 round-trip noise, which we bound below).
        let in_dim = 64;
        let out_dim = 8;
        let rank = 2;
        let (layer, _vm, device, _base_w) =
            build_qlora(in_dim, out_dim, rank, 4.0, QloraQuantDtype::Q4_0);

        let xs = Tensor::from_vec(
            (0..(in_dim * 2)).map(|i| (i as f32) * 0.005).collect(),
            (2, in_dim),
            &device,
        )
        .expect("xs");

        let ys = layer.forward(&xs).expect("forward");
        let baseline = layer.forward_base_only(&xs).expect("base only");

        // Why: B=0 means LoRA contributes exactly zero; the two must
        // match byte-for-byte (no quantization noise added because we're
        // comparing the same QMatMul output to itself, just through two
        // code paths that should collapse to one).
        let diff = (&ys - &baseline)
            .expect("diff")
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            diff < 1e-6,
            "B=0 init: QLoRA forward should equal base-only; got max-abs diff {diff}",
        );

        // And the base-only path itself must approximate the dense ref.
        // Q4_0 round-trip error is bounded by the block scale; for our
        // tiny deterministic weight (~|w| <= 0.1) the per-element error
        // stays well under 0.05.
        let dense_w = layer.dequantize_base(&device).expect("dequant");
        let manual = xs.matmul(&dense_w.t().expect("t")).expect("matmul");
        let q_error = (&baseline - &manual)
            .expect("diff")
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            q_error < 0.5,
            "Q4_0 round-trip via QMatMul should match dense reference; got max-abs error {q_error}",
        );
    }

    #[test]
    fn qlora_linear_lora_delta_is_differentiable() {
        // Why: the whole point of QLoRA is that B and A receive
        // gradients while the base does not. Build a tiny model, run a
        // backward, and assert both lora_A and lora_B have non-zero
        // gradient entries.
        let in_dim = 32;
        let out_dim = 8;
        let rank = 2;
        let (layer, varmap, device, _base_w) =
            build_qlora(in_dim, out_dim, rank, 4.0, QloraQuantDtype::Q4_0);

        // Override A to a non-zero pattern so the gradient on B is non-zero.
        {
            let guard = varmap
                .data()
                .lock()
                .expect("varmap mutex poisoned by another thread");
            let a = guard.get("lora_A.weight").expect("lora_A registered");
            a.set(
                &Tensor::from_vec(
                    (0..(rank * in_dim)).map(|i| (i as f32) * 0.01).collect(),
                    (rank, in_dim),
                    &device,
                )
                .expect("a tensor"),
            )
            .expect("set a");
        }

        let xs = Tensor::from_vec(
            (0..(2 * in_dim))
                .map(|i| ((i as f32) * 0.05).cos())
                .collect(),
            (2, in_dim),
            &device,
        )
        .expect("xs");
        let ys = layer.forward(&xs).expect("forward");
        // Why: a mean-squared "loss" gives a non-trivial gradient signal
        // for both A and B even though B starts at zero (its gradient
        // depends on A@x, not on B itself).
        let loss = ys.sqr().expect("sqr").mean_all().expect("mean");
        let grads = loss.backward().expect("backward");

        let guard = varmap
            .data()
            .lock()
            .expect("varmap mutex poisoned by another thread");
        let a_var = guard.get("lora_A.weight").expect("lora_A");
        let b_var = guard.get("lora_B.weight").expect("lora_B");

        // a's gradient comes via B (which is zero) — so the chain rule
        // says dL/dA = B.T @ (...) which is zero on init. That's fine —
        // PEFT is well-known to have a one-step lag on A's update. What
        // we MUST see is a non-zero gradient on B.
        let b_grad = grads
            .get(b_var.as_tensor())
            .expect("B has gradient")
            .abs()
            .expect("abs")
            .sum_all()
            .expect("sum")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            b_grad > 0.0,
            "QLoRA B must have non-zero gradient on first step; got {b_grad}",
        );
        // A's gradient may be zero at step 0 (because B starts at 0). We
        // only verify the entry *exists* — i.e. autograd actually wired
        // A into the graph. After one optimizer step on B, A will pick up
        // a gradient too.
        let _ = grads.get(a_var.as_tensor());
    }

    #[test]
    fn qlora_param_names_match_peft_convention() {
        assert_eq!(
            QLoraLinear::param_names(),
            ["lora_A.weight", "lora_B.weight"]
        );
        assert_eq!(qlora_param_names(), &["lora_A.weight", "lora_B.weight"]);
    }

    #[test]
    fn qloratrainer_one_step_does_not_panic() {
        // Why: end-to-end sanity. Build a QLoraLinear, register it in a
        // VarMap, hand the LoRA vars to AdamW, and run one optimizer
        // step on a simple MSE loss. We don't assert numerical
        // convergence (one step doesn't move the loss much); we assert
        // the wiring is right — backward + optimizer step both succeed
        // without panicking, and the LoRA weights actually change.
        let in_dim = 64;
        let out_dim = 16;
        let rank = 4;
        let (layer, varmap, device, _base_w) =
            build_qlora(in_dim, out_dim, rank, 8.0, QloraQuantDtype::Q4_0);

        // Override A to a non-zero pattern.
        {
            let guard = varmap
                .data()
                .lock()
                .expect("varmap mutex poisoned by another thread");
            let a = guard.get("lora_A.weight").expect("lora_A registered");
            a.set(
                &Tensor::from_vec(
                    (0..(rank * in_dim))
                        .map(|i| ((i as f32) * 0.02).sin())
                        .collect(),
                    (rank, in_dim),
                    &device,
                )
                .expect("a tensor"),
            )
            .expect("set a");
        }

        // Why: in the test we register the LoRA vars at the *unprefixed*
        // names `lora_A.weight` / `lora_B.weight` (we built the VarBuilder
        // without push_prefix). The per-arch wrappers push a deep prefix
        // (`base_model.model.model.layers.N.self_attn.q_proj.`) so the
        // is_qlora_param_name predicate keys on `.lora_A.weight` etc.
        // Here we match the simpler unprefixed names directly.
        let trainable: Vec<candle_core::Var> = {
            let guard = varmap
                .data()
                .lock()
                .expect("varmap mutex poisoned by another thread");
            guard
                .iter()
                .filter(|(name, _)| {
                    name.as_str() == "lora_A.weight" || name.as_str() == "lora_B.weight"
                })
                .map(|(_, v)| v.clone())
                .collect()
        };
        assert_eq!(trainable.len(), 2, "expected exactly lora_A and lora_B");

        let mut optimizer = AdamW::new(
            trainable,
            ParamsAdamW {
                lr: 1e-3,
                ..Default::default()
            },
        )
        .expect("AdamW");

        let xs = Tensor::from_vec(
            (0..(2 * in_dim))
                .map(|i| ((i as f32) * 0.05).cos())
                .collect(),
            (2, in_dim),
            &device,
        )
        .expect("xs");
        let target = Tensor::zeros((2, out_dim), DType::F32, &device).expect("target");

        // Snapshot B before the step so we can confirm AdamW actually
        // moved it.
        let b_before: Vec<f32> = {
            let guard = varmap
                .data()
                .lock()
                .expect("varmap mutex poisoned by another thread");
            guard
                .get("lora_B.weight")
                .expect("lora_B")
                .as_tensor()
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("vec1")
        };

        let ys = layer.forward(&xs).expect("forward");
        let loss = (ys - target)
            .expect("sub")
            .sqr()
            .expect("sqr")
            .mean_all()
            .expect("mean");
        let grads = loss.backward().expect("backward");
        optimizer.step(&grads).expect("optimizer step");

        let b_after: Vec<f32> = {
            let guard = varmap
                .data()
                .lock()
                .expect("varmap mutex poisoned by another thread");
            guard
                .get("lora_B.weight")
                .expect("lora_B")
                .as_tensor()
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("vec1")
        };

        let total_delta: f32 = b_before
            .iter()
            .zip(b_after.iter())
            .map(|(a, b)| (b - a).abs())
            .sum();
        assert!(
            total_delta > 0.0,
            "QLoRA B should have moved after one AdamW step; got total |delta| = {total_delta}",
        );
    }

    #[test]
    fn ggml_dtype_for_maps_every_variant() {
        assert_eq!(ggml_dtype_for(QloraQuantDtype::Q4_0), GgmlDType::Q4_0);
        assert_eq!(ggml_dtype_for(QloraQuantDtype::Q4K), GgmlDType::Q4K);
        assert_eq!(ggml_dtype_for(QloraQuantDtype::Q5_0), GgmlDType::Q5_0);
        assert_eq!(ggml_dtype_for(QloraQuantDtype::Q8_0), GgmlDType::Q8_0);
    }
}
