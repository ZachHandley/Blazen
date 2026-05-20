//! Trainable LoRA-augmented linear layer.
//!
//! Wraps a frozen base [`Linear`] with two trainable low-rank matrices A
//! (`rank × in_dim`) and B (`out_dim × rank`).
//!
//! `forward(x) = base(x) + scale * B(A(x))`
//!
//! where `scale = (alpha / rank) * runtime_scale` — the PEFT-canonical
//! formula mirrored from `crates/blazen-llm-candle/src/lora.rs::LoraLayer`
//! so trained adapters round-trip through that loader.

use std::collections::HashMap;
use std::sync::MutexGuard;

use candle_core::{Result, Tensor, Var};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};

/// LoRA-augmented linear layer with a frozen base and trainable A/B.
///
/// The base [`Linear`] is loaded from frozen tensors; the A and B
/// projections are built against a [`VarBuilder::from_varmap`]-scoped
/// builder so their weights are registered as `Var` entries in the
/// caller's [`VarMap`] and pick up gradients during `loss.backward()`.
///
/// `B` is initialized to all zeros so `B @ A == 0` at training start —
/// the model's initial behavior is identical to the unadapted base.
/// `A` uses Kaiming-uniform (PEFT convention) for non-pathological
/// gradient magnitudes once `B` starts moving away from zero.
pub struct LoraLinear {
    base: Linear,
    a: Linear,
    b: Linear,
    scale: f32,
}

impl LoraLinear {
    /// Wrap an existing base [`Linear`] with new trainable LoRA matrices.
    ///
    /// `vb` is the [`VarBuilder`] scope for the LoRA params; the produced
    /// var names are `lora_A.weight` and `lora_B.weight` under that prefix
    /// (the PEFT-canonical layout PR1's `LoadedAdapter::from_dir` expects
    /// after stripping `base_model.model.`).
    ///
    /// `alpha / rank` is folded into `scale` once at construction time so
    /// the forward path is a single `affine` multiply rather than a
    /// per-call division.
    ///
    /// # Errors
    ///
    /// Returns any candle error from the underlying `VarBuilder` allocations.
    // Why: VarBuilder is a small Arc-cloning handle, but candle_nn's idiomatic
    // call style (mirroring `candle_nn::linear`) takes it by value because
    // the typical site moves a freshly `push_prefix`-scoped builder in.
    #[allow(clippy::needless_pass_by_value)]
    pub fn wrap(
        base: Linear,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let a_ws = vb.get_with_hints(
            (rank, in_dim),
            "lora_A.weight",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let b_ws = vb.get_with_hints((out_dim, rank), "lora_B.weight", Init::Const(0.0))?;

        #[allow(clippy::cast_precision_loss)]
        let scale = alpha / rank as f32;

        Ok(Self {
            base,
            a: Linear::new(a_ws, None),
            b: Linear::new(b_ws, None),
            scale,
        })
    }

    /// Names of the trainable parameters this layer registers, relative
    /// to the [`VarBuilder`]'s prefix.
    #[must_use]
    pub fn param_names() -> [&'static str; 2] {
        ["lora_A.weight", "lora_B.weight"]
    }

    /// Effective scaling factor `alpha / rank` baked at construction.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

impl Module for LoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(xs)?;
        let lora_out = self
            .b
            .forward(&self.a.forward(xs)?)?
            .affine(f64::from(self.scale), 0.0)?;
        base_out + lora_out
    }
}

/// Freeze base-model parameters by removing every var whose name does NOT
/// contain a LoRA suffix from the trainable-vars set.
///
/// The returned `Vec<Var>` is the trainable subset, suitable for passing
/// directly to `candle_nn::AdamW::new(vars, params)`. The full [`VarMap`]
/// is unchanged so it can still be checkpointed.
///
/// `target_modules` is currently informational — the LoRA suffix
/// (`lora_A.weight` / `lora_B.weight`) is what the filter actually keys on
/// because the per-arch wrappers (Wave 2A/B/C) only build LoraLinear at
/// the target sites in the first place. Keeping the param mirrors the
/// PEFT config field name for callers.
///
/// # Panics
///
/// Panics if the [`VarMap`]'s internal mutex has been poisoned by a
/// concurrent panic in another thread.
#[must_use]
pub fn freeze_base_params(varmap: &VarMap, _target_modules: &[&str]) -> Vec<Var> {
    let guard: MutexGuard<'_, HashMap<String, Var>> = varmap
        .data()
        .lock()
        .expect("varmap mutex poisoned by another thread");
    guard
        .iter()
        .filter(|(name, _)| is_lora_param_name(name))
        .map(|(_, v)| v.clone())
        .collect()
}

/// Enumerate the LoRA-param leaf names this crate writes per layer.
///
/// Matches the PEFT suffix convention so PR1's `LoadedAdapter` recognizes
/// the keys without translation.
#[must_use]
pub fn lora_param_names() -> &'static [&'static str] {
    &["lora_A.weight", "lora_B.weight"]
}

fn is_lora_param_name(name: &str) -> bool {
    name.ends_with(".lora_A.weight") || name.ends_with(".lora_B.weight")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_base(in_dim: usize, out_dim: usize, device: &Device) -> Linear {
        let w = Tensor::ones((out_dim, in_dim), DType::F32, device).expect("base weight ones");
        Linear::new(w, None)
    }

    #[test]
    fn lora_linear_zero_init_b_means_zero_delta() {
        let device = Device::Cpu;
        let in_dim = 4;
        let out_dim = 3;
        let rank = 2;

        let base = make_base(in_dim, out_dim, &device);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lora = LoraLinear::wrap(base, in_dim, out_dim, rank, 8.0, vb).expect("wrap");

        let base_only = make_base(in_dim, out_dim, &device);

        let x = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2, in_dim),
            &device,
        )
        .expect("input tensor");

        let lora_out = lora.forward(&x).expect("lora forward");
        let base_out = base_only.forward(&x).expect("base forward");

        let diff = (&lora_out - &base_out).expect("diff");
        let max_abs = diff
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");

        assert!(
            max_abs < 1e-6,
            "B-init-to-zero means LoraLinear must equal base; got max-abs delta {max_abs}",
        );
    }

    #[test]
    fn lora_linear_scale_correctly_applied() {
        let device = Device::Cpu;
        let in_dim = 2;
        let out_dim = 2;
        let rank = 1;
        let alpha = 4.0_f32;

        let base_w =
            Tensor::zeros((out_dim, in_dim), DType::F32, &device).expect("base weight zeros");
        let base = Linear::new(base_w, None);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lora = LoraLinear::wrap(base, in_dim, out_dim, rank, alpha, vb).expect("wrap");

        // Why: overwrite the randomly-initialized A and zero-initialized B with
        // hand-picked values so the expected output is exact.
        {
            let guard = varmap
                .data()
                .lock()
                .expect("varmap mutex poisoned by another thread");
            let a = guard.get("lora_A.weight").expect("lora_A registered");
            a.set(
                &Tensor::from_vec(vec![1.0_f32, 0.0], (rank, in_dim), &device).expect("a override"),
            )
            .expect("set a");
            let b = guard.get("lora_B.weight").expect("lora_B registered");
            b.set(
                &Tensor::from_vec(vec![2.0_f32, 0.0], (out_dim, rank), &device)
                    .expect("b override"),
            )
            .expect("set b");
        }

        let x = Tensor::from_vec(vec![3.0_f32, 0.0], (1, in_dim), &device).expect("input");
        let out = lora.forward(&x).expect("forward");
        let rows: Vec<Vec<f32>> = out.to_vec2().expect("to_vec2");

        // base contributes zero. A(x) = [[3]], B(A(x)) = [[6],[0]], scale = alpha/rank = 4.
        // expected first dim = 4 * 6 = 24, second dim = 0.
        let expected_first = f64::from(alpha) / f64::from(u32::try_from(rank).unwrap()) * 6.0;
        let got_first = f64::from(rows[0][0]);
        assert!(
            (got_first - expected_first).abs() < 1e-5,
            "scale wrong: got {got_first}, want {expected_first}",
        );
        assert!(rows[0][1].abs() < 1e-5);

        #[allow(clippy::cast_precision_loss)]
        let expected_scale = alpha / rank as f32;
        assert!(
            (lora.scale() - expected_scale).abs() < f32::EPSILON,
            "scale getter wrong: got {}, want {expected_scale}",
            lora.scale(),
        );
    }

    #[test]
    fn freeze_base_params_returns_only_lora_vars() {
        let device = Device::Cpu;
        let varmap = VarMap::new();

        // Why: simulate a mixed varmap with both a "base" var (frozen) and
        // two LoRA vars (trainable). freeze_base_params must return only
        // the latter two.
        let vb_lora = VarBuilder::from_varmap(&varmap, DType::F32, &device)
            .push_prefix("model.layers.0.self_attn.q_proj");
        let _ = LoraLinear::wrap(make_base(4, 4, &device), 4, 4, 2, 4.0, vb_lora).expect("wrap");

        // Why: register a non-LoRA "base" var by going through VarMap::get,
        // which is the same path the per-arch wrappers will use for frozen
        // base weights when feeding them from a VarBuilder::from_varmap.
        let _frozen = varmap
            .get(
                (2, 2),
                "model.layers.0.self_attn.q_proj.weight",
                Init::Const(0.0),
                DType::F32,
                &device,
            )
            .expect("register frozen base var");

        let trainable = freeze_base_params(&varmap, &["q_proj"]);
        assert_eq!(trainable.len(), 2, "expected exactly the two LoRA vars");
    }

    #[test]
    fn param_names_match_peft_convention() {
        assert_eq!(
            LoraLinear::param_names(),
            ["lora_A.weight", "lora_B.weight"]
        );
        assert_eq!(lora_param_names(), &["lora_A.weight", "lora_B.weight"]);
    }
}
