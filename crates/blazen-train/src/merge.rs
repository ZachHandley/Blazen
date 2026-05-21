//! Offline LoRA adapter merge.
//!
//! Two surfaces:
//!
//! * [`merge_lora_into_base`] — fold a single LoRA adapter into the base
//!   weights and write a plain (non-LoRA) safetensors file. The merged
//!   model is the same size as the base; runtime inference has zero LoRA
//!   overhead but the adapter can no longer be swapped out.
//! * [`merge_lora_blend`] — weighted-sum multiple LoRAs into the base.
//!   Useful for "task vector arithmetic"
//!   (e.g. `base + good_behavior - bad_behavior`) and for the
//!   normalized-blend case (weights summing to one yields the weighted
//!   average of the adapters).
//!
//! Both functions assume the PEFT-canonical key layout the inference-side
//! loader at `crates/blazen-llm-candle/src/lora.rs::build_layers`
//! consumes:
//! `base_model.model.<module_path>.lora_<A|B>.weight`. The
//! `base_model.model.` prefix is stripped to match against the base
//! safetensors' key `<module_path>.weight`.
//!
//! The math (per target Linear, per adapter):
//!
//! * single: `W_new = W_base + scale * (B @ A)`
//! * blend:  `W_new = W_base + Σ_i (w_i * scale_i * (B_i @ A_i))`
//!
//! `scale` is the PEFT-canonical `alpha / r` read from the adapter's
//! `adapter_config.json`. For [`merge_lora_into_base`] the caller can
//! supply an additional runtime multiplier; for [`merge_lora_blend`] each
//! `(path, weight)` pair carries its own blend weight applied on top of
//! the adapter's intrinsic scale.

#![cfg(feature = "engine")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::error::MergeError;

/// PEFT prefix that wraps every adapter weight name in
/// `adapter_model.safetensors`. Mirrors
/// `crates/blazen-llm-candle/src/lora.rs::PEFT_KEY_PREFIX` and
/// `crates/blazen-train/src/export.rs::PEFT_KEY_PREFIX`.
const PEFT_KEY_PREFIX: &str = "base_model.model.";

/// Minimal subset of PEFT's `adapter_config.json` schema needed to
/// recover the per-adapter `alpha / r` scale.
///
/// Mirrors `blazen_llm_candle::lora::LoraAdapterConfig` without taking a
/// circular dependency on the inference crate. Unknown keys are
/// accepted-and-ignored.
#[derive(Debug, Deserialize)]
struct AdapterConfigSubset {
    r: usize,
    lora_alpha: f32,
}

/// Parsed LoRA half-pair for one target Linear module.
struct AdapterModule {
    /// `lora_A.weight` — shape `(rank, in_dim)`.
    a: Tensor,
    /// `lora_B.weight` — shape `(out_dim, rank)`.
    b: Tensor,
    /// Rank dimension recovered from the A tensor (= `a.dims()[0]`).
    /// Stored explicitly so the rank-mismatch check across adapters
    /// doesn't have to re-inspect shapes.
    rank: usize,
}

/// Fully-parsed adapter ready for the merge math: per-module low-rank
/// pair plus the adapter's intrinsic `alpha / r` scale.
struct ParsedAdapter {
    /// Module-path → (A, B, rank) table. Keys have the
    /// `base_model.model.` prefix stripped so they match the base
    /// safetensors' module-path-without-suffix layout.
    modules: HashMap<String, AdapterModule>,
    /// PEFT-canonical scale `alpha / r`.
    alpha_over_r: f32,
}

/// Merge a single LoRA adapter into the base model.
///
/// Writes a new safetensors file at `output` containing every tensor from
/// `base_safetensors`, with each weight that the adapter targets
/// replaced by `W_base + scale * alpha/r * (B @ A)`. Tensors that the
/// adapter does not target are copied through verbatim.
///
/// `scale` is a runtime multiplier the caller layers on top of the
/// adapter's intrinsic `alpha / r` (use `1.0` for "full strength" — the
/// PEFT convention). The merged result is a plain dense safetensors
/// file: no `adapter_config.json` is emitted because the merged model
/// is no longer LoRA-shaped.
///
/// # Errors
///
/// * [`MergeError::Io`] / [`MergeError::Safetensors`] on file or parse
///   failure of either the base or the adapter.
/// * [`MergeError::MissingBaseTensor`] if the adapter targets a module
///   that does not exist in `base_safetensors` (almost always a base
///   model mismatch).
/// * [`MergeError::ShapeMismatch`] if `B @ A` does not match the base
///   weight's `(out_dim, in_dim)` shape.
/// * [`MergeError::Candle`] for tensor-op failures during the matmul or
///   addition.
pub fn merge_lora_into_base(
    base_safetensors: &Path,
    adapter_safetensors: &Path,
    output: &Path,
    scale: f32,
) -> Result<(), MergeError> {
    let device = Device::Cpu;
    let adapter_dir = adapter_safetensors
        .parent()
        .ok_or_else(|| MergeError::MalformedAdapter {
            path: adapter_safetensors.display().to_string(),
            reason: "adapter safetensors path has no parent directory".into(),
        })?;
    let adapter = load_adapter(adapter_safetensors, adapter_dir, &device)?;

    merge_into_base_inner(base_safetensors, output, &[(adapter, scale)], &device)
}

/// Multi-adapter weighted-sum merge.
///
/// Writes a new safetensors file at `output` containing every tensor
/// from `base`, with each weight that any adapter targets replaced by
/// `W_base + Σ_i (w_i * (alpha_i / r_i) * (B_i @ A_i))`.
///
/// Semantics of `adapters: &[(PathBuf, f32)]`:
///
/// * `PathBuf` is the path to an `adapter_model.safetensors` file. Its
///   sibling `adapter_config.json` is read to recover `alpha / r`.
/// * `f32` is the blend weight `w_i` applied on top of the adapter's
///   intrinsic `alpha / r`. Weights are **not** normalized — pass
///   weights that sum to 1.0 for "weighted average" semantics; pass
///   unnormalized weights for "task vector arithmetic" (negative
///   weights are accepted and subtract the adapter's contribution).
///
/// # Errors
///
/// * [`MergeError::EmptyBlend`] if `adapters` is empty.
/// * [`MergeError::RankMismatch`] if two adapters target the same
///   module with different ranks. (The math itself does not require
///   this — `B @ A` collapses rank — but mixing ranks is almost always
///   a sign of base-model mismatch; surfacing it explicitly is the safe
///   default.)
/// * Same I/O / shape / candle errors as [`merge_lora_into_base`].
pub fn merge_lora_blend(
    base: &Path,
    adapters: &[(PathBuf, f32)],
    output: &Path,
) -> Result<(), MergeError> {
    if adapters.is_empty() {
        return Err(MergeError::EmptyBlend);
    }

    let device = Device::Cpu;
    let mut parsed: Vec<(ParsedAdapter, f32)> = Vec::with_capacity(adapters.len());
    for (path, weight) in adapters {
        let dir = path.parent().ok_or_else(|| MergeError::MalformedAdapter {
            path: path.display().to_string(),
            reason: "adapter safetensors path has no parent directory".into(),
        })?;
        parsed.push((load_adapter(path, dir, &device)?, *weight));
    }

    // Why: rank consistency is checked once, up-front, so a partial-write
    // is impossible — we either fail before opening the output file or we
    // proceed to write the entire merged shard.
    if let Some((head, tail)) = parsed.split_first() {
        for (idx, (other, _)) in tail.iter().enumerate() {
            for (module, m_head) in &head.0.modules {
                if let Some(m_other) = other.modules.get(module)
                    && m_head.rank != m_other.rank
                {
                    return Err(MergeError::RankMismatch {
                        module: module.clone(),
                        first_rank: m_head.rank,
                        other_index: idx + 1,
                        other_rank: m_other.rank,
                    });
                }
            }
        }
    }

    merge_into_base_inner(base, output, &parsed, &device)
}

/// Shared body of the single-adapter and multi-adapter merge paths.
///
/// `adapters` is `&[(ParsedAdapter, runtime_weight)]`. The total delta
/// applied to each base tensor is
/// `Σ_i runtime_weight_i * (alpha_i / r_i) * (B_i @ A_i)`.
fn merge_into_base_inner(
    base_path: &Path,
    output_path: &Path,
    adapters: &[(ParsedAdapter, f32)],
    device: &Device,
) -> Result<(), MergeError> {
    let base_tensors = load_safetensors_into_tensors(base_path, device)?;

    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(base_tensors.len());

    // Walk the base once, accumulating any adapter deltas that target each key.
    for (key, base_t) in base_tensors {
        let module_path = key.strip_suffix(".weight");

        if let Some(module_path) = module_path {
            let mut delta_accum: Option<Tensor> = None;
            for (adapter, runtime_weight) in adapters {
                let Some(m) = adapter.modules.get(module_path) else {
                    continue;
                };
                let effective = adapter.alpha_over_r * *runtime_weight;
                let delta = compute_delta(module_path, m, effective, &base_t, device)?;
                delta_accum = Some(match delta_accum {
                    Some(prev) => prev.add(&delta)?,
                    None => delta,
                });
            }

            if let Some(delta) = delta_accum {
                // Why: cast delta to the base dtype before adding so the
                // output preserves the base's storage precision (bf16/f16
                // bases stay bf16/f16; f32 stays f32) — otherwise an f32
                // delta against a bf16 base would silently upcast the whole
                // shard and double its file size.
                let delta = if delta.dtype() == base_t.dtype() {
                    delta
                } else {
                    delta.to_dtype(base_t.dtype())?
                };
                let merged_t = base_t.add(&delta)?;
                merged.insert(key, merged_t);
                continue;
            }
        }

        // Either the key didn't end in `.weight` (e.g. embedding tables
        // some checkpoints ship under non-`.weight` keys) or no adapter
        // touched it. Pass through unchanged.
        merged.insert(key, base_t);
    }

    // Fail loudly if any adapter targets a module the base doesn't have —
    // this almost always indicates a base-model mismatch and silently
    // dropping the delta would be the wrong default.
    for (adapter, _) in adapters {
        for module_path in adapter.modules.keys() {
            let expected = format!("{module_path}.weight");
            if !merged.contains_key(&expected) {
                return Err(MergeError::MissingBaseTensor(expected));
            }
        }
    }

    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    safetensors::tensor::serialize_to_file(merged.iter(), None, output_path).map_err(|e| {
        MergeError::Safetensors(format!(
            "serialize to {} failed: {e}",
            output_path.display()
        ))
    })?;

    Ok(())
}

/// Compute `effective * (B @ A)` shaped like the base linear's weight.
fn compute_delta(
    module: &str,
    pair: &AdapterModule,
    effective: f32,
    base_w: &Tensor,
    device: &Device,
) -> Result<Tensor, MergeError> {
    // Promote to f32 on the chosen device. PEFT adapters commonly ship in
    // bf16/f16 — we do the matmul + scale in f32 to avoid overflow and
    // silent precision-mismatch failures, then downcast back to the base
    // dtype at write time (see merge_into_base_inner).
    let a = ensure_f32_on(&pair.a, device)?;
    let b = ensure_f32_on(&pair.b, device)?;

    // A: (rank, in_dim), B: (out_dim, rank). Product: (out_dim, in_dim).
    let delta = b.matmul(&a)?.affine(f64::from(effective), 0.0)?;

    let base_dims = base_w.dims();
    let delta_dims = delta.dims();
    if base_dims != delta_dims {
        return Err(MergeError::ShapeMismatch {
            module: module.to_string(),
            base: base_dims.to_vec(),
            delta: delta_dims.to_vec(),
        });
    }
    Ok(delta)
}

fn ensure_f32_on(t: &Tensor, device: &Device) -> Result<Tensor, MergeError> {
    let on_dev = if t.device().same_device(device) {
        t.clone()
    } else {
        t.to_device(device)?
    };
    if on_dev.dtype() == DType::F32 {
        Ok(on_dev)
    } else {
        Ok(on_dev.to_dtype(DType::F32)?)
    }
}

/// Load + parse a PEFT adapter (`adapter_model.safetensors` +
/// `adapter_config.json`) into a [`ParsedAdapter`].
fn load_adapter(
    adapter_safetensors: &Path,
    adapter_dir: &Path,
    device: &Device,
) -> Result<ParsedAdapter, MergeError> {
    let cfg_path = adapter_dir.join("adapter_config.json");
    let cfg_bytes = std::fs::read(&cfg_path)?;
    let cfg: AdapterConfigSubset =
        serde_json::from_slice(&cfg_bytes).map_err(|e| MergeError::MalformedAdapter {
            path: cfg_path.display().to_string(),
            reason: format!("adapter_config.json parse failed: {e}"),
        })?;
    if cfg.r == 0 {
        return Err(MergeError::MalformedAdapter {
            path: cfg_path.display().to_string(),
            reason: "adapter_config.json: r must be > 0".into(),
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let alpha_over_r = cfg.lora_alpha / cfg.r as f32;

    let tensors = load_safetensors_into_tensors(adapter_safetensors, device)?;

    let mut a_map: HashMap<String, Tensor> = HashMap::new();
    let mut b_map: HashMap<String, Tensor> = HashMap::new();

    for (raw_key, tensor) in tensors {
        let key = raw_key.strip_prefix(PEFT_KEY_PREFIX).unwrap_or(&raw_key);
        if let Some(module) = key.strip_suffix(".lora_A.weight") {
            a_map.insert(module.to_string(), tensor);
        } else if let Some(module) = key.strip_suffix(".lora_B.weight") {
            b_map.insert(module.to_string(), tensor);
        } else if key.contains(".lora_A.") || key.contains(".lora_B.") {
            return Err(MergeError::MalformedAdapter {
                path: adapter_safetensors.display().to_string(),
                reason: format!("unexpected PEFT key shape (no .weight terminator): {raw_key}"),
            });
        }
        // Anything else (rare embedding-resize tables some PEFT trainers
        // ship alongside) is ignored — they are not consumed by the
        // delta path.
    }

    let mut modules: HashMap<String, AdapterModule> = HashMap::with_capacity(a_map.len());
    for (module, a_w) in a_map {
        let Some(b_w) = b_map.remove(&module) else {
            return Err(MergeError::UnpairedLora {
                module,
                missing: "lora_B.weight",
            });
        };
        let rank = a_w
            .dims()
            .first()
            .copied()
            .ok_or_else(|| MergeError::MalformedAdapter {
                path: adapter_safetensors.display().to_string(),
                reason: format!("lora_A for {module} has zero dimensions"),
            })?;
        modules.insert(
            module,
            AdapterModule {
                a: a_w,
                b: b_w,
                rank,
            },
        );
    }
    if let Some((module, _)) = b_map.into_iter().next() {
        return Err(MergeError::UnpairedLora {
            module,
            missing: "lora_A.weight",
        });
    }

    Ok(ParsedAdapter {
        modules,
        alpha_over_r,
    })
}

/// Load a safetensors file into an owned `HashMap<key, Tensor>` on
/// `device`. Thin wrapper over `candle_core::safetensors::load` that
/// rewrites the candle error into [`MergeError::Safetensors`] tagged
/// with the offending path.
fn load_safetensors_into_tensors(
    path: &Path,
    device: &Device,
) -> Result<HashMap<String, Tensor>, MergeError> {
    candle_core::safetensors::load(path, device)
        .map_err(|e| MergeError::Safetensors(format!("load from {} failed: {e}", path.display())))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Test-fixture root under ~/.cache/blazen-pr-am-research/test-fixtures
    /// (repo policy forbids /tmp). Each test allocates a UUID-ish unique
    /// subdir so parallel cargo-test runs don't collide.
    fn fixture_root() -> PathBuf {
        let home =
            std::env::var_os("HOME").expect("HOME env var must be set for PR-AM test fixtures");
        let root = PathBuf::from(home)
            .join(".cache")
            .join("blazen-pr-am-research")
            .join("test-fixtures");
        std::fs::create_dir_all(&root).expect("create fixture root");
        root
    }

    /// Allocate a unique scratch dir under the fixture root and clean it
    /// up at drop. We can't use the `tempfile` crate's `TempDir::path`
    /// pointing at /tmp because the project rule forbids /tmp scratch.
    struct ScratchDir(PathBuf);
    impl ScratchDir {
        fn new(tag: &str) -> Self {
            // Why: pid + nanos gives a collision-free name without pulling
            // in `uuid` for a one-off test fixture.
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let p = fixture_root().join(format!("{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&p).expect("create scratch dir");
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_base_safetensors(
        path: &Path,
        entries: &[(&str, &[usize], Vec<f32>)],
    ) -> candle_core::Result<()> {
        let device = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (key, shape, data) in entries {
            let t = Tensor::from_vec((*data).clone(), *shape, &device)?;
            tensors.insert((*key).to_string(), t);
        }
        safetensors::tensor::serialize_to_file(tensors.iter(), None, path).expect("serialize base");
        Ok(())
    }

    /// `(module_path, a_shape, a_data, b_shape, b_data)`.
    type AdapterModuleSpec<'a> = (&'a str, &'a [usize], Vec<f32>, &'a [usize], Vec<f32>);

    fn write_adapter(
        dir: &Path,
        rank: usize,
        alpha: f32,
        modules: &[AdapterModuleSpec<'_>],
    ) -> candle_core::Result<PathBuf> {
        std::fs::create_dir_all(dir).expect("mkdir adapter");
        let cfg = serde_json::json!({
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": modules.iter().map(|(m, ..)| *m).collect::<Vec<_>>(),
            "base_model_name_or_path": "test/base",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        });
        std::fs::write(dir.join("adapter_config.json"), cfg.to_string())
            .expect("write adapter_config");
        let device = Device::Cpu;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        for (module, a_shape, a_data, b_shape, b_data) in modules {
            let a = Tensor::from_vec((*a_data).clone(), *a_shape, &device)?;
            let b = Tensor::from_vec((*b_data).clone(), *b_shape, &device)?;
            ts.insert(format!("{PEFT_KEY_PREFIX}{module}.lora_A.weight"), a);
            ts.insert(format!("{PEFT_KEY_PREFIX}{module}.lora_B.weight"), b);
        }
        let adapter_path = dir.join("adapter_model.safetensors");
        safetensors::tensor::serialize_to_file(ts.iter(), None, &adapter_path)
            .expect("serialize adapter");
        Ok(adapter_path)
    }

    fn load_tensor(path: &Path, key: &str) -> Vec<f32> {
        let device = Device::Cpu;
        let map = candle_core::safetensors::load(path, &device).expect("load output");
        let t = map.get(key).expect("key present");
        t.flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1")
    }

    #[test]
    fn merge_single_adapter_preserves_shapes() {
        let scratch = ScratchDir::new("merge-single-shapes");
        let base_path = scratch.path().join("base.safetensors");
        let out_path = scratch.path().join("merged.safetensors");

        // base.q_proj.weight: (out=3, in=4) all zeros so the merged
        // result equals the pure delta — easy to assert.
        write_base_safetensors(
            &base_path,
            &[
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    &[3usize, 4],
                    vec![0.0_f32; 12],
                ),
                // A non-LoRA tensor (untargeted) — must pass through unchanged.
                (
                    "model.layers.0.input_layernorm.weight",
                    &[3usize],
                    vec![7.0_f32, 8.0, 9.0],
                ),
            ],
        )
        .unwrap();

        // rank=2, alpha=4 → alpha/r = 2.0. With scale=1.0 the effective
        // multiplier is 2.0.
        // A: (2,4), B: (3,2) — B @ A: (3,4).
        let adapter_dir = scratch.path().join("adapter");
        let adapter_path = write_adapter(
            &adapter_dir,
            2,
            4.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[2usize, 4],
                vec![
                    1.0_f32, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0,
                ],
                &[3usize, 2],
                vec![
                    1.0_f32, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0,
                ],
            )],
        )
        .unwrap();

        merge_lora_into_base(&base_path, &adapter_path, &out_path, 1.0).expect("merge ok");

        // Untargeted passthrough preserved.
        let ln = load_tensor(&out_path, "model.layers.0.input_layernorm.weight");
        assert_eq!(ln, vec![7.0, 8.0, 9.0]);

        // Targeted weight: shape preserved as (3,4) = 12 elems. The
        // exact value is 2.0 * (B @ A) since base is zero.
        // B @ A row 0 = [1,0,0,0]; row 1 = [0,1,0,0]; row 2 = [1,1,0,0].
        // Times 2.0:
        // [[2,0,0,0],[0,2,0,0],[2,2,0,0]].
        let q = load_tensor(&out_path, "model.layers.0.self_attn.q_proj.weight");
        let expected = vec![
            2.0_f32, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            2.0, 2.0, 0.0, 0.0,
        ];
        assert_eq!(q.len(), 12, "shape preserved as 12 elems");
        for (g, e) in q.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "delta math wrong: got {g}, want {e}");
        }
    }

    #[test]
    fn merge_blend_normalizes_correctly() {
        // Two adapters with the same rank, distinct deltas; blend with
        // positive and negative weights. The combined delta must equal
        // w1*Δ1 + w2*Δ2 (unnormalized — task-vector arithmetic semantics).
        let scratch = ScratchDir::new("merge-blend-normalize");
        let base_path = scratch.path().join("base.safetensors");
        let out_path = scratch.path().join("merged.safetensors");

        write_base_safetensors(
            &base_path,
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                &[2usize, 2],
                vec![0.0_f32; 4],
            )],
        )
        .unwrap();

        // Adapter 1: A=I_2, B=I_2, rank=1 each => B@A = I_2.
        // Wait — for rank=1 A is (1,2), B is (2,1). B@A = (2,2).
        // Use rank=2, alpha=2 → alpha/r=1.0 for both so the blend weight
        // alone controls each adapter's contribution.
        let a1_dir = scratch.path().join("a1");
        let a1 = write_adapter(
            &a1_dir,
            2,
            2.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0], // identity
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0], // identity
            )],
        )
        .unwrap();
        // B@A = I_2 — so Δ1 = 1.0 * I = [[1,0],[0,1]].

        let a2_dir = scratch.path().join("a2");
        let a2 = write_adapter(
            &a2_dir,
            2,
            2.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[2usize, 2],
                vec![0.0_f32, 1.0, 1.0, 0.0], // swap
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        // B@A for adapter 2 = [[0,1],[1,0]].

        // Weights: w1=0.75, w2=-0.25 → combined delta =
        //   0.75 * [[1,0],[0,1]] + (-0.25) * [[0,1],[1,0]]
        // = [[0.75, -0.25], [-0.25, 0.75]].
        let adapters = vec![(a1, 0.75_f32), (a2, -0.25_f32)];
        merge_lora_blend(&base_path, &adapters, &out_path).expect("blend ok");

        let q = load_tensor(&out_path, "model.layers.0.self_attn.q_proj.weight");
        let expected = [0.75_f32, -0.25, -0.25, 0.75];
        for (g, e) in q.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "blend math wrong: got {g}, want {e} (full: {q:?})"
            );
        }
    }

    #[test]
    fn merge_blend_normalized_weights_sum_to_one_yields_average() {
        // Two adapters with identical scales and weights summing to 1.0
        // should yield the weighted-average delta. We use 0.5/0.5 across
        // two distinct adapters; the merged delta must equal the average
        // of the two individual deltas.
        let scratch = ScratchDir::new("merge-blend-avg");
        let base_path = scratch.path().join("base.safetensors");
        let out_path = scratch.path().join("merged.safetensors");

        write_base_safetensors(
            &base_path,
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                &[2usize, 2],
                vec![10.0_f32, 20.0, 30.0, 40.0],
            )],
        )
        .unwrap();

        // Adapter 1: Δ1 = [[2,0],[0,2]] (with alpha/r=1.0, identity A & 2I B)
        let a1_dir = scratch.path().join("a1");
        let a1 = write_adapter(
            &a1_dir,
            2,
            2.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
                &[2usize, 2],
                vec![2.0_f32, 0.0, 0.0, 2.0],
            )],
        )
        .unwrap();

        // Adapter 2: Δ2 = [[0,4],[4,0]]
        let a2_dir = scratch.path().join("a2");
        let a2 = write_adapter(
            &a2_dir,
            2,
            2.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[2usize, 2],
                vec![0.0_f32, 1.0, 1.0, 0.0],
                &[2usize, 2],
                vec![4.0_f32, 0.0, 0.0, 4.0],
            )],
        )
        .unwrap();

        let adapters = vec![(a1, 0.5_f32), (a2, 0.5_f32)];
        merge_lora_blend(&base_path, &adapters, &out_path).expect("blend ok");

        // Combined delta = 0.5 * Δ1 + 0.5 * Δ2 = 0.5*[[2,0],[0,2]] + 0.5*[[0,4],[4,0]]
        //                = [[1,2],[2,1]]
        // Merged = base + delta = [[10+1, 20+2], [30+2, 40+1]]
        //        = [[11, 22], [32, 41]]
        let q = load_tensor(&out_path, "model.layers.0.self_attn.q_proj.weight");
        let expected = [11.0_f32, 22.0, 32.0, 41.0];
        for (g, e) in q.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "average-blend wrong: got {g}, want {e} (full: {q:?})"
            );
        }
    }

    #[test]
    fn merge_rejects_rank_mismatch() {
        let scratch = ScratchDir::new("merge-rank-mismatch");
        let base_path = scratch.path().join("base.safetensors");
        let out_path = scratch.path().join("merged.safetensors");

        write_base_safetensors(
            &base_path,
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                &[2usize, 2],
                vec![0.0_f32; 4],
            )],
        )
        .unwrap();

        // Adapter 1: rank=2.
        let a1_dir = scratch.path().join("a1");
        let a1 = write_adapter(
            &a1_dir,
            2,
            2.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        // Adapter 2: rank=1, same module.
        let a2_dir = scratch.path().join("a2");
        let a2 = write_adapter(
            &a2_dir,
            1,
            1.0,
            &[(
                "model.layers.0.self_attn.q_proj",
                &[1usize, 2],
                vec![1.0_f32, 1.0],
                &[2usize, 1],
                vec![1.0_f32, 1.0],
            )],
        )
        .unwrap();

        let adapters = vec![(a1, 1.0_f32), (a2, 1.0_f32)];
        let err = merge_lora_blend(&base_path, &adapters, &out_path)
            .expect_err("rank mismatch must be rejected");
        match err {
            MergeError::RankMismatch {
                module,
                first_rank,
                other_index,
                other_rank,
            } => {
                assert_eq!(module, "model.layers.0.self_attn.q_proj");
                assert_eq!(first_rank, 2);
                assert_eq!(other_index, 1);
                assert_eq!(other_rank, 1);
            }
            other => panic!("expected RankMismatch, got {other:?}"),
        }
    }

    #[test]
    fn merge_rejects_missing_base_tensor() {
        let scratch = ScratchDir::new("merge-missing-base");
        let base_path = scratch.path().join("base.safetensors");
        let out_path = scratch.path().join("merged.safetensors");

        // Base has q_proj, but adapter targets k_proj (not present).
        write_base_safetensors(
            &base_path,
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                &[2usize, 2],
                vec![0.0_f32; 4],
            )],
        )
        .unwrap();

        let adapter_dir = scratch.path().join("adapter");
        let adapter_path = write_adapter(
            &adapter_dir,
            2,
            2.0,
            &[(
                "model.layers.0.self_attn.k_proj",
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
                &[2usize, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();

        let err = merge_lora_into_base(&base_path, &adapter_path, &out_path, 1.0)
            .expect_err("missing base must be rejected");
        match err {
            MergeError::MissingBaseTensor(name) => {
                assert!(
                    name.contains("k_proj"),
                    "diagnostic must name the missing module, got {name}"
                );
            }
            other => panic!("expected MissingBaseTensor, got {other:?}"),
        }
    }

    #[test]
    fn merge_lora_blend_empty_input_errors() {
        let scratch = ScratchDir::new("merge-empty");
        let base_path = scratch.path().join("base.safetensors");
        let out_path = scratch.path().join("merged.safetensors");
        write_base_safetensors(&base_path, &[("x.weight", &[2usize, 2], vec![0.0_f32; 4])])
            .unwrap();
        let err =
            merge_lora_blend(&base_path, &[], &out_path).expect_err("empty blend must be rejected");
        assert!(matches!(err, MergeError::EmptyBlend));
    }
}
