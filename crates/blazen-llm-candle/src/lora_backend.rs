//! `SimpleBackend` that merges PEFT `LoRA` deltas into base safetensors
//! reads at `VarBuilder` load time.
//!
//! This is the Wave C piece that makes `load_adapter` actually do
//! something on the candle safetensors path. The backend wraps a
//! [`MmapedSafetensors`] handle plus a precomputed table of
//! `tensor_name -> delta` entries. On each [`SimpleBackend::get`] call:
//!
//!   1. Read the base tensor from the wrapped safetensors.
//!   2. If a delta exists for this tensor name, add it elementwise.
//!   3. Otherwise return the base tensor unchanged.
//!
//! Because the merge happens *inside* the backend, model code stays
//! completely unmodified — `Llama::load(vb, ...)` does not need to know
//! whether the underlying `vb` is plain safetensors or merged-with-LoRA.
//! The same `LoraMergingBackend` works for any architecture the
//! safetensors path eventually supports (Llama / Qwen2 / Mistral / …)
//! because every `candle_transformers` model uses `vb.get(...)` for its
//! `Linear` weights and that funnels through the same trait method.
//!
//! Multi-adapter composition is handled by summing deltas during
//! construction — if two adapters both target `q_proj` on the same
//! layer, their `(B@A)*scale` matrices are added before being installed
//! in the table. This matches PEFT's `add_weighted_adapter` semantics
//! and is what Wave A did at the llama.cpp layer.

use std::collections::HashMap;

use candle_core::{DType, Device, Shape, Tensor, safetensors::MmapedSafetensors};
use candle_nn::var_builder::SimpleBackend;

use crate::lora::LoadedAdapter;

/// PEFT-stripped module path prefix needed to reconstruct the
/// safetensors tensor name expected by `candle_transformers::models::llama`.
///
/// PEFT's `adapter_model.safetensors` keys look like
/// `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`;
/// [`crate::lora`] strips the `base_model.model.` envelope, leaving
/// `model.layers.0.self_attn.q_proj` (when the underlying HF model uses
/// the conventional `model.` namespace, which Llama / Qwen2 / Mistral
/// all do). Some PEFT trainers also write the inner `model.` themselves
/// — we accept both shapes and only re-prepend `model.` if it isn't
/// already there.
const MODEL_NAMESPACE: &str = "model.";

/// Reconstruct the full safetensors tensor name for a base `Linear`
/// weight from a PEFT-stripped module path.
///
/// Why: PEFT key stripping yields e.g. `layers.0.self_attn.q_proj`;
/// candle-transformers' Llama loader queries
/// `model.layers.0.self_attn.q_proj.weight`. We append `.weight` and
/// prepend `model.` if missing.
fn full_safetensors_key(module_path: &str) -> String {
    let with_prefix = if module_path.starts_with(MODEL_NAMESPACE)
        || module_path.starts_with("lm_head")
        || module_path.starts_with("embed_tokens")
    {
        module_path.to_string()
    } else {
        format!("{MODEL_NAMESPACE}{module_path}")
    };
    format!("{with_prefix}.weight")
}

/// Compute `(B @ A) * scale` for a single [`crate::lora::LoraLayer`].
///
/// Stored A is `[rank, in_dim]`, stored B is `[out_dim, rank]`, so
/// `B @ A` is `[out_dim, in_dim]` — matching the base `Linear`'s
/// `weight` shape. The result is cast to `target_dtype` so the
/// downstream `broadcast_add` against the base tensor (which lives at
/// the model's runtime dtype, typically f16 / bf16) succeeds without
/// further conversion.
fn compute_delta(
    layer: &crate::lora::LoraLayer,
    device: &Device,
    target_dtype: DType,
) -> candle_core::Result<Tensor> {
    // Why: `Linear::forward` would impose a batch dim; we want the raw
    // matmul `B @ A`. `Linear::weight()` returns the stored tensor
    // directly, no copy.
    let a = layer.a.weight();
    let b = layer.b.weight();
    let a = if a.device().same_device(device) {
        a.clone()
    } else {
        a.to_device(device)?
    };
    let b = if b.device().same_device(device) {
        b.clone()
    } else {
        b.to_device(device)?
    };
    let delta = b.matmul(&a)?;
    let delta = delta.affine(f64::from(layer.scale), 0.0)?;
    if delta.dtype() == target_dtype {
        Ok(delta)
    } else {
        delta.to_dtype(target_dtype)
    }
}

/// A [`SimpleBackend`] that wraps [`MmapedSafetensors`] and applies
/// merged PEFT `LoRA` deltas on the fly during model load.
///
/// Construct via [`LoraMergingBackend::new`]; pass the result through
/// `Box::new` and `VarBuilder::from_backend` to drop it into any
/// candle-transformers model load path.
pub struct LoraMergingBackend {
    base: MmapedSafetensors,
    /// Full safetensors tensor name → summed delta tensor (already cast
    /// to the load dtype). Tensors not present in the base map are
    /// silently dropped at construction time so an adapter that mentions
    /// modules absent from the base never panics the model loader.
    deltas: HashMap<String, Tensor>,
}

impl LoraMergingBackend {
    /// Build a merging backend from a base safetensors handle plus a
    /// list of parsed PEFT adapters.
    ///
    /// `dtype` is the load dtype of the wrapped model — deltas are cast
    /// to this dtype so the per-`get` `broadcast_add` is dtype-clean.
    ///
    /// Adapters targeting modules not present in the base are silently
    /// skipped (the base check is what `MmapedSafetensors::get` would
    /// have raised on anyway, so dropping them early keeps the table
    /// small without changing behavior).
    ///
    /// If multiple adapters target the same base module, the deltas are
    /// summed in-place — matching PEFT's `add_weighted_adapter`.
    ///
    /// # Errors
    ///
    /// Forwards any tensor error from the `B @ A` matmul or dtype cast.
    pub fn new(
        base: MmapedSafetensors,
        adapters: &[LoadedAdapter],
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Self> {
        let mut deltas: HashMap<String, Tensor> = HashMap::new();
        for adapter in adapters {
            for (module_path, layer) in &adapter.layers {
                let tensor_name = full_safetensors_key(module_path);
                if base.get(&tensor_name).is_err() {
                    tracing::debug!(
                        adapter = %adapter.id,
                        module = %module_path,
                        tensor = %tensor_name,
                        "skipping LoRA delta: base tensor not present"
                    );
                    continue;
                }
                let delta = compute_delta(layer, device, dtype)?;
                if let Some(existing) = deltas.get_mut(&tensor_name) {
                    let summed = existing.add(&delta)?;
                    *existing = summed;
                } else {
                    deltas.insert(tensor_name, delta);
                }
            }
        }
        Ok(Self { base, deltas })
    }

    /// Number of base tensors that will receive a merged delta on
    /// `get`. Mostly useful for diagnostics + tests.
    #[must_use]
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }
}

impl SimpleBackend for LoraMergingBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let base_tensor =
            <MmapedSafetensors as SimpleBackend>::get(&self.base, s, name, h, dtype, dev)?;
        let Some(delta) = self.deltas.get(name) else {
            return Ok(base_tensor);
        };
        let delta = if delta.dtype() == base_tensor.dtype() {
            delta.clone()
        } else {
            delta.to_dtype(base_tensor.dtype())?
        };
        let delta = if delta.device().same_device(base_tensor.device()) {
            delta
        } else {
            delta.to_device(base_tensor.device())?
        };
        base_tensor.broadcast_add(&delta)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let base_tensor =
            <MmapedSafetensors as SimpleBackend>::get_unchecked(&self.base, name, dtype, dev)?;
        let Some(delta) = self.deltas.get(name) else {
            return Ok(base_tensor);
        };
        let delta = if delta.dtype() == base_tensor.dtype() {
            delta.clone()
        } else {
            delta.to_dtype(base_tensor.dtype())?
        };
        let delta = if delta.device().same_device(base_tensor.device()) {
            delta
        } else {
            delta.to_device(base_tensor.device())?
        };
        base_tensor.broadcast_add(&delta)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        <MmapedSafetensors as SimpleBackend>::contains_tensor(&self.base, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoraLayer;
    use candle_nn::Linear;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap as StdHashMap;

    /// Write a single-tensor safetensors file under the per-test
    /// scratch directory and return its path.
    fn write_single_tensor_safetensors(
        name: &str,
        rows: usize,
        cols: usize,
        fill: f32,
        scratch_subdir: &str,
    ) -> std::path::PathBuf {
        let dir = std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(".cache/blazen-candle-lora-merge"))
            .expect("HOME env required")
            .join(scratch_subdir);
        std::fs::create_dir_all(&dir).expect("mkdir scratch");
        let path = dir.join(format!("{name}.safetensors"));

        let data: Vec<f32> = vec![fill; rows * cols];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = TensorView::new(safetensors::Dtype::F32, vec![rows, cols], &bytes)
            .expect("tensor view");
        let mut map: StdHashMap<&str, TensorView<'_>> = StdHashMap::new();
        map.insert(name, view);
        let payload = safetensors::serialize(&map, None).expect("serialize");
        std::fs::write(&path, payload).expect("write safetensors");
        path
    }

    fn make_layer(scale: f32, device: &Device) -> LoraLayer {
        // Why: A=[1,2] of 1s, B=[2,1] of 1s → B@A = [[1,1],[1,1]] ;
        // scale baked in via LoraLayer::scale.
        let a = Tensor::from_vec(vec![1.0_f32, 1.0], (1, 2), device).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 1.0], (2, 1), device).unwrap();
        LoraLayer {
            a: Linear::new(a, None),
            b: Linear::new(b, None),
            scale,
        }
    }

    #[test]
    fn empty_adapter_list_acts_as_passthrough() {
        let device = Device::Cpu;
        let path = write_single_tensor_safetensors(
            "model.layers.0.self_attn.q_proj.weight",
            2,
            2,
            7.0,
            "passthrough",
        );
        // Safety: file is immutable for the duration of this test.
        #[allow(unsafe_code)]
        let mmap = unsafe { MmapedSafetensors::new(&path).expect("mmap base") };

        let backend = LoraMergingBackend::new(mmap, &[], &device, DType::F32).expect("backend");
        assert_eq!(backend.delta_count(), 0);

        let t = <LoraMergingBackend as SimpleBackend>::get(
            &backend,
            Shape::from((2, 2)),
            "model.layers.0.self_attn.q_proj.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("get");
        let vals: Vec<Vec<f32>> = t.to_vec2().unwrap();
        assert!(vals.iter().flatten().all(|v| (*v - 7.0).abs() < 1e-6));
    }

    #[test]
    fn single_adapter_adds_delta_to_matching_key_only() {
        let device = Device::Cpu;
        let q_path = write_single_tensor_safetensors(
            "model.layers.0.self_attn.q_proj.weight",
            2,
            2,
            5.0,
            "single_q",
        );
        // Why: separate file so the second `get` against an untouched
        // tensor sees its true base value.
        let k_path = write_single_tensor_safetensors(
            "model.layers.0.self_attn.k_proj.weight",
            2,
            2,
            3.0,
            "single_k",
        );

        #[allow(unsafe_code)]
        let mmap = unsafe { MmapedSafetensors::multi(&[q_path, k_path]).expect("mmap base") };

        let layer = make_layer(2.0, &device);
        let mut layers = HashMap::new();
        layers.insert("layers.0.self_attn.q_proj".to_string(), layer);
        let adapter = LoadedAdapter {
            id: "adapter-a".into(),
            source_dir: std::path::PathBuf::from("."),
            scale: 1.0,
            layers,
        };

        let backend =
            LoraMergingBackend::new(mmap, &[adapter], &device, DType::F32).expect("backend");
        assert_eq!(backend.delta_count(), 1);

        // q_proj: base 5 + delta (B@A=[[1,1],[1,1]]) * scale 2 = [[7,7],[7,7]]
        let q = <LoraMergingBackend as SimpleBackend>::get(
            &backend,
            Shape::from((2, 2)),
            "model.layers.0.self_attn.q_proj.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .unwrap();
        let qv: Vec<Vec<f32>> = q.to_vec2().unwrap();
        for row in &qv {
            for v in row {
                assert!((*v - 7.0).abs() < 1e-5, "q_proj got {v}");
            }
        }

        // k_proj untouched: still 3.0 everywhere.
        let k = <LoraMergingBackend as SimpleBackend>::get(
            &backend,
            Shape::from((2, 2)),
            "model.layers.0.self_attn.k_proj.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .unwrap();
        let kv: Vec<Vec<f32>> = k.to_vec2().unwrap();
        for row in &kv {
            for v in row {
                assert!((*v - 3.0).abs() < 1e-5, "k_proj got {v}");
            }
        }
    }

    #[test]
    fn two_adapters_targeting_same_module_sum_deltas() {
        let device = Device::Cpu;
        let path = write_single_tensor_safetensors(
            "model.layers.0.self_attn.q_proj.weight",
            2,
            2,
            0.0,
            "sum",
        );

        #[allow(unsafe_code)]
        let mmap = unsafe { MmapedSafetensors::new(&path).expect("mmap base") };

        let mut layers_a = HashMap::new();
        layers_a.insert(
            "layers.0.self_attn.q_proj".to_string(),
            make_layer(1.0, &device),
        );
        let adapter_a = LoadedAdapter {
            id: "a".into(),
            source_dir: std::path::PathBuf::from("."),
            scale: 1.0,
            layers: layers_a,
        };

        let mut layers_b = HashMap::new();
        layers_b.insert(
            "layers.0.self_attn.q_proj".to_string(),
            make_layer(3.0, &device),
        );
        let adapter_b = LoadedAdapter {
            id: "b".into(),
            source_dir: std::path::PathBuf::from("."),
            scale: 1.0,
            layers: layers_b,
        };

        let backend = LoraMergingBackend::new(mmap, &[adapter_a, adapter_b], &device, DType::F32)
            .expect("backend");
        assert_eq!(backend.delta_count(), 1);

        // Two B@A=[[1,1],[1,1]] deltas summed with scales 1 and 3 →
        // [[1+3,1+3],[1+3,1+3]] = [[4,4],[4,4]].
        let t = <LoraMergingBackend as SimpleBackend>::get(
            &backend,
            Shape::from((2, 2)),
            "model.layers.0.self_attn.q_proj.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .unwrap();
        let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
        for row in &v {
            for value in row {
                assert!((*value - 4.0).abs() < 1e-5, "got {value}");
            }
        }
    }

    #[test]
    fn adapter_targeting_missing_base_tensor_is_silently_skipped() {
        let device = Device::Cpu;
        let path = write_single_tensor_safetensors(
            "model.layers.0.self_attn.q_proj.weight",
            2,
            2,
            5.0,
            "skip",
        );
        #[allow(unsafe_code)]
        let mmap = unsafe { MmapedSafetensors::new(&path).expect("mmap base") };

        let mut layers = HashMap::new();
        layers.insert(
            "layers.99.self_attn.v_proj".to_string(),
            make_layer(1.0, &device),
        );
        let adapter = LoadedAdapter {
            id: "ghost".into(),
            source_dir: std::path::PathBuf::from("."),
            scale: 1.0,
            layers,
        };

        let backend =
            LoraMergingBackend::new(mmap, &[adapter], &device, DType::F32).expect("backend");
        assert_eq!(backend.delta_count(), 0, "ghost adapter must not register");

        // q_proj still equals base (5.0) because nothing was merged.
        let t = <LoraMergingBackend as SimpleBackend>::get(
            &backend,
            Shape::from((2, 2)),
            "model.layers.0.self_attn.q_proj.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .unwrap();
        let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
        for row in &v {
            for value in row {
                assert!((*value - 5.0).abs() < 1e-5, "got {value}");
            }
        }
    }

    #[test]
    fn full_safetensors_key_handles_both_prefix_shapes() {
        assert_eq!(
            full_safetensors_key("layers.0.self_attn.q_proj"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            full_safetensors_key("model.layers.0.self_attn.q_proj"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(full_safetensors_key("lm_head"), "lm_head.weight");
    }
}
