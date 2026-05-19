//! PEFT-format `LoRA` adapter loading + per-layer delta math for the candle
//! backend.
//!
//! This module provides the data structures and weight-loading machinery
//! for `LoRA` adapters in the PEFT canonical layout (`adapter_config.json`
//! + `adapter_model.safetensors`). Each per-target-module pair of
//!   low-rank matrices is stored as a [`LoraLayer`] whose [`forward`]
//!   computes the standard PEFT delta `scale * b(a(x))` where
//!   `scale = alpha / rank` × caller-supplied multiplier.
//!
//! The structures here are intentionally model-agnostic. Wiring the
//! per-layer delta into a base model's forward pass requires a
//! model-specific wrapper because `candle_transformers` does not expose
//! per-`Linear` forward hooks on its built-in model structs. The current
//! provider runs `candle_transformers::models::quantized_llama::ModelWeights`
//! (a monolithic GGUF model) and therefore cannot consume these layers
//! in-place; the adapter parsing + state-table machinery is exposed
//! nonetheless so future per-architecture wrappers (and external
//! consumers) can apply the deltas correctly.
//!
//! [`forward`]: LoraLayer::forward

#![cfg(feature = "engine")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use serde::{Deserialize, Serialize};

use crate::CandleLlmError;

// ---------------------------------------------------------------------------
// Adapter config (subset of PEFT's adapter_config.json)
// ---------------------------------------------------------------------------

/// Subset of PEFT's `adapter_config.json` schema that this loader needs.
///
/// PEFT writes many additional keys (`peft_type`, `task_type`,
/// `bias`, `inference_mode`, etc.) that are accepted-and-ignored here via
/// `serde`'s default unknown-field behavior. Required keys are `r`,
/// `lora_alpha`, and `target_modules`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapterConfig {
    /// Low-rank dimension (the "r" in `LoRA`).
    pub r: usize,
    /// Scaling numerator. Effective adapter scale is `lora_alpha / r`.
    pub lora_alpha: f32,
    /// Module name suffixes that `LoRA` was trained against
    /// (e.g. `["q_proj", "v_proj"]`).
    pub target_modules: Vec<String>,
    /// Base model the adapter was trained against. Recorded for
    /// diagnostics only — this loader does not validate compatibility.
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
}

impl LoraAdapterConfig {
    /// Parse `adapter_config.json` from `adapter_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::InvalidOptions`] if the file is missing,
    /// unreadable, or fails JSON deserialization.
    pub fn from_dir(adapter_dir: &Path) -> Result<Self, CandleLlmError> {
        let path = adapter_dir.join("adapter_config.json");
        let bytes = std::fs::read(&path).map_err(|e| {
            CandleLlmError::InvalidOptions(format!(
                "adapter_config.json read failed at {}: {e}",
                path.display()
            ))
        })?;
        serde_json::from_slice::<Self>(&bytes).map_err(|e| {
            CandleLlmError::InvalidOptions(format!(
                "adapter_config.json parse failed at {}: {e}",
                path.display()
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// LoraLayer: the per-target-module low-rank pair
// ---------------------------------------------------------------------------

/// A single `LoRA` delta layer for one target `Linear` in the base model.
///
/// The delta applied on top of the base linear's output is
/// `effective_scale * b(a(x))`, where `effective_scale` is the product of
/// the PEFT-canonical `lora_alpha / r` and any caller-supplied runtime
/// scale multiplier (the `scale` field on this struct already folds in
/// both factors).
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// The "A" projection (`input_dim` → r). No bias.
    pub a: Linear,
    /// The "B" projection (r → `output_dim`). No bias.
    pub b: Linear,
    /// Effective scaling factor, already `(lora_alpha / r) * runtime_scale`.
    pub scale: f32,
}

impl LoraLayer {
    /// Compute the `LoRA` delta for input `x`. Returns
    /// `scale * b(a(x))` shaped like the base linear's output.
    ///
    /// # Errors
    ///
    /// Forwards any tensor error from candle (dtype/shape mismatch).
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        use candle_core::Module;
        let down = self.a.forward(x)?;
        let up = self.b.forward(&down)?;
        // Why: f64 cast keeps the scalar multiply consistent with candle's
        // broadcast helpers, which take f64 even though weights are f32/f16.
        up.affine(f64::from(self.scale), 0.0)
    }
}

// ---------------------------------------------------------------------------
// LoadedAdapter: a full adapter's layer table + bookkeeping
// ---------------------------------------------------------------------------

/// Fully-loaded `LoRA` adapter: parsed config plus per-module
/// [`LoraLayer`] table keyed by the stripped module path
/// (e.g. `"model.layers.0.self_attn.q_proj"`).
pub struct LoadedAdapter {
    /// Caller-chosen identifier.
    pub id: String,
    /// Directory the adapter was loaded from.
    pub source_dir: PathBuf,
    /// Caller-supplied runtime scale multiplier (the `LoRA` `alpha/r`
    /// is already folded into each [`LoraLayer::scale`]). Preserved
    /// here for diagnostics / `list_adapters` reporting.
    pub scale: f32,
    /// Map of stripped module path → low-rank delta. The keys match
    /// what a caller would name target Linear modules
    /// (e.g. `model.layers.0.self_attn.q_proj`); the
    /// `base_model.model.` prefix that PEFT writes into the
    /// safetensors keys has already been stripped.
    pub layers: HashMap<String, LoraLayer>,
}

impl LoadedAdapter {
    /// Load + parse a PEFT-format adapter directory.
    ///
    /// `adapter_dir` must contain `adapter_config.json` and
    /// `adapter_model.safetensors`. `runtime_scale` is the caller's
    /// runtime multiplier (PEFT convention: `1.0` = full strength);
    /// this is multiplied into the canonical `lora_alpha / r` before
    /// being baked into each layer.
    ///
    /// # Errors
    ///
    /// Returns [`CandleLlmError::InvalidOptions`] for missing files or
    /// malformed config, [`CandleLlmError::ModelLoad`] for safetensors
    /// I/O or tensor-construction failures.
    pub fn from_dir(
        adapter_dir: &Path,
        adapter_id: String,
        runtime_scale: f32,
        device: &Device,
    ) -> Result<Self, CandleLlmError> {
        let config = LoraAdapterConfig::from_dir(adapter_dir)?;

        if config.r == 0 {
            return Err(CandleLlmError::InvalidOptions(
                "adapter_config.json: r must be > 0".into(),
            ));
        }

        #[allow(clippy::cast_precision_loss)]
        let alpha_over_r = config.lora_alpha / config.r as f32;
        let effective_scale = alpha_over_r * runtime_scale;

        let weights_path = adapter_dir.join("adapter_model.safetensors");
        if !weights_path.exists() {
            return Err(CandleLlmError::InvalidOptions(format!(
                "adapter_model.safetensors not found at {}",
                weights_path.display()
            )));
        }

        let tensors = candle_core::safetensors::load(&weights_path, device).map_err(|e| {
            CandleLlmError::ModelLoad(format!(
                "safetensors load failed at {}: {e}",
                weights_path.display()
            ))
        })?;

        let layers = build_layers(&tensors, effective_scale)?;

        Ok(Self {
            id: adapter_id,
            source_dir: adapter_dir.to_path_buf(),
            scale: runtime_scale,
            layers,
        })
    }
}

// ---------------------------------------------------------------------------
// Safetensors → LoraLayer table construction
// ---------------------------------------------------------------------------

/// PEFT key prefix that wraps every adapter weight name. Stripped to
/// produce the module path used as the side-table key.
const PEFT_KEY_PREFIX: &str = "base_model.model.";

/// Convert a flat `HashMap<key, Tensor>` from a PEFT safetensors file
/// into a per-module `HashMap<module_path, LoraLayer>` table.
///
/// PEFT keys follow the convention
/// `base_model.model.<module_path>.lora_<A|B>.weight`. This function
/// strips the `base_model.model.` prefix, splits keys on the
/// `.lora_A.weight` / `.lora_B.weight` suffixes, and pairs the two
/// halves per module path.
///
/// # Errors
///
/// Returns [`CandleLlmError::ModelLoad`] if a key has the `LoRA`-A half
/// but is missing the `LoRA`-B half (or vice versa), if a key fails to
/// match the expected pattern, or if `Linear::new` rejects a tensor.
fn build_layers(
    tensors: &HashMap<String, Tensor>,
    effective_scale: f32,
) -> Result<HashMap<String, LoraLayer>, CandleLlmError> {
    let mut a_map: HashMap<String, Tensor> = HashMap::new();
    let mut b_map: HashMap<String, Tensor> = HashMap::new();

    for (raw_key, tensor) in tensors {
        let key = raw_key.strip_prefix(PEFT_KEY_PREFIX).unwrap_or(raw_key);

        if let Some(module) = key.strip_suffix(".lora_A.weight") {
            a_map.insert(module.to_string(), tensor.clone());
        } else if let Some(module) = key.strip_suffix(".lora_B.weight") {
            b_map.insert(module.to_string(), tensor.clone());
        } else if key.contains(".lora_A.") || key.contains(".lora_B.") {
            return Err(CandleLlmError::ModelLoad(format!(
                "unexpected PEFT key shape (no .weight terminator): {raw_key}"
            )));
        }
        // Why: silently ignore non-`LoRA` tensors (e.g. embedding
        // resizing tables some PEFT trainers ship alongside) — they
        // are not consumed by the delta path.
    }

    let mut layers = HashMap::with_capacity(a_map.len());

    for (module, a_weight) in a_map {
        let b_weight = b_map.remove(&module).ok_or_else(|| {
            CandleLlmError::ModelLoad(format!(
                "PEFT adapter has lora_A but no matching lora_B for module '{module}'"
            ))
        })?;

        let a_weight = ensure_f32(&a_weight).map_err(|e| {
            CandleLlmError::ModelLoad(format!(
                "`LoRA` A weight dtype cast failed for {module}: {e}"
            ))
        })?;
        let b_weight = ensure_f32(&b_weight).map_err(|e| {
            CandleLlmError::ModelLoad(format!(
                "`LoRA` B weight dtype cast failed for {module}: {e}"
            ))
        })?;

        layers.insert(
            module.clone(),
            LoraLayer {
                a: Linear::new(a_weight, None),
                b: Linear::new(b_weight, None),
                scale: effective_scale,
            },
        );
    }

    if !b_map.is_empty() {
        let stragglers: Vec<&String> = b_map.keys().collect();
        return Err(CandleLlmError::ModelLoad(format!(
            "PEFT adapter has lora_B but no matching lora_A for modules: {stragglers:?}"
        )));
    }

    Ok(layers)
}

/// Cast a tensor to f32 if it isn't already. PEFT adapters are commonly
/// shipped in bf16/f16; the delta math is computed in f32 to avoid
/// silent precision-mismatch errors when the base model runs a
/// different dtype.
fn ensure_f32(t: &Tensor) -> candle_core::Result<Tensor> {
    if t.dtype() == DType::F32 {
        Ok(t.clone())
    } else {
        t.to_dtype(DType::F32)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn write_minimal_config(dir: &Path, r: usize, alpha: f32, modules: &[&str]) {
        let cfg = serde_json::json!({
            "r": r,
            "lora_alpha": alpha,
            "target_modules": modules,
            "base_model_name_or_path": "test/base",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        });
        std::fs::write(
            dir.join("adapter_config.json"),
            serde_json::to_vec(&cfg).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn config_parses_required_fields() {
        let tmp = std::env::temp_dir().join("blazen_candle_lora_cfg_test");
        std::fs::create_dir_all(&tmp).unwrap();
        write_minimal_config(&tmp, 8, 16.0, &["q_proj", "v_proj"]);

        let cfg = LoraAdapterConfig::from_dir(&tmp).expect("config parses");
        assert_eq!(cfg.r, 8);
        assert!((cfg.lora_alpha - 16.0).abs() < f32::EPSILON);
        assert_eq!(cfg.target_modules, vec!["q_proj", "v_proj"]);
        assert_eq!(cfg.base_model_name_or_path.as_deref(), Some("test/base"));

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn config_missing_file_returns_error() {
        let tmp = std::env::temp_dir().join("blazen_candle_lora_missing_test");
        std::fs::create_dir_all(&tmp).unwrap();
        // Why: ensure no stale config from a previous run sneaks in.
        std::fs::remove_file(tmp.join("adapter_config.json")).ok();

        let err = LoraAdapterConfig::from_dir(&tmp).unwrap_err();
        assert!(
            matches!(err, CandleLlmError::InvalidOptions(_)),
            "expected InvalidOptions, got {err:?}"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn lora_layer_forward_applies_scale() {
        let device = Device::Cpu;
        // Why: tiny 2→r=1→2 layer so the delta is hand-computable.
        let a = Tensor::from_vec(vec![1.0_f32, 0.0_f32], (1, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![2.0_f32, 0.0_f32], (2, 1), &device).unwrap();
        let layer = LoraLayer {
            a: Linear::new(a, None),
            b: Linear::new(b, None),
            scale: 0.5,
        };

        let x = Tensor::from_vec(vec![3.0_f32, 0.0_f32], (1, 2), &device).unwrap();
        let out = layer.forward(&x).unwrap();
        let out_vec: Vec<Vec<f32>> = out.to_vec2().unwrap();
        // a(x) = [3], b(a(x)) = [6, 0], * scale 0.5 = [3, 0]
        assert!((out_vec[0][0] - 3.0).abs() < 1e-5);
        assert!(out_vec[0][1].abs() < 1e-5);
    }

    #[test]
    fn build_layers_pairs_ab_and_strips_prefix() {
        let device = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".into(),
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap(),
        );
        tensors.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap(),
        );

        let layers = build_layers(&tensors, 0.25).expect("layers build");
        assert_eq!(layers.len(), 1);
        let key = "layers.0.self_attn.q_proj";
        let layer = layers.get(key).expect("module key present");
        assert!((layer.scale - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn build_layers_errors_on_missing_b() {
        let device = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "base_model.model.q_proj.lora_A.weight".into(),
            Tensor::from_vec(vec![0.0_f32, 0.0], (1, 2), &device).unwrap(),
        );

        let err = build_layers(&tensors, 1.0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no matching lora_B"), "got: {msg}");
    }

    #[test]
    fn build_layers_errors_on_missing_a() {
        let device = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "base_model.model.q_proj.lora_B.weight".into(),
            Tensor::from_vec(vec![0.0_f32, 0.0], (2, 1), &device).unwrap(),
        );

        let err = build_layers(&tensors, 1.0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no matching lora_A"), "got: {msg}");
    }
}
