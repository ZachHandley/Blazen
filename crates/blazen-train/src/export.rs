//! PEFT-canonical adapter export and full fine-tune safetensors export.
//!
//! [`save_peft_adapter`] writes a PEFT-format adapter directory
//! ([`blazen_llm_candle::lora::LoadedAdapter::from_dir`]-compatible).
//!
//! [`save_full_safetensors`] writes a full model checkpoint
//! (`model.safetensors`, plus an optional `config.json`) for the
//! [`crate::arch::TrainMode::FullFineTune`] training path. v1 only emits
//! a single shard — anything above 2GB returns
//! [`BlazenTrainError::Unsupported`]. Use LoRA training when a model
//! would exceed the cap.

use std::collections::HashMap;
use std::path::Path;

use candle_nn::VarMap;
use serde::Serialize;

use crate::config::LoraConfig;
use crate::error::BlazenTrainError;

/// PEFT key prefix every adapter tensor name is wrapped in. Matches
/// `crates/blazen-llm-candle/src/lora.rs::PEFT_KEY_PREFIX`.
const PEFT_KEY_PREFIX: &str = "base_model.model.";

#[derive(Debug, Serialize)]
struct PeftAdapterConfig<'a> {
    r: usize,
    lora_alpha: f32,
    lora_dropout: f32,
    target_modules: &'a [String],
    base_model_name_or_path: &'a str,
    peft_type: &'static str,
    task_type: &'static str,
    bias: &'static str,
    fan_in_fan_out: bool,
    inference_mode: bool,
}

/// Serialize the trained LoRA params in `varmap` to a PEFT-format adapter
/// directory.
///
/// Filters `varmap` to entries whose names end in `.lora_A.weight` or
/// `.lora_B.weight`, prepends the PEFT `base_model.model.` prefix, and
/// writes both files under `output_dir`. `output_dir` is created if it
/// does not already exist.
///
/// # Errors
///
/// Returns [`BlazenTrainError::Export`] on filesystem errors,
/// [`BlazenTrainError::Serde`] on JSON failures, and
/// [`BlazenTrainError::Export`] on safetensors-serialization failures.
/// Returns [`BlazenTrainError::Export`] if `varmap` contains zero LoRA
/// parameters (almost always indicates the wrapper layer did not register
/// vars against the correct VarBuilder).
pub fn save_peft_adapter(
    varmap: &VarMap,
    output_dir: &Path,
    lora_config: &LoraConfig,
    base_model_repo: &str,
) -> Result<(), BlazenTrainError> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        BlazenTrainError::Export(format!(
            "failed to create adapter dir {}: {e}",
            output_dir.display()
        ))
    })?;

    let cfg = PeftAdapterConfig {
        r: lora_config.rank,
        lora_alpha: lora_config.alpha,
        lora_dropout: lora_config.dropout,
        target_modules: &lora_config.target_modules,
        base_model_name_or_path: base_model_repo,
        peft_type: "LORA",
        task_type: "CAUSAL_LM",
        bias: "none",
        fan_in_fan_out: false,
        inference_mode: true,
    };
    let cfg_bytes = serde_json::to_vec_pretty(&cfg)?;
    let cfg_path = output_dir.join("adapter_config.json");
    std::fs::write(&cfg_path, &cfg_bytes).map_err(|e| {
        BlazenTrainError::Export(format!(
            "failed to write adapter_config.json at {}: {e}",
            cfg_path.display()
        ))
    })?;

    let guard = varmap
        .data()
        .lock()
        .map_err(|e| BlazenTrainError::Export(format!("varmap mutex poisoned: {e}")))?;

    let mut filtered: HashMap<String, candle_core::Tensor> = HashMap::new();
    for (name, var) in guard.iter() {
        if name.ends_with(".lora_A.weight") || name.ends_with(".lora_B.weight") {
            let peft_key = format!("{PEFT_KEY_PREFIX}{name}");
            filtered.insert(peft_key, var.as_tensor().clone());
        }
    }
    drop(guard);

    if filtered.is_empty() {
        return Err(BlazenTrainError::Export(
            "varmap contains no LoRA parameters (lora_A.weight / lora_B.weight) — nothing to export"
                .to_string(),
        ));
    }

    let weights_path = output_dir.join("adapter_model.safetensors");
    safetensors::tensor::serialize_to_file(filtered.iter(), None, &weights_path).map_err(|e| {
        BlazenTrainError::Export(format!(
            "safetensors serialize failed at {}: {e}",
            weights_path.display()
        ))
    })?;

    Ok(())
}

/// Maximum total payload (sum of tensor byte sizes) the v1 single-shard
/// `save_full_safetensors` will write.
///
/// safetensors has no hard upper bound, but transformers-style sharded
/// layouts (`model-00001-of-NNNN.safetensors` + `model.safetensors.index.json`)
/// are how the ecosystem ships multi-GB checkpoints. Writing one giant
/// shard hits `mmap`-on-load size limits on common deploy targets, so
/// this exporter caps single-shard output at 2GB and returns
/// [`BlazenTrainError::Unsupported`] for anything larger. Use LoRA
/// training in that regime.
///
/// Lowered to a small value under `#[cfg(test)]` so unit tests can drive
/// the cap-rejection path without allocating a multi-GB tensor.
#[cfg(not(test))]
const FULL_SAFETENSORS_MAX_BYTES: usize = 2_000_000_000;

#[cfg(test)]
const FULL_SAFETENSORS_MAX_BYTES: usize = 500_000;

/// Serialize every [`candle_nn::Var`] in `varmap` to a single
/// `model.safetensors` under `output_dir`.
///
/// Optionally writes `output_dir/config.json` if `model_config` is
/// provided (pretty-printed JSON, matching the conventions of the HF
/// `transformers` `PreTrainedModel.save_pretrained` layout).
///
/// `output_dir` is created (recursively) if it does not already exist.
///
/// # Errors
///
/// Returns [`BlazenTrainError::Unsupported`] if the total tensor byte
/// payload exceeds [`FULL_SAFETENSORS_MAX_BYTES`] (single-shard
/// limit — multi-shard support is intentionally deferred; use LoRA
/// training for larger models).
///
/// Returns [`BlazenTrainError::Export`] on filesystem / safetensors
/// serialization failures, [`BlazenTrainError::Io`] on directory
/// creation failures, and [`BlazenTrainError::Serde`] on JSON failures
/// while writing `config.json`.
pub fn save_full_safetensors(
    varmap: &candle_nn::VarMap,
    output_dir: &Path,
    model_config: Option<&serde_json::Value>,
) -> Result<std::path::PathBuf, BlazenTrainError> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        BlazenTrainError::Export(format!(
            "failed to create output dir {}: {e}",
            output_dir.display()
        ))
    })?;

    let guard = varmap
        .data()
        .lock()
        .map_err(|e| BlazenTrainError::Export(format!("varmap mutex poisoned: {e}")))?;

    let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::with_capacity(guard.len());
    let mut total_bytes: usize = 0;
    for (name, var) in guard.iter() {
        let t = var.as_tensor();
        total_bytes = total_bytes.saturating_add(t.elem_count() * t.dtype().size_in_bytes());
        tensors.insert(name.clone(), t.clone());
    }
    drop(guard);

    if total_bytes > FULL_SAFETENSORS_MAX_BYTES {
        return Err(BlazenTrainError::Unsupported(format!(
            "model >{FULL_SAFETENSORS_MAX_BYTES} bytes ({total_bytes} bytes); sharding not yet implemented — use LoRA training instead"
        )));
    }

    if tensors.is_empty() {
        return Err(BlazenTrainError::Export(
            "varmap is empty — nothing to save".to_string(),
        ));
    }

    let weights_path = output_dir.join("model.safetensors");
    safetensors::tensor::serialize_to_file(tensors.iter(), None, &weights_path).map_err(|e| {
        BlazenTrainError::Export(format!(
            "safetensors serialize failed at {}: {e}",
            weights_path.display()
        ))
    })?;

    if let Some(cfg) = model_config {
        let cfg_bytes = serde_json::to_vec_pretty(cfg)?;
        let cfg_path = output_dir.join("config.json");
        std::fs::write(&cfg_path, &cfg_bytes).map_err(|e| {
            BlazenTrainError::Export(format!(
                "failed to write config.json at {}: {e}",
                cfg_path.display()
            ))
        })?;
    }

    Ok(output_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Init, VarBuilder};
    use tempfile::TempDir;

    use crate::lora::LoraLinear;

    fn populate_varmap(varmap: &VarMap, device: &Device) {
        // Why: simulate two LoRA-wrapped target modules so the export has
        // multiple A/B pairs to round-trip.
        for module in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ] {
            let vb = VarBuilder::from_varmap(varmap, DType::F32, device).push_prefix(module);
            let base_w = candle_core::Tensor::ones((4, 4), DType::F32, device).unwrap();
            let base = candle_nn::Linear::new(base_w, None);
            let _ = LoraLinear::wrap(base, 4, 4, 2, 4.0, vb).expect("wrap");
        }
        // Why: also register a non-LoRA frozen base var to confirm it is
        // filtered out of the safetensors payload.
        let _ = varmap
            .get(
                (4, 4),
                "model.layers.0.self_attn.q_proj.weight",
                Init::Const(0.0),
                DType::F32,
                device,
            )
            .unwrap();
    }

    #[test]
    fn save_peft_adapter_creates_both_files() {
        let tmp = TempDir::new().unwrap();
        let varmap = VarMap::new();
        populate_varmap(&varmap, &Device::Cpu);

        let cfg = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        };

        save_peft_adapter(&varmap, tmp.path(), &cfg, "test/base").expect("export");
        assert!(tmp.path().join("adapter_config.json").exists());
        assert!(tmp.path().join("adapter_model.safetensors").exists());
    }

    #[test]
    fn export_rejects_empty_varmap() {
        let tmp = TempDir::new().unwrap();
        let varmap = VarMap::new();
        let cfg = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".into()],
        };
        let err = save_peft_adapter(&varmap, tmp.path(), &cfg, "test/base").unwrap_err();
        assert!(matches!(err, BlazenTrainError::Export(_)));
    }

    #[test]
    fn save_full_safetensors_writes_model_safetensors() {
        let tmp = TempDir::new().unwrap();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        // Why: register two trainable weights via VarMap::get so they land
        // as Vars under known names, mirroring how the FFT arch path
        // populates the trainer's varmap.
        let _ = varmap
            .get((4, 4), "w1.weight", Init::Const(0.5), DType::F32, &device)
            .unwrap();
        let _ = varmap
            .get((2,), "w1.bias", Init::Const(0.0), DType::F32, &device)
            .unwrap();

        let returned =
            save_full_safetensors(&varmap, tmp.path(), None).expect("save_full_safetensors");
        assert_eq!(returned, tmp.path());
        let weights = tmp.path().join("model.safetensors");
        assert!(weights.exists(), "model.safetensors should exist");
        let meta = std::fs::metadata(&weights).expect("stat model.safetensors");
        assert!(meta.len() > 0, "model.safetensors should be non-empty");
        // config.json must NOT exist when model_config is None.
        assert!(
            !tmp.path().join("config.json").exists(),
            "config.json should be absent when model_config = None"
        );
    }

    #[test]
    fn save_full_safetensors_writes_config_when_provided() {
        let tmp = TempDir::new().unwrap();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let _ = varmap
            .get((4, 4), "w1.weight", Init::Const(0.0), DType::F32, &device)
            .unwrap();

        let cfg = serde_json::json!({ "hidden_size": 16, "model_type": "qwen2" });
        save_full_safetensors(&varmap, tmp.path(), Some(&cfg)).expect("save with config");

        let cfg_path = tmp.path().join("config.json");
        assert!(cfg_path.exists(), "config.json should exist");
        let cfg_bytes = std::fs::read(&cfg_path).expect("read config.json");
        let parsed: serde_json::Value =
            serde_json::from_slice(&cfg_bytes).expect("parse config.json");
        assert_eq!(parsed["hidden_size"], 16);
        assert_eq!(parsed["model_type"], "qwen2");
    }

    #[test]
    fn save_full_safetensors_rejects_oversized_models() {
        // The #[cfg(test)] limit is 500KB; (512, 512) f32 = 512*512*4 = ~1MB,
        // which exceeds the cap and forces the Unsupported branch.
        let tmp = TempDir::new().unwrap();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let _ = varmap
            .get(
                (512, 512),
                "huge.weight",
                Init::Const(0.0),
                DType::F32,
                &device,
            )
            .unwrap();

        let err = save_full_safetensors(&varmap, tmp.path(), None)
            .expect_err("oversized varmap should be rejected");
        match err {
            BlazenTrainError::Unsupported(msg) => {
                assert!(
                    msg.contains("sharding"),
                    "unexpected Unsupported message: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn save_full_safetensors_rejects_empty_varmap() {
        let tmp = TempDir::new().unwrap();
        let varmap = VarMap::new();
        let err = save_full_safetensors(&varmap, tmp.path(), None)
            .expect_err("empty varmap should be rejected");
        assert!(matches!(err, BlazenTrainError::Export(_)));
    }

    #[test]
    fn saved_adapter_loads_via_inference_side_loader() {
        let tmp = TempDir::new().unwrap();
        let varmap = VarMap::new();
        populate_varmap(&varmap, &Device::Cpu);

        let cfg = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        };
        save_peft_adapter(&varmap, tmp.path(), &cfg, "test/base").expect("export");

        let loaded = blazen_llm_candle::lora::LoadedAdapter::from_dir(
            tmp.path(),
            "round-trip".to_string(),
            1.0,
            &Device::Cpu,
        )
        .expect("inference loader accepts our PEFT layout");

        assert_eq!(loaded.layers.len(), 2);
        assert!(
            loaded
                .layers
                .contains_key("model.layers.0.self_attn.q_proj")
        );
        assert!(
            loaded
                .layers
                .contains_key("model.layers.0.self_attn.v_proj")
        );
    }
}
