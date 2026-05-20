//! PEFT-canonical adapter export.
//!
//! Output round-trips through
//! [`blazen_llm_candle::lora::LoadedAdapter::from_dir`]: writes
//! `adapter_config.json` plus `adapter_model.safetensors` with keys shaped
//! `base_model.model.<module_path>.lora_{A,B}.weight`.

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
