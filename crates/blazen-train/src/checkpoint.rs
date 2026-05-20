//! Training-run checkpoints.
//!
//! A checkpoint is the pair `(VarMap snapshot, training metadata)` written
//! to `output_dir/step_<N>/`:
//!
//! * `varmap.safetensors` — full [`VarMap`] state (LoRA + any other
//!   registered vars), via [`VarMap::save`].
//! * `metadata.json` — `{ "global_step": N, "config": <TrainConfig> }`.
//!
//! Known limitation (PR7): optimizer state — the AdamW first/second
//! moments — is **not** persisted. `candle_nn::AdamW` does not expose its
//! internal state for serialization, so resuming from a checkpoint
//! reinitializes the moments from zero. Loss/LR continuity will be
//! perfect (varmap + step are exact), but the first few hundred steps
//! after resume effectively re-warm the optimizer. PR8 may revisit this
//! if a moment-snapshot hook lands upstream.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use candle_core::Device;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::config::TrainConfig;
use crate::error::BlazenTrainError;

/// In-memory checkpoint payload returned by [`load_checkpoint`].
pub struct Checkpoint {
    /// Optimizer step the checkpoint was taken at.
    pub global_step: usize,
    /// Restored varmap (vars present in the metadata are populated;
    /// see [`VarMap::load`] semantics).
    pub varmap: VarMap,
}

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointMetadata {
    global_step: usize,
    config: TrainConfig,
    /// Names of every var in the saved varmap. Used at load time to
    /// preregister entries so [`VarMap::load`] (which only updates
    /// already-present keys) actually populates them.
    var_names: Vec<VarEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VarEntry {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

fn dtype_from_label(label: &str) -> Result<candle_core::DType, BlazenTrainError> {
    candle_core::DType::from_str(label).map_err(|e| {
        BlazenTrainError::Checkpoint(format!("unknown dtype label in checkpoint metadata: {e}"))
    })
}

fn step_dir(output_dir: &Path, step: usize) -> PathBuf {
    output_dir.join(format!("step_{step}"))
}

/// Persist a checkpoint to `output_dir/step_<step>/`.
///
/// # Errors
///
/// Returns [`BlazenTrainError::Checkpoint`] on filesystem failures,
/// [`BlazenTrainError::Serde`] on metadata JSON failures, and forwards
/// candle errors from [`VarMap::save`].
pub fn save_checkpoint(
    output_dir: &Path,
    step: usize,
    varmap: &VarMap,
    config: &TrainConfig,
) -> Result<(), BlazenTrainError> {
    let dir = step_dir(output_dir, step);
    std::fs::create_dir_all(&dir).map_err(|e| {
        BlazenTrainError::Checkpoint(format!(
            "failed to create checkpoint dir {}: {e}",
            dir.display()
        ))
    })?;

    let var_names = {
        let guard = varmap
            .data()
            .lock()
            .map_err(|e| BlazenTrainError::Checkpoint(format!("varmap mutex poisoned: {e}")))?;
        let mut entries: Vec<VarEntry> = guard
            .iter()
            .map(|(name, var)| {
                let t = var.as_tensor();
                VarEntry {
                    name: name.clone(),
                    shape: t.dims().to_vec(),
                    dtype: t.dtype().as_str().to_string(),
                }
            })
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    };

    if var_names.is_empty() {
        return Err(BlazenTrainError::Checkpoint(
            "varmap is empty — refusing to write an empty checkpoint".to_string(),
        ));
    }

    varmap
        .save(dir.join("varmap.safetensors"))
        .map_err(|e| BlazenTrainError::Checkpoint(format!("VarMap::save failed: {e}")))?;

    let meta = CheckpointMetadata {
        global_step: step,
        config: config.clone(),
        var_names,
    };
    let meta_bytes = serde_json::to_vec_pretty(&meta)?;
    std::fs::write(dir.join("metadata.json"), meta_bytes)
        .map_err(|e| BlazenTrainError::Checkpoint(format!("failed to write metadata.json: {e}")))?;

    Ok(())
}

/// Load a checkpoint directory written by [`save_checkpoint`].
///
/// `input_dir` is the per-step directory (e.g. `.../step_200/`), not the
/// parent output dir. Returns the restored varmap + step counter along
/// with the original training config so callers can reconstruct the
/// [`crate::Trainer`] with matching hyperparameters.
///
/// # Errors
///
/// Returns [`BlazenTrainError::Checkpoint`] if `metadata.json` or
/// `varmap.safetensors` is missing or fails to parse / load.
pub fn load_checkpoint(
    input_dir: &Path,
    device: &Device,
) -> Result<(Checkpoint, TrainConfig), BlazenTrainError> {
    let meta_path = input_dir.join("metadata.json");
    let weights_path = input_dir.join("varmap.safetensors");

    if !meta_path.exists() {
        return Err(BlazenTrainError::Checkpoint(format!(
            "metadata.json missing in checkpoint dir {}",
            input_dir.display()
        )));
    }
    if !weights_path.exists() {
        return Err(BlazenTrainError::Checkpoint(format!(
            "varmap.safetensors missing in checkpoint dir {}",
            input_dir.display()
        )));
    }

    let meta_bytes = std::fs::read(&meta_path).map_err(|e| {
        BlazenTrainError::Checkpoint(format!(
            "failed to read metadata.json at {}: {e}",
            meta_path.display()
        ))
    })?;
    let meta: CheckpointMetadata = serde_json::from_slice(&meta_bytes)?;

    let mut varmap = VarMap::new();
    // Why: VarMap::load only refreshes vars already present in the map, so
    // we have to pre-register one entry per saved tensor using the shapes
    // + dtypes recorded in metadata.json. Init::Const(0.0) is a throwaway
    // — every cell is overwritten by the safetensors payload on the next
    // line.
    for entry in &meta.var_names {
        let dt = dtype_from_label(&entry.dtype)?;
        let shape: Vec<usize> = entry.shape.clone();
        varmap
            .get(shape, &entry.name, candle_nn::Init::Const(0.0), dt, device)
            .map_err(|e| {
                BlazenTrainError::Checkpoint(format!("preregister var {} failed: {e}", entry.name))
            })?;
    }

    varmap
        .load(&weights_path)
        .map_err(|e| BlazenTrainError::Checkpoint(format!("VarMap::load failed: {e}")))?;

    Ok((
        Checkpoint {
            global_step: meta.global_step,
            varmap,
        },
        meta.config,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{Init, VarBuilder};
    use tempfile::TempDir;

    use crate::lora::LoraLinear;

    fn populate(varmap: &VarMap, device: &Device) {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device)
            .push_prefix("model.layers.0.self_attn.q_proj");
        let base_w = candle_core::Tensor::ones((4, 4), DType::F32, device).unwrap();
        let base = candle_nn::Linear::new(base_w, None);
        let _ = LoraLinear::wrap(base, 4, 4, 2, 4.0, vb).expect("wrap");
        let _ = varmap
            .get(
                (4, 4),
                "model.layers.0.self_attn.q_proj.weight",
                Init::Const(0.5),
                DType::F32,
                device,
            )
            .unwrap();
    }

    #[test]
    fn checkpoint_roundtrip_preserves_varmap_and_step() {
        let tmp = TempDir::new().unwrap();
        let device = Device::Cpu;

        let varmap = VarMap::new();
        populate(&varmap, &device);

        let cfg = TrainConfig::default();
        save_checkpoint(tmp.path(), 200, &varmap, &cfg).expect("save");

        let step_path = tmp.path().join("step_200");
        let (cp, restored_cfg) = load_checkpoint(&step_path, &device).expect("load");
        assert_eq!(cp.global_step, 200);
        assert_eq!(restored_cfg.max_steps, cfg.max_steps);

        let original_names: Vec<String> = {
            let g = varmap.data().lock().unwrap();
            let mut v: Vec<String> = g.keys().cloned().collect();
            v.sort();
            v
        };
        let restored_names: Vec<String> = {
            let g = cp.varmap.data().lock().unwrap();
            let mut v: Vec<String> = g.keys().cloned().collect();
            v.sort();
            v
        };
        assert_eq!(original_names, restored_names);

        let original_frozen = {
            let g = varmap.data().lock().unwrap();
            g.get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .as_tensor()
                .to_vec2::<f32>()
                .unwrap()
        };
        let restored_frozen = {
            let g = cp.varmap.data().lock().unwrap();
            g.get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .as_tensor()
                .to_vec2::<f32>()
                .unwrap()
        };
        assert_eq!(original_frozen, restored_frozen);
    }

    #[test]
    fn checkpoint_load_rejects_missing_metadata() {
        let tmp = TempDir::new().unwrap();
        match load_checkpoint(tmp.path(), &Device::Cpu) {
            Err(BlazenTrainError::Checkpoint(msg)) => assert!(msg.contains("metadata.json")),
            Err(other) => panic!("expected Checkpoint error, got {other}"),
            Ok(_) => panic!("expected error on empty dir"),
        }
    }

    #[test]
    fn save_checkpoint_rejects_empty_varmap() {
        let tmp = TempDir::new().unwrap();
        let varmap = VarMap::new();
        let cfg = TrainConfig::default();
        let err = save_checkpoint(tmp.path(), 0, &varmap, &cfg).unwrap_err();
        assert!(matches!(err, BlazenTrainError::Checkpoint(_)));
    }
}
