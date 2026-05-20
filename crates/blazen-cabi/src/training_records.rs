//! C-ABI mirrors for [`blazen_train`] configuration + result types and the
//! opaque [`BlazenJsonlDataset`] handle consumed by
//! [`crate::manager::blazen_model_manager_train_lora_blocking`] (and its
//! future-returning sibling).
//!
//! ## Layout strategy
//!
//! Config records are flat `#[repr(C)]` aggregates that the Ruby wrapper (or
//! any future cabi consumer) populates on the stack and hands across as
//! `*const BlazenTrainConfig`. The cabi side copies every string and the
//! `target_modules` array on entry, so callers may free their backing
//! storage as soon as the call returns. Nullable scalar slots — `eval_steps`,
//! `save_steps`, `gradient_clip` — use a sibling `has_*` byte plus the
//! payload value, matching the discriminated-record convention already in
//! `manager_records.rs`.
//!
//! ## Result + dataset handles
//!
//! [`BlazenTrainedAdapter`] is a small flat record returned by-value through
//! an out-param; its single owned string (`adapter_dir`) is released by
//! [`blazen_trained_adapter_free`]. [`BlazenJsonlDataset`] is opaque (the
//! inner `Arc<JsonlDataset>` carries the loaded tensors + tokenizer) and is
//! constructed via [`blazen_jsonl_dataset_from_path`] / freed via
//! [`blazen_jsonl_dataset_free`].
//!
//! Why: Ruby progress callback uses Fiber.scheduler-aware polling, deferred
//! to a follow-up. Sync/async without progress is the v1 surface.

#![allow(dead_code)]

use std::ffi::c_char;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "training")]
use blazen_train::dataset::{
    JsonlDataset as InnerJsonlDataset, PreferenceJsonlDataset as InnerPreferenceJsonlDataset,
    RatedJsonlDataset as InnerRatedJsonlDataset,
};
#[cfg(feature = "training")]
use blazen_train::{
    DpoConfig as InnerDpoConfig, FullFineTuneConfig as InnerFullFineTuneConfig,
    FullFineTuneResult as InnerFullFineTuneResult, KtoConfig as InnerKtoConfig,
    LoraConfig as InnerLoraConfig, MixedPrecision as InnerMixedPrecision,
    OptimConfig as InnerOptimConfig, OrpoConfig as InnerOrpoConfig,
    SchedulerConfig as InnerSchedulerConfig, SchedulerKind as InnerSchedulerKind,
    SimpoConfig as InnerSimpoConfig, TrainConfig as InnerTrainConfig,
    TrainCoreConfig as InnerTrainCoreConfig,
};
#[cfg(feature = "training")]
use blazen_uniffi::errors::BlazenError as InnerError;
#[cfg(feature = "training")]
use tokenizers::Tokenizer;

#[cfg(feature = "training")]
use crate::error::BlazenError;
#[cfg(feature = "training")]
use crate::string::{alloc_cstring, cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Tag constants
// ---------------------------------------------------------------------------

/// LR scheduler tag — peak LR stays flat after warmup.
pub const BLAZEN_SCHEDULER_CONSTANT: i32 = 0;
/// LR scheduler tag — linear decay from peak to zero.
pub const BLAZEN_SCHEDULER_LINEAR: i32 = 1;
/// LR scheduler tag — half-cosine decay from peak to zero.
pub const BLAZEN_SCHEDULER_COSINE: i32 = 2;

/// Mixed-precision tag — full fp32 forward/backward.
pub const BLAZEN_MIXED_PRECISION_NONE: i32 = 0;
/// Mixed-precision tag — bf16 forward/backward, fp32 master weights.
pub const BLAZEN_MIXED_PRECISION_BF16: i32 = 1;

// ---------------------------------------------------------------------------
// BlazenLoraConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::LoraConfig`].
///
/// `target_modules` is a null-terminated-or-`_len`-bounded array of NUL-
/// terminated UTF-8 C strings (e.g. `"q_proj"`, `"k_proj"`). The cabi clones
/// every string before [`blazen_model_manager_train_lora_blocking`] returns,
/// so the caller may release the source storage immediately after the call.
#[repr(C)]
pub struct BlazenLoraConfig {
    /// Low-rank dimension (PEFT "r"). Must be > 0.
    pub rank: u32,
    /// Scaling numerator. Effective per-layer scale is `alpha / rank`.
    pub alpha: f32,
    /// Dropout probability applied to LoRA-A input. Must be in `[0.0, 1.0)`.
    pub dropout: f32,
    /// Pointer to an array of `target_modules_len` C strings. Borrowed only
    /// for the duration of the train call.
    pub target_modules: *const *const c_char,
    /// Number of entries in `target_modules`.
    pub target_modules_len: usize,
}

// ---------------------------------------------------------------------------
// BlazenOptimConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::OptimConfig`].
///
/// `gradient_clip` is wrapped with a discriminator byte: set
/// `has_gradient_clip = 1` and populate `gradient_clip` to enable global L2
/// clipping; set `has_gradient_clip = 0` to disable.
#[repr(C)]
pub struct BlazenOptimConfig {
    /// Peak learning rate (applied at end of warmup).
    pub learning_rate: f64,
    /// `AdamW` beta1.
    pub beta1: f64,
    /// `AdamW` beta2.
    pub beta2: f64,
    /// `AdamW` numerical-stability epsilon.
    pub epsilon: f64,
    /// `AdamW` weight decay (decoupled).
    pub weight_decay: f64,
    /// 0 = `None`, 1 = `Some(gradient_clip)`.
    pub has_gradient_clip: i32,
    /// Global L2 clip value when `has_gradient_clip == 1`.
    pub gradient_clip: f32,
}

// ---------------------------------------------------------------------------
// BlazenSchedulerConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::SchedulerConfig`].
#[repr(C)]
pub struct BlazenSchedulerConfig {
    /// One of `BLAZEN_SCHEDULER_*`.
    pub kind: i32,
    /// Linear-warmup duration in optimizer steps.
    pub warmup_steps: u32,
}

// ---------------------------------------------------------------------------
// BlazenTrainConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::TrainConfig`].
///
/// String pointers (`base_model_repo`, `output_dir`, `device`) and the
/// `target_modules` array inside `lora` are borrowed only for the call.
/// Nullable scalar slots use the `has_*` discriminator convention shared by
/// `BlazenOptimConfig::has_gradient_clip`.
#[repr(C)]
pub struct BlazenTrainConfig {
    /// Hugging Face repo id of the base model. Required (non-null).
    pub base_model_repo: *const c_char,
    /// Filesystem directory where the adapter + checkpoints land. Required.
    pub output_dir: *const c_char,
    /// `LoRA` hyperparameters.
    pub lora: BlazenLoraConfig,
    /// `AdamW` hyperparameters.
    pub optim: BlazenOptimConfig,
    /// LR-schedule configuration.
    pub scheduler: BlazenSchedulerConfig,
    /// Total optimizer steps to run. Must be > 0.
    pub max_steps: u32,
    /// Micro-batch size per forward pass. Must be > 0.
    pub batch_size: u32,
    /// Micro-batches to accumulate before each optimizer step. Must be > 0.
    pub gradient_accumulation_steps: u32,
    /// Max tokenized sequence length per example. Must be > 0.
    pub max_seq_len: u32,
    /// 0 = `None`, 1 = `Some(eval_steps)`.
    pub has_eval_steps: i32,
    /// Run evaluation every N steps when `has_eval_steps == 1`.
    pub eval_steps: u32,
    /// 0 = `None`, 1 = `Some(save_steps)`.
    pub has_save_steps: i32,
    /// Write a checkpoint every N steps when `has_save_steps == 1`.
    pub save_steps: u32,
    /// RNG seed (controls dataset shuffling + `LoRA` `A` init).
    pub seed: u64,
    /// One of `BLAZEN_MIXED_PRECISION_*`.
    pub mixed_precision: i32,
    /// Device specifier (`"cpu"`, `"cuda:0"`, `"metal"`). Null = `"cpu"`.
    pub device: *const c_char,
}

// ---------------------------------------------------------------------------
// BlazenTrainedAdapter
// ---------------------------------------------------------------------------

/// `#[repr(C)]` flat record returned by-value from
/// [`crate::manager::blazen_model_manager_train_lora_blocking`] (via an
/// out-param) and from [`crate::future::blazen_future_take_trained_adapter`].
///
/// `adapter_dir` is a caller-owned heap C string; release the whole record
/// (which frees the string too) with [`blazen_trained_adapter_free`].
#[repr(C)]
pub struct BlazenTrainedAdapter {
    /// PEFT-layout adapter directory (caller-owned C string). Free with
    /// [`blazen_trained_adapter_free`].
    pub adapter_dir: *mut c_char,
    /// Final training loss reported by the trainer.
    pub final_loss: f32,
    /// Total optimizer steps actually executed.
    pub total_steps: u64,
}

/// Releases the heap C string carried by `adapter`. The struct itself is
/// caller-owned (typically a stack value populated by the train verb), so
/// this just zeroes out the pointer slot after freeing.
///
/// # Safety
///
/// `adapter` must be null OR a pointer to a valid [`BlazenTrainedAdapter`]
/// whose `adapter_dir` field is null OR was previously produced by the cabi
/// surface. Double-free on the same `adapter_dir` is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_trained_adapter_free(adapter: *mut BlazenTrainedAdapter) {
    if adapter.is_null() {
        return;
    }
    // SAFETY: caller upholds the pointer contract per the per-fn docs.
    let a = unsafe { &mut *adapter };
    if !a.adapter_dir.is_null() {
        // SAFETY: out-param contract — adapter_dir was produced by alloc_cstring
        // (which uses CString::into_raw), so blazen_string_free reclaims it on
        // the matching allocator.
        unsafe {
            crate::string::blazen_string_free(a.adapter_dir);
        }
        a.adapter_dir = std::ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// BlazenJsonlDataset
// ---------------------------------------------------------------------------

/// Opaque handle wrapping `Arc<blazen_train::dataset::JsonlDataset>`. The
/// `Arc` lets one dataset be reused across multiple `train_lora` calls (each
/// call clones the handle's Arc when it builds the trainer-side adapter).
#[cfg(feature = "training")]
pub struct BlazenJsonlDataset {
    pub(crate) inner: Arc<InnerJsonlDataset>,
}

#[cfg(feature = "training")]
impl BlazenJsonlDataset {
    pub(crate) fn into_ptr(self) -> *mut BlazenJsonlDataset {
        Box::into_raw(Box::new(self))
    }
}

/// Loads a JSONL training file with the tokenizer at `tokenizer_path` and
/// returns a caller-owned [`BlazenJsonlDataset`]. Free with
/// [`blazen_jsonl_dataset_free`].
///
/// Returns null on failure and writes `*out_err` (validation /
/// tokenizer-load / dataset-parse errors).
///
/// # Safety
///
/// `path` and `tokenizer_path` must be NUL-terminated UTF-8 buffers valid for
/// the call. `chat_template` and `device` are null OR same contract.
/// `out_err` is null OR a single-writer destination.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_jsonl_dataset_from_path(
    path: *const c_char,
    tokenizer_path: *const c_char,
    chat_template: *const c_char,
    max_seq_len: u32,
    device: *const c_char,
    pad_token_id: u32,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenJsonlDataset {
    // SAFETY: caller upholds the NUL + lifetime contracts on the input strings.
    let Some(path) = (unsafe { cstr_to_str(path) }).map(PathBuf::from) else {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_jsonl_dataset_from_path: null or non-UTF-8 path",
            );
        }
        return std::ptr::null_mut();
    };
    let Some(tokenizer_path) = (unsafe { cstr_to_str(tokenizer_path) }).map(str::to_owned) else {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_jsonl_dataset_from_path: null or non-UTF-8 tokenizer_path",
            );
        }
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let chat_template = unsafe { cstr_to_opt_string(chat_template) };
    let device_spec = unsafe { cstr_to_opt_string(device) }.unwrap_or_else(|| "cpu".to_string());

    if max_seq_len == 0 {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_jsonl_dataset_from_path: max_seq_len must be > 0",
            );
        }
        return std::ptr::null_mut();
    }

    let cdev = match parse_train_device_cabi(&device_spec) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_err(out_err, e);
            }
            return std::ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_validation_err(
                    out_err,
                    &format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
                );
            }
            return std::ptr::null_mut();
        }
    };

    let ds = match InnerJsonlDataset::from_path(
        path.as_path(),
        Arc::new(tokenizer),
        chat_template.as_deref(),
        max_seq_len as usize,
        cdev,
        pad_token_id,
    ) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_validation_err(out_err, &format!("JsonlDataset load failed: {e}"));
            }
            return std::ptr::null_mut();
        }
    };

    BlazenJsonlDataset {
        inner: Arc::new(ds),
    }
    .into_ptr()
}

/// Frees a [`BlazenJsonlDataset`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `dataset` must be null OR a pointer previously produced by
/// [`blazen_jsonl_dataset_from_path`]. Double-free is undefined behavior.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_jsonl_dataset_free(dataset: *mut BlazenJsonlDataset) {
    if dataset.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(dataset) });
}

// ---------------------------------------------------------------------------
// Conversion helpers (Rust-side only; not exposed across the FFI boundary)
// ---------------------------------------------------------------------------

/// Convert the cabi-side `BlazenTrainConfig` into the Rust-side `TrainConfig`,
/// validating tagged enum values and copying every borrowed string.
///
/// # Safety
///
/// `cfg` must be a valid pointer to a fully-populated `BlazenTrainConfig`
/// whose string slots are null OR NUL-terminated UTF-8 buffers, and whose
/// `lora.target_modules` array (when non-null) carries `lora.target_modules_len`
/// valid C-string pointers — all valid for the duration of this call.
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)] // Why: matches the convention in crate::manager (funnels into Box<BlazenError> at the FFI boundary).
pub(crate) unsafe fn convert_train_config(
    cfg: *const BlazenTrainConfig,
) -> Result<InnerTrainConfig, InnerError> {
    let Some(cfg) = (unsafe { cfg.as_ref() }) else {
        return Err(InnerError::Validation {
            message: "blazen_model_manager_train_lora: config is null".to_string(),
        });
    };

    let Some(base_model_repo) = (unsafe { cstr_to_str(cfg.base_model_repo) }).map(str::to_owned)
    else {
        return Err(InnerError::Validation {
            message: "TrainConfig.base_model_repo must be a non-null UTF-8 string".to_string(),
        });
    };
    if base_model_repo.trim().is_empty() {
        return Err(InnerError::Validation {
            message: "TrainConfig.base_model_repo must be non-empty".to_string(),
        });
    }

    let Some(output_dir) = (unsafe { cstr_to_str(cfg.output_dir) }).map(PathBuf::from) else {
        return Err(InnerError::Validation {
            message: "TrainConfig.output_dir must be a non-null UTF-8 string".to_string(),
        });
    };
    if output_dir.as_os_str().is_empty() {
        return Err(InnerError::Validation {
            message: "TrainConfig.output_dir must be non-empty".to_string(),
        });
    }

    if cfg.max_steps == 0 {
        return Err(InnerError::Validation {
            message: "TrainConfig.max_steps must be > 0".to_string(),
        });
    }
    if cfg.batch_size == 0 {
        return Err(InnerError::Validation {
            message: "TrainConfig.batch_size must be > 0".to_string(),
        });
    }
    if cfg.gradient_accumulation_steps == 0 {
        return Err(InnerError::Validation {
            message: "TrainConfig.gradient_accumulation_steps must be > 0".to_string(),
        });
    }
    if cfg.max_seq_len == 0 {
        return Err(InnerError::Validation {
            message: "TrainConfig.max_seq_len must be > 0".to_string(),
        });
    }

    let lora = unsafe { convert_lora_config(&cfg.lora) }?;
    let optim = convert_optim_config(&cfg.optim)?;
    let scheduler = convert_scheduler_config(&cfg.scheduler)?;
    let mixed_precision = convert_mixed_precision(cfg.mixed_precision)?;
    // SAFETY: caller upholds the NUL + lifetime contract on `device`.
    let device = unsafe { cstr_to_opt_string(cfg.device) };

    Ok(InnerTrainConfig {
        base_model_repo,
        output_dir,
        lora,
        optim,
        scheduler,
        max_steps: cfg.max_steps as usize,
        batch_size: cfg.batch_size as usize,
        gradient_accumulation_steps: cfg.gradient_accumulation_steps as usize,
        max_seq_len: cfg.max_seq_len as usize,
        eval_steps: if cfg.has_eval_steps == 0 {
            None
        } else {
            Some(cfg.eval_steps as usize)
        },
        save_steps: if cfg.has_save_steps == 0 {
            None
        } else {
            Some(cfg.save_steps as usize)
        },
        seed: cfg.seed,
        mixed_precision,
        device,
    })
}

#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
unsafe fn convert_lora_config(c: &BlazenLoraConfig) -> Result<InnerLoraConfig, InnerError> {
    if c.rank == 0 {
        return Err(InnerError::Validation {
            message: "LoraConfig.rank must be > 0".to_string(),
        });
    }
    if !c.alpha.is_finite() || c.alpha <= 0.0 {
        return Err(InnerError::Validation {
            message: "LoraConfig.alpha must be > 0".to_string(),
        });
    }
    if !(0.0..1.0).contains(&c.dropout) {
        return Err(InnerError::Validation {
            message: "LoraConfig.dropout must be in [0.0, 1.0)".to_string(),
        });
    }

    let target_modules = if c.target_modules.is_null() || c.target_modules_len == 0 {
        Vec::new()
    } else {
        // SAFETY: caller upholds the contract that `target_modules` points to
        // `target_modules_len` valid C-string pointers for the duration of
        // this call.
        let slice = unsafe { std::slice::from_raw_parts(c.target_modules, c.target_modules_len) };
        let mut out = Vec::with_capacity(slice.len());
        for (i, &ptr) in slice.iter().enumerate() {
            // SAFETY: same contract.
            let Some(s) = (unsafe { cstr_to_str(ptr) }) else {
                return Err(InnerError::Validation {
                    message: format!(
                        "LoraConfig.target_modules[{i}] must be a non-null UTF-8 string"
                    ),
                });
            };
            out.push(s.to_owned());
        }
        out
    };

    if target_modules.is_empty() {
        return Err(InnerError::Validation {
            message: "LoraConfig.target_modules must be non-empty".to_string(),
        });
    }

    Ok(InnerLoraConfig {
        rank: c.rank as usize,
        alpha: c.alpha,
        dropout: c.dropout,
        target_modules,
    })
}

#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
fn convert_optim_config(c: &BlazenOptimConfig) -> Result<InnerOptimConfig, InnerError> {
    if !c.learning_rate.is_finite() || c.learning_rate <= 0.0 {
        return Err(InnerError::Validation {
            message: "OptimConfig.learning_rate must be > 0".to_string(),
        });
    }
    if !(0.0..1.0).contains(&c.beta1) || !(0.0..1.0).contains(&c.beta2) {
        return Err(InnerError::Validation {
            message: "OptimConfig.beta1 / beta2 must be in [0.0, 1.0)".to_string(),
        });
    }
    if !c.epsilon.is_finite() || c.epsilon <= 0.0 {
        return Err(InnerError::Validation {
            message: "OptimConfig.epsilon must be > 0".to_string(),
        });
    }
    if !c.weight_decay.is_finite() || c.weight_decay < 0.0 {
        return Err(InnerError::Validation {
            message: "OptimConfig.weight_decay must be >= 0".to_string(),
        });
    }
    let gradient_clip = if c.has_gradient_clip == 0 {
        None
    } else {
        if !c.gradient_clip.is_finite() || c.gradient_clip <= 0.0 {
            return Err(InnerError::Validation {
                message: "OptimConfig.gradient_clip, when set, must be > 0".to_string(),
            });
        }
        Some(c.gradient_clip)
    };
    Ok(InnerOptimConfig {
        learning_rate: c.learning_rate,
        beta1: c.beta1,
        beta2: c.beta2,
        epsilon: c.epsilon,
        weight_decay: c.weight_decay,
        gradient_clip,
    })
}

#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
fn convert_scheduler_config(c: &BlazenSchedulerConfig) -> Result<InnerSchedulerConfig, InnerError> {
    let kind = match c.kind {
        BLAZEN_SCHEDULER_CONSTANT => InnerSchedulerKind::Constant,
        BLAZEN_SCHEDULER_LINEAR => InnerSchedulerKind::Linear,
        BLAZEN_SCHEDULER_COSINE => InnerSchedulerKind::Cosine,
        other => {
            return Err(InnerError::Validation {
                message: format!(
                    "SchedulerConfig.kind {other} is not one of BLAZEN_SCHEDULER_CONSTANT (0), \
                     BLAZEN_SCHEDULER_LINEAR (1), or BLAZEN_SCHEDULER_COSINE (2)"
                ),
            });
        }
    };
    Ok(InnerSchedulerConfig {
        kind,
        warmup_steps: c.warmup_steps as usize,
    })
}

#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
fn convert_mixed_precision(tag: i32) -> Result<InnerMixedPrecision, InnerError> {
    match tag {
        BLAZEN_MIXED_PRECISION_NONE => Ok(InnerMixedPrecision::None),
        BLAZEN_MIXED_PRECISION_BF16 => Ok(InnerMixedPrecision::Bf16),
        other => Err(InnerError::Validation {
            message: format!(
                "TrainConfig.mixed_precision {other} is not one of \
                 BLAZEN_MIXED_PRECISION_NONE (0) or BLAZEN_MIXED_PRECISION_BF16 (1)"
            ),
        }),
    }
}

/// Convert the Rust-side [`blazen_train::TrainedAdapter`] into the cabi flat
/// record. Allocates a heap C string for `adapter_dir`.
#[cfg(feature = "training")]
pub(crate) fn trained_adapter_to_cabi(a: &blazen_train::TrainedAdapter) -> BlazenTrainedAdapter {
    BlazenTrainedAdapter {
        adapter_dir: alloc_cstring(&a.adapter_dir.display().to_string()),
        final_loss: a.final_loss,
        total_steps: a.total_steps as u64,
    }
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
unsafe fn write_err(out_err: *mut *mut BlazenError, err: InnerError) {
    if out_err.is_null() {
        return;
    }
    // SAFETY: out-param contract.
    unsafe {
        *out_err = BlazenError::from(err).into_ptr();
    }
}

#[cfg(feature = "training")]
unsafe fn write_validation_err(out_err: *mut *mut BlazenError, msg: &str) {
    // SAFETY: forwarded to `write_err`.
    unsafe {
        write_err(
            out_err,
            InnerError::Validation {
                message: msg.to_owned(),
            },
        );
    }
}

/// Parse the `device` field of a [`BlazenTrainConfig`] (or the `device`
/// argument of [`blazen_jsonl_dataset_from_path`]) into a `candle_core::Device`.
///
/// Mirrors the helper in `blazen-manager::parse_train_device` so the cabi
/// pre-flight validation matches what `ModelManager::train_lora` would do
/// internally — surfacing bad device specs to the C caller before the train
/// loop spins up.
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
fn parse_train_device_cabi(spec: &str) -> Result<candle_core::Device, InnerError> {
    let normalized = spec.trim().to_ascii_lowercase();
    if normalized == "cpu" {
        return Ok(candle_core::Device::Cpu);
    }
    let (kind, idx) = match normalized.split_once(':') {
        Some((k, rest)) => {
            let parsed = rest.parse::<usize>().map_err(|e| InnerError::Validation {
                message: format!("training device '{spec}' has non-numeric index '{rest}': {e}"),
            })?;
            (k, parsed)
        }
        None => (normalized.as_str(), 0),
    };
    match kind {
        "cuda" => candle_core::Device::new_cuda(idx).map_err(|e| InnerError::Validation {
            message: format!("cuda:{idx} unavailable: {e}"),
        }),
        "metal" => candle_core::Device::new_metal(idx).map_err(|e| InnerError::Validation {
            message: format!("metal:{idx} unavailable: {e}"),
        }),
        other => Err(InnerError::Validation {
            message: format!(
                "unknown training device '{other}' (want one of: cpu, cuda[:N], metal[:N])"
            ),
        }),
    }
}

// ===========================================================================
// PR8 Wave 18 — DPO / ORPO / SimPO / KTO / full fine-tune
//
// Same convention as the Wave 17 LoRA surface above: flat `#[repr(C)]` config
// records the caller populates on the stack; nullable scalar slots use `has_*`
// discriminator bytes (no Option<T> across the FFI boundary); strings are
// `*const c_char` borrowed only for the call; PreferenceJsonlDataset and
// RatedJsonlDataset are opaque (Arc-wrapped so a single dataset can be reused
// across multiple train_* calls without re-tokenizing).
//
// Why: Ruby per-language wrapper consumes these in Wave 19; the cbindgen-
// regenerated `bindings/ruby/ext/blazen/blazen.h` carries the declarations
// after `cargo build -p blazen-cabi --release`. Progress callback stays
// deferred (same caveat documented above for train_lora).
// ===========================================================================

// ---------------------------------------------------------------------------
// BlazenTrainCoreConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::TrainCoreConfig`]. Composed by the
/// preference / KTO / full-fine-tune config records as their `core` slot;
/// the SFT-only [`BlazenTrainConfig`] above stays flat for backward
/// compatibility with PR7 consumers.
///
/// String pointers (`base_model_repo`, `base_model_revision`, `output_dir`,
/// `device`) are borrowed only for the call. Nullable scalar slots
/// (`has_eval_steps`, `has_save_steps`) and nullable strings
/// (`base_model_revision = NULL`, `device = NULL`) use the conventions
/// established by [`BlazenTrainConfig`].
#[repr(C)]
pub struct BlazenTrainCoreConfig {
    /// Hugging Face repo id of the base model. Required (non-null).
    pub base_model_repo: *const c_char,
    /// Optional revision (branch / tag / commit) for the base model. Null for
    /// "default" (the HF-Hub default branch).
    pub base_model_revision: *const c_char,
    /// Filesystem directory for trained weights and checkpoints. Required.
    pub output_dir: *const c_char,
    /// `AdamW` hyperparameters.
    pub optim: BlazenOptimConfig,
    /// LR-schedule configuration.
    pub scheduler: BlazenSchedulerConfig,
    /// Total optimizer steps to run. Must be > 0.
    pub max_steps: u32,
    /// Micro-batch size per forward pass. Must be > 0.
    pub batch_size: u32,
    /// Micro-batches to accumulate before each optimizer step. Must be > 0.
    pub gradient_accumulation_steps: u32,
    /// Max tokenized sequence length per example. Must be > 0.
    pub max_seq_len: u32,
    /// 0 = `None`, 1 = `Some(eval_steps)`.
    pub has_eval_steps: i32,
    /// Run evaluation every N steps when `has_eval_steps == 1`.
    pub eval_steps: u32,
    /// 0 = `None`, 1 = `Some(save_steps)`.
    pub has_save_steps: i32,
    /// Write a checkpoint every N steps when `has_save_steps == 1`.
    pub save_steps: u32,
    /// RNG seed.
    pub seed: u64,
    /// One of `BLAZEN_MIXED_PRECISION_*`.
    pub mixed_precision: i32,
    /// Device specifier (`"cpu"`, `"cuda:0"`, `"metal"`). Null = `"cpu"`.
    pub device: *const c_char,
}

// ---------------------------------------------------------------------------
// BlazenDpoConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::DpoConfig`].
///
/// `reference_model_repo` / `reference_model_revision` are nullable — passing
/// `NULL` makes the trainer use `core.base_model_repo` as the reference. The
/// LoRA-A init seed is taken from `core.seed`.
#[repr(C)]
pub struct BlazenDpoConfig {
    /// Shared training hyperparameters.
    pub core: BlazenTrainCoreConfig,
    /// `LoRA` hyperparameters applied to the policy model.
    pub lora: BlazenLoraConfig,
    /// KL-regularization strength (TRL default 0.1).
    pub beta: f32,
    /// Conservative DPO label smoothing (cDPO). 0.0 disables.
    pub label_smoothing: f32,
    /// Reference model repo. Null = reuse `core.base_model_repo`.
    pub reference_model_repo: *const c_char,
    /// Optional revision for the reference model.
    pub reference_model_revision: *const c_char,
}

// ---------------------------------------------------------------------------
// BlazenOrpoConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::OrpoConfig`]. Reference-free.
#[repr(C)]
pub struct BlazenOrpoConfig {
    /// Shared training hyperparameters.
    pub core: BlazenTrainCoreConfig,
    /// `LoRA` hyperparameters.
    pub lora: BlazenLoraConfig,
    /// Weight of the odds-ratio term relative to the SFT term.
    pub lambda: f32,
}

// ---------------------------------------------------------------------------
// BlazenSimpoConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::SimpoConfig`]. Reference-free.
#[repr(C)]
pub struct BlazenSimpoConfig {
    /// Shared training hyperparameters.
    pub core: BlazenTrainCoreConfig,
    /// `LoRA` hyperparameters.
    pub lora: BlazenLoraConfig,
    /// Logit scaling for the length-normalized preference margin.
    pub beta: f32,
    /// Target reward margin between chosen and rejected.
    pub gamma: f32,
}

// ---------------------------------------------------------------------------
// BlazenKtoConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::KtoConfig`]. Requires a frozen
/// reference model — pass `NULL` for `reference_model_repo` to reuse
/// `core.base_model_repo`.
#[repr(C)]
pub struct BlazenKtoConfig {
    /// Shared training hyperparameters.
    pub core: BlazenTrainCoreConfig,
    /// `LoRA` hyperparameters applied to the policy model.
    pub lora: BlazenLoraConfig,
    /// KL-regularization strength.
    pub beta: f32,
    /// Loss weight applied to desirable examples.
    pub lambda_d: f32,
    /// Loss weight applied to undesirable examples.
    pub lambda_u: f32,
    /// Reference model repo. Null = reuse `core.base_model_repo`.
    pub reference_model_repo: *const c_char,
    /// Optional revision for the reference model.
    pub reference_model_revision: *const c_char,
}

// ---------------------------------------------------------------------------
// BlazenFullFineTuneConfig
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of [`blazen_train::FullFineTuneConfig`] — every
/// parameter is trainable, no `LoRA` wrapping. Currently the trainer rejects
/// `gradient_checkpointing = 1` because candle 0.10.2 has no checkpointing
/// primitive; the field is exposed for forward compatibility.
#[repr(C)]
pub struct BlazenFullFineTuneConfig {
    /// Shared training hyperparameters.
    pub core: BlazenTrainCoreConfig,
    /// Activation checkpointing flag. 0 = disabled (the only currently
    /// supported value); 1 will surface `Unsupported` at trainer init.
    pub gradient_checkpointing: i32,
}

// ---------------------------------------------------------------------------
// BlazenFullFineTuneResult
// ---------------------------------------------------------------------------

/// `#[repr(C)]` flat record returned by-value from
/// [`crate::manager::blazen_model_manager_fine_tune_blocking`] (via an
/// out-param) and from
/// [`crate::future::blazen_future_take_full_finetune_result`].
///
/// `output_dir` is a caller-owned heap C string — release the whole record
/// (which frees the string too) with [`blazen_full_finetune_result_free`].
#[repr(C)]
pub struct BlazenFullFineTuneResult {
    /// Directory the full-model weights were written to (caller-owned C
    /// string). Free via [`blazen_full_finetune_result_free`].
    pub output_dir: *mut c_char,
    /// Final training loss reported by the trainer.
    pub final_loss: f32,
    /// Total optimizer steps actually executed.
    pub steps_completed: u64,
}

/// Releases the heap C string carried by `result`. Defensive on a null pointer
/// AND on a record whose `output_dir` slot is already null (so callers can
/// safely free a partially-initialized stack value after an error).
///
/// # Safety
///
/// `result` must be null OR a pointer to a valid [`BlazenFullFineTuneResult`]
/// whose `output_dir` field is null OR was previously produced by the cabi
/// surface. Double-free on the same `output_dir` is undefined behavior.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_full_finetune_result_free(result: *mut BlazenFullFineTuneResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the pointer contract per the per-fn docs.
    let r = unsafe { &mut *result };
    if !r.output_dir.is_null() {
        // SAFETY: out-param contract — output_dir was produced by
        // `alloc_cstring` (i.e. `CString::into_raw`), so `blazen_string_free`
        // reclaims it on the matching allocator.
        unsafe {
            crate::string::blazen_string_free(r.output_dir);
        }
        r.output_dir = std::ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// BlazenPreferenceJsonlDataset + BlazenRatedJsonlDataset (opaque handles)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping `Arc<blazen_train::dataset::PreferenceJsonlDataset>`.
/// Constructed via [`blazen_preference_jsonl_dataset_from_path`] and freed via
/// [`blazen_preference_jsonl_dataset_free`]. The `Arc` lets a single dataset
/// be reused across multiple `train_dpo` / `train_orpo` / `train_simpo`
/// invocations without re-tokenizing.
#[cfg(feature = "training")]
pub struct BlazenPreferenceJsonlDataset {
    pub(crate) inner: Arc<InnerPreferenceJsonlDataset>,
}

#[cfg(feature = "training")]
impl BlazenPreferenceJsonlDataset {
    pub(crate) fn into_ptr(self) -> *mut BlazenPreferenceJsonlDataset {
        Box::into_raw(Box::new(self))
    }
}

/// Loads a preference-pair JSONL file at `path` with the tokenizer at
/// `tokenizer_path` and returns a caller-owned
/// [`BlazenPreferenceJsonlDataset`]. Free with
/// [`blazen_preference_jsonl_dataset_free`].
///
/// Returns null on failure and writes `*out_err` (validation / tokenizer-load
/// / dataset-parse errors).
///
/// # Safety
///
/// `path` and `tokenizer_path` must be NUL-terminated UTF-8 buffers valid for
/// the call. `chat_template` and `device` are null OR same contract.
/// `out_err` is null OR a single-writer destination.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_preference_jsonl_dataset_from_path(
    path: *const c_char,
    tokenizer_path: *const c_char,
    chat_template: *const c_char,
    max_seq_len: u32,
    device: *const c_char,
    pad_token_id: u32,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenPreferenceJsonlDataset {
    // SAFETY: caller upholds the NUL + lifetime contracts on the input strings.
    let Some(path) = (unsafe { cstr_to_str(path) }).map(PathBuf::from) else {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_preference_jsonl_dataset_from_path: null or non-UTF-8 path",
            );
        }
        return std::ptr::null_mut();
    };
    let Some(tokenizer_path) = (unsafe { cstr_to_str(tokenizer_path) }).map(str::to_owned) else {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_preference_jsonl_dataset_from_path: null or non-UTF-8 tokenizer_path",
            );
        }
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let chat_template = unsafe { cstr_to_opt_string(chat_template) };
    let device_spec = unsafe { cstr_to_opt_string(device) }.unwrap_or_else(|| "cpu".to_string());

    if max_seq_len == 0 {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_preference_jsonl_dataset_from_path: max_seq_len must be > 0",
            );
        }
        return std::ptr::null_mut();
    }

    let cdev = match parse_train_device_cabi(&device_spec) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_err(out_err, e);
            }
            return std::ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_validation_err(
                    out_err,
                    &format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
                );
            }
            return std::ptr::null_mut();
        }
    };

    let ds = match InnerPreferenceJsonlDataset::from_path(
        path.as_path(),
        Arc::new(tokenizer),
        chat_template.as_deref(),
        max_seq_len as usize,
        cdev,
        pad_token_id,
    ) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_validation_err(out_err, &format!("PreferenceJsonlDataset load failed: {e}"));
            }
            return std::ptr::null_mut();
        }
    };

    BlazenPreferenceJsonlDataset {
        inner: Arc::new(ds),
    }
    .into_ptr()
}

/// Frees a [`BlazenPreferenceJsonlDataset`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `dataset` must be null OR a pointer previously produced by
/// [`blazen_preference_jsonl_dataset_from_path`]. Double-free is undefined
/// behavior.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_preference_jsonl_dataset_free(
    dataset: *mut BlazenPreferenceJsonlDataset,
) {
    if dataset.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(dataset) });
}

/// Opaque handle wrapping `Arc<blazen_train::dataset::RatedJsonlDataset>`.
/// Constructed via [`blazen_rated_jsonl_dataset_from_path`] and freed via
/// [`blazen_rated_jsonl_dataset_free`]. Consumed by
/// [`crate::manager::blazen_model_manager_train_kto`].
#[cfg(feature = "training")]
pub struct BlazenRatedJsonlDataset {
    pub(crate) inner: Arc<InnerRatedJsonlDataset>,
}

#[cfg(feature = "training")]
impl BlazenRatedJsonlDataset {
    pub(crate) fn into_ptr(self) -> *mut BlazenRatedJsonlDataset {
        Box::into_raw(Box::new(self))
    }
}

/// Loads a rated JSONL file (KTO format) at `path` with the tokenizer at
/// `tokenizer_path` and returns a caller-owned [`BlazenRatedJsonlDataset`].
/// Free with [`blazen_rated_jsonl_dataset_free`].
///
/// Returns null on failure and writes `*out_err`.
///
/// # Safety
///
/// Same contract as [`blazen_preference_jsonl_dataset_from_path`].
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rated_jsonl_dataset_from_path(
    path: *const c_char,
    tokenizer_path: *const c_char,
    chat_template: *const c_char,
    max_seq_len: u32,
    device: *const c_char,
    pad_token_id: u32,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenRatedJsonlDataset {
    // SAFETY: caller upholds the NUL + lifetime contracts on the input strings.
    let Some(path) = (unsafe { cstr_to_str(path) }).map(PathBuf::from) else {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_rated_jsonl_dataset_from_path: null or non-UTF-8 path",
            );
        }
        return std::ptr::null_mut();
    };
    let Some(tokenizer_path) = (unsafe { cstr_to_str(tokenizer_path) }).map(str::to_owned) else {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_rated_jsonl_dataset_from_path: null or non-UTF-8 tokenizer_path",
            );
        }
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let chat_template = unsafe { cstr_to_opt_string(chat_template) };
    let device_spec = unsafe { cstr_to_opt_string(device) }.unwrap_or_else(|| "cpu".to_string());

    if max_seq_len == 0 {
        // SAFETY: out-param contract.
        unsafe {
            write_validation_err(
                out_err,
                "blazen_rated_jsonl_dataset_from_path: max_seq_len must be > 0",
            );
        }
        return std::ptr::null_mut();
    }

    let cdev = match parse_train_device_cabi(&device_spec) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_err(out_err, e);
            }
            return std::ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_validation_err(
                    out_err,
                    &format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
                );
            }
            return std::ptr::null_mut();
        }
    };

    let ds = match InnerRatedJsonlDataset::from_path(
        path.as_path(),
        Arc::new(tokenizer),
        chat_template.as_deref(),
        max_seq_len as usize,
        cdev,
        pad_token_id,
    ) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: out-param contract.
            unsafe {
                write_validation_err(out_err, &format!("RatedJsonlDataset load failed: {e}"));
            }
            return std::ptr::null_mut();
        }
    };

    BlazenRatedJsonlDataset {
        inner: Arc::new(ds),
    }
    .into_ptr()
}

/// Frees a [`BlazenRatedJsonlDataset`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `dataset` must be null OR a pointer previously produced by
/// [`blazen_rated_jsonl_dataset_from_path`]. Double-free is undefined
/// behavior.
#[cfg(feature = "training")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rated_jsonl_dataset_free(dataset: *mut BlazenRatedJsonlDataset) {
    if dataset.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(dataset) });
}

// ---------------------------------------------------------------------------
// Conversion helpers — BlazenTrainCoreConfig → TrainCoreConfig
// ---------------------------------------------------------------------------

/// Convert a `BlazenTrainCoreConfig` reference into a Rust-side
/// [`TrainCoreConfig`]. Validates non-null required strings, valid UTF-8,
/// non-zero counts, and tagged-enum payloads.
///
/// # Safety
///
/// `c` must reference a fully-populated `BlazenTrainCoreConfig` whose string
/// slots are null OR NUL-terminated UTF-8 buffers valid for the duration of
/// this call.
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
pub(crate) unsafe fn convert_train_core_config(
    c: &BlazenTrainCoreConfig,
    where_: &str,
) -> Result<InnerTrainCoreConfig, InnerError> {
    let Some(base_model_repo) = (unsafe { cstr_to_str(c.base_model_repo) }).map(str::to_owned)
    else {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.base_model_repo must be a non-null UTF-8 string"),
        });
    };
    if base_model_repo.trim().is_empty() {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.base_model_repo must be non-empty"),
        });
    }
    // SAFETY: same contract on `c.base_model_revision`.
    let base_model_revision = unsafe { cstr_to_opt_string(c.base_model_revision) };

    let Some(output_dir) = (unsafe { cstr_to_str(c.output_dir) }).map(PathBuf::from) else {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.output_dir must be a non-null UTF-8 string"),
        });
    };
    if output_dir.as_os_str().is_empty() {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.output_dir must be non-empty"),
        });
    }

    if c.max_steps == 0 {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.max_steps must be > 0"),
        });
    }
    if c.batch_size == 0 {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.batch_size must be > 0"),
        });
    }
    if c.gradient_accumulation_steps == 0 {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.gradient_accumulation_steps must be > 0"),
        });
    }
    if c.max_seq_len == 0 {
        return Err(InnerError::Validation {
            message: format!("{where_}: core.max_seq_len must be > 0"),
        });
    }

    let optim = convert_optim_config(&c.optim)?;
    let scheduler = convert_scheduler_config(&c.scheduler)?;
    let mixed_precision = convert_mixed_precision(c.mixed_precision)?;
    // SAFETY: same contract on `c.device`.
    let device = unsafe { cstr_to_opt_string(c.device) };

    Ok(InnerTrainCoreConfig {
        base_model_repo,
        base_model_revision,
        output_dir,
        max_steps: c.max_steps as usize,
        batch_size: c.batch_size as usize,
        gradient_accumulation_steps: c.gradient_accumulation_steps as usize,
        max_seq_len: c.max_seq_len as usize,
        eval_steps: if c.has_eval_steps == 0 {
            None
        } else {
            Some(c.eval_steps as usize)
        },
        save_steps: if c.has_save_steps == 0 {
            None
        } else {
            Some(c.save_steps as usize)
        },
        seed: c.seed,
        mixed_precision,
        device,
        optim,
        scheduler,
    })
}

// ---------------------------------------------------------------------------
// TryFrom impls — cabi config refs → blazen-train configs.
// Implemented as helper functions (not std::convert::TryFrom) because the
// conversion is `unsafe` (dereferences raw pointers inside the config).
// ---------------------------------------------------------------------------

/// Convert a `*const BlazenDpoConfig` into a Rust-side [`DpoConfig`].
///
/// # Safety
///
/// `cfg` must be null OR a pointer to a fully-populated `BlazenDpoConfig`
/// whose pointer fields are each null OR NUL-terminated UTF-8 buffers valid
/// for the duration of this call.
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
pub(crate) unsafe fn convert_dpo_config(
    cfg: *const BlazenDpoConfig,
) -> Result<InnerDpoConfig, InnerError> {
    let Some(cfg) = (unsafe { cfg.as_ref() }) else {
        return Err(InnerError::Validation {
            message: "blazen_model_manager_train_dpo: config is null".to_string(),
        });
    };
    let core = unsafe { convert_train_core_config(&cfg.core, "DpoConfig") }?;
    let lora = unsafe { convert_lora_config(&cfg.lora) }?;
    // SAFETY: caller upholds the NUL + lifetime contract on the two
    // reference-model pointer slots.
    let reference_model_repo = unsafe { cstr_to_opt_string(cfg.reference_model_repo) };
    let reference_model_revision = unsafe { cstr_to_opt_string(cfg.reference_model_revision) };

    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(InnerError::Validation {
            message: format!("DpoConfig.beta must be > 0 (got {})", cfg.beta),
        });
    }
    if !cfg.label_smoothing.is_finite() || cfg.label_smoothing < 0.0 || cfg.label_smoothing >= 0.5 {
        return Err(InnerError::Validation {
            message: format!(
                "DpoConfig.label_smoothing must be in [0.0, 0.5) (got {})",
                cfg.label_smoothing
            ),
        });
    }

    Ok(InnerDpoConfig {
        core,
        lora,
        beta: cfg.beta,
        reference_model_repo,
        reference_model_revision,
        label_smoothing: cfg.label_smoothing,
    })
}

/// Convert a `*const BlazenOrpoConfig` into a Rust-side [`OrpoConfig`].
///
/// # Safety
///
/// See [`convert_dpo_config`].
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
pub(crate) unsafe fn convert_orpo_config(
    cfg: *const BlazenOrpoConfig,
) -> Result<InnerOrpoConfig, InnerError> {
    let Some(cfg) = (unsafe { cfg.as_ref() }) else {
        return Err(InnerError::Validation {
            message: "blazen_model_manager_train_orpo: config is null".to_string(),
        });
    };
    let core = unsafe { convert_train_core_config(&cfg.core, "OrpoConfig") }?;
    let lora = unsafe { convert_lora_config(&cfg.lora) }?;

    if !cfg.lambda.is_finite() || cfg.lambda < 0.0 {
        return Err(InnerError::Validation {
            message: format!("OrpoConfig.lambda must be >= 0 (got {})", cfg.lambda),
        });
    }

    Ok(InnerOrpoConfig {
        core,
        lora,
        lambda: cfg.lambda,
    })
}

/// Convert a `*const BlazenSimpoConfig` into a Rust-side [`SimpoConfig`].
///
/// # Safety
///
/// See [`convert_dpo_config`].
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
pub(crate) unsafe fn convert_simpo_config(
    cfg: *const BlazenSimpoConfig,
) -> Result<InnerSimpoConfig, InnerError> {
    let Some(cfg) = (unsafe { cfg.as_ref() }) else {
        return Err(InnerError::Validation {
            message: "blazen_model_manager_train_simpo: config is null".to_string(),
        });
    };
    let core = unsafe { convert_train_core_config(&cfg.core, "SimpoConfig") }?;
    let lora = unsafe { convert_lora_config(&cfg.lora) }?;

    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(InnerError::Validation {
            message: format!("SimpoConfig.beta must be > 0 (got {})", cfg.beta),
        });
    }
    if !cfg.gamma.is_finite() || cfg.gamma < 0.0 {
        return Err(InnerError::Validation {
            message: format!("SimpoConfig.gamma must be >= 0 (got {})", cfg.gamma),
        });
    }

    Ok(InnerSimpoConfig {
        core,
        lora,
        beta: cfg.beta,
        gamma: cfg.gamma,
    })
}

/// Convert a `*const BlazenKtoConfig` into a Rust-side [`KtoConfig`].
///
/// # Safety
///
/// See [`convert_dpo_config`].
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
pub(crate) unsafe fn convert_kto_config(
    cfg: *const BlazenKtoConfig,
) -> Result<InnerKtoConfig, InnerError> {
    let Some(cfg) = (unsafe { cfg.as_ref() }) else {
        return Err(InnerError::Validation {
            message: "blazen_model_manager_train_kto: config is null".to_string(),
        });
    };
    let core = unsafe { convert_train_core_config(&cfg.core, "KtoConfig") }?;
    let lora = unsafe { convert_lora_config(&cfg.lora) }?;
    // SAFETY: caller upholds the NUL + lifetime contract on the two
    // reference-model pointer slots.
    let reference_model_repo = unsafe { cstr_to_opt_string(cfg.reference_model_repo) };
    let reference_model_revision = unsafe { cstr_to_opt_string(cfg.reference_model_revision) };

    if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
        return Err(InnerError::Validation {
            message: format!("KtoConfig.beta must be > 0 (got {})", cfg.beta),
        });
    }
    if !cfg.lambda_d.is_finite() || cfg.lambda_d < 0.0 {
        return Err(InnerError::Validation {
            message: format!("KtoConfig.lambda_d must be >= 0 (got {})", cfg.lambda_d),
        });
    }
    if !cfg.lambda_u.is_finite() || cfg.lambda_u < 0.0 {
        return Err(InnerError::Validation {
            message: format!("KtoConfig.lambda_u must be >= 0 (got {})", cfg.lambda_u),
        });
    }

    Ok(InnerKtoConfig {
        core,
        lora,
        beta: cfg.beta,
        lambda_d: cfg.lambda_d,
        lambda_u: cfg.lambda_u,
        reference_model_repo,
        reference_model_revision,
    })
}

/// Convert a `*const BlazenFullFineTuneConfig` into a Rust-side
/// [`FullFineTuneConfig`].
///
/// # Safety
///
/// See [`convert_dpo_config`].
#[cfg(feature = "training")]
#[allow(clippy::result_large_err)]
pub(crate) unsafe fn convert_full_finetune_config(
    cfg: *const BlazenFullFineTuneConfig,
) -> Result<InnerFullFineTuneConfig, InnerError> {
    let Some(cfg) = (unsafe { cfg.as_ref() }) else {
        return Err(InnerError::Validation {
            message: "blazen_model_manager_fine_tune: config is null".to_string(),
        });
    };
    let core = unsafe { convert_train_core_config(&cfg.core, "FullFineTuneConfig") }?;
    Ok(InnerFullFineTuneConfig {
        core,
        gradient_checkpointing: cfg.gradient_checkpointing != 0,
    })
}

/// Convert the Rust-side [`InnerFullFineTuneResult`] into the cabi flat
/// record. Allocates a heap C string for `output_dir`.
#[cfg(feature = "training")]
pub(crate) fn full_finetune_result_to_cabi(
    r: &InnerFullFineTuneResult,
) -> BlazenFullFineTuneResult {
    BlazenFullFineTuneResult {
        output_dir: alloc_cstring(&r.output_dir.display().to_string()),
        final_loss: r.final_loss,
        steps_completed: r.steps_completed as u64,
    }
}

// ---------------------------------------------------------------------------
// Tests (PR8 Wave 18)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "training"))]
mod tests {
    use std::ffi::CString;

    use super::*;

    /// Build a fully-populated `BlazenTrainCoreConfig` whose string slots are
    /// owned by the returned `Vec<CString>` (kept alive by the caller for the
    /// duration of the test). Returns the core record and the backing strings.
    fn make_core_record(repo: &str) -> (BlazenTrainCoreConfig, Vec<CString>) {
        let repo_c = CString::new(repo).unwrap();
        let out_c = CString::new("./out").unwrap();
        let core = BlazenTrainCoreConfig {
            base_model_repo: repo_c.as_ptr(),
            base_model_revision: std::ptr::null(),
            output_dir: out_c.as_ptr(),
            optim: BlazenOptimConfig {
                learning_rate: 2e-4,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                has_gradient_clip: 1,
                gradient_clip: 1.0,
            },
            scheduler: BlazenSchedulerConfig {
                kind: BLAZEN_SCHEDULER_COSINE,
                warmup_steps: 10,
            },
            max_steps: 100,
            batch_size: 1,
            gradient_accumulation_steps: 4,
            max_seq_len: 128,
            has_eval_steps: 0,
            eval_steps: 0,
            has_save_steps: 0,
            save_steps: 0,
            seed: 42,
            mixed_precision: BLAZEN_MIXED_PRECISION_NONE,
            device: std::ptr::null(),
        };
        (core, vec![repo_c, out_c])
    }

    fn make_lora_record(modules: &[CString]) -> (BlazenLoraConfig, Vec<*const c_char>) {
        let ptrs: Vec<*const c_char> = modules.iter().map(|c| c.as_ptr()).collect();
        let lora = BlazenLoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: ptrs.as_ptr(),
            target_modules_len: ptrs.len(),
        };
        (lora, ptrs)
    }

    #[test]
    fn dpo_config_tryfrom_rejects_null_repo() {
        // core.base_model_repo is null → should hit the validation branch in
        // `convert_train_core_config` and surface as InnerError::Validation
        // mentioning core.base_model_repo.
        let out_c = CString::new("./out").unwrap();
        let core = BlazenTrainCoreConfig {
            base_model_repo: std::ptr::null(), // <-- intentionally null
            base_model_revision: std::ptr::null(),
            output_dir: out_c.as_ptr(),
            optim: BlazenOptimConfig {
                learning_rate: 2e-4,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                has_gradient_clip: 0,
                gradient_clip: 0.0,
            },
            scheduler: BlazenSchedulerConfig {
                kind: BLAZEN_SCHEDULER_COSINE,
                warmup_steps: 0,
            },
            max_steps: 10,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            max_seq_len: 32,
            has_eval_steps: 0,
            eval_steps: 0,
            has_save_steps: 0,
            save_steps: 0,
            seed: 42,
            mixed_precision: BLAZEN_MIXED_PRECISION_NONE,
            device: std::ptr::null(),
        };
        let modules = [CString::new("q_proj").unwrap()];
        let (lora, _ptrs) = make_lora_record(&modules);
        let cfg = BlazenDpoConfig {
            core,
            lora,
            beta: 0.1,
            label_smoothing: 0.0,
            reference_model_repo: std::ptr::null(),
            reference_model_revision: std::ptr::null(),
        };
        // SAFETY: `cfg` is a live stack value for the duration of the call;
        // every pointer field is null OR points into a CString that outlives
        // the conversion.
        let err = unsafe { convert_dpo_config(std::ptr::from_ref(&cfg)) }.unwrap_err();
        match err {
            InnerError::Validation { message } => {
                assert!(
                    message.contains("base_model_repo"),
                    "expected base_model_repo in error, got: {message}"
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn dpo_config_tryfrom_rejects_null_config_pointer() {
        // SAFETY: explicit null-pointer input — the conversion must detect and
        // surface the null without dereferencing.
        let err = unsafe { convert_dpo_config(std::ptr::null()) }.unwrap_err();
        match err {
            InnerError::Validation { message } => {
                assert!(message.contains("config is null"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)] // Why: round-trip test deliberately uses `as u32` for clarity, and the
    // record-construction body unavoidably runs ~100 lines.
    fn orpo_config_tryfrom_round_trips() {
        // Build a Rust OrpoConfig, marshal to BlazenOrpoConfig, unmarshal back
        // via the conversion, assert the round-trip preserves every field.
        let original = InnerOrpoConfig {
            core: InnerTrainCoreConfig {
                base_model_repo: "Qwen/Qwen2.5-0.5B".to_string(),
                base_model_revision: None,
                output_dir: PathBuf::from("./out"),
                max_steps: 200,
                batch_size: 2,
                gradient_accumulation_steps: 8,
                max_seq_len: 256,
                eval_steps: None,
                save_steps: Some(50),
                seed: 7,
                mixed_precision: InnerMixedPrecision::None,
                device: None,
                optim: InnerOptimConfig {
                    learning_rate: 1e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.01,
                    gradient_clip: Some(0.5),
                },
                scheduler: InnerSchedulerConfig {
                    kind: InnerSchedulerKind::Cosine,
                    warmup_steps: 20,
                },
            },
            lora: InnerLoraConfig {
                rank: 16,
                alpha: 32.0,
                dropout: 0.05,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            },
            lambda: 0.25,
        };

        // Marshal: build CStrings + cabi record matching `original`.
        let repo_c = CString::new(original.core.base_model_repo.as_str()).unwrap();
        let out_c = CString::new(original.core.output_dir.display().to_string()).unwrap();
        let modules: Vec<CString> = original
            .lora
            .target_modules
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let module_ptrs: Vec<*const c_char> = modules.iter().map(|c| c.as_ptr()).collect();

        let core = BlazenTrainCoreConfig {
            base_model_repo: repo_c.as_ptr(),
            base_model_revision: std::ptr::null(),
            output_dir: out_c.as_ptr(),
            optim: BlazenOptimConfig {
                learning_rate: original.core.optim.learning_rate,
                beta1: original.core.optim.beta1,
                beta2: original.core.optim.beta2,
                epsilon: original.core.optim.epsilon,
                weight_decay: original.core.optim.weight_decay,
                has_gradient_clip: 1,
                gradient_clip: original.core.optim.gradient_clip.unwrap(),
            },
            scheduler: BlazenSchedulerConfig {
                kind: BLAZEN_SCHEDULER_COSINE,
                warmup_steps: 20,
            },
            max_steps: original.core.max_steps as u32,
            batch_size: original.core.batch_size as u32,
            gradient_accumulation_steps: original.core.gradient_accumulation_steps as u32,
            max_seq_len: original.core.max_seq_len as u32,
            has_eval_steps: 0,
            eval_steps: 0,
            has_save_steps: 1,
            save_steps: original.core.save_steps.unwrap() as u32,
            seed: original.core.seed,
            mixed_precision: BLAZEN_MIXED_PRECISION_NONE,
            device: std::ptr::null(),
        };
        let lora = BlazenLoraConfig {
            rank: original.lora.rank as u32,
            alpha: original.lora.alpha,
            dropout: original.lora.dropout,
            target_modules: module_ptrs.as_ptr(),
            target_modules_len: module_ptrs.len(),
        };
        let cabi = BlazenOrpoConfig {
            core,
            lora,
            lambda: original.lambda,
        };

        // SAFETY: every pointer in `cabi` points into a CString that lives at
        // least until the end of this scope.
        let round_tripped =
            unsafe { convert_orpo_config(std::ptr::from_ref(&cabi)) }.expect("round-trip");

        assert_eq!(
            round_tripped.core.base_model_repo,
            original.core.base_model_repo
        );
        assert_eq!(round_tripped.core.output_dir, original.core.output_dir);
        assert_eq!(round_tripped.core.max_steps, original.core.max_steps);
        assert_eq!(round_tripped.core.batch_size, original.core.batch_size);
        assert_eq!(round_tripped.core.save_steps, original.core.save_steps);
        assert_eq!(round_tripped.core.seed, original.core.seed);
        assert_eq!(round_tripped.lora.rank, original.lora.rank);
        assert_eq!(
            round_tripped.lora.target_modules,
            original.lora.target_modules
        );
        assert!((round_tripped.lambda - original.lambda).abs() < f32::EPSILON);
        assert!(
            (round_tripped.core.optim.gradient_clip.unwrap()
                - original.core.optim.gradient_clip.unwrap())
            .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn preference_jsonl_dataset_from_path_rejects_nonexistent_file() {
        let bad = CString::new("/this/path/definitely/does/not/exist/preference.jsonl").unwrap();
        let tk = CString::new("/this/path/also/does/not/exist/tokenizer.json").unwrap();
        let mut err: *mut BlazenError = std::ptr::null_mut();
        // SAFETY: pointers are valid for the duration of the call; `err` is a
        // single-writer destination.
        let ds = unsafe {
            blazen_preference_jsonl_dataset_from_path(
                bad.as_ptr(),
                tk.as_ptr(),
                std::ptr::null(),
                32,
                std::ptr::null(),
                0,
                std::ptr::addr_of_mut!(err),
            )
        };
        assert!(ds.is_null(), "expected null dataset on missing file");
        assert!(!err.is_null(), "expected non-null err on missing file");
        // SAFETY: err was just populated above; reclaim the BlazenError box.
        unsafe {
            drop(Box::from_raw(err));
        }
    }

    #[test]
    fn full_finetune_result_free_safe_with_null() {
        // SAFETY: explicit null input — the free path must short-circuit
        // without UB.
        unsafe {
            blazen_full_finetune_result_free(std::ptr::null_mut());
        }
        // Also exercise the "record present but output_dir is null" path
        // (e.g. a stack value that was zeroed but never populated).
        let mut empty = BlazenFullFineTuneResult {
            output_dir: std::ptr::null_mut(),
            final_loss: 0.0,
            steps_completed: 0,
        };
        // SAFETY: `empty` is a live stack value; output_dir is null and the
        // free path must tolerate that.
        unsafe {
            blazen_full_finetune_result_free(std::ptr::addr_of_mut!(empty));
        }
    }
}
