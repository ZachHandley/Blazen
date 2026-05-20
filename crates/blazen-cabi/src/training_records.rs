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
use blazen_train::dataset::JsonlDataset as InnerJsonlDataset;
#[cfg(feature = "training")]
use blazen_train::{
    LoraConfig as InnerLoraConfig, MixedPrecision as InnerMixedPrecision,
    OptimConfig as InnerOptimConfig, SchedulerConfig as InnerSchedulerConfig,
    SchedulerKind as InnerSchedulerKind, TrainConfig as InnerTrainConfig,
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
