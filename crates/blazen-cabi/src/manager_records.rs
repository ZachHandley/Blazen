//! Records returned by the [`crate::manager`] surface.
//!
//! Three opaque list handles are exposed — adapter status, model status, and
//! pool status — each with a `_len` plus a per-index accessor mirror of the
//! `BlazenVoiceHandleArray` pattern in `compute_results.rs`. Individual entry
//! values are read via flat C accessor functions returning either
//! heap-allocated C strings (free with [`crate::string::blazen_string_free`])
//! or `u64`/`f64`/`bool` scalars.
//!
//! ## Mount-strategy tags
//!
//! [`AdapterMountStrategy`] is enum-shaped in Rust. The C ABI flattens it to
//! the `BLAZEN_ADAPTER_MOUNT_STRATEGY_*` constants below so FFI hosts can
//! switch on the value without depending on cbindgen's enum emission.

#![allow(dead_code)]

use std::ffi::c_char;

use blazen_llm::{AdapterMountStrategy, AdapterStatus as InnerAdapterStatus};
use blazen_manager::ModelStatus as InnerModelStatus;

use crate::string::alloc_cstring;

/// Backend tag for [`BlazenHfLoadOptions::backend_hint`] — mistral.rs
/// (broad architecture coverage, safetensors + GGUF, multimodal).
pub const BLAZEN_BACKEND_HINT_MISTRALRS: i32 = 0;
/// Backend tag for [`BlazenHfLoadOptions::backend_hint`] — pure-Rust candle.
pub const BLAZEN_BACKEND_HINT_CANDLE: i32 = 1;
/// Backend tag for [`BlazenHfLoadOptions::backend_hint`] — llama.cpp
/// (GGUF only, best CPU performance).
pub const BLAZEN_BACKEND_HINT_LLAMACPP: i32 = 2;
/// Sentinel for [`BlazenHfLoadOptions::backend_hint`] meaning "no hint —
/// auto-detect from the repo layout".
pub const BLAZEN_BACKEND_HINT_NONE: i32 = -1;

/// `#[repr(C)]` mirror of [`blazen_manager::hf_loader::HfLoadOptions`] for the
/// C ABI. Every pointer field is nullable; integer/byte fields use sentinel
/// values to mean "unset" (see per-field docs).
///
/// Construct in C as a stack value, populate the fields you care about, then
/// pass the address to [`crate::manager::blazen_model_manager_load_from_hf`]
/// or [`crate::manager::blazen_model_manager_load_from_hf_blocking`]. The
/// strings are copied during the call — the caller may free them as soon as
/// the function returns (sync) or as soon as `BlazenFuture*` has been spawned
/// (async). The struct itself is borrowed; do not keep it alive beyond the
/// call boundary.
#[repr(C)]
pub struct BlazenHfLoadOptions {
    /// One of `BLAZEN_BACKEND_HINT_*`. Use [`BLAZEN_BACKEND_HINT_NONE`] (-1)
    /// to let the loader auto-detect from the repo's file layout.
    pub backend_hint: i32,
    /// Git revision (branch, tag, or commit sha). Null = default branch.
    pub revision: *const c_char,
    /// Hugging Face access token. Null falls back to `$HF_TOKEN`, then
    /// anonymous.
    pub hf_token: *const c_char,
    /// Override the on-disk cache directory used by `hf-hub`. Null uses the
    /// upstream default (`$HF_HOME` or `~/.cache/huggingface/`).
    pub cache_dir: *const c_char,
    /// Device specifier forwarded to the chosen provider (`"cpu"`,
    /// `"cuda:0"`, `"metal"`, …). Null uses the provider default.
    pub device: *const c_char,
    /// Explicit GGUF filename for repos that ship multiple quantizations.
    /// Required when [`choose_backend`](blazen_manager::hf_loader::choose_backend)
    /// would otherwise pick `Llamacpp` from a repo with multiple `*.gguf`
    /// siblings.
    pub gguf_file: *const c_char,
    /// Override the manager's memory budgeting estimate, in bytes. `0` =
    /// unset (manager sums the chosen backend's weight files from repo
    /// metadata).
    pub memory_estimate_bytes: u64,
    /// Target pool label (`"cpu"` / `"gpu"` / `"gpu:N"`). Null defaults to
    /// `Pool::Cpu`. An unparseable label surfaces as a validation error on
    /// the call that consumes the options struct.
    pub pool: *const c_char,
}

/// Tag for [`AdapterMountStrategy::Attached`].
pub const BLAZEN_ADAPTER_MOUNT_STRATEGY_ATTACHED: u32 = 1;
/// Tag for [`AdapterMountStrategy::Rebuilt`].
pub const BLAZEN_ADAPTER_MOUNT_STRATEGY_REBUILT: u32 = 2;
/// Tag for [`AdapterMountStrategy::Merged`].
pub const BLAZEN_ADAPTER_MOUNT_STRATEGY_MERGED: u32 = 3;

pub(crate) fn mount_strategy_tag(s: AdapterMountStrategy) -> u32 {
    match s {
        AdapterMountStrategy::Attached => BLAZEN_ADAPTER_MOUNT_STRATEGY_ATTACHED,
        AdapterMountStrategy::Rebuilt => BLAZEN_ADAPTER_MOUNT_STRATEGY_REBUILT,
        AdapterMountStrategy::Merged => BLAZEN_ADAPTER_MOUNT_STRATEGY_MERGED,
    }
}

// ---------------------------------------------------------------------------
// BlazenAdapterStatus
// ---------------------------------------------------------------------------

/// Opaque snapshot of one mounted adapter — wraps
/// [`blazen_llm::AdapterStatus`] plus the mount-strategy tag reported by the
/// underlying [`AdapterHandle`](blazen_llm::AdapterHandle) at mount time
/// (status snapshots stored on the manager don't normally carry that field;
/// the manager records it on a per-handle basis instead, see
/// [`crate::manager::blazen_model_manager_load_adapter_blocking`]).
pub struct BlazenAdapterStatus {
    pub(crate) inner: InnerAdapterStatus,
    pub(crate) mount_strategy: AdapterMountStrategy,
}

impl BlazenAdapterStatus {
    pub(crate) fn into_ptr(self) -> *mut BlazenAdapterStatus {
        Box::into_raw(Box::new(self))
    }
}

/// Returns the adapter id as a caller-owned C string. Null on a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenAdapterStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_adapter_id(
    s: *const BlazenAdapterStatus,
) -> *mut c_char {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    alloc_cstring(&s.inner.adapter_id)
}

/// Returns the adapter source directory as a caller-owned C string. Null on
/// a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenAdapterStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_source_dir(
    s: *const BlazenAdapterStatus,
) -> *mut c_char {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    alloc_cstring(&s.inner.source_dir.display().to_string())
}

/// Returns the adapter scale (delta-weight multiplier). Returns `0.0` on a
/// null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenAdapterStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_scale(s: *const BlazenAdapterStatus) -> f64 {
    if s.is_null() {
        return 0.0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    f64::from(s.inner.scale)
}

/// Returns the adapter's runtime memory footprint in bytes (as reported by
/// the backend at mount time). Returns `0` on a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenAdapterStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_memory_bytes(s: *const BlazenAdapterStatus) -> u64 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.inner.memory_bytes
}

/// Returns the mount-strategy tag (one of `BLAZEN_ADAPTER_MOUNT_STRATEGY_*`).
/// Returns `0` on a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenAdapterStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_mount_strategy(
    s: *const BlazenAdapterStatus,
) -> u32 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    mount_strategy_tag(s.mount_strategy)
}

/// Frees a [`BlazenAdapterStatus`] handle. No-op on a null pointer.
///
/// # Safety
///
/// `s` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_free(s: *mut BlazenAdapterStatus) {
    if s.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(s) });
}

// ---------------------------------------------------------------------------
// BlazenAdapterStatusList
// ---------------------------------------------------------------------------

/// Opaque list of [`BlazenAdapterStatus`] snapshots. Owned by the caller —
/// free with [`blazen_adapter_status_list_free`]. Iterate by calling
/// [`blazen_adapter_status_list_len`] and either
/// [`blazen_adapter_status_list_get`] (borrows the entry, no allocation) or
/// [`blazen_adapter_status_list_take`] (transfers ownership of one entry).
pub struct BlazenAdapterStatusList {
    pub(crate) inner: Vec<BlazenAdapterStatus>,
}

impl BlazenAdapterStatusList {
    pub(crate) fn into_ptr(self) -> *mut BlazenAdapterStatusList {
        Box::into_raw(Box::new(self))
    }

    pub(crate) fn from_statuses(items: Vec<InnerAdapterStatus>) -> Self {
        Self {
            inner: items
                .into_iter()
                .map(|inner| BlazenAdapterStatus {
                    inner,
                    mount_strategy: AdapterMountStrategy::Attached,
                })
                .collect(),
        }
    }
}

/// Returns the number of entries in the list. Returns `0` on a null handle.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenAdapterStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_list_len(
    list: *const BlazenAdapterStatusList,
) -> usize {
    if list.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner.len()
}

/// Borrows the `idx`-th entry as a `*const BlazenAdapterStatus`. The borrow
/// is valid until the list is freed or modified. Returns null if `list` is
/// null or `idx` is out of range.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenAdapterStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_list_get(
    list: *const BlazenAdapterStatusList,
    idx: usize,
) -> *const BlazenAdapterStatus {
    if list.is_null() {
        return std::ptr::null();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner
        .get(idx)
        .map_or(std::ptr::null(), std::ptr::from_ref)
}

/// Pops the `idx`-th entry and returns it as a caller-owned
/// [`BlazenAdapterStatus`] handle (free with
/// [`blazen_adapter_status_free`]). Returns null if `list` is null or `idx`
/// is out of range. The list shrinks by one on success.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenAdapterStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_list_take(
    list: *mut BlazenAdapterStatusList,
    idx: usize,
) -> *mut BlazenAdapterStatus {
    if list.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &mut *list };
    if idx >= l.inner.len() {
        return std::ptr::null_mut();
    }
    let entry = l.inner.remove(idx);
    entry.into_ptr()
}

/// Frees a [`BlazenAdapterStatusList`], dropping any remaining entries.
///
/// # Safety
///
/// `list` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_status_list_free(list: *mut BlazenAdapterStatusList) {
    if list.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(list) });
}

// ---------------------------------------------------------------------------
// BlazenModelStatus
// ---------------------------------------------------------------------------

/// Opaque snapshot of one registered model — wraps
/// [`blazen_manager::ModelStatus`]. The pool label is rendered via
/// [`Pool::Display`](blazen_llm::Pool); adapters are exposed as a count plus
/// a take-by-index accessor that produces a fresh list handle scoped to one
/// model.
pub struct BlazenModelStatus(pub(crate) InnerModelStatus);

impl BlazenModelStatus {
    pub(crate) fn into_ptr(self) -> *mut BlazenModelStatus {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerModelStatus> for BlazenModelStatus {
    fn from(inner: InnerModelStatus) -> Self {
        Self(inner)
    }
}

/// Returns the model id as a caller-owned C string. Null on a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenModelStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_id(s: *const BlazenModelStatus) -> *mut c_char {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    alloc_cstring(&s.0.id)
}

/// Returns `true` if the model is currently loaded. Returns `false` on a
/// null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenModelStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_loaded(s: *const BlazenModelStatus) -> bool {
    if s.is_null() {
        return false;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.0.loaded
}

/// Returns the model's estimated memory footprint in bytes. Returns `0` on a
/// null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenModelStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_memory_bytes(s: *const BlazenModelStatus) -> u64 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.0.memory_estimate_bytes
}

/// Returns the pool label (`"cpu"` or `"gpu:N"`) as a caller-owned C string.
/// Null on a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenModelStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_pool(s: *const BlazenModelStatus) -> *mut c_char {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    alloc_cstring(&format!("{}", s.0.pool))
}

/// Returns the number of adapters mounted on this model. Returns `0` on a
/// null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenModelStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_adapters_count(s: *const BlazenModelStatus) -> usize {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.0.adapters.len()
}

/// Returns a caller-owned [`BlazenAdapterStatusList`] cloned from the
/// model's mounted-adapters snapshot. Null on a null handle. Free with
/// [`blazen_adapter_status_list_free`].
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenModelStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_adapters(
    s: *const BlazenModelStatus,
) -> *mut BlazenAdapterStatusList {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    BlazenAdapterStatusList::from_statuses(s.0.adapters.clone()).into_ptr()
}

/// Frees a [`BlazenModelStatus`] handle.
///
/// # Safety
///
/// `s` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_free(s: *mut BlazenModelStatus) {
    if s.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(s) });
}

// ---------------------------------------------------------------------------
// BlazenModelStatusList
// ---------------------------------------------------------------------------

/// Opaque list of [`BlazenModelStatus`] snapshots. Same access pattern as
/// [`BlazenAdapterStatusList`].
pub struct BlazenModelStatusList {
    pub(crate) inner: Vec<BlazenModelStatus>,
}

impl BlazenModelStatusList {
    pub(crate) fn into_ptr(self) -> *mut BlazenModelStatusList {
        Box::into_raw(Box::new(self))
    }
}

impl From<Vec<InnerModelStatus>> for BlazenModelStatusList {
    fn from(items: Vec<InnerModelStatus>) -> Self {
        Self {
            inner: items.into_iter().map(BlazenModelStatus::from).collect(),
        }
    }
}

/// Returns the number of entries in the list. Returns `0` on a null handle.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenModelStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_list_len(list: *const BlazenModelStatusList) -> usize {
    if list.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner.len()
}

/// Borrows the `idx`-th entry. Returns null if `list` is null or `idx` is
/// out of range.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenModelStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_list_get(
    list: *const BlazenModelStatusList,
    idx: usize,
) -> *const BlazenModelStatus {
    if list.is_null() {
        return std::ptr::null();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner
        .get(idx)
        .map_or(std::ptr::null(), std::ptr::from_ref)
}

/// Pops the `idx`-th entry and returns it as a caller-owned handle. Returns
/// null if `list` is null or `idx` is out of range.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenModelStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_list_take(
    list: *mut BlazenModelStatusList,
    idx: usize,
) -> *mut BlazenModelStatus {
    if list.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &mut *list };
    if idx >= l.inner.len() {
        return std::ptr::null_mut();
    }
    l.inner.remove(idx).into_ptr()
}

/// Frees a [`BlazenModelStatusList`], dropping any remaining entries.
///
/// # Safety
///
/// `list` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_model_status_list_free(list: *mut BlazenModelStatusList) {
    if list.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(list) });
}

// ---------------------------------------------------------------------------
// BlazenPoolStatus
// ---------------------------------------------------------------------------

/// Opaque snapshot of one configured pool: its label, total budget, current
/// usage, and the number of currently-loaded models. Produced by
/// [`crate::manager::blazen_model_manager_pools`] (which packages the
/// `(Pool, budget)` pairs reported by [`blazen_manager::ModelManager::pools`]
/// together with live usage / count from the status snapshot).
pub struct BlazenPoolStatus {
    pub(crate) pool_label: String,
    pub(crate) budget_bytes: u64,
    pub(crate) used_bytes: u64,
    pub(crate) loaded_models: usize,
}

impl BlazenPoolStatus {
    pub(crate) fn into_ptr(self) -> *mut BlazenPoolStatus {
        Box::into_raw(Box::new(self))
    }
}

/// Returns the pool label as a caller-owned C string. Null on a null handle.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenPoolStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_id(s: *const BlazenPoolStatus) -> *mut c_char {
    if s.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    alloc_cstring(&s.pool_label)
}

/// Returns the pool's total budget in bytes.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenPoolStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_budget_bytes(s: *const BlazenPoolStatus) -> u64 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.budget_bytes
}

/// Returns bytes currently used by loaded models in this pool.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenPoolStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_used_bytes(s: *const BlazenPoolStatus) -> u64 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.used_bytes
}

/// Returns the number of currently-loaded models charged to this pool.
///
/// # Safety
///
/// `s` must be null OR a live [`BlazenPoolStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_loaded_models(s: *const BlazenPoolStatus) -> usize {
    if s.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let s = unsafe { &*s };
    s.loaded_models
}

/// Frees a [`BlazenPoolStatus`] handle.
///
/// # Safety
///
/// `s` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_free(s: *mut BlazenPoolStatus) {
    if s.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(s) });
}

// ---------------------------------------------------------------------------
// BlazenPoolStatusList
// ---------------------------------------------------------------------------

/// Opaque list of [`BlazenPoolStatus`] snapshots.
pub struct BlazenPoolStatusList {
    pub(crate) inner: Vec<BlazenPoolStatus>,
}

impl BlazenPoolStatusList {
    pub(crate) fn into_ptr(self) -> *mut BlazenPoolStatusList {
        Box::into_raw(Box::new(self))
    }
}

/// Returns the number of entries in the list.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenPoolStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_list_len(list: *const BlazenPoolStatusList) -> usize {
    if list.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner.len()
}

/// Borrows the `idx`-th entry. Returns null on null `list` or out-of-range
/// `idx`.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenPoolStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_list_get(
    list: *const BlazenPoolStatusList,
    idx: usize,
) -> *const BlazenPoolStatus {
    if list.is_null() {
        return std::ptr::null();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner
        .get(idx)
        .map_or(std::ptr::null(), std::ptr::from_ref)
}

/// Pops the `idx`-th entry and returns a caller-owned handle.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenPoolStatusList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_list_take(
    list: *mut BlazenPoolStatusList,
    idx: usize,
) -> *mut BlazenPoolStatus {
    if list.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &mut *list };
    if idx >= l.inner.len() {
        return std::ptr::null_mut();
    }
    l.inner.remove(idx).into_ptr()
}

/// Frees a [`BlazenPoolStatusList`], dropping any remaining entries.
///
/// # Safety
///
/// `list` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pool_status_list_free(list: *mut BlazenPoolStatusList) {
    if list.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(list) });
}

// ---------------------------------------------------------------------------
// BlazenAdapterHandleInfo
// ---------------------------------------------------------------------------

/// Opaque result returned by
/// [`crate::manager::blazen_model_manager_load_adapter_blocking`] and its
/// future variant. Carries the adapter id echoed by the backend plus the
/// runtime memory footprint and the mount-strategy tag.
pub struct BlazenAdapterHandleInfo {
    pub(crate) adapter_id: String,
    pub(crate) memory_bytes: u64,
    pub(crate) mount_strategy: AdapterMountStrategy,
}

impl BlazenAdapterHandleInfo {
    pub(crate) fn into_ptr(self) -> *mut BlazenAdapterHandleInfo {
        Box::into_raw(Box::new(self))
    }
}

impl From<blazen_llm::AdapterHandle> for BlazenAdapterHandleInfo {
    fn from(h: blazen_llm::AdapterHandle) -> Self {
        Self {
            adapter_id: h.adapter_id,
            memory_bytes: h.memory_bytes,
            mount_strategy: h.mount_strategy,
        }
    }
}

/// Returns the adapter id as a caller-owned C string. Null on a null handle.
///
/// # Safety
///
/// `h` must be null OR a live [`BlazenAdapterHandleInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_handle_info_adapter_id(
    h: *const BlazenAdapterHandleInfo,
) -> *mut c_char {
    if h.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let h = unsafe { &*h };
    alloc_cstring(&h.adapter_id)
}

/// Returns the adapter's runtime memory footprint in bytes.
///
/// # Safety
///
/// `h` must be null OR a live [`BlazenAdapterHandleInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_handle_info_memory_bytes(
    h: *const BlazenAdapterHandleInfo,
) -> u64 {
    if h.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let h = unsafe { &*h };
    h.memory_bytes
}

/// Returns the mount-strategy tag (one of `BLAZEN_ADAPTER_MOUNT_STRATEGY_*`).
///
/// # Safety
///
/// `h` must be null OR a live [`BlazenAdapterHandleInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_handle_info_mount_strategy(
    h: *const BlazenAdapterHandleInfo,
) -> u32 {
    if h.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let h = unsafe { &*h };
    mount_strategy_tag(h.mount_strategy)
}

/// Frees a [`BlazenAdapterHandleInfo`] handle.
///
/// # Safety
///
/// `h` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_adapter_handle_info_free(h: *mut BlazenAdapterHandleInfo) {
    if h.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(h) });
}
