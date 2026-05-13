//! Persistence opaque object: [`BlazenCheckpointStore`].
//!
//! Wraps an `Arc<blazen_uniffi::persist::CheckpointStore>` (which itself
//! abstracts any `blazen_persist::CheckpointStore` backend) and exposes
//! its five async methods through both `*_blocking` and future-returning
//! C entry points:
//!
//! - `save(checkpoint)` — persist a checkpoint, overwriting any prior
//!   entry with the same `run_id`. The input `BlazenWorkflowCheckpoint`
//!   handle is **consumed** by the call (whether sync or async) — callers
//!   must NOT separately free it.
//! - `load(run_id)` — fetch a checkpoint by run id; returns a found / not
//!   found discriminator alongside the optional handle.
//! - `delete(run_id)` — drop the checkpoint for a given run id.
//! - `list()` — return every stored checkpoint as a heap-allocated array
//!   of `*mut BlazenWorkflowCheckpoint`.
//! - `list_run_ids()` — return every stored run id (UUID string) as a
//!   heap-allocated array of `*mut c_char`.
//!
//! ## Ownership conventions
//!
//! - Store handles are heap-allocated `Box<BlazenCheckpointStore>` and
//!   freed via [`blazen_checkpoint_store_free`].
//! - The list-returning methods produce a heap-allocated, contiguous,
//!   caller-owned slice of element pointers. The accompanying free
//!   helpers ([`blazen_workflow_checkpoint_array_free`] and
//!   [`blazen_string_array_free`]) drop each element AND the backing
//!   slice in one call.
//! - Error handles produced on the failure path are caller-owned and
//!   freed via [`crate::error::blazen_error_free`].

#![allow(dead_code)]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::persist::{
    CheckpointStore as InnerCheckpointStore, WorkflowCheckpoint as InnerWorkflowCheckpoint,
};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::persist_records::BlazenWorkflowCheckpoint;
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `Vec<InnerWorkflowCheckpoint>` into a heap-allocated array of
/// `*mut BlazenWorkflowCheckpoint`. Returns `(base_ptr, count)`.
///
/// Each element is wrapped via [`BlazenWorkflowCheckpoint::into_ptr`] so it
/// can be freed individually. The backing slice is leaked via
/// `Box::into_raw` so the FFI host can index into it; the matching
/// [`blazen_workflow_checkpoint_array_free`] reconstructs the slice via
/// `slice::from_raw_parts_mut` + `Box::from_raw` to release everything.
fn checkpoints_to_c_array(
    items: Vec<InnerWorkflowCheckpoint>,
) -> (*mut *mut BlazenWorkflowCheckpoint, usize) {
    let boxed: Box<[*mut BlazenWorkflowCheckpoint]> = items
        .into_iter()
        .map(|v| BlazenWorkflowCheckpoint::from(v).into_ptr())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let len = boxed.len();
    let raw = Box::into_raw(boxed);
    // `*mut [T]` -> `*mut T` via `.cast()` keeps provenance correct as long
    // as the matching free reconstructs the same `[T]` shape via
    // `slice::from_raw_parts_mut(base, len)` + `Box::from_raw`.
    (raw.cast::<*mut BlazenWorkflowCheckpoint>(), len)
}

/// Convert a `Vec<String>` into a heap-allocated array of `*mut c_char`.
/// Returns `(base_ptr, count)`.
///
/// Each element goes through [`alloc_cstring`] so it can be freed via
/// `blazen_string_free`. The slice itself is freed via
/// [`blazen_string_array_free`]. An interior-NUL string collapses to a
/// null element pointer (matching `alloc_cstring`'s behavior) — the array
/// free still tolerates null entries.
fn strings_to_c_array(items: Vec<String>) -> (*mut *mut c_char, usize) {
    let boxed: Box<[*mut c_char]> = items
        .into_iter()
        .map(|s| alloc_cstring(&s))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let len = boxed.len();
    let raw = Box::into_raw(boxed);
    (raw.cast::<*mut c_char>(), len)
}

// ---------------------------------------------------------------------------
// BlazenCheckpointStore
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::persist::CheckpointStore>`.
///
/// Produced by the per-backend store factories (`new_redb_checkpoint_store`,
/// `new_valkey_checkpoint_store`) wired in Phase R4. Free with
/// [`blazen_checkpoint_store_free`].
pub struct BlazenCheckpointStore(pub(crate) Arc<InnerCheckpointStore>);

impl BlazenCheckpointStore {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenCheckpointStore {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerCheckpointStore>> for BlazenCheckpointStore {
    fn from(inner: Arc<InnerCheckpointStore>) -> Self {
        Self(inner)
    }
}

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

/// Synchronously persist `checkpoint`. The `checkpoint` pointer is
/// **consumed** by this call — callers must NOT separately free it.
///
/// Returns `0` on success, `-1` on failure (writes into `out_err`), `-2`
/// on invalid input (null `store` or null `checkpoint`).
///
/// # Safety
///
/// - `store` must be a valid pointer to a `BlazenCheckpointStore` produced
///   by the cabi surface.
/// - `checkpoint` must be a pointer previously produced by
///   [`crate::persist_records::blazen_workflow_checkpoint_new`] (or any
///   cabi function producing a `BlazenWorkflowCheckpoint`) and not yet
///   freed.
/// - `out_err` must be null OR a writable pointer to a `*mut BlazenError`
///   slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_save_blocking(
    store: *const BlazenCheckpointStore,
    checkpoint: *mut BlazenWorkflowCheckpoint,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if store.is_null() || checkpoint.is_null() {
        if !checkpoint.is_null() {
            // SAFETY: caller upholds the `Box::into_raw` contract on `checkpoint`.
            drop(unsafe { Box::from_raw(checkpoint) });
        }
        return -2;
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };
    // SAFETY: caller upholds the `Box::into_raw` contract on `checkpoint`.
    let checkpoint = unsafe { Box::from_raw(checkpoint) };
    let inner_checkpoint = checkpoint.0;

    let store_inner = Arc::clone(&store.0);
    let result = runtime().block_on(async move { store_inner.save(inner_checkpoint).await });
    match result {
        Ok(()) => 0,
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously persist `checkpoint`. The `checkpoint` pointer is
/// **consumed** by this call — callers must NOT separately free it.
///
/// Returns a future handle whose result is popped with
/// [`blazen_future_take_unit`]. Returns null if either input is null
/// (in which case the consumed `checkpoint` is dropped to avoid a leak).
///
/// # Safety
///
/// Same contracts as [`blazen_checkpoint_store_save_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_save(
    store: *const BlazenCheckpointStore,
    checkpoint: *mut BlazenWorkflowCheckpoint,
) -> *mut BlazenFuture {
    if store.is_null() || checkpoint.is_null() {
        if !checkpoint.is_null() {
            // SAFETY: caller upholds the `Box::into_raw` contract on `checkpoint`.
            drop(unsafe { Box::from_raw(checkpoint) });
        }
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };
    // SAFETY: caller upholds the `Box::into_raw` contract on `checkpoint`.
    let checkpoint = unsafe { Box::from_raw(checkpoint) };
    let inner_checkpoint = checkpoint.0;

    let store_inner = Arc::clone(&store.0);
    BlazenFuture::spawn::<(), _>(async move { store_inner.save(inner_checkpoint).await })
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/// Synchronously load a checkpoint by `run_id`.
///
/// On success, writes `1` into `out_found` and a freshly-allocated
/// `*mut BlazenWorkflowCheckpoint` into `out_checkpoint` (when `Some`), or
/// `0` into `out_found` (when `None`). Returns `0` on success, `-1` on
/// failure, `-2` on invalid input (null `store` / null `run_id` /
/// non-UTF-8 `run_id`).
///
/// # Safety
///
/// - `store` must be a valid pointer to a `BlazenCheckpointStore`.
/// - `run_id` must be a valid NUL-terminated UTF-8 buffer.
/// - `out_checkpoint` / `out_found` / `out_err` must be null OR writable
///   pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_load_blocking(
    store: *const BlazenCheckpointStore,
    run_id: *const c_char,
    out_checkpoint: *mut *mut BlazenWorkflowCheckpoint,
    out_found: *mut i32,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if store.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `run_id`.
    let Some(run_id) = (unsafe { cstr_to_str(run_id) }) else {
        return -2;
    };
    let run_id = run_id.to_owned();

    let store_inner = Arc::clone(&store.0);
    let result = runtime().block_on(async move { store_inner.load(run_id).await });
    match result {
        Ok(Some(v)) => {
            if !out_checkpoint.is_null() {
                // SAFETY: caller has guaranteed `out_checkpoint` is writable.
                unsafe {
                    *out_checkpoint = BlazenWorkflowCheckpoint::from(v).into_ptr();
                }
            }
            if !out_found.is_null() {
                // SAFETY: caller has guaranteed `out_found` is writable.
                unsafe {
                    *out_found = 1;
                }
            }
            0
        }
        Ok(None) => {
            if !out_found.is_null() {
                // SAFETY: caller has guaranteed `out_found` is writable.
                unsafe {
                    *out_found = 0;
                }
            }
            0
        }
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously load a checkpoint by `run_id`. Returns a future handle
/// whose result is popped with
/// [`blazen_future_take_workflow_checkpoint_option`].
///
/// Returns null on invalid input.
///
/// # Safety
///
/// Same string contracts as [`blazen_checkpoint_store_load_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_load(
    store: *const BlazenCheckpointStore,
    run_id: *const c_char,
) -> *mut BlazenFuture {
    if store.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `run_id`.
    let Some(run_id) = (unsafe { cstr_to_str(run_id) }) else {
        return std::ptr::null_mut();
    };
    let run_id = run_id.to_owned();

    let store_inner = Arc::clone(&store.0);
    BlazenFuture::spawn::<Option<InnerWorkflowCheckpoint>, _>(async move {
        store_inner.load(run_id).await
    })
}

// ---------------------------------------------------------------------------
// delete
// ---------------------------------------------------------------------------

/// Synchronously delete the checkpoint for `run_id`. The underlying
/// backends treat delete-of-missing as a no-op, so the call succeeds even
/// when no checkpoint exists for the id.
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input.
///
/// # Safety
///
/// Same contracts as [`blazen_checkpoint_store_load_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_delete_blocking(
    store: *const BlazenCheckpointStore,
    run_id: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if store.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `run_id`.
    let Some(run_id) = (unsafe { cstr_to_str(run_id) }) else {
        return -2;
    };
    let run_id = run_id.to_owned();

    let store_inner = Arc::clone(&store.0);
    let result = runtime().block_on(async move { store_inner.delete(run_id).await });
    match result {
        Ok(()) => 0,
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously delete the checkpoint for `run_id`. Returns a future
/// handle whose result is popped with [`blazen_future_take_unit`].
///
/// # Safety
///
/// Same contracts as [`blazen_checkpoint_store_delete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_delete(
    store: *const BlazenCheckpointStore,
    run_id: *const c_char,
) -> *mut BlazenFuture {
    if store.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `run_id`.
    let Some(run_id) = (unsafe { cstr_to_str(run_id) }) else {
        return std::ptr::null_mut();
    };
    let run_id = run_id.to_owned();

    let store_inner = Arc::clone(&store.0);
    BlazenFuture::spawn::<(), _>(async move { store_inner.delete(run_id).await })
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

/// Synchronously list every stored checkpoint, ordered by timestamp
/// descending.
///
/// On success writes a heap-allocated array of
/// `*mut BlazenWorkflowCheckpoint` into `*out_array` plus its length into
/// `*out_count`. Free the array with
/// [`blazen_workflow_checkpoint_array_free`].
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input.
///
/// # Safety
///
/// `store` must be a valid pointer to a `BlazenCheckpointStore`.
/// `out_array` / `out_count` / `out_err` must be null OR writable pointers
/// to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_list_blocking(
    store: *const BlazenCheckpointStore,
    out_array: *mut *mut *mut BlazenWorkflowCheckpoint,
    out_count: *mut usize,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if store.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };

    let store_inner = Arc::clone(&store.0);
    let result = runtime().block_on(async move { store_inner.list().await });
    match result {
        Ok(items) => {
            let (base, count) = checkpoints_to_c_array(items);
            if out_array.is_null() {
                // Caller doesn't want the array — release it immediately.
                // SAFETY: `base` + `count` were just produced by
                // `checkpoints_to_c_array`; reconstructing the boxed slice is
                // sound. The `Box<[_]>` drop releases the slice; each element
                // pointer is released via `Box::from_raw` below.
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(base, count);
                    let owned = Box::from_raw(slice);
                    for &ptr in &owned {
                        if !ptr.is_null() {
                            drop(Box::from_raw(ptr));
                        }
                    }
                    drop(owned);
                }
            } else {
                // SAFETY: caller has guaranteed `out_array` is writable.
                unsafe {
                    *out_array = base;
                }
            }
            if !out_count.is_null() {
                // SAFETY: caller has guaranteed `out_count` is writable.
                unsafe {
                    *out_count = count;
                }
            }
            0
        }
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously list every stored checkpoint. Returns a future whose
/// result is popped with [`blazen_future_take_workflow_checkpoint_list`].
///
/// # Safety
///
/// `store` must be a valid pointer to a `BlazenCheckpointStore`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_list(
    store: *const BlazenCheckpointStore,
) -> *mut BlazenFuture {
    if store.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };

    let store_inner = Arc::clone(&store.0);
    BlazenFuture::spawn::<Vec<InnerWorkflowCheckpoint>, _>(async move { store_inner.list().await })
}

// ---------------------------------------------------------------------------
// list_run_ids
// ---------------------------------------------------------------------------

/// Synchronously list every stored run id, ordered by timestamp
/// descending.
///
/// On success writes a heap-allocated array of `*mut c_char` into
/// `*out_array` plus its length into `*out_count`. Free the array with
/// [`blazen_string_array_free`].
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input.
///
/// # Safety
///
/// Same contracts as [`blazen_checkpoint_store_list_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_list_run_ids_blocking(
    store: *const BlazenCheckpointStore,
    out_array: *mut *mut *mut c_char,
    out_count: *mut usize,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if store.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };

    let store_inner = Arc::clone(&store.0);
    let result = runtime().block_on(async move { store_inner.list_run_ids().await });
    match result {
        Ok(items) => {
            let (base, count) = strings_to_c_array(items);
            if out_array.is_null() {
                // Caller doesn't want the array — release it immediately.
                // SAFETY: `base` + `count` were just produced by
                // `strings_to_c_array`; reconstructing the boxed slice is
                // sound and `Box::from_raw` on each non-null element pointer
                // returns the original `CString` allocation.
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(base, count);
                    let owned = Box::from_raw(slice);
                    for &ptr in &owned {
                        if !ptr.is_null() {
                            drop(std::ffi::CString::from_raw(ptr));
                        }
                    }
                    drop(owned);
                }
            } else {
                // SAFETY: caller has guaranteed `out_array` is writable.
                unsafe {
                    *out_array = base;
                }
            }
            if !out_count.is_null() {
                // SAFETY: caller has guaranteed `out_count` is writable.
                unsafe {
                    *out_count = count;
                }
            }
            0
        }
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously list every stored run id. Returns a future whose result
/// is popped with [`blazen_future_take_string_list`].
///
/// # Safety
///
/// `store` must be a valid pointer to a `BlazenCheckpointStore`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_list_run_ids(
    store: *const BlazenCheckpointStore,
) -> *mut BlazenFuture {
    if store.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `store` is a live `BlazenCheckpointStore`.
    let store = unsafe { &*store };

    let store_inner = Arc::clone(&store.0);
    BlazenFuture::spawn::<Vec<String>, _>(async move { store_inner.list_run_ids().await })
}

// ---------------------------------------------------------------------------
// free
// ---------------------------------------------------------------------------

/// Frees a `BlazenCheckpointStore` handle. No-op on a null pointer.
///
/// # Safety
///
/// `store` must be null OR a pointer previously produced by the cabi
/// surface's checkpoint-store factory functions. Double-free is undefined
/// behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_free(store: *mut BlazenCheckpointStore) {
    if store.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(store) });
}

// ---------------------------------------------------------------------------
// Array free helpers
// ---------------------------------------------------------------------------

/// Frees an array of `*mut BlazenWorkflowCheckpoint` previously produced
/// by [`blazen_checkpoint_store_list_blocking`] or
/// [`blazen_future_take_workflow_checkpoint_list`].
///
/// Releases each element handle AND the backing slice in one call. No-op
/// on a null `arr` (regardless of `count`).
///
/// # Safety
///
/// `arr` must be null OR a pointer previously produced by the cabi
/// surface's checkpoint-list entry points, with `count` matching its
/// length. Double-free is undefined behavior; modifying or freeing
/// individual element pointers before this call is also undefined.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_array_free(
    arr: *mut *mut BlazenWorkflowCheckpoint,
    count: usize,
) {
    if arr.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `arr` + `count` describe a live
    // `Box<[*mut BlazenWorkflowCheckpoint]>` allocation. Reconstructing the
    // slice and reboxing releases the backing storage; each non-null
    // element pointer is then `Box::from_raw`'d back to release its
    // checkpoint allocation.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(arr, count);
        let owned = Box::from_raw(slice);
        for &ptr in &owned {
            if !ptr.is_null() {
                drop(Box::from_raw(ptr));
            }
        }
        drop(owned);
    }
}

/// Frees an array of `*mut c_char` previously produced by
/// [`blazen_checkpoint_store_list_run_ids_blocking`] or
/// [`blazen_future_take_string_list`].
///
/// Releases each element string AND the backing slice in one call. No-op
/// on a null `arr` (regardless of `count`).
///
/// # Safety
///
/// `arr` must be null OR a pointer previously produced by the cabi
/// surface's string-list entry points, with `count` matching its length.
/// Double-free is undefined behavior; modifying or freeing individual
/// element pointers before this call is also undefined.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_string_array_free(arr: *mut *mut c_char, count: usize) {
    if arr.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `arr` + `count` describe a live
    // `Box<[*mut c_char]>` allocation. Reconstructing the slice and reboxing
    // releases the backing storage; each non-null element pointer is then
    // `CString::from_raw`'d back to release its string allocation.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(arr, count);
        let owned = Box::from_raw(slice);
        for &ptr in &owned {
            if !ptr.is_null() {
                drop(std::ffi::CString::from_raw(ptr));
            }
        }
        drop(owned);
    }
}

// ---------------------------------------------------------------------------
// Typed future-take entry points
// ---------------------------------------------------------------------------

/// Pops a `()` result out of `fut` — used for `save` and `delete`. On
/// success returns `0`; on failure returns `-1` and writes a caller-owned
/// `*mut BlazenError` into `err`.
///
/// `err` may be null when the caller wants to discard the error.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by a cabi async wrapper
/// whose underlying future returns `Result<(), BlazenError>`, not yet
/// freed. `err` must be null OR a writable pointer to a `*mut BlazenError`
/// slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_unit(
    fut: *mut BlazenFuture,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<()>(fut) } {
        Ok(()) => 0,
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller has guaranteed `err` is writable.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Pops an `Option<WorkflowCheckpoint>` result out of `fut`. Mirrors the
/// blocking `load` semantics:
///
/// - On success Some: returns `0`, writes a fresh handle into `out`, and
///   writes `1` into `out_found`.
/// - On success None: returns `0` and writes `0` into `out_found`.
/// - On failure: returns `-1` and writes a `*mut BlazenError` into `err`.
///
/// `out` / `out_found` / `err` may individually be null to discard.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_checkpoint_store_load`], not yet freed. `out` / `out_found` /
/// `err` must be null OR writable pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_workflow_checkpoint_option(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenWorkflowCheckpoint,
    out_found: *mut i32,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<Option<InnerWorkflowCheckpoint>>(fut) } {
        Ok(Some(v)) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenWorkflowCheckpoint::from(v).into_ptr();
                }
            }
            if !out_found.is_null() {
                // SAFETY: caller has guaranteed `out_found` is writable.
                unsafe {
                    *out_found = 1;
                }
            }
            0
        }
        Ok(None) => {
            if !out_found.is_null() {
                // SAFETY: caller has guaranteed `out_found` is writable.
                unsafe {
                    *out_found = 0;
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller has guaranteed `err` is writable.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Pops a `Vec<WorkflowCheckpoint>` result out of `fut`. Mirrors the
/// blocking `list` semantics — on success returns `0` and writes a
/// heap-allocated array of `*mut BlazenWorkflowCheckpoint` into
/// `*out_array` with its length in `*out_count`. Free with
/// [`blazen_workflow_checkpoint_array_free`].
///
/// `out_array` / `out_count` / `err` may individually be null to discard.
/// When `out_array` is null the array is freed immediately to avoid a
/// leak.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_checkpoint_store_list`], not yet freed. `out_array` /
/// `out_count` / `err` must be null OR writable pointers to the
/// appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_workflow_checkpoint_list(
    fut: *mut BlazenFuture,
    out_array: *mut *mut *mut BlazenWorkflowCheckpoint,
    out_count: *mut usize,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<Vec<InnerWorkflowCheckpoint>>(fut) } {
        Ok(items) => {
            let (base, count) = checkpoints_to_c_array(items);
            if out_array.is_null() {
                // SAFETY: `base` + `count` describe the array just produced;
                // reconstructing the boxed slice is sound.
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(base, count);
                    let owned = Box::from_raw(slice);
                    for &ptr in &owned {
                        if !ptr.is_null() {
                            drop(Box::from_raw(ptr));
                        }
                    }
                    drop(owned);
                }
            } else {
                // SAFETY: caller has guaranteed `out_array` is writable.
                unsafe {
                    *out_array = base;
                }
            }
            if !out_count.is_null() {
                // SAFETY: caller has guaranteed `out_count` is writable.
                unsafe {
                    *out_count = count;
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller has guaranteed `err` is writable.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Pops a `Vec<String>` result out of `fut`. Mirrors the blocking
/// `list_run_ids` semantics — on success returns `0` and writes a
/// heap-allocated array of `*mut c_char` into `*out_array` with its
/// length in `*out_count`. Free with [`blazen_string_array_free`].
///
/// `out_array` / `out_count` / `err` may individually be null to discard.
/// When `out_array` is null the array is freed immediately to avoid a
/// leak.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_checkpoint_store_list_run_ids`], not yet freed. `out_array` /
/// `out_count` / `err` must be null OR writable pointers to the
/// appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_string_list(
    fut: *mut BlazenFuture,
    out_array: *mut *mut *mut c_char,
    out_count: *mut usize,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<Vec<String>>(fut) } {
        Ok(items) => {
            let (base, count) = strings_to_c_array(items);
            if out_array.is_null() {
                // SAFETY: `base` + `count` describe the array just produced;
                // reconstructing the boxed slice is sound.
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(base, count);
                    let owned = Box::from_raw(slice);
                    for &ptr in &owned {
                        if !ptr.is_null() {
                            drop(std::ffi::CString::from_raw(ptr));
                        }
                    }
                    drop(owned);
                }
            } else {
                // SAFETY: caller has guaranteed `out_array` is writable.
                unsafe {
                    *out_array = base;
                }
            }
            if !out_count.is_null() {
                // SAFETY: caller has guaranteed `out_count` is writable.
                unsafe {
                    *out_count = count;
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller has guaranteed `err` is writable.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}
