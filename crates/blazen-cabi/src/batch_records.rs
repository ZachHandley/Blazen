//! Batch-related record marshalling. Wraps [`blazen_uniffi::batch::BatchItem`]
//! (enum) and [`blazen_uniffi::batch::BatchResult`] as opaque C handles.
//!
//! # Ownership conventions
//!
//! `BatchItem` and `BatchResult` are both output-only: they are produced by
//! the Phase R4 `complete_batch` / `complete_batch_blocking` free functions
//! and consumed by foreign callers. No public constructors are provided.
//!
//! Variant discrimination on `BatchItem` goes through [`blazen_batch_item_kind`]
//! which returns one of the [`BLAZEN_BATCH_ITEM_SUCCESS`] / [`BLAZEN_BATCH_ITEM_FAILURE`]
//! tags. Variant-specific accessors return null when called against the
//! wrong variant.
//!
//! `_responses_get(idx)` clones the indexed [`BatchItem`] into a fresh
//! caller-owned handle. Strings and nested handles produced by getters
//! transfer ownership to the caller; release with the matching `_free`
//! function or [`crate::string::blazen_string_free`].

// Foundation utility consumed by R4+ wrappers; the public extern fns stay
// linker-resident regardless.
#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::batch::{BatchItem as InnerBatchItem, BatchResult as InnerBatchResult};

use crate::llm_records::{BlazenCompletionResponse, BlazenTokenUsage};
use crate::string::alloc_cstring;

/// Variant tag for [`InnerBatchItem::Success`]: the request returned a
/// completion response.
pub const BLAZEN_BATCH_ITEM_SUCCESS: u32 = 0;
/// Variant tag for [`InnerBatchItem::Failure`]: the request failed and only
/// carries an error message.
pub const BLAZEN_BATCH_ITEM_FAILURE: u32 = 1;

// ---------------------------------------------------------------------------
// BlazenBatchItem
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::batch::BatchItem`].
pub struct BlazenBatchItem(pub(crate) InnerBatchItem);

impl BlazenBatchItem {
    pub(crate) fn into_ptr(self) -> *mut BlazenBatchItem {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerBatchItem> for BlazenBatchItem {
    fn from(inner: InnerBatchItem) -> Self {
        Self(inner)
    }
}

/// Returns the variant tag for `handle` — one of [`BLAZEN_BATCH_ITEM_SUCCESS`]
/// or [`BLAZEN_BATCH_ITEM_FAILURE`]. Returns [`BLAZEN_BATCH_ITEM_FAILURE`] on
/// a null handle (treating a missing slot as a degenerate failure with no
/// message).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchItem` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_item_kind(handle: *const BlazenBatchItem) -> u32 {
    if handle.is_null() {
        return BLAZEN_BATCH_ITEM_FAILURE;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchItem`.
    let item = unsafe { &*handle };
    match &item.0 {
        InnerBatchItem::Success { .. } => BLAZEN_BATCH_ITEM_SUCCESS,
        InnerBatchItem::Failure { .. } => BLAZEN_BATCH_ITEM_FAILURE,
    }
}

/// Returns a fresh `BlazenCompletionResponse` cloned from the
/// [`InnerBatchItem::Success`] variant's payload. Returns null if `handle` is
/// null or the variant is `Failure`. Caller frees with
/// `blazen_completion_response_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchItem` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_item_success_response(
    handle: *const BlazenBatchItem,
) -> *mut BlazenCompletionResponse {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchItem`.
    let item = unsafe { &*handle };
    match &item.0 {
        InnerBatchItem::Success { response } => {
            BlazenCompletionResponse(response.clone()).into_ptr()
        }
        InnerBatchItem::Failure { .. } => std::ptr::null_mut(),
    }
}

/// Returns the [`InnerBatchItem::Failure`] variant's `error_message` as a
/// caller-owned C string. Returns null if `handle` is null or the variant is
/// `Success`. Caller frees with `blazen_string_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchItem` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_item_failure_message(
    handle: *const BlazenBatchItem,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchItem`.
    let item = unsafe { &*handle };
    match &item.0 {
        InnerBatchItem::Failure { error_message } => alloc_cstring(error_message),
        InnerBatchItem::Success { .. } => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenBatchItem` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_item_free(handle: *mut BlazenBatchItem) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenBatchResult
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::batch::BatchResult`].
pub struct BlazenBatchResult(pub(crate) InnerBatchResult);

impl BlazenBatchResult {
    pub(crate) fn into_ptr(self) -> *mut BlazenBatchResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerBatchResult> for BlazenBatchResult {
    fn from(inner: InnerBatchResult) -> Self {
        Self(inner)
    }
}

/// Returns the number of per-request response slots. Returns `0` on a null
/// handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchResult` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_result_responses_count(
    handle: *const BlazenBatchResult,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchResult`.
    let r = unsafe { &*handle };
    r.0.responses.len()
}

/// Clones the `idx`-th response slot into a fresh `BlazenBatchItem` handle
/// the caller owns. Returns null on a null handle or out-of-range index.
/// Caller frees with `blazen_batch_item_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchResult` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_result_responses_get(
    handle: *const BlazenBatchResult,
    idx: usize,
) -> *mut BlazenBatchItem {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchResult`.
    let r = unsafe { &*handle };
    match r.0.responses.get(idx) {
        Some(item) => BlazenBatchItem(item.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Returns a fresh `BlazenTokenUsage` handle cloned from the aggregated
/// `total_usage` counters. Returns null on a null handle. Caller frees with
/// `blazen_token_usage_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchResult` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_result_total_usage(
    handle: *const BlazenBatchResult,
) -> *mut BlazenTokenUsage {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchResult`.
    let r = unsafe { &*handle };
    BlazenTokenUsage(r.0.total_usage.clone()).into_ptr()
}

/// Returns the aggregated `total_cost_usd` across successful responses.
/// Returns `0.0` on a null handle. The wire format does not distinguish
/// "no cost data reported" from "exactly zero" — both produce `0.0`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBatchResult` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_result_total_cost_usd(
    handle: *const BlazenBatchResult,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenBatchResult`.
    let r = unsafe { &*handle };
    r.0.total_cost_usd
}

/// Frees a `BlazenBatchResult` handle and all owned per-slot contents. No-op
/// on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_batch_result_free(handle: *mut BlazenBatchResult) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
