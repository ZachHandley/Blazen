//! Batch completion free functions. Phase R4 Agent B.
//!
//! Wraps the upstream [`blazen_uniffi::batch::complete_batch`] /
//! [`blazen_uniffi::batch::complete_batch_blocking`] functions in C-callable
//! form. Two execution modes:
//!
//! - [`blazen_complete_batch_blocking`] — synchronous, drives the future on
//!   the cabi tokio runtime via `runtime().block_on(...)`.
//! - [`blazen_complete_batch`] — returns a `*mut BlazenFuture` immediately and
//!   spawns the batch task onto the runtime. The caller observes completion
//!   via `blazen_future_poll` / `blazen_future_wait` / `blazen_future_fd` and
//!   then pops the typed result with [`blazen_future_take_batch_result`].
//!
//! ## Ownership conventions
//!
//! - `model` is BORROWED; the caller retains ownership of the
//!   `BlazenCompletionModel*`. The inner `Arc<CompletionModel>` is cloned into
//!   the task so the model handle can be freed independently of the spawned
//!   batch.
//! - The `requests` array itself is BORROWED at the array level — the caller
//!   frees the outer `BlazenCompletionRequest*` array after the cabi call
//!   returns — but each `BlazenCompletionRequest*` element is CONSUMED. The
//!   cabi reclaims each via `Box::from_raw`, moves the inner
//!   `blazen_uniffi::llm::CompletionRequest` record out, and drops the empty
//!   wrapper. Calling
//!   [`crate::llm_records::blazen_completion_request_free`] on any element
//!   passed to a batch call afterwards is a double-free.
//! - The result handle produced via either the blocking call or
//!   [`blazen_future_take_batch_result`] is caller-owned and freed via
//!   [`crate::batch_records::blazen_batch_result_free`].
//! - Error handles produced on the failure path are caller-owned and freed
//!   via [`crate::error::blazen_error_free`].

use std::sync::Arc;

use blazen_uniffi::batch::BatchResult as InnerBatchResult;
use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::llm::CompletionRequest as InnerCompletionRequest;

use crate::batch_records::BlazenBatchResult;
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm::BlazenCompletionModel;
use crate::llm_records::BlazenCompletionRequest;
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Local error helpers
//
// Mirror the pattern used in `llm.rs`, `agent.rs`, and `compute_factories.rs`
// rather than reaching for a shared helper — each cabi module stays
// self-contained.
// ---------------------------------------------------------------------------

/// Writes a caller-owned `BlazenError` into `out_err` if the slot is non-null
/// and returns `-1` so the caller can `return write_error(...)` in tail
/// position.
fn write_error(out_err: *mut *mut BlazenError, err: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: `out_err` is non-null per the branch above. The caller has
        // guaranteed it points to a writable `*mut BlazenError` slot.
        unsafe {
            *out_err = BlazenError::from(err).into_ptr();
        }
    }
    -1
}

/// Builds + writes a `BlazenError::Internal { message }` into `out_err`.
fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) -> i32 {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    )
}

// ---------------------------------------------------------------------------
// Request-array consumption helper
// ---------------------------------------------------------------------------

/// Drain the C `requests` array, taking ownership of each
/// `BlazenCompletionRequest*` element and moving the inner record out.
///
/// Returns `None` if `requests` is null while `count > 0`, or if any indexed
/// pointer is null. On the `None` path, every `BlazenCompletionRequest*`
/// element that was successfully reclaimed before the first null is dropped
/// (so callers don't leak prefix elements when a later element is invalid).
///
/// An empty array (`count == 0`) is valid and yields `Some(Vec::new())`; the
/// `requests` outer pointer is only required to be non-null when `count > 0`.
///
/// # Safety
///
/// When `count > 0`, `requests` must point to an array of exactly `count`
/// `BlazenCompletionRequest*` entries. Each entry must be null OR a pointer
/// previously produced by [`crate::llm_records::blazen_completion_request_new`]
/// (or equivalent `Box::into_raw` over a `BlazenCompletionRequest`), and
/// ownership of every non-null entry transfers to this function.
unsafe fn take_requests(
    requests: *const *mut BlazenCompletionRequest,
    count: usize,
) -> Option<Vec<InnerCompletionRequest>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if requests.is_null() {
        return None;
    }
    let mut out: Vec<InnerCompletionRequest> = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: caller guarantees `requests` indexes exactly `count` valid
        // pointer slots; reading slot `i < count` is in-bounds.
        let p = unsafe { *requests.add(i) };
        if p.is_null() {
            // Drop everything we've already reclaimed so the caller doesn't
            // leak prefix entries on the validation-error path. The Rust
            // borrow checker is satisfied because `out` is dropped as we
            // return.
            return None;
        }
        // SAFETY: caller has transferred ownership of slot `i`; per the
        // module-level contract, `p` came from `Box::into_raw` over a
        // `BlazenCompletionRequest`, so reconstructing the `Box` is sound.
        let req_box = unsafe { Box::from_raw(p) };
        out.push(req_box.0);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Synchronously run a batch of completions on the cabi tokio runtime.
///
/// On success returns `0` and writes a fresh `BlazenBatchResult*` into
/// `*out_result`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Both out-parameters may be null to discard the matching
/// side of the result (typically only meaningful on the error path during a
/// smoke test).
///
/// `max_concurrency` is a hard cap on in-flight requests; `0` means unlimited
/// (the upstream default — all dispatched in parallel).
///
/// **Each `BlazenCompletionRequest*` element of `requests` is consumed.** See
/// the module-level docs for the full ownership contract; calling
/// [`crate::llm_records::blazen_completion_request_free`] on any element
/// afterwards is a double-free.
///
/// # Safety
///
/// - `model` must be null OR a live `BlazenCompletionModel` produced by the
///   cabi surface.
/// - When `requests_count > 0`, `requests` must point to an array of exactly
///   `requests_count` `BlazenCompletionRequest*` entries; each entry must be
///   a live `BlazenCompletionRequest` produced by the cabi surface, and
///   ownership of every entry transfers to this function. When
///   `requests_count == 0`, `requests` may be null.
/// - `out_result` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_complete_batch_blocking(
    model: *const BlazenCompletionModel,
    requests: *const *mut BlazenCompletionRequest,
    requests_count: usize,
    max_concurrency: u32,
    out_result: *mut *mut BlazenBatchResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(out_err, "blazen_complete_batch_blocking: null model");
    }
    // SAFETY: caller has guaranteed `requests` is a valid array of `requests_count`
    // owned `BlazenCompletionRequest*` entries (or `requests_count == 0`).
    let Some(core_requests) = (unsafe { take_requests(requests, requests_count) }) else {
        return write_internal_error(
            out_err,
            "blazen_complete_batch_blocking: null requests array or null request element",
        );
    };
    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let m = unsafe { &*model };
    let inner_model = Arc::clone(&m.0);

    let result: Result<InnerBatchResult, InnerError> = runtime().block_on(async move {
        blazen_uniffi::batch::complete_batch(inner_model, core_requests, max_concurrency).await
    });

    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenBatchResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Spawn a batch of completions onto the cabi tokio runtime and return an
/// opaque future handle. Observe completion via `blazen_future_poll` /
/// `blazen_future_wait` / `blazen_future_fd`, then pop the typed result with
/// [`blazen_future_take_batch_result`]. Free the future with
/// `blazen_future_free`.
///
/// Returns null if `model` is null, or if `requests` is null when
/// `requests_count > 0`, or if any indexed `BlazenCompletionRequest*` element
/// is null. On every null-return path, ownership of any
/// `BlazenCompletionRequest*` elements that were already reclaimed is
/// dropped (no leaks).
///
/// **Each `BlazenCompletionRequest*` element of `requests` is consumed** on
/// the success path AND on the validation-failure path (so the caller is
/// always relieved of ownership of every successfully-passed element). See
/// the module-level docs for the full contract.
///
/// # Safety
///
/// Same as [`blazen_complete_batch_blocking`]: `model` must be null OR a live
/// `BlazenCompletionModel`; the `requests` array elements transfer ownership.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_complete_batch(
    model: *const BlazenCompletionModel,
    requests: *const *mut BlazenCompletionRequest,
    requests_count: usize,
    max_concurrency: u32,
) -> *mut BlazenFuture {
    if model.is_null() {
        // We can't safely reclaim the request elements without first running
        // `take_requests`, so do that anyway to avoid leaking when the model
        // is the only null. `take_requests` will either drop everything (on
        // its own None path) or return owned Rust values that we drop on the
        // next line.
        //
        // SAFETY: caller has guaranteed the array shape contract on `requests`.
        let _ = unsafe { take_requests(requests, requests_count) };
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed the array shape contract on `requests`.
    let Some(core_requests) = (unsafe { take_requests(requests, requests_count) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let m = unsafe { &*model };
    let inner_model = Arc::clone(&m.0);

    BlazenFuture::spawn(async move {
        blazen_uniffi::batch::complete_batch(inner_model, core_requests, max_concurrency).await
    })
}

/// Pops the [`BlazenBatchResult`] out of a future produced by
/// [`blazen_complete_batch`].
///
/// Returns `0` on success (writes the result into `*out` when non-null) or
/// `-1` on failure (writes a fresh `BlazenError*` into `*err` when non-null).
/// Both out-parameters may be null to discard the corresponding side.
///
/// Calling this against a future produced by any other cabi entry point
/// (e.g. `blazen_completion_model_complete`) yields a `BlazenError::Internal`
/// with a `type mismatch` message — see `BlazenFuture::take_typed`.
///
/// # Safety
///
/// `fut` must be null OR a pointer previously produced by
/// [`blazen_complete_batch`] (and not yet freed, not concurrently freed from
/// another thread). `out` and `err` must each be null OR point to a writable
/// slot of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_batch_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenBatchResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerBatchResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenBatchResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(err, e),
    }
}
