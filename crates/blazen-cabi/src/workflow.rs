//! Workflow opaque objects: `WorkflowBuilder` and `Workflow`. Each async
//! method exposes both a `_blocking` C entry point and a future-returning
//! variant.
//!
//! ## Ownership conventions
//!
//! - `blazen_workflow_builder_new` returns a caller-owned `*mut BlazenWorkflowBuilder`.
//!   Free with [`blazen_workflow_builder_free`] unless it has been consumed
//!   by [`blazen_workflow_builder_build`], which transfers ownership of the
//!   inner builder into the resulting `Workflow` (the original handle stays
//!   alive but its inner builder slot is empty — subsequent method calls on
//!   the same handle will fail with `Validation`).
//! - `blazen_workflow_builder_build` writes a caller-owned `*mut BlazenWorkflow`
//!   into `out_workflow` on success. Free that with [`blazen_workflow_free`].
//! - The `*_blocking` and future-returning `_run` wrappers both produce
//!   caller-owned values. `_blocking` writes a `*mut BlazenWorkflowResult` into
//!   `out_result`; the future variant hands the result through
//!   [`blazen_future_take_workflow_result`].
//! - String accessors that produce `*mut c_char` (e.g.
//!   [`blazen_workflow_step_names_get`]) return caller-owned heap strings;
//!   free with [`crate::string::blazen_string_free`].
//! - Errors flow through `*mut *mut BlazenError` out-params on fallible calls;
//!   the future-returning variants instead funnel the error through
//!   `blazen_future_take_*`.
//!
//! `WorkflowBuilder::step` requires the `StepHandler` callback trampoline and
//! is deferred to Phase R5 (see the doc comment at the bottom of this file).

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::workflow::{
    Workflow as InnerWorkflow, WorkflowBuilder as InnerWorkflowBuilder,
    WorkflowResult as InnerWorkflowResult,
};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_str};
use crate::workflow_records::BlazenWorkflowResult;

// ---------------------------------------------------------------------------
// Shared error-out helpers
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`. Centralises the
/// fallible-call epilogue so the per-method bodies stay focused on the happy
/// path.
///
/// # Safety
///
/// `out_err` must be null OR a valid destination for a single
/// `*mut BlazenError` write (typically a caller's stack-local
/// `*mut BlazenError` initialised to null).
unsafe fn write_error(out_err: *mut *mut BlazenError, e: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: caller-supplied out-param; per the function-level contract
        // it's either null (handled above) or a valid destination for a
        // single pointer-sized write.
        unsafe {
            *out_err = BlazenError::from(e).into_ptr();
        }
    }
    -1
}

/// Writes a synthesised `Internal` error to the out-param and returns `-1`.
/// Used for null-pointer / UTF-8 input failures where there isn't an
/// originating `InnerError`.
///
/// # Safety
///
/// Same contract as [`write_error`]: `out_err` is null OR points at a single
/// writable `*mut BlazenError` slot.
unsafe fn write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded to `write_error`; caller upholds the same contract.
    unsafe {
        write_error(
            out_err,
            InnerError::Internal {
                message: msg.into(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// BlazenWorkflowBuilder
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::workflow::WorkflowBuilder`.
///
/// The inner `Arc` matches the `self: Arc<Self>` shape of the underlying
/// builder methods — the methods reach through an internal `Mutex<Option<_>>`
/// to consume + replace the in-progress core builder, so cloning the `Arc`
/// here is sound and lets each C entry point hand a fresh ref into the
/// inner method.
pub struct BlazenWorkflowBuilder(pub(crate) Arc<InnerWorkflowBuilder>);

/// Construct a new builder with the given UTF-8 `name`. Returns null on a
/// null pointer or non-UTF-8 input.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with
/// [`blazen_workflow_builder_free`] unless it has been consumed by
/// [`blazen_workflow_builder_build`].
///
/// # Safety
///
/// `name` must be null OR a valid NUL-terminated UTF-8 buffer that remains
/// live for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_builder_new(
    name: *const c_char,
) -> *mut BlazenWorkflowBuilder {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        return std::ptr::null_mut();
    };
    let inner = InnerWorkflowBuilder::new(name.to_owned());
    Box::into_raw(Box::new(BlazenWorkflowBuilder(inner)))
}

/// Sets the per-step timeout in milliseconds. Returns `0` on success or `-1`
/// on failure (writing the inner error to `out_err`).
///
/// # Safety
///
/// `builder` must be a valid pointer to a `BlazenWorkflowBuilder` previously
/// produced by [`blazen_workflow_builder_new`] (and not yet freed). `out_err`
/// is null OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_builder_step_timeout_ms(
    builder: *mut BlazenWorkflowBuilder,
    millis: u64,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if builder.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }
    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder = unsafe { &*builder };
    let inner = Arc::clone(&builder.0);
    match inner.step_timeout_ms(millis) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Sets the workflow-wide timeout in milliseconds. Returns `0` on success or
/// `-1` on failure (writing the inner error to `out_err`).
///
/// # Safety
///
/// Same as [`blazen_workflow_builder_step_timeout_ms`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_builder_timeout_ms(
    builder: *mut BlazenWorkflowBuilder,
    millis: u64,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if builder.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }
    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder = unsafe { &*builder };
    let inner = Arc::clone(&builder.0);
    match inner.timeout_ms(millis) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Consumes the in-progress builder and produces a runnable `Workflow`,
/// writing the caller-owned `*mut BlazenWorkflow` into `out_workflow` on
/// success. Returns `0` on success or `-1` on failure.
///
/// On the success path the `BlazenWorkflowBuilder` handle remains live but
/// its internal builder slot is now empty; subsequent calls on the same
/// handle will fail with a `Validation` error. The handle itself must still
/// be released with [`blazen_workflow_builder_free`].
///
/// # Safety
///
/// `builder` must be a valid pointer to a `BlazenWorkflowBuilder` previously
/// produced by [`blazen_workflow_builder_new`] (and not yet freed).
/// `out_workflow` is null OR a valid destination for one `*mut BlazenWorkflow`
/// write. `out_err` is null OR a valid destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_builder_build(
    builder: *mut BlazenWorkflowBuilder,
    out_workflow: *mut *mut BlazenWorkflow,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if builder.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }
    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder = unsafe { &*builder };
    let inner = Arc::clone(&builder.0);
    match inner.build() {
        Ok(workflow) => {
            if !out_workflow.is_null() {
                // SAFETY: caller-supplied out-param; per the function-level
                // contract it's either null (handled above) or a valid
                // destination for a single pointer-sized write.
                unsafe {
                    *out_workflow = Box::into_raw(Box::new(BlazenWorkflow(workflow)));
                }
            }
            0
        }
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Frees a `BlazenWorkflowBuilder` handle previously produced by the cabi
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `builder` must be null OR a pointer previously produced by
/// [`blazen_workflow_builder_new`]. Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_builder_free(builder: *mut BlazenWorkflowBuilder) {
    if builder.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(builder) });
}

// ---------------------------------------------------------------------------
// BlazenWorkflow
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::workflow::Workflow`. The inner `Arc`
/// supports `self: Arc<Self>` async methods on the inner type.
pub struct BlazenWorkflow(pub(crate) Arc<InnerWorkflow>);

/// Synchronously runs the workflow with the given JSON `input_json` payload.
/// Blocks the calling thread on the cabi tokio runtime. Returns `0` on
/// success (writing a caller-owned `*mut BlazenWorkflowResult` to
/// `out_result`) or `-1` on failure (writing the inner error to `out_err`).
///
/// # Safety
///
/// `wf` must be a valid pointer to a `BlazenWorkflow`. `input_json` must be a
/// valid NUL-terminated UTF-8 buffer. `out_result` is null OR a valid
/// destination for one `*mut BlazenWorkflowResult` write. `out_err` is null
/// OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_run_blocking(
    wf: *const BlazenWorkflow,
    input_json: *const c_char,
    out_result: *mut *mut BlazenWorkflowResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if wf.is_null() || input_json.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `wf` is a live pointer.
    let wf = unsafe { &*wf };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `input_json`.
    let input = match unsafe { cstr_to_str(input_json) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => return unsafe { write_internal_error(out_err, "input_json not valid UTF-8") },
    };
    let inner = Arc::clone(&wf.0);
    match runtime().block_on(async move { inner.run(input).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: caller-supplied out-param; per the function-level
                // contract it's either null (handled above) or a valid
                // destination for a single pointer-sized write.
                unsafe {
                    *out_result = BlazenWorkflowResult::from(result).into_ptr();
                }
            }
            0
        }
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Runs the workflow asynchronously, returning an opaque future handle
/// immediately. The caller waits via `blazen_future_wait` / `blazen_future_fd`
/// / `blazen_future_poll`, then takes the result via
/// [`blazen_future_take_workflow_result`].
///
/// Returns null if `wf` or `input_json` is null, or if `input_json` is not
/// valid UTF-8. Errors that surface during the async run are delivered
/// through `blazen_future_take_workflow_result`'s `err` out-param.
///
/// # Safety
///
/// `wf` must be a valid pointer to a `BlazenWorkflow`. `input_json` must be a
/// valid NUL-terminated UTF-8 buffer that remains valid for the duration of
/// this call (the buffer is copied before this function returns).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_run(
    wf: *const BlazenWorkflow,
    input_json: *const c_char,
) -> *mut BlazenFuture {
    if wf.is_null() || input_json.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `wf` is a live pointer.
    let wf = unsafe { &*wf };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `input_json`.
    let input = match unsafe { cstr_to_str(input_json) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    let inner = Arc::clone(&wf.0);
    BlazenFuture::spawn(async move { inner.run(input).await })
}

/// Returns the number of registered steps in this workflow. Returns `0` if
/// `wf` is null.
///
/// Used together with [`blazen_workflow_step_names_get`] to iterate step
/// names without round-tripping a heap-allocated array across the FFI.
///
/// # Safety
///
/// `wf` must be null OR a valid pointer to a `BlazenWorkflow` previously
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_step_names_count(wf: *const BlazenWorkflow) -> usize {
    if wf.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `wf` is a live pointer.
    let wf = unsafe { &*wf };
    Arc::clone(&wf.0).step_names().len()
}

/// Returns the step name at position `idx` as a heap-allocated NUL-terminated
/// UTF-8 C string. Returns null if `wf` is null or `idx` is out of bounds.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `wf` must be null OR a valid pointer to a `BlazenWorkflow` previously
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_step_names_get(
    wf: *const BlazenWorkflow,
    idx: usize,
) -> *mut c_char {
    if wf.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `wf` is a live pointer.
    let wf = unsafe { &*wf };
    Arc::clone(&wf.0)
        .step_names()
        .get(idx)
        .map_or(std::ptr::null_mut(), |name| alloc_cstring(name))
}

/// Frees a `BlazenWorkflow` handle previously produced by the cabi surface.
/// No-op on a null pointer.
///
/// # Safety
///
/// `wf` must be null OR a pointer previously produced by
/// [`blazen_workflow_builder_build`] (or any other cabi function documenting
/// `BlazenWorkflow` ownership-transfer-to-caller semantics). Calling this
/// twice on the same non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_free(wf: *mut BlazenWorkflow) {
    if wf.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(wf) });
}

// ---------------------------------------------------------------------------
// Typed future-take for WorkflowResult
// ---------------------------------------------------------------------------

/// Pops the `WorkflowResult` out of a future produced by any of the
/// workflow-result-returning cabi async wrappers (`blazen_workflow_run`,
/// `blazen_pipeline_run`, and — once R5 lands — `blazen_peer_client_run_remote_workflow`).
/// Returns `0` on success or `-1` on error.
///
/// On success, `out` receives a caller-owned `*mut BlazenWorkflowResult` (free
/// with [`crate::workflow_records::blazen_workflow_result_free`]). On error,
/// `err` receives a caller-owned `*mut BlazenError` (free with
/// [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// All three pointers must follow the cabi-future contract: `fut` is a live
/// future produced by `blazen_workflow_run` / `blazen_pipeline_run` /
/// `blazen_peer_client_run_remote_workflow`, observed completed via
/// `blazen_future_poll` / `_wait` / `_fd`. `out` and `err` are valid
/// destinations for a single `*mut` write (typically stack `*mut BlazenX`
/// locals — can be null to discard).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_workflow_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenWorkflowResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the future-pointer contract documented above.
    match unsafe { BlazenFuture::take_typed::<InnerWorkflowResult>(fut) } {
        Ok(result) => {
            if !out.is_null() {
                // SAFETY: caller-supplied out-param; per the contract above
                // it's either null (handled) or a valid destination for a
                // single pointer-sized write.
                unsafe {
                    *out = BlazenWorkflowResult::from(result).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller-supplied out-param; same contract as `out`.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Deferred surface
// ---------------------------------------------------------------------------

// `blazen_workflow_builder_step` is intentionally **not** exposed here.
// Registering a step requires a foreign-language `StepHandler` trampoline —
// a callback that the Rust workflow engine can invoke from arbitrary tokio
// worker threads, with the foreign-language method coupled to a
// `BlazenStepOutput` return value. That trampoline lands in Phase R5
// alongside the `blazen_agent_*` tool-handler bridge; until then, the cabi
// surface is consumer-only (build a Workflow with no steps, run it, observe
// the result).
