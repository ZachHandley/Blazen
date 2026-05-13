//! Pipeline opaque objects: `PipelineBuilder` and `Pipeline`. Each async
//! method exposes both a `_blocking` C entry point and a future-returning
//! variant.
//!
//! ## Ownership conventions
//!
//! - `blazen_pipeline_builder_new` returns a caller-owned `*mut BlazenPipelineBuilder`.
//!   Free with [`blazen_pipeline_builder_free`] unless it has been consumed by
//!   [`blazen_pipeline_builder_build`], which transfers ownership of the inner
//!   state into the resulting `Pipeline` (the original handle stays alive but
//!   its inner state slot is empty — subsequent method calls on the same
//!   handle will fail with `Validation`).
//! - `blazen_pipeline_builder_add_workflow`, `blazen_pipeline_builder_stage`,
//!   and `blazen_pipeline_builder_parallel` **consume** the `*mut BlazenWorkflow`
//!   pointers passed in. Callers MUST NOT separately free those workflow
//!   handles after these calls return (regardless of success or failure — on
//!   the failure path the consumed handles are still dropped here to avoid
//!   leaking caller-allocated input). On the parallel variant every workflow
//!   in the `workflows` array is consumed.
//! - `blazen_pipeline_builder_build` writes a caller-owned `*mut BlazenPipeline`
//!   into `out_pipeline`. Free with [`blazen_pipeline_free`].
//! - The `*_blocking` and future-returning `_run` wrappers both produce
//!   caller-owned `*mut BlazenWorkflowResult` values (pipelines reuse the
//!   workflow-result type). Future variants funnel through
//!   [`crate::workflow::blazen_future_take_workflow_result`].
//! - String accessors that produce `*mut c_char` return caller-owned heap
//!   strings; free with [`crate::string::blazen_string_free`].

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::pipeline::{Pipeline as InnerPipeline, PipelineBuilder as InnerPipelineBuilder};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_str};
use crate::workflow::BlazenWorkflow;
use crate::workflow_records::BlazenWorkflowResult;

// ---------------------------------------------------------------------------
// Shared error-out helpers (mirror workflow.rs — kept local so the two files
// stay self-contained for cbindgen and reviewers).
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`.
///
/// # Safety
///
/// `out_err` must be null OR a valid destination for a single
/// `*mut BlazenError` write.
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
///
/// # Safety
///
/// Same contract as [`write_error`].
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

/// Reclaims ownership of a `*mut BlazenWorkflow` previously produced by
/// [`crate::workflow::blazen_workflow_builder_build`] (or any other cabi
/// function documenting `BlazenWorkflow` ownership-transfer semantics) and
/// returns the inner `Arc<InnerWorkflow>`.
///
/// Used by the pipeline-builder methods that take ownership of workflow
/// handles to fold them into the pipeline's stage list.
///
/// # Safety
///
/// `wf` must be a non-null pointer previously produced by the cabi surface
/// over a `BlazenWorkflow`. Calling this twice on the same pointer is a
/// double-free.
unsafe fn take_workflow(wf: *mut BlazenWorkflow) -> Arc<blazen_uniffi::workflow::Workflow> {
    // SAFETY: caller upholds the `Box::into_raw` provenance contract on `wf`.
    let boxed = unsafe { Box::from_raw(wf) };
    boxed.0
}

// ---------------------------------------------------------------------------
// BlazenPipelineBuilder
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::pipeline::PipelineBuilder`.
///
/// The inner `Arc` matches the `self: Arc<Self>` shape of the underlying
/// builder methods — those methods reach through an internal
/// `Mutex<Option<_>>` to consume + replace the in-progress builder state, so
/// cloning the `Arc` per cabi call is sound.
pub struct BlazenPipelineBuilder(pub(crate) Arc<InnerPipelineBuilder>);

/// Construct a new pipeline builder with the given UTF-8 `name`. Returns
/// null on a null pointer or non-UTF-8 input.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with
/// [`blazen_pipeline_builder_free`] unless it has been consumed by
/// [`blazen_pipeline_builder_build`].
///
/// # Safety
///
/// `name` must be null OR a valid NUL-terminated UTF-8 buffer that remains
/// live for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_new(
    name: *const c_char,
) -> *mut BlazenPipelineBuilder {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        return std::ptr::null_mut();
    };
    let inner = InnerPipelineBuilder::new(name.to_owned());
    Box::into_raw(Box::new(BlazenPipelineBuilder(inner)))
}

/// Appends a sequential workflow stage with an auto-generated stage name.
/// Consumes ownership of `workflow` — the caller MUST NOT separately free
/// it after this call returns (regardless of return value; the workflow is
/// reclaimed even on the error path).
///
/// Returns `0` on success or `-1` on failure (writing the inner error to
/// `out_err`).
///
/// # Safety
///
/// `builder` must be a valid pointer to a `BlazenPipelineBuilder`.
/// `workflow` must be a non-null pointer previously produced by
/// [`crate::workflow::blazen_workflow_builder_build`] (and not yet freed).
/// `out_err` is null OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_add_workflow(
    builder: *mut BlazenPipelineBuilder,
    workflow: *mut BlazenWorkflow,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if builder.is_null() {
        if !workflow.is_null() {
            // SAFETY: caller upholds the `Box::into_raw` contract on `workflow`.
            drop(unsafe { Box::from_raw(workflow) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }
    if workflow.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null workflow pointer") };
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract on `workflow`.
    let wf_arc = unsafe { take_workflow(workflow) };
    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder = unsafe { &*builder };
    let inner = Arc::clone(&builder.0);
    match inner.add_workflow(wf_arc) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Appends a sequential workflow stage with an explicit `name`. Consumes
/// ownership of `workflow`. Returns `0` on success or `-1` on failure.
///
/// # Safety
///
/// Same as [`blazen_pipeline_builder_add_workflow`], plus `name` must be a
/// valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_stage(
    builder: *mut BlazenPipelineBuilder,
    name: *const c_char,
    workflow: *mut BlazenWorkflow,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if builder.is_null() {
        if !workflow.is_null() {
            // SAFETY: caller upholds the `Box::into_raw` contract on `workflow`.
            drop(unsafe { Box::from_raw(workflow) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }
    if workflow.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null workflow pointer") };
    }
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `name`.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        // SAFETY: caller upholds the `Box::into_raw` contract on `workflow`.
        drop(unsafe { Box::from_raw(workflow) });
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "name not valid UTF-8") };
    };
    let name = name.to_owned();
    // SAFETY: caller upholds the `Box::into_raw` provenance contract on `workflow`.
    let wf_arc = unsafe { take_workflow(workflow) };
    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder = unsafe { &*builder };
    let inner = Arc::clone(&builder.0);
    match inner.stage(name, wf_arc) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Appends a parallel stage running multiple workflows concurrently.
///
/// - `name`: name for the parallel stage as a whole.
/// - `branch_names` + `branch_count`: array of NUL-terminated UTF-8 branch
///   names. The array itself is borrowed (not consumed); the strings it
///   points at must remain valid for the duration of this call.
/// - `workflows` + `workflow_count`: array of `*mut BlazenWorkflow` pointers
///   to be **consumed** by this call. Every workflow is reclaimed regardless
///   of return value to avoid leaking caller-allocated input on the failure
///   path. The `workflows` array storage itself is owned by the caller.
/// - `wait_all`: if true, every branch must complete; otherwise the stage
///   finishes as soon as the first branch produces a result.
///
/// `branch_count` and `workflow_count` must match; a mismatch yields a
/// `Validation` error (after consuming every workflow in `workflows`).
///
/// Returns `0` on success or `-1` on failure (writing the inner error to
/// `out_err`).
///
/// # Safety
///
/// `builder` is a valid pointer to a `BlazenPipelineBuilder`. `name` is a
/// valid NUL-terminated UTF-8 buffer. `branch_names` is a valid pointer to
/// `branch_count` contiguous NUL-terminated UTF-8 string pointers (each
/// individually live for the duration of the call). `workflows` is a valid
/// pointer to `workflow_count` contiguous `*mut BlazenWorkflow` values, each
/// of which was previously produced by the cabi surface and has not yet been
/// freed. `out_err` is null OR a valid destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_parallel(
    builder: *mut BlazenPipelineBuilder,
    name: *const c_char,
    branch_names: *const *const c_char,
    branch_count: usize,
    workflows: *mut *mut BlazenWorkflow,
    workflow_count: usize,
    wait_all: bool,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // Reclaim every workflow up-front so a downstream input-validation
    // failure can't leak them. We do this before any other validation so the
    // cleanup path is uniform.
    let mut taken_workflows: Vec<Arc<blazen_uniffi::workflow::Workflow>> =
        Vec::with_capacity(workflow_count);
    if !workflows.is_null() {
        for i in 0..workflow_count {
            // SAFETY: per the function-level contract, `workflows` points at
            // `workflow_count` contiguous `*mut BlazenWorkflow` values.
            let wf_ptr = unsafe { *workflows.add(i) };
            if wf_ptr.is_null() {
                continue;
            }
            // SAFETY: caller upholds the `Box::into_raw` contract on each
            // workflow pointer.
            taken_workflows.push(unsafe { take_workflow(wf_ptr) });
        }
    }

    if builder.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }
    if branch_count != workflow_count {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "branch_names length does not match workflows length",
            )
        };
    }
    if name.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null name pointer") };
    }
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `name`.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "name not valid UTF-8") };
    };
    let name = name.to_owned();

    if branch_count > 0 && branch_names.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null branch_names pointer") };
    }
    let mut names: Vec<String> = Vec::with_capacity(branch_count);
    for i in 0..branch_count {
        // SAFETY: per the function-level contract, `branch_names` points at
        // `branch_count` contiguous `*const c_char` values.
        let s_ptr = unsafe { *branch_names.add(i) };
        // SAFETY: each entry is a valid NUL-terminated UTF-8 buffer.
        let Some(s) = (unsafe { cstr_to_str(s_ptr) }) else {
            // SAFETY: `out_err` upholds the function-level contract.
            return unsafe { write_internal_error(out_err, "branch name not valid UTF-8") };
        };
        names.push(s.to_owned());
    }

    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder = unsafe { &*builder };
    let inner = Arc::clone(&builder.0);
    match inner.parallel(name, names, taken_workflows, wait_all) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Sets the per-stage timeout in milliseconds. Returns `0` on success or
/// `-1` on failure.
///
/// # Safety
///
/// `builder` must be a valid pointer to a `BlazenPipelineBuilder`. `out_err`
/// is null OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_timeout_per_stage_ms(
    builder: *mut BlazenPipelineBuilder,
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
    match inner.timeout_per_stage_ms(millis) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Sets the total pipeline wall-clock timeout in milliseconds. Returns `0`
/// on success or `-1` on failure.
///
/// # Safety
///
/// Same as [`blazen_pipeline_builder_timeout_per_stage_ms`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_total_timeout_ms(
    builder: *mut BlazenPipelineBuilder,
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
    match inner.total_timeout_ms(millis) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Validates the pipeline definition and produces a runnable `Pipeline`,
/// writing the caller-owned `*mut BlazenPipeline` into `out_pipeline` on
/// success. Returns `0` on success or `-1` on failure.
///
/// On the success path the `BlazenPipelineBuilder` handle remains live but
/// its internal state slot is now empty; subsequent calls on the same handle
/// will fail with `Validation`. The handle itself must still be released
/// with [`blazen_pipeline_builder_free`].
///
/// # Safety
///
/// `builder` must be a valid pointer to a `BlazenPipelineBuilder`.
/// `out_pipeline` is null OR a valid destination for one `*mut BlazenPipeline`
/// write. `out_err` is null OR a valid destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_build(
    builder: *mut BlazenPipelineBuilder,
    out_pipeline: *mut *mut BlazenPipeline,
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
        Ok(pipeline) => {
            if !out_pipeline.is_null() {
                // SAFETY: caller-supplied out-param.
                unsafe {
                    *out_pipeline = Box::into_raw(Box::new(BlazenPipeline(pipeline)));
                }
            }
            0
        }
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Frees a `BlazenPipelineBuilder` handle. No-op on a null pointer.
///
/// # Safety
///
/// `builder` must be null OR a pointer previously produced by
/// [`blazen_pipeline_builder_new`]. Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_builder_free(builder: *mut BlazenPipelineBuilder) {
    if builder.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(builder) });
}

// ---------------------------------------------------------------------------
// BlazenPipeline
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::pipeline::Pipeline`. The inner `Arc`
/// supports the `self: Arc<Self>` async methods on the inner type and lets
/// `_blocking` / future-returning variants share a single live reference.
pub struct BlazenPipeline(pub(crate) Arc<InnerPipeline>);

/// Synchronously runs the pipeline with the given JSON `input_json` payload.
/// Blocks the calling thread on the cabi tokio runtime. Returns `0` on
/// success (writing a caller-owned `*mut BlazenWorkflowResult` to
/// `out_result`) or `-1` on failure (writing the inner error to `out_err`).
///
/// Pipelines produce the same result type as workflows — a `WorkflowResult`
/// whose terminal event is a synthetic `StopEvent` carrying the final stage's
/// output and whose `total_*_tokens` / `total_cost_usd` are summed across
/// every stage.
///
/// # Safety
///
/// `pipe` must be a valid pointer to a `BlazenPipeline`. `input_json` must be
/// a valid NUL-terminated UTF-8 buffer. `out_result` is null OR a valid
/// destination for one `*mut BlazenWorkflowResult` write. `out_err` is null
/// OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_run_blocking(
    pipe: *const BlazenPipeline,
    input_json: *const c_char,
    out_result: *mut *mut BlazenWorkflowResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if pipe.is_null() || input_json.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `pipe` is a live pointer.
    let pipe = unsafe { &*pipe };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `input_json`.
    let input = match unsafe { cstr_to_str(input_json) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => return unsafe { write_internal_error(out_err, "input_json not valid UTF-8") },
    };
    let inner = Arc::clone(&pipe.0);
    match runtime().block_on(async move { inner.run(input).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: caller-supplied out-param.
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

/// Runs the pipeline asynchronously, returning an opaque future handle
/// immediately. Caller drives it with `blazen_future_wait` / `_fd` / `_poll`
/// and then takes the result via
/// [`crate::workflow::blazen_future_take_workflow_result`] (pipelines and
/// workflows produce the same result type).
///
/// Returns null if `pipe` or `input_json` is null, or if `input_json` is not
/// valid UTF-8.
///
/// # Safety
///
/// `pipe` must be a valid pointer to a `BlazenPipeline`. `input_json` must be
/// a valid NUL-terminated UTF-8 buffer that remains valid for the duration of
/// this call (the buffer is copied before this function returns).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_run(
    pipe: *const BlazenPipeline,
    input_json: *const c_char,
) -> *mut BlazenFuture {
    if pipe.is_null() || input_json.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `pipe` is a live pointer.
    let pipe = unsafe { &*pipe };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `input_json`.
    let input = match unsafe { cstr_to_str(input_json) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    let inner = Arc::clone(&pipe.0);
    BlazenFuture::spawn(async move { inner.run(input).await })
}

/// Returns the number of stages in this pipeline. Returns `0` if `pipe` is
/// null.
///
/// Used together with [`blazen_pipeline_stage_names_get`] to iterate stage
/// names.
///
/// # Safety
///
/// `pipe` must be null OR a valid pointer to a `BlazenPipeline` previously
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_stage_names_count(pipe: *const BlazenPipeline) -> usize {
    if pipe.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `pipe` is a live pointer.
    let pipe = unsafe { &*pipe };
    Arc::clone(&pipe.0).stage_names().len()
}

/// Returns the stage name at position `idx` as a heap-allocated NUL-terminated
/// UTF-8 C string. Returns null if `pipe` is null or `idx` is out of bounds.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `pipe` must be null OR a valid pointer to a `BlazenPipeline` previously
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_stage_names_get(
    pipe: *const BlazenPipeline,
    idx: usize,
) -> *mut c_char {
    if pipe.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `pipe` is a live pointer.
    let pipe = unsafe { &*pipe };
    Arc::clone(&pipe.0)
        .stage_names()
        .get(idx)
        .map_or(std::ptr::null_mut(), |name| alloc_cstring(name))
}

/// Frees a `BlazenPipeline` handle previously produced by the cabi surface.
/// No-op on a null pointer.
///
/// # Safety
///
/// `pipe` must be null OR a pointer previously produced by
/// [`blazen_pipeline_builder_build`]. Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_pipeline_free(pipe: *mut BlazenPipeline) {
    if pipe.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(pipe) });
}
