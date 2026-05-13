//! `StepHandler` foreign-callback trampoline. Bridges a C vtable of function
//! pointers into a Rust `Arc<dyn StepHandler>` so the workflow engine can
//! invoke foreign-language step handlers (Ruby blocks, Dart callbacks,
//! Crystal procs, etc.).
//!
//! ## Ownership / lifecycle
//!
//! [`blazen_workflow_builder_add_step`] takes a [`BlazenStepHandlerVTable`] by
//! value. The vtable's `user_data` ownership transfers into the wrapper
//! [`CStepHandler`] for the lifetime of the workflow. `drop_user_data` is
//! invoked exactly once when the wrapper drops (workflow drop, or builder
//! drop before consuming).
//!
//! Critically, EVERY early-return path inside `blazen_workflow_builder_add_step`
//! that fails before constructing the wrapper MUST call
//! `(vtable.drop_user_data)(vtable.user_data)` — otherwise the foreign side
//! has already considered ownership transferred and won't free it itself,
//! leaking the foreign-side context.
//!
//! ## Thread safety
//!
//! The Rust workflow engine schedules step invocations on tokio worker
//! threads, which differ from the thread that registered the step. The
//! foreign side must guarantee that its `user_data` pointer and function
//! pointers are safe to invoke from any thread. In Ruby, the `ffi` gem's
//! `FFI::Callback` reacquires the GVL automatically before invoking the
//! user-provided block; in Dart, `NativeCallable.listener` marshals the
//! callback back to the isolate event loop; in native hosts the
//! responsibility falls to the embedder.
//!
//! ## Sync-over-async bridging
//!
//! `StepHandler::invoke` is `async fn` on the Rust side but the C vtable
//! exposes a synchronous function pointer (a non-trivial async ABI across
//! the C boundary would require a callback-based promise — not worth it for
//! the typical step-handler workload). We bridge by spawning the C
//! invocation onto Tokio's blocking thread pool via `spawn_blocking`, so the
//! synchronous foreign callback never starves the runtime's async workers.
//!
//! Phase R5 Agent A.

use std::ffi::{c_char, c_void};
use std::sync::Arc;

use async_trait::async_trait;
use blazen_uniffi::errors::{BlazenError as InnerError, BlazenResult};
use blazen_uniffi::workflow::{Event as InnerEvent, StepHandler, StepOutput as InnerStepOutput};

use crate::error::BlazenError;
use crate::string::cstr_to_str;
use crate::workflow::BlazenWorkflowBuilder;
use crate::workflow_records::{BlazenEvent, BlazenStepOutput};

// ---------------------------------------------------------------------------
// Shared error-out helpers
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

/// Converts a C array of NUL-terminated UTF-8 pointers into a `Vec<String>`.
/// Returns `None` if any element pointer is null or contains invalid UTF-8.
/// A null `ptrs` with `count == 0` yields an empty `Vec`.
///
/// # Safety
///
/// `ptrs` must be null OR point to an array of `count` `*const c_char`
/// values, each of which is either null or a valid NUL-terminated UTF-8
/// buffer that remains live for the duration of this call.
unsafe fn cstr_array_to_vec_string(
    ptrs: *const *const c_char,
    count: usize,
) -> Option<Vec<String>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if ptrs.is_null() {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: caller has guaranteed `ptrs` points to at least `count`
        // valid `*const c_char` slots; `i < count`.
        let item = unsafe { *ptrs.add(i) };
        // SAFETY: caller-supplied item pointer per the function-level
        // contract (NUL-terminated or null).
        let s = unsafe { cstr_to_str(item) }?;
        out.push(s.to_owned());
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// VTable struct
// ---------------------------------------------------------------------------

/// C-side vtable for a foreign `StepHandler` implementation.
///
/// All three fields are required (no nullable function pointers). The vtable
/// is consumed by [`blazen_workflow_builder_add_step`] which takes ownership
/// of `user_data` — the foreign side must NOT free `user_data` after the
/// add-step call returns; the cabi will invoke `drop_user_data` exactly once
/// when the workflow drops the step handler.
///
/// # Thread safety
///
/// The foreign side guarantees that `user_data` and the function pointers are
/// safe to invoke from any thread (the cabi may invoke `invoke` on a Tokio
/// blocking-pool thread, which differs from the thread that created the
/// vtable). In Ruby this works because the `ffi` gem reacquires the GVL
/// automatically for declared `FFI::Callback` signatures; Dart's
/// `NativeCallable.listener` marshals back to the isolate event loop;
/// native hosts must opt into thread-safety in their own runtime model.
#[repr(C)]
pub struct BlazenStepHandlerVTable {
    /// Opaque foreign-side context handed back to each function pointer.
    /// Owned by this vtable struct (released via `drop_user_data` on drop).
    pub user_data: *mut c_void,

    /// Called exactly once when the wrapping `CStepHandler` drops.
    /// Implementations should reclaim and release `user_data`.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),

    /// Synchronous invocation of the step.
    ///
    /// Arguments:
    /// - `user_data`: the vtable's `user_data` pointer (copied in for convenience)
    /// - `event`: caller-owned `*mut BlazenEvent` — the callback OWNS this and
    ///   must free it via `blazen_event_free` before returning, OR consume it
    ///   into a derivative structure
    /// - `out_output`: writable slot for the resulting `*mut BlazenStepOutput`
    /// - `out_err`: writable slot for the error on failure
    ///
    /// Returns: 0 on success (`out_output` set), -1 on failure (`out_err` set).
    pub invoke: extern "C" fn(
        user_data: *mut c_void,
        event: *mut BlazenEvent,
        out_output: *mut *mut BlazenStepOutput,
        out_err: *mut *mut BlazenError,
    ) -> i32,
}

// SAFETY: the foreign side guarantees thread-safety of `user_data` and the
// function pointers, as documented on `BlazenStepHandlerVTable`. Ruby's `ffi`
// gem automatically reacquires the GVL for callbacks; Dart's
// `NativeCallable.listener` marshals back to the isolate event loop; native
// hosts must opt into thread-safety in their own runtime model.
unsafe impl Send for BlazenStepHandlerVTable {}
// SAFETY: see the `Send` impl above — same foreign-side guarantee covers
// shared-reference access from multiple threads.
unsafe impl Sync for BlazenStepHandlerVTable {}

// ---------------------------------------------------------------------------
// Wrapper struct
// ---------------------------------------------------------------------------

/// Rust-side trampoline wrapping a foreign [`BlazenStepHandlerVTable`].
/// Implements the [`StepHandler`] trait by calling into the vtable's
/// function pointers.
///
/// Owns the vtable's `user_data` — drops it via `drop_user_data` exactly
/// once when this wrapper drops.
pub(crate) struct CStepHandler {
    vtable: BlazenStepHandlerVTable,
}

impl Drop for CStepHandler {
    fn drop(&mut self) {
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

#[async_trait]
impl StepHandler for CStepHandler {
    // `InnerError` is large (it carries every variant's payload inline), but
    // it's the shared error type across `blazen_uniffi` and we don't get to
    // choose its representation here.
    #[allow(clippy::result_large_err)]
    async fn invoke(&self, event: InnerEvent) -> BlazenResult<InnerStepOutput> {
        // Wrap event in cabi handle; the foreign callback owns it for the
        // duration of the call and is contractually required to free it.
        let event_ptr = BlazenEvent::from(event).into_ptr();

        // Copy the vtable fields out by value so the `spawn_blocking` closure
        // can be 'static + Send. We cast the raw pointer to usize so the
        // closure doesn't need to capture a `*mut c_void` (which is `!Send`).
        let user_data_addr = self.vtable.user_data as usize;
        let invoke_fn = self.vtable.invoke;
        let event_addr = event_ptr as usize;

        // SAFETY: the foreign side guarantees thread-safe access to
        // `user_data` (see `BlazenStepHandlerVTable` docs). The function
        // pointer is a plain `extern "C" fn` — `Copy + Send + Sync`. The
        // `BlazenEvent` pointer was just minted from `Box::into_raw` so it's
        // a unique allocation we're handing off to the callback.
        let join_result =
            tokio::task::spawn_blocking(move || -> Result<InnerStepOutput, InnerError> {
                let user_data = user_data_addr as *mut c_void;
                let event_ptr = event_addr as *mut BlazenEvent;
                let mut out_output: *mut BlazenStepOutput = std::ptr::null_mut();
                let mut out_err: *mut BlazenError = std::ptr::null_mut();

                let status = invoke_fn(user_data, event_ptr, &raw mut out_output, &raw mut out_err);

                if status == 0 {
                    if out_output.is_null() {
                        return Err(InnerError::Internal {
                            message: "step handler returned success but null output".into(),
                        });
                    }
                    // SAFETY: per the vtable contract, on a success return
                    // (status == 0) the foreign callback has written a valid
                    // `*mut BlazenStepOutput` produced by one of the
                    // `blazen_step_output_new_*` constructors. Ownership of
                    // that allocation transfers to us; reclaim via
                    // `Box::from_raw`.
                    let bs = unsafe { Box::from_raw(out_output) };
                    Ok(bs.0)
                } else {
                    if out_err.is_null() {
                        return Err(InnerError::Internal {
                            message: "step handler returned -1 without setting out_err".into(),
                        });
                    }
                    // SAFETY: per the vtable contract, on a failure return
                    // (status != 0) the foreign callback has written a valid
                    // `*mut BlazenError`. Ownership transfers to us; reclaim
                    // via `Box::from_raw` to recover the inner error.
                    let be = unsafe { Box::from_raw(out_err) };
                    Err(be.inner)
                }
            })
            .await;

        match join_result {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("step handler task panicked: {join_err}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// C entry point
// ---------------------------------------------------------------------------

/// Registers a step on a workflow builder. The vtable is consumed (its
/// `user_data` becomes owned by the workflow until the workflow drops).
///
/// `name`, `accepts`, `emits` are NUL-terminated UTF-8 strings; `accepts` and
/// `emits` are arrays of strings (or null, paired with a `_count` of `0`).
///
/// Returns 0 on success, -1 on failure (writing the error to `out_err`).
///
/// # Ownership transfer
///
/// On EVERY return path — success or failure — the cabi takes responsibility
/// for releasing `vtable.user_data`. Foreign callers MUST NOT call
/// `drop_user_data` themselves after handing the vtable to this function. On
/// failure paths that abort before constructing the [`CStepHandler`] wrapper,
/// this function explicitly invokes `(vtable.drop_user_data)(vtable.user_data)`
/// before returning, honouring the same ownership contract.
///
/// # Safety
///
/// `builder` must be a live `BlazenWorkflowBuilder` previously produced by
/// the cabi surface (and not yet freed). `name` must be a valid
/// NUL-terminated UTF-8 buffer. `accepts` and `emits` must each be either
/// null (with the corresponding `_count` being 0) or arrays of `_count`
/// valid NUL-terminated UTF-8 buffers. `vtable` must contain valid function
/// pointers and the `user_data` pointer transfers ownership to the wrapper.
/// `out_err` is null OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_builder_add_step(
    builder: *mut BlazenWorkflowBuilder,
    name: *const c_char,
    accepts: *const *const c_char,
    accepts_count: usize,
    emits: *const *const c_char,
    emits_count: usize,
    vtable: BlazenStepHandlerVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if builder.is_null() {
        // Honor the ownership-transfer contract: the foreign side has already
        // handed `user_data` to us, so we must release it even on this early-
        // return path.
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null builder pointer") };
    }

    // SAFETY: caller upholds the NUL-termination + lifetime contract on `name`.
    let Some(name_str) = (unsafe { cstr_to_str(name) }) else {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null or non-UTF-8 step name") };
    };
    let name_owned = name_str.to_owned();

    // SAFETY: caller upholds the array shape contract.
    let Some(accepts_vec) = (unsafe { cstr_array_to_vec_string(accepts, accepts_count) }) else {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null or non-UTF-8 accepts entry") };
    };

    // SAFETY: caller upholds the array shape contract.
    let Some(emits_vec) = (unsafe { cstr_array_to_vec_string(emits, emits_count) }) else {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null or non-UTF-8 emits entry") };
    };

    // Wrap the vtable; from here on, `CStepHandler::drop` is responsible for
    // calling `drop_user_data`.
    let handler: Arc<dyn StepHandler> = Arc::new(CStepHandler { vtable });

    // SAFETY: caller has guaranteed `builder` is a live pointer.
    let builder_ref = unsafe { &*builder };
    let inner = Arc::clone(&builder_ref.0);
    match inner.step(name_owned, accepts_vec, emits_vec, handler) {
        Ok(_) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}
