//! `CompletionStreamSink` foreign-callback trampoline + `blazen_complete_streaming`
//! free functions. Bridges three C function pointers (`on_chunk` / `on_done` /
//! `on_error`) into a Rust `Arc<dyn CompletionStreamSink>` so streaming
//! completions can drive foreign-language callbacks (Ruby blocks, Dart
//! callbacks, Crystal procs, etc.).
//!
//! Phase R5 Agent C.
//!
//! ## Why a vtable, not three loose callbacks
//!
//! Same rationale as the [`crate::tool_handler`] / [`crate::step_handler`]
//! trampolines: each foreign host has its own way of producing function
//! pointers, but they all understand a flat `#[repr(C)]` struct of
//! `(user_data, drop_user_data, on_chunk, on_done, on_error)` pointers. The
//! foreign side owns the lifecycle of `user_data` and provides the
//! `drop_user_data` thunk we call exactly once when the wrapper drops.
//!
//! ## Wire-format of the callback arguments
//!
//! - `on_chunk` receives `*mut BlazenStreamChunk` — caller-owned, the foreign
//!   callback OWNS it and MUST free it via [`crate::streaming_records::blazen_stream_chunk_free`]
//!   (or consume it into a derivative structure).
//! - `on_done` receives `(*mut c_char, *mut BlazenTokenUsage)` — both
//!   caller-owned. The `finish_reason` string is freed via
//!   [`crate::string::blazen_string_free`]; the usage handle via
//!   [`crate::llm_records::blazen_token_usage_free`].
//! - `on_error` receives `*mut BlazenError` — caller-owned. The foreign
//!   callback frees it via [`crate::error::blazen_error_free`].
//!
//! Every callback returns `0` on success / `-1` on failure (writing a fresh
//! `*mut BlazenError` into `*out_err`). A `-1` from `on_chunk` or `on_done`
//! propagates up: the streaming engine treats it as a sink-side abort, calls
//! `on_error` with the propagated error (if `complete_streaming` is still able
//! to), and returns. A `-1` from `on_error` is logged through the resulting
//! `BlazenResult` but cannot trigger another `on_error` (the engine never
//! double-fires error callbacks).
//!
//! ## Sync-over-async bridging
//!
//! The three trait methods are `async fn` on the Rust side but the C vtable
//! exposes synchronous function pointers. We bridge by dispatching each
//! callback through `tokio::task::spawn_blocking`, matching the pattern in
//! the `StepHandler` / `ToolHandler` trampolines: foreign callbacks may block
//! (Ruby's GVL acquisition can park, Dart isolates may shuttle work) and we
//! don't want a slow sink starving the async runtime.

use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;

use async_trait::async_trait;
use blazen_uniffi::errors::{BlazenError as InnerError, BlazenResult};
use blazen_uniffi::llm::{CompletionModel as InnerCompletionModel, TokenUsage as InnerTokenUsage};
use blazen_uniffi::streaming::{
    CompletionStreamSink, StreamChunk as InnerStreamChunk, complete_streaming,
    complete_streaming_blocking,
};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm::BlazenCompletionModel;
use crate::llm_records::{BlazenCompletionRequest, BlazenTokenUsage};
use crate::streaming_records::BlazenStreamChunk;

// ---------------------------------------------------------------------------
// Error-out helpers (module-private; mirror tool_handler.rs / step_handler.rs)
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

// ---------------------------------------------------------------------------
// VTable
// ---------------------------------------------------------------------------

/// Function-pointer vtable a foreign caller fills out to implement
/// [`CompletionStreamSink`] across the C ABI.
///
/// All three callbacks plus `drop_user_data` are required (no nullable
/// function pointers). The vtable is consumed by
/// [`blazen_complete_streaming`] / [`blazen_complete_streaming_blocking`],
/// which take ownership of `user_data` and the function pointers for the
/// lifetime of the streaming call. `drop_user_data` runs exactly once when
/// the wrapping [`CStreamSink`] drops (after the stream has terminated, or on
/// an early-return failure path before the stream starts).
///
/// ## Thread safety
///
/// The streaming engine schedules callbacks on tokio worker threads which
/// will generally differ from the thread that registered the sink. The
/// foreign side guarantees that `user_data` and the function pointers are
/// safe to invoke from any thread. In Ruby, the `ffi` gem's `FFI::Callback`
/// reacquires the GVL automatically before invoking the user-provided block;
/// in Dart, `NativeCallable.listener` marshals back to the isolate event
/// loop; in native hosts the responsibility falls to the embedder.
///
/// ## Callback contracts
///
/// See the module-level docs for the per-callback ownership rules. In
/// summary: every pointer passed *into* a callback is caller-owned and the
/// callback must free it (or consume it into a derivative structure) before
/// returning. Every pointer the callback writes *out* (`out_err`) becomes
/// caller-owned and the cabi will free it after consuming.
#[repr(C)]
pub struct BlazenCompletionStreamSinkVTable {
    /// Opaque foreign-side context handed back to each callback. Owned by
    /// this vtable struct; released via `drop_user_data` when the wrapper
    /// drops.
    pub user_data: *mut c_void,

    /// Invoked exactly once when the wrapping `CStreamSink` drops.
    /// Implementations should reclaim and release `user_data`.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),

    /// Receive a single chunk. `chunk` is caller-owned; the callback MUST
    /// free it via `blazen_stream_chunk_free` (or consume it into a
    /// derivative structure). Returns `0` on success or `-1` on failure
    /// (writing a fresh `*mut BlazenError` into `*out_err`). A `-1` aborts
    /// the stream.
    pub on_chunk: extern "C" fn(
        user_data: *mut c_void,
        chunk: *mut BlazenStreamChunk,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// Receive the terminal completion signal. `finish_reason` is a
    /// caller-owned `*mut c_char` freed via `blazen_string_free`; `usage`
    /// is a caller-owned `*mut BlazenTokenUsage` freed via
    /// `blazen_token_usage_free`. Returns `0` on success or `-1` on
    /// failure (writing a fresh `*mut BlazenError` into `*out_err`).
    pub on_done: extern "C" fn(
        user_data: *mut c_void,
        finish_reason: *mut c_char,
        usage: *mut BlazenTokenUsage,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// Receive a fatal error from the stream. `err` is a caller-owned
    /// `*mut BlazenError` freed via `blazen_error_free`. Returns `0` on
    /// success or `-1` on failure (writing the callback's own error into
    /// `*out_err`). A `-1` from `on_error` is logged via the resulting
    /// `BlazenResult` but does NOT trigger a second `on_error` invocation.
    pub on_error: extern "C" fn(
        user_data: *mut c_void,
        err: *mut BlazenError,
        out_err: *mut *mut BlazenError,
    ) -> i32,
}

// SAFETY: the foreign side guarantees thread-safety of `user_data` and the
// function pointers, as documented on `BlazenCompletionStreamSinkVTable`.
// Ruby's `ffi` gem automatically reacquires the GVL for callbacks; Dart's
// `NativeCallable.listener` marshals back to the isolate event loop; native
// hosts must opt into thread-safety in their own runtime model.
unsafe impl Send for BlazenCompletionStreamSinkVTable {}
// SAFETY: see the `Send` impl above — same foreign-side guarantee covers
// shared-reference access from multiple threads.
unsafe impl Sync for BlazenCompletionStreamSinkVTable {}

// ---------------------------------------------------------------------------
// Wrapper struct
// ---------------------------------------------------------------------------

/// Rust-side trampoline wrapping a foreign [`BlazenCompletionStreamSinkVTable`].
/// Implements the [`CompletionStreamSink`] trait by dispatching through the
/// vtable's function pointers.
///
/// Owns the vtable's `user_data` — drops it via `drop_user_data` exactly once
/// when this wrapper drops (after the stream terminates, or on an early-return
/// failure path before the stream starts).
pub(crate) struct CStreamSink {
    vtable: BlazenCompletionStreamSinkVTable,
}

impl Drop for CStreamSink {
    fn drop(&mut self) {
        // SAFETY: by the vtable contract, `drop_user_data` is the foreign
        // side's release thunk for `user_data` and is safe to call exactly
        // once when the wrapper is destroyed. We haven't called it before:
        // every early-return path in `blazen_complete_streaming` /
        // `blazen_complete_streaming_blocking` that aborts BEFORE constructing
        // `CStreamSink` calls `drop_user_data` directly, and once the wrapper
        // is constructed only this `Drop` impl invokes it.
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

#[async_trait]
impl CompletionStreamSink for CStreamSink {
    // `InnerError` is large (it carries every variant's payload inline), but
    // it's the shared error type across `blazen_uniffi` and we don't get to
    // choose its representation here.
    #[allow(clippy::result_large_err)]
    async fn on_chunk(&self, chunk: InnerStreamChunk) -> BlazenResult<()> {
        // Wrap the chunk in a cabi handle the foreign callback can free.
        let chunk_ptr = BlazenStreamChunk::from(chunk).into_ptr();

        // Capture pointer + fn-pointer as primitives so the `spawn_blocking`
        // closure can be `'static + Send`. We cast the raw pointers to
        // `usize` so the closure doesn't need to capture a `*mut c_void`
        // (which is `!Send`).
        let user_data_addr = self.vtable.user_data as usize;
        let on_chunk_fn = self.vtable.on_chunk;
        let chunk_addr = chunk_ptr as usize;

        // SAFETY: the foreign side guarantees thread-safe access to
        // `user_data` (see `BlazenCompletionStreamSinkVTable` docs). The
        // function pointer is a plain `extern "C" fn` — `Copy + Send + Sync`.
        // The `BlazenStreamChunk` pointer was just minted from `Box::into_raw`
        // so it's a unique allocation we're handing off to the callback.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let chunk_ptr = chunk_addr as *mut BlazenStreamChunk;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_chunk_fn(user_data, chunk_ptr, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "stream sink on_chunk returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract, on a failure return the
                // foreign callback has written a valid `*mut BlazenError`.
                // Ownership transfers to us; reclaim via `Box::from_raw` to
                // recover the inner error.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("stream sink on_chunk task panicked: {join_err}"),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn on_done(&self, finish_reason: String, usage: InnerTokenUsage) -> BlazenResult<()> {
        // Build C strings from the inputs. `CString::new` fails on interior
        // NUL bytes — surface that as an Internal error rather than a panic.
        let Ok(finish_cstring) = CString::new(finish_reason) else {
            return Err(InnerError::Internal {
                message: "stream sink on_done: finish_reason contains interior NUL byte".into(),
            });
        };
        let finish_raw = finish_cstring.into_raw();
        let usage_ptr = BlazenTokenUsage::from(usage).into_ptr();

        let user_data_addr = self.vtable.user_data as usize;
        let on_done_fn = self.vtable.on_done;
        let finish_addr = finish_raw as usize;
        let usage_addr = usage_ptr as usize;

        // SAFETY: same justification as `on_chunk` — foreign side guarantees
        // thread-safety; the `*mut c_char` and `*mut BlazenTokenUsage` pointers
        // were just minted from `CString::into_raw` / `Box::into_raw` so they
        // are unique allocations we're handing off to the callback.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let finish_ptr = finish_addr as *mut c_char;
            let usage_ptr = usage_addr as *mut BlazenTokenUsage;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_done_fn(user_data, finish_ptr, usage_ptr, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "stream sink on_done returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract, on a failure return the
                // foreign callback has written a valid `*mut BlazenError`.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("stream sink on_done task panicked: {join_err}"),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn on_error(&self, err: InnerError) -> BlazenResult<()> {
        // Wrap the inner error in a cabi handle the foreign callback can free.
        let err_ptr = BlazenError::from(err).into_ptr();

        let user_data_addr = self.vtable.user_data as usize;
        let on_error_fn = self.vtable.on_error;
        let err_addr = err_ptr as usize;

        // SAFETY: same justification as `on_chunk` — foreign side guarantees
        // thread-safety; the `*mut BlazenError` was just minted from
        // `Box::into_raw` so it's a unique allocation we're handing off.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let err_ptr = err_addr as *mut BlazenError;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_error_fn(user_data, err_ptr, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "stream sink on_error returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract, on a failure return the
                // foreign callback has written a valid `*mut BlazenError`.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("stream sink on_error task panicked: {join_err}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

/// Synchronously drives a streaming chat completion, dispatching chunks to
/// the supplied sink. Blocks the calling thread on the cabi tokio runtime.
///
/// On success returns `0` (the sink's `on_done` has fired). On a stream-start
/// failure returns `-1` and writes a fresh `*mut BlazenError` into `*out_err`;
/// sink-side failures during the stream are delivered through `on_error`
/// rather than this return path — see the upstream
/// [`blazen_uniffi::streaming::complete_streaming`] for the full contract.
///
/// ## Ownership transfer
///
/// - `model` is BORROWED — the underlying `Arc<CompletionModel>` is cloned
///   into the streaming call. Caller retains its handle.
/// - `request` is CONSUMED — internally we `Box::from_raw` it and move the
///   inner record out. Callers must NOT call `blazen_completion_request_free`
///   on the same pointer afterwards (double-free).
/// - `sink` (the vtable) is CONSUMED — ownership of `user_data` transfers to
///   the wrapping `CStreamSink`, which releases it via `drop_user_data` on
///   drop. On every early-return failure path that aborts BEFORE constructing
///   `CStreamSink`, this function explicitly invokes
///   `(sink.drop_user_data)(sink.user_data)` to honour the same contract.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCompletionModel` produced by the
/// cabi surface. `request` must be null OR a live `BlazenCompletionRequest`
/// produced by the cabi surface; ownership transfers to this function.
/// `sink.user_data` and the four `sink` function pointers must satisfy the
/// contracts documented on [`BlazenCompletionStreamSinkVTable`]. `out_err`
/// must be null OR a writable slot for a single `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_complete_streaming_blocking(
    model: *const BlazenCompletionModel,
    request: *mut BlazenCompletionRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // ---- Validate inputs, honouring the consume-on-call contract on every
    // early-return path -------------------------------------------------

    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership of `request`; reclaim and
            // drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "blazen_complete_streaming: null model") };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "blazen_complete_streaming: null request") };
    }

    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let model_handle = unsafe { &*model };
    let model_arc: Arc<InnerCompletionModel> = Arc::clone(&model_handle.0);

    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    // Wrap the vtable; from here on, `CStreamSink::drop` is responsible for
    // calling `drop_user_data`.
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });

    // The upstream `complete_streaming_blocking` calls `runtime().block_on`
    // on `blazen-uniffi`'s runtime — not the cabi runtime. That's fine: we
    // don't need any cabi-runtime context here, and reusing the upstream
    // runtime keeps semantics identical to a direct UniFFI call.
    let result = complete_streaming_blocking(model_arc, inner_request, sink_arc);
    match result {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drives a streaming chat completion onto the cabi tokio
/// runtime, returning an opaque future handle. The caller observes
/// completion via the future's fd / [`crate::future::blazen_future_poll`] /
/// [`crate::future::blazen_future_wait`], then calls
/// [`crate::persist::blazen_future_take_unit`] to pop the result. Free the
/// future with [`crate::future::blazen_future_free`].
///
/// Returns null if `model` is null or `request` is null. On those error
/// paths the `request` (if non-null) is still consumed and freed, and the
/// `sink.drop_user_data` thunk is still invoked, so no resources leak.
///
/// ## Ownership transfer
///
/// Same as [`blazen_complete_streaming_blocking`]: `model` is BORROWED,
/// `request` is CONSUMED, `sink` is CONSUMED. Every code path — success or
/// early-return failure — guarantees `request` is dropped and
/// `sink.drop_user_data` is invoked exactly once.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCompletionModel`. `request` must be
/// null OR a live `BlazenCompletionRequest`; ownership transfers to this
/// function regardless of whether the call returns null. `sink` satisfies
/// the [`BlazenCompletionStreamSinkVTable`] contract; its `user_data` is
/// consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_complete_streaming(
    model: *const BlazenCompletionModel,
    request: *mut BlazenCompletionRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership of `request`; reclaim and
            // drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }

    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let model_handle = unsafe { &*model };
    let model_arc: Arc<InnerCompletionModel> = Arc::clone(&model_handle.0);

    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    // Wrap the vtable; from here on, `CStreamSink::drop` is responsible for
    // calling `drop_user_data`.
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });

    BlazenFuture::spawn::<(), _>(async move {
        complete_streaming(model_arc, inner_request, sink_arc).await
    })
}
