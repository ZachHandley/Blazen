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
use blazen_uniffi::compute_music::{
    MusicChunk as InnerMusicChunk, MusicStreamSink, stream_generate_music_to_sink,
    stream_generate_music_to_sink_blocking, stream_generate_sfx_to_sink,
    stream_generate_sfx_to_sink_blocking,
};
use blazen_uniffi::compute_vc::{
    VcChunk as InnerVcChunk, VcStreamSink, stream_convert_pcm_to_sink,
    stream_convert_pcm_to_sink_blocking,
};
use blazen_uniffi::errors::{BlazenError as InnerError, BlazenResult};
use blazen_uniffi::llm::{Model as InnerModel, TokenUsage as InnerTokenUsage};
use blazen_uniffi::streaming::{
    CompletionStreamSink, StreamChunk as InnerStreamChunk, complete_streaming,
    complete_streaming_blocking,
};

use crate::compute_music::BlazenMusicModel;
use crate::compute_records::{BlazenMusicChunk, BlazenVcChunk};
use crate::compute_vc::BlazenVcModel;
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm::BlazenModel;
#[cfg(feature = "candle-llm")]
use crate::llm_providers::BlazenCandleLlmProvider;
#[cfg(feature = "llamacpp")]
use crate::llm_providers::BlazenLlamaCppProvider;
#[cfg(feature = "mistralrs")]
use crate::llm_providers::BlazenMistralRsProvider;
use crate::llm_providers::{
    BlazenAnthropicProvider, BlazenAzureOpenAiProvider, BlazenBedrockProvider,
    BlazenCohereProvider, BlazenDeepSeekProvider, BlazenFalLlmProvider, BlazenFireworksProvider,
    BlazenGeminiProvider, BlazenGroqProvider, BlazenLmStudioProvider, BlazenMistralProvider,
    BlazenOllamaProvider, BlazenOpenAiCompatProvider, BlazenOpenAiProvider,
    BlazenOpenRouterProvider, BlazenPerplexityProvider, BlazenTogetherProvider, BlazenXaiProvider,
};
use crate::llm_records::{BlazenModelRequest, BlazenTokenUsage};
use crate::streaming_records::BlazenStreamChunk;

#[cfg(feature = "audio-music-audiogen")]
use crate::music::BlazenAudioGenProvider;
#[cfg(feature = "audio-music-musicgen")]
use crate::music::BlazenMusicGenProvider;
#[cfg(feature = "audio-music-stable-audio")]
use crate::music::BlazenStableAudioProvider;
#[cfg(feature = "audio-vc-rvc")]
use crate::vc::BlazenRvcProvider;

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

impl CStreamSink {
    /// Construct a `CStreamSink` from a caller-supplied vtable. Ownership of
    /// `vtable.user_data` transfers to the returned sink; the foreign-side
    /// `drop_user_data` thunk is invoked exactly once on drop.
    pub(crate) fn from_vtable(vtable: BlazenCompletionStreamSinkVTable) -> Self {
        Self { vtable }
    }
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
/// - `model` is BORROWED — the underlying `Arc<Model>` is cloned
///   into the streaming call. Caller retains its handle.
/// - `request` is CONSUMED — internally we `Box::from_raw` it and move the
///   inner record out. Callers must NOT call `blazen_model_request_free`
///   on the same pointer afterwards (double-free).
/// - `sink` (the vtable) is CONSUMED — ownership of `user_data` transfers to
///   the wrapping `CStreamSink`, which releases it via `drop_user_data` on
///   drop. On every early-return failure path that aborts BEFORE constructing
///   `CStreamSink`, this function explicitly invokes
///   `(sink.drop_user_data)(sink.user_data)` to honour the same contract.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenModel` produced by the
/// cabi surface. `request` must be null OR a live `BlazenModelRequest`
/// produced by the cabi surface; ownership transfers to this function.
/// `sink.user_data` and the four `sink` function pointers must satisfy the
/// contracts documented on [`BlazenCompletionStreamSinkVTable`]. `out_err`
/// must be null OR a writable slot for a single `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_complete_streaming_blocking(
    model: *const BlazenModel,
    request: *mut BlazenModelRequest,
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

    // SAFETY: caller has guaranteed `model` is a live `BlazenModel`.
    let model_handle = unsafe { &*model };
    let model_arc: Arc<InnerModel> = Arc::clone(&model_handle.0);

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
/// `model` must be null OR a live `BlazenModel`. `request` must be
/// null OR a live `BlazenModelRequest`; ownership transfers to this
/// function regardless of whether the call returns null. `sink` satisfies
/// the [`BlazenCompletionStreamSinkVTable`] contract; its `user_data` is
/// consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_complete_streaming(
    model: *const BlazenModel,
    request: *mut BlazenModelRequest,
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

    // SAFETY: caller has guaranteed `model` is a live `BlazenModel`.
    let model_handle = unsafe { &*model };
    let model_arc: Arc<InnerModel> = Arc::clone(&model_handle.0);

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

// ===========================================================================
// Per-engine LLM streaming entry points
// ===========================================================================
//
// Each pair mirrors `blazen_complete_streaming` / `_blocking` exactly — same
// null checks, same `CStreamSink` construction from the vtable, same
// consume-`request` / consume-`sink` ownership discipline — but borrows a
// per-engine cabi provider opaque (cloning its inner `Arc<...Provider>` like
// the `_complete` fns in `llm_providers.rs`) and dispatches to the matching
// per-engine `blazen_uniffi::concrete::llm` streaming free function instead of
// the central `complete_streaming`. The central fns + the `CStreamSink`
// trampoline above are reused verbatim (no duplicate trampoline / sink impl).
//
// Written longhand (no `macro_rules!`): cbindgen does not expand declarative
// macros, so each `#[no_mangle]` symbol must appear in source for the Ruby
// header to pick it up.

// OpenAiProvider -----------------------------------------------------------

/// Synchronously drive a streaming chat completion through the
/// `OpenAiProvider` concrete provider. Mirrors
/// [`blazen_complete_streaming_blocking`] semantics (null checks,
/// `request`/`sink` consume discipline) but dispatches to
/// [`blazen_uniffi::concrete::llm::openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenOpenAiProvider`. `request` must be
/// null OR a live `BlazenModelRequest`; ownership transfers to this function.
/// `sink` satisfies the [`BlazenCompletionStreamSinkVTable`] contract; its
/// `user_data` is consumed. `out_err` must be null OR a writable slot for a
/// single `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_complete_streaming_blocking(
    provider: *const BlazenOpenAiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_openai_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_openai_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::openai_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drive a streaming chat completion through the
/// `OpenAiProvider` concrete provider, returning an opaque future handle.
/// Mirrors [`blazen_complete_streaming`] semantics.
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenOpenAiProvider`. `request` must be
/// null OR a live `BlazenModelRequest`; ownership transfers to this function
/// regardless of whether the call returns null. `sink` satisfies the
/// [`BlazenCompletionStreamSinkVTable`] contract; its `user_data` is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_complete_streaming(
    provider: *const BlazenOpenAiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::openai_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// AnthropicProvider --------------------------------------------------------

/// Synchronous per-engine streaming for `AnthropicProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenAnthropicProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_complete_streaming_blocking(
    provider: *const BlazenAnthropicProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_anthropic_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_anthropic_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::anthropic_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `AnthropicProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenAnthropicProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_complete_streaming(
    provider: *const BlazenAnthropicProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::anthropic_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// GeminiProvider -----------------------------------------------------------

/// Synchronous per-engine streaming for `GeminiProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenGeminiProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_complete_streaming_blocking(
    provider: *const BlazenGeminiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_gemini_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_gemini_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::gemini_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `GeminiProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenGeminiProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_complete_streaming(
    provider: *const BlazenGeminiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::gemini_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// AzureOpenAiProvider ------------------------------------------------------

/// Synchronous per-engine streaming for `AzureOpenAiProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenAzureOpenAiProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_complete_streaming_blocking(
    provider: *const BlazenAzureOpenAiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_azure_openai_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_azure_openai_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::azure_openai_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `AzureOpenAiProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenAzureOpenAiProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_complete_streaming(
    provider: *const BlazenAzureOpenAiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::azure_openai_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// BedrockProvider ----------------------------------------------------------

/// Synchronous per-engine streaming for `BedrockProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenBedrockProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_complete_streaming_blocking(
    provider: *const BlazenBedrockProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_bedrock_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_bedrock_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::bedrock_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `BedrockProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenBedrockProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_complete_streaming(
    provider: *const BlazenBedrockProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::bedrock_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// MistralProvider ----------------------------------------------------------

/// Synchronous per-engine streaming for `MistralProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenMistralProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_complete_streaming_blocking(
    provider: *const BlazenMistralProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_mistral_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_mistral_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::mistral_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `MistralProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenMistralProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_complete_streaming(
    provider: *const BlazenMistralProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::mistral_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// FireworksProvider --------------------------------------------------------

/// Synchronous per-engine streaming for `FireworksProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenFireworksProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_complete_streaming_blocking(
    provider: *const BlazenFireworksProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_fireworks_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_fireworks_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::fireworks_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `FireworksProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenFireworksProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_complete_streaming(
    provider: *const BlazenFireworksProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::fireworks_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// DeepSeekProvider ---------------------------------------------------------

/// Synchronous per-engine streaming for `DeepSeekProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenDeepSeekProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_complete_streaming_blocking(
    provider: *const BlazenDeepSeekProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_deepseek_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_deepseek_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::deepseek_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `DeepSeekProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenDeepSeekProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_complete_streaming(
    provider: *const BlazenDeepSeekProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::deepseek_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// PerplexityProvider -------------------------------------------------------

/// Synchronous per-engine streaming for `PerplexityProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenPerplexityProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_complete_streaming_blocking(
    provider: *const BlazenPerplexityProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_perplexity_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_perplexity_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::perplexity_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `PerplexityProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenPerplexityProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_complete_streaming(
    provider: *const BlazenPerplexityProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::perplexity_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// TogetherProvider ---------------------------------------------------------

/// Synchronous per-engine streaming for `TogetherProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenTogetherProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_complete_streaming_blocking(
    provider: *const BlazenTogetherProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_together_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_together_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::together_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `TogetherProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenTogetherProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_complete_streaming(
    provider: *const BlazenTogetherProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::together_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// GroqProvider -------------------------------------------------------------

/// Synchronous per-engine streaming for `GroqProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenGroqProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_complete_streaming_blocking(
    provider: *const BlazenGroqProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_groq_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_groq_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::groq_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `GroqProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenGroqProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_complete_streaming(
    provider: *const BlazenGroqProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::groq_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// OpenRouterProvider -------------------------------------------------------

/// Synchronous per-engine streaming for `OpenRouterProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenOpenRouterProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_complete_streaming_blocking(
    provider: *const BlazenOpenRouterProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_openrouter_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_openrouter_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::openrouter_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `OpenRouterProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenOpenRouterProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_complete_streaming(
    provider: *const BlazenOpenRouterProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::openrouter_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// CohereProvider -----------------------------------------------------------

/// Synchronous per-engine streaming for `CohereProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenCohereProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_complete_streaming_blocking(
    provider: *const BlazenCohereProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_cohere_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_cohere_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::cohere_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `CohereProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenCohereProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_complete_streaming(
    provider: *const BlazenCohereProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::cohere_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// XaiProvider --------------------------------------------------------------

/// Synchronous per-engine streaming for `XaiProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenXaiProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_complete_streaming_blocking(
    provider: *const BlazenXaiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_xai_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_xai_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::xai_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `XaiProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenXaiProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_complete_streaming(
    provider: *const BlazenXaiProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::xai_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// FalLlmProvider -----------------------------------------------------------

/// Synchronous per-engine streaming for `FalLlmProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenFalLlmProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_complete_streaming_blocking(
    provider: *const BlazenFalLlmProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_fal_llm_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_fal_llm_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::fal_llm_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `FalLlmProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenFalLlmProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_complete_streaming(
    provider: *const BlazenFalLlmProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::fal_llm_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// OpenAiCompatProvider -----------------------------------------------------

/// Synchronous per-engine streaming for `OpenAiCompatProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenOpenAiCompatProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_complete_streaming_blocking(
    provider: *const BlazenOpenAiCompatProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_openai_compat_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_openai_compat_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::openai_compat_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `OpenAiCompatProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenOpenAiCompatProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_complete_streaming(
    provider: *const BlazenOpenAiCompatProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::openai_compat_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// OllamaProvider -----------------------------------------------------------

/// Synchronous per-engine streaming for `OllamaProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenOllamaProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_complete_streaming_blocking(
    provider: *const BlazenOllamaProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_ollama_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_ollama_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::ollama_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `OllamaProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenOllamaProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_complete_streaming(
    provider: *const BlazenOllamaProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::ollama_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// LmStudioProvider ---------------------------------------------------------

/// Synchronous per-engine streaming for `LmStudioProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenLmStudioProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_complete_streaming_blocking(
    provider: *const BlazenLmStudioProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_lm_studio_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_lm_studio_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::lm_studio_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `LmStudioProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenLmStudioProvider`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_complete_streaming(
    provider: *const BlazenLmStudioProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::lm_studio_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// MistralRsProvider --------------------------------------------------------

/// Synchronous per-engine streaming for `MistralRsProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenMistralRsProvider`).
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_complete_streaming_blocking(
    provider: *const BlazenMistralRsProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_mistralrs_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_mistralrs_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::mistralrs_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `MistralRsProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenMistralRsProvider`).
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_complete_streaming(
    provider: *const BlazenMistralRsProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::mistralrs_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// LlamaCppProvider ---------------------------------------------------------

/// Synchronous per-engine streaming for `LlamaCppProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenLlamaCppProvider`).
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_complete_streaming_blocking(
    provider: *const BlazenLlamaCppProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_llamacpp_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_llamacpp_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::llamacpp_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `LlamaCppProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenLlamaCppProvider`).
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_complete_streaming(
    provider: *const BlazenLlamaCppProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::llamacpp_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// CandleLlmProvider --------------------------------------------------------

/// Synchronous per-engine streaming for `CandleLlmProvider`. See
/// [`blazen_openai_provider_complete_streaming_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming_blocking`]
/// (`provider` is a `BlazenCandleLlmProvider`).
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_complete_streaming_blocking(
    provider: *const BlazenCandleLlmProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_candle_provider_complete_streaming: null provider",
            )
        };
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_candle_provider_complete_streaming: null request",
            )
        };
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    match blazen_uniffi::concrete::llm::candle_provider_complete_streaming_blocking(
        provider_arc,
        inner_request,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async per-engine streaming for `CandleLlmProvider`. See
/// [`blazen_openai_provider_complete_streaming`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_streaming`]
/// (`provider` is a `BlazenCandleLlmProvider`).
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_complete_streaming(
    provider: *const BlazenCandleLlmProvider,
    request: *mut BlazenModelRequest,
    sink: BlazenCompletionStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        if !request.is_null() {
            // SAFETY: caller transferred ownership; reclaim and drop.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;
    let sink_arc: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::llm::candle_provider_complete_streaming(
            provider_arc,
            inner_request,
            sink_arc,
        )
        .await
    })
}

// ===========================================================================
// MusicStreamSink trampoline + stream pump entry points
// ===========================================================================
//
// Symmetric to the `CompletionStreamSink` trampoline above. Foreign callers
// implement [`MusicStreamSink`] (`on_chunk` + `on_done` + `on_error`) via a
// flat `#[repr(C)]` vtable; the cabi side wraps that vtable in a Rust trait
// impl that dispatches through `tokio::task::spawn_blocking` (same rationale
// as the completion sink: foreign callbacks may block, we don't want a slow
// sink starving the async runtime).
//
// Wire-format differences vs the completion sink:
//
// - `on_chunk` receives `*mut BlazenMusicChunk` instead of
//   `*mut BlazenStreamChunk`. Caller-owned; foreign callback frees via
//   [`crate::compute_records::blazen_music_chunk_free`].
// - `on_done` takes no auxiliary args — music streams don't carry a finish
//   reason or token-usage record (the upstream
//   [`MusicStreamSink::on_done`](blazen_uniffi::compute_music::MusicStreamSink::on_done)
//   is a bare `async fn(&self) -> BlazenResult<()>`).
// - `on_error` matches the completion-sink shape: receives a caller-owned
//   `*mut BlazenError` the foreign callback frees via `blazen_error_free`.

/// Function-pointer vtable a foreign caller fills out to implement
/// [`MusicStreamSink`] across the C ABI.
///
/// All three callbacks plus `drop_user_data` are required. The vtable is
/// consumed by the music stream-pump functions below
/// ([`blazen_music_model_stream_generate_music`] /
/// [`blazen_music_model_stream_generate_music_blocking`] /
/// [`blazen_music_model_stream_generate_sfx`] /
/// [`blazen_music_model_stream_generate_sfx_blocking`]); they take ownership
/// of `user_data` and the function pointers for the lifetime of the streaming
/// call. `drop_user_data` runs exactly once when the wrapping
/// [`CMusicStreamSink`] drops (after the stream has terminated, or on an
/// early-return failure path before the stream starts).
#[repr(C)]
pub struct BlazenMusicStreamSinkVTable {
    /// Opaque foreign-side context handed back to each callback. Owned by
    /// this vtable struct; released via `drop_user_data` when the wrapper
    /// drops.
    pub user_data: *mut c_void,

    /// Invoked exactly once when the wrapping `CMusicStreamSink` drops.
    /// Implementations should reclaim and release `user_data`.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),

    /// Receive a single chunk. `chunk` is caller-owned; the callback MUST
    /// free it via [`crate::compute_records::blazen_music_chunk_free`] (or
    /// consume it into a derivative structure). Returns `0` on success or
    /// `-1` on failure (writing a fresh `*mut BlazenError` into `*out_err`).
    /// A `-1` aborts the stream.
    pub on_chunk: extern "C" fn(
        user_data: *mut c_void,
        chunk: *mut BlazenMusicChunk,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// Receive the terminal completion signal. Music streams carry no
    /// auxiliary `on_done` payload (no finish reason, no token usage).
    /// Returns `0` on success or `-1` on failure (writing a fresh
    /// `*mut BlazenError` into `*out_err`).
    pub on_done: extern "C" fn(user_data: *mut c_void, out_err: *mut *mut BlazenError) -> i32,

    /// Receive a fatal error from the stream. `err` is a caller-owned
    /// `*mut BlazenError` freed via `blazen_error_free`. Returns `0` on
    /// success or `-1` on failure (writing the callback's own error into
    /// `*out_err`).
    pub on_error: extern "C" fn(
        user_data: *mut c_void,
        err: *mut BlazenError,
        out_err: *mut *mut BlazenError,
    ) -> i32,
}

// SAFETY: the foreign side guarantees thread-safety of `user_data` and the
// function pointers, matching the documented contract on
// `BlazenCompletionStreamSinkVTable`.
unsafe impl Send for BlazenMusicStreamSinkVTable {}
// SAFETY: see the `Send` impl above — same foreign-side guarantee covers
// shared-reference access from multiple threads.
unsafe impl Sync for BlazenMusicStreamSinkVTable {}

/// Rust-side trampoline wrapping a foreign [`BlazenMusicStreamSinkVTable`].
/// Implements the [`MusicStreamSink`] trait by dispatching through the
/// vtable's function pointers.
///
/// Owns the vtable's `user_data` — drops it via `drop_user_data` exactly
/// once when this wrapper drops (after the stream terminates, or on an
/// early-return failure path before the stream starts).
pub(crate) struct CMusicStreamSink {
    vtable: BlazenMusicStreamSinkVTable,
}

impl Drop for CMusicStreamSink {
    fn drop(&mut self) {
        // SAFETY: by the vtable contract, `drop_user_data` is the foreign
        // side's release thunk for `user_data` and is safe to call exactly
        // once when the wrapper is destroyed. Every early-return path in
        // the stream pump functions below that aborts BEFORE constructing
        // `CMusicStreamSink` calls `drop_user_data` directly; once the
        // wrapper is constructed only this `Drop` impl invokes it.
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

#[async_trait]
impl MusicStreamSink for CMusicStreamSink {
    // `InnerError` is large (it carries every variant's payload inline), but
    // it's the shared error type across `blazen_uniffi` and we don't get to
    // choose its representation here. Mirrors the same `allow` on
    // `CStreamSink::on_chunk` above.
    #[allow(clippy::result_large_err)]
    async fn on_chunk(&self, chunk: InnerMusicChunk) -> BlazenResult<()> {
        // Wrap the chunk in a cabi handle the foreign callback can free.
        let chunk_ptr = BlazenMusicChunk::from(chunk).into_ptr();

        // Capture pointer + fn-pointer as primitives so the
        // `spawn_blocking` closure can be `'static + Send`. Cast the raw
        // pointers to `usize` so the closure doesn't need to capture a
        // `*mut c_void` (which is `!Send`).
        let user_data_addr = self.vtable.user_data as usize;
        let on_chunk_fn = self.vtable.on_chunk;
        let chunk_addr = chunk_ptr as usize;

        // SAFETY: see the `CStreamSink::on_chunk` impl above — same
        // foreign-side thread-safety guarantee covers `user_data`; the
        // function pointer is `Copy + Send + Sync`; the chunk pointer was
        // just minted from `Box::into_raw` so it's a unique allocation we're
        // handing off.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let chunk_ptr = chunk_addr as *mut BlazenMusicChunk;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_chunk_fn(user_data, chunk_ptr, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "music stream sink on_chunk returned non-zero status ({status}) without setting out_err"
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
                message: format!("music stream sink on_chunk task panicked: {join_err}"),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn on_done(&self) -> BlazenResult<()> {
        let user_data_addr = self.vtable.user_data as usize;
        let on_done_fn = self.vtable.on_done;

        // SAFETY: same thread-safety justification as `on_chunk`.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_done_fn(user_data, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "music stream sink on_done returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("music stream sink on_done task panicked: {join_err}"),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn on_error(&self, err: InnerError) -> BlazenResult<()> {
        // Wrap the inner error in a cabi handle the foreign callback can
        // free.
        let err_ptr = BlazenError::from(err).into_ptr();

        let user_data_addr = self.vtable.user_data as usize;
        let on_error_fn = self.vtable.on_error;
        let err_addr = err_ptr as usize;

        // SAFETY: same thread-safety justification as `on_chunk`.
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
                            "music stream sink on_error returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("music stream sink on_error task panicked: {join_err}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Music stream-pump entry points (music)
// ---------------------------------------------------------------------------

/// Synchronously drive a music streaming generation, dispatching each
/// chunk to the supplied sink. Blocks the calling thread on the upstream
/// tokio runtime.
///
/// On success returns `0` (the sink's `on_done` has fired). On a
/// stream-start failure returns `-1` and writes a fresh `*mut BlazenError`
/// into `*out_err`; backend-side and sink-side failures during the stream
/// are delivered through `on_error` rather than this return path — see
/// [`blazen_uniffi::compute_music::stream_generate_music_to_sink`] for the
/// full contract.
///
/// ## Ownership transfer
///
/// - `model` is BORROWED — the underlying `Arc<MusicModel>` is cloned
///   into the streaming call. Caller retains its handle.
/// - `sink` (the vtable) is CONSUMED — ownership of `user_data` transfers
///   to the wrapping `CMusicStreamSink`, which releases it via
///   `drop_user_data` on drop. On every early-return failure path that
///   aborts BEFORE constructing `CMusicStreamSink`, this function
///   explicitly invokes `(sink.drop_user_data)(sink.user_data)` to honour
///   the same contract.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenMusicModel` produced by the cabi
/// surface. `prompt` must be null OR a valid NUL-terminated UTF-8 buffer.
/// `sink.user_data` and the four `sink` function pointers must satisfy
/// the contracts documented on [`BlazenMusicStreamSinkVTable`]. `out_err`
/// must be null OR a writable slot for a single `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_stream_generate_music_blocking(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_music_model_stream_generate_music: null model",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_music_model_stream_generate_music: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();

    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    // Wrap the vtable; from here on, `CMusicStreamSink::drop` is responsible
    // for calling `drop_user_data`.
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });

    match stream_generate_music_to_sink_blocking(model_arc, prompt, duration_seconds, sink_arc) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drive a music streaming generation onto the upstream
/// tokio runtime, returning an opaque future handle that resolves to
/// `()` on completion. The caller observes completion via the future's
/// fd / [`crate::future::blazen_future_poll`] /
/// [`crate::future::blazen_future_wait`], then calls
/// [`crate::persist::blazen_future_take_unit`] to pop the result. Free
/// the future with [`crate::future::blazen_future_free`].
///
/// Returns null if `model` is null or `prompt` is null / non-UTF-8. On
/// those error paths `sink.drop_user_data` is still invoked so no
/// resources leak.
///
/// ## Ownership transfer
///
/// Same as [`blazen_music_model_stream_generate_music_blocking`]:
/// `model` is BORROWED, `sink` is CONSUMED.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenMusicModel`. `prompt` must be
/// null OR a valid NUL-terminated UTF-8 buffer. `sink` satisfies the
/// [`BlazenMusicStreamSinkVTable`] contract; its `user_data` is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_stream_generate_music(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });

    BlazenFuture::spawn::<(), _>(async move {
        stream_generate_music_to_sink(model_arc, prompt, duration_seconds, sink_arc).await
    })
}

// ---------------------------------------------------------------------------
// Music stream-pump entry points (sfx)
// ---------------------------------------------------------------------------

/// Synchronously drive an SFX streaming generation, dispatching each chunk
/// to the supplied sink. Mirrors
/// [`blazen_music_model_stream_generate_music_blocking`] semantics
/// (including the early-return `drop_user_data` discipline).
///
/// # Safety
///
/// Same contracts as
/// [`blazen_music_model_stream_generate_music_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_stream_generate_sfx_blocking(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_music_model_stream_generate_sfx: null model",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_music_model_stream_generate_sfx: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();

    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });

    match stream_generate_sfx_to_sink_blocking(model_arc, prompt, duration_seconds, sink_arc) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drive an SFX streaming generation. Mirrors
/// [`blazen_music_model_stream_generate_music`] semantics; the returned
/// future resolves to `()`, popped via
/// [`crate::persist::blazen_future_take_unit`].
///
/// # Safety
///
/// Same contracts as [`blazen_music_model_stream_generate_music`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_stream_generate_sfx(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });

    BlazenFuture::spawn::<(), _>(async move {
        stream_generate_sfx_to_sink(model_arc, prompt, duration_seconds, sink_arc).await
    })
}

// ===========================================================================
// VcStreamSink trampoline + stream pump entry points
// ===========================================================================
//
// Symmetric to the `MusicStreamSink` trampoline above. Foreign callers
// implement [`VcStreamSink`] (`on_chunk` + `on_done` + `on_error`) via a
// flat `#[repr(C)]` vtable; the cabi side wraps that vtable in a Rust trait
// impl that dispatches through `tokio::task::spawn_blocking` (same rationale
// as the music sink: foreign callbacks may block, we don't want a slow sink
// starving the async runtime).
//
// Wire-format differences vs the music sink:
//
// - `on_chunk` receives `*mut BlazenVcChunk` instead of
//   `*mut BlazenMusicChunk`. Caller-owned; foreign callback frees via
//   [`crate::compute_records::blazen_vc_chunk_free`].
// - `on_done` matches the music-sink shape (no auxiliary payload).
// - `on_error` matches the completion-sink shape: receives a caller-owned
//   `*mut BlazenError` the foreign callback frees via `blazen_error_free`.

/// Function-pointer vtable a foreign caller fills out to implement
/// [`VcStreamSink`] across the C ABI.
///
/// All three callbacks plus `drop_user_data` are required. The vtable is
/// consumed by the voice-conversion stream-pump functions below
/// ([`blazen_vc_model_stream_convert_pcm_to_sink`] /
/// [`blazen_vc_model_stream_convert_pcm_to_sink_blocking`]); they take
/// ownership of `user_data` and the function pointers for the lifetime of
/// the streaming call. `drop_user_data` runs exactly once when the wrapping
/// [`CVcStreamSink`] drops (after the stream has terminated, or on an
/// early-return failure path before the stream starts).
#[repr(C)]
pub struct BlazenVcStreamSinkVTable {
    /// Opaque foreign-side context handed back to each callback. Owned by
    /// this vtable struct; released via `drop_user_data` when the wrapper
    /// drops.
    pub user_data: *mut c_void,

    /// Invoked exactly once when the wrapping `CVcStreamSink` drops.
    /// Implementations should reclaim and release `user_data`.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),

    /// Receive a single chunk. `chunk` is caller-owned; the callback MUST
    /// free it via [`crate::compute_records::blazen_vc_chunk_free`] (or
    /// consume it into a derivative structure). Returns `0` on success or
    /// `-1` on failure (writing a fresh `*mut BlazenError` into `*out_err`).
    /// A `-1` aborts the stream.
    pub on_chunk: extern "C" fn(
        user_data: *mut c_void,
        chunk: *mut BlazenVcChunk,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// Receive the terminal completion signal. Voice-conversion streams
    /// carry no auxiliary `on_done` payload. Returns `0` on success or
    /// `-1` on failure (writing a fresh `*mut BlazenError` into `*out_err`).
    pub on_done: extern "C" fn(user_data: *mut c_void, out_err: *mut *mut BlazenError) -> i32,

    /// Receive a fatal error from the stream. `err` is a caller-owned
    /// `*mut BlazenError` freed via `blazen_error_free`. Returns `0` on
    /// success or `-1` on failure (writing the callback's own error into
    /// `*out_err`).
    pub on_error: extern "C" fn(
        user_data: *mut c_void,
        err: *mut BlazenError,
        out_err: *mut *mut BlazenError,
    ) -> i32,
}

// SAFETY: the foreign side guarantees thread-safety of `user_data` and the
// function pointers, matching the documented contract on
// `BlazenCompletionStreamSinkVTable`.
unsafe impl Send for BlazenVcStreamSinkVTable {}
// SAFETY: see the `Send` impl above — same foreign-side guarantee covers
// shared-reference access from multiple threads.
unsafe impl Sync for BlazenVcStreamSinkVTable {}

/// Rust-side trampoline wrapping a foreign [`BlazenVcStreamSinkVTable`].
/// Implements the [`VcStreamSink`] trait by dispatching through the
/// vtable's function pointers.
///
/// Owns the vtable's `user_data` — drops it via `drop_user_data` exactly
/// once when this wrapper drops (after the stream terminates, or on an
/// early-return failure path before the stream starts).
pub(crate) struct CVcStreamSink {
    vtable: BlazenVcStreamSinkVTable,
}

impl Drop for CVcStreamSink {
    fn drop(&mut self) {
        // SAFETY: by the vtable contract, `drop_user_data` is the foreign
        // side's release thunk for `user_data` and is safe to call exactly
        // once when the wrapper is destroyed. Every early-return path in
        // the stream pump functions below that aborts BEFORE constructing
        // `CVcStreamSink` calls `drop_user_data` directly; once the
        // wrapper is constructed only this `Drop` impl invokes it.
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

#[async_trait]
impl VcStreamSink for CVcStreamSink {
    // `InnerError` is large (it carries every variant's payload inline), but
    // it's the shared error type across `blazen_uniffi` and we don't get to
    // choose its representation here. Mirrors the same `allow` on
    // `CStreamSink::on_chunk` above.
    #[allow(clippy::result_large_err)]
    async fn on_chunk(&self, chunk: InnerVcChunk) -> BlazenResult<()> {
        // Wrap the chunk in a cabi handle the foreign callback can free.
        let chunk_ptr = BlazenVcChunk::from(chunk).into_ptr();

        // Capture pointer + fn-pointer as primitives so the
        // `spawn_blocking` closure can be `'static + Send`. Cast the raw
        // pointers to `usize` so the closure doesn't need to capture a
        // `*mut c_void` (which is `!Send`).
        let user_data_addr = self.vtable.user_data as usize;
        let on_chunk_fn = self.vtable.on_chunk;
        let chunk_addr = chunk_ptr as usize;

        // SAFETY: see the `CStreamSink::on_chunk` impl above — same
        // foreign-side thread-safety guarantee covers `user_data`; the
        // function pointer is `Copy + Send + Sync`; the chunk pointer was
        // just minted from `Box::into_raw` so it's a unique allocation we're
        // handing off.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let chunk_ptr = chunk_addr as *mut BlazenVcChunk;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_chunk_fn(user_data, chunk_ptr, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "vc stream sink on_chunk returned non-zero status ({status}) without setting out_err"
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
                message: format!("vc stream sink on_chunk task panicked: {join_err}"),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn on_done(&self) -> BlazenResult<()> {
        let user_data_addr = self.vtable.user_data as usize;
        let on_done_fn = self.vtable.on_done;

        // SAFETY: same thread-safety justification as `on_chunk`.
        let join = tokio::task::spawn_blocking(move || -> Result<(), InnerError> {
            let user_data = user_data_addr as *mut c_void;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = on_done_fn(user_data, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "vc stream sink on_done returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("vc stream sink on_done task panicked: {join_err}"),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn on_error(&self, err: InnerError) -> BlazenResult<()> {
        // Wrap the inner error in a cabi handle the foreign callback can
        // free.
        let err_ptr = BlazenError::from(err).into_ptr();

        let user_data_addr = self.vtable.user_data as usize;
        let on_error_fn = self.vtable.on_error;
        let err_addr = err_ptr as usize;

        // SAFETY: same thread-safety justification as `on_chunk`.
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
                            "vc stream sink on_error returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("vc stream sink on_error task panicked: {join_err}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Voice-conversion stream-pump entry points
// ---------------------------------------------------------------------------

/// Synchronously drive a voice-conversion streaming call, dispatching each
/// chunk to the supplied sink. Blocks the calling thread on the upstream
/// tokio runtime.
///
/// `input_pcm` / `input_pcm_len` describe the source utterance as 32-bit
/// float PCM at the backend's expected source sample rate (typically
/// 16 kHz mono for RVC). The buffer is copied into an owned `Vec<f32>`
/// before the stream starts, so the caller may free / reuse the buffer the
/// instant this function returns. A null `input_pcm` is treated as an
/// empty input vector.
///
/// On success returns `0` (the sink's `on_done` has fired). On a
/// stream-start failure returns `-1` and writes a fresh `*mut BlazenError`
/// into `*out_err`; backend-side and sink-side failures during the stream
/// are delivered through `on_error` rather than this return path — see
/// [`blazen_uniffi::compute_vc::stream_convert_pcm_to_sink`] for the full
/// contract.
///
/// ## Ownership transfer
///
/// - `model` is BORROWED — the underlying `Arc<VcModel>` is cloned into
///   the streaming call. Caller retains its handle.
/// - `sink` (the vtable) is CONSUMED — ownership of `user_data` transfers
///   to the wrapping `CVcStreamSink`, which releases it via
///   `drop_user_data` on drop. On every early-return failure path that
///   aborts BEFORE constructing `CVcStreamSink`, this function explicitly
///   invokes `(sink.drop_user_data)(sink.user_data)` to honour the same
///   contract.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenVcModel` produced by the cabi
/// surface. `input_pcm` must be null OR point to a readable `f32` buffer
/// of at least `input_pcm_len` elements. `target_voice_id` must be null
/// OR a valid NUL-terminated UTF-8 buffer. `sink.user_data` and the four
/// `sink` function pointers must satisfy the contracts documented on
/// [`BlazenVcStreamSinkVTable`]. `out_err` must be null OR a writable slot
/// for a single `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_stream_convert_pcm_to_sink_blocking(
    model: *const BlazenVcModel,
    input_pcm: *const f32,
    input_pcm_len: usize,
    target_voice_id: *const c_char,
    sink: BlazenVcStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_vc_model_stream_convert_pcm_to_sink: null model",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on
    // `target_voice_id`.
    let Some(target_voice_id) = (unsafe { crate::string::cstr_to_str(target_voice_id) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_vc_model_stream_convert_pcm_to_sink: null or non-UTF-8 target_voice_id",
            )
        };
    };
    let target_voice_id = target_voice_id.to_owned();

    let pcm = if input_pcm.is_null() || input_pcm_len == 0 {
        Vec::<f32>::new()
    } else {
        // SAFETY: caller has guaranteed `input_pcm` points to at least
        // `input_pcm_len` readable `f32` elements.
        unsafe { std::slice::from_raw_parts(input_pcm, input_pcm_len) }.to_vec()
    };

    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    // Wrap the vtable; from here on, `CVcStreamSink::drop` is responsible
    // for calling `drop_user_data`.
    let sink_arc: Arc<dyn VcStreamSink> = Arc::new(CVcStreamSink { vtable: sink });

    match stream_convert_pcm_to_sink_blocking(model_arc, pcm, target_voice_id, sink_arc) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drive a voice-conversion streaming call onto the upstream
/// tokio runtime, returning an opaque future handle that resolves to `()`
/// on completion. The caller observes completion via the future's fd /
/// [`crate::future::blazen_future_poll`] /
/// [`crate::future::blazen_future_wait`], then calls
/// [`crate::persist::blazen_future_take_unit`] to pop the result. Free the
/// future with [`crate::future::blazen_future_free`].
///
/// Returns null if `model` is null or `target_voice_id` is null /
/// non-UTF-8. On those error paths `sink.drop_user_data` is still invoked
/// so no resources leak.
///
/// ## Ownership transfer
///
/// Same as [`blazen_vc_model_stream_convert_pcm_to_sink_blocking`]:
/// `model` is BORROWED, `sink` is CONSUMED. `input_pcm` is copied into an
/// owned `Vec<f32>` before the future is spawned, so the caller may free /
/// reuse the buffer the instant this function returns.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenVcModel`. `input_pcm` must be
/// null OR point to a readable `f32` buffer of at least `input_pcm_len`
/// elements. `target_voice_id` must be null OR a valid NUL-terminated
/// UTF-8 buffer. `sink` satisfies the [`BlazenVcStreamSinkVTable`]
/// contract; its `user_data` is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_stream_convert_pcm_to_sink(
    model: *const BlazenVcModel,
    input_pcm: *const f32,
    input_pcm_len: usize,
    target_voice_id: *const c_char,
    sink: BlazenVcStreamSinkVTable,
) -> *mut BlazenFuture {
    if model.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on
    // `target_voice_id`.
    let Some(target_voice_id) = (unsafe { crate::string::cstr_to_str(target_voice_id) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let target_voice_id = target_voice_id.to_owned();

    let pcm = if input_pcm.is_null() || input_pcm_len == 0 {
        Vec::<f32>::new()
    } else {
        // SAFETY: caller has guaranteed `input_pcm` points to at least
        // `input_pcm_len` readable `f32` elements.
        unsafe { std::slice::from_raw_parts(input_pcm, input_pcm_len) }.to_vec()
    };

    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    let sink_arc: Arc<dyn VcStreamSink> = Arc::new(CVcStreamSink { vtable: sink });

    BlazenFuture::spawn::<(), _>(async move {
        stream_convert_pcm_to_sink(model_arc, pcm, target_voice_id, sink_arc).await
    })
}

// ===========================================================================
// Per-engine music streaming entry points
// ===========================================================================
//
// Each pair mirrors `blazen_music_model_stream_generate_music` / `_blocking`
// (or the `_sfx` variants) exactly — same null + UTF-8 checks, same
// `CMusicStreamSink` construction from the vtable, same consume-`sink`
// discipline — but borrows a per-engine cabi music provider opaque (cloning
// its inner `Arc<...Provider>`) and dispatches to the matching per-engine
// `blazen_uniffi::concrete::music` streaming free function instead of the
// central `stream_generate_*_to_sink`. The central fns + the `CMusicStreamSink`
// trampoline above are reused verbatim (no duplicate trampoline / sink impl).
//
// MusicGen is music-only; AudioGen + StableAudio expose both music and sfx.

// MusicGenProvider (music only) --------------------------------------------

/// Synchronously drive a music streaming generation through the
/// `MusicGenProvider` concrete provider. Mirrors
/// [`blazen_music_model_stream_generate_music_blocking`] semantics but
/// dispatches to
/// [`blazen_uniffi::concrete::music::musicgen_provider_stream_music_to_sink_blocking`].
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenMusicGenProvider`. `prompt` must
/// be null OR a valid NUL-terminated UTF-8 buffer. `sink` satisfies the
/// [`BlazenMusicStreamSinkVTable`] contract; its `user_data` is consumed.
/// `out_err` must be null OR a writable slot for a single `*mut BlazenError`
/// write.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_stream_music_blocking(
    provider: *const BlazenMusicGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_musicgen_provider_stream_music: null provider",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_musicgen_provider_stream_music: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    match blazen_uniffi::concrete::music::musicgen_provider_stream_music_to_sink_blocking(
        provider_arc,
        prompt,
        duration_seconds,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drive a music streaming generation through the
/// `MusicGenProvider` concrete provider. Mirrors
/// [`blazen_music_model_stream_generate_music`] semantics.
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenMusicGenProvider`. `prompt` must
/// be null OR a valid NUL-terminated UTF-8 buffer. `sink` satisfies the
/// [`BlazenMusicStreamSinkVTable`] contract; its `user_data` is consumed.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_stream_music(
    provider: *const BlazenMusicGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::music::musicgen_provider_stream_music_to_sink(
            provider_arc,
            prompt,
            duration_seconds,
            sink_arc,
        )
        .await
    })
}

// AudioGenProvider (music + sfx) -------------------------------------------

/// Synchronous music streaming for `AudioGenProvider`. `AudioGen` is
/// sfx-primary;
/// its `stream_generate_music` delivers `Unsupported` through the sink's
/// `on_error`. Prefer [`blazen_audiogen_provider_stream_sfx_blocking`]. See
/// [`blazen_musicgen_provider_stream_music_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music_blocking`]
/// (`provider` is a `BlazenAudioGenProvider`).
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_stream_music_blocking(
    provider: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_audiogen_provider_stream_music: null provider",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_audiogen_provider_stream_music: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    match blazen_uniffi::concrete::music::audiogen_provider_stream_music_to_sink_blocking(
        provider_arc,
        prompt,
        duration_seconds,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async music streaming for `AudioGenProvider`. See
/// [`blazen_musicgen_provider_stream_music`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music`]
/// (`provider` is a `BlazenAudioGenProvider`).
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_stream_music(
    provider: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::music::audiogen_provider_stream_music_to_sink(
            provider_arc,
            prompt,
            duration_seconds,
            sink_arc,
        )
        .await
    })
}

/// Synchronous SFX streaming for `AudioGenProvider`. See
/// [`blazen_musicgen_provider_stream_music_blocking`] (chunks flow through the
/// same [`BlazenMusicStreamSinkVTable`]).
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music_blocking`]
/// (`provider` is a `BlazenAudioGenProvider`).
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_stream_sfx_blocking(
    provider: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_audiogen_provider_stream_sfx: null provider",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_audiogen_provider_stream_sfx: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    match blazen_uniffi::concrete::music::audiogen_provider_stream_sfx_to_sink_blocking(
        provider_arc,
        prompt,
        duration_seconds,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async SFX streaming for `AudioGenProvider`. See
/// [`blazen_musicgen_provider_stream_music`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music`]
/// (`provider` is a `BlazenAudioGenProvider`).
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_stream_sfx(
    provider: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::music::audiogen_provider_stream_sfx_to_sink(
            provider_arc,
            prompt,
            duration_seconds,
            sink_arc,
        )
        .await
    })
}

// StableAudioProvider (music + sfx) ----------------------------------------

/// Synchronous music streaming for `StableAudioProvider`. See
/// [`blazen_musicgen_provider_stream_music_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music_blocking`]
/// (`provider` is a `BlazenStableAudioProvider`).
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_stream_music_blocking(
    provider: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_stable_audio_provider_stream_music: null provider",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_stable_audio_provider_stream_music: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    match blazen_uniffi::concrete::music::stable_audio_provider_stream_music_to_sink_blocking(
        provider_arc,
        prompt,
        duration_seconds,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async music streaming for `StableAudioProvider`. See
/// [`blazen_musicgen_provider_stream_music`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music`]
/// (`provider` is a `BlazenStableAudioProvider`).
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_stream_music(
    provider: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::music::stable_audio_provider_stream_music_to_sink(
            provider_arc,
            prompt,
            duration_seconds,
            sink_arc,
        )
        .await
    })
}

/// Synchronous SFX streaming for `StableAudioProvider`. See
/// [`blazen_musicgen_provider_stream_music_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music_blocking`]
/// (`provider` is a `BlazenStableAudioProvider`).
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_stream_sfx_blocking(
    provider: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_stable_audio_provider_stream_sfx: null provider",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_stable_audio_provider_stream_sfx: null or non-UTF-8 prompt",
            )
        };
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    match blazen_uniffi::concrete::music::stable_audio_provider_stream_sfx_to_sink_blocking(
        provider_arc,
        prompt,
        duration_seconds,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Async SFX streaming for `StableAudioProvider`. See
/// [`blazen_musicgen_provider_stream_music`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_stream_music`]
/// (`provider` is a `BlazenStableAudioProvider`).
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_stream_sfx(
    provider: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    sink: BlazenMusicStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { crate::string::cstr_to_str(prompt) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn MusicStreamSink> = Arc::new(CMusicStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::music::stable_audio_provider_stream_sfx_to_sink(
            provider_arc,
            prompt,
            duration_seconds,
            sink_arc,
        )
        .await
    })
}

// ===========================================================================
// Per-engine voice-conversion streaming entry points
// ===========================================================================
//
// Mirrors `blazen_vc_model_stream_convert_pcm_to_sink` / `_blocking` exactly
// — same null + UTF-8 checks, same PCM copy, same `CVcStreamSink` construction
// from the vtable, same consume-`sink` discipline — but borrows the per-engine
// `BlazenRvcProvider` opaque (cloning its inner `Arc<RvcProvider>`) and
// dispatches to
// [`blazen_uniffi::concrete::vc::rvc_provider_stream_convert_pcm_to_sink`]
// instead of the central `stream_convert_pcm_to_sink`. The central fns + the
// `CVcStreamSink` trampoline above are reused verbatim.

// RvcProvider --------------------------------------------------------------

/// Synchronously drive a voice-conversion streaming call through the
/// `RvcProvider` concrete provider. Mirrors
/// [`blazen_vc_model_stream_convert_pcm_to_sink_blocking`] semantics but
/// dispatches to
/// [`blazen_uniffi::concrete::vc::rvc_provider_stream_convert_pcm_to_sink_blocking`].
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenRvcProvider`. `input_pcm` must be
/// null OR point to a readable `f32` buffer of at least `input_pcm_len`
/// elements. `target_voice_id` must be null OR a valid NUL-terminated UTF-8
/// buffer. `sink` satisfies the [`BlazenVcStreamSinkVTable`] contract; its
/// `user_data` is consumed. `out_err` must be null OR a writable slot for a
/// single `*mut BlazenError` write.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_stream_convert_pcm_blocking(
    provider: *const BlazenRvcProvider,
    input_pcm: *const f32,
    input_pcm_len: usize,
    target_voice_id: *const c_char,
    sink: BlazenVcStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_rvc_provider_stream_convert_pcm: null provider",
            )
        };
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(target_voice_id) = (unsafe { crate::string::cstr_to_str(target_voice_id) }) else {
        (sink.drop_user_data)(sink.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(
                out_err,
                "blazen_rvc_provider_stream_convert_pcm: null or non-UTF-8 target_voice_id",
            )
        };
    };
    let target_voice_id = target_voice_id.to_owned();
    let pcm = if input_pcm.is_null() || input_pcm_len == 0 {
        Vec::<f32>::new()
    } else {
        // SAFETY: caller has guaranteed `input_pcm` points to at least
        // `input_pcm_len` readable `f32` elements.
        unsafe { std::slice::from_raw_parts(input_pcm, input_pcm_len) }.to_vec()
    };
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn VcStreamSink> = Arc::new(CVcStreamSink { vtable: sink });
    match blazen_uniffi::concrete::vc::rvc_provider_stream_convert_pcm_to_sink_blocking(
        provider_arc,
        pcm,
        target_voice_id,
        sink_arc,
    ) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Asynchronously drive a voice-conversion streaming call through the
/// `RvcProvider` concrete provider. Mirrors
/// [`blazen_vc_model_stream_convert_pcm_to_sink`] semantics.
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenRvcProvider`. `input_pcm` must be
/// null OR point to a readable `f32` buffer of at least `input_pcm_len`
/// elements. `target_voice_id` must be null OR a valid NUL-terminated UTF-8
/// buffer. `sink` satisfies the [`BlazenVcStreamSinkVTable`] contract; its
/// `user_data` is consumed.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_stream_convert_pcm(
    provider: *const BlazenRvcProvider,
    input_pcm: *const f32,
    input_pcm_len: usize,
    target_voice_id: *const c_char,
    sink: BlazenVcStreamSinkVTable,
) -> *mut BlazenFuture {
    if provider.is_null() {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(target_voice_id) = (unsafe { crate::string::cstr_to_str(target_voice_id) }) else {
        (sink.drop_user_data)(sink.user_data);
        return std::ptr::null_mut();
    };
    let target_voice_id = target_voice_id.to_owned();
    let pcm = if input_pcm.is_null() || input_pcm_len == 0 {
        Vec::<f32>::new()
    } else {
        // SAFETY: caller has guaranteed `input_pcm` points to at least
        // `input_pcm_len` readable `f32` elements.
        unsafe { std::slice::from_raw_parts(input_pcm, input_pcm_len) }.to_vec()
    };
    // SAFETY: caller has guaranteed `provider` is a live handle.
    let provider_handle = unsafe { &*provider };
    let provider_arc = Arc::clone(&provider_handle.0);
    let sink_arc: Arc<dyn VcStreamSink> = Arc::new(CVcStreamSink { vtable: sink });
    BlazenFuture::spawn::<(), _>(async move {
        blazen_uniffi::concrete::vc::rvc_provider_stream_convert_pcm_to_sink(
            provider_arc,
            pcm,
            target_voice_id,
            sink_arc,
        )
        .await
    })
}
