//! Cabi `CustomProvider` — opaque handle wrapping `blazen_llm`'s typed-trait
//! [`CustomProviderHandle`] plus a C-callable vtable that Ruby (or any other
//! cbindgen-driven host) fills out to provide a *typed* foreign implementation
//! of the 16-method `CustomProvider` trait.
//!
//! ## V2 model (Phase B-Pivot.1)
//!
//! `blazen_llm::providers::custom::CustomProvider` is a foreign-implementable
//! trait with 16 typed async methods (completion + streaming + embedding plus
//! 13 compute/media capabilities). Every method has a default
//! `BlazenError::Unsupported` implementation; users override only the methods
//! their provider actually supports.
//!
//! The cabi exposes that trait via [`BlazenCustomProviderVTable`] — a flat
//! `#[repr(C)]` struct of 16 `extern "C"` function pointers plus a
//! `user_data` opaque context and a `drop_user_data` thunk. Each function
//! pointer accepts typed `*const`/`*mut` opaque request handles (mirroring the
//! types already exposed in [`crate::compute_requests`] and
//! [`crate::llm_records`]) and writes a typed result back into a caller-
//! supplied out-pointer (or an error into `out_err`).
//!
//! The Rust-side adapter [`CabiCustomProviderAdapter`] implements the typed
//! trait by dispatching each call to the vtable on a `spawn_blocking` worker
//! (so foreign callbacks may block while doing their own async work without
//! starving the tokio scheduler — same pattern as
//! [`crate::step_handler::CStepHandler`] / [`crate::stream_sink::CStreamSink`]).
//!
//! ## Factories exposed
//!
//! - [`blazen_custom_provider_from_vtable`] — fully-typed foreign override.
//!   Caller supplies a [`BlazenCustomProviderVTable`] (the vtable is consumed
//!   and its `user_data` ownership transfers to the adapter; the adapter
//!   invokes `drop_user_data` exactly once on drop).
//! - [`blazen_custom_provider_openai_compat`] — arbitrary OpenAI-compatible
//!   server (clones the supplied [`crate::provider_api_protocol::BlazenOpenAiCompatConfig`]).
//! - [`blazen_custom_provider_ollama`] — convenience for Ollama servers.
//! - [`blazen_custom_provider_lm_studio`] — convenience for LM Studio servers.
//!
//! ## Async return values
//!
//! Each typed-call C entry point returns a `*mut BlazenFuture` that the host
//! polls / waits on, then pops the typed result with the matching
//! `blazen_future_take_custom_*` entry point. The typed-take wrappers in this
//! module monomorphise [`crate::future::BlazenFuture::take_typed`] onto the
//! result types from [`blazen_llm::compute`].
//!
//! ## Ownership conventions
//!
//! - Every `blazen_custom_provider_*` factory returns a heap-allocated handle
//!   the caller owns; release with [`blazen_custom_provider_free`].
//! - Compute methods take the request handle by value (consuming, via
//!   `Box::from_raw`) so they can move the inner request into the async task
//!   without an extra clone. After the call, the caller must NOT free the
//!   request handle — ownership transferred.
//! - [`blazen_custom_provider_as_base_provider`] clones the inner
//!   `CustomProviderHandle` (which implements `CompletionModel`) into a brand-
//!   new `BaseProvider` whose inner `CompletionModel` is an
//!   `Arc<CustomProviderHandle>`. The returned `BlazenBaseProvider` is
//!   independent of the source handle's lifetime and must be freed with
//!   [`crate::provider_base::blazen_base_provider_free`].

#![allow(dead_code)]
// `blazen_uniffi::BlazenError` is ~168 bytes (every variant inlined). Boxing
// it would force allocations on every `?` and break `From<...> for InnerError`
// patterns; the existing `step_handler` / `stream_sink` modules accept the
// same lint locally. We follow the same convention here.
#![allow(clippy::result_large_err)]

use std::ffi::{c_char, c_void};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::{self, Stream};

use blazen_llm::compute::{
    AudioResult as InnerAudioResult, BackgroundRemovalRequest as InnerBackgroundRemovalRequest,
    ImageRequest as InnerImageRequest, ImageResult as InnerImageResult,
    MusicRequest as InnerMusicRequest, SpeechRequest as InnerSpeechRequest,
    ThreeDRequest as InnerThreeDRequest, ThreeDResult as InnerThreeDResult,
    TranscriptionRequest as InnerTranscriptionRequest,
    TranscriptionResult as InnerTranscriptionResult, UpscaleRequest as InnerUpscaleRequest,
    VideoRequest as InnerVideoRequest, VideoResult as InnerVideoResult,
    VoiceCloneRequest as InnerVoiceCloneRequest, VoiceHandle as InnerVoiceHandle,
};
use blazen_llm::error::BlazenError as LlmError;
use blazen_llm::providers::base::BaseProvider as InnerBaseProvider;
use blazen_llm::providers::custom::{
    self as inner_custom, CustomProvider as InnerCustomProviderTrait,
    CustomProviderHandle as InnerCustomProviderHandle,
};
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::{
    CompletionRequest as LlmCompletionRequest, CompletionResponse as LlmCompletionResponse,
    EmbeddingResponse as LlmEmbeddingResponse, StreamChunk as LlmStreamChunk,
};

use blazen_uniffi::errors::BlazenError as UniffiError;
use blazen_uniffi::llm::{
    CompletionRequest as UniffiCompletionRequest, CompletionResponse as UniffiCompletionResponse,
    EmbeddingResponse as UniffiEmbeddingResponse,
};
use blazen_uniffi::streaming::StreamChunk as UniffiStreamChunk;

use crate::compute_requests::{
    BlazenBackgroundRemovalRequest, BlazenImageRequest, BlazenMusicRequest, BlazenSpeechRequest,
    BlazenThreeDRequest, BlazenTranscriptionRequest, BlazenUpscaleRequest, BlazenVideoRequest,
    BlazenVoiceCloneRequest,
};
use crate::compute_results::{
    BlazenAudioResult, BlazenImageResult, BlazenThreeDResult, BlazenTranscriptionResult,
    BlazenVideoResult, BlazenVoiceHandle,
};
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm_records::{
    BlazenCompletionRequest, BlazenCompletionResponse, BlazenEmbeddingResponse,
};
use crate::provider_api_protocol::BlazenOpenAiCompatConfig;
use crate::provider_base::BlazenBaseProvider;
use crate::streaming_records::BlazenStreamChunk;
use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_llm::providers::CustomProviderHandle>`.
///
/// `Arc` so the compute methods can clone-and-move the handle into the
/// spawned async task without forcing a deep copy of the inner defaults
/// bundle on every call.
#[repr(C)]
pub struct BlazenCustomProvider(pub(crate) Arc<InnerCustomProviderHandle>);

impl BlazenCustomProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenCustomProvider {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerCustomProviderHandle>> for BlazenCustomProvider {
    fn from(inner: Arc<InnerCustomProviderHandle>) -> Self {
        Self(inner)
    }
}

impl From<InnerCustomProviderHandle> for BlazenCustomProvider {
    fn from(inner: InnerCustomProviderHandle) -> Self {
        Self(Arc::new(inner))
    }
}

// ---------------------------------------------------------------------------
// Error bridging
// ---------------------------------------------------------------------------

/// Bridge a `blazen_uniffi` error (the error type the cabi surface speaks
/// natively) back into the `blazen_llm` error type expected by the
/// `CustomProvider` trait. Lossy by design — variants the upstream trait
/// can't represent collapse to `Request` / `Provider`.
fn uniffi_err_to_llm(err: UniffiError) -> LlmError {
    match err {
        UniffiError::Auth { message } => LlmError::Auth { message },
        UniffiError::RateLimit { retry_after_ms, .. } => LlmError::RateLimit { retry_after_ms },
        UniffiError::Timeout { elapsed_ms, .. } => LlmError::Timeout { elapsed_ms },
        UniffiError::Validation { message } => LlmError::Validation {
            field: None,
            message,
        },
        UniffiError::ContentPolicy { message } => LlmError::ContentPolicy { message },
        UniffiError::Unsupported { message } => LlmError::Unsupported { message },
        UniffiError::Tool { message } => LlmError::Tool {
            name: None,
            message,
        },
        UniffiError::Provider {
            provider,
            message,
            status,
            ..
        } => LlmError::Provider {
            provider: provider.unwrap_or_default(),
            message,
            status_code: status.and_then(|s| u16::try_from(s).ok()),
        },
        UniffiError::Compute { message }
        | UniffiError::Media { message }
        | UniffiError::Internal { message }
        | UniffiError::Workflow { message }
        | UniffiError::Peer { message, .. }
        | UniffiError::Persist { message }
        | UniffiError::Prompt { message, .. }
        | UniffiError::Memory { message, .. }
        | UniffiError::Cache { message, .. } => LlmError::request(message),
        UniffiError::Cancelled => LlmError::request("cancelled"),
    }
}

// ---------------------------------------------------------------------------
// VTable
// ---------------------------------------------------------------------------

/// C-side vtable for a foreign [`InnerCustomProviderTrait`] implementation.
///
/// All 18 fields are required (no nullable function pointers): the four
/// metadata fields (`provider_id`, `model_id`, `user_data`, `drop_user_data`),
/// and 16 typed-method fn-pointers — one per [`InnerCustomProviderTrait`]
/// method.
///
/// The vtable is consumed by [`blazen_custom_provider_from_vtable`] which
/// takes ownership of `user_data` for the lifetime of the resulting handle.
/// `drop_user_data` is invoked exactly once when the wrapping
/// [`CabiCustomProviderAdapter`] drops.
///
/// ## Ownership conventions per call
///
/// - **Request pointers** (e.g. `*mut BlazenSpeechRequest`): caller-owned and
///   the foreign callback OWNS them — must free via the matching `_free`
///   function before returning, OR consume them into a derivative structure.
/// - **Result pointers** (e.g. `out_result: *mut *mut BlazenAudioResult`):
///   on a success return (status 0) the foreign callback writes a fresh
///   caller-owned `*mut BlazenAudioResult` (produced by one of the
///   `compute_results` constructors) into the slot; ownership transfers back
///   to the Rust adapter.
/// - **Error pointers** (`out_err: *mut *mut BlazenError`): on a failure
///   return (status -1) the foreign callback writes a fresh caller-owned
///   `*mut BlazenError` into the slot; ownership transfers back to the Rust
///   adapter.
///
/// Every callback returns `0` on success or `-1` on failure. A `-1` return
/// surfaces as the corresponding [`InnerCustomProviderTrait`] method's
/// `Err(BlazenError)`.
///
/// ## Thread safety
///
/// The Rust adapter dispatches each callback through
/// `tokio::task::spawn_blocking`, so the foreign side is invoked from a
/// blocking-pool worker that may differ from the thread that registered the
/// vtable. The foreign side guarantees `user_data` and the function pointers
/// are safe to invoke from any thread (Ruby's `ffi` gem reacquires the GVL
/// automatically for declared `FFI::Callback` signatures; native hosts must
/// opt into thread-safety in their own runtime model).
#[repr(C)]
pub struct BlazenCustomProviderVTable {
    /// Opaque foreign-side context handed back to each function pointer.
    /// Owned by this vtable struct (released via `drop_user_data` on drop).
    pub user_data: *mut c_void,

    /// Called exactly once when the wrapping `CabiCustomProviderAdapter`
    /// drops. Implementations should reclaim and release `user_data`.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),

    /// Stable provider identifier (NUL-terminated UTF-8) — fed back by
    /// [`InnerCustomProviderTrait::provider_id`]. Owned by the vtable
    /// struct (NOT released by the adapter; the foreign side is expected to
    /// either return a static string or to free this pointer in
    /// `drop_user_data`).
    pub provider_id: *const c_char,

    /// Model identifier (NUL-terminated UTF-8) — fed back by
    /// [`InnerCustomProviderTrait::model_id`]. Same ownership as
    /// `provider_id`. A null pointer reuses `provider_id`.
    pub model_id: *const c_char,

    // ---- Completion / streaming / embedding ----
    /// Non-streaming chat completion.
    pub complete: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenCompletionRequest,
        out_response: *mut *mut BlazenCompletionResponse,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// Streaming chat completion. The foreign callback receives an opaque
    /// [`BlazenStreamPusher`] sink it pushes chunks into via
    /// [`blazen_stream_pusher_push`], finishing with either
    /// [`blazen_stream_pusher_end`] or [`blazen_stream_pusher_error`].
    ///
    /// Returns `0` on a successful stream START (chunks are then pushed
    /// asynchronously); `-1` if the stream can't be opened at all (writing a
    /// fresh `*mut BlazenError` into `*out_err`).
    pub stream: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenCompletionRequest,
        pusher: *mut BlazenStreamPusher,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// Batch text embedding. `texts` is an array of `count` NUL-terminated
    /// UTF-8 buffers — caller-owned for the duration of the call; the
    /// foreign callback MUST NOT free the underlying string memory (the
    /// adapter releases the backing Rust allocations after the call returns).
    pub embed: extern "C" fn(
        user_data: *mut c_void,
        texts: *const *const c_char,
        count: usize,
        out_response: *mut *mut BlazenEmbeddingResponse,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    // ---- Audio ----
    pub text_to_speech: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenSpeechRequest,
        out_result: *mut *mut BlazenAudioResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub generate_music: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenMusicRequest,
        out_result: *mut *mut BlazenAudioResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub generate_sfx: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenMusicRequest,
        out_result: *mut *mut BlazenAudioResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    // ---- Voice cloning ----
    pub clone_voice: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenVoiceCloneRequest,
        out_result: *mut *mut BlazenVoiceHandle,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    /// On success, the callback writes a fresh array of caller-owned
    /// `*mut BlazenVoiceHandle` pointers (count in `*out_count`). The Rust
    /// adapter takes ownership of each element and frees the outer pointer-
    /// array via [`blazen_voice_handle_list_free`].
    pub list_voices: extern "C" fn(
        user_data: *mut c_void,
        out_array: *mut *mut *mut BlazenVoiceHandle,
        out_count: *mut usize,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub delete_voice: extern "C" fn(
        user_data: *mut c_void,
        voice: *mut BlazenVoiceHandle,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    // ---- Image / video ----
    pub generate_image: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenImageRequest,
        out_result: *mut *mut BlazenImageResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub upscale_image: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenUpscaleRequest,
        out_result: *mut *mut BlazenImageResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub text_to_video: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenVideoRequest,
        out_result: *mut *mut BlazenVideoResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub image_to_video: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenVideoRequest,
        out_result: *mut *mut BlazenVideoResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub transcribe: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenTranscriptionRequest,
        out_result: *mut *mut BlazenTranscriptionResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub generate_3d: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenThreeDRequest,
        out_result: *mut *mut BlazenThreeDResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,

    pub remove_background: extern "C" fn(
        user_data: *mut c_void,
        request: *mut BlazenBackgroundRemovalRequest,
        out_result: *mut *mut BlazenImageResult,
        out_err: *mut *mut BlazenError,
    ) -> i32,
}

// SAFETY: the foreign side guarantees thread-safety of `user_data` and the
// function pointers, as documented on `BlazenCustomProviderVTable`. Ruby's
// `ffi` gem automatically reacquires the GVL for callbacks; Dart's
// `NativeCallable.listener` marshals back to the isolate event loop; native
// hosts must opt into thread-safety in their own runtime model.
unsafe impl Send for BlazenCustomProviderVTable {}
// SAFETY: see the `Send` impl above — same foreign-side guarantee covers
// shared-reference access from multiple threads.
unsafe impl Sync for BlazenCustomProviderVTable {}

// ---------------------------------------------------------------------------
// Stream pusher (one-shot channel from foreign callback into a Rust stream)
// ---------------------------------------------------------------------------

/// Opaque handle the foreign `stream` callback writes chunks into. The Rust
/// adapter wraps the receiving end in a `Pin<Box<dyn Stream<...>>>` that
/// drives the typed [`InnerCustomProviderTrait::stream`] return value.
///
/// Each pushed chunk is a `*mut BlazenStreamChunk` whose ownership transfers
/// into the pusher. The foreign caller terminates the stream with EXACTLY
/// ONE of:
/// - [`blazen_stream_pusher_end`] — graceful EOF.
/// - [`blazen_stream_pusher_error`] — fatal error.
///
/// Sending after termination is silently ignored (the channel half is
/// already closed).
pub struct BlazenStreamPusher {
    tx: tokio::sync::mpsc::UnboundedSender<Result<LlmStreamChunk, UniffiError>>,
}

impl BlazenStreamPusher {
    fn into_ptr(self) -> *mut BlazenStreamPusher {
        Box::into_raw(Box::new(self))
    }
}

/// Pushes a single `BlazenStreamChunk` into the stream. Consumes `chunk`.
/// No-op on null `pusher` or `chunk`. Returns `0` on success, `-1` if the
/// receiver has already been dropped (the stream consumer disappeared).
///
/// # Safety
///
/// `pusher` must be null OR a live `BlazenStreamPusher` previously handed to
/// the foreign callback by the cabi adapter. `chunk` must be null OR a live
/// `BlazenStreamChunk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_pusher_push(
    pusher: *mut BlazenStreamPusher,
    chunk: *mut BlazenStreamChunk,
) -> i32 {
    if pusher.is_null() || chunk.is_null() {
        if !chunk.is_null() {
            // SAFETY: per the contract, `chunk` came from `Box::into_raw`.
            drop(unsafe { Box::from_raw(chunk) });
        }
        return -1;
    }
    // SAFETY: caller guarantees `pusher` is live.
    let p = unsafe { &*pusher };
    // SAFETY: caller transferred ownership of `chunk`.
    let chunk_box = unsafe { Box::from_raw(chunk) };
    // The cabi handle holds a `blazen_uniffi::streaming::StreamChunk`;
    // convert to the `blazen_llm::types::StreamChunk` shape that the
    // upstream stream consumes.
    let uniffi_chunk: UniffiStreamChunk = chunk_box.0;
    let delta = if uniffi_chunk.content_delta.is_empty() {
        None
    } else {
        Some(uniffi_chunk.content_delta)
    };
    let finish_reason = if uniffi_chunk.is_final {
        Some("stop".to_owned())
    } else {
        None
    };
    // Tool-call deltas don't round-trip via the cabi wire today; the
    // foreign side that needs them can synthesise them through the
    // upstream `complete_streaming` path instead.
    let llm_chunk = LlmStreamChunk {
        delta,
        finish_reason,
        ..LlmStreamChunk::default()
    };
    match p.tx.send(Ok(llm_chunk)) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Signals end-of-stream gracefully. After this call the pusher is closed —
/// further pushes are no-ops. No-op on null `pusher`.
///
/// The foreign callback MUST call exactly one of `blazen_stream_pusher_end`
/// or [`blazen_stream_pusher_error`] before returning ownership of the
/// pusher to the Rust adapter; the adapter assumes the channel is sealed
/// once the synchronous callback returns success.
///
/// # Safety
///
/// `pusher` must be null OR a live `BlazenStreamPusher`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_pusher_end(pusher: *mut BlazenStreamPusher) {
    if pusher.is_null() {
        return;
    }
    // SAFETY: per the contract, `pusher` came from `Box::into_raw` inside the
    // adapter's stream-method dispatch. Reclaiming via `Box::from_raw` drops
    // the sender, which closes the channel.
    drop(unsafe { Box::from_raw(pusher) });
}

/// Signals a fatal stream error. Consumes `err`. No-op on null `pusher`
/// (still frees `err` if non-null).
///
/// # Safety
///
/// `pusher` must be null OR a live `BlazenStreamPusher`. `err` must be null
/// OR a live `BlazenError` produced by the cabi surface; ownership transfers
/// into the pusher (or is dropped on a null-pusher path).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_pusher_error(
    pusher: *mut BlazenStreamPusher,
    err: *mut BlazenError,
) {
    if pusher.is_null() {
        if !err.is_null() {
            // SAFETY: per the contract, `err` came from `Box::into_raw`.
            drop(unsafe { Box::from_raw(err) });
        }
        return;
    }
    // SAFETY: per the contract, `pusher` came from `Box::into_raw`.
    let p_box = unsafe { Box::from_raw(pusher) };
    let inner_err = if err.is_null() {
        UniffiError::Internal {
            message: "stream pusher closed without an error payload".into(),
        }
    } else {
        // SAFETY: per the contract, `err` came from `Box::into_raw`.
        unsafe { Box::from_raw(err) }.inner
    };
    // Send the error then let the pusher box drop, sealing the channel.
    let _ = p_box.tx.send(Err(inner_err));
}

// ---------------------------------------------------------------------------
// Adapter — bridges the C vtable into `InnerCustomProviderTrait`
// ---------------------------------------------------------------------------

/// Rust-side trampoline wrapping a foreign [`BlazenCustomProviderVTable`].
/// Implements [`InnerCustomProviderTrait`] by dispatching each method to the
/// vtable's function pointers on a `tokio::task::spawn_blocking` worker.
///
/// Owns the vtable's `user_data` — drops it via `drop_user_data` exactly
/// once when this adapter drops.
pub struct BlazenCabiCustomProviderAdapter {
    vtable: BlazenCustomProviderVTable,
    provider_id_cached: String,
    model_id_cached: String,
}

impl Drop for BlazenCabiCustomProviderAdapter {
    fn drop(&mut self) {
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

impl std::fmt::Debug for BlazenCabiCustomProviderAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlazenCabiCustomProviderAdapter")
            .field("provider_id", &self.provider_id_cached)
            .field("model_id", &self.model_id_cached)
            .finish_non_exhaustive()
    }
}

impl BlazenCabiCustomProviderAdapter {
    /// Reads `provider_id` / `model_id` out of the vtable's C-string fields
    /// and caches them as owned `String`s. Falls back to sensible defaults
    /// when the foreign side supplies null pointers (e.g. `model_id == NULL`
    /// reuses `provider_id`).
    fn new(vtable: BlazenCustomProviderVTable) -> Self {
        // SAFETY: per the vtable contract, `provider_id` is either null or a
        // valid NUL-terminated UTF-8 buffer.
        let provider_id_cached = unsafe { cstr_to_str(vtable.provider_id) }
            .unwrap_or("custom")
            .to_owned();
        // SAFETY: same contract for `model_id`.
        let model_id_cached = unsafe { cstr_to_str(vtable.model_id) }
            .map_or_else(|| provider_id_cached.clone(), str::to_owned);
        Self {
            vtable,
            provider_id_cached,
            model_id_cached,
        }
    }
}

// Helpers --------------------------------------------------------------------

/// Spawn-blocking a typed foreign call that consumes a request handle and
/// produces a typed result handle. Bridges the C ABI (status + out-ptrs)
/// into a Rust `Result<Inner, LlmError>`.
///
/// `req_addr` is the `usize`-encoded address of a `*mut Req` (a raw pointer
/// is `!Send`; passing the address as a primitive lets the closure cross
/// the spawn-blocking boundary cleanly). The foreign callback takes
/// ownership of the request and must free it.
async fn dispatch_typed_with_request<Req, ResHandle, Res>(
    user_data_addr: usize,
    func: extern "C" fn(*mut c_void, *mut Req, *mut *mut ResHandle, *mut *mut BlazenError) -> i32,
    req_addr: usize,
) -> Result<Res, LlmError>
where
    Req: 'static,
    ResHandle: 'static,
    Res: Send + 'static,
    Box<ResHandle>: TakeInner<Res>,
{
    // SAFETY: foreign side guarantees thread-safe access to `user_data`
    // (see the `BlazenCustomProviderVTable` docs). Function pointer is
    // `Copy + Send + Sync`. The request pointer was just minted from
    // `Box::into_raw` upstream so it's a unique allocation we're handing
    // off to the callback.
    let join =
        tokio::task::spawn_blocking(move || -> Result<Res, UniffiError> {
            let user_data = user_data_addr as *mut c_void;
            let req_ptr = req_addr as *mut Req;
            let mut out_handle: *mut ResHandle = std::ptr::null_mut();
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = func(user_data, req_ptr, &raw mut out_handle, &raw mut out_err);
            if status == 0 {
                if out_handle.is_null() {
                    return Err(UniffiError::Internal {
                        message: "CustomProvider vtable returned 0 without writing out_handle"
                            .into(),
                    });
                }
                // SAFETY: per the vtable contract, on success the foreign
                // callback wrote a fresh caller-owned `*mut ResHandle` produced
                // by one of the cabi result constructors. Ownership transfers
                // to us; reclaim via `Box::from_raw` and unwrap to the inner.
                let boxed = unsafe { Box::from_raw(out_handle) };
                Ok(<Box<ResHandle> as TakeInner<Res>>::take_inner(boxed))
            } else {
                if out_err.is_null() {
                    return Err(UniffiError::Internal {
                        message: format!(
                            "CustomProvider vtable returned -1 (status={status}) without writing out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract, on failure the foreign
                // callback wrote a fresh `*mut BlazenError`.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

    match join {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(uniffi_err_to_llm(e)),
        Err(join_err) => Err(LlmError::request(format!(
            "CustomProvider vtable task panicked: {join_err}"
        ))),
    }
}

/// Helper trait letting `dispatch_typed_with_request` unbox cabi result
/// handles into their inner `blazen_llm` core types without leaking the
/// `Box<BlazenXxx>(InnerXxx)` shape into the generic signature.
trait TakeInner<Out> {
    fn take_inner(self) -> Out;
}

impl TakeInner<InnerAudioResult> for Box<BlazenAudioResult> {
    fn take_inner(self) -> InnerAudioResult {
        self.0
    }
}
impl TakeInner<InnerImageResult> for Box<BlazenImageResult> {
    fn take_inner(self) -> InnerImageResult {
        self.0
    }
}
impl TakeInner<InnerVideoResult> for Box<BlazenVideoResult> {
    fn take_inner(self) -> InnerVideoResult {
        self.0
    }
}
impl TakeInner<InnerTranscriptionResult> for Box<BlazenTranscriptionResult> {
    fn take_inner(self) -> InnerTranscriptionResult {
        self.0
    }
}
impl TakeInner<InnerThreeDResult> for Box<BlazenThreeDResult> {
    fn take_inner(self) -> InnerThreeDResult {
        self.0
    }
}
impl TakeInner<InnerVoiceHandle> for Box<BlazenVoiceHandle> {
    fn take_inner(self) -> InnerVoiceHandle {
        self.0
    }
}

#[async_trait]
impl InnerCustomProviderTrait for BlazenCabiCustomProviderAdapter {
    fn provider_id(&self) -> &str {
        &self.provider_id_cached
    }

    fn model_id(&self) -> &str {
        &self.model_id_cached
    }

    async fn complete(
        &self,
        request: LlmCompletionRequest,
    ) -> Result<LlmCompletionResponse, LlmError> {
        // Convert from the blazen_llm typed request into the cabi wire shape
        // (which the foreign side already understands via `blazen_completion_request_*`).
        let uniffi_req: UniffiCompletionRequest = llm_to_uniffi_request(request);
        let req_handle = BlazenCompletionRequest::from(uniffi_req).into_ptr();

        let user_data_addr = self.vtable.user_data as usize;
        let complete_fn = self.vtable.complete;
        let req_addr = req_handle as usize;

        // SAFETY: foreign side guarantees thread-safe access; the request
        // pointer was just minted from `Box::into_raw`.
        let join = tokio::task::spawn_blocking(move || -> Result<LlmCompletionResponse, UniffiError> {
            let user_data = user_data_addr as *mut c_void;
            let req_ptr = req_addr as *mut BlazenCompletionRequest;
            let mut out_response: *mut BlazenCompletionResponse = std::ptr::null_mut();
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = complete_fn(user_data, req_ptr, &raw mut out_response, &raw mut out_err);
            if status == 0 {
                if out_response.is_null() {
                    return Err(UniffiError::Internal {
                        message: "CustomProvider complete returned 0 without writing out_response".into(),
                    });
                }
                // SAFETY: per the vtable contract, ownership transferred to us.
                let resp_box = unsafe { Box::from_raw(out_response) };
                Ok(uniffi_to_llm_response(resp_box.0))
            } else {
                if out_err.is_null() {
                    return Err(UniffiError::Internal {
                        message: format!("CustomProvider complete returned -1 (status={status}) without writing out_err"),
                    });
                }
                // SAFETY: per the vtable contract.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(uniffi_err_to_llm(e)),
            Err(join_err) => Err(LlmError::request(format!(
                "CustomProvider complete task panicked: {join_err}"
            ))),
        }
    }

    async fn stream(
        &self,
        request: LlmCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LlmStreamChunk, LlmError>> + Send>>, LlmError>
    {
        // Build cabi request handle.
        let uniffi_req: UniffiCompletionRequest = llm_to_uniffi_request(request);
        let req_handle = BlazenCompletionRequest::from(uniffi_req).into_ptr();

        // Build the pusher: the channel's RX side becomes the returned
        // Stream; the TX side is moved into the BlazenStreamPusher handle
        // we hand to the foreign callback.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let pusher_ptr = BlazenStreamPusher { tx }.into_ptr();

        let user_data_addr = self.vtable.user_data as usize;
        let stream_fn = self.vtable.stream;
        let req_addr = req_handle as usize;
        let pusher_addr = pusher_ptr as usize;

        // Drive the foreign callback synchronously on a blocking worker. The
        // foreign side is expected to either push chunks inline before
        // returning OR spawn its own producer that holds the pusher pointer
        // for the duration of the stream. Either way: `status` reports
        // whether the stream STARTED successfully.
        //
        // SAFETY: foreign side guarantees thread-safe access; the request
        // and pusher pointers were just minted from `Box::into_raw`.
        let start_status = tokio::task::spawn_blocking(move || -> Result<(), UniffiError> {
            let user_data = user_data_addr as *mut c_void;
            let req_ptr = req_addr as *mut BlazenCompletionRequest;
            let pusher = pusher_addr as *mut BlazenStreamPusher;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = stream_fn(user_data, req_ptr, pusher, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(UniffiError::Internal {
                        message: format!(
                            "CustomProvider stream returned -1 (status={status}) without writing out_err"
                        ),
                    });
                }
                // SAFETY: per the vtable contract.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match start_status {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(uniffi_err_to_llm(e)),
            Err(join_err) => {
                return Err(LlmError::request(format!(
                    "CustomProvider stream start task panicked: {join_err}"
                )));
            }
        }

        // Build the stream that pulls from the channel and converts each
        // item from the uniffi-error to the llm-error type. After a fatal
        // error the foreign side has already dropped the pusher (via
        // `blazen_stream_pusher_error`), so `recv` returns `None` on the
        // next iteration and the stream ends.
        let s = stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(Ok(chunk)) => Some((Ok(chunk), rx)),
                Some(Err(e)) => Some((Err(uniffi_err_to_llm(e)), rx)),
                None => None,
            }
        });
        Ok(Box::pin(s))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<LlmEmbeddingResponse, LlmError> {
        // Build the owning `CString` vector in the outer scope; move it
        // (and only `Send`-safe values) into the blocking closure. The
        // pointer array gets rebuilt inside the closure right before the
        // foreign call — raw pointers are `!Send`, so we never carry them
        // across the closure boundary.
        let owned: Vec<std::ffi::CString> = texts
            .into_iter()
            .map(|t| {
                std::ffi::CString::new(t).unwrap_or_else(|_| std::ffi::CString::new("").unwrap())
            })
            .collect();

        let user_data_addr = self.vtable.user_data as usize;
        let embed_fn = self.vtable.embed;

        // SAFETY: foreign side guarantees thread-safe access. The CString
        // backing store lives inside the closure (moved by value) so it
        // outlives the foreign call.
        let join = tokio::task::spawn_blocking(
            move || -> Result<LlmEmbeddingResponse, UniffiError> {
                let user_data = user_data_addr as *mut c_void;
                // Build the ptr array INSIDE the closure so we never need
                // to send raw pointers across the `Send` boundary.
                let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
                let count = ptrs.len();

                let mut out_response: *mut BlazenEmbeddingResponse = std::ptr::null_mut();
                let mut out_err: *mut BlazenError = std::ptr::null_mut();

                let status = embed_fn(
                    user_data,
                    ptrs.as_ptr(),
                    count,
                    &raw mut out_response,
                    &raw mut out_err,
                );

                // Keep `owned` alive across the call by referencing it after.
                let _keepalive = &owned;

                if status == 0 {
                    if out_response.is_null() {
                        return Err(UniffiError::Internal {
                            message: "CustomProvider embed returned 0 without writing out_response"
                                .into(),
                        });
                    }
                    // SAFETY: per the vtable contract.
                    let resp_box = unsafe { Box::from_raw(out_response) };
                    Ok(uniffi_to_llm_embedding_response(resp_box.0))
                } else {
                    if out_err.is_null() {
                        return Err(UniffiError::Internal {
                            message: format!(
                                "CustomProvider embed returned -1 (status={status}) without writing out_err"
                            ),
                        });
                    }
                    // SAFETY: per the vtable contract.
                    let be = unsafe { Box::from_raw(out_err) };
                    Err(be.inner)
                }
            },
        )
        .await;

        match join {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(uniffi_err_to_llm(e)),
            Err(join_err) => Err(LlmError::request(format!(
                "CustomProvider embed task panicked: {join_err}"
            ))),
        }
    }

    async fn text_to_speech(&self, req: InnerSpeechRequest) -> Result<InnerAudioResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.text_to_speech;
        let req_ptr = BlazenSpeechRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenSpeechRequest, BlazenAudioResult, InnerAudioResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn generate_music(&self, req: InnerMusicRequest) -> Result<InnerAudioResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.generate_music;
        let req_ptr = BlazenMusicRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenMusicRequest, BlazenAudioResult, InnerAudioResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn generate_sfx(&self, req: InnerMusicRequest) -> Result<InnerAudioResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.generate_sfx;
        let req_ptr = BlazenMusicRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenMusicRequest, BlazenAudioResult, InnerAudioResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn clone_voice(&self, req: InnerVoiceCloneRequest) -> Result<InnerVoiceHandle, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.clone_voice;
        let req_ptr = BlazenVoiceCloneRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenVoiceCloneRequest, BlazenVoiceHandle, InnerVoiceHandle>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn list_voices(&self) -> Result<Vec<InnerVoiceHandle>, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.list_voices;

        // SAFETY: foreign side guarantees thread-safe access to `user_data`.
        let join = tokio::task::spawn_blocking(move || -> Result<Vec<InnerVoiceHandle>, UniffiError> {
            let user_data = user_data_addr as *mut c_void;
            let mut out_array: *mut *mut BlazenVoiceHandle = std::ptr::null_mut();
            let mut out_count: usize = 0;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = f(user_data, &raw mut out_array, &raw mut out_count, &raw mut out_err);
            if status == 0 {
                if out_count == 0 {
                    // Empty list path: foreign side may legitimately return
                    // a null array with count 0. Anything else is treated as
                    // an empty list.
                    return Ok(Vec::new());
                }
                if out_array.is_null() {
                    return Err(UniffiError::Internal {
                        message: "CustomProvider list_voices returned non-zero count with null array".into(),
                    });
                }
                // Reclaim the outer pointer array as a Box<[*mut _]> so we
                // can free it correctly later. The foreign side allocated
                // the array with `blazen_voice_handle_list_alloc`, which is
                // the dual of `blazen_voice_handle_list_free`.
                let mut out: Vec<InnerVoiceHandle> = Vec::with_capacity(out_count);
                // SAFETY: caller wrote `out_count` element pointers into a
                // fresh array (as documented on the vtable). Walk the slice
                // and reclaim each handle.
                for i in 0..out_count {
                    // SAFETY: `i < out_count`, and the array spans `out_count`
                    // valid pointer slots per the vtable contract.
                    let h_ptr = unsafe { *out_array.add(i) };
                    if h_ptr.is_null() {
                        return Err(UniffiError::Internal {
                            message: "CustomProvider list_voices: null entry in voice array".into(),
                        });
                    }
                    // SAFETY: per the vtable contract, each entry came from
                    // `Box::into_raw`. Ownership transfers here.
                    let h_box = unsafe { Box::from_raw(h_ptr) };
                    out.push(h_box.0);
                }
                // Free the outer pointer array as a boxed slice. The foreign
                // side allocates it; the cabi reclaims it via the shared
                // `blazen_voice_handle_list_free` shape (boxed slice from
                // `Box::into_raw`).
                let slice_ptr: *mut [*mut BlazenVoiceHandle] =
                    std::ptr::slice_from_raw_parts_mut(out_array, out_count);
                // SAFETY: same boxed-slice round-trip as documented on
                // `blazen_voice_handle_list_free`; the foreign side must
                // allocate the array as a boxed slice for this to be sound.
                drop(unsafe { Box::from_raw(slice_ptr) });
                Ok(out)
            } else {
                if out_err.is_null() {
                    return Err(UniffiError::Internal {
                        message: format!(
                            "CustomProvider list_voices returned -1 (status={status}) without writing out_err"
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
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(uniffi_err_to_llm(e)),
            Err(join_err) => Err(LlmError::request(format!(
                "CustomProvider list_voices task panicked: {join_err}"
            ))),
        }
    }

    async fn delete_voice(&self, voice: InnerVoiceHandle) -> Result<(), LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.delete_voice;
        let v_ptr = BlazenVoiceHandle::from(voice).into_ptr();
        let v_addr = v_ptr as usize;

        // SAFETY: foreign side guarantees thread-safe access. The voice
        // pointer was just minted from `Box::into_raw`.
        let join = tokio::task::spawn_blocking(move || -> Result<(), UniffiError> {
            let user_data = user_data_addr as *mut c_void;
            let v_ptr = v_addr as *mut BlazenVoiceHandle;
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = f(user_data, v_ptr, &raw mut out_err);
            if status == 0 {
                Ok(())
            } else {
                if out_err.is_null() {
                    return Err(UniffiError::Internal {
                        message: format!(
                            "CustomProvider delete_voice returned -1 (status={status}) without writing out_err"
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
            Ok(Err(e)) => Err(uniffi_err_to_llm(e)),
            Err(join_err) => Err(LlmError::request(format!(
                "CustomProvider delete_voice task panicked: {join_err}"
            ))),
        }
    }

    async fn generate_image(&self, req: InnerImageRequest) -> Result<InnerImageResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.generate_image;
        let req_ptr = BlazenImageRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenImageRequest, BlazenImageResult, InnerImageResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn upscale_image(&self, req: InnerUpscaleRequest) -> Result<InnerImageResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.upscale_image;
        let req_ptr = BlazenUpscaleRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenUpscaleRequest, BlazenImageResult, InnerImageResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn text_to_video(&self, req: InnerVideoRequest) -> Result<InnerVideoResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.text_to_video;
        let req_ptr = BlazenVideoRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenVideoRequest, BlazenVideoResult, InnerVideoResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn image_to_video(&self, req: InnerVideoRequest) -> Result<InnerVideoResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.image_to_video;
        let req_ptr = BlazenVideoRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenVideoRequest, BlazenVideoResult, InnerVideoResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn transcribe(
        &self,
        req: InnerTranscriptionRequest,
    ) -> Result<InnerTranscriptionResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.transcribe;
        let req_ptr = BlazenTranscriptionRequest::from(req).into_ptr();
        dispatch_typed_with_request::<
            BlazenTranscriptionRequest,
            BlazenTranscriptionResult,
            InnerTranscriptionResult,
        >(user_data_addr, f, req_ptr as usize)
        .await
    }

    async fn generate_3d(&self, req: InnerThreeDRequest) -> Result<InnerThreeDResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.generate_3d;
        let req_ptr = BlazenThreeDRequest::from(req).into_ptr();
        dispatch_typed_with_request::<BlazenThreeDRequest, BlazenThreeDResult, InnerThreeDResult>(
            user_data_addr,
            f,
            req_ptr as usize,
        )
        .await
    }

    async fn remove_background(
        &self,
        req: InnerBackgroundRemovalRequest,
    ) -> Result<InnerImageResult, LlmError> {
        let user_data_addr = self.vtable.user_data as usize;
        let f = self.vtable.remove_background;
        let req_ptr = BlazenBackgroundRemovalRequest::from(req).into_ptr();
        dispatch_typed_with_request::<
            BlazenBackgroundRemovalRequest,
            BlazenImageResult,
            InnerImageResult,
        >(user_data_addr, f, req_ptr as usize)
        .await
    }
}

// ---------------------------------------------------------------------------
// blazen_llm <-> blazen_uniffi request/response/embedding conversions
// ---------------------------------------------------------------------------

/// Convert a `blazen_llm::types::CompletionRequest` (core trait input) into
/// the `blazen_uniffi::llm::CompletionRequest` shape consumed by the cabi
/// handle. Loss is bounded to fields that have no first-class slot in the
/// uniffi wire format (`stop`, `frequency_penalty`, `presence_penalty`,
/// `seed`, modality / image / audio configs, etc.) — the foreign side that
/// implements `complete` doesn't see them anyway; if it needs them it can
/// add them through the framework's existing extensibility.
fn llm_to_uniffi_request(req: LlmCompletionRequest) -> UniffiCompletionRequest {
    let response_format_json = req.response_format.map(|v| v.to_string());

    let messages = req
        .messages
        .into_iter()
        .map(llm_message_to_uniffi)
        .collect();

    let tools = req
        .tools
        .into_iter()
        .map(|t| blazen_uniffi::llm::Tool {
            name: t.name,
            description: t.description,
            parameters_json: t.parameters.to_string(),
        })
        .collect();

    UniffiCompletionRequest {
        messages,
        tools,
        temperature: req.temperature.map(f64::from),
        max_tokens: req.max_tokens,
        top_p: req.top_p.map(f64::from),
        model: req.model,
        response_format_json,
        system: None,
    }
}

fn llm_message_to_uniffi(msg: blazen_llm::types::ChatMessage) -> blazen_uniffi::llm::ChatMessage {
    use blazen_llm::types::{ContentPart, ImageSource, MessageContent, Role};

    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
    .to_owned();

    let (content, media_parts) = match msg.content {
        MessageContent::Text(s) => (s, Vec::new()),
        MessageContent::Image(img) => (String::new(), vec![image_content_to_media(img)]),
        MessageContent::Parts(parts) => {
            let mut text = String::new();
            let mut media = Vec::new();
            for part in parts {
                match part {
                    ContentPart::Text { text: t } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&t);
                    }
                    ContentPart::Image(img) => {
                        media.push(image_content_to_media(img));
                    }
                    ContentPart::Audio(a) => {
                        let m = blazen_uniffi::llm::Media {
                            kind: "audio".into(),
                            mime_type: a.media_type.unwrap_or_default(),
                            data_base64: match a.source {
                                ImageSource::Base64 { data } => data,
                                other => serde_json::to_string(&other).unwrap_or_default(),
                            },
                        };
                        media.push(m);
                    }
                    ContentPart::Video(v) => {
                        let m = blazen_uniffi::llm::Media {
                            kind: "video".into(),
                            mime_type: v.media_type.unwrap_or_default(),
                            data_base64: match v.source {
                                ImageSource::Base64 { data } => data,
                                other => serde_json::to_string(&other).unwrap_or_default(),
                            },
                        };
                        media.push(m);
                    }
                    ContentPart::File(_) => {}
                }
            }
            (text, media)
        }
    };

    let tool_calls = msg
        .tool_calls
        .into_iter()
        .map(|tc| blazen_uniffi::llm::ToolCall {
            id: tc.id,
            name: tc.name,
            arguments_json: tc.arguments.to_string(),
        })
        .collect();

    blazen_uniffi::llm::ChatMessage {
        role,
        content,
        media_parts,
        tool_calls,
        tool_call_id: msg.tool_call_id,
        name: msg.name,
    }
}

fn image_content_to_media(img: blazen_llm::types::ImageContent) -> blazen_uniffi::llm::Media {
    use blazen_llm::types::ImageSource;
    let data_base64 = match img.source {
        ImageSource::Base64 { data } => data,
        other => serde_json::to_string(&other).unwrap_or_default(),
    };
    blazen_uniffi::llm::Media {
        kind: "image".into(),
        mime_type: img.media_type.unwrap_or_default(),
        data_base64,
    }
}

/// Convert the uniffi-flavoured response back into the core llm shape.
fn uniffi_to_llm_response(resp: UniffiCompletionResponse) -> LlmCompletionResponse {
    let finish_reason = if resp.finish_reason.is_empty() {
        None
    } else {
        Some(resp.finish_reason)
    };
    let usage = if resp.usage.total_tokens == 0
        && resp.usage.prompt_tokens == 0
        && resp.usage.completion_tokens == 0
    {
        None
    } else {
        Some(blazen_llm::types::TokenUsage {
            prompt_tokens: u32::try_from(resp.usage.prompt_tokens).unwrap_or(u32::MAX),
            completion_tokens: u32::try_from(resp.usage.completion_tokens).unwrap_or(u32::MAX),
            total_tokens: u32::try_from(resp.usage.total_tokens).unwrap_or(u32::MAX),
            cached_input_tokens: u32::try_from(resp.usage.cached_input_tokens).unwrap_or(u32::MAX),
            reasoning_tokens: u32::try_from(resp.usage.reasoning_tokens).unwrap_or(u32::MAX),
            audio_input_tokens: 0,
            audio_output_tokens: 0,
        })
    };
    let tool_calls = resp
        .tool_calls
        .into_iter()
        .map(|tc| blazen_llm::types::ToolCall {
            id: tc.id,
            name: tc.name,
            arguments: serde_json::from_str(&tc.arguments_json).unwrap_or_default(),
        })
        .collect();

    LlmCompletionResponse {
        content: if resp.content.is_empty() {
            None
        } else {
            Some(resp.content)
        },
        tool_calls,
        reasoning: None,
        citations: Vec::new(),
        artifacts: Vec::new(),
        usage,
        model: resp.model,
        finish_reason,
        cost: None,
        timing: None,
        images: Vec::new(),
        audio: Vec::new(),
        videos: Vec::new(),
        metadata: serde_json::Value::Null,
    }
}

fn uniffi_to_llm_embedding_response(resp: UniffiEmbeddingResponse) -> LlmEmbeddingResponse {
    // Narrowing f64 -> f32 is intentional: blazen_llm's `EmbeddingResponse`
    // stores f32 (saving 50% memory and matching native model precision);
    // uniffi widens to f64 for cross-language uniformity. Round-trip loss
    // is bounded to the bottom 29 mantissa bits — irrelevant for cosine
    // similarity over typical embedding vectors.
    #[allow(clippy::cast_possible_truncation)]
    let embeddings: Vec<Vec<f32>> = resp
        .embeddings
        .into_iter()
        .map(|v| v.into_iter().map(|d| d as f32).collect())
        .collect();
    let usage = blazen_llm::types::TokenUsage {
        prompt_tokens: u32::try_from(resp.usage.prompt_tokens).unwrap_or(u32::MAX),
        completion_tokens: u32::try_from(resp.usage.completion_tokens).unwrap_or(u32::MAX),
        total_tokens: u32::try_from(resp.usage.total_tokens).unwrap_or(u32::MAX),
        cached_input_tokens: u32::try_from(resp.usage.cached_input_tokens).unwrap_or(u32::MAX),
        reasoning_tokens: u32::try_from(resp.usage.reasoning_tokens).unwrap_or(u32::MAX),
        audio_input_tokens: 0,
        audio_output_tokens: 0,
    };
    let usage =
        if usage.total_tokens == 0 && usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
            None
        } else {
            Some(usage)
        };
    LlmEmbeddingResponse {
        embeddings,
        model: resp.model,
        usage,
        cost: None,
        timing: None,
        metadata: serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/// Wraps a foreign-supplied [`BlazenCustomProviderVTable`] in an
/// [`InnerCustomProviderHandle`] and hands back a caller-owned handle.
///
/// The vtable's `user_data` ownership transfers into the wrapping
/// [`BlazenCabiCustomProviderAdapter`] for the lifetime of the handle.
/// `drop_user_data` is invoked exactly once when the handle is freed via
/// [`blazen_custom_provider_free`].
///
/// # Safety
///
/// The vtable's function pointers must satisfy the contracts documented on
/// [`BlazenCustomProviderVTable`]. `user_data` ownership transfers; foreign
/// callers must NOT call `drop_user_data` themselves after this returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_from_vtable(
    vtable: BlazenCustomProviderVTable,
) -> *mut BlazenCustomProvider {
    let adapter =
        Arc::new(BlazenCabiCustomProviderAdapter::new(vtable)) as Arc<dyn InnerCustomProviderTrait>;
    let handle = InnerCustomProviderHandle::new(adapter);
    BlazenCustomProvider::from(handle).into_ptr()
}

/// Constructs a `CustomProvider` speaking the `OpenAI` Chat Completions wire
/// protocol against an arbitrary `OpenAI`-compatible server.
///
/// Clones the supplied [`BlazenOpenAiCompatConfig`] into the new provider —
/// caller retains ownership of the source config handle. Returns null if
/// `provider_id` or `config` is null / non-UTF-8.
///
/// # Safety
///
/// `provider_id` must be a valid NUL-terminated UTF-8 buffer. `config` must
/// be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_openai_compat(
    provider_id: *const c_char,
    config: *const BlazenOpenAiCompatConfig,
) -> *mut BlazenCustomProvider {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(provider_id) = (unsafe { cstr_to_str(provider_id) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller guarantees live config handle.
    let cfg = unsafe { &*config }.0.clone();
    let handle = inner_custom::openai_compat(provider_id.to_owned(), cfg);
    BlazenCustomProvider::from(handle).into_ptr()
}

/// Convenience factory for an Ollama server.
///
/// Builds the canonical `http://<host>:<port>/v1` base URL and constructs an
/// OpenAI-protocol `CustomProviderHandle` with `provider_id = "ollama"`.
/// `host` being null is treated as `"localhost"`. `port` of `0` is treated
/// as the upstream default (`11434`).
///
/// # Safety
///
/// `host` must be null OR a valid NUL-terminated UTF-8 buffer. `model` must
/// be a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_ollama(
    model: *const c_char,
    host: *const c_char,
    port: u16,
) -> *mut BlazenCustomProvider {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `host`.
    let host_str = unsafe { cstr_to_str(host) }.unwrap_or("localhost");
    let resolved_port = if port == 0 { 11434 } else { port };
    let handle = inner_custom::ollama(host_str, resolved_port, model.to_owned());
    BlazenCustomProvider::from(handle).into_ptr()
}

/// Convenience factory for an LM Studio server.
///
/// Builds the canonical `http://<host>:<port>/v1` base URL and constructs an
/// OpenAI-protocol `CustomProviderHandle` with `provider_id = "lm_studio"`.
/// `host` being null is treated as `"localhost"`. `port` of `0` is treated
/// as the upstream default (`1234`).
///
/// # Safety
///
/// `host` must be null OR a valid NUL-terminated UTF-8 buffer. `model` must
/// be a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_lm_studio(
    model: *const c_char,
    host: *const c_char,
    port: u16,
) -> *mut BlazenCustomProvider {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `host`.
    let host_str = unsafe { cstr_to_str(host) }.unwrap_or("localhost");
    let resolved_port = if port == 0 { 1234 } else { port };
    let handle = inner_custom::lm_studio(host_str, resolved_port, model.to_owned());
    BlazenCustomProvider::from(handle).into_ptr()
}

// ---------------------------------------------------------------------------
// Cross-handle bridges
// ---------------------------------------------------------------------------

/// Returns the inner provider's `provider_id` as a caller-owned C string.
/// Free with [`crate::string::blazen_string_free`]. Returns null on null
/// `p`.
///
/// # Safety
///
/// `p` must be null OR a live `BlazenCustomProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_provider_id(
    p: *const BlazenCustomProvider,
) -> *mut c_char {
    if p.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let h = unsafe { &*p };
    alloc_cstring(h.0.provider_id_str())
}

/// Returns the inner provider's `model_id` as a caller-owned C string. Free
/// with [`crate::string::blazen_string_free`]. Returns null on null `p`.
///
/// # Safety
///
/// `p` must be null OR a live `BlazenCustomProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_model_id(
    p: *const BlazenCustomProvider,
) -> *mut c_char {
    if p.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let h = unsafe { &*p };
    alloc_cstring(<InnerCustomProviderHandle as CompletionModel>::model_id(
        &h.0,
    ))
}

/// Wraps the underlying [`InnerCustomProviderHandle`] inside a brand-new
/// [`InnerBaseProvider`] and returns it as a caller-owned
/// `BlazenBaseProvider` handle.
///
/// This lets the host attach base-provider builder mutations (system
/// prompts, default tools, defaults bundles) without reaching into the
/// private `base` field on `CustomProviderHandle`. The returned
/// `BaseProvider` holds an `Arc<CustomProviderHandle>` as its inner
/// `CompletionModel`, so `complete()` / `stream()` on the result delegate
/// back through the `CustomProviderHandle` (including any vtable dispatch).
///
/// Returns null if `p` is null.
///
/// # Safety
///
/// `p` must be null OR a live `BlazenCustomProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_as_base_provider(
    p: *const BlazenCustomProvider,
) -> *mut BlazenBaseProvider {
    if p.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let inner: Arc<dyn CompletionModel> = unsafe { (*p).0.clone() };
    let bp = InnerBaseProvider::new(inner);
    BlazenBaseProvider::from(bp).into_ptr()
}

/// Frees a `BlazenCustomProvider`. No-op on null.
///
/// # Safety
///
/// `p` must be null OR a pointer produced by one of the
/// `blazen_custom_provider_*` factories. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_free(p: *mut BlazenCustomProvider) {
    if p.is_null() {
        return;
    }
    // SAFETY: per the contract above, `p` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(p) });
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Clones the inner `Arc<CustomProviderHandle>` out of a live handle. The
/// caller is responsible for confirming `p.is_null()` first.
///
/// # Safety
///
/// `p` must be a live, non-null pointer to a `BlazenCustomProvider`.
#[inline]
unsafe fn clone_provider(p: *const BlazenCustomProvider) -> Arc<InnerCustomProviderHandle> {
    // SAFETY: caller guarantees live handle.
    Arc::clone(unsafe { &(*p).0 })
}

// ---------------------------------------------------------------------------
// 13 compute methods + completion/embedding C entry points
// ---------------------------------------------------------------------------

/// Spawns an async `text_to_speech` against the host's `CustomProvider`.
///
/// Consumes the request handle (the inner [`InnerSpeechRequest`] is moved
/// into the async task). On completion the typed [`InnerAudioResult`] is
/// popped with [`blazen_future_take_custom_audio_result`].
///
/// Returns null if either argument is null.
///
/// # Safety
///
/// `p` must be null OR a live `BlazenCustomProvider`. `request` must be null
/// OR a live `BlazenSpeechRequest`. Ownership of `request` transfers to the
/// spawned task — do NOT call `blazen_speech_request_free` afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_text_to_speech(
    p: *const BlazenCustomProvider,
    request: *mut BlazenSpeechRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerAudioResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::text_to_speech(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `generate_music`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenMusicRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_generate_music(
    p: *const BlazenCustomProvider,
    request: *mut BlazenMusicRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerAudioResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::generate_music(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `generate_sfx`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenMusicRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_generate_sfx(
    p: *const BlazenCustomProvider,
    request: *mut BlazenMusicRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerAudioResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::generate_sfx(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `clone_voice`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenVoiceCloneRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_clone_voice(
    p: *const BlazenCustomProvider,
    request: *mut BlazenVoiceCloneRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerVoiceHandle, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::clone_voice(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `list_voices`. Takes no request handle.
///
/// On completion, pop the resulting `Vec<VoiceHandle>` via
/// [`blazen_future_take_custom_voice_list`].
///
/// Returns null if `p` is null.
///
/// # Safety
///
/// `p` must be null OR a live `BlazenCustomProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_list_voices(
    p: *const BlazenCustomProvider,
) -> *mut BlazenFuture {
    if p.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    BlazenFuture::spawn::<Vec<InnerVoiceHandle>, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::list_voices(&provider)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `delete_voice`. Consumes `voice` — ownership of the
/// handle transfers into the spawned task. After this call, the caller must
/// NOT call `blazen_voice_handle_free`.
///
/// Returns a future whose typed result is unit; observers should drain via
/// [`blazen_future_take_custom_unit`].
///
/// Returns null if `p` or `voice` is null.
///
/// # Safety
///
/// `p` must be null OR a live `BlazenCustomProvider`. `voice` must be null
/// OR a live `BlazenVoiceHandle`; ownership transfers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_delete_voice(
    p: *const BlazenCustomProvider,
    voice: *mut BlazenVoiceHandle,
) -> *mut BlazenFuture {
    if p.is_null() || voice.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `voice` came from `Box::into_raw`.
    let v = unsafe { Box::from_raw(voice) }.0;
    BlazenFuture::spawn::<(), _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::delete_voice(&provider, v)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `generate_image`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_generate_image(
    p: *const BlazenCustomProvider,
    request: *mut BlazenImageRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerImageResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::generate_image(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `upscale_image`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenUpscaleRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_upscale_image(
    p: *const BlazenCustomProvider,
    request: *mut BlazenUpscaleRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerImageResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::upscale_image(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `text_to_video`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_text_to_video(
    p: *const BlazenCustomProvider,
    request: *mut BlazenVideoRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerVideoResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::text_to_video(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `image_to_video`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_image_to_video(
    p: *const BlazenCustomProvider,
    request: *mut BlazenVideoRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerVideoResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::image_to_video(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `transcribe`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenTranscriptionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_transcribe(
    p: *const BlazenCustomProvider,
    request: *mut BlazenTranscriptionRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerTranscriptionResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::transcribe(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `generate_3d`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenThreeDRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_generate_3d(
    p: *const BlazenCustomProvider,
    request: *mut BlazenThreeDRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerThreeDResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::generate_3d(&provider, req)
            .await
            .map_err(Into::into)
    })
}

/// Spawns an async `remove_background`. See
/// [`blazen_custom_provider_text_to_speech`] for ownership semantics.
///
/// # Safety
///
/// Same as [`blazen_custom_provider_text_to_speech`] but `request` must be
/// a `BlazenBackgroundRemovalRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_custom_provider_remove_background(
    p: *const BlazenCustomProvider,
    request: *mut BlazenBackgroundRemovalRequest,
) -> *mut BlazenFuture {
    if p.is_null() || request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { clone_provider(p) };
    // SAFETY: per the contract, `request` came from `Box::into_raw`.
    let req = unsafe { Box::from_raw(request) }.0;
    BlazenFuture::spawn::<InnerImageResult, _>(async move {
        <InnerCustomProviderHandle as InnerCustomProviderTrait>::remove_background(&provider, req)
            .await
            .map_err(Into::into)
    })
}

// ---------------------------------------------------------------------------
// Typed future-take entry points
// ---------------------------------------------------------------------------

/// Pops a typed [`InnerAudioResult`] (the `blazen_llm::compute::AudioResult`
/// returned by `text_to_speech`, `generate_music`, `generate_sfx`) out of
/// `fut`.
///
/// On success returns `0` and writes a caller-owned `*mut BlazenAudioResult`
/// into `out`; on failure returns `-1` and writes a caller-owned
/// `*mut BlazenError` into `err`. Either out-pointer may be null.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by one of the audio-result
/// `blazen_custom_provider_*` entry points, not yet freed. `out` / `err`
/// must be null OR writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_audio_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenAudioResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerAudioResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenAudioResult::from(v).into_ptr();
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

/// Pops a typed [`InnerImageResult`] (`generate_image`, `upscale_image`,
/// `remove_background`) out of `fut`. Mirrors
/// [`blazen_future_take_custom_audio_result`] semantics.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by one of the image-result
/// `blazen_custom_provider_*` entry points. `out` / `err` must be null OR
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_image_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenImageResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerImageResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenImageResult::from(v).into_ptr();
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

/// Pops a typed [`InnerVideoResult`] (`text_to_video`, `image_to_video`) out
/// of `fut`. Mirrors [`blazen_future_take_custom_audio_result`] semantics.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by one of the video-result
/// `blazen_custom_provider_*` entry points. `out` / `err` must be null OR
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_video_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenVideoResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerVideoResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenVideoResult::from(v).into_ptr();
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

/// Pops a typed [`InnerTranscriptionResult`] (`transcribe`) out of `fut`.
/// Mirrors [`blazen_future_take_custom_audio_result`] semantics.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_custom_provider_transcribe`]. `out` / `err` must be null OR
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_transcription_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenTranscriptionResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerTranscriptionResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenTranscriptionResult::from(v).into_ptr();
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

/// Pops a typed [`InnerThreeDResult`] (`generate_3d`) out of `fut`. Mirrors
/// [`blazen_future_take_custom_audio_result`] semantics.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_custom_provider_generate_3d`]. `out` / `err` must be null OR
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_three_d_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenThreeDResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerThreeDResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenThreeDResult::from(v).into_ptr();
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

/// Pops a typed [`InnerVoiceHandle`] (`clone_voice`) out of `fut`. Mirrors
/// [`blazen_future_take_custom_audio_result`] semantics.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_custom_provider_clone_voice`]. `out` / `err` must be null OR
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_voice_handle(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenVoiceHandle,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerVoiceHandle>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenVoiceHandle::from(v).into_ptr();
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

/// Pops the typed `Vec<InnerVoiceHandle>` result of `list_voices` out of
/// `fut`. On success returns `0`, writes the count into `*out_count`, and
/// writes a caller-owned `*mut *mut BlazenVoiceHandle` array into
/// `*out_array`.
///
/// On failure returns `-1` and writes a caller-owned `*mut BlazenError`
/// into `err`. Any of `out_array` / `out_count` / `err` may be null.
///
/// ## Releasing the returned array
///
/// Free each `BlazenVoiceHandle` element with
/// [`crate::compute_results::blazen_voice_handle_free`], then release the
/// outer pointer array with [`blazen_voice_handle_list_free`].
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_custom_provider_list_voices`]. `out_array` / `out_count` / `err`
/// must each be null OR writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_voice_list(
    fut: *mut BlazenFuture,
    out_array: *mut *mut *mut BlazenVoiceHandle,
    out_count: *mut usize,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<Vec<InnerVoiceHandle>>(fut) } {
        Ok(v) => {
            let count = v.len();
            // Convert each `VoiceHandle` into a caller-owned
            // `*mut BlazenVoiceHandle`, then leak the boxed slice so the
            // caller owns the array.
            let boxed_slice: Box<[*mut BlazenVoiceHandle]> = v
                .into_iter()
                .map(|h| BlazenVoiceHandle::from(h).into_ptr())
                .collect();
            let raw_slice = Box::into_raw(boxed_slice);
            let ptr: *mut *mut BlazenVoiceHandle = raw_slice.cast::<*mut BlazenVoiceHandle>();
            if out_array.is_null() {
                // Caller doesn't care about the array — leak it to avoid
                // freeing the inner handles. Documented contract for
                // "discard the result" callers.
                let _ = ptr;
            } else {
                // SAFETY: caller has guaranteed `out_array` is writable.
                unsafe {
                    *out_array = ptr;
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

/// Frees the array of `*mut BlazenVoiceHandle` returned by
/// [`blazen_future_take_custom_voice_list`]. Does NOT free the individual
/// `BlazenVoiceHandle` elements — call
/// [`crate::compute_results::blazen_voice_handle_free`] for each pointer
/// first. No-op on null.
///
/// # Safety
///
/// `array` must be null OR a pointer-array produced by
/// [`blazen_future_take_custom_voice_list`]. `count` must be the exact
/// length reported by that call. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_list_free(
    array: *mut *mut BlazenVoiceHandle,
    count: usize,
) {
    if array.is_null() {
        return;
    }
    // SAFETY: per the contract, `array` was leaked from a boxed slice with
    // length `count`. Reconstructing the `Box<[_]>` here is sound and
    // `drop` releases the outer allocation. The element pointers remain
    // owned by the caller.
    let slice_ptr: *mut [*mut BlazenVoiceHandle] = std::ptr::slice_from_raw_parts_mut(array, count);
    // SAFETY: see comment above.
    drop(unsafe { Box::from_raw(slice_ptr) });
}

/// Pops the unit `()` result of `delete_voice` out of `fut`. On success
/// returns `0`; on failure returns `-1` and writes a caller-owned
/// `*mut BlazenError` into `err`. `err` may be null.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_custom_provider_delete_voice`]. `err` must be null OR a
/// writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_custom_unit(
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
