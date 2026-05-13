//! LLM opaque objects: `CompletionModel` and `EmbeddingModel`. Phase R3 Agent B.
//!
//! Both types wrap an `Arc<blazen_uniffi::llm::{CompletionModel,EmbeddingModel}>`
//! so multiple cabi handles can share the same underlying provider client
//! (per-provider factories in Phase R4 construct one inner model and
//! optionally hand back multiple wrappers).
//!
//! # Ownership conventions
//!
//! - **Model handles** (`*mut BlazenCompletionModel` / `*mut BlazenEmbeddingModel`):
//!   produced by provider factories in Phase R4, owned by the C caller, freed
//!   with the matching `*_free` function.
//! - **Request consumption** (`BlazenCompletionRequest`): the
//!   `complete_blocking` / `complete` entry points CONSUME the request
//!   pointer. Internally we `Box::from_raw` it, move the inner record out, and
//!   drop the (now-empty) wrapper. Callers must NOT call
//!   `blazen_completion_request_free` on a pointer that was passed to one of
//!   these two functions — that's a double-free.
//! - **String-array inputs** (`embed_blocking` / `embed`): the
//!   `*const *const c_char` argument is BORROWED. For `embed_blocking` we
//!   copy each string out before the blocking call returns; for the future
//!   variant we also copy each string out before the spawned task starts, so
//!   the caller is only contractually bound to keep the array alive for the
//!   duration of the cabi call itself (the spawned task owns its own
//!   `Vec<String>` copy).
//! - **Response handles**: the typed-take functions
//!   `blazen_future_take_completion_response` and
//!   `blazen_future_take_embedding_response` produce
//!   `*mut BlazenCompletionResponse` / `*mut BlazenEmbeddingResponse` that the
//!   caller owns and must free with the matching `*_free`.
//! - **`*mut c_char` getters** (`model_id`): allocated via `alloc_cstring`,
//!   freed with `blazen_string_free`.

// `BlazenCompletionModel::into_ptr` / `BlazenEmbeddingModel::into_ptr` are the
// foundation helpers Phase R4 provider factories will reach for. The extern
// FFI functions are linker-preserved regardless, but the helpers fire
// dead-code until the factories wire in.
#![allow(dead_code)]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::llm::{
    CompletionModel as InnerCompletionModel, CompletionResponse as InnerCompletionResponse,
    EmbeddingModel as InnerEmbeddingModel, EmbeddingResponse as InnerEmbeddingResponse,
};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm_records::{
    BlazenCompletionRequest, BlazenCompletionResponse, BlazenEmbeddingResponse,
};
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// Local error/status helpers
//
// Mirror the pattern in `compute.rs` / `persist.rs` / `agent.rs` rather than
// importing a shared helper: the cabi keeps each module self-contained so a
// future split (e.g. moving llm into its own cdylib) doesn't drag in unrelated
// surface area. Kept module-private.
// ---------------------------------------------------------------------------

/// Writes a `BlazenError` into `out_err` if the pointer is non-null. The
/// caller takes ownership of the produced pointer.
fn write_error(out_err: *mut *mut BlazenError, err: InnerError) {
    if !out_err.is_null() {
        // SAFETY: `out_err` is non-null per the branch above. The caller has
        // guaranteed it points to a writable `*mut BlazenError` slot.
        unsafe {
            *out_err = BlazenError::from(err).into_ptr();
        }
    }
}

/// Builds + writes a `BlazenError::Internal { message }` into `out_err`.
/// Used for argument-shape errors (null pointers where non-null is required,
/// non-UTF-8 strings) that don't originate from a `blazen_uniffi` call.
fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    );
}

/// Borrows a C array of C strings and clones each entry into an owned
/// `Vec<String>`. Returns `None` if any element is null or non-UTF-8.
///
/// An empty array (`count == 0`) is valid and returns `Some(vec![])`; the
/// `ptrs` pointer is only required to be non-null when `count > 0`.
///
/// # Safety
///
/// When `count > 0`, `ptrs` must point to an array of exactly `count` valid
/// `*const c_char` entries, each of which is a NUL-terminated UTF-8 buffer
/// valid for the duration of this call.
unsafe fn ptr_array_to_strings(ptrs: *const *const c_char, count: usize) -> Option<Vec<String>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if ptrs.is_null() {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: caller guarantees `ptrs` indexes `count` valid pointers.
        let p = unsafe { *ptrs.add(i) };
        // SAFETY: caller guarantees each indexed pointer is a NUL-terminated
        // UTF-8 buffer valid for the duration of this call.
        let s = unsafe { cstr_to_str(p) }?;
        out.push(s.to_owned());
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// BlazenCompletionModel
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::CompletionModel`]. Construct
/// via the per-provider factories in Phase R4 (e.g.
/// `blazen_completion_model_openai`).
pub struct BlazenCompletionModel(pub(crate) Arc<InnerCompletionModel>);

impl BlazenCompletionModel {
    /// Leaks a fresh handle for the C caller. Used by Phase R4 provider
    /// factories.
    pub(crate) fn into_ptr(self) -> *mut BlazenCompletionModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerCompletionModel>> for BlazenCompletionModel {
    fn from(inner: Arc<InnerCompletionModel>) -> Self {
        Self(inner)
    }
}

/// Returns the model's identifier (e.g. `"gpt-4o"`) as a caller-owned C
/// string. Returns null on a null handle. Caller frees with
/// `blazen_string_free`.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCompletionModel` produced by the
/// cabi surface (and not yet freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_model_id(
    model: *const BlazenCompletionModel,
) -> *mut c_char {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let m = unsafe { &*model };
    let id = Arc::clone(&m.0).model_id();
    alloc_cstring(&id)
}

/// Synchronously runs a chat completion on the cabi tokio runtime.
///
/// On success returns `0` and writes a fresh `BlazenCompletionResponse*` into
/// `*out_response`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Either out-parameter may be null to discard that side of
/// the result (typically only meaningful on the error path during a smoke
/// test).
///
/// **The `request` pointer is consumed.** Internally we `Box::from_raw` it
/// and move its inner record out. Calling
/// [`blazen_completion_request_free`](crate::llm_records::blazen_completion_request_free)
/// on the same pointer afterwards is a double-free.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCompletionModel`. `request` must be
/// null OR a live `BlazenCompletionRequest` produced by the cabi surface;
/// ownership transfers to this function. `out_response` and `out_err` must
/// each be null OR point to a writable slot of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_complete_blocking(
    model: *const BlazenCompletionModel,
    request: *mut BlazenCompletionRequest,
    out_response: *mut *mut BlazenCompletionResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(
            out_err,
            "blazen_completion_model_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_completion_model_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let m = unsafe { &*model };
    let inner_model = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result: Result<InnerCompletionResponse, InnerError> =
        runtime().block_on(async move { inner_model.complete(inner_request).await });

    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenCompletionResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Spawns a chat completion onto the cabi tokio runtime and returns an
/// opaque future handle. The caller observes completion via the future's
/// fd / `blazen_future_poll` / `blazen_future_wait`, then calls
/// [`blazen_future_take_completion_response`] to pop the result. Free the
/// future with `blazen_future_free`.
///
/// Returns null if either `model` or `request` is null (in which case the
/// `request`, if non-null, is still consumed and freed to avoid a leak).
///
/// **The `request` pointer is consumed.** See
/// [`blazen_completion_model_complete_blocking`] for details.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCompletionModel`. `request` must be
/// null OR a live `BlazenCompletionRequest`; ownership transfers to this
/// function regardless of whether the call returns null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_complete(
    model: *const BlazenCompletionModel,
    request: *mut BlazenCompletionRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let m = unsafe { &*model };
    let inner_model = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn(async move { inner_model.complete(inner_request).await })
}

/// Frees a `BlazenCompletionModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by a provider
/// factory in the cabi surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_free(model: *mut BlazenCompletionModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: per the contract, `model` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// BlazenEmbeddingModel
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::EmbeddingModel`]. Construct
/// via the per-provider factories in Phase R4.
pub struct BlazenEmbeddingModel(pub(crate) Arc<InnerEmbeddingModel>);

impl BlazenEmbeddingModel {
    /// Leaks a fresh handle for the C caller. Used by Phase R4 provider
    /// factories.
    pub(crate) fn into_ptr(self) -> *mut BlazenEmbeddingModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerEmbeddingModel>> for BlazenEmbeddingModel {
    fn from(inner: Arc<InnerEmbeddingModel>) -> Self {
        Self(inner)
    }
}

/// Returns the embedding model's identifier as a caller-owned C string.
/// Returns null on a null handle.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenEmbeddingModel`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_model_id(
    model: *const BlazenEmbeddingModel,
) -> *mut c_char {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenEmbeddingModel`.
    let m = unsafe { &*model };
    let id = Arc::clone(&m.0).model_id();
    alloc_cstring(&id)
}

/// Returns the embedding vector dimensionality. Returns `0` on a null handle
/// (which is otherwise an invalid input — callers should null-check before
/// calling).
///
/// # Safety
///
/// `model` must be null OR a live `BlazenEmbeddingModel`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_dimensions(
    model: *const BlazenEmbeddingModel,
) -> u32 {
    if model.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenEmbeddingModel`.
    let m = unsafe { &*model };
    Arc::clone(&m.0).dimensions()
}

/// Synchronously embeds one or more input strings on the cabi tokio runtime.
///
/// On success returns `0` and writes a fresh `BlazenEmbeddingResponse*` into
/// `*out_response`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`.
///
/// The `inputs` array is BORROWED — the cabi copies each string before the
/// blocking call returns. The caller retains ownership of the array and its
/// strings.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenEmbeddingModel`. When
/// `inputs_count > 0`, `inputs` must point to an array of exactly
/// `inputs_count` valid `*const c_char` entries, each a NUL-terminated UTF-8
/// buffer valid for the duration of this call. `out_response` and `out_err`
/// must each be null OR point to a writable slot of the matching pointer
/// type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_embed_blocking(
    model: *const BlazenEmbeddingModel,
    inputs: *const *const c_char,
    inputs_count: usize,
    out_response: *mut *mut BlazenEmbeddingResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_embedding_model_embed_blocking: null model");
        return -1;
    }
    // SAFETY: caller has guaranteed `inputs` indexes `inputs_count` valid
    // NUL-terminated UTF-8 buffers (or `inputs_count == 0`).
    let Some(owned_inputs) = (unsafe { ptr_array_to_strings(inputs, inputs_count) }) else {
        write_internal_error(
            out_err,
            "blazen_embedding_model_embed_blocking: null or non-UTF-8 input string",
        );
        return -1;
    };
    // SAFETY: caller has guaranteed `model` is a live `BlazenEmbeddingModel`.
    let m = unsafe { &*model };
    let inner_model = Arc::clone(&m.0);

    let result: Result<InnerEmbeddingResponse, InnerError> =
        runtime().block_on(async move { inner_model.embed(owned_inputs).await });

    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenEmbeddingResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Spawns an embed task onto the cabi tokio runtime and returns an opaque
/// future handle. Pop the result with
/// [`blazen_future_take_embedding_response`].
///
/// Returns null if `model` is null, if `inputs` is null and `inputs_count >
/// 0`, or if any indexed string is null / non-UTF-8. The `inputs` array is
/// copied out before the future is spawned, so the caller only has to keep
/// the array alive for the duration of this cabi call.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenEmbeddingModel`. When
/// `inputs_count > 0`, `inputs` must point to an array of exactly
/// `inputs_count` valid `*const c_char` entries, each a NUL-terminated UTF-8
/// buffer valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_embed(
    model: *const BlazenEmbeddingModel,
    inputs: *const *const c_char,
    inputs_count: usize,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `inputs` indexes `inputs_count` valid
    // NUL-terminated UTF-8 buffers (or `inputs_count == 0`).
    let Some(owned_inputs) = (unsafe { ptr_array_to_strings(inputs, inputs_count) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller has guaranteed `model` is a live `BlazenEmbeddingModel`.
    let m = unsafe { &*model };
    let inner_model = Arc::clone(&m.0);

    BlazenFuture::spawn(async move { inner_model.embed(owned_inputs).await })
}

/// Frees a `BlazenEmbeddingModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by a provider
/// factory in the cabi surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_free(model: *mut BlazenEmbeddingModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: per the contract, `model` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// Typed future-take entry points
//
// One per typed result produced by the `complete` / `embed` future variants.
// C can't see Rust generics, so we monomorphise here.
// ---------------------------------------------------------------------------

/// Pops the [`BlazenCompletionResponse`] out of a future produced by
/// [`blazen_completion_model_complete`].
///
/// Returns `0` on success (writes the response into `*out` when non-null) or
/// `-1` on failure (writes a fresh `BlazenError*` into `*err` when non-null).
/// Both out-parameters may be null to discard the corresponding side.
///
/// Calling this against a future produced by any other cabi entry point
/// (e.g. `blazen_embedding_model_embed`) yields a `BlazenError::Internal`
/// with a `type mismatch` message — see `BlazenFuture::take_typed`.
///
/// # Safety
///
/// `fut` must be null OR a pointer previously produced by
/// [`blazen_completion_model_complete`] (and not yet freed, not concurrently
/// freed from another thread). `out` and `err` must each be null OR point to
/// a writable slot of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_completion_response(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenCompletionResponse,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the contract on `fut` (live or null `BlazenFuture`
    // pointer produced by the cabi surface).
    match unsafe { BlazenFuture::take_typed::<InnerCompletionResponse>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: `out` is non-null per the branch above.
                unsafe {
                    *out = BlazenCompletionResponse::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(err, e);
            -1
        }
    }
}

/// Pops the [`BlazenEmbeddingResponse`] out of a future produced by
/// [`blazen_embedding_model_embed`].
///
/// Returns `0` on success (writes the response into `*out` when non-null) or
/// `-1` on failure (writes a fresh `BlazenError*` into `*err` when non-null).
///
/// # Safety
///
/// `fut` must be null OR a pointer previously produced by
/// [`blazen_embedding_model_embed`] (and not yet freed). `out` and `err`
/// must each be null OR point to a writable slot of the matching pointer
/// type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_embedding_response(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenEmbeddingResponse,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the contract on `fut`.
    match unsafe { BlazenFuture::take_typed::<InnerEmbeddingResponse>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: `out` is non-null per the branch above.
                unsafe {
                    *out = BlazenEmbeddingResponse::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(err, e);
            -1
        }
    }
}
