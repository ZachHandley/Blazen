//! Per-engine embedding provider opaques + C ABI factories.
//!
//! Each engine wraps an `Arc<blazen_uniffi::concrete::embed::<Engine>Provider>`
//! in a `Blazen<Engine>Provider` opaque, exposing:
//!
//! - `blazen_<engine>_provider_new` -- factory
//! - `blazen_<engine>_provider_embed` -- async future returning a
//!   `*mut BlazenFuture` whose typed result is popped with
//!   [`blazen_future_take_embedding_vectors`] (declared in this module)
//! - `blazen_<engine>_provider_embed_blocking` -- synchronous variant
//! - `blazen_<engine>_provider_dimensions` -- sync u32 accessor
//! - `blazen_<engine>_provider_free` -- destructor (no-op on null)
//!
//! ## Result type
//!
//! The polymorphic capability trait
//! [`blazen_uniffi::concrete::bases::EmbeddingProvider::embed`] returns
//! `Vec<Vec<f32>>` -- a flat, model-agnostic shape that mirrors what the
//! foreign bindings consume. The existing
//! [`crate::llm_records::BlazenEmbeddingResponse`] / `blazen_future_take_embedding_response`
//! pair carries the richer `blazen_uniffi::llm::EmbeddingResponse` struct
//! (`Vec<Vec<f64>>` + model id + usage) used by the central
//! `BlazenEmbeddingModel`, so it is not reusable here. This module
//! introduces a sibling [`BlazenEmbeddingVectors`] opaque + matching
//! accessors that owns the flat `Vec<Vec<f32>>` produced by per-engine
//! providers.
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<Blazen<Engine>Provider>`
//!   returned by the factory functions. Callers free with the matching
//!   `*_free`. Double-free is undefined behavior.
//! - String inputs are borrowed for the duration of the call only -- the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.
//! - The `texts` array is consumed by-value into a heap `Vec<String>`
//!   before the call returns; callers may free the source array
//!   immediately afterward.
//!
//! ## Relationship to the central [`crate::compute::BlazenEmbeddingModel`]
//!
//! The central `BlazenEmbeddingModel` + `blazen_embedding_model_*` family
//! in [`crate::llm`] / [`crate::compute`] remain in place -- this module
//! is purely additive. Foreign hosts (Ruby, future Dart / Crystal / Lua /
//! PHP) can migrate to the per-engine surface incrementally without
//! breaking the existing entry points.

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// Writes a caller-owned `BlazenError` into `out_err` (if non-null) and
/// returns `-1` for use in tail position.
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

/// Copy a `*const *const c_char` + len array of NUL-terminated UTF-8
/// strings into an owned `Vec<String>`. Returns `None` if `ptr` is null
/// (and `len > 0`), or any element is null / non-UTF-8.
///
/// # Safety
///
/// - When `len > 0`, `ptr` must point to `len` valid `*const c_char`
///   entries.
/// - Each non-null element must be a NUL-terminated UTF-8 buffer.
unsafe fn collect_strings(ptr: *const *const c_char, len: usize) -> Option<Vec<String>> {
    if len == 0 {
        return Some(Vec::new());
    }
    if ptr.is_null() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // SAFETY: caller has guaranteed `ptr` points to `len` entries.
        let entry = unsafe { *ptr.add(i) };
        // SAFETY: caller upholds NUL-termination + UTF-8 on each non-null entry.
        let s = unsafe { cstr_to_str(entry) }?;
        out.push(s.to_owned());
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// BlazenEmbeddingVectors -- flat Vec<Vec<f32>> opaque + accessors
// ---------------------------------------------------------------------------

/// Opaque wrapper around the flat `Vec<Vec<f32>>` produced by per-engine
/// embedding providers in this module. Distinct from
/// [`crate::llm_records::BlazenEmbeddingResponse`] (which carries the
/// richer model-aware `EmbeddingResponse` from `blazen_uniffi::llm`).
pub struct BlazenEmbeddingVectors(pub(crate) Vec<Vec<f32>>);

impl BlazenEmbeddingVectors {
    pub(crate) fn into_ptr(self) -> *mut BlazenEmbeddingVectors {
        Box::into_raw(Box::new(self))
    }
}

/// Pops the [`BlazenEmbeddingVectors`] out of a future produced by any
/// `blazen_<engine>_provider_embed` in this module.
///
/// Returns `0` on success (writes the handle into `*out` when non-null)
/// or `-1` on failure (writes a fresh `BlazenError*` into `*err` when
/// non-null).
///
/// # Safety
///
/// `fut` must be null OR a pointer previously produced by a per-engine
/// `_embed` function (and not yet freed). `out` and `err` must each be
/// null OR point to a writable slot of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_embedding_vectors(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenEmbeddingVectors,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the contract on `fut`.
    match unsafe { BlazenFuture::take_typed::<Vec<Vec<f32>>>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: `out` is non-null per the branch above.
                unsafe {
                    *out = BlazenEmbeddingVectors(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(err, e),
    }
}

/// Returns the number of embedding vectors. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingVectors`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_vectors_count(
    handle: *const BlazenEmbeddingVectors,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is live.
    let r = unsafe { &*handle };
    r.0.len()
}

/// Returns the dimensionality of the `vec_idx`-th embedding vector.
/// Returns `0` on a null handle or an out-of-range index.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingVectors`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_vectors_dim(
    handle: *const BlazenEmbeddingVectors,
    vec_idx: usize,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is live.
    let r = unsafe { &*handle };
    r.0.get(vec_idx).map_or(0, Vec::len)
}

/// Returns the `dim_idx`-th coordinate of the `vec_idx`-th embedding
/// vector. Returns `0.0` on a null handle or an out-of-range index.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingVectors`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_vectors_get(
    handle: *const BlazenEmbeddingVectors,
    vec_idx: usize,
    dim_idx: usize,
) -> f32 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller has guaranteed `handle` is live.
    let r = unsafe { &*handle };
    r.0.get(vec_idx)
        .and_then(|v| v.get(dim_idx))
        .copied()
        .unwrap_or(0.0)
}

/// Bulk-copy the `vec_idx`-th embedding vector into the caller-supplied
/// `f32` buffer. Returns the number of coordinates written (capped at
/// `out_len`). Returns `0` on a null handle, out-of-range index, or null
/// `out_buf`. Designed for hot paths in embedding-heavy workloads (RAG,
/// semantic search).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingVectors`. `out_buf`
/// must be null OR point to a writable buffer of at least `out_len`
/// `f32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_vectors_to_buffer(
    handle: *const BlazenEmbeddingVectors,
    vec_idx: usize,
    out_buf: *mut f32,
    out_len: usize,
) -> usize {
    if handle.is_null() || out_buf.is_null() || out_len == 0 {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is live.
    let r = unsafe { &*handle };
    let Some(vec) = r.0.get(vec_idx) else {
        return 0;
    };
    let n = vec.len().min(out_len);
    // SAFETY: caller has guaranteed `out_buf` points to >= `out_len` f32s,
    // and `vec` is borrowed read-only from the handle; the two regions are
    // non-overlapping.
    unsafe {
        std::ptr::copy_nonoverlapping(vec.as_ptr(), out_buf, n);
    }
    n
}

/// Frees a `BlazenEmbeddingVectors` handle. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by
/// [`blazen_future_take_embedding_vectors`] (or the blocking variant of
/// any per-engine `_embed_blocking`). Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_vectors_free(handle: *mut BlazenEmbeddingVectors) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// FastembedProvider -- local fastembed (ORT) (feature = "fastembed")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::embed::FastembedProvider>`.
///
/// Free with [`blazen_fastembed_provider_free`].
#[cfg(feature = "fastembed")]
pub struct BlazenFastembedProvider(
    pub(crate) Arc<blazen_uniffi::concrete::embed::FastembedProvider>,
);

#[cfg(feature = "fastembed")]
impl BlazenFastembedProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFastembedProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFastembedProvider`. Both inputs are optional --
/// pass null for `model_id` to default to fastembed's `BGESmallENV15`,
/// and null for `cache_dir` to defer to the model-cache default.
///
/// On success returns `0` and writes a fresh `BlazenFastembedProvider*`
/// into `*out_model`. On failure returns `-1` and writes a fresh
/// `BlazenError*` into `*out_err`. Both out-parameters may be null to
/// discard that side of the result.
///
/// # Safety
///
/// - `model_id` / `cache_dir` must be null OR valid NUL-terminated UTF-8
///   buffers.
/// - `out_model` / `out_err` must each be null OR point to a writable
///   slot of the matching pointer type.
#[cfg(feature = "fastembed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fastembed_provider_new(
    model_id: *const c_char,
    cache_dir: *const c_char,
    out_model: *mut *mut BlazenFastembedProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };

    match blazen_uniffi::concrete::embed::FastembedProvider::new(model_id, cache_dir) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenFastembedProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously compute embedding vectors for `texts`. Returns a
/// `*mut BlazenFuture` the caller polls / waits on; the typed result is
/// popped with [`blazen_future_take_embedding_vectors`].
///
/// Returns null if `model` is null, `texts` is null with `texts_len > 0`,
/// or any text is non-UTF-8.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenFastembedProvider`.
/// - When `texts_len > 0`, `texts` must point to `texts_len` valid
///   `*const c_char` entries, each a NUL-terminated UTF-8 buffer.
#[cfg(feature = "fastembed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fastembed_provider_embed(
    model: *const BlazenFastembedProvider,
    texts: *const *const c_char,
    texts_len: usize,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return std::ptr::null_mut();
    };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<Vec<f32>>, _>(async move { inner.embed(texts).await })
}

/// Synchronous variant of [`blazen_fastembed_provider_embed`]. Returns
/// `0` on success (handle written into `*out_result`), `-1` on failure
/// (error written into `*out_err`), `-2` on invalid input (null model
/// pointer or non-UTF-8 text element).
///
/// # Safety
///
/// Same string + array contracts as [`blazen_fastembed_provider_embed`].
/// `out_result` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "fastembed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fastembed_provider_embed_blocking(
    model: *const BlazenFastembedProvider,
    texts: *const *const c_char,
    texts_len: usize,
    out_result: *mut *mut BlazenEmbeddingVectors,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return -2;
    };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.embed(texts).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenEmbeddingVectors(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Returns the dimensionality of vectors produced by this provider.
/// Returns `0` on a null handle.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenFastembedProvider`.
#[cfg(feature = "fastembed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fastembed_provider_dimensions(
    model: *const BlazenFastembedProvider,
) -> u32 {
    if model.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    model.0.dimensions()
}

/// Frees a `BlazenFastembedProvider` handle. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_fastembed_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "fastembed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fastembed_provider_free(model: *mut BlazenFastembedProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// TractEmbedProvider -- pure-Rust ONNX (feature = "tract")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::embed::TractEmbedProvider>`.
#[cfg(feature = "tract")]
pub struct BlazenTractEmbedProvider(
    pub(crate) Arc<blazen_uniffi::concrete::embed::TractEmbedProvider>,
);

#[cfg(feature = "tract")]
impl BlazenTractEmbedProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenTractEmbedProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenTractEmbedProvider`. See
/// [`blazen_fastembed_provider_new`] for argument conventions.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_new`].
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tract_embed_provider_new(
    model_id: *const c_char,
    cache_dir: *const c_char,
    out_model: *mut *mut BlazenTractEmbedProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };

    match blazen_uniffi::concrete::embed::TractEmbedProvider::new(model_id, cache_dir) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenTractEmbedProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously compute embedding vectors.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed`].
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tract_embed_provider_embed(
    model: *const BlazenTractEmbedProvider,
    texts: *const *const c_char,
    texts_len: usize,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return std::ptr::null_mut();
    };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<Vec<f32>>, _>(async move { inner.embed(texts).await })
}

/// Synchronous variant of [`blazen_tract_embed_provider_embed`].
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed_blocking`].
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tract_embed_provider_embed_blocking(
    model: *const BlazenTractEmbedProvider,
    texts: *const *const c_char,
    texts_len: usize,
    out_result: *mut *mut BlazenEmbeddingVectors,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return -2;
    };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.embed(texts).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenEmbeddingVectors(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Returns the dimensionality of vectors produced by this provider.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenTractEmbedProvider`.
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tract_embed_provider_dimensions(
    model: *const BlazenTractEmbedProvider,
) -> u32 {
    if model.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    model.0.dimensions()
}

/// Frees a `BlazenTractEmbedProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_free`].
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tract_embed_provider_free(model: *mut BlazenTractEmbedProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// CandleEmbedProvider -- candle BERT-family (feature = "candle-embed")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::embed::CandleEmbedProvider>`.
///
/// Note: the underlying `CandleEmbedProvider::new` constructor is async
/// (HF download + weights load), so this factory drives it through the
/// shared cabi tokio runtime via `block_on` -- matching the pattern in
/// `crate::three_d` for `TripoSrProvider`. Callers should not invoke
/// this factory from inside another tokio runtime on the same thread.
#[cfg(feature = "candle-embed")]
pub struct BlazenCandleEmbedProvider(
    pub(crate) Arc<blazen_uniffi::concrete::embed::CandleEmbedProvider>,
);

#[cfg(feature = "candle-embed")]
impl BlazenCandleEmbedProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenCandleEmbedProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenCandleEmbedProvider`. Both inputs are optional --
/// pass null for `model_id` to default to
/// `sentence-transformers/all-MiniLM-L6-v2`, and null for `cache_dir` to
/// defer to the model-cache default.
///
/// The underlying constructor is async; this factory blocks on the
/// shared cabi tokio runtime to resolve it.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_new`].
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_embed_provider_new(
    model_id: *const c_char,
    cache_dir: *const c_char,
    out_model: *mut *mut BlazenCandleEmbedProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };

    // The uniffi-side `CandleEmbedProvider::new` is a sync
    // `#[uniffi::constructor]` that already drives the upstream `async fn`
    // through the shared tokio runtime internally (see
    // `crates/blazen-uniffi/src/concrete/embed.rs`). We delegate to it
    // directly here -- the runtime is already a process-wide singleton, so
    // there is no double-`block_on` hazard from this entry point.
    match blazen_uniffi::concrete::embed::CandleEmbedProvider::new(model_id, cache_dir) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenCandleEmbedProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously compute embedding vectors.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed`].
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_embed_provider_embed(
    model: *const BlazenCandleEmbedProvider,
    texts: *const *const c_char,
    texts_len: usize,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return std::ptr::null_mut();
    };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<Vec<f32>>, _>(async move { inner.embed(texts).await })
}

/// Synchronous variant of [`blazen_candle_embed_provider_embed`].
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed_blocking`].
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_embed_provider_embed_blocking(
    model: *const BlazenCandleEmbedProvider,
    texts: *const *const c_char,
    texts_len: usize,
    out_result: *mut *mut BlazenEmbeddingVectors,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return -2;
    };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.embed(texts).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenEmbeddingVectors(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Returns the dimensionality of vectors produced by this provider.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCandleEmbedProvider`.
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_embed_provider_dimensions(
    model: *const BlazenCandleEmbedProvider,
) -> u32 {
    if model.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    model.0.dimensions()
}

/// Frees a `BlazenCandleEmbedProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_free`].
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_embed_provider_free(model: *mut BlazenCandleEmbedProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// OpenAiEmbeddingProvider -- OpenAI /v1/embeddings (no feature gate)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::embed::OpenAiEmbeddingProvider>`.
pub struct BlazenOpenAiEmbeddingProvider(
    pub(crate) Arc<blazen_uniffi::concrete::embed::OpenAiEmbeddingProvider>,
);

impl BlazenOpenAiEmbeddingProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenOpenAiEmbeddingProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenOpenAiEmbeddingProvider` from an `OpenAI` API key.
///
/// `model` overrides the default `text-embedding-3-small` model id when
/// non-null. An empty `api_key` falls back to `OPENAI_API_KEY` from the
/// environment at request time.
///
/// This constructor is infallible (the underlying Rust constructor
/// returns `Arc<Self>` directly), so it returns the handle by pointer
/// rather than an out-param. Returns null only if `api_key` is null or
/// non-UTF-8.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null;
///   empty string is allowed).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_err` may be null to discard the error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_embedding_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenOpenAiEmbeddingProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_openai_embedding_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::embed::OpenAiEmbeddingProvider::new(api_key, model_id);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenOpenAiEmbeddingProvider(arc).into_ptr();
        }
    }
    0
}

/// Asynchronously compute embedding vectors.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_embedding_provider_embed(
    model: *const BlazenOpenAiEmbeddingProvider,
    texts: *const *const c_char,
    texts_len: usize,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return std::ptr::null_mut();
    };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<Vec<f32>>, _>(async move { inner.embed(texts).await })
}

/// Synchronous variant of [`blazen_openai_embedding_provider_embed`].
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_embedding_provider_embed_blocking(
    model: *const BlazenOpenAiEmbeddingProvider,
    texts: *const *const c_char,
    texts_len: usize,
    out_result: *mut *mut BlazenEmbeddingVectors,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return -2;
    };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.embed(texts).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenEmbeddingVectors(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Returns the dimensionality of vectors produced by this provider.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenOpenAiEmbeddingProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_embedding_provider_dimensions(
    model: *const BlazenOpenAiEmbeddingProvider,
) -> u32 {
    if model.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    model.0.dimensions()
}

/// Frees a `BlazenOpenAiEmbeddingProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_embedding_provider_free(
    model: *mut BlazenOpenAiEmbeddingProvider,
) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// FalEmbeddingProvider -- fal.ai (no feature gate)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::embed::FalEmbeddingProvider>`.
pub struct BlazenFalEmbeddingProvider(
    pub(crate) Arc<blazen_uniffi::concrete::embed::FalEmbeddingProvider>,
);

impl BlazenFalEmbeddingProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalEmbeddingProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFalEmbeddingProvider` from a fal.ai API key.
///
/// `model` overrides the default `openai/text-embedding-3-small` model
/// id when non-null. An empty `api_key` falls back to `FAL_KEY` from
/// the environment at request time.
///
/// # Safety
///
/// Same contracts as [`blazen_openai_embedding_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_embedding_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenFalEmbeddingProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_embedding_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::embed::FalEmbeddingProvider::new(api_key, model_id);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFalEmbeddingProvider(arc).into_ptr();
        }
    }
    0
}

/// Asynchronously compute embedding vectors.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_embedding_provider_embed(
    model: *const BlazenFalEmbeddingProvider,
    texts: *const *const c_char,
    texts_len: usize,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return std::ptr::null_mut();
    };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<Vec<f32>>, _>(async move { inner.embed(texts).await })
}

/// Synchronous variant of [`blazen_fal_embedding_provider_embed`].
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_embed_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_embedding_provider_embed_blocking(
    model: *const BlazenFalEmbeddingProvider,
    texts: *const *const c_char,
    texts_len: usize,
    out_result: *mut *mut BlazenEmbeddingVectors,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the array + UTF-8 contract on `texts`.
    let Some(texts) = (unsafe { collect_strings(texts, texts_len) }) else {
        return -2;
    };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.embed(texts).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenEmbeddingVectors(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Returns the dimensionality of vectors produced by this provider.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenFalEmbeddingProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_embedding_provider_dimensions(
    model: *const BlazenFalEmbeddingProvider,
) -> u32 {
    if model.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    model.0.dimensions()
}

/// Frees a `BlazenFalEmbeddingProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_fastembed_provider_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_embedding_provider_free(
    model: *mut BlazenFalEmbeddingProvider,
) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// Polymorphic `as_embedding_provider` conversions
// ===========================================================================
//
// One C function per engine that clones the inner per-engine
// `Arc<...Provider>`, coerces it to
// `Arc<dyn blazen_uniffi::concrete::bases::EmbeddingProvider>`, and boxes the
// result into a [`crate::embedding_provider::BlazenEmbeddingProvider`] handle.
// Used by callers that need to pass a polymorphic embedding provider into a
// surface that takes the trait object.
//
// The original per-engine handle is BORROWED and remains valid after the
// conversion — both handles clean up independently.

use crate::embedding_provider::BlazenEmbeddingProvider;

/// Returns a fresh [`BlazenEmbeddingProvider`] cloned from this per-engine
/// handle. Returns null on a null input. Caller frees with
/// [`crate::embedding_provider::blazen_embedding_provider_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenFastembedProvider`.
#[cfg(feature = "fastembed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fastembed_provider_as_embedding_provider(
    handle: *const BlazenFastembedProvider,
) -> *mut BlazenEmbeddingProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenEmbeddingProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenTractEmbedProvider`.
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tract_embed_provider_as_embedding_provider(
    handle: *const BlazenTractEmbedProvider,
) -> *mut BlazenEmbeddingProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenEmbeddingProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenCandleEmbedProvider`.
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_embed_provider_as_embedding_provider(
    handle: *const BlazenCandleEmbedProvider,
) -> *mut BlazenEmbeddingProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenEmbeddingProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiEmbeddingProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_embedding_provider_as_embedding_provider(
    handle: *const BlazenOpenAiEmbeddingProvider,
) -> *mut BlazenEmbeddingProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenEmbeddingProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenFalEmbeddingProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_embedding_provider_as_embedding_provider(
    handle: *const BlazenFalEmbeddingProvider,
) -> *mut BlazenEmbeddingProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenEmbeddingProvider(h.0.clone()).into_ptr()
}
