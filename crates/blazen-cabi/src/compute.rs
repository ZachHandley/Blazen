//! Compute opaque objects: [`BlazenTtsModel`], [`BlazenSttModel`], and
//! [`BlazenImageGenModel`].
//!
//! Each opaque type wraps an `Arc<blazen_uniffi::compute::*Model>` and
//! exposes its async method via three C entry points:
//!
//! - `*_blocking` — synchronous, drives the underlying future on the
//!   cabi tokio runtime via [`crate::runtime::runtime`]`::block_on`.
//! - non-suffixed — returns a `*mut BlazenFuture` immediately and spawns
//!   the underlying async task onto the runtime. Consumers observe
//!   completion via [`crate::future`] and then pop the typed result with
//!   the matching `blazen_future_take_*` extern defined below.
//! - `*_free` — releases the heap allocation for the model handle.
//!
//! ## Ownership conventions
//!
//! - Model handles are heap-allocated `Box<BlazenXModel>` returned by
//!   factory functions (wired in Phase R4). Callers free with the matching
//!   `*_free`. Double-free is undefined behavior.
//! - String inputs are borrowed for the duration of the call only — the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.
//! - Result handles produced through `*_blocking` and through the typed
//!   `blazen_future_take_*` are caller-owned and freed via the matching
//!   `blazen_*_result_free` in [`crate::compute_records`].
//! - Error handles produced on the failure path are caller-owned and
//!   freed via [`crate::error::blazen_error_free`].
//!
//! ## Optional inputs
//!
//! `voice` / `language` / `negative_prompt` and the per-call image
//! `model_override` are `Option<String>` on the Rust side; the C
//! signatures accept `*const c_char` and treat null as `None`. The image
//! `width` / `height` / `num_images` are `Option<u32>` on the Rust side;
//! the C signatures accept `i32` and treat `-1` as `None` (any other
//! negative value is clamped to `None`).

#![allow(dead_code)]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute::{
    ImageGenModel as InnerImageGenModel, ImageGenResult as InnerImageGenResult,
    SttModel as InnerSttModel, SttResult as InnerSttResult, TtsModel as InnerTtsModel,
    TtsResult as InnerTtsResult,
};

use crate::compute_records::{BlazenImageGenResult, BlazenSttResult, BlazenTtsResult};
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a C `i32` Option-as-sentinel into the equivalent `Option<u32>`.
///
/// Any negative value collapses to `None`; non-negative values pass through
/// as `Some(v as u32)`. Used by the image-generation wrappers where width /
/// height / `num_images` are `Option<u32>` on the Rust side.
#[inline]
fn opt_u32_from_i32(v: i32) -> Option<u32> {
    if v < 0 {
        None
    } else {
        Some(u32::try_from(v).unwrap_or(0))
    }
}

// ---------------------------------------------------------------------------
// BlazenTtsModel
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::compute::TtsModel>`.
///
/// Produced by the per-backend TTS factory functions (Phase R4). Free with
/// [`blazen_tts_model_free`].
pub struct BlazenTtsModel(pub(crate) Arc<InnerTtsModel>);

impl BlazenTtsModel {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenTtsModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerTtsModel>> for BlazenTtsModel {
    fn from(inner: Arc<InnerTtsModel>) -> Self {
        Self(inner)
    }
}

/// Synchronously synthesize speech for `text` and write the result into
/// `out_result` on success, or a `BlazenError` into `out_err` on failure.
///
/// Returns `0` on success, `-1` on failure (`out_err` may be inspected),
/// `-2` on invalid input (null model / non-UTF-8 `text`).
///
/// Pass `voice` / `language` as null to leave them unset.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenTtsModel` produced by the
///   cabi surface.
/// - `text` must be a valid NUL-terminated UTF-8 buffer.
/// - `voice` / `language` must be null OR valid NUL-terminated UTF-8 buffers.
/// - `out_result` / `out_err` must be null OR writable pointers to a
///   `*mut BlazenTtsResult` / `*mut BlazenError` slot respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_model_synthesize_blocking(
    model: *const BlazenTtsModel,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenTtsModel` pointer.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `text`.
    let Some(text) = (unsafe { cstr_to_str(text) }) else {
        return -2;
    };
    let text = text.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let voice = unsafe { cstr_to_opt_string(voice) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.synthesize(text, voice, language).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously synthesize speech for `text`. Returns a `*mut BlazenFuture`
/// the caller polls / waits on; the typed result is popped with
/// [`blazen_future_take_tts_result`].
///
/// Returns null if `model` is null or `text` is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as [`blazen_tts_model_synthesize_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_model_synthesize(
    model: *const BlazenTtsModel,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenTtsModel` pointer.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `text`.
    let Some(text) = (unsafe { cstr_to_str(text) }) else {
        return std::ptr::null_mut();
    };
    let text = text.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let voice = unsafe { cstr_to_opt_string(voice) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerTtsResult, _>(async move {
        inner.synthesize(text, voice, language).await
    })
}

/// Frees a `BlazenTtsModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by the cabi
/// surface's TTS factory functions. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_model_free(model: *mut BlazenTtsModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// BlazenSttModel
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::compute::SttModel>`.
///
/// Produced by the per-backend STT factory functions (Phase R4). Free with
/// [`blazen_stt_model_free`].
pub struct BlazenSttModel(pub(crate) Arc<InnerSttModel>);

impl BlazenSttModel {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenSttModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerSttModel>> for BlazenSttModel {
    fn from(inner: Arc<InnerSttModel>) -> Self {
        Self(inner)
    }
}

/// Synchronously transcribe audio at `audio_source` and write the result
/// into `out_result` on success, or a `BlazenError` into `out_err` on
/// failure.
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input (null
/// model / non-UTF-8 `audio_source`).
///
/// Pass `language` as null to leave it unset.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenSttModel` produced by the
///   cabi surface.
/// - `audio_source` must be a valid NUL-terminated UTF-8 buffer.
/// - `language` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_result` / `out_err` must be null OR writable pointers to the
///   appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_model_transcribe_blocking(
    model: *const BlazenSttModel,
    audio_source: *const c_char,
    language: *const c_char,
    out_result: *mut *mut BlazenSttResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenSttModel` pointer.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `audio_source`.
    let Some(audio_source) = (unsafe { cstr_to_str(audio_source) }) else {
        return -2;
    };
    let audio_source = audio_source.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.transcribe(audio_source, language).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenSttResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously transcribe audio at `audio_source`. Returns a future
/// handle whose typed result is popped with
/// [`blazen_future_take_stt_result`].
///
/// Returns null if `model` is null or `audio_source` is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as [`blazen_stt_model_transcribe_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_model_transcribe(
    model: *const BlazenSttModel,
    audio_source: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenSttModel` pointer.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `audio_source`.
    let Some(audio_source) = (unsafe { cstr_to_str(audio_source) }) else {
        return std::ptr::null_mut();
    };
    let audio_source = audio_source.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerSttResult, _>(async move {
        inner.transcribe(audio_source, language).await
    })
}

/// Frees a `BlazenSttModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by the cabi
/// surface's STT factory functions. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_model_free(model: *mut BlazenSttModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// BlazenImageGenModel
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::compute::ImageGenModel>`.
///
/// Produced by the per-backend image-generation factory functions
/// (Phase R4). Free with [`blazen_image_gen_model_free`].
pub struct BlazenImageGenModel(pub(crate) Arc<InnerImageGenModel>);

impl BlazenImageGenModel {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenImageGenModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerImageGenModel>> for BlazenImageGenModel {
    fn from(inner: Arc<InnerImageGenModel>) -> Self {
        Self(inner)
    }
}

/// Synchronously generate one or more images for `prompt` and write the
/// result into `out_result` on success, or a `BlazenError` into `out_err`
/// on failure.
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input (null
/// model / non-UTF-8 `prompt` or `negative_prompt`).
///
/// `width` / `height` / `num_images`: pass `-1` (or any negative value) to
/// leave unset. `model_override`: null leaves it unset (the per-model
/// default endpoint configured at construction time is used).
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenImageGenModel` produced
///   by the cabi surface.
/// - `prompt` / `negative_prompt` must be valid NUL-terminated UTF-8
///   buffers (non-null).
/// - `model_override` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_result` / `out_err` must be null OR writable pointers to the
///   appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_model_generate_blocking(
    model: *const BlazenImageGenModel,
    prompt: *const c_char,
    negative_prompt: *const c_char,
    width: i32,
    height: i32,
    num_images: i32,
    model_override: *const c_char,
    out_result: *mut *mut BlazenImageGenResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenImageGenModel` pointer.
    let model_ref = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return -2;
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `negative_prompt`.
    //
    // `negative_prompt` is declared non-nullable on the C side but the
    // upstream Rust signature accepts `Option<String>`. A null pointer
    // collapses to `None` for forward-compat with callers that prefer to
    // express "no negative prompt" via null instead of an empty string.
    let negative_prompt = unsafe { cstr_to_opt_string(negative_prompt) };
    let width = opt_u32_from_i32(width);
    let height = opt_u32_from_i32(height);
    let num_images = opt_u32_from_i32(num_images);
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_override = unsafe { cstr_to_opt_string(model_override) };

    let inner = Arc::clone(&model_ref.0);
    let result = runtime().block_on(async move {
        inner
            .generate(
                prompt,
                negative_prompt,
                width,
                height,
                num_images,
                model_override,
            )
            .await
    });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenImageGenResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !out_err.is_null() {
                // SAFETY: caller has guaranteed `out_err` is writable.
                unsafe {
                    *out_err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Asynchronously generate one or more images for `prompt`. Returns a
/// future handle whose typed result is popped with
/// [`blazen_future_take_image_gen_result`].
///
/// Returns null if `model` is null or `prompt` is null / non-UTF-8.
///
/// # Safety
///
/// Same contracts as [`blazen_image_gen_model_generate_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_model_generate(
    model: *const BlazenImageGenModel,
    prompt: *const c_char,
    negative_prompt: *const c_char,
    width: i32,
    height: i32,
    num_images: i32,
    model_override: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenImageGenModel` pointer.
    let model_ref = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `negative_prompt`.
    let negative_prompt = unsafe { cstr_to_opt_string(negative_prompt) };
    let width = opt_u32_from_i32(width);
    let height = opt_u32_from_i32(height);
    let num_images = opt_u32_from_i32(num_images);
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_override = unsafe { cstr_to_opt_string(model_override) };

    let inner = Arc::clone(&model_ref.0);
    BlazenFuture::spawn::<InnerImageGenResult, _>(async move {
        inner
            .generate(
                prompt,
                negative_prompt,
                width,
                height,
                num_images,
                model_override,
            )
            .await
    })
}

/// Frees a `BlazenImageGenModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by the cabi
/// surface's image-generation factory functions. Double-free is undefined
/// behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_model_free(model: *mut BlazenImageGenModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// Typed future-take entry points
// ---------------------------------------------------------------------------

/// Pops a typed `TtsResult` out of `fut`. On success returns `0` and writes
/// a caller-owned `*mut BlazenTtsResult` into `out`; on failure returns
/// `-1` and writes a caller-owned `*mut BlazenError` into `err`.
///
/// `out` / `err` may be null when the caller wants to discard the value.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_tts_model_synthesize`], not yet freed, and not concurrently
/// freed from another thread. `out` / `err` must be null OR writable
/// pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_tts_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenTtsResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerTtsResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenTtsResult::from(v).into_ptr();
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

/// Pops a typed `SttResult` out of `fut`. Mirrors
/// [`blazen_future_take_tts_result`] semantics.
///
/// # Safety
///
/// Same contracts as [`blazen_future_take_tts_result`], with `fut`
/// produced by [`blazen_stt_model_transcribe`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_stt_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenSttResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerSttResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenSttResult::from(v).into_ptr();
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

/// Pops a typed `ImageGenResult` out of `fut`. Mirrors
/// [`blazen_future_take_tts_result`] semantics.
///
/// # Safety
///
/// Same contracts as [`blazen_future_take_tts_result`], with `fut`
/// produced by [`blazen_image_gen_model_generate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_image_gen_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenImageGenResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerImageGenResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenImageGenResult::from(v).into_ptr();
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
