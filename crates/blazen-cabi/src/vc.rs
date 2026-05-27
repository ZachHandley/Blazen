//! Per-engine voice-conversion provider opaques + C ABI factories.
//!
//! Mirrors [`crate::tts`] for the two VC engines currently exposed by
//! [`blazen_uniffi::concrete::vc`]:
//!
//! - [`BlazenRvcProvider`] — native Retrieval-based Voice Conversion
//!   (feature `audio-vc-rvc`, transitively `audio-vc`).
//! - [`BlazenFalVcProvider`] — fal.ai-hosted cloud VC (no extra feature gate;
//!   only the parent `audio-vc` is required since the upstream
//!   `concrete::vc` module is implicitly `#[cfg(feature = "audio-vc")]`).
//!
//! ## Async future plumbing
//!
//! - `_convert_voice` returns `*mut BlazenFuture` whose typed result is
//!   popped with [`crate::compute_vc::blazen_future_take_vc_result`].
//! - `_clone_voice` returns `*mut BlazenFuture` whose unit result is popped
//!   with [`crate::persist::blazen_future_take_unit`] (the canonical taker
//!   for `Result<(), BlazenError>` futures across the cabi surface).
//! - `_list_target_voices` returns `*mut BlazenFuture` whose
//!   `Vec<TargetVoice>` result is popped with
//!   [`crate::compute_vc::blazen_future_take_target_voice_list`].
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<Blazen<Engine>Provider>`
//!   returned by the factory functions. Callers free with the matching
//!   `*_free`. Double-free is undefined behavior.
//! - String inputs are borrowed for the call only — the wrappers copy out
//!   everything they need into owned `String`s before spawning the
//!   underlying task.
//! - Result handles (`BlazenVcResult`, `BlazenTargetVoiceList`) flow
//!   through the existing pipeline in [`crate::compute_vc`] /
//!   [`crate::compute_records`] and are freed with
//!   [`crate::compute_records::blazen_vc_result_free`] /
//!   [`crate::compute_records::blazen_target_voice_list_free`].
//!
//! ## Relationship to the central [`crate::compute_vc::BlazenVcModel`]
//!
//! The central `BlazenVcModel` + `blazen_vc_model_*` entry points in
//! [`crate::compute_vc`] remain in place — this module is purely additive.
//! Foreign hosts (Ruby, future Dart / Crystal / Lua / PHP) can migrate to
//! the per-engine surface incrementally without breaking the existing Ruby
//! gem entry points.

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute_vc::{TargetVoice as InnerTargetVoice, VcResult as InnerVcResult};
use blazen_uniffi::concrete::bases::VcProvider as _;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::cstr_to_str;

// ---------------------------------------------------------------------------
// Local helpers (mirroring crate::tts)
// ---------------------------------------------------------------------------

/// Writes a caller-owned `BlazenError` into `out_err` (if non-null) and
/// returns `-1` for use in tail position.
fn write_error(out_err: *mut *mut BlazenError, err: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: `out_err` is non-null per the branch above. The caller
        // has guaranteed it points to a writable `*mut BlazenError` slot.
        unsafe {
            *out_err = BlazenError::from(err).into_ptr();
        }
    }
    -1
}

/// Builds + writes a `BlazenError::Internal { message }` into `out_err`.
/// Used for argument-shape errors (null required pointers, non-UTF-8
/// strings) that don't originate from a `blazen_uniffi` call.
fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) -> i32 {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    )
}

// ===========================================================================
// RvcProvider — native Retrieval-based Voice Conversion
// (feature = "audio-vc-rvc")
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::vc::RvcProvider>`.
///
/// Free with [`blazen_rvc_provider_free`].
#[cfg(feature = "audio-vc-rvc")]
pub struct BlazenRvcProvider(pub(crate) Arc<blazen_uniffi::concrete::vc::RvcProvider>);

#[cfg(feature = "audio-vc-rvc")]
impl BlazenRvcProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenRvcProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenRvcProvider` with the default CPU backend.
///
/// Target voices are loaded lazily from `$BLAZEN_RVC_VOICE_DIR/<voice_id>/`
/// on the first `convert_voice` call. This constructor is infallible.
///
/// # Safety
///
/// Always safe to call; returns a non-null heap-allocated handle.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub extern "C" fn blazen_rvc_provider_new() -> *mut BlazenRvcProvider {
    let arc = blazen_uniffi::concrete::vc::RvcProvider::new();
    BlazenRvcProvider(arc).into_ptr()
}

/// Asynchronously convert the source utterance at `input_path` into the
/// registered target voice `target_voice_id`. Returns a `*mut BlazenFuture`
/// the caller polls / waits on; the typed result is popped with
/// [`crate::compute_vc::blazen_future_take_vc_result`].
///
/// Returns null if `model` is null or either string is null / non-UTF-8.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenRvcProvider` produced by
///   this surface.
/// - `input_path` / `target_voice_id` must each be valid NUL-terminated
///   UTF-8 buffers (non-null).
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_convert_voice(
    model: *const BlazenRvcProvider,
    input_path: *const c_char,
    target_voice_id: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenRvcProvider`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(input_path) = (unsafe { cstr_to_str(input_path) }) else {
        return std::ptr::null_mut();
    };
    let input_path = input_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(target_voice_id) = (unsafe { cstr_to_str(target_voice_id) }) else {
        return std::ptr::null_mut();
    };
    let target_voice_id = target_voice_id.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerVcResult, _>(async move {
        inner.convert_voice(input_path, target_voice_id).await
    })
}

/// Synchronous variant of [`blazen_rvc_provider_convert_voice`]. Returns
/// `0` on success (typed result in `*out_result`), `-1` on failure (error
/// in `*out_err`), `-2` on invalid input (null model / non-UTF-8 string).
///
/// # Safety
///
/// Same string contracts as [`blazen_rvc_provider_convert_voice`].
/// `out_result` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_convert_voice_blocking(
    model: *const BlazenRvcProvider,
    input_path: *const c_char,
    target_voice_id: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenVcResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_rvc_provider_convert_voice: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(input_path) = (unsafe { cstr_to_str(input_path) }) else {
        write_internal_error(
            out_err,
            "blazen_rvc_provider_convert_voice: null or non-UTF-8 input_path",
        );
        return -2;
    };
    let input_path = input_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(target_voice_id) = (unsafe { cstr_to_str(target_voice_id) }) else {
        write_internal_error(
            out_err,
            "blazen_rvc_provider_convert_voice: null or non-UTF-8 target_voice_id",
        );
        return -2;
    };
    let target_voice_id = target_voice_id.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.convert_voice(input_path, target_voice_id).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenVcResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously register a new target voice from the reference utterance
/// at `reference_path` under the backend-scoped identifier `voice_id`.
/// Returns a `*mut BlazenFuture` whose unit result is popped with
/// [`crate::persist::blazen_future_take_unit`].
///
/// Returns null if `model` is null or either string is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as [`blazen_rvc_provider_convert_voice`].
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_clone_voice(
    model: *const BlazenRvcProvider,
    voice_id: *const c_char,
    reference_path: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        return std::ptr::null_mut();
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(reference_path) = (unsafe { cstr_to_str(reference_path) }) else {
        return std::ptr::null_mut();
    };
    let reference_path = reference_path.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<(), _>(async move { inner.clone_voice(voice_id, reference_path).await })
}

/// Synchronous variant of [`blazen_rvc_provider_clone_voice`]. Returns
/// `0` on success, `-1` on failure (error in `*out_err`), `-2` on invalid
/// input (null model / non-UTF-8 string).
///
/// # Safety
///
/// Same string contracts as [`blazen_rvc_provider_convert_voice_blocking`].
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_clone_voice_blocking(
    model: *const BlazenRvcProvider,
    voice_id: *const c_char,
    reference_path: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_rvc_provider_clone_voice: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        write_internal_error(
            out_err,
            "blazen_rvc_provider_clone_voice: null or non-UTF-8 voice_id",
        );
        return -2;
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(reference_path) = (unsafe { cstr_to_str(reference_path) }) else {
        write_internal_error(
            out_err,
            "blazen_rvc_provider_clone_voice: null or non-UTF-8 reference_path",
        );
        return -2;
    };
    let reference_path = reference_path.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.clone_voice(voice_id, reference_path).await });
    match result {
        Ok(()) => 0,
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously list the target voices currently known to the backend.
/// Returns a `*mut BlazenFuture` whose `Vec<TargetVoice>` result is popped
/// with [`crate::compute_vc::blazen_future_take_target_voice_list`].
///
/// Returns null if `model` is null.
///
/// # Safety
///
/// `model` must be a valid pointer to a `BlazenRvcProvider` produced by
/// this surface.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_list_target_voices(
    model: *const BlazenRvcProvider,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<InnerTargetVoice>, _>(async move { inner.list_target_voices().await })
}

/// Synchronous variant of [`blazen_rvc_provider_list_target_voices`].
/// Returns `0` on success (list handle in `*out_list`), `-1` on failure
/// (error in `*out_err`), `-2` on invalid input (null model).
///
/// # Safety
///
/// `model` must be a valid pointer to a `BlazenRvcProvider`. `out_list` /
/// `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_list_target_voices_blocking(
    model: *const BlazenRvcProvider,
    out_list: *mut *mut crate::compute_records::BlazenTargetVoiceList,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(
            out_err,
            "blazen_rvc_provider_list_target_voices: null model",
        );
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.list_target_voices().await });
    match result {
        Ok(voices) => {
            if !out_list.is_null() {
                // SAFETY: caller has guaranteed `out_list` is writable.
                unsafe {
                    *out_list =
                        crate::compute_records::BlazenTargetVoiceList::from(voices).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenRvcProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_rvc_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_rvc_provider_free(model: *mut BlazenRvcProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// FalVcProvider — fal.ai cloud voice conversion (feature = "audio-vc")
// ===========================================================================
//
// The upstream `blazen_uniffi::concrete::vc::FalVcProvider` is gated by
// the parent `#[cfg(feature = "audio-vc")]` on `concrete::vc`, so this
// section mirrors that gate. `clone_voice` / `list_target_voices` are
// exposed for API parity even though fal returns
// `BlazenError::Unsupported` / an empty list respectively — keeping the
// surface uniform makes binding-side feature detection easier (the symbol
// is present, the runtime tells you what's supported).

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::vc::FalVcProvider>`.
///
/// Free with [`blazen_fal_vc_provider_free`].
#[cfg(feature = "audio-vc")]
pub struct BlazenFalVcProvider(pub(crate) Arc<blazen_uniffi::concrete::vc::FalVcProvider>);

#[cfg(feature = "audio-vc")]
impl BlazenFalVcProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalVcProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFalVcProvider` from a fal.ai API key.
///
/// `api_key` may be empty when the fal client resolves it from the
/// `FAL_KEY` environment variable.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null;
///   empty string is allowed).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_new(
    api_key: *const c_char,
    out_model: *mut *mut BlazenFalVcProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_vc_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();

    let arc = blazen_uniffi::concrete::vc::FalVcProvider::new(api_key);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFalVcProvider(arc).into_ptr();
        }
    }
    0
}

/// Asynchronously convert the source utterance at `input_path` (an
/// `http(s)://` / `data:` URL reachable by fal's workers) into the target
/// voice `target_voice_id`. Returns a `*mut BlazenFuture` whose typed
/// result is popped with
/// [`crate::compute_vc::blazen_future_take_vc_result`].
///
/// Unlike the native [`BlazenRvcProvider`], fal requires a URL — local
/// file paths won't work since fal's workers can't reach the caller's
/// disk. Pass a presigned / public URL or a `data:` URI.
///
/// Returns null if `model` is null or either string is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as [`blazen_rvc_provider_convert_voice`].
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_convert_voice(
    model: *const BlazenFalVcProvider,
    input_path: *const c_char,
    target_voice_id: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(input_path) = (unsafe { cstr_to_str(input_path) }) else {
        return std::ptr::null_mut();
    };
    let input_path = input_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(target_voice_id) = (unsafe { cstr_to_str(target_voice_id) }) else {
        return std::ptr::null_mut();
    };
    let target_voice_id = target_voice_id.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerVcResult, _>(async move {
        inner.convert_voice(input_path, target_voice_id).await
    })
}

/// Synchronous variant of [`blazen_fal_vc_provider_convert_voice`].
///
/// # Safety
///
/// Same contracts as [`blazen_rvc_provider_convert_voice_blocking`].
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_convert_voice_blocking(
    model: *const BlazenFalVcProvider,
    input_path: *const c_char,
    target_voice_id: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenVcResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_fal_vc_provider_convert_voice: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(input_path) = (unsafe { cstr_to_str(input_path) }) else {
        write_internal_error(
            out_err,
            "blazen_fal_vc_provider_convert_voice: null or non-UTF-8 input_path",
        );
        return -2;
    };
    let input_path = input_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(target_voice_id) = (unsafe { cstr_to_str(target_voice_id) }) else {
        write_internal_error(
            out_err,
            "blazen_fal_vc_provider_convert_voice: null or non-UTF-8 target_voice_id",
        );
        return -2;
    };
    let target_voice_id = target_voice_id.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.convert_voice(input_path, target_voice_id).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenVcResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously attempt to clone a voice through fal. Today the
/// upstream surface always surfaces `BlazenError::Unsupported` — the
/// future will resolve to an error, but the entry point is exposed for
/// binding-side parity with [`blazen_rvc_provider_clone_voice`]. Pop the
/// unit result with [`crate::persist::blazen_future_take_unit`].
///
/// Returns null if `model` is null or either string is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as [`blazen_rvc_provider_clone_voice`].
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_clone_voice(
    model: *const BlazenFalVcProvider,
    voice_id: *const c_char,
    reference_path: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        return std::ptr::null_mut();
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(reference_path) = (unsafe { cstr_to_str(reference_path) }) else {
        return std::ptr::null_mut();
    };
    let reference_path = reference_path.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<(), _>(async move { inner.clone_voice(voice_id, reference_path).await })
}

/// Synchronous variant of [`blazen_fal_vc_provider_clone_voice`].
///
/// # Safety
///
/// Same contracts as [`blazen_rvc_provider_clone_voice_blocking`].
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_clone_voice_blocking(
    model: *const BlazenFalVcProvider,
    voice_id: *const c_char,
    reference_path: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_fal_vc_provider_clone_voice: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        write_internal_error(
            out_err,
            "blazen_fal_vc_provider_clone_voice: null or non-UTF-8 voice_id",
        );
        return -2;
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(reference_path) = (unsafe { cstr_to_str(reference_path) }) else {
        write_internal_error(
            out_err,
            "blazen_fal_vc_provider_clone_voice: null or non-UTF-8 reference_path",
        );
        return -2;
    };
    let reference_path = reference_path.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.clone_voice(voice_id, reference_path).await });
    match result {
        Ok(()) => 0,
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously list target voices. Fal returns an empty list today;
/// the entry point exists for binding-side parity with
/// [`blazen_rvc_provider_list_target_voices`]. Pop with
/// [`crate::compute_vc::blazen_future_take_target_voice_list`].
///
/// Returns null if `model` is null.
///
/// # Safety
///
/// `model` must be a valid pointer to a `BlazenFalVcProvider`.
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_list_target_voices(
    model: *const BlazenFalVcProvider,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<InnerTargetVoice>, _>(async move { inner.list_target_voices().await })
}

/// Synchronous variant of [`blazen_fal_vc_provider_list_target_voices`].
///
/// # Safety
///
/// Same contracts as [`blazen_rvc_provider_list_target_voices_blocking`].
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_list_target_voices_blocking(
    model: *const BlazenFalVcProvider,
    out_list: *mut *mut crate::compute_records::BlazenTargetVoiceList,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(
            out_err,
            "blazen_fal_vc_provider_list_target_voices: null model",
        );
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { inner.list_target_voices().await });
    match result {
        Ok(voices) => {
            if !out_list.is_null() {
                // SAFETY: caller has guaranteed `out_list` is writable.
                unsafe {
                    *out_list =
                        crate::compute_records::BlazenTargetVoiceList::from(voices).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenFalVcProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_fal_vc_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "audio-vc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_vc_provider_free(model: *mut BlazenFalVcProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
