//! Voice-conversion (RVC and friends) surface for the C ABI.
//!
//! Mirrors [`crate::compute_music`] for the upstream
//! [`blazen_uniffi::compute_vc`] module:
//!
//! - [`BlazenVcModel`] — opaque handle wrapping `Arc<VcModel>`.
//! - One per-backend factory shipping today: [`blazen_vc_model_new_rvc`]
//!   (feature `audio-vc-rvc`).
//! - Non-streaming convert + voice-management (sync + async) wrappers that
//!   route through the upstream tokio runtime.
//! - Typed future-take entry point [`blazen_future_take_vc_result`] for
//!   `convert_voice`; the unit-returning `register_target_voice` async
//!   variant reuses [`crate::persist::blazen_future_take_unit`]; the
//!   list-returning `list_target_voices` async variant gets its own typed
//!   taker [`blazen_future_take_target_voice_list`].
//!
//! Streaming entry points (vtable + pump functions) live in
//! [`crate::stream_sink`] next to the existing `MusicStreamSink`
//! trampoline.
//!
//! ## Ownership conventions
//!
//! - Model handles are heap-allocated `Box<BlazenVcModel>` returned by the
//!   factory functions; caller frees with [`blazen_vc_model_free`].
//! - Result handles produced by `*_blocking` and by
//!   [`blazen_future_take_vc_result`] are caller-owned and freed via
//!   [`crate::compute_records::blazen_vc_result_free`].
//! - Voice-list handles produced by `list_target_voices*_blocking` and by
//!   [`blazen_future_take_target_voice_list`] are caller-owned and freed
//!   via [`crate::compute_records::blazen_target_voice_list_free`].
//! - Error handles produced on the failure path are caller-owned and freed
//!   via [`crate::error::blazen_error_free`].
//! - String inputs are borrowed for the duration of the call only — the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.

#![allow(dead_code)]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute_vc::{
    TargetVoice as InnerTargetVoice, VcModel as InnerVcModel, VcResult as InnerVcResult,
};
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::compute_records::{BlazenTargetVoiceList, BlazenVcResult};
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local error / sentinel helpers
//
// Mirrors `compute_music.rs` — each module keeps its own helpers
// self-contained so a future split (e.g. shipping voice conversion as its
// own cdylib) doesn't drag in unrelated surface area.
// ---------------------------------------------------------------------------

/// Writes a caller-owned `BlazenError` into `out_err` if the slot is non-null
/// and returns `-1`.
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
/// Used for argument-shape errors (null required pointers, non-UTF-8 strings)
/// that don't originate from a `blazen_uniffi` call.
fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) -> i32 {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    )
}

// ---------------------------------------------------------------------------
// BlazenVcModel handle
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::compute_vc::VcModel>`.
///
/// Produced by the per-backend factory functions ([`blazen_vc_model_new_rvc`]
/// today). Free with [`blazen_vc_model_free`].
pub struct BlazenVcModel(pub(crate) Arc<InnerVcModel>);

impl BlazenVcModel {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenVcModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerVcModel>> for BlazenVcModel {
    fn from(inner: Arc<InnerVcModel>) -> Self {
        Self(inner)
    }
}

/// Frees a `BlazenVcModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by the cabi
/// surface's voice-conversion factory functions. Double-free is undefined
/// behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_free(model: *mut BlazenVcModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// Native RVC factory (feature = "audio-vc-rvc")
// ---------------------------------------------------------------------------

/// Build a native RVC-backed [`BlazenVcModel`].
///
/// `voice_dir` may be null to leave the per-process
/// `BLAZEN_RVC_VOICE_DIR` environment variable untouched; pass a
/// NUL-terminated UTF-8 buffer to override it (the RVC pipeline reads the
/// variable lazily on the first conversion call, so callers who construct
/// the model up-front before spinning off threads are safe). `device`
/// follows the `blazen_llm::Device::parse` format
/// (`"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`, `"metal:N"`); null defers to
/// CPU.
///
/// On success returns `0` and writes a fresh `BlazenVcModel*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`.
///
/// # Safety
///
/// - `voice_dir` / `device` must each be null OR a valid NUL-terminated
///   UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable
///   slot of the matching pointer type.
#[cfg(feature = "audio-vc-rvc")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_new_rvc(
    voice_dir: *const c_char,
    device: *const c_char,
    out_model: *mut *mut BlazenVcModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on each input.
    let voice_dir = unsafe { cstr_to_opt_string(voice_dir) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on each input.
    let device = unsafe { cstr_to_opt_string(device) };

    match blazen_uniffi::compute_vc::new_rvc_model(voice_dir, device) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenVcModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Non-streaming convert_voice — sync + async
// ---------------------------------------------------------------------------

/// Synchronously convert the source utterance at `input_audio_path` into
/// the voice of registered target speaker `target_voice_id` and write the
/// result into `out_result` on success, or a `BlazenError` into `out_err`
/// on failure.
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input (null
/// model / null or non-UTF-8 path / null or non-UTF-8 voice id).
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenVcModel` produced by the
///   cabi surface.
/// - `input_audio_path` / `target_voice_id` must each be valid
///   NUL-terminated UTF-8 buffers.
/// - `out_result` / `out_err` must each be null OR writable pointers to
///   the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_convert_voice_blocking(
    model: *const BlazenVcModel,
    input_audio_path: *const c_char,
    target_voice_id: *const c_char,
    out_result: *mut *mut BlazenVcResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_vc_model_convert_voice: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `input_audio_path`.
    let Some(input_audio_path) = (unsafe { cstr_to_str(input_audio_path) }) else {
        write_internal_error(
            out_err,
            "blazen_vc_model_convert_voice: null or non-UTF-8 input_audio_path",
        );
        return -2;
    };
    let input_audio_path = input_audio_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `target_voice_id`.
    let Some(target_voice_id) = (unsafe { cstr_to_str(target_voice_id) }) else {
        write_internal_error(
            out_err,
            "blazen_vc_model_convert_voice: null or non-UTF-8 target_voice_id",
        );
        return -2;
    };
    let target_voice_id = target_voice_id.to_owned();

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move {
        Arc::clone(&inner)
            .convert_voice(input_audio_path, target_voice_id)
            .await
    });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenVcResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously convert the source utterance at `input_audio_path` into
/// the voice of registered target speaker `target_voice_id`. Returns a
/// `*mut BlazenFuture` the caller polls / waits on; the typed result is
/// popped with [`blazen_future_take_vc_result`].
///
/// Returns null if `model` is null or either string is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as [`blazen_vc_model_convert_voice_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_convert_voice(
    model: *const BlazenVcModel,
    input_audio_path: *const c_char,
    target_voice_id: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `input_audio_path`.
    let Some(input_audio_path) = (unsafe { cstr_to_str(input_audio_path) }) else {
        return std::ptr::null_mut();
    };
    let input_audio_path = input_audio_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `target_voice_id`.
    let Some(target_voice_id) = (unsafe { cstr_to_str(target_voice_id) }) else {
        return std::ptr::null_mut();
    };
    let target_voice_id = target_voice_id.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerVcResult, _>(async move {
        Arc::clone(&inner)
            .convert_voice(input_audio_path, target_voice_id)
            .await
    })
}

// ---------------------------------------------------------------------------
// Typed future-take entry points
// ---------------------------------------------------------------------------

/// Pops a typed `VcResult` out of `fut`. On success returns `0` and writes
/// a caller-owned `*mut BlazenVcResult` into `out`; on failure returns
/// `-1` and writes a caller-owned `*mut BlazenError` into `err`.
///
/// `out` / `err` may be null when the caller wants to discard the value.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_vc_model_convert_voice`], not yet freed, and not concurrently
/// freed from another thread. `out` / `err` must be null OR writable
/// pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_vc_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenVcResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerVcResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenVcResult::from(v).into_ptr();
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

/// Pops a typed `Vec<TargetVoice>` out of `fut`, wraps it in a
/// caller-owned [`BlazenTargetVoiceList`], and writes the handle into
/// `out`. On failure returns `-1` and writes a caller-owned
/// `*mut BlazenError` into `err`.
///
/// `out` / `err` may be null when the caller wants to discard the value.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_vc_model_list_target_voices`], not yet freed, and not
/// concurrently freed from another thread. `out` / `err` must be null OR
/// writable pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_target_voice_list(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenTargetVoiceList,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<Vec<InnerTargetVoice>>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenTargetVoiceList::from(v).into_ptr();
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

// ---------------------------------------------------------------------------
// list_target_voices — sync + async
// ---------------------------------------------------------------------------

/// Synchronously list the target voices this backend can currently
/// render. Writes a caller-owned [`BlazenTargetVoiceList`] into `out_list`
/// on success, or a `BlazenError` into `out_err` on failure.
///
/// Returns `0` on success, `-1` on failure, `-2` on null `model`.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenVcModel` produced by the
///   cabi surface.
/// - `out_list` / `out_err` must each be null OR writable pointers to the
///   appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_list_target_voices_blocking(
    model: *const BlazenVcModel,
    out_list: *mut *mut BlazenTargetVoiceList,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_vc_model_list_target_voices: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model = unsafe { &*model };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move { Arc::clone(&inner).list_target_voices().await });
    match result {
        Ok(voices) => {
            if !out_list.is_null() {
                // SAFETY: caller has guaranteed `out_list` is writable.
                unsafe {
                    *out_list = BlazenTargetVoiceList::from(voices).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously list the target voices this backend can render. Returns
/// a `*mut BlazenFuture` the caller polls / waits on; the typed list is
/// popped with [`blazen_future_take_target_voice_list`].
///
/// Returns null if `model` is null.
///
/// # Safety
///
/// `model` must be null OR a valid pointer to a `BlazenVcModel` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_list_target_voices(
    model: *const BlazenVcModel,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model = unsafe { &*model };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<Vec<InnerTargetVoice>, _>(async move {
        Arc::clone(&inner).list_target_voices().await
    })
}

// ---------------------------------------------------------------------------
// register_target_voice — sync + async
// ---------------------------------------------------------------------------

/// Synchronously register a new target voice for the backend, sourcing
/// its identity from the reference utterance at `reference_audio_path`.
///
/// Returns `0` on success, `-1` on backend failure, `-2` on invalid input
/// (null model / null or non-UTF-8 voice id / null or non-UTF-8
/// reference path).
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenVcModel` produced by the
///   cabi surface.
/// - `voice_id` / `reference_audio_path` must each be valid
///   NUL-terminated UTF-8 buffers.
/// - `out_err` must be null OR a writable slot for a single
///   `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_register_target_voice_blocking(
    model: *const BlazenVcModel,
    voice_id: *const c_char,
    reference_audio_path: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_vc_model_register_target_voice: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `voice_id`.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        write_internal_error(
            out_err,
            "blazen_vc_model_register_target_voice: null or non-UTF-8 voice_id",
        );
        return -2;
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `reference_audio_path`.
    let Some(reference_audio_path) = (unsafe { cstr_to_str(reference_audio_path) }) else {
        write_internal_error(
            out_err,
            "blazen_vc_model_register_target_voice: null or non-UTF-8 reference_audio_path",
        );
        return -2;
    };
    let reference_audio_path = reference_audio_path.to_owned();

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move {
        Arc::clone(&inner)
            .register_target_voice(voice_id, reference_audio_path)
            .await
    });
    match result {
        Ok(()) => 0,
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously register a new target voice for the backend, sourcing
/// its identity from the reference utterance at `reference_audio_path`.
/// Returns a `*mut BlazenFuture` whose result is popped with
/// [`crate::persist::blazen_future_take_unit`].
///
/// Returns null if `model` is null or either string is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as
/// [`blazen_vc_model_register_target_voice_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_model_register_target_voice(
    model: *const BlazenVcModel,
    voice_id: *const c_char,
    reference_audio_path: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenVcModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `voice_id`.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        return std::ptr::null_mut();
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `reference_audio_path`.
    let Some(reference_audio_path) = (unsafe { cstr_to_str(reference_audio_path) }) else {
        return std::ptr::null_mut();
    };
    let reference_audio_path = reference_audio_path.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<(), _>(async move {
        Arc::clone(&inner)
            .register_target_voice(voice_id, reference_audio_path)
            .await
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_uniffi::compute_vc::{
        TargetVoice as InnerTargetVoiceForTest, VcChunk as InnerVcChunkForTest,
        VcResult as InnerVcResultForTest,
    };

    use crate::compute_records::{
        BlazenTargetVoice, BlazenTargetVoiceList, BlazenVcChunk, BlazenVcResult,
    };

    /// Plumbing test: the RVC factory should not touch the voice dir or
    /// load weights during construction, so a CPU device produces a live
    /// model handle that `blazen_vc_model_free` cleans up without leaking.
    #[cfg(feature = "audio-vc-rvc")]
    #[test]
    fn rvc_vc_model_new_returns_nonnull_handle() {
        let device = std::ffi::CString::new("cpu").unwrap();
        let mut model: *mut BlazenVcModel = std::ptr::null_mut();
        let mut err: *mut BlazenError = std::ptr::null_mut();
        // SAFETY: device is a live NUL-terminated buffer; out-params are
        // writable stack slots; voice_dir is null (factory leaves env var
        // untouched).
        let rc = unsafe {
            blazen_vc_model_new_rvc(
                std::ptr::null(),
                device.as_ptr(),
                &raw mut model,
                &raw mut err,
            )
        };
        assert_eq!(rc, 0, "expected ok, got rc={rc}, err={:?}", err.is_null());
        assert!(!model.is_null(), "expected non-null model handle");
        assert!(err.is_null(), "expected no error on success");
        // SAFETY: model came from the factory above; freeing through the
        // matching `blazen_vc_model_free` is exactly the documented
        // ownership contract.
        unsafe {
            blazen_vc_model_free(model);
        }
    }

    /// `BlazenVcChunk` round-trips the `samples`/`is_final`/`latency`
    /// fields through the C accessor functions.
    #[test]
    fn blazen_vc_chunk_round_trips_accessors() {
        let inner = InnerVcChunkForTest {
            samples: vec![0.0_f32, 0.25, -0.5, 1.0, -1.0],
            is_final: true,
            latency_seconds: Some(0.125),
        };
        let chunk = BlazenVcChunk::from(inner).into_ptr();

        let mut len: usize = 0;
        // SAFETY: `chunk` is a live cabi handle; `len` is a writable stack slot.
        let ptr = unsafe { crate::compute_records::blazen_vc_chunk_samples(chunk, &raw mut len) };
        assert!(!ptr.is_null(), "samples ptr must be non-null");
        assert_eq!(len, 5);
        // SAFETY: ptr/len describe a live `Vec<f32>` borrowed from the chunk.
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert_eq!(slice, &[0.0_f32, 0.25, -0.5, 1.0, -1.0]);

        // SAFETY: live chunk pointer.
        assert!(unsafe { crate::compute_records::blazen_vc_chunk_is_final(chunk) });
        // SAFETY: live chunk pointer.
        let latency = unsafe { crate::compute_records::blazen_vc_chunk_latency_seconds(chunk) };
        assert!((latency - 0.125).abs() < f32::EPSILON);

        // SAFETY: `chunk` came from `into_ptr` above; freeing through the
        // documented free function is exactly the ownership contract.
        unsafe {
            crate::compute_records::blazen_vc_chunk_free(chunk);
        }
    }

    /// `BlazenVcResult` round-trips bytes / mime / `sample_rate` / duration
    /// through the C accessor functions.
    #[test]
    fn blazen_vc_result_round_trips_accessors() {
        let inner = InnerVcResultForTest {
            bytes: vec![0x52, 0x49, 0x46, 0x46], // "RIFF"
            mime_type: "audio/wav".to_string(),
            sample_rate: 40_000,
            duration_seconds: 2.25,
        };
        let result = BlazenVcResult::from(inner).into_ptr();

        let mut len: usize = 0;
        // SAFETY: `result` is a live cabi handle; `len` is a writable stack slot.
        let bytes_ptr =
            unsafe { crate::compute_records::blazen_vc_result_bytes(result, &raw mut len) };
        assert!(!bytes_ptr.is_null());
        assert_eq!(len, 4);
        // SAFETY: ptr/len describe a live `Vec<u8>` borrowed from the result.
        let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, len) };
        assert_eq!(bytes, &[0x52_u8, 0x49, 0x46, 0x46]);

        // SAFETY: live result pointer.
        let mime = unsafe { crate::compute_records::blazen_vc_result_mime_type(result) };
        assert!(!mime.is_null());
        // SAFETY: pointer minted by `alloc_cstring` above — valid NUL-terminated
        // UTF-8 we can recover via `CStr`.
        let mime_str = unsafe { std::ffi::CStr::from_ptr(mime).to_str().unwrap().to_owned() };
        assert_eq!(mime_str, "audio/wav");
        // SAFETY: `mime` was minted by `alloc_cstring` (which calls
        // `CString::into_raw`); free via the matching `blazen_string_free`.
        unsafe { crate::string::blazen_string_free(mime) };

        // SAFETY: live result pointer for the scalar accessors.
        assert_eq!(
            unsafe { crate::compute_records::blazen_vc_result_sample_rate(result) },
            40_000
        );
        let duration = unsafe { crate::compute_records::blazen_vc_result_duration_seconds(result) };
        assert!((duration - 2.25).abs() < f32::EPSILON);

        // SAFETY: `result` came from `into_ptr` above; free via the matching
        // free function.
        unsafe {
            crate::compute_records::blazen_vc_result_free(result);
        }
    }

    /// `BlazenTargetVoice` round-trips id / label / sample-rate through
    /// the C accessor functions; `label` collapses `None` to an empty
    /// string per the cabi convention.
    #[test]
    fn blazen_target_voice_round_trips_accessors() {
        // With label present.
        let inner = InnerTargetVoiceForTest {
            id: "speaker-01".to_string(),
            label: Some("Alice".to_string()),
            sample_rate_hz: 40_000,
        };
        let voice = BlazenTargetVoice::from(inner).into_ptr();

        // SAFETY: live voice pointer.
        let id = unsafe { crate::compute_records::blazen_target_voice_id(voice) };
        assert!(!id.is_null());
        // SAFETY: pointer minted by `alloc_cstring`.
        let id_str = unsafe { std::ffi::CStr::from_ptr(id).to_str().unwrap().to_owned() };
        assert_eq!(id_str, "speaker-01");
        // SAFETY: `id` was minted by `alloc_cstring`.
        unsafe { crate::string::blazen_string_free(id) };

        // SAFETY: live voice pointer.
        let label = unsafe { crate::compute_records::blazen_target_voice_label(voice) };
        assert!(!label.is_null());
        // SAFETY: pointer minted by `alloc_cstring`.
        let label_str = unsafe { std::ffi::CStr::from_ptr(label).to_str().unwrap().to_owned() };
        assert_eq!(label_str, "Alice");
        // SAFETY: minted by `alloc_cstring`.
        unsafe { crate::string::blazen_string_free(label) };

        // SAFETY: live voice pointer.
        assert_eq!(
            unsafe { crate::compute_records::blazen_target_voice_sample_rate_hz(voice) },
            40_000
        );

        // SAFETY: `voice` came from `into_ptr` above.
        unsafe {
            crate::compute_records::blazen_target_voice_free(voice);
        }

        // With label absent — should collapse to empty string, not null.
        let inner_no_label = InnerTargetVoiceForTest {
            id: "speaker-02".to_string(),
            label: None,
            sample_rate_hz: 16_000,
        };
        let voice_no_label = BlazenTargetVoice::from(inner_no_label).into_ptr();

        // SAFETY: live voice pointer.
        let label = unsafe { crate::compute_records::blazen_target_voice_label(voice_no_label) };
        assert!(!label.is_null(), "label must be empty string, not null");
        // SAFETY: pointer minted by `alloc_cstring`.
        let label_str = unsafe { std::ffi::CStr::from_ptr(label).to_str().unwrap().to_owned() };
        assert_eq!(label_str, "");
        // SAFETY: minted by `alloc_cstring`.
        unsafe { crate::string::blazen_string_free(label) };

        // SAFETY: came from `into_ptr` above.
        unsafe {
            crate::compute_records::blazen_target_voice_free(voice_no_label);
        }
    }

    /// `BlazenTargetVoiceList` iteration round-trips via `_len`, `_get`,
    /// and `_take`.
    #[test]
    fn blazen_target_voice_list_iteration() {
        let items = vec![
            InnerTargetVoiceForTest {
                id: "v0".to_string(),
                label: Some("Voice 0".to_string()),
                sample_rate_hz: 16_000,
            },
            InnerTargetVoiceForTest {
                id: "v1".to_string(),
                label: None,
                sample_rate_hz: 40_000,
            },
        ];
        let list = BlazenTargetVoiceList::from(items).into_ptr();

        // SAFETY: live list pointer.
        assert_eq!(
            unsafe { crate::compute_records::blazen_target_voice_list_len(list) },
            2
        );

        // Borrow entry 0 via `_get` (read-only, do not free).
        // SAFETY: live list pointer.
        let entry0 = unsafe { crate::compute_records::blazen_target_voice_list_get(list, 0) };
        assert!(!entry0.is_null());
        // SAFETY: borrow valid for list's lifetime.
        let id0 = unsafe { crate::compute_records::blazen_target_voice_id(entry0) };
        // SAFETY: minted by `alloc_cstring`.
        let id0_str = unsafe { std::ffi::CStr::from_ptr(id0).to_str().unwrap().to_owned() };
        assert_eq!(id0_str, "v0");
        // SAFETY: minted by `alloc_cstring`.
        unsafe { crate::string::blazen_string_free(id0) };

        // Out-of-range `_get` returns null.
        // SAFETY: live list pointer.
        let oob = unsafe { crate::compute_records::blazen_target_voice_list_get(list, 99) };
        assert!(oob.is_null());

        // Pop entry 1 via `_take` — caller now owns the handle.
        // SAFETY: live list pointer.
        let popped = unsafe { crate::compute_records::blazen_target_voice_list_take(list, 1) };
        assert!(!popped.is_null());
        // SAFETY: live popped pointer.
        assert_eq!(
            unsafe { crate::compute_records::blazen_target_voice_sample_rate_hz(popped) },
            40_000
        );
        // SAFETY: caller owns the popped handle.
        unsafe {
            crate::compute_records::blazen_target_voice_free(popped);
        }

        // After popping, list length is 1 (we removed index 1 from a
        // length-2 list, leaving index 0).
        // SAFETY: live list pointer.
        assert_eq!(
            unsafe { crate::compute_records::blazen_target_voice_list_len(list) },
            1
        );

        // Out-of-range `_take` returns null.
        // SAFETY: live list pointer.
        let oob = unsafe { crate::compute_records::blazen_target_voice_list_take(list, 99) };
        assert!(oob.is_null());

        // SAFETY: list came from `into_ptr` above.
        unsafe {
            crate::compute_records::blazen_target_voice_list_free(list);
        }
    }
}
