//! Per-engine STT provider opaques + C ABI factories.
//!
//! Each engine wraps an `Arc<blazen_uniffi::concrete::stt::<Engine>Provider>`
//! in a `Blazen<Engine>Provider` opaque, exposing:
//!
//! - `blazen_<engine>_provider_new` -- factory
//! - `blazen_<engine>_provider_transcribe` -- async future returning a
//!   `*mut BlazenFuture` whose typed result is popped with
//!   [`crate::compute::blazen_future_take_stt_result`]
//! - `blazen_<engine>_provider_transcribe_blocking` -- synchronous variant
//! - `blazen_<engine>_provider_free` -- destructor (no-op on null)
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<Blazen<Engine>Provider>` returned
//!   by the factory functions. Callers free with the matching `*_free`.
//!   Double-free is undefined behavior.
//! - String inputs are borrowed for the duration of the call only — the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.
//! - The futures and `BlazenSttResult`s produced here share the existing
//!   [`crate::compute::blazen_future_take_stt_result`] /
//!   [`crate::compute_records::blazen_stt_result_free`] pipeline used by the
//!   central [`crate::compute::BlazenSttModel`] — there is no per-engine
//!   result type.
//!
//! ## Optional inputs
//!
//! All `Option<String>` parameters on the Rust side accept `*const c_char`
//! and treat null as `None`. `WhisperStreamingProvider`'s `chunk_seconds` /
//! `chunk_overlap_seconds` (`Option<f32>`) use NaN as the `None` sentinel
//! (any non-finite value collapses to `None`).
//!
//! ## Feature gates
//!
//! The whole upstream [`blazen_uniffi::concrete::stt`] module is gated on
//! `whispercpp`, so every engine here carries that gate at minimum. Two
//! engines layer additional gates on top:
//!
//! - [`BlazenFasterWhisperProvider`] — `audio-stt-faster-whisper`
//! - [`BlazenWhisperStreamingProvider`] — `audio-stt-whisper-streaming`
//!
//! ## Relationship to the central [`crate::compute::BlazenSttModel`]
//!
//! The central `BlazenSttModel` + `blazen_stt_model_new_*` factories in
//! [`crate::compute`] / [`crate::compute_factories`] remain in place — this
//! module is purely additive. Foreign hosts (Ruby, future Dart / Crystal /
//! Lua / PHP) can migrate to the per-engine surface incrementally without
//! breaking the existing Ruby gem entry points.

#![cfg(feature = "whispercpp")]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute::SttResult as InnerSttResult;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// Convert an `f32` Option-as-sentinel into the equivalent `Option<f32>`.
///
/// NaN (and any non-finite value) collapses to `None`; finite values pass
/// through as `Some(v)`.
#[inline]
#[cfg(feature = "audio-stt-whisper-streaming")]
fn opt_f32_from_f32(v: f32) -> Option<f32> {
    if v.is_finite() { Some(v) } else { None }
}

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

/// Builds + writes a `BlazenError::Internal { message }` into `out_err`. Used
/// for argument-shape errors (null required pointers, non-UTF-8 strings) that
/// don't originate from a `blazen_uniffi` call.
fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) -> i32 {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    )
}

// ---------------------------------------------------------------------------
// WhisperCppProvider — local whisper.cpp (feature = "whispercpp")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::stt::WhisperCppProvider>`.
///
/// Free with [`blazen_whispercpp_provider_free`].
pub struct BlazenWhisperCppProvider(
    pub(crate) Arc<blazen_uniffi::concrete::stt::WhisperCppProvider>,
);

impl BlazenWhisperCppProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenWhisperCppProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenWhisperCppProvider`.
///
/// All three inputs are optional — pass null to defer to the engine's
/// defaults. `model` selects the whisper.cpp variant (e.g. `"tiny"`,
/// `"base"`, `"small"`, `"medium"`, `"large-v3"`). `device` picks the
/// runtime device (`"cpu"`, `"cuda"`, ...). `language` is an optional
/// ISO-639-1 default-language hint.
///
/// On success returns `0` and writes a fresh `BlazenWhisperCppProvider*`
/// into `*out_model`. On failure returns `-1` and writes a fresh
/// `BlazenError*` into `*out_err`. Both out-parameters may be null to
/// discard that side of the result.
///
/// # Safety
///
/// - `model` / `device` / `language` must each be null OR a valid
///   NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whispercpp_provider_new(
    model: *const c_char,
    device: *const c_char,
    language: *const c_char,
    out_model: *mut *mut BlazenWhisperCppProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };

    match blazen_uniffi::concrete::stt::WhisperCppProvider::new(model, device, language) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenWhisperCppProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously transcribe audio at `audio_source`. Returns a
/// `*mut BlazenFuture` the caller polls / waits on; the typed result is
/// popped with [`crate::compute::blazen_future_take_stt_result`].
///
/// `audio_source` is a local file path (16-bit PCM mono WAV at 16 kHz) or
/// an `http(s)://` / `data:` URL. `language` is an optional per-call
/// ISO-639-1 override; when null the constructor's `language` hint (if
/// any) is used.
///
/// Returns null if `model` is null or `audio_source` is null / non-UTF-8.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenWhisperCppProvider`
///   produced by this surface.
/// - `audio_source` must be a valid NUL-terminated UTF-8 buffer.
/// - `language` must be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whispercpp_provider_transcribe(
    model: *const BlazenWhisperCppProvider,
    audio_source: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenWhisperCppProvider`.
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

/// Synchronous variant of [`blazen_whispercpp_provider_transcribe`].
/// Returns `0` on success (typed result in `*out_result`), `-1` on failure
/// (error in `*out_err`), `-2` on invalid input (null model / non-UTF-8
/// `audio_source`).
///
/// # Safety
///
/// Same string contracts as [`blazen_whispercpp_provider_transcribe`].
/// `out_result` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whispercpp_provider_transcribe_blocking(
    model: *const BlazenWhisperCppProvider,
    audio_source: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenSttResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenSttResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenWhisperCppProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_whispercpp_provider_new`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whispercpp_provider_free(model: *mut BlazenWhisperCppProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// FasterWhisperProvider — local CTranslate2 (feature = "audio-stt-faster-whisper")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::stt::FasterWhisperProvider>`.
#[cfg(feature = "audio-stt-faster-whisper")]
pub struct BlazenFasterWhisperProvider(
    pub(crate) Arc<blazen_uniffi::concrete::stt::FasterWhisperProvider>,
);

#[cfg(feature = "audio-stt-faster-whisper")]
impl BlazenFasterWhisperProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFasterWhisperProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFasterWhisperProvider`. All three inputs are
/// optional — pass null to defer to the engine's defaults.
///
/// `model_id` selects a Hugging Face bundle id (default
/// `"Systran/faster-whisper-tiny"`). `model_dir` provides a pre-resolved
/// local `CTranslate2` bundle directory; when supplied the HF download is
/// skipped. `revision` pins a specific branch / tag / commit on the HF
/// repo (default `"main"`).
///
/// Unlike most engines this constructor is infallible (the underlying Rust
/// constructor returns `Arc<Self>` directly), so it returns the handle by
/// pointer rather than an out-param. Returns null only when one of the
/// input strings is non-null but contains non-UTF-8 bytes.
///
/// # Safety
///
/// `model_id` / `model_dir` / `revision` must each be null OR a valid
/// NUL-terminated UTF-8 buffer.
#[cfg(feature = "audio-stt-faster-whisper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_faster_whisper_provider_new(
    model_id: *const c_char,
    model_dir: *const c_char,
    revision: *const c_char,
) -> *mut BlazenFasterWhisperProvider {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_dir = unsafe { cstr_to_opt_string(model_dir) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let revision = unsafe { cstr_to_opt_string(revision) };

    let arc =
        blazen_uniffi::concrete::stt::FasterWhisperProvider::new(model_id, model_dir, revision);
    BlazenFasterWhisperProvider(arc).into_ptr()
}

/// Asynchronously transcribe audio. See
/// [`blazen_whispercpp_provider_transcribe`] for the future pipeline and
/// `audio_source` / `language` semantics.
///
/// # Safety
///
/// Same contracts as [`blazen_whispercpp_provider_transcribe`].
#[cfg(feature = "audio-stt-faster-whisper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_faster_whisper_provider_transcribe(
    model: *const BlazenFasterWhisperProvider,
    audio_source: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of
/// [`blazen_faster_whisper_provider_transcribe`].
///
/// # Safety
///
/// Same contracts as
/// [`blazen_whispercpp_provider_transcribe_blocking`].
#[cfg(feature = "audio-stt-faster-whisper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_faster_whisper_provider_transcribe_blocking(
    model: *const BlazenFasterWhisperProvider,
    audio_source: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenSttResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenSttResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenFasterWhisperProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_whispercpp_provider_free`].
#[cfg(feature = "audio-stt-faster-whisper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_faster_whisper_provider_free(
    model: *mut BlazenFasterWhisperProvider,
) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// WhisperStreamingProvider — chunked candle Whisper + Silero VAD
// (feature = "audio-stt-whisper-streaming")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::stt::WhisperStreamingProvider>`.
///
/// Note: the underlying backend's blocking `transcribe` entry point
/// returns `Unsupported` — only the streaming surface is functional in
/// the upstream backend. The async / blocking transcribe methods below
/// are exposed for API parity but will surface that error to the caller.
#[cfg(feature = "audio-stt-whisper-streaming")]
pub struct BlazenWhisperStreamingProvider(
    pub(crate) Arc<blazen_uniffi::concrete::stt::WhisperStreamingProvider>,
);

#[cfg(feature = "audio-stt-whisper-streaming")]
impl BlazenWhisperStreamingProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenWhisperStreamingProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenWhisperStreamingProvider`. All four inputs are
/// optional — pass null for `model_id` / `vad_model_path` and `NaN` (or
/// any non-finite value) for `chunk_seconds` / `chunk_overlap_seconds` to
/// defer to the engine's defaults (`"openai/whisper-base"`, on-demand HF
/// download, `30.0`s windows with `5.0`s overlap).
///
/// Infallible — returns the handle directly. Returns null only when one
/// of the input strings is non-null but contains non-UTF-8 bytes.
///
/// # Safety
///
/// `model_id` / `vad_model_path` must each be null OR a valid
/// NUL-terminated UTF-8 buffer.
#[cfg(feature = "audio-stt-whisper-streaming")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whisper_streaming_provider_new(
    model_id: *const c_char,
    vad_model_path: *const c_char,
    chunk_seconds: f32,
    chunk_overlap_seconds: f32,
) -> *mut BlazenWhisperStreamingProvider {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let vad_model_path = unsafe { cstr_to_opt_string(vad_model_path) };
    let chunk_seconds = opt_f32_from_f32(chunk_seconds);
    let chunk_overlap_seconds = opt_f32_from_f32(chunk_overlap_seconds);

    let arc = blazen_uniffi::concrete::stt::WhisperStreamingProvider::new(
        model_id,
        vad_model_path,
        chunk_seconds,
        chunk_overlap_seconds,
    );
    BlazenWhisperStreamingProvider(arc).into_ptr()
}

/// Asynchronously transcribe audio. See
/// [`blazen_whispercpp_provider_transcribe`] for the future pipeline and
/// `audio_source` / `language` semantics.
///
/// # Safety
///
/// Same contracts as [`blazen_whispercpp_provider_transcribe`].
#[cfg(feature = "audio-stt-whisper-streaming")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whisper_streaming_provider_transcribe(
    model: *const BlazenWhisperStreamingProvider,
    audio_source: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of
/// [`blazen_whisper_streaming_provider_transcribe`].
///
/// # Safety
///
/// Same contracts as
/// [`blazen_whispercpp_provider_transcribe_blocking`].
#[cfg(feature = "audio-stt-whisper-streaming")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whisper_streaming_provider_transcribe_blocking(
    model: *const BlazenWhisperStreamingProvider,
    audio_source: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenSttResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenSttResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenWhisperStreamingProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_whispercpp_provider_free`].
#[cfg(feature = "audio-stt-whisper-streaming")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_whisper_streaming_provider_free(
    model: *mut BlazenWhisperStreamingProvider,
) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// FalSttProvider — hosted fal.ai Whisper / Wizper
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::stt::FalSttProvider>`.
pub struct BlazenFalSttProvider(pub(crate) Arc<blazen_uniffi::concrete::stt::FalSttProvider>);

impl BlazenFalSttProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalSttProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFalSttProvider` from a fal.ai API key.
///
/// An empty `api_key` falls back to the `FAL_KEY` environment variable
/// at call time. Construction is infallible — returns `0` and writes the
/// handle into `*out_model` unconditionally (the `out_err` slot is left
/// alone). The error tail is preserved for parity with the other STT
/// constructors and future expansion.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null;
///   empty string is allowed).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_stt_provider_new(
    api_key: *const c_char,
    out_model: *mut *mut BlazenFalSttProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_stt_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();

    let arc = blazen_uniffi::concrete::stt::FalSttProvider::new(api_key);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFalSttProvider(arc).into_ptr();
        }
    }
    0
}

/// Asynchronously transcribe audio. See
/// [`blazen_whispercpp_provider_transcribe`] for the future pipeline and
/// `audio_source` / `language` semantics.
///
/// `audio_source` should be an `http(s)://` URL or a `data:` URI reachable
/// by fal.ai's workers (local paths are not supported by the hosted
/// backend).
///
/// # Safety
///
/// Same contracts as [`blazen_whispercpp_provider_transcribe`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_stt_provider_transcribe(
    model: *const BlazenFalSttProvider,
    audio_source: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_fal_stt_provider_transcribe`].
///
/// # Safety
///
/// Same contracts as
/// [`blazen_whispercpp_provider_transcribe_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_stt_provider_transcribe_blocking(
    model: *const BlazenFalSttProvider,
    audio_source: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenSttResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenSttResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenFalSttProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_whispercpp_provider_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_stt_provider_free(model: *mut BlazenFalSttProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
