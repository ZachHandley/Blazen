//! Per-engine TTS provider opaques + C ABI factories.
//!
//! Each engine wraps an `Arc<blazen_uniffi::concrete::tts::<Engine>Provider>`
//! in a `Blazen<Engine>Provider` opaque, exposing:
//!
//! - `blazen_<engine>_provider_new[_<variant>]` -- factory(ies)
//! - `blazen_<engine>_provider_synthesize` -- async future returning a
//!   `*mut BlazenFuture` whose typed result is popped with
//!   [`crate::compute::blazen_future_take_tts_result`]
//! - `blazen_<engine>_provider_synthesize_blocking` -- synchronous variant
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
//! - The futures and `BlazenTtsResult`s produced here share the existing
//!   [`crate::compute::blazen_future_take_tts_result`] /
//!   [`crate::compute_records::blazen_tts_result_free`] pipeline used by the
//!   central [`crate::compute::BlazenTtsModel`] — there is no per-engine
//!   result type.
//!
//! ## Optional inputs
//!
//! All `Option<String>` parameters on the Rust side accept `*const c_char`
//! and treat null as `None`. The Piper `default_speaker_id` (`Option<i64>`)
//! uses `-1` as the `None` sentinel. Kokoro / `VibeVoice` / Qwen3 `sample_rate`
//! (`Option<u32>`) uses `-1` (or any negative `i32`) as the `None` sentinel —
//! matching the convention established in [`crate::compute_factories`].
//!
//! ## Relationship to the central [`crate::compute::BlazenTtsModel`]
//!
//! The central `BlazenTtsModel` + `blazen_tts_model_new_*` factories in
//! [`crate::compute`] / [`crate::compute_factories`] remain in place — this
//! module is purely additive. Foreign hosts (Ruby, future Dart / Crystal /
//! Lua / PHP) can migrate to the per-engine surface incrementally without
//! breaking the existing Ruby gem entry points.

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute::TtsResult as InnerTtsResult;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local helpers (mirroring crate::compute_factories)
// ---------------------------------------------------------------------------

/// Convert a C `i32` Option-as-sentinel into the equivalent `Option<u32>`.
///
/// Any negative value collapses to `None`; non-negative values pass through
/// as `Some(v as u32)`. Matches the encoding documented in
/// [`crate::compute_factories`].
#[inline]
fn opt_u32_from_i32(v: i32) -> Option<u32> {
    if v < 0 {
        None
    } else {
        Some(u32::try_from(v).unwrap_or(0))
    }
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
// PiperProvider — local Piper ONNX (feature = "audio-tts-piper")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::PiperProvider>`.
///
/// Free with [`blazen_piper_provider_free`].
#[cfg(feature = "audio-tts-piper")]
pub struct BlazenPiperProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::PiperProvider>);

#[cfg(feature = "audio-tts-piper")]
impl BlazenPiperProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenPiperProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenPiperProvider` from a Piper voice id + already-resolved
/// local file paths.
///
/// `voice_id` and `onnx_path` are required (non-null UTF-8). `config_path`
/// may be null to derive the sidecar path automatically (`.onnx.json`).
/// `default_speaker_id` uses `-1` as the `None` sentinel (any other negative
/// value also collapses to `None`).
///
/// On success returns `0` and writes a fresh `BlazenPiperProvider*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Both out-parameters may be null to discard that side of
/// the result.
///
/// # Safety
///
/// - `voice_id`, `onnx_path` must be valid NUL-terminated UTF-8 buffers
///   (non-null).
/// - `config_path` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[cfg(feature = "audio-tts-piper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_piper_provider_new(
    voice_id: *const c_char,
    onnx_path: *const c_char,
    config_path: *const c_char,
    default_speaker_id: i64,
    out_model: *mut *mut BlazenPiperProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(voice_id) = (unsafe { cstr_to_str(voice_id) }) else {
        return write_internal_error(
            out_err,
            "blazen_piper_provider_new: voice_id must not be null",
        );
    };
    let voice_id = voice_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(onnx_path) = (unsafe { cstr_to_str(onnx_path) }) else {
        return write_internal_error(
            out_err,
            "blazen_piper_provider_new: onnx_path must not be null",
        );
    };
    let onnx_path = onnx_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let config_path = unsafe { cstr_to_opt_string(config_path) };
    let default_speaker_id = if default_speaker_id < 0 {
        None
    } else {
        Some(default_speaker_id)
    };

    match blazen_uniffi::concrete::tts::PiperProvider::new(
        voice_id,
        onnx_path,
        config_path,
        default_speaker_id,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenPiperProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously synthesize speech for `text`. Returns a `*mut BlazenFuture`
/// the caller polls / waits on; the typed result is popped with
/// [`crate::compute::blazen_future_take_tts_result`].
///
/// Returns null if `model` is null or `text` is null / non-UTF-8.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenPiperProvider` produced by
///   this surface.
/// - `text` must be a valid NUL-terminated UTF-8 buffer.
/// - `voice` / `language` must be null OR valid NUL-terminated UTF-8 buffers.
#[cfg(feature = "audio-tts-piper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_piper_provider_synthesize(
    model: *const BlazenPiperProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenPiperProvider`.
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

/// Synchronous variant of [`blazen_piper_provider_synthesize`]. Returns `0`
/// on success (typed result in `*out_result`), `-1` on failure (error in
/// `*out_err`), `-2` on invalid input (null model / non-UTF-8 `text`).
///
/// # Safety
///
/// Same string contracts as [`blazen_piper_provider_synthesize`]. `out_result`
/// / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "audio-tts-piper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_piper_provider_synthesize_blocking(
    model: *const BlazenPiperProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenPiperProvider`.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenPiperProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_piper_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "audio-tts-piper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_piper_provider_free(model: *mut BlazenPiperProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// KokoroProvider — local Kokoro-82M (feature = "tts")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::KokoroProvider>`.
#[cfg(feature = "tts")]
pub struct BlazenKokoroProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::KokoroProvider>);

#[cfg(feature = "tts")]
impl BlazenKokoroProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenKokoroProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenKokoroProvider` with optional defaults.
///
/// All inputs are optional — pass null for `voice` / `language` and any
/// negative value for `sample_rate` to defer to the engine's defaults.
///
/// # Safety
///
/// `voice` / `language` must be null OR valid NUL-terminated UTF-8 buffers.
/// `out_model` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_kokoro_provider_new(
    voice: *const c_char,
    language: *const c_char,
    sample_rate: i32,
    out_model: *mut *mut BlazenKokoroProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let voice = unsafe { cstr_to_opt_string(voice) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };
    let sample_rate = opt_u32_from_i32(sample_rate);

    match blazen_uniffi::concrete::tts::KokoroProvider::new(voice, language, sample_rate) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenKokoroProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously synthesize speech. See
/// [`blazen_piper_provider_synthesize`] for the future / `take_tts_result`
/// pipeline.
///
/// # Safety
///
/// Same string contracts as [`blazen_piper_provider_synthesize`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_kokoro_provider_synthesize(
    model: *const BlazenKokoroProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_kokoro_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_kokoro_provider_synthesize_blocking(
    model: *const BlazenKokoroProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenKokoroProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_kokoro_provider_free(model: *mut BlazenKokoroProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// VibeVoiceProvider — local VibeVoice (feature = "tts")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::VibeVoiceProvider>`.
#[cfg(feature = "tts")]
pub struct BlazenVibeVoiceProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::VibeVoiceProvider>);

#[cfg(feature = "tts")]
impl BlazenVibeVoiceProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenVibeVoiceProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenVibeVoiceProvider`. See
/// [`blazen_kokoro_provider_new`] for argument conventions.
///
/// # Safety
///
/// Same contracts as [`blazen_kokoro_provider_new`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vibevoice_provider_new(
    voice: *const c_char,
    language: *const c_char,
    sample_rate: i32,
    out_model: *mut *mut BlazenVibeVoiceProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let voice = unsafe { cstr_to_opt_string(voice) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };
    let sample_rate = opt_u32_from_i32(sample_rate);

    match blazen_uniffi::concrete::tts::VibeVoiceProvider::new(voice, language, sample_rate) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenVibeVoiceProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously synthesize speech. See
/// [`blazen_piper_provider_synthesize`] for the future pipeline.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vibevoice_provider_synthesize(
    model: *const BlazenVibeVoiceProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_vibevoice_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vibevoice_provider_synthesize_blocking(
    model: *const BlazenVibeVoiceProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenVibeVoiceProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vibevoice_provider_free(model: *mut BlazenVibeVoiceProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// Qwen3TtsProvider — local Qwen3-TTS (feature = "tts")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::Qwen3TtsProvider>`.
#[cfg(feature = "tts")]
pub struct BlazenQwen3TtsProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::Qwen3TtsProvider>);

#[cfg(feature = "tts")]
impl BlazenQwen3TtsProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenQwen3TtsProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenQwen3TtsProvider`. See
/// [`blazen_kokoro_provider_new`] for argument conventions.
///
/// # Safety
///
/// Same contracts as [`blazen_kokoro_provider_new`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_qwen3_tts_provider_new(
    voice: *const c_char,
    language: *const c_char,
    sample_rate: i32,
    out_model: *mut *mut BlazenQwen3TtsProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let voice = unsafe { cstr_to_opt_string(voice) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };
    let sample_rate = opt_u32_from_i32(sample_rate);

    match blazen_uniffi::concrete::tts::Qwen3TtsProvider::new(voice, language, sample_rate) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenQwen3TtsProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously synthesize speech.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_qwen3_tts_provider_synthesize(
    model: *const BlazenQwen3TtsProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_qwen3_tts_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_qwen3_tts_provider_synthesize_blocking(
    model: *const BlazenQwen3TtsProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenQwen3TtsProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[cfg(feature = "tts")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_qwen3_tts_provider_free(model: *mut BlazenQwen3TtsProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// SparkTtsProvider — local Spark-TTS (feature = "audio-tts-spark")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::SparkTtsProvider>`.
#[cfg(feature = "audio-tts-spark")]
pub struct BlazenSparkTtsProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::SparkTtsProvider>);

#[cfg(feature = "audio-tts-spark")]
impl BlazenSparkTtsProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenSparkTtsProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenSparkTtsProvider`. All three inputs are optional —
/// pass null to defer to the upstream defaults.
///
/// Unlike most engines this constructor is infallible (the underlying Rust
/// constructor returns `Arc<Self>` directly), so it returns the handle by
/// pointer rather than an out-param. Returns `null` only when one of the
/// optional input strings is non-null but contains non-UTF-8 bytes.
///
/// # Safety
///
/// `model_id` / `model_dir` / `revision` must be null OR valid NUL-terminated
/// UTF-8 buffers.
#[cfg(feature = "audio-tts-spark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_spark_tts_provider_new(
    model_id: *const c_char,
    model_dir: *const c_char,
    revision: *const c_char,
) -> *mut BlazenSparkTtsProvider {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_dir = unsafe { cstr_to_opt_string(model_dir) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let revision = unsafe { cstr_to_opt_string(revision) };

    let arc = blazen_uniffi::concrete::tts::SparkTtsProvider::new(model_id, model_dir, revision);
    BlazenSparkTtsProvider(arc).into_ptr()
}

/// Asynchronously synthesize speech.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize`].
#[cfg(feature = "audio-tts-spark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_spark_tts_provider_synthesize(
    model: *const BlazenSparkTtsProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_spark_tts_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[cfg(feature = "audio-tts-spark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_spark_tts_provider_synthesize_blocking(
    model: *const BlazenSparkTtsProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenSparkTtsProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[cfg(feature = "audio-tts-spark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_spark_tts_provider_free(model: *mut BlazenSparkTtsProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// BarkProvider — local Bark (feature = "audio-tts-bark")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::BarkProvider>`.
#[cfg(feature = "audio-tts-bark")]
pub struct BlazenBarkProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::BarkProvider>);

#[cfg(feature = "audio-tts-bark")]
impl BlazenBarkProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenBarkProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenBarkProvider` with default configuration. Infallible.
///
/// # Safety
///
/// Always safe to call; returns a non-null heap-allocated handle.
#[cfg(feature = "audio-tts-bark")]
#[unsafe(no_mangle)]
pub extern "C" fn blazen_bark_provider_new() -> *mut BlazenBarkProvider {
    let arc = blazen_uniffi::concrete::tts::BarkProvider::new();
    BlazenBarkProvider(arc).into_ptr()
}

/// Asynchronously synthesize speech.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize`].
#[cfg(feature = "audio-tts-bark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bark_provider_synthesize(
    model: *const BlazenBarkProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_bark_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[cfg(feature = "audio-tts-bark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bark_provider_synthesize_blocking(
    model: *const BlazenBarkProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenBarkProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[cfg(feature = "audio-tts-bark")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bark_provider_free(model: *mut BlazenBarkProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// F5Provider — local F5-TTS (feature = "audio-tts-f5")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::F5Provider>`.
#[cfg(feature = "audio-tts-f5")]
pub struct BlazenF5Provider(pub(crate) Arc<blazen_uniffi::concrete::tts::F5Provider>);

#[cfg(feature = "audio-tts-f5")]
impl BlazenF5Provider {
    pub(crate) fn into_ptr(self) -> *mut BlazenF5Provider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenF5Provider` with default configuration. Infallible.
///
/// # Safety
///
/// Always safe to call; returns a non-null heap-allocated handle.
#[cfg(feature = "audio-tts-f5")]
#[unsafe(no_mangle)]
pub extern "C" fn blazen_f5_provider_new() -> *mut BlazenF5Provider {
    let arc = blazen_uniffi::concrete::tts::F5Provider::new();
    BlazenF5Provider(arc).into_ptr()
}

/// Asynchronously synthesize speech.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize`].
#[cfg(feature = "audio-tts-f5")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_f5_provider_synthesize(
    model: *const BlazenF5Provider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_f5_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[cfg(feature = "audio-tts-f5")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_f5_provider_synthesize_blocking(
    model: *const BlazenF5Provider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenF5Provider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[cfg(feature = "audio-tts-f5")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_f5_provider_free(model: *mut BlazenF5Provider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// FalTtsProvider — fal.ai hosted (no feature gate)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::tts::FalTtsProvider>`.
pub struct BlazenFalTtsProvider(pub(crate) Arc<blazen_uniffi::concrete::tts::FalTtsProvider>);

impl BlazenFalTtsProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalTtsProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFalTtsProvider` from a fal.ai API key.
///
/// An empty `api_key` falls back to the `FAL_KEY` environment variable.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null;
///   empty string is allowed).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_tts_provider_new(
    api_key: *const c_char,
    out_model: *mut *mut BlazenFalTtsProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_tts_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();

    match blazen_uniffi::concrete::tts::FalTtsProvider::new(api_key) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenFalTtsProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Construct a `BlazenFalTtsProvider` with an explicit default fal TTS
/// endpoint (e.g. `"fal-ai/dia-tts"`). Pass `default_model` as null to leave
/// the upstream default in place.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `default_model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_tts_provider_new_with_model(
    api_key: *const c_char,
    default_model: *const c_char,
    out_model: *mut *mut BlazenFalTtsProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_tts_provider_new_with_model: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let default_model = unsafe { cstr_to_opt_string(default_model) };

    match blazen_uniffi::concrete::tts::FalTtsProvider::with_model(api_key, default_model) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenFalTtsProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously synthesize speech.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_tts_provider_synthesize(
    model: *const BlazenFalTtsProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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

/// Synchronous variant of [`blazen_fal_tts_provider_synthesize`].
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_synthesize_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_tts_provider_synthesize_blocking(
    model: *const BlazenFalTtsProvider,
    text: *const c_char,
    voice: *const c_char,
    language: *const c_char,
    out_result: *mut *mut crate::compute_records::BlazenTtsResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
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
                    *out_result = crate::compute_records::BlazenTtsResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenFalTtsProvider`. No-op on null.
///
/// # Safety
///
/// Same contracts as [`blazen_piper_provider_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_tts_provider_free(model: *mut BlazenFalTtsProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
