//! Music + sound-effect generation surface for the C ABI.
//!
//! Mirrors [`crate::compute`] + [`crate::compute_factories`] for the upstream
//! [`blazen_uniffi::compute_music`] module:
//!
//! - [`BlazenMusicModel`] — opaque handle wrapping `Arc<MusicModel>`.
//! - Four per-backend factories: [`blazen_music_model_new_fal`] (always on),
//!   [`blazen_music_model_new_musicgen`] (feature `audio-music-musicgen`),
//!   [`blazen_music_model_new_stable_audio`] (feature
//!   `audio-music-stable-audio`), and [`blazen_music_model_new_audiogen`]
//!   (feature `audio-music-audiogen`).
//! - Non-streaming generate (sync + async, music + SFX) wrappers that route
//!   through the cabi tokio runtime.
//! - Typed future-take entry point [`blazen_future_take_music_result`] for the
//!   async non-streaming variants.
//!
//! Streaming entry points (vtable + pump functions) live in
//! [`crate::stream_sink`] next to the existing `CompletionStreamSink`
//! trampoline.
//!
//! ## Optional-numeric encoding
//!
//! - `Option<f32>` (the `max_duration_seconds` factory arg) uses NaN as the
//!   sentinel for "use the backend default". Any non-NaN value (including
//!   zero or negative) passes through verbatim — `f32::NAN` collapses to
//!   `None` so foreign callers can omit the override cleanly.
//!
//! ## Ownership conventions
//!
//! - Model handles are heap-allocated `Box<BlazenMusicModel>` returned by the
//!   factory functions; caller frees with [`blazen_music_model_free`].
//! - Result handles produced by `*_blocking` and by
//!   [`blazen_future_take_music_result`] are caller-owned and freed via
//!   [`crate::compute_records::blazen_music_result_free`].
//! - Error handles produced on the failure path are caller-owned and freed
//!   via [`crate::error::blazen_error_free`].
//! - String inputs are borrowed for the duration of the call only — the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.

#![allow(dead_code)]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute_music::{
    MusicModel as InnerMusicModel, MusicResult as InnerMusicResult,
};
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::compute_records::BlazenMusicResult;
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local error / sentinel helpers
//
// Mirrors the pattern in `compute_factories.rs` — each module keeps its own
// helpers self-contained so a future split (e.g. shipping music as its own
// cdylib) doesn't drag in unrelated surface area.
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

/// Convert a C `f32` Option-as-sentinel into the equivalent `Option<f32>`.
///
/// NaN is the sentinel for "unset" — any non-NaN value (including zero or
/// negative) passes through as `Some(v)`. Mirrors
/// `crate::compute_factories::opt_f32_from_f32`.
#[inline]
fn opt_f32_from_f32(v: f32) -> Option<f32> {
    if v.is_nan() { None } else { Some(v) }
}

// ---------------------------------------------------------------------------
// BlazenMusicModel handle
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::compute_music::MusicModel>`.
///
/// Produced by the per-backend factory functions
/// ([`blazen_music_model_new_fal`], [`blazen_music_model_new_musicgen`],
/// [`blazen_music_model_new_stable_audio`],
/// [`blazen_music_model_new_audiogen`]). Free with
/// [`blazen_music_model_free`].
pub struct BlazenMusicModel(pub(crate) Arc<InnerMusicModel>);

impl BlazenMusicModel {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenMusicModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerMusicModel>> for BlazenMusicModel {
    fn from(inner: Arc<InnerMusicModel>) -> Self {
        Self(inner)
    }
}

/// Frees a `BlazenMusicModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by the cabi
/// surface's music factory functions. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_free(model: *mut BlazenMusicModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// fal.ai music factory (always available)
// ---------------------------------------------------------------------------

/// Build a fal.ai-backed [`BlazenMusicModel`].
///
/// `api_key` must be a NUL-terminated UTF-8 buffer (empty is allowed — the
/// upstream factory resolves `FAL_KEY` from the environment in that case).
/// `model` may be null to leave the default fal music / SFX endpoint in
/// place; pass a NUL-terminated UTF-8 buffer to override.
///
/// On success returns `0` and writes a fresh `BlazenMusicModel*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Returns `-2` on invalid argument shape (null
/// `api_key`).
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_new_fal(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenMusicModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `api_key`.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_music_model_new_fal: api_key must not be null",
        );
        return -2;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `model`.
    let model = unsafe { cstr_to_opt_string(model) };

    match blazen_uniffi::compute_music::new_fal_music_model(api_key, model) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenMusicModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Native MusicGen factory (feature = "audio-music-musicgen")
// ---------------------------------------------------------------------------

/// Build a native MusicGen-backed [`BlazenMusicModel`].
///
/// `variant` selects the `MusicGen` checkpoint by name (case-insensitive:
/// `"small"`, `"medium"`, `"large"`); pass null to default to `"small"`.
/// `device` follows `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`,
/// `"cuda:N"`, `"metal"`, `"metal:N"`); null defers to the backend's
/// auto-detection. `cache_dir` overrides the Hugging Face Hub cache
/// directory. `max_duration_seconds` follows the NaN-as-`None` encoding
/// (see [`opt_f32_from_f32`]); `None` defaults to 30 s upstream.
///
/// # Safety
///
/// - `variant` / `device` / `cache_dir` must each be null OR a valid
///   NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_new_musicgen(
    variant: *const c_char,
    device: *const c_char,
    cache_dir: *const c_char,
    max_duration_seconds: f32,
    out_model: *mut *mut BlazenMusicModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on each input.
    let variant = unsafe { cstr_to_opt_string(variant) };
    let device = unsafe { cstr_to_opt_string(device) };
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };
    let max_duration_seconds = opt_f32_from_f32(max_duration_seconds);

    match blazen_uniffi::compute_music::new_musicgen_model(
        variant,
        device,
        cache_dir,
        max_duration_seconds,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenMusicModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Native Stable Audio factory (feature = "audio-music-stable-audio")
// ---------------------------------------------------------------------------

/// Build a native Stable Audio Open-backed [`BlazenMusicModel`].
///
/// `variant` selects the Stable Audio Open checkpoint by name
/// (case-insensitive: `"small"`, `"open-1.0"` / `"open1.0"`); pass null to
/// default to `"small"`. `tokenizer_path` is REQUIRED — it must point at
/// the T5 `SentencePiece` `tokenizer.json` shipped with the Stable Audio
/// Open repo (Stable Audio's tokenizer is not auto-downloaded by the
/// backend today). `device` follows the same device-string format as the
/// `MusicGen` factory. `max_duration_seconds` follows the NaN-as-`None`
/// encoding; Stable Audio enforces its own variant-dependent ceiling
/// internally regardless.
///
/// Returns `0` on success, `-1` on a backend init failure, or `-2` on an
/// invalid argument shape (null `tokenizer_path`).
///
/// # Safety
///
/// - `tokenizer_path` must be a valid NUL-terminated UTF-8 buffer
///   (non-null).
/// - `variant` / `device` must each be null OR a valid NUL-terminated UTF-8
///   buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_new_stable_audio(
    variant: *const c_char,
    tokenizer_path: *const c_char,
    device: *const c_char,
    max_duration_seconds: f32,
    out_model: *mut *mut BlazenMusicModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let variant = unsafe { cstr_to_opt_string(variant) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `tokenizer_path`.
    let Some(tokenizer_path) = (unsafe { cstr_to_str(tokenizer_path) }) else {
        write_internal_error(
            out_err,
            "blazen_music_model_new_stable_audio: tokenizer_path must not be null",
        );
        return -2;
    };
    let tokenizer_path = tokenizer_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    let max_duration_seconds = opt_f32_from_f32(max_duration_seconds);

    match blazen_uniffi::compute_music::new_stable_audio_model(
        variant,
        tokenizer_path,
        device,
        max_duration_seconds,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenMusicModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Native AudioGen factory (feature = "audio-music-audiogen")
// ---------------------------------------------------------------------------

/// Build a native AudioGen-backed [`BlazenMusicModel`].
///
/// `repo_id` overrides the default Hugging Face repo
/// (`facebook/audiogen-medium`); pass null to use the default. `revision`
/// pins a specific commit / tag. `device` / `cache_dir` /
/// `max_duration_seconds` follow the `MusicGen` factory's conventions.
///
/// # Safety
///
/// - `repo_id` / `revision` / `device` / `cache_dir` must each be null OR a
///   valid NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_new_audiogen(
    repo_id: *const c_char,
    revision: *const c_char,
    device: *const c_char,
    cache_dir: *const c_char,
    max_duration_seconds: f32,
    out_model: *mut *mut BlazenMusicModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on each input.
    let repo_id = unsafe { cstr_to_opt_string(repo_id) };
    let revision = unsafe { cstr_to_opt_string(revision) };
    let device = unsafe { cstr_to_opt_string(device) };
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };
    let max_duration_seconds = opt_f32_from_f32(max_duration_seconds);

    match blazen_uniffi::compute_music::new_audiogen_model(
        repo_id,
        revision,
        device,
        cache_dir,
        max_duration_seconds,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenMusicModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Non-streaming generate (music) — sync + async
// ---------------------------------------------------------------------------

/// Synchronously generate `duration_seconds` of music conditioned on
/// `prompt` and write the result into `out_result` on success, or a
/// `BlazenError` into `out_err` on failure.
///
/// Returns `0` on success, `-1` on failure, `-2` on invalid input (null
/// model / null or non-UTF-8 `prompt`).
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenMusicModel` produced by
///   the cabi surface.
/// - `prompt` must be a valid NUL-terminated UTF-8 buffer.
/// - `out_result` / `out_err` must each be null OR writable pointers to the
///   appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_generate_music_blocking(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_music_model_generate_music: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        write_internal_error(
            out_err,
            "blazen_music_model_generate_music: null or non-UTF-8 prompt",
        );
        return -2;
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move {
        Arc::clone(&inner)
            .generate_music(prompt, duration_seconds)
            .await
    });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of music conditioned on
/// `prompt`. Returns a `*mut BlazenFuture` the caller polls / waits on;
/// the typed result is popped with
/// [`blazen_future_take_music_result`].
///
/// Returns null if `model` is null or `prompt` is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as
/// [`blazen_music_model_generate_music_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_generate_music(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        Arc::clone(&inner)
            .generate_music(prompt, duration_seconds)
            .await
    })
}

// ---------------------------------------------------------------------------
// Non-streaming generate (sfx) — sync + async
// ---------------------------------------------------------------------------

/// Synchronously generate `duration_seconds` of sound-effect audio
/// conditioned on `prompt`. Result / error / status codes mirror
/// [`blazen_music_model_generate_music_blocking`].
///
/// # Safety
///
/// Same string contracts as
/// [`blazen_music_model_generate_music_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_generate_sfx_blocking(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(out_err, "blazen_music_model_generate_sfx: null model");
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        write_internal_error(
            out_err,
            "blazen_music_model_generate_sfx: null or non-UTF-8 prompt",
        );
        return -2;
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move {
        Arc::clone(&inner)
            .generate_sfx(prompt, duration_seconds)
            .await
    });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of sound-effect audio
/// conditioned on `prompt`. Returns a `*mut BlazenFuture` the caller
/// polls / waits on; the typed result is popped with
/// [`blazen_future_take_music_result`].
///
/// Returns null if `model` is null or `prompt` is null / non-UTF-8.
///
/// # Safety
///
/// Same string contracts as
/// [`blazen_music_model_generate_music_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_model_generate_sfx(
    model: *const BlazenMusicModel,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenMusicModel`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        Arc::clone(&inner)
            .generate_sfx(prompt, duration_seconds)
            .await
    })
}

// ---------------------------------------------------------------------------
// Typed future-take entry point
// ---------------------------------------------------------------------------

/// Pops a typed `MusicResult` out of `fut`. On success returns `0` and
/// writes a caller-owned `*mut BlazenMusicResult` into `out`; on failure
/// returns `-1` and writes a caller-owned `*mut BlazenError` into `err`.
///
/// `out` / `err` may be null when the caller wants to discard the value.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by either
/// [`blazen_music_model_generate_music`] or
/// [`blazen_music_model_generate_sfx`], not yet freed, and not
/// concurrently freed from another thread. `out` / `err` must be null OR
/// writable pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_music_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenMusicResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerMusicResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenMusicResult::from(v).into_ptr();
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blazen_uniffi::compute_music::MusicChunk as InnerMusicChunkForTest;

    use crate::compute_records::{BlazenMusicChunk, BlazenMusicResult};

    /// Plumbing test: the fal music factory should not contact fal during
    /// construction, so a dummy key returns a live model handle that
    /// `blazen_music_model_free` cleans up without leaking.
    #[test]
    fn fal_music_model_new_returns_nonnull_handle() {
        let api_key = std::ffi::CString::new("dummy").unwrap();
        let mut model: *mut BlazenMusicModel = std::ptr::null_mut();
        let mut err: *mut BlazenError = std::ptr::null_mut();
        // SAFETY: api_key is a live NUL-terminated buffer; out-params are
        // writable stack slots.
        let rc = unsafe {
            blazen_music_model_new_fal(
                api_key.as_ptr(),
                std::ptr::null(),
                &raw mut model,
                &raw mut err,
            )
        };
        assert_eq!(rc, 0, "expected ok, got rc={rc}, err={:?}", err.is_null());
        assert!(!model.is_null(), "expected non-null model handle");
        assert!(err.is_null(), "expected no error on success");
        // SAFETY: model came from the factory above; freeing through the
        // matching `blazen_music_model_free` is exactly the documented
        // ownership contract.
        unsafe {
            blazen_music_model_free(model);
        }
    }

    /// `BlazenMusicChunk` round-trips the `samples`/`is_final`/`latency`
    /// fields through the C accessor functions.
    #[test]
    fn blazen_music_chunk_round_trips_accessors() {
        let inner = InnerMusicChunkForTest {
            samples: vec![0.0_f32, 0.25, -0.5, 1.0, -1.0],
            is_final: true,
            latency_seconds: Some(0.125),
        };
        let chunk = BlazenMusicChunk::from(inner).into_ptr();

        let mut len: usize = 0;
        // SAFETY: `chunk` is a live cabi handle; `len` is a writable stack slot.
        let ptr =
            unsafe { crate::compute_records::blazen_music_chunk_samples(chunk, &raw mut len) };
        assert!(!ptr.is_null(), "samples ptr must be non-null");
        assert_eq!(len, 5);
        // SAFETY: ptr/len describe a live `Vec<f32>` borrowed from the chunk.
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert_eq!(slice, &[0.0_f32, 0.25, -0.5, 1.0, -1.0]);

        // SAFETY: live chunk pointer.
        assert!(unsafe { crate::compute_records::blazen_music_chunk_is_final(chunk) });
        // SAFETY: live chunk pointer.
        let latency = unsafe { crate::compute_records::blazen_music_chunk_latency_seconds(chunk) };
        assert!((latency - 0.125).abs() < f32::EPSILON);

        // SAFETY: `chunk` came from `into_ptr` above; freeing through the
        // documented free function is exactly the ownership contract.
        unsafe {
            crate::compute_records::blazen_music_chunk_free(chunk);
        }
    }

    /// `BlazenMusicResult` round-trips bytes / mime / `sample_rate` / channels /
    /// duration / url through the C accessor functions.
    #[test]
    fn blazen_music_result_round_trips_accessors() {
        let inner = InnerMusicResult {
            bytes: vec![0x01, 0x02, 0x03, 0x04],
            mime_type: "audio/wav".to_string(),
            sample_rate: 44_100,
            channels: 2,
            duration_seconds: 3.5,
            url: "https://example.invalid/clip.wav".to_string(),
        };
        let result = BlazenMusicResult::from(inner).into_ptr();

        let mut len: usize = 0;
        // SAFETY: `result` is a live cabi handle; `len` is a writable stack slot.
        let bytes_ptr =
            unsafe { crate::compute_records::blazen_music_result_bytes(result, &raw mut len) };
        assert!(!bytes_ptr.is_null());
        assert_eq!(len, 4);
        // SAFETY: ptr/len describe a live `Vec<u8>` borrowed from the result.
        let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, len) };
        assert_eq!(bytes, &[0x01_u8, 0x02, 0x03, 0x04]);

        // SAFETY: live result pointer.
        let mime = unsafe { crate::compute_records::blazen_music_result_mime_type(result) };
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
            unsafe { crate::compute_records::blazen_music_result_sample_rate(result) },
            44_100
        );
        assert_eq!(
            unsafe { crate::compute_records::blazen_music_result_channels(result) },
            2
        );
        let duration =
            unsafe { crate::compute_records::blazen_music_result_duration_seconds(result) };
        assert!((duration - 3.5).abs() < f32::EPSILON);

        // SAFETY: live result pointer.
        let url = unsafe { crate::compute_records::blazen_music_result_url(result) };
        assert!(!url.is_null());
        // SAFETY: pointer minted by `alloc_cstring` above.
        let url_str = unsafe { std::ffi::CStr::from_ptr(url).to_str().unwrap().to_owned() };
        assert_eq!(url_str, "https://example.invalid/clip.wav");
        // SAFETY: `url` was minted by `alloc_cstring`.
        unsafe { crate::string::blazen_string_free(url) };

        // SAFETY: `result` came from `into_ptr` above; free via the matching
        // free function.
        unsafe {
            crate::compute_records::blazen_music_result_free(result);
        }
    }
}
