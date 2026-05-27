//! Per-engine Music provider opaques + C ABI factories.
//!
//! Each engine wraps an `Arc<blazen_uniffi::concrete::music::<Engine>Provider>`
//! in a `Blazen<Engine>Provider` opaque, exposing:
//!
//! - `blazen_<engine>_provider_new` -- factory
//! - `blazen_<engine>_provider_generate_music` -- async future returning a
//!   `*mut BlazenFuture` whose typed result is popped with
//!   [`crate::compute_music::blazen_future_take_music_result`]
//! - `blazen_<engine>_provider_generate_music_blocking` -- synchronous variant
//! - `blazen_<engine>_provider_generate_sfx` -- async future (same future
//!   accessor as music)
//! - `blazen_<engine>_provider_generate_sfx_blocking` -- synchronous variant
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
//! - The futures and `BlazenMusicResult`s produced here share the existing
//!   [`crate::compute_music::blazen_future_take_music_result`] /
//!   [`crate::compute_records::blazen_music_result_free`] pipeline used by the
//!   central [`crate::compute_music::BlazenMusicModel`] — there is no
//!   per-engine result type.
//!
//! ## Music vs SFX unsupported routing
//!
//! `MusicGenProvider` only supports music (its `generate_sfx` returns
//! `BlazenError::Unsupported`). `AudioGenProvider` only supports SFX (its
//! `generate_music` returns `BlazenError::Unsupported`). Rather than omit the
//! unsupported pairs from the cabi surface, every engine exposes both
//! variants — the cabi wrappers route through the
//! [`blazen_uniffi::concrete::bases::MusicProvider`] trait which surfaces the
//! upstream `Unsupported` error to the foreign caller. This keeps the C ABI
//! shape uniform across engines so wrappers in the Ruby gem (and any future
//! cabi consumer) can be generated programmatically without per-engine
//! special-casing.
//!
//! ## Optional inputs
//!
//! There are no optional inputs at the generate call site — `prompt` is
//! required and `duration_seconds` is a plain `f32`. Constructor-side options
//! follow the same NaN-as-`None` / null-as-`None` conventions documented in
//! [`crate::compute_music`].
//!
//! ## Relationship to the central [`crate::compute_music::BlazenMusicModel`]
//!
//! The central `BlazenMusicModel` + `blazen_music_model_new_*` factories in
//! [`crate::compute_music`] remain in place — this module is purely additive.
//! Foreign hosts (Ruby, future Dart / Crystal / Lua / PHP) can migrate to the
//! per-engine surface incrementally without breaking the existing Ruby gem
//! entry points.

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute_music::MusicResult as InnerMusicResult;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local helpers (mirroring crate::compute_music + crate::tts)
// ---------------------------------------------------------------------------

/// Convert a C `f32` Option-as-sentinel into the equivalent `Option<f32>`.
///
/// NaN is the sentinel for "unset" — any non-NaN value (including zero or
/// negative) passes through as `Some(v)`. Mirrors
/// `crate::compute_music::opt_f32_from_f32`.
#[inline]
fn opt_f32_from_f32(v: f32) -> Option<f32> {
    if v.is_nan() { None } else { Some(v) }
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
// MusicGenProvider — Meta MusicGen text-to-music (feature = "audio-music-musicgen")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::music::MusicGenProvider>`.
///
/// Free with [`blazen_musicgen_provider_free`].
#[cfg(feature = "audio-music-musicgen")]
pub struct BlazenMusicGenProvider(pub(crate) Arc<blazen_uniffi::concrete::music::MusicGenProvider>);

#[cfg(feature = "audio-music-musicgen")]
impl BlazenMusicGenProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenMusicGenProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenMusicGenProvider`.
///
/// `variant` selects the `MusicGen` checkpoint by name (case-insensitive:
/// `"small"`, `"medium"`, `"large"`); pass null to default to `"small"`.
/// `device` follows `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`,
/// `"cuda:N"`, `"metal"`, `"metal:N"`); null defers to the backend's
/// auto-detection. `cache_dir` overrides the Hugging Face Hub cache
/// directory. `max_duration_seconds` follows the NaN-as-`None` encoding;
/// `None` defaults to 30 s upstream.
///
/// On success returns `0` and writes a fresh `BlazenMusicGenProvider*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Both out-parameters may be null to discard that side of
/// the result.
///
/// # Safety
///
/// - `variant` / `device` / `cache_dir` must each be null OR a valid
///   NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_new(
    variant: *const c_char,
    device: *const c_char,
    cache_dir: *const c_char,
    max_duration_seconds: f32,
    out_model: *mut *mut BlazenMusicGenProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let variant = unsafe { cstr_to_opt_string(variant) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };
    let max_duration_seconds = opt_f32_from_f32(max_duration_seconds);

    match blazen_uniffi::concrete::music::MusicGenProvider::new(
        variant,
        device,
        cache_dir,
        max_duration_seconds,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenMusicGenProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of music conditioned on
/// `prompt`. Returns a `*mut BlazenFuture` the caller polls / waits on; the
/// typed result is popped with
/// [`crate::compute_music::blazen_future_take_music_result`].
///
/// Returns null if `model` is null or `prompt` is null / non-UTF-8.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenMusicGenProvider` produced
///   by this surface.
/// - `prompt` must be a valid NUL-terminated UTF-8 buffer.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_generate_music(
    model: *const BlazenMusicGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_music(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_musicgen_provider_generate_music`].
/// Returns `0` on success (typed result in `*out_result`), `-1` on failure
/// (error in `*out_err`), `-2` on invalid input (null model / non-UTF-8
/// `prompt`).
///
/// # Safety
///
/// Same string contracts as
/// [`blazen_musicgen_provider_generate_music`]. `out_result` / `out_err` must
/// each be null OR point to a writable slot.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_generate_music_blocking(
    model: *const BlazenMusicGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_musicgen_provider_generate_music_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_musicgen_provider_generate_music_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_music(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of sound-effect audio
/// conditioned on `prompt`. For `MusicGenProvider` the upstream impl
/// returns `BlazenError::Unsupported` (`MusicGen` is music-only) — the
/// future resolves to that error and the foreign caller pops it through the
/// standard `blazen_future_take_music_result` accessor.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_generate_sfx(
    model: *const BlazenMusicGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        // MusicGen has no inherent `generate_sfx` — route through the
        // capability trait, whose impl returns `BlazenError::Unsupported`.
        use blazen_uniffi::concrete::bases::MusicProvider as _;
        inner.generate_sfx(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_musicgen_provider_generate_sfx`]. The
/// `MusicGen` backend returns `BlazenError::Unsupported`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_generate_sfx_blocking(
    model: *const BlazenMusicGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_musicgen_provider_generate_sfx_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_musicgen_provider_generate_sfx_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move {
        // MusicGen has no inherent `generate_sfx` — route through the
        // capability trait, whose impl returns `BlazenError::Unsupported`.
        use blazen_uniffi::concrete::bases::MusicProvider as _;
        inner.generate_sfx(prompt, duration_seconds).await
    });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenMusicGenProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_musicgen_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_musicgen_provider_free(model: *mut BlazenMusicGenProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// AudioGenProvider — Meta AudioGen text-to-sfx (feature = "audio-music-audiogen")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::music::AudioGenProvider>`.
#[cfg(feature = "audio-music-audiogen")]
pub struct BlazenAudioGenProvider(pub(crate) Arc<blazen_uniffi::concrete::music::AudioGenProvider>);

#[cfg(feature = "audio-music-audiogen")]
impl BlazenAudioGenProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenAudioGenProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenAudioGenProvider`.
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
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_new(
    repo_id: *const c_char,
    revision: *const c_char,
    device: *const c_char,
    cache_dir: *const c_char,
    max_duration_seconds: f32,
    out_model: *mut *mut BlazenAudioGenProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on each input.
    let repo_id = unsafe { cstr_to_opt_string(repo_id) };
    let revision = unsafe { cstr_to_opt_string(revision) };
    let device = unsafe { cstr_to_opt_string(device) };
    let cache_dir = unsafe { cstr_to_opt_string(cache_dir) };
    let max_duration_seconds = opt_f32_from_f32(max_duration_seconds);

    match blazen_uniffi::concrete::music::AudioGenProvider::new(
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
                    *out_model = BlazenAudioGenProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of music conditioned on
/// `prompt`. For `AudioGenProvider` the upstream impl returns
/// `BlazenError::Unsupported` (`AudioGen` is sfx-only).
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_generate_music(
    model: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_music(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_audiogen_provider_generate_music`]. The
/// `AudioGen` backend returns `BlazenError::Unsupported`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_generate_music_blocking(
    model: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_audiogen_provider_generate_music_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_audiogen_provider_generate_music_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_music(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of sound-effect audio
/// conditioned on `prompt`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_generate_sfx(
    model: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_sfx(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_audiogen_provider_generate_sfx`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_generate_sfx_blocking(
    model: *const BlazenAudioGenProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_audiogen_provider_generate_sfx_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_audiogen_provider_generate_sfx_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_sfx(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenAudioGenProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_audiogen_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "audio-music-audiogen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audiogen_provider_free(model: *mut BlazenAudioGenProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// StableAudioProvider — Stability AI Stable Audio Open
// (feature = "audio-music-stable-audio")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::music::StableAudioProvider>`.
#[cfg(feature = "audio-music-stable-audio")]
pub struct BlazenStableAudioProvider(
    pub(crate) Arc<blazen_uniffi::concrete::music::StableAudioProvider>,
);

#[cfg(feature = "audio-music-stable-audio")]
impl BlazenStableAudioProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenStableAudioProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenStableAudioProvider`.
///
/// Unlike the other music engines this constructor is async upstream
/// (Stable Audio loads weights at construction time); the cabi factory
/// drives the shared cabi Tokio runtime via [`runtime().block_on`].
///
/// `variant` selects the Stable Audio Open checkpoint by name
/// (case-insensitive: `"small"`, `"open-1.0"` / `"open1.0"`); pass null to
/// default to `"small"`. `tokenizer_path` is REQUIRED — it must point at
/// the T5 `SentencePiece` `tokenizer.json` shipped with the Stable Audio
/// Open repo (Stable Audio's tokenizer is not auto-downloaded by the
/// backend today). `device` follows the same device-string format as the
/// `MusicGen` factory; null defaults to CPU. `max_duration_seconds` is
/// accepted for API symmetry but Stable Audio enforces its own
/// variant-dependent ceiling internally.
///
/// Returns `0` on success, `-1` on backend init failure, or `-2` on
/// invalid argument shape (null `tokenizer_path`).
///
/// # Safety
///
/// - `tokenizer_path` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `variant` / `device` must each be null OR a valid NUL-terminated UTF-8
///   buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_new(
    variant: *const c_char,
    tokenizer_path: *const c_char,
    device: *const c_char,
    max_duration_seconds: f32,
    out_model: *mut *mut BlazenStableAudioProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let variant = unsafe { cstr_to_opt_string(variant) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `tokenizer_path`.
    let Some(tokenizer_path) = (unsafe { cstr_to_str(tokenizer_path) }) else {
        write_internal_error(
            out_err,
            "blazen_stable_audio_provider_new: tokenizer_path must not be null",
        );
        return -2;
    };
    let tokenizer_path = tokenizer_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    let max_duration_seconds = opt_f32_from_f32(max_duration_seconds);

    let result = runtime().block_on(async move {
        blazen_uniffi::concrete::music::StableAudioProvider::new(
            variant,
            tokenizer_path,
            device,
            max_duration_seconds,
        )
        .await
    });
    match result {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenStableAudioProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of music conditioned on
/// `prompt`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_generate_music(
    model: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_music(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_stable_audio_provider_generate_music`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_generate_music_blocking(
    model: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_stable_audio_provider_generate_music_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_stable_audio_provider_generate_music_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_music(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of sound-effect audio
/// conditioned on `prompt`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_generate_sfx(
    model: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_sfx(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_stable_audio_provider_generate_sfx`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_generate_sfx_blocking(
    model: *const BlazenStableAudioProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_stable_audio_provider_generate_sfx_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_stable_audio_provider_generate_sfx_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_sfx(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenStableAudioProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_stable_audio_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "audio-music-stable-audio")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stable_audio_provider_free(model: *mut BlazenStableAudioProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// FalMusicProvider — fal.ai hosted music + sfx (no feature gate)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::music::FalMusicProvider>`.
///
/// Note: the upstream `MusicProvider` trait is itself gated behind
/// `audio-music-musicgen` (the trait's `#[uniffi::export]` only emits when
/// the music feature group is on). When that feature is off the
/// `generate_*` cabi entry points compile away too — host bindings should
/// gate calls behind the same feature. The handle constructor +
/// destructor remain available unconditionally for symmetry with the
/// always-on `crate::compute_music::blazen_music_model_new_fal` factory.
pub struct BlazenFalMusicProvider(pub(crate) Arc<blazen_uniffi::concrete::music::FalMusicProvider>);

impl BlazenFalMusicProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalMusicProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFalMusicProvider` from a fal.ai API key.
///
/// An empty `api_key` falls back to the `FAL_KEY` environment variable.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null; empty
///   string is allowed).
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_music_provider_new(
    api_key: *const c_char,
    out_model: *mut *mut BlazenFalMusicProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_music_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();

    let arc = blazen_uniffi::concrete::music::FalMusicProvider::new(api_key);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFalMusicProvider(arc).into_ptr();
        }
    }
    0
}

/// Asynchronously generate `duration_seconds` of music conditioned on
/// `prompt`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_music_provider_generate_music(
    model: *const BlazenFalMusicProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_music(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_fal_music_provider_generate_music`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_music_provider_generate_music_blocking(
    model: *const BlazenFalMusicProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_fal_music_provider_generate_music_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_music_provider_generate_music_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_music(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate `duration_seconds` of sound-effect audio
/// conditioned on `prompt`.
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music`].
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_music_provider_generate_sfx(
    model: *const BlazenFalMusicProvider,
    prompt: *const c_char,
    duration_seconds: f32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerMusicResult, _>(async move {
        inner.generate_sfx(prompt, duration_seconds).await
    })
}

/// Synchronous variant of [`blazen_fal_music_provider_generate_sfx`].
///
/// # Safety
///
/// Same contracts as [`blazen_musicgen_provider_generate_music_blocking`].
#[cfg(feature = "audio-music-musicgen")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_music_provider_generate_sfx_blocking(
    model: *const BlazenFalMusicProvider,
    prompt: *const c_char,
    duration_seconds: f32,
    out_result: *mut *mut crate::compute_records::BlazenMusicResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return write_internal_error(
            out_err,
            "blazen_fal_music_provider_generate_sfx_blocking: null model",
        );
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_music_provider_generate_sfx_blocking: null or non-UTF-8 prompt",
        );
    };
    let prompt = prompt.to_owned();

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_sfx(prompt, duration_seconds).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenMusicResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenFalMusicProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_fal_music_provider_new`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_music_provider_free(model: *mut BlazenFalMusicProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
