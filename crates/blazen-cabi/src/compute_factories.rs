//! Compute model factories: TTS, STT, image-gen (6 factories, some
//! feature-gated). Phase R4 Agent B.
//!
//! Each factory wraps the matching `blazen_uniffi::compute::new_<x>_<kind>_model`
//! call and hands back an opaque `*mut BlazenXModel` via an out-parameter:
//!
//! | C function                                | upstream                       | feature gate  |
//! |-------------------------------------------|--------------------------------|---------------|
//! | [`blazen_tts_model_new_fal`]              | `new_fal_tts_model`            | (always on)   |
//! | [`blazen_stt_model_new_fal`]              | `new_fal_stt_model`            | (always on)   |
//! | [`blazen_image_gen_model_new_fal`]        | `new_fal_image_gen_model`      | (always on)   |
//! | [`blazen_tts_model_new_piper`]            | `new_piper_tts_model`          | `piper`       |
//! | [`blazen_stt_model_new_whisper`]          | `new_whisper_stt_model`        | `whispercpp`  |
//! | [`blazen_image_gen_model_new_diffusion`]  | `new_diffusion_model`          | `diffusion`   |
//!
//! ## Optional-numeric encodings
//!
//! - `Option<u32>` → `i32`: `-1` (or any negative value) means `None`. See
//!   [`opt_u32_from_i32`].
//! - `Option<f32>` → `f32`: NaN means `None`. See [`opt_f32_from_f32`].
//!
//! ## Ownership conventions
//!
//! - Returned model handles are heap-allocated via the matching
//!   `BlazenXModel::into_ptr` and owned by the C caller. Free with the matching
//!   `blazen_x_model_free` in [`crate::compute`].
//! - Error handles produced on the failure path are caller-owned and freed
//!   via [`crate::error::blazen_error_free`].
//! - String inputs are borrowed for the duration of the factory call only —
//!   the wrappers copy out everything they need into owned `String`s before
//!   handing off to the upstream factory.

use std::ffi::c_char;

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::compute::{BlazenImageGenModel, BlazenSttModel, BlazenTtsModel};
use crate::error::BlazenError;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local error / sentinel helpers
//
// Mirrors the pattern in `llm.rs` / `persist.rs` / `agent.rs` instead of
// pulling from a shared helper — keeping each module self-contained means a
// future split (e.g. shipping compute as its own cdylib) doesn't drag in
// unrelated surface area.
// ---------------------------------------------------------------------------

/// Writes a caller-owned `BlazenError` into `out_err` if the slot is non-null
/// and returns `-1` so the caller can `return write_error(...)` in tail position.
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

/// Convert a C `i32` Option-as-sentinel into the equivalent `Option<u32>`.
///
/// Any negative value collapses to `None`; non-negative values pass through as
/// `Some(v as u32)`. Matches the encoding used by the image-gen wrappers in
/// [`crate::compute`].
#[inline]
fn opt_u32_from_i32(v: i32) -> Option<u32> {
    if v < 0 {
        None
    } else {
        Some(u32::try_from(v).unwrap_or(0))
    }
}

/// Convert a C `f32` Option-as-sentinel into the equivalent `Option<f32>`.
///
/// NaN is the sentinel for "unset" — any non-NaN value (including zero or
/// negative) passes through as `Some(v)`. Used by the diffusion image-gen
/// factory's `guidance_scale` parameter.
#[inline]
fn opt_f32_from_f32(v: f32) -> Option<f32> {
    if v.is_nan() { None } else { Some(v) }
}

// ---------------------------------------------------------------------------
// fal.ai compute factories (always available)
// ---------------------------------------------------------------------------

/// Build a fal.ai-backed TTS model.
///
/// `api_key` must be a NUL-terminated UTF-8 buffer (empty is allowed — the
/// upstream factory resolves `FAL_KEY` from the environment in that case).
/// `model` may be null to leave the default fal TTS endpoint in place; pass a
/// NUL-terminated UTF-8 buffer to override (e.g. `"fal-ai/dia-tts"`).
///
/// On success returns `0` and writes a fresh `BlazenTtsModel*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Both out-parameters may be null to discard the matching
/// side of the result.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_model_new_fal(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenTtsModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `api_key`.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_tts_model_new_fal: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `model`.
    let model = unsafe { cstr_to_opt_string(model) };

    match blazen_uniffi::compute::new_fal_tts_model(api_key, model) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenTtsModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Build a fal.ai-backed STT model.
///
/// Argument conventions mirror [`blazen_tts_model_new_fal`].
///
/// # Safety
///
/// Same string and out-pointer contracts as [`blazen_tts_model_new_fal`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_model_new_fal(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenSttModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `api_key`.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_stt_model_new_fal: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `model`.
    let model = unsafe { cstr_to_opt_string(model) };

    match blazen_uniffi::compute::new_fal_stt_model(api_key, model) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenSttModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Build a fal.ai-backed image-generation model.
///
/// Argument conventions mirror [`blazen_tts_model_new_fal`]; pass `model` as
/// the default fal image endpoint override (e.g. `"fal-ai/flux/dev"`).
///
/// # Safety
///
/// Same string and out-pointer contracts as [`blazen_tts_model_new_fal`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_model_new_fal(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenImageGenModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `api_key`.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_image_gen_model_new_fal: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `model`.
    let model = unsafe { cstr_to_opt_string(model) };

    match blazen_uniffi::compute::new_fal_image_gen_model(api_key, model) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenImageGenModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Piper TTS factory (feature = "piper")
// ---------------------------------------------------------------------------

/// Build a local Piper text-to-speech model.
///
/// `model_id` selects a Piper voice (e.g. `"en_US-amy-medium"`); pass null to
/// leave it unset (the upstream factory falls back to its default). `speaker_id`
/// and `sample_rate` follow the Option-as-`-1` encoding documented in
/// [`opt_u32_from_i32`].
///
/// Construction succeeds today but synthesise calls surface the upstream
/// "engine not yet wired" error until the Piper Phase 9 work lands — see
/// `blazen_uniffi::compute::new_piper_tts_model`.
///
/// On success returns `0` and writes a fresh `BlazenTtsModel*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`.
///
/// # Safety
///
/// - `model_id` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[cfg(feature = "piper")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_model_new_piper(
    model_id: *const c_char,
    speaker_id: i32,
    sample_rate: i32,
    out_model: *mut *mut BlazenTtsModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `model_id`.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    let speaker_id = opt_u32_from_i32(speaker_id);
    let sample_rate = opt_u32_from_i32(sample_rate);

    match blazen_uniffi::compute::new_piper_tts_model(model_id, speaker_id, sample_rate) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenTtsModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Whisper STT factory (feature = "whispercpp")
// ---------------------------------------------------------------------------

/// Build a local whisper.cpp speech-to-text model.
///
/// `model` selects a whisper variant by name (`"tiny"`, `"base"`, `"small"`,
/// `"medium"`, `"large-v3"`); pass null to default to `"small"` upstream.
/// `device` follows `blazen_llm::Device::parse` (`"cpu"`, `"cuda"`,
/// `"cuda:N"`, `"metal"`). `language` is an optional default ISO-639-1 hint
/// (overridable per-call on [`crate::compute::blazen_stt_model_transcribe_blocking`]).
///
/// # Safety
///
/// - `model` / `device` / `language` must each be null OR a valid
///   NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[cfg(feature = "whispercpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_model_new_whisper(
    model: *const c_char,
    device: *const c_char,
    language: *const c_char,
    out_model: *mut *mut BlazenSttModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let language = unsafe { cstr_to_opt_string(language) };

    match blazen_uniffi::compute::new_whisper_stt_model(model, device, language) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenSttModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Diffusion image-gen factory (feature = "diffusion")
// ---------------------------------------------------------------------------

/// Build a local diffusion-rs image-generation model.
///
/// `model_id` is a Hugging Face repo id (e.g. `"stabilityai/stable-diffusion-2-1"`);
/// pass null to use the upstream default. `device` follows the same device
/// string format as the LLM factories. `width` / `height` /
/// `num_inference_steps` use the `-1`-as-`None` encoding (see
/// [`opt_u32_from_i32`]); `guidance_scale` uses NaN-as-`None`
/// (see [`opt_f32_from_f32`]).
///
/// Construction succeeds today but generate calls surface the upstream
/// "engine not yet wired" error until the Phase 5.3 work lands — see
/// `blazen_uniffi::compute::new_diffusion_model`.
///
/// # Safety
///
/// - `model_id` / `device` must each be null OR a valid NUL-terminated UTF-8
///   buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable slot
///   of the matching pointer type.
#[cfg(feature = "diffusion")]
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn blazen_image_gen_model_new_diffusion(
    model_id: *const c_char,
    device: *const c_char,
    width: i32,
    height: i32,
    num_inference_steps: i32,
    guidance_scale: f32,
    out_model: *mut *mut BlazenImageGenModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    let width = opt_u32_from_i32(width);
    let height = opt_u32_from_i32(height);
    let num_inference_steps = opt_u32_from_i32(num_inference_steps);
    let guidance_scale = opt_f32_from_f32(guidance_scale);

    match blazen_uniffi::compute::new_diffusion_model(
        model_id,
        device,
        width,
        height,
        num_inference_steps,
        guidance_scale,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenImageGenModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}
