//! Per-engine `ImageGen` provider opaques + C ABI factories.
//!
//! Each engine wraps an `Arc<blazen_uniffi::concrete::image::<Engine>Provider>`
//! in a `Blazen<Engine>Provider` opaque, exposing:
//!
//! - `blazen_<engine>_provider_new` -- factory
//! - `blazen_<engine>_provider_generate_image` -- async future returning a
//!   `*mut BlazenFuture` whose typed result is popped with
//!   [`crate::compute::blazen_future_take_image_gen_result`]
//! - `blazen_<engine>_provider_generate_image_blocking` -- synchronous variant
//! - `blazen_<engine>_provider_free` -- destructor (no-op on null)
//!
//! ## Feature gating
//!
//! Both engines exposed here are gated on `feature = "diffusion"`. The
//! polymorphic [`blazen_uniffi::concrete::bases::ImageGenProvider`] trait is
//! itself gated on `diffusion`, and the per-engine
//! [`blazen_uniffi::concrete::image`] module is `#![cfg(feature = "diffusion")]`,
//! so `FalImageGenProvider` -- although purely HTTP and unconditional in
//! `blazen-llm` -- only surfaces through this cabi when the `diffusion`
//! feature is enabled. This mirrors the uniffi surface.
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<Blazen<Engine>Provider>` returned
//!   by the factory functions. Callers free with the matching `*_free`.
//!   Double-free is undefined behavior.
//! - String inputs are borrowed for the duration of the call only -- the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.
//! - The futures and `BlazenImageGenResult`s produced here share the existing
//!   [`crate::compute::blazen_future_take_image_gen_result`] /
//!   [`crate::compute_records::blazen_image_gen_result_free`] pipeline used by
//!   the central [`crate::compute::BlazenImageGenModel`] -- there is no
//!   per-engine result type.
//!
//! ## Optional inputs
//!
//! All `Option<String>` parameters on the Rust side accept `*const c_char`
//! and treat null as `None`. The `width` / `height` parameters use `0` as
//! the `None` sentinel -- a zero-dimensioned image makes no semantic sense,
//! so collapsing it to `None` lets the upstream provider fall back to its
//! default size.
//!
//! ## Relationship to the central [`crate::compute::BlazenImageGenModel`]
//!
//! The central `BlazenImageGenModel` + `blazen_image_gen_model_new_*`
//! factories in [`crate::compute`] / [`crate::compute_factories`] remain in
//! place -- this module is purely additive. Foreign hosts (Ruby, future
//! Dart / Crystal / Lua / PHP) can migrate to the per-engine surface
//! incrementally without breaking the existing Ruby gem entry points.

#![cfg(feature = "diffusion")]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute::ImageGenResult as InnerImageGenResult;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local helpers (mirroring crate::tts)
// ---------------------------------------------------------------------------

/// Convert a C `u32` Option-as-sentinel into the equivalent `Option<u32>`.
///
/// A value of `0` collapses to `None`; non-zero values pass through as
/// `Some(v)`. A zero-dimensioned image is semantically meaningless, so the
/// sentinel reuse is safe.
#[inline]
fn opt_u32_from_zero(v: u32) -> Option<u32> {
    if v == 0 { None } else { Some(v) }
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
// DiffusionProvider -- local stable-diffusion.cpp (feature = "diffusion")
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::image::DiffusionProvider>`.
///
/// Free with [`blazen_diffusion_provider_free`].
pub struct BlazenDiffusionProvider(
    pub(crate) Arc<blazen_uniffi::concrete::image::DiffusionProvider>,
);

impl BlazenDiffusionProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenDiffusionProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenDiffusionProvider` from an optional JSON-encoded
/// [`blazen_llm::DiffusionOptions`] payload.
///
/// Pass null `options_json` to use the upstream defaults (512x512, `EulerA`
/// scheduler, 20 inference steps). Invalid JSON or option-validation
/// failures surface as a fresh `BlazenError*` in `*out_err`.
///
/// On success returns `0` and writes a fresh `BlazenDiffusionProvider*` into
/// `*out_model`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`. Both out-parameters may be null to discard that side of
/// the result.
///
/// # Safety
///
/// - `options_json` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_diffusion_provider_new(
    options_json: *const c_char,
    out_model: *mut *mut BlazenDiffusionProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let options_json = unsafe { cstr_to_opt_string(options_json) };

    match blazen_uniffi::concrete::image::DiffusionProvider::new(options_json) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenDiffusionProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate an image from `prompt`. Returns a `*mut BlazenFuture`
/// the caller polls / waits on; the typed result is popped with
/// [`crate::compute::blazen_future_take_image_gen_result`].
///
/// `width` and `height` use `0` as the `None` sentinel -- pass `0` for either
/// to defer to the provider's default size. When either is `0` the request
/// omits the size override entirely (the inner builder only applies the
/// override when both are present).
///
/// Returns null if `model` is null or `prompt` is null / non-UTF-8.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenDiffusionProvider` produced
///   by this surface.
/// - `prompt` must be a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_diffusion_provider_generate_image(
    model: *const BlazenDiffusionProvider,
    prompt: *const c_char,
    width: u32,
    height: u32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenDiffusionProvider`.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    let prompt = prompt.to_owned();
    let width = opt_u32_from_zero(width);
    let height = opt_u32_from_zero(height);

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerImageGenResult, _>(async move {
        inner.generate_image(prompt, width, height).await
    })
}

/// Synchronous variant of [`blazen_diffusion_provider_generate_image`]. Returns
/// `0` on success (typed result in `*out_result`), `-1` on failure (error in
/// `*out_err`), `-2` on invalid input (null model / non-UTF-8 `prompt`).
///
/// # Safety
///
/// Same string contracts as [`blazen_diffusion_provider_generate_image`].
/// `out_result` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_diffusion_provider_generate_image_blocking(
    model: *const BlazenDiffusionProvider,
    prompt: *const c_char,
    width: u32,
    height: u32,
    out_result: *mut *mut crate::compute_records::BlazenImageGenResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return -2;
    };
    let prompt = prompt.to_owned();
    let width = opt_u32_from_zero(width);
    let height = opt_u32_from_zero(height);

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_image(prompt, width, height).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenImageGenResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenDiffusionProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_diffusion_provider_new`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_diffusion_provider_free(model: *mut BlazenDiffusionProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// FalImageGenProvider -- fal.ai hosted (still gated on `diffusion` to mirror
// the uniffi surface, where the polymorphic ImageGenProvider trait is gated)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::image::FalImageGenProvider>`.
///
/// Free with [`blazen_fal_image_gen_provider_free`].
pub struct BlazenFalImageGenProvider(
    pub(crate) Arc<blazen_uniffi::concrete::image::FalImageGenProvider>,
);

impl BlazenFalImageGenProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalImageGenProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenFalImageGenProvider` from a fal.ai API key.
///
/// `default_model` is an optional default image model id
/// (e.g. `"fal-ai/flux/schnell"`). `base_url` overrides the default fal
/// queue URL (`https://queue.fal.run`) -- used for proxies / staging
/// environments.
///
/// This constructor is infallible on the Rust side (the uniffi wrapper
/// returns `Arc<Self>` directly), so it returns `0` and writes the handle
/// into `*out_model`. The `out_err` parameter exists only for argument-shape
/// errors (null required pointers, non-UTF-8 strings).
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `default_model` / `base_url` must be null OR valid NUL-terminated UTF-8
///   buffers.
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_image_gen_provider_new(
    api_key: *const c_char,
    default_model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenFalImageGenProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return write_internal_error(
            out_err,
            "blazen_fal_image_gen_provider_new: api_key must not be null",
        );
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let default_model = unsafe { cstr_to_opt_string(default_model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };

    let arc =
        blazen_uniffi::concrete::image::FalImageGenProvider::new(api_key, default_model, base_url);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFalImageGenProvider(arc).into_ptr();
        }
    }
    0
}

/// Asynchronously generate an image from `prompt`. See
/// [`blazen_diffusion_provider_generate_image`] for the future /
/// `take_image_gen_result` pipeline and the `0`-as-`None` sentinel
/// convention on `width` / `height`.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenFalImageGenProvider`
///   produced by this surface.
/// - `prompt` must be a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_image_gen_provider_generate_image(
    model: *const BlazenFalImageGenProvider,
    prompt: *const c_char,
    width: u32,
    height: u32,
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
    let width = opt_u32_from_zero(width);
    let height = opt_u32_from_zero(height);

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerImageGenResult, _>(async move {
        inner.generate_image(prompt, width, height).await
    })
}

/// Synchronous variant of [`blazen_fal_image_gen_provider_generate_image`].
/// Returns `0` on success, `-1` on failure (error in `*out_err`).
///
/// # Safety
///
/// Same contracts as [`blazen_diffusion_provider_generate_image_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_image_gen_provider_generate_image_blocking(
    model: *const BlazenFalImageGenProvider,
    prompt: *const c_char,
    width: u32,
    height: u32,
    out_result: *mut *mut crate::compute_records::BlazenImageGenResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on `prompt`.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return -2;
    };
    let prompt = prompt.to_owned();
    let width = opt_u32_from_zero(width);
    let height = opt_u32_from_zero(height);

    let inner = Arc::clone(&model.0);
    let result =
        runtime().block_on(async move { inner.generate_image(prompt, width, height).await });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = crate::compute_records::BlazenImageGenResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenFalImageGenProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_fal_image_gen_provider_new`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_image_gen_provider_free(model: *mut BlazenFalImageGenProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
