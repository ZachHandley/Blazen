//! Native 3D model surface for the C ABI.
//!
//! Mirrors [`crate::compute_music`] for the upstream
//! [`blazen_uniffi::compute`] 3D types:
//!
//! - [`BlazenThreeDModel`] — opaque handle wrapping
//!   `Arc<blazen_uniffi::compute::ThreeDModel>`.
//! - [`blazen_three_d_model_new_triposr`] — local `TripoSR`
//!   single-image-to-3D factory.
//! - [`blazen_three_d_model_generate_from_image_blocking`] +
//!   [`blazen_three_d_model_generate_from_image`] — sync + async wrappers
//!   that produce a [`BlazenThreeDGenerateResult`] (GLB / gltf-binary
//!   bytes + MIME type).
//! - [`blazen_future_take_three_d_generate_result`] — typed future-take
//!   for the async variant.
//!
//! This module is distinct from [`crate::threed`] (the legacy HTTP-proxy
//! `Compat3dProvider` surface gated on `threed-compat-proxy`). Both can
//! coexist in a single cabi build.
//!
//! ## Ownership conventions
//!
//! - Model handles are heap-allocated `Box<BlazenThreeDModel>` returned by
//!   the factory functions; caller frees with
//!   [`blazen_three_d_model_free`].
//! - Result handles produced by `*_blocking` and by
//!   [`blazen_future_take_three_d_generate_result`] are caller-owned and
//!   freed via [`blazen_three_d_generate_result_free`].
//! - Error handles produced on the failure path are caller-owned and freed
//!   via [`crate::error::blazen_error_free`].
//! - String inputs are borrowed for the duration of the call only — the
//!   wrappers copy out everything they need into owned `String`s before
//!   spawning the underlying task.
//! - Image-byte inputs (`image_bytes` + `image_bytes_len`) are borrowed
//!   for the duration of the call only — the wrappers copy out the slice
//!   into an owned `Vec<u8>` before spawning the underlying task.

#![allow(dead_code)]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute::{
    ThreeDGenerateResult as InnerThreeDGenerateResult, ThreeDModel as InnerThreeDModel,
};
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_opt_string};

// ---------------------------------------------------------------------------
// Local error helpers — mirror compute_music.rs / compute_factories.rs.
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
/// Used for argument-shape errors (null required pointers, null model)
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
// BlazenThreeDModel handle
// ---------------------------------------------------------------------------

/// Opaque handle wrapping an `Arc<blazen_uniffi::compute::ThreeDModel>`.
///
/// Produced by [`blazen_three_d_model_new_triposr`]. Free with
/// [`blazen_three_d_model_free`].
pub struct BlazenThreeDModel(pub(crate) Arc<InnerThreeDModel>);

impl BlazenThreeDModel {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenThreeDModel {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerThreeDModel>> for BlazenThreeDModel {
    fn from(inner: Arc<InnerThreeDModel>) -> Self {
        Self(inner)
    }
}

/// Frees a `BlazenThreeDModel` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_three_d_model_new_triposr`]. Double-free is undefined
/// behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_model_free(model: *mut BlazenThreeDModel) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ---------------------------------------------------------------------------
// TripoSR factory
// ---------------------------------------------------------------------------

/// Build a local `TripoSR` single-image-to-3D model.
///
/// `hf_repo_id` selects the Hugging Face repo (default
/// `"stabilityai/TripoSR"`). `revision` pins a specific branch / tag /
/// commit on that repo. `weights_path` provides a pre-resolved local
/// directory containing the `image_encoder.safetensors` /
/// `transformer.safetensors` / `nerf_field.safetensors` triple; when
/// supplied, the HF download is skipped entirely.
///
/// All three inputs are optional — pass null to use the upstream
/// defaults (HF download from `stabilityai/TripoSR` at `main`).
///
/// Returns `0` on success and writes a fresh `BlazenThreeDModel*` into
/// `*out_model`. Returns `-1` on backend init failure and writes a fresh
/// `BlazenError*` into `*out_err`.
///
/// # Safety
///
/// - `hf_repo_id` / `revision` / `weights_path` must each be null OR a
///   valid NUL-terminated UTF-8 buffer.
/// - `out_model` and `out_err` must each be null OR point to a writable
///   slot of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_model_new_triposr(
    hf_repo_id: *const c_char,
    revision: *const c_char,
    weights_path: *const c_char,
    out_model: *mut *mut BlazenThreeDModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract on each input.
    let hf_repo_id = unsafe { cstr_to_opt_string(hf_repo_id) };
    let revision = unsafe { cstr_to_opt_string(revision) };
    let weights_path = unsafe { cstr_to_opt_string(weights_path) };

    match blazen_uniffi::compute::new_triposr_3d_model(hf_repo_id, revision, weights_path) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenThreeDModel::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

// ---------------------------------------------------------------------------
// Non-streaming generate_from_image — sync + async
// ---------------------------------------------------------------------------

/// Synchronously render a 3D mesh from a single input image.
///
/// `image_bytes` + `image_bytes_len` describe an encoded PNG or JPEG
/// payload. `mesh_resolution` controls the side length of the density
/// grid sampled from the triplane during marching cubes; `256` matches
/// the upstream `TripoSR` reference and is a reasonable default.
///
/// Returns `0` on success, `-1` on backend failure, `-2` on invalid
/// argument shape (null model / null `image_bytes`).
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenThreeDModel` produced
///   by the cabi surface.
/// - `image_bytes` must be a valid pointer to a buffer of at least
///   `image_bytes_len` bytes (or null, which yields `-2`).
/// - `out_result` / `out_err` must each be null OR writable pointers to
///   the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_model_generate_from_image_blocking(
    model: *const BlazenThreeDModel,
    image_bytes: *const u8,
    image_bytes_len: usize,
    mesh_resolution: u32,
    out_result: *mut *mut BlazenThreeDGenerateResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        write_internal_error(
            out_err,
            "blazen_three_d_model_generate_from_image: null model",
        );
        return -2;
    }
    if image_bytes.is_null() && image_bytes_len != 0 {
        write_internal_error(
            out_err,
            "blazen_three_d_model_generate_from_image: null image_bytes with non-zero len",
        );
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenThreeDModel`.
    let model = unsafe { &*model };
    // SAFETY: caller has guaranteed `image_bytes` points to a buffer of at
    // least `image_bytes_len` bytes (or zero-len with any pointer, in which
    // case we hand the inner factory an empty Vec).
    let bytes_vec: Vec<u8> = if image_bytes_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(image_bytes, image_bytes_len) }.to_vec()
    };

    let inner = Arc::clone(&model.0);
    let result = runtime().block_on(async move {
        Arc::clone(&inner)
            .generate_from_image(bytes_vec, mesh_resolution)
            .await
    });
    match result {
        Ok(v) => {
            if !out_result.is_null() {
                // SAFETY: caller has guaranteed `out_result` is writable.
                unsafe {
                    *out_result = BlazenThreeDGenerateResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously render a 3D mesh from a single input image. Returns a
/// `*mut BlazenFuture` the caller polls / waits on; the typed result is
/// popped with [`blazen_future_take_three_d_generate_result`].
///
/// Returns null if `model` is null or `image_bytes` is null with a
/// non-zero `image_bytes_len`.
///
/// # Safety
///
/// Same buffer contracts as
/// [`blazen_three_d_model_generate_from_image_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_model_generate_from_image(
    model: *const BlazenThreeDModel,
    image_bytes: *const u8,
    image_bytes_len: usize,
    mesh_resolution: u32,
) -> *mut BlazenFuture {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    if image_bytes.is_null() && image_bytes_len != 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live `BlazenThreeDModel`.
    let model = unsafe { &*model };
    // SAFETY: see the blocking variant above for the buffer contract.
    let bytes_vec: Vec<u8> = if image_bytes_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(image_bytes, image_bytes_len) }.to_vec()
    };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerThreeDGenerateResult, _>(async move {
        Arc::clone(&inner)
            .generate_from_image(bytes_vec, mesh_resolution)
            .await
    })
}

// ---------------------------------------------------------------------------
// Typed future-take entry point
// ---------------------------------------------------------------------------

/// Pops a typed `ThreeDGenerateResult` out of `fut`. On success returns
/// `0` and writes a caller-owned `*mut BlazenThreeDGenerateResult` into
/// `out`; on failure returns `-1` and writes a caller-owned
/// `*mut BlazenError` into `err`.
///
/// `out` / `err` may be null when the caller wants to discard the value.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_three_d_model_generate_from_image`], not yet freed, and not
/// concurrently freed from another thread. `out` / `err` must be null OR
/// writable pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_three_d_generate_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenThreeDGenerateResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<InnerThreeDGenerateResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = BlazenThreeDGenerateResult::from(v).into_ptr();
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
// BlazenThreeDGenerateResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// [`blazen_uniffi::compute::ThreeDGenerateResult`].
///
/// Produced by the cabi `ThreeDModel::generate_from_image` wrappers and
/// by the typed [`blazen_future_take_three_d_generate_result`] taker.
/// Holds the encoded 3D model bytes (typically GLB / gltf-binary) and an
/// IANA MIME type string so foreign callers can dispatch on the format
/// without sniffing the buffer.
pub struct BlazenThreeDGenerateResult(pub(crate) InnerThreeDGenerateResult);

impl BlazenThreeDGenerateResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenThreeDGenerateResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerThreeDGenerateResult> for BlazenThreeDGenerateResult {
    fn from(inner: InnerThreeDGenerateResult) -> Self {
        Self(inner)
    }
}

/// Borrows the result's encoded 3D model bytes (GLB / gltf-binary).
/// Writes the slice length into `*out_len` and returns the pointer to
/// the first byte. The returned pointer is valid for the lifetime of the
/// result handle (i.e. until [`blazen_three_d_generate_result_free`] is
/// called); callers must NOT free the buffer directly.
///
/// Returns null and writes `0` into `*out_len` if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a
/// `BlazenThreeDGenerateResult` produced by the cabi surface.
/// `out_len` must be null OR a writable pointer to a single `usize` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_generate_result_model_bytes(
    result: *const BlazenThreeDGenerateResult,
    out_len: *mut usize,
) -> *const u8 {
    if result.is_null() {
        if !out_len.is_null() {
            // SAFETY: caller upholds the out-pointer contract.
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    let slice = result.0.model_bytes.as_slice();
    if !out_len.is_null() {
        // SAFETY: caller upholds the out-pointer contract.
        unsafe {
            *out_len = slice.len();
        }
    }
    slice.as_ptr()
}

/// Returns the IANA MIME type of the encoded 3D model as a
/// heap-allocated C string (typically `"model/gltf-binary"`). Caller
/// frees with `blazen_string_free`. Returns null if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a
/// `BlazenThreeDGenerateResult` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_generate_result_mime_type(
    result: *const BlazenThreeDGenerateResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.mime_type)
}

/// Frees a `BlazenThreeDGenerateResult` produced by the cabi surface.
/// Passing null is a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// 3D-generation wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_generate_result_free(
    result: *mut BlazenThreeDGenerateResult,
) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `BlazenThreeDGenerateResult` round-trips bytes + mime through the C
    /// accessor functions.
    #[test]
    fn blazen_three_d_generate_result_round_trips_accessors() {
        let inner = InnerThreeDGenerateResult {
            model_bytes: vec![0x67, 0x6c, 0x54, 0x46], // "glTF" magic
            mime_type: "model/gltf-binary".to_string(),
        };
        let result = BlazenThreeDGenerateResult::from(inner).into_ptr();

        let mut len: usize = 0;
        // SAFETY: `result` is a live cabi handle; `len` is a writable stack slot.
        let bytes_ptr = unsafe { blazen_three_d_generate_result_model_bytes(result, &raw mut len) };
        assert!(!bytes_ptr.is_null());
        assert_eq!(len, 4);
        // SAFETY: ptr/len describe a live `Vec<u8>` borrowed from the result.
        let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, len) };
        assert_eq!(bytes, &[0x67_u8, 0x6c, 0x54, 0x46]);

        // SAFETY: live result pointer.
        let mime = unsafe { blazen_three_d_generate_result_mime_type(result) };
        assert!(!mime.is_null());
        // SAFETY: pointer minted by `alloc_cstring` above — valid NUL-terminated
        // UTF-8 we can recover via `CStr`.
        let mime_str = unsafe { std::ffi::CStr::from_ptr(mime).to_str().unwrap().to_owned() };
        assert_eq!(mime_str, "model/gltf-binary");
        // SAFETY: `mime` was minted by `alloc_cstring`.
        unsafe { crate::string::blazen_string_free(mime) };

        // SAFETY: `result` came from `into_ptr` above.
        unsafe {
            blazen_three_d_generate_result_free(result);
        }
    }
}
