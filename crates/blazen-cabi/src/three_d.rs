//! Per-engine 3D provider opaques + C ABI factories.
//!
//! Mirrors the structural template in [`crate::tts`] for the
//! generation-only 3D engine exported by
//! [`blazen_uniffi::concrete::three_d`]:
//!
//! - [`BlazenTripoSrProvider`] (gated on `triposr`) — native candle
//!   image-to-3D. Generation-only; pipe the produced GLB through a
//!   post-proc backend (the legacy [`crate::threed`] HTTP-proxy
//!   surface, or a future per-engine
//!   `crate::three_d::BlazenCompat3dProvider`) for texturize / rig /
//!   refine / animate.
//!
//! ## Scope
//!
//! The `Compat3dProvider` (HTTP-proxy post-processing) wrapper is
//! deferred to a follow-up task. The existing
//! [`crate::threed`] module already exposes a full
//! `BlazenCompat3dProvider` surface (with JSON-string knobs) that
//! conflicts with both the Rust struct path and the cbindgen-generated
//! C type name. Replacing it requires coordinating the deletion of
//! that legacy module — out-of-scope for the additive Part U `TripoSR`
//! wrapper here. See the follow-up task report.
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<BlazenTripoSrProvider>`
//!   returned by [`blazen_triposr_provider_new`]. Callers free with
//!   [`blazen_triposr_provider_free`]. Double-free is undefined
//!   behavior.
//! - String inputs are borrowed for the duration of the call only —
//!   the wrapper copies out everything it needs into owned `String`s
//!   before spawning the underlying task.
//! - Byte-slice inputs (`image_bytes` + `image_bytes_len`) are
//!   borrowed for the duration of the call only — the wrapper copies
//!   them into an owned `Vec<u8>` before spawning.
//!
//! ## Relationship to the central [`crate::compute_3d::BlazenThreeDModel`]
//!
//! The central `BlazenThreeDModel` + `blazen_three_d_model_new_triposr`
//! factory in [`crate::compute_3d`] remain in place — this module is
//! purely additive. The existing
//! [`crate::compute_3d::blazen_future_take_three_d_generate_result`]
//! taker is reused for `_generate_from_image` futures here too; no
//! per-engine result type is introduced for the generation surface.

#![cfg(feature = "triposr")]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::compute::ThreeDGenerateResult as InnerThreeDGenerateResult;
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::cstr_to_opt_string;

// ---------------------------------------------------------------------------
// Local error helpers (mirror crate::tts / crate::compute_3d)
// ---------------------------------------------------------------------------

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

/// Copy a borrowed C buffer into an owned `Vec<u8>`. Null pointer with a
/// non-zero length is rejected by the calling site; null with `len == 0`
/// produces an empty `Vec`.
///
/// # Safety
///
/// `ptr` must be null OR point to a buffer of at least `len` bytes that
/// remains valid for the duration of this call.
#[inline]
unsafe fn copy_bytes(ptr: *const u8, len: usize) -> Vec<u8> {
    if len == 0 {
        Vec::new()
    } else {
        // SAFETY: caller guarantees `ptr` is valid for `len` bytes.
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }
}

// ===========================================================================
// TripoSrProvider — native candle image-to-3D (feature = "triposr")
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::three_d::TripoSrProvider>`.
///
/// Free with [`blazen_triposr_provider_free`].
pub struct BlazenTripoSrProvider(pub(crate) Arc<blazen_uniffi::concrete::three_d::TripoSrProvider>);

impl BlazenTripoSrProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenTripoSrProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `BlazenTripoSrProvider`. All three inputs are optional —
/// pass null to defer to the upstream defaults (`HuggingFace` download
/// from `"stabilityai/TripoSR"` at `main`).
///
/// Construction may block on a `HuggingFace` download (the inner
/// constructor is async and is driven through the shared cabi runtime).
///
/// On success returns `0` and writes a fresh `BlazenTripoSrProvider*`
/// into `*out_model`. On failure returns `-1` and writes a fresh
/// `BlazenError*` into `*out_err`. Both out-parameters may be null to
/// discard that side of the result.
///
/// # Safety
///
/// - `hf_repo_id` / `revision` / `weights_path` must each be null OR a
///   valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable
///   slot of the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_triposr_provider_new(
    hf_repo_id: *const c_char,
    revision: *const c_char,
    weights_path: *const c_char,
    out_model: *mut *mut BlazenTripoSrProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let hf_repo_id = unsafe { cstr_to_opt_string(hf_repo_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let revision = unsafe { cstr_to_opt_string(revision) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let weights_path = unsafe { cstr_to_opt_string(weights_path) };

    match blazen_uniffi::concrete::three_d::TripoSrProvider::new(hf_repo_id, revision, weights_path)
    {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenTripoSrProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Asynchronously generate a 3D mesh from a single input image. Returns
/// a `*mut BlazenFuture` the caller polls / waits on; the typed result
/// is popped with
/// [`crate::compute_3d::blazen_future_take_three_d_generate_result`].
///
/// `image_bytes` is encoded PNG or JPEG payload. `mesh_resolution`
/// controls the side length of the density grid sampled from the
/// triplane during marching cubes; `256` matches the upstream
/// `TripoSR` reference and is a reasonable default.
///
/// Returns null if `model` is null OR `image_bytes` is null with a
/// non-zero `image_bytes_len`.
///
/// # Safety
///
/// - `model` must be a valid pointer to a `BlazenTripoSrProvider`
///   produced by [`blazen_triposr_provider_new`].
/// - `image_bytes` must be null OR point to a buffer of at least
///   `image_bytes_len` bytes (null with `image_bytes_len == 0` is
///   accepted and yields an empty payload).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_triposr_provider_generate_from_image(
    model: *const BlazenTripoSrProvider,
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
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: see contract above.
    let bytes_vec = unsafe { copy_bytes(image_bytes, image_bytes_len) };

    let inner = Arc::clone(&model.0);
    BlazenFuture::spawn::<InnerThreeDGenerateResult, _>(async move {
        Arc::clone(&inner)
            .generate_from_image(bytes_vec, mesh_resolution)
            .await
    })
}

/// Synchronous variant of
/// [`blazen_triposr_provider_generate_from_image`].
///
/// Returns `0` on success (typed result in `*out_result`), `-1` on
/// failure (error in `*out_err`), `-2` on invalid input (null model /
/// null `image_bytes` with non-zero len).
///
/// # Safety
///
/// Same buffer contracts as
/// [`blazen_triposr_provider_generate_from_image`]. `out_result` /
/// `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_triposr_provider_generate_from_image_blocking(
    model: *const BlazenTripoSrProvider,
    image_bytes: *const u8,
    image_bytes_len: usize,
    mesh_resolution: u32,
    out_result: *mut *mut crate::compute_3d::BlazenThreeDGenerateResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        return -2;
    }
    if image_bytes.is_null() && image_bytes_len != 0 {
        return -2;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let model = unsafe { &*model };
    // SAFETY: see contract above.
    let bytes_vec = unsafe { copy_bytes(image_bytes, image_bytes_len) };

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
                    *out_result = crate::compute_3d::BlazenThreeDGenerateResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => write_error(out_err, e),
    }
}

/// Frees a `BlazenTripoSrProvider` handle. No-op on a null pointer.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_triposr_provider_new`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_triposr_provider_free(model: *mut BlazenTripoSrProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
