//! Per-engine 3D provider opaques + C ABI factories.
//!
//! Mirrors the structural template in [`crate::tts`] for the two 3D
//! engines exported by [`blazen_uniffi::concrete::three_d`]:
//!
//! - [`BlazenTripoSrProvider`] (gated on `triposr`) — native candle
//!   image-to-3D. Generation-only; pipe the produced GLB through a
//!   post-proc backend ([`BlazenCompat3dProvider`]) for texturize /
//!   rig / refine / animate.
//! - [`BlazenCompat3dProvider`] (gated on `threed-compat-proxy`) — the
//!   HTTP-proxy post-processing backend. Post-processing-only;
//!   `generate_from_image` surfaces `Unsupported`. Wraps the unified
//!   [`blazen_uniffi::concrete::three_d::Compat3dProvider`].
//!
//! The two engines share no FFI symbols — the `blazen_triposr_*` and
//! `blazen_compat3d_*` surfaces are independent and each gated on its
//! own feature. The module compiles whenever EITHER feature is on.
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<…Provider>` returned by
//!   the matching `…_provider_new` factory. Callers free with the
//!   matching `…_provider_free`. Double-free is undefined behavior.
//! - String inputs are borrowed for the duration of the call only —
//!   the wrapper copies out everything it needs into owned `String`s
//!   before spawning the underlying task.
//! - Byte-slice inputs (`image_bytes` + `image_bytes_len`, mesh GLB,
//!   reference image, driving video, BVH motion) are borrowed for the
//!   duration of the call only — the wrapper copies them into owned
//!   `Vec<u8>`s before spawning.
//!
//! ## Relationship to the central [`crate::compute_3d::BlazenThreeDModel`]
//!
//! The central `BlazenThreeDModel` + `blazen_three_d_model_new_triposr`
//! factory in [`crate::compute_3d`] remain in place — this module is
//! purely additive. The existing
//! [`crate::compute_3d::blazen_future_take_three_d_generate_result`]
//! taker is reused for `_generate_from_image` futures here too; no
//! per-engine result type is introduced for the generation surface.
//!
//! ## Compat3d wire format
//!
//! `Compat3dProvider` request "knobs" cross the boundary as a
//! NUL-terminated UTF-8 JSON string mirroring the relevant
//! [`blazen_uniffi::concrete::three_d`] record fields. Binary payloads
//! (mesh GLB, reference image, driving video, BVH motion) cross as
//! `(*const u8, usize)` borrowed slices. The Rust side copies bytes and
//! parses the JSON before spawning the underlying async task, so
//! callers may free their input buffers as soon as the C entry point
//! returns (sync flavour) / as soon as the future handle is in hand
//! (async flavour). Result bytes are owned by the
//! [`BlazenCompat3dResult`] handle and released wholesale by
//! [`blazen_compat3d_result_free`]; copy what you need before freeing.

#![cfg(any(feature = "triposr", feature = "threed-compat-proxy"))]
#![allow(dead_code)] // crate-private helpers are linker-preserved on the cdylib

use std::ffi::c_char;
use std::sync::Arc;

#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
pub struct BlazenTripoSrProvider(pub(crate) Arc<blazen_uniffi::concrete::three_d::TripoSrProvider>);

#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_triposr_provider_free(model: *mut BlazenTripoSrProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// Compat3dProvider — HTTP-proxy 3D post-processing (feature = "threed-compat-proxy")
// ===========================================================================
//
// Folded from the former `crate::threed` module (P4.2.x.3.three_d). The
// `blazen_compat3d_*` C ABI symbol names are preserved verbatim so the
// Ruby gem (`bindings/ruby/lib/blazen/{threed,ffi}.rb`) keeps working
// without changes. Wraps the unified
// `blazen_uniffi::concrete::three_d::Compat3dProvider`.

#[cfg(feature = "threed-compat-proxy")]
use blazen_uniffi::concrete::three_d::{
    AnimateRequest, AnimateResult, Compat3dProvider as InnerCompat3dProvider, RefineRequest,
    RefineResult, RigRequest, RigResult, TexturizeRequest, TexturizeResult,
};
#[cfg(feature = "threed-compat-proxy")]
use serde::Deserialize;

#[cfg(feature = "threed-compat-proxy")]
use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// JSON request shapes
// ---------------------------------------------------------------------------
//
// Mirror `blazen_uniffi::concrete::three_d::{Texturize,Rig,Refine,Animate}Request`
// minus the binary fields, which the cabi accepts out-of-band as
// `(ptr, len)` byte slices. The JSON shapes are deliberately permissive
// (all fields default to `None` / sensible falsy values) so callers can
// pass `"{}"` for a no-knobs request.

#[cfg(feature = "threed-compat-proxy")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TexturizeRequestJson {
    prompt: Option<String>,
    style: Option<String>,
    resolution: Option<u32>,
    pbr: bool,
}

#[cfg(feature = "threed-compat-proxy")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RigRequestJson {
    template: Option<String>,
    skin: bool,
    pose_hint: Option<String>,
}

#[cfg(feature = "threed-compat-proxy")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RefineRequestJson {
    decimate_target_tris: Option<u32>,
    fill_holes: bool,
    unwrap_uvs: bool,
    retopologize: bool,
    smooth_iterations: Option<u32>,
}

#[cfg(feature = "threed-compat-proxy")]
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AnimateRequestJson {
    prompt: Option<String>,
    duration_seconds: Option<f32>,
    fps: Option<u32>,
    loop_animation: bool,
}

/// Parse a NUL-terminated UTF-8 JSON string into `T`, returning an
/// `InvalidInput`-flavoured `InnerError` on any parse failure. An
/// empty string OR a null pointer collapses to `T::default()` so
/// callers can pass `null` / `""` for a no-knobs request.
///
/// `InnerError` is large (~168 bytes); allow the lint here because every
/// consumer immediately funnels it into a `Box<BlazenError>` via
/// `BlazenError::into_ptr` before crossing the FFI boundary, so the
/// large temporary doesn't actually live on a hot path.
#[cfg(feature = "threed-compat-proxy")]
#[allow(clippy::result_large_err)]
fn parse_json_or_default<T>(json: *const c_char, what: &str) -> Result<T, InnerError>
where
    T: Default + for<'de> Deserialize<'de>,
{
    // SAFETY: caller-supplied — `cstr_to_str` upholds the contract.
    let s = unsafe { cstr_to_str(json) };
    match s {
        None | Some("") => Ok(T::default()),
        Some(s) => serde_json::from_str::<T>(s).map_err(|e| InnerError::Validation {
            message: format!("invalid {what} request JSON: {e}"),
        }),
    }
}

/// Materialise a `Vec<u8>` from a borrowed `(*const u8, usize)` pair.
/// Treats a null pointer as an empty vec (matches the upstream
/// `Compat3dProvider` semantics — every binary input is conceptually
/// `&[u8]`, so an empty slice is the sensible zero-arg value).
///
/// # Safety
///
/// `ptr` must be null OR point at a buffer of at least `len` bytes that
/// remains valid for the duration of this call.
#[cfg(feature = "threed-compat-proxy")]
unsafe fn bytes_to_vec(ptr: *const u8, len: usize) -> Vec<u8> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: caller-supplied pointer + length describe a live buffer.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

// ---------------------------------------------------------------------------
// Compat3dProvider opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`blazen_uniffi::concrete::three_d::Compat3dProvider`].
///
/// Produced by [`blazen_compat3d_provider_new`]; freed by
/// [`blazen_compat3d_provider_free`]. The handle is `Send + Sync` — the
/// inner [`Arc`] is cloned into each spawned task so the provider stays
/// alive for the duration of the async call without pinning the caller's
/// stack-local handle.
#[cfg(feature = "threed-compat-proxy")]
pub struct BlazenCompat3dProvider(pub(crate) Arc<InnerCompat3dProvider>);

/// Construct an HTTP-proxy 3D provider pointed at `base_url`.
///
/// `api_key` and `timeout_secs` are optional — pass null / `0` to take
/// the defaults (no bearer auth, 600-second per-request timeout). When
/// `timeout_secs` is non-zero the inner client is rebuilt with that
/// timeout via `with_timeout`.
///
/// Returns null if `base_url` is null, not valid UTF-8, or contains an
/// interior NUL byte. Otherwise returns a caller-owned pointer that
/// must be released with [`blazen_compat3d_provider_free`].
///
/// # Safety
///
/// `base_url` must be a NUL-terminated UTF-8 buffer that remains valid
/// for the duration of this call (the string is copied before return).
/// `api_key` must be null OR a NUL-terminated UTF-8 buffer with the same
/// liveness requirement.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_provider_new(
    base_url: *const c_char,
    api_key: *const c_char,
    timeout_secs: u32,
) -> *mut BlazenCompat3dProvider {
    // SAFETY: caller upholds the NUL + UTF-8 contract.
    let Some(base) = (unsafe { cstr_to_str(base_url) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the NUL + UTF-8 contract.
    let key = unsafe { cstr_to_opt_string(api_key) };
    let timeout = if timeout_secs == 0 {
        None
    } else {
        Some(timeout_secs)
    };
    let inner = InnerCompat3dProvider::new(base.to_owned(), key, timeout);
    Box::into_raw(Box::new(BlazenCompat3dProvider(inner)))
}

/// Frees a [`BlazenCompat3dProvider`] previously produced by the cabi
/// surface. No-op when `provider` is null.
///
/// # Safety
///
/// `provider` must be null OR a pointer previously produced by
/// [`blazen_compat3d_provider_new`]. Double-free is undefined behavior.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_provider_free(provider: *mut BlazenCompat3dProvider) {
    if provider.is_null() {
        return;
    }
    // SAFETY: per the contract above, `provider` was produced by
    // `Box::into_raw` over a `BlazenCompat3dProvider`.
    drop(unsafe { Box::from_raw(provider) });
}

// ---------------------------------------------------------------------------
// Result handle
// ---------------------------------------------------------------------------

/// Internal discriminant for the four stages. The C side never inspects
/// this directly — it's used only to keep the stage-specific accessors
/// from blowing up if a caller mixes accessors across stages (e.g.
/// calling `_bone_names_count` on a `Texturize` result).
#[cfg(feature = "threed-compat-proxy")]
#[derive(Debug, Clone)]
enum ResultInner {
    Texturize(TexturizeResult),
    Rig(RigResult),
    Refine(RefineResult),
    Animate(AnimateResult),
}

/// Opaque handle holding the result of any of the four stages.
///
/// Produced by the four `blazen_compat3d_*_blocking` wrappers (out
/// param) or by the matching [`blazen_future_take_compat3d_result`] taker.
/// Free with [`blazen_compat3d_result_free`]; do NOT call any stage-specific
/// accessor after freeing.
#[cfg(feature = "threed-compat-proxy")]
pub struct BlazenCompat3dResult(ResultInner);

#[cfg(feature = "threed-compat-proxy")]
impl BlazenCompat3dResult {
    fn into_ptr(self) -> *mut BlazenCompat3dResult {
        Box::into_raw(Box::new(self))
    }
}

/// Frees a [`BlazenCompat3dResult`] handle. No-op on null. Double-free is
/// undefined behavior.
///
/// # Safety
///
/// `result` must be null OR a pointer previously produced by the cabi
/// 3D surface (`blazen_compat3d_*_blocking` out-param or
/// [`blazen_future_take_compat3d_result`]).
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_free(result: *mut BlazenCompat3dResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: per the contract above, produced by `Box::into_raw`.
    drop(unsafe { Box::from_raw(result) });
}

/// Returns the produced GLB bytes for any stage.
///
/// Writes the raw pointer into `*out_ptr` (borrow into the result
/// handle's internal `Vec<u8>`) and the length into `*out_len`. The
/// borrow is valid until [`blazen_compat3d_result_free`] is called on the
/// handle. Returns `0` on success, `-1` on null `result`.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
/// `out_ptr` / `out_len` must be null OR valid single-writer
/// destinations for `*const u8` and `usize` respectively.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_glb_bytes(
    result: *const BlazenCompat3dResult,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    if result.is_null() {
        return -1;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    let bytes: &[u8] = match &result.0 {
        ResultInner::Texturize(r) => &r.textured_glb,
        ResultInner::Rig(r) => &r.rigged_glb,
        ResultInner::Refine(r) => &r.refined_glb,
        ResultInner::Animate(r) => &r.animated_glb,
    };
    if !out_ptr.is_null() {
        // SAFETY: caller-supplied out-param.
        unsafe {
            *out_ptr = bytes.as_ptr();
        }
    }
    if !out_len.is_null() {
        // SAFETY: caller-supplied out-param.
        unsafe {
            *out_len = bytes.len();
        }
    }
    0
}

/// Returns the MIME type for any stage as a heap-allocated NUL-terminated
/// UTF-8 C string. Caller frees with `blazen_string_free`. Returns null
/// if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_mime_type(
    result: *const BlazenCompat3dResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    let mime: &str = match &result.0 {
        ResultInner::Texturize(r) => &r.mime_type,
        ResultInner::Rig(r) => &r.mime_type,
        ResultInner::Refine(r) => &r.mime_type,
        ResultInner::Animate(r) => &r.mime_type,
    };
    alloc_cstring(mime)
}

// ---- Texturize-only PBR accessors ----------------------------------------

/// Returns `1` if this result is a `Texturize` result carrying a PBR
/// bundle, `0` otherwise (or if `result` is null).
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_has_pbr_maps(
    result: *const BlazenCompat3dResult,
) -> i32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Texturize(r) if r.pbr_maps.is_some() => 1,
        _ => 0,
    }
}

/// PBR-channel discriminants accepted by
/// [`blazen_compat3d_result_pbr_map_bytes`].
#[cfg(feature = "threed-compat-proxy")]
pub const BLAZEN_PBR_MAP_ALBEDO: u32 = 0;
#[cfg(feature = "threed-compat-proxy")]
pub const BLAZEN_PBR_MAP_NORMAL: u32 = 1;
#[cfg(feature = "threed-compat-proxy")]
pub const BLAZEN_PBR_MAP_ROUGHNESS: u32 = 2;
#[cfg(feature = "threed-compat-proxy")]
pub const BLAZEN_PBR_MAP_METALLIC: u32 = 3;

/// Borrows the bytes of a single PBR channel for a `Texturize` result.
///
/// `channel` is one of the `BLAZEN_PBR_MAP_*` constants. Writes the
/// borrowed pointer into `*out_ptr` and length into `*out_len`. The
/// borrow is valid until [`blazen_compat3d_result_free`] is called.
///
/// Returns `0` if the channel is populated, `1` if the result is not a
/// `Texturize` result / has no PBR bundle / the requested channel is
/// `None`, `-1` if `result` is null or `channel` is out of range. The
/// `1` return path zeros `*out_len` so callers can early-return on
/// "channel not present".
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
/// `out_ptr` / `out_len` must be null OR valid single-writer
/// destinations.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_pbr_map_bytes(
    result: *const BlazenCompat3dResult,
    channel: u32,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    if result.is_null() {
        return -1;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    let ResultInner::Texturize(r) = &result.0 else {
        if !out_len.is_null() {
            // SAFETY: out-param contract.
            unsafe {
                *out_len = 0;
            }
        }
        return 1;
    };
    let Some(maps) = r.pbr_maps.as_ref() else {
        if !out_len.is_null() {
            // SAFETY: out-param contract.
            unsafe {
                *out_len = 0;
            }
        }
        return 1;
    };
    let bytes: Option<&[u8]> = match channel {
        BLAZEN_PBR_MAP_ALBEDO => Some(maps.albedo_png.as_slice()),
        BLAZEN_PBR_MAP_NORMAL => maps.normal_png.as_deref(),
        BLAZEN_PBR_MAP_ROUGHNESS => maps.roughness_png.as_deref(),
        BLAZEN_PBR_MAP_METALLIC => maps.metallic_png.as_deref(),
        _ => return -1,
    };
    if let Some(slice) = bytes {
        if !out_ptr.is_null() {
            // SAFETY: out-param contract.
            unsafe {
                *out_ptr = slice.as_ptr();
            }
        }
        if !out_len.is_null() {
            // SAFETY: out-param contract.
            unsafe {
                *out_len = slice.len();
            }
        }
        0
    } else {
        if !out_len.is_null() {
            // SAFETY: out-param contract.
            unsafe {
                *out_len = 0;
            }
        }
        1
    }
}

// ---- Rig-only bone-name accessors ----------------------------------------

/// Returns the number of bone names carried by a `Rig` result, or `0`
/// for any other result variant / null pointer.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_bone_names_count(
    result: *const BlazenCompat3dResult,
) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Rig(r) => r.bone_names.len(),
        _ => 0,
    }
}

/// Returns a heap-allocated NUL-terminated UTF-8 C string for the bone
/// name at `index` (caller frees with `blazen_string_free`). Returns
/// null if `result` isn't a `Rig` result, `index` is out of bounds, or
/// `result` is null.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_bone_name_get(
    result: *const BlazenCompat3dResult,
    index: usize,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    let ResultInner::Rig(r) = &result.0 else {
        return std::ptr::null_mut();
    };
    r.bone_names
        .get(index)
        .map_or(std::ptr::null_mut(), |s| alloc_cstring(s))
}

// ---- Refine-only stat accessors ------------------------------------------

/// Returns the input triangle count for a `Refine` result, or `0` for
/// any other variant / null pointer.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_refine_input_tri_count(
    result: *const BlazenCompat3dResult,
) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Refine(r) => r.stats.input_tri_count,
        _ => 0,
    }
}

/// Returns the output triangle count for a `Refine` result, or `0` for
/// any other variant / null pointer.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_refine_output_tri_count(
    result: *const BlazenCompat3dResult,
) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Refine(r) => r.stats.output_tri_count,
        _ => 0,
    }
}

/// Returns the UV chart count for a `Refine` result. Returns `-1` when
/// the field is `None` (UV unwrapping wasn't requested), the result
/// isn't a `Refine` variant, or `result` is null.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_refine_uv_chart_count(
    result: *const BlazenCompat3dResult,
) -> i64 {
    if result.is_null() {
        return -1;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Refine(r) => r.stats.uv_chart_count.map_or(-1, i64::from),
        _ => -1,
    }
}

// ---- Animate-only knobs --------------------------------------------------

/// Returns the produced animation duration in seconds for an `Animate`
/// result. Returns `0.0` for any other variant / null pointer.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_animate_duration_seconds(
    result: *const BlazenCompat3dResult,
) -> f32 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Animate(r) => r.duration_seconds,
        _ => 0.0,
    }
}

/// Returns the produced animation FPS for an `Animate` result. Returns
/// `0` for any other variant / null pointer.
///
/// # Safety
///
/// `result` must be null OR a live `BlazenCompat3dResult` pointer.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_result_animate_fps(
    result: *const BlazenCompat3dResult,
) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract.
    let result = unsafe { &*result };
    match &result.0 {
        ResultInner::Animate(r) => r.fps,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Compat3d error-out helpers
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`.
///
/// # Safety
///
/// `out_err` must be null OR a valid single-writer destination for a
/// `*mut BlazenError` value.
#[cfg(feature = "threed-compat-proxy")]
unsafe fn compat3d_write_error(out_err: *mut *mut BlazenError, e: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: out-param contract upheld by caller.
        unsafe {
            *out_err = BlazenError::from(e).into_ptr();
        }
    }
    -1
}

/// Writes a synthesised `Internal` error to the out-param and returns
/// `-1`. Used for null-pointer / UTF-8 input failures.
///
/// # Safety
///
/// Same as [`compat3d_write_error`].
#[cfg(feature = "threed-compat-proxy")]
unsafe fn compat3d_write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded to compat3d_write_error.
    unsafe {
        compat3d_write_error(
            out_err,
            InnerError::Internal {
                message: msg.into(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Texturize
// ---------------------------------------------------------------------------

/// Synchronously runs the `texturize` stage.
///
/// Returns `0` on success (writing a caller-owned `*mut BlazenCompat3dResult`
/// to `out_result`) or `-1` on failure (writing the inner error to
/// `out_err`).
///
/// `request_json` may be null or empty for an all-defaults request; see
/// the module-level docs for the JSON shape. `mesh_ptr` / `mesh_len`
/// describe the input mesh GLB bytes; `reference_image_ptr` /
/// `reference_image_len` describe the optional reference image PNG/JPEG
/// bytes (null / `0` for no reference).
///
/// # Safety
///
/// `provider` must be a valid pointer to a `BlazenCompat3dProvider`.
/// `mesh_ptr` must be null OR point to a buffer of at least `mesh_len`
/// bytes. Same for `reference_image_ptr` / `reference_image_len`.
/// `request_json` follows the standard NUL+UTF-8 contract. `out_result`
/// / `out_err` are null OR valid single-writer destinations.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_texturize_blocking(
    provider: *const BlazenCompat3dProvider,
    mesh_ptr: *const u8,
    mesh_len: usize,
    reference_image_ptr: *const u8,
    reference_image_len: usize,
    request_json: *const c_char,
    out_result: *mut *mut BlazenCompat3dResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        // SAFETY: out-err contract.
        return unsafe { compat3d_write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: TexturizeRequestJson = match parse_json_or_default(request_json, "texturize") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { compat3d_write_error(out_err, e) },
    };
    // SAFETY: caller upholds the (ptr,len) liveness contract.
    let mesh = unsafe { bytes_to_vec(mesh_ptr, mesh_len) };
    // SAFETY: same.
    let reference_image = unsafe { bytes_to_vec(reference_image_ptr, reference_image_len) };
    let request = TexturizeRequest {
        prompt: req_json.prompt,
        reference_image: if reference_image.is_empty() {
            None
        } else {
            Some(reference_image)
        },
        style: req_json.style,
        resolution: req_json.resolution,
        pbr: req_json.pbr,
    };
    let inner = Arc::clone(&provider.0);
    match runtime().block_on(async move { inner.texturize(mesh, request).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_result = BlazenCompat3dResult(ResultInner::Texturize(result)).into_ptr();
                }
            }
            0
        }
        // SAFETY: out-err contract.
        Err(e) => unsafe { compat3d_write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_texturize_blocking`].
///
/// Returns a `BlazenFuture *` immediately (or null on null inputs / JSON
/// parse failures). The result resolves via
/// [`blazen_future_take_compat3d_result`].
///
/// # Safety
///
/// Same per-pointer contract as [`blazen_compat3d_texturize_blocking`];
/// every input buffer is copied before the function returns, so callers
/// may release their buffers as soon as this returns.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_texturize(
    provider: *const BlazenCompat3dProvider,
    mesh_ptr: *const u8,
    mesh_len: usize,
    reference_image_ptr: *const u8,
    reference_image_len: usize,
    request_json: *const c_char,
) -> *mut BlazenFuture {
    if provider.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: TexturizeRequestJson = match parse_json_or_default(request_json, "texturize") {
        Ok(r) => r,
        // Surface the parse error through a ready-resolved future so
        // foreign hosts get the same shape as the blocking variant.
        Err(e) => return BlazenFuture::spawn(async move { Err::<ResultInner, _>(e) }),
    };
    // SAFETY: caller-supplied (ptr,len) contract.
    let mesh = unsafe { bytes_to_vec(mesh_ptr, mesh_len) };
    // SAFETY: same.
    let reference_image = unsafe { bytes_to_vec(reference_image_ptr, reference_image_len) };
    let request = TexturizeRequest {
        prompt: req_json.prompt,
        reference_image: if reference_image.is_empty() {
            None
        } else {
            Some(reference_image)
        },
        style: req_json.style,
        resolution: req_json.resolution,
        pbr: req_json.pbr,
    };
    let inner = Arc::clone(&provider.0);
    BlazenFuture::spawn(async move {
        inner
            .texturize(mesh, request)
            .await
            .map(ResultInner::Texturize)
    })
}

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

/// Synchronously runs the `rig` stage. See
/// [`blazen_compat3d_texturize_blocking`] for the shared semantics; the
/// `rig`-specific request fields are `template` / `skin` / `pose_hint`.
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize_blocking`].
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_rig_blocking(
    provider: *const BlazenCompat3dProvider,
    mesh_ptr: *const u8,
    mesh_len: usize,
    request_json: *const c_char,
    out_result: *mut *mut BlazenCompat3dResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        // SAFETY: out-err contract.
        return unsafe { compat3d_write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: RigRequestJson = match parse_json_or_default(request_json, "rig") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { compat3d_write_error(out_err, e) },
    };
    // SAFETY: caller-supplied (ptr,len) contract.
    let mesh = unsafe { bytes_to_vec(mesh_ptr, mesh_len) };
    let request = RigRequest {
        template: req_json.template,
        skin: req_json.skin,
        pose_hint: req_json.pose_hint,
    };
    let inner = Arc::clone(&provider.0);
    match runtime().block_on(async move { inner.rig(mesh, request).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_result = BlazenCompat3dResult(ResultInner::Rig(result)).into_ptr();
                }
            }
            0
        }
        // SAFETY: out-err contract.
        Err(e) => unsafe { compat3d_write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_rig_blocking`].
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize`].
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_rig(
    provider: *const BlazenCompat3dProvider,
    mesh_ptr: *const u8,
    mesh_len: usize,
    request_json: *const c_char,
) -> *mut BlazenFuture {
    if provider.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: RigRequestJson = match parse_json_or_default(request_json, "rig") {
        Ok(r) => r,
        Err(e) => return BlazenFuture::spawn(async move { Err::<ResultInner, _>(e) }),
    };
    // SAFETY: caller-supplied (ptr,len) contract.
    let mesh = unsafe { bytes_to_vec(mesh_ptr, mesh_len) };
    let request = RigRequest {
        template: req_json.template,
        skin: req_json.skin,
        pose_hint: req_json.pose_hint,
    };
    let inner = Arc::clone(&provider.0);
    BlazenFuture::spawn(async move { inner.rig(mesh, request).await.map(ResultInner::Rig) })
}

// ---------------------------------------------------------------------------
// Refine
// ---------------------------------------------------------------------------

/// Synchronously runs the `refine` stage.
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize_blocking`].
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_refine_blocking(
    provider: *const BlazenCompat3dProvider,
    mesh_ptr: *const u8,
    mesh_len: usize,
    request_json: *const c_char,
    out_result: *mut *mut BlazenCompat3dResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        // SAFETY: out-err contract.
        return unsafe { compat3d_write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: RefineRequestJson = match parse_json_or_default(request_json, "refine") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { compat3d_write_error(out_err, e) },
    };
    // SAFETY: caller-supplied (ptr,len) contract.
    let mesh = unsafe { bytes_to_vec(mesh_ptr, mesh_len) };
    let request = RefineRequest {
        decimate_target_tris: req_json.decimate_target_tris,
        fill_holes: req_json.fill_holes,
        unwrap_uvs: req_json.unwrap_uvs,
        retopologize: req_json.retopologize,
        smooth_iterations: req_json.smooth_iterations,
    };
    let inner = Arc::clone(&provider.0);
    match runtime().block_on(async move { inner.refine(mesh, request).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_result = BlazenCompat3dResult(ResultInner::Refine(result)).into_ptr();
                }
            }
            0
        }
        // SAFETY: out-err contract.
        Err(e) => unsafe { compat3d_write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_refine_blocking`].
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize`].
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_refine(
    provider: *const BlazenCompat3dProvider,
    mesh_ptr: *const u8,
    mesh_len: usize,
    request_json: *const c_char,
) -> *mut BlazenFuture {
    if provider.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: RefineRequestJson = match parse_json_or_default(request_json, "refine") {
        Ok(r) => r,
        Err(e) => return BlazenFuture::spawn(async move { Err::<ResultInner, _>(e) }),
    };
    // SAFETY: caller-supplied (ptr,len) contract.
    let mesh = unsafe { bytes_to_vec(mesh_ptr, mesh_len) };
    let request = RefineRequest {
        decimate_target_tris: req_json.decimate_target_tris,
        fill_holes: req_json.fill_holes,
        unwrap_uvs: req_json.unwrap_uvs,
        retopologize: req_json.retopologize,
        smooth_iterations: req_json.smooth_iterations,
    };
    let inner = Arc::clone(&provider.0);
    BlazenFuture::spawn(async move { inner.refine(mesh, request).await.map(ResultInner::Refine) })
}

// ---------------------------------------------------------------------------
// Animate
// ---------------------------------------------------------------------------

/// Synchronously runs the `animate` stage. `rigged_glb` is the
/// already-rigged input mesh; `driving_video_ptr` / `bvh_motion_ptr` are
/// the optional motion-source byte buffers (null / `0` to omit each).
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize_blocking`], plus
/// the `driving_video_ptr` / `bvh_motion_ptr` pointers follow the same
/// (ptr,len) liveness rule.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)] // multiple optional binary inputs
pub unsafe extern "C" fn blazen_compat3d_animate_blocking(
    provider: *const BlazenCompat3dProvider,
    rigged_glb_ptr: *const u8,
    rigged_glb_len: usize,
    driving_video_ptr: *const u8,
    driving_video_len: usize,
    bvh_motion_ptr: *const u8,
    bvh_motion_len: usize,
    request_json: *const c_char,
    out_result: *mut *mut BlazenCompat3dResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        // SAFETY: out-err contract.
        return unsafe { compat3d_write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: AnimateRequestJson = match parse_json_or_default(request_json, "animate") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { compat3d_write_error(out_err, e) },
    };
    // SAFETY: caller-supplied (ptr,len) contract for each input.
    let rigged = unsafe { bytes_to_vec(rigged_glb_ptr, rigged_glb_len) };
    // SAFETY: same.
    let driving = unsafe { bytes_to_vec(driving_video_ptr, driving_video_len) };
    // SAFETY: same.
    let bvh = unsafe { bytes_to_vec(bvh_motion_ptr, bvh_motion_len) };
    let request = AnimateRequest {
        prompt: req_json.prompt,
        driving_video: if driving.is_empty() {
            None
        } else {
            Some(driving)
        },
        bvh_motion: if bvh.is_empty() { None } else { Some(bvh) },
        duration_seconds: req_json.duration_seconds,
        fps: req_json.fps,
        loop_animation: req_json.loop_animation,
    };
    let inner = Arc::clone(&provider.0);
    match runtime().block_on(async move { inner.animate(rigged, request).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out_result = BlazenCompat3dResult(ResultInner::Animate(result)).into_ptr();
                }
            }
            0
        }
        // SAFETY: out-err contract.
        Err(e) => unsafe { compat3d_write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_animate_blocking`].
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize`], extended to the
/// `driving_video_ptr` / `bvh_motion_ptr` inputs.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)] // multiple optional binary inputs
pub unsafe extern "C" fn blazen_compat3d_animate(
    provider: *const BlazenCompat3dProvider,
    rigged_glb_ptr: *const u8,
    rigged_glb_len: usize,
    driving_video_ptr: *const u8,
    driving_video_len: usize,
    bvh_motion_ptr: *const u8,
    bvh_motion_len: usize,
    request_json: *const c_char,
) -> *mut BlazenFuture {
    if provider.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: AnimateRequestJson = match parse_json_or_default(request_json, "animate") {
        Ok(r) => r,
        Err(e) => return BlazenFuture::spawn(async move { Err::<ResultInner, _>(e) }),
    };
    // SAFETY: caller-supplied (ptr,len) contract.
    let rigged = unsafe { bytes_to_vec(rigged_glb_ptr, rigged_glb_len) };
    // SAFETY: same.
    let driving = unsafe { bytes_to_vec(driving_video_ptr, driving_video_len) };
    // SAFETY: same.
    let bvh = unsafe { bytes_to_vec(bvh_motion_ptr, bvh_motion_len) };
    let request = AnimateRequest {
        prompt: req_json.prompt,
        driving_video: if driving.is_empty() {
            None
        } else {
            Some(driving)
        },
        bvh_motion: if bvh.is_empty() { None } else { Some(bvh) },
        duration_seconds: req_json.duration_seconds,
        fps: req_json.fps,
        loop_animation: req_json.loop_animation,
    };
    let inner = Arc::clone(&provider.0);
    BlazenFuture::spawn(async move {
        inner
            .animate(rigged, request)
            .await
            .map(ResultInner::Animate)
    })
}

// ---------------------------------------------------------------------------
// Generation surface (Unsupported on the compat-proxy backend)
// ---------------------------------------------------------------------------

/// Surfaces the `Unsupported` generation error for the compat-proxy
/// backend. The HTTP-proxy wire contract only covers texturize / rig /
/// refine / animate — base generation must come from a separate backend
/// such as [`BlazenTripoSrProvider`]. Always returns `-1` and writes a
/// `BlazenError::Unsupported` into `out_err`.
///
/// `image_bytes` / `mesh_resolution` are accepted (and ignored) so the
/// signature mirrors the per-engine generation surface across engines.
///
/// # Safety
///
/// `provider` must be null OR a live `BlazenCompat3dProvider` pointer.
/// `image_bytes` must be null OR point to a buffer of at least
/// `image_bytes_len` bytes. `out_err` must be null OR a valid
/// single-writer destination for a `*mut BlazenError`.
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_compat3d_provider_generate_from_image(
    provider: *const BlazenCompat3dProvider,
    _image_bytes: *const u8,
    _image_bytes_len: usize,
    _mesh_resolution: u32,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if provider.is_null() {
        // SAFETY: out-err contract.
        return unsafe { compat3d_write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: out-err contract.
    unsafe {
        compat3d_write_error(
            out_err,
            InnerError::Unsupported {
                message: "Compat3dProvider does not support generate_from_image — the \
                          HTTP-proxy upstream only exposes texturize / rig / refine / \
                          animate. Use a generation backend (e.g. TripoSrProvider) to \
                          produce the base mesh, then forward it through Compat3dProvider's \
                          post-proc methods."
                    .to_string(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Typed future taker
// ---------------------------------------------------------------------------

/// Pops a [`BlazenCompat3dResult`] out of any 3D future produced by the
/// `blazen_compat3d_{texturize,rig,refine,animate}` async wrappers.
/// Returns `0` on success (writing the caller-owned result handle into
/// `*out`), or `-1` on failure (writing the inner error into `*err`).
///
/// # Safety
///
/// `fut` must be null OR a live `BlazenFuture` produced by the cabi
/// 3D async surface, already observed completed via
/// `blazen_future_poll` / `_wait` / `_fd`. `out` / `err` follow the
/// usual single-writer out-param contract (null is OK to discard the
/// matching slot).
#[cfg(feature = "threed-compat-proxy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_compat3d_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenCompat3dResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<ResultInner>(fut) } {
        Ok(inner) => {
            if !out.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *out = BlazenCompat3dResult(inner).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: out-param contract.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests (host-side; no live HTTP, no async runtime needed)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "threed-compat-proxy"))]
mod compat3d_tests {
    use super::*;

    #[test]
    fn parse_texturize_defaults_on_empty_json() {
        let parsed: TexturizeRequestJson =
            parse_json_or_default(std::ptr::null(), "texturize").expect("null defaults");
        assert!(parsed.prompt.is_none());
        assert!(!parsed.pbr);
    }

    #[test]
    fn parse_texturize_unknown_field_is_validation_error() {
        let s = std::ffi::CString::new("{\"bogus\": 1}").unwrap();
        let err = parse_json_or_default::<TexturizeRequestJson>(s.as_ptr(), "texturize")
            .expect_err("unknown field");
        match err {
            InnerError::Validation { message } => assert!(message.contains("bogus")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn provider_new_returns_handle_and_frees_cleanly() {
        let url = std::ffi::CString::new("https://example.test").unwrap();
        // SAFETY: well-formed inputs.
        let ptr = unsafe { blazen_compat3d_provider_new(url.as_ptr(), std::ptr::null(), 0) };
        assert!(!ptr.is_null());
        // SAFETY: produced just above.
        unsafe { blazen_compat3d_provider_free(ptr) };
    }

    #[test]
    fn provider_new_rejects_null_base_url() {
        // SAFETY: null base_url is explicitly handled.
        let ptr = unsafe { blazen_compat3d_provider_new(std::ptr::null(), std::ptr::null(), 0) };
        assert!(ptr.is_null());
    }

    #[test]
    fn pbr_accessor_returns_one_on_non_texturize_result() {
        // Build a Rig result directly (no HTTP) and confirm the PBR
        // accessor reports "not a Texturize variant".
        let result = BlazenCompat3dResult(ResultInner::Rig(RigResult {
            rigged_glb: vec![1, 2, 3],
            mime_type: "model/gltf-binary".into(),
            bone_names: vec!["root".into(), "spine".into()],
        }))
        .into_ptr();
        let mut out_ptr: *const u8 = std::ptr::null();
        let mut out_len: usize = 9999;
        // SAFETY: result was just produced and is live.
        let rc = unsafe {
            blazen_compat3d_result_pbr_map_bytes(
                result,
                BLAZEN_PBR_MAP_ALBEDO,
                &raw mut out_ptr,
                &raw mut out_len,
            )
        };
        assert_eq!(rc, 1);
        assert_eq!(out_len, 0);
        // SAFETY: still live.
        assert_eq!(
            unsafe { blazen_compat3d_result_bone_names_count(result) },
            2
        );
        // SAFETY: live + valid index.
        let name_ptr = unsafe { blazen_compat3d_result_bone_name_get(result, 1) };
        assert!(!name_ptr.is_null());
        // SAFETY: produced by alloc_cstring just above.
        unsafe { crate::string::blazen_string_free(name_ptr) };
        // SAFETY: still live.
        unsafe { blazen_compat3d_result_free(result) };
    }

    #[test]
    fn timeout_secs_zero_is_treated_as_default() {
        // Smoke test: timeout_secs=0 must not panic; it short-circuits to
        // None which `Compat3dProvider::new` interprets as the default
        // 10-minute timeout.
        let url = std::ffi::CString::new("https://example.test").unwrap();
        let key = std::ffi::CString::new("sk-test").unwrap();
        // SAFETY: well-formed inputs.
        let ptr = unsafe { blazen_compat3d_provider_new(url.as_ptr(), key.as_ptr(), 0) };
        assert!(!ptr.is_null());
        // SAFETY: produced just above.
        unsafe { blazen_compat3d_provider_free(ptr) };
    }
}
