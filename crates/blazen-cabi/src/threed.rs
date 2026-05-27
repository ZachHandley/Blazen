//! C-ABI surface for the [`blazen_3d`] HTTP-proxy backend.
//!
//! Exposes [`blazen_uniffi::concrete::three_d::Compat3dProvider`] (a `UniFFI` object
//! wrapping `blazen_3d::backends::compat::Compat3dProvider`) to FFI
//! hosts (the Ruby gem today, future Dart/Crystal/etc.). The surface is
//! deliberately small and consistent across the four 3D-pipeline stages:
//!
//! * Opaque handle [`BlazenCompat3dProvider`] produced by
//!   [`blazen_compat3d_provider_new`].
//! * Four async verbs — `texturize` / `rig` / `refine` / `animate` —
//!   each exposed in two flavours: a `*_blocking` synchronous wrapper
//!   that drives the underlying future on the cabi tokio runtime, and
//!   a future-returning variant whose `BlazenFuture *` resolves via the
//!   matching `blazen_future_take_compat3d_result` taker.
//! * One opaque result type [`BlazenCompat3dResult`] with accessors for
//!   the GLB bytes, MIME type, and stage-specific extras (PBR maps,
//!   bone names, refine stats, animation duration/fps).
//!
//! # Wire format
//!
//! Request "knobs" cross the boundary as a NUL-terminated UTF-8 JSON
//! string mirroring the relevant
//! [`blazen_uniffi::concrete::three_d`] record fields. Binary payloads (mesh GLB,
//! reference image, driving video, BVH motion) cross as
//! `(*const u8, usize)` borrowed slices. The Rust side copies bytes
//! and parses the JSON before spawning the underlying async task, so
//! callers may free their input buffers as soon as the C entry point
//! returns (sync flavour) / as soon as the future handle is in hand
//! (async flavour).
//!
//! Result bytes are owned by the [`BlazenCompat3dResult`] handle and
//! released wholesale by [`blazen_compat3d_result_free`]; copy what you
//! need before freeing.

#![cfg(feature = "threed-compat-proxy")]
#![allow(dead_code)] // crate-private helpers are linker-preserved on the cdylib

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::concrete::three_d::{
    AnimateRequest, AnimateResult, Compat3dProvider as InnerProvider, RefineRequest, RefineResult,
    RigRequest, RigResult, TexturizeRequest, TexturizeResult,
};
use blazen_uniffi::errors::BlazenError as InnerError;
use serde::Deserialize;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// JSON request shapes
// ---------------------------------------------------------------------------
//
// Mirror `blazen_uniffi::concrete::three_d::{Texturize,Rig,Refine,Animate}Request`
// minus the binary fields, which the cabi accepts out-of-band as
// `(ptr, len)` byte slices. The JSON shapes are deliberately permissive
// (all fields default to `None` / sensible falsy values) so callers can
// pass `"{}"` for a no-knobs request.

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TexturizeRequestJson {
    prompt: Option<String>,
    style: Option<String>,
    resolution: Option<u32>,
    pbr: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RigRequestJson {
    template: Option<String>,
    skin: bool,
    pose_hint: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RefineRequestJson {
    decimate_target_tris: Option<u32>,
    fill_holes: bool,
    unwrap_uvs: bool,
    retopologize: bool,
    smooth_iterations: Option<u32>,
}

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
pub struct BlazenCompat3dProvider(pub(crate) Arc<InnerProvider>);

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
    let inner = InnerProvider::new(base.to_owned(), key, timeout);
    Box::into_raw(Box::new(BlazenCompat3dProvider(inner)))
}

/// Frees a [`BlazenCompat3dProvider`] previously produced by the cabi
/// surface. No-op when `provider` is null.
///
/// # Safety
///
/// `provider` must be null OR a pointer previously produced by
/// [`blazen_compat3d_provider_new`]. Double-free is undefined behavior.
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
pub struct BlazenCompat3dResult(ResultInner);

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
pub const BLAZEN_PBR_MAP_ALBEDO: u32 = 0;
pub const BLAZEN_PBR_MAP_NORMAL: u32 = 1;
pub const BLAZEN_PBR_MAP_ROUGHNESS: u32 = 2;
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
// Shared error-out helpers
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`.
///
/// # Safety
///
/// `out_err` must be null OR a valid single-writer destination for a
/// `*mut BlazenError` value.
unsafe fn write_error(out_err: *mut *mut BlazenError, e: InnerError) -> i32 {
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
/// Same as [`write_error`].
unsafe fn write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded to write_error.
    unsafe {
        write_error(
            out_err,
            InnerError::Internal {
                message: msg.into(),
            },
        )
    }
}

// Note: the unified `Compat3dProvider` in `blazen_uniffi::concrete::three_d`
// now returns `BlazenError` (the central uniffi error type) directly
// from every method, so the legacy `ThreeDError` → `InnerError`
// translation that lived here (`threed_err_to_inner`) was retired
// alongside the `crate::threed` module deletion in P4.2.x.3.three_d.

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
        return unsafe { write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: TexturizeRequestJson = match parse_json_or_default(request_json, "texturize") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { write_error(out_err, e) },
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
        Err(e) => unsafe { write_error(out_err, e) },
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
        return unsafe { write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: RigRequestJson = match parse_json_or_default(request_json, "rig") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { write_error(out_err, e) },
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
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_rig_blocking`].
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize`].
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
        return unsafe { write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: RefineRequestJson = match parse_json_or_default(request_json, "refine") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { write_error(out_err, e) },
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
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_refine_blocking`].
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize`].
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
        return unsafe { write_internal_error(out_err, "null provider pointer") };
    }
    // SAFETY: caller has guaranteed `provider` is a live pointer.
    let provider = unsafe { &*provider };
    let req_json: AnimateRequestJson = match parse_json_or_default(request_json, "animate") {
        Ok(r) => r,
        // SAFETY: out-err contract.
        Err(e) => return unsafe { write_error(out_err, e) },
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
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Future-returning variant of [`blazen_compat3d_animate_blocking`].
///
/// # Safety
///
/// Same contract as [`blazen_compat3d_texturize`], extended to the
/// `driving_video_ptr` / `bvh_motion_ptr` inputs.
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

#[cfg(test)]
mod tests {
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
