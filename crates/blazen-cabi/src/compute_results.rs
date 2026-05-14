//! Compute result marshalling. Opaque handles wrap the seven typed result
//! records from `blazen_llm::compute::*`:
//!
//! - [`ImageResult`](blazen_llm::compute::ImageResult)
//! - [`VideoResult`](blazen_llm::compute::VideoResult)
//! - [`AudioResult`](blazen_llm::compute::AudioResult)
//! - [`ThreeDResult`](blazen_llm::compute::ThreeDResult)
//! - [`TranscriptionSegment`](blazen_llm::compute::TranscriptionSegment)
//! - [`TranscriptionResult`](blazen_llm::compute::TranscriptionResult)
//! - [`VoiceHandle`](blazen_llm::compute::VoiceHandle)
//!
//! Results are not constructed by the caller — they fall out of the
//! `BaseProvider` compute methods (wired in Phase A and beyond). The cabi
//! surface exposes getters + `_free` only.
//!
//! # Ownership
//!
//! Every `*_free` consumes the heap allocation. String getters allocate fresh
//! caller-owned C strings (release with [`crate::string::blazen_string_free`]).
//! Nested handle getters (`_segments_get`, `_images_get`, ...) return
//! caller-owned pointers cloned from the source. Optional fields fall back to
//! `null` for missing C strings, `0` for missing numerics, and dedicated
//! `*_has_*` flags where the distinction between absent and zero matters.
//!
//! # JSON-encoded fields
//!
//! The `metadata` (`serde_json::Value`) and embedded `media: MediaOutput`
//! records are exposed as JSON-encoded caller-owned C strings rather than as
//! deeply nested opaque handles. The C surface stays flat and Ruby/other FFI
//! consumers parse the JSON natively — far simpler than building a parallel
//! opaque-handle tree.

#![allow(dead_code)]

use std::ffi::{CStr, c_char};

use blazen_llm::compute::{
    AudioResult as InnerAudioResult, ImageResult as InnerImageResult,
    ThreeDResult as InnerThreeDResult, TranscriptionResult as InnerTranscriptionResult,
    TranscriptionSegment as InnerTranscriptionSegment, VideoResult as InnerVideoResult,
    VoiceHandle as InnerVoiceHandle,
};
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::string::alloc_cstring;

// ---------------------------------------------------------------------------
// Shared JSON-constructor helpers (Wave 3a)
//
// The Ruby binding (via `CustomProvider` trampolines) needs to materialise a
// typed result handle from a JSON-encoded Rust struct. Each `_from_json`
// constructor below shares the same plumbing: read a NUL-terminated UTF-8 C
// string, deserialize it as the inner `blazen_llm::compute::*` type, box it
// into the opaque wrapper. Errors funnel through `write_internal_err` which
// writes a fresh `BlazenError::Internal { message }` into the out-param.
// ---------------------------------------------------------------------------

/// Writes a fresh `BlazenError::Internal { message }` through `out_err` if
/// `out_err` is non-null. Mirrors the `write_error` / `write_internal_error`
/// helpers used in `llm.rs` / `compute.rs`; duplicated here to keep this
/// module self-contained (matches the cabi convention of per-module helpers).
fn write_internal_err(out_err: *mut *mut BlazenError, message: String) {
    if out_err.is_null() {
        return;
    }
    // SAFETY: `out_err` is non-null per the branch above; the caller has
    // guaranteed it points to a writable `*mut BlazenError` slot.
    unsafe {
        *out_err = BlazenError::from(InnerError::Internal { message }).into_ptr();
    }
}

/// Reads `json` (a NUL-terminated UTF-8 C string) as `&str`, writing an
/// `Internal` error through `out_err` on null or non-UTF-8 input. Returns
/// `None` in those cases so the caller can early-return with a null result
/// pointer.
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated buffer that remains valid
/// for the duration of this call.
unsafe fn read_json_input<'a>(
    json: *const c_char,
    fn_name: &str,
    out_err: *mut *mut BlazenError,
) -> Option<&'a str> {
    if json.is_null() {
        write_internal_err(out_err, format!("{fn_name}: json pointer is null"));
        return None;
    }
    // SAFETY: per the contract, `json` points to a NUL-terminated buffer.
    let cstr = unsafe { CStr::from_ptr(json) };
    match cstr.to_str() {
        Ok(s) => Some(s),
        Err(e) => {
            write_internal_err(out_err, format!("{fn_name}: input is not valid UTF-8: {e}"));
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Returns the JSON-encoded text of a `serde_json::Value` as a caller-owned
/// C string. Returns null only if the value contains data that can't be
/// stringified by `serde_json::to_string` (in practice unreachable for the
/// values we round-trip out of `blazen-llm`).
fn alloc_json_cstring(value: &serde_json::Value) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

// ===========================================================================
// TranscriptionSegment (also used by TranscriptionResult)
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::TranscriptionSegment`].
#[repr(C)]
pub struct BlazenTranscriptionSegment(pub(crate) InnerTranscriptionSegment);

impl BlazenTranscriptionSegment {
    pub(crate) fn into_ptr(self) -> *mut BlazenTranscriptionSegment {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTranscriptionSegment> for BlazenTranscriptionSegment {
    fn from(inner: InnerTranscriptionSegment) -> Self {
        Self(inner)
    }
}

/// Returns the segment's `text` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionSegment`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_segment_text(
    handle: *const BlazenTranscriptionSegment,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let s = unsafe { &*handle };
    alloc_cstring(&s.0.text)
}

/// Returns the segment's `start` time in seconds. Returns `0.0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionSegment`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_segment_start(
    handle: *const BlazenTranscriptionSegment,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let s = unsafe { &*handle };
    s.0.start
}

/// Returns the segment's `end` time in seconds. Returns `0.0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionSegment`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_segment_end(
    handle: *const BlazenTranscriptionSegment,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let s = unsafe { &*handle };
    s.0.end
}

/// Returns the segment's optional `speaker` label as a caller-owned C string,
/// or null if no speaker was reported (e.g. diarization disabled).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionSegment`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_segment_speaker(
    handle: *const BlazenTranscriptionSegment,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let s = unsafe { &*handle };
    match &s.0.speaker {
        Some(v) => alloc_cstring(v),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenTranscriptionSegment`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_segment_free(
    handle: *mut BlazenTranscriptionSegment,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// ImageResult
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::ImageResult`].
#[repr(C)]
pub struct BlazenImageResult(pub(crate) InnerImageResult);

impl BlazenImageResult {
    pub(crate) fn into_ptr(self) -> *mut BlazenImageResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerImageResult> for BlazenImageResult {
    fn from(inner: InnerImageResult) -> Self {
        Self(inner)
    }
}

/// Returns the number of images in the result. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_images_count(
    handle: *const BlazenImageResult,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.images.len()
}

/// Returns the `idx`-th image as a JSON-encoded caller-owned C string
/// (serialized [`GeneratedImage`](blazen_llm::media::GeneratedImage) record),
/// or null on null/out-of-range. The JSON carries the embedded
/// `MediaOutput` (`url` / `base64` / `raw_content` / `media_type`) plus the
/// optional `width` and `height` dimensions.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_images_get_json(
    handle: *const BlazenImageResult,
    idx: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match r.0.images.get(idx) {
        Some(img) => match serde_json::to_string(img) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Returns the `image_count` reported by the provider. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_image_count(handle: *const BlazenImageResult) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.image_count
}

/// Returns the provider-reported `cost` in USD. Returns `0.0` on null or when
/// the provider did not report a cost — check
/// [`blazen_image_result_has_cost`] first to distinguish.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_cost(handle: *const BlazenImageResult) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.unwrap_or(0.0)
}

/// Returns `true` if the provider reported a cost. Returns `false` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_has_cost(handle: *const BlazenImageResult) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.is_some()
}

/// Returns the `metadata` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_metadata_json(
    handle: *const BlazenImageResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.metadata)
}

/// Returns the request timing as a JSON-encoded caller-owned C string with
/// keys `queue_ms`, `execution_ms`, `total_ms` (each optional).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_timing_json(
    handle: *const BlazenImageResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match serde_json::to_string(&r.0.timing) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the token-usage record as a JSON-encoded caller-owned C string, or
/// null if the provider did not report usage.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_usage_json(
    handle: *const BlazenImageResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.usage {
        Some(u) => match serde_json::to_string(u) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenImageResult`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_free(handle: *mut BlazenImageResult) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// VideoResult
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::VideoResult`].
#[repr(C)]
pub struct BlazenVideoResult(pub(crate) InnerVideoResult);

impl BlazenVideoResult {
    pub(crate) fn into_ptr(self) -> *mut BlazenVideoResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVideoResult> for BlazenVideoResult {
    fn from(inner: InnerVideoResult) -> Self {
        Self(inner)
    }
}

/// Returns the number of videos in the result. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_videos_count(
    handle: *const BlazenVideoResult,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.videos.len()
}

/// Returns the `idx`-th video as a JSON-encoded caller-owned C string
/// (serialized [`GeneratedVideo`](blazen_llm::media::GeneratedVideo) record).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_videos_get_json(
    handle: *const BlazenVideoResult,
    idx: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match r.0.videos.get(idx) {
        Some(v) => match serde_json::to_string(v) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Returns the total `video_seconds` across all returned videos.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_video_seconds(
    handle: *const BlazenVideoResult,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.video_seconds
}

/// Returns the provider-reported `cost` in USD. Returns `0.0` on null or
/// when the provider did not report a cost — check
/// [`blazen_video_result_has_cost`] first.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_cost(handle: *const BlazenVideoResult) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.unwrap_or(0.0)
}

/// Returns `true` if the provider reported a cost.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_has_cost(handle: *const BlazenVideoResult) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.is_some()
}

/// Returns the `metadata` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_metadata_json(
    handle: *const BlazenVideoResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.metadata)
}

/// Returns the request timing as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_timing_json(
    handle: *const BlazenVideoResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match serde_json::to_string(&r.0.timing) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the token-usage record as a JSON-encoded caller-owned C string,
/// or null if not reported.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_usage_json(
    handle: *const BlazenVideoResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.usage {
        Some(u) => match serde_json::to_string(u) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenVideoResult`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_free(handle: *mut BlazenVideoResult) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// AudioResult
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::AudioResult`].
#[repr(C)]
pub struct BlazenAudioResult(pub(crate) InnerAudioResult);

impl BlazenAudioResult {
    pub(crate) fn into_ptr(self) -> *mut BlazenAudioResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerAudioResult> for BlazenAudioResult {
    fn from(inner: InnerAudioResult) -> Self {
        Self(inner)
    }
}

/// Returns the number of audio clips in the result. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_audio_count(
    handle: *const BlazenAudioResult,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.audio.len()
}

/// Returns the `idx`-th audio clip as a JSON-encoded caller-owned C string
/// (serialized [`GeneratedAudio`](blazen_llm::media::GeneratedAudio) record).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_audio_get_json(
    handle: *const BlazenAudioResult,
    idx: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match r.0.audio.get(idx) {
        Some(a) => match serde_json::to_string(a) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Returns the total `audio_seconds` across all returned clips.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_audio_seconds(
    handle: *const BlazenAudioResult,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.audio_seconds
}

/// Returns the provider-reported `cost` in USD. Check
/// [`blazen_audio_result_has_cost`] to distinguish missing from zero.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_cost(handle: *const BlazenAudioResult) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.unwrap_or(0.0)
}

/// Returns `true` if the provider reported a cost.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_has_cost(handle: *const BlazenAudioResult) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.is_some()
}

/// Returns the `metadata` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_metadata_json(
    handle: *const BlazenAudioResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.metadata)
}

/// Returns the request timing as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_timing_json(
    handle: *const BlazenAudioResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match serde_json::to_string(&r.0.timing) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the token-usage record as a JSON-encoded caller-owned C string,
/// or null if not reported.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_usage_json(
    handle: *const BlazenAudioResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.usage {
        Some(u) => match serde_json::to_string(u) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenAudioResult`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_free(handle: *mut BlazenAudioResult) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// ThreeDResult
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::ThreeDResult`].
#[repr(C)]
pub struct BlazenThreeDResult(pub(crate) InnerThreeDResult);

impl BlazenThreeDResult {
    pub(crate) fn into_ptr(self) -> *mut BlazenThreeDResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerThreeDResult> for BlazenThreeDResult {
    fn from(inner: InnerThreeDResult) -> Self {
        Self(inner)
    }
}

/// Returns the number of generated 3D models. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_models_count(
    handle: *const BlazenThreeDResult,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.models.len()
}

/// Returns the `idx`-th 3D model as a JSON-encoded caller-owned C string
/// (serialized [`Generated3DModel`](blazen_llm::media::Generated3DModel)).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_models_get_json(
    handle: *const BlazenThreeDResult,
    idx: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match r.0.models.get(idx) {
        Some(m) => match serde_json::to_string(m) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Returns the provider-reported `cost` in USD. Check
/// [`blazen_three_d_result_has_cost`] to distinguish missing from zero.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_cost(handle: *const BlazenThreeDResult) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.unwrap_or(0.0)
}

/// Returns `true` if the provider reported a cost.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_has_cost(handle: *const BlazenThreeDResult) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.is_some()
}

/// Returns the `metadata` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_metadata_json(
    handle: *const BlazenThreeDResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.metadata)
}

/// Returns the request timing as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_timing_json(
    handle: *const BlazenThreeDResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match serde_json::to_string(&r.0.timing) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the token-usage record as a JSON-encoded caller-owned C string,
/// or null if not reported.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_usage_json(
    handle: *const BlazenThreeDResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.usage {
        Some(u) => match serde_json::to_string(u) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenThreeDResult`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_free(handle: *mut BlazenThreeDResult) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// TranscriptionResult
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::TranscriptionResult`].
#[repr(C)]
pub struct BlazenTranscriptionResult(pub(crate) InnerTranscriptionResult);

impl BlazenTranscriptionResult {
    pub(crate) fn into_ptr(self) -> *mut BlazenTranscriptionResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTranscriptionResult> for BlazenTranscriptionResult {
    fn from(inner: InnerTranscriptionResult) -> Self {
        Self(inner)
    }
}

/// Returns the full transcript text as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_text(
    handle: *const BlazenTranscriptionResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.text)
}

/// Returns the number of time-aligned segments. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_segments_count(
    handle: *const BlazenTranscriptionResult,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.segments.len()
}

/// Clones the `idx`-th segment into a fresh caller-owned
/// `BlazenTranscriptionSegment`. Caller frees with
/// `blazen_transcription_segment_free`. Returns null on null/out-of-range.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_segments_get(
    handle: *const BlazenTranscriptionResult,
    idx: usize,
) -> *mut BlazenTranscriptionSegment {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match r.0.segments.get(idx) {
        Some(seg) => BlazenTranscriptionSegment(seg.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Returns the detected/specified `language` as a caller-owned C string, or
/// null if not reported.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_language(
    handle: *const BlazenTranscriptionResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.language {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Returns the duration of input audio that was transcribed, in seconds.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_audio_seconds(
    handle: *const BlazenTranscriptionResult,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.audio_seconds
}

/// Returns the provider-reported `cost` in USD. Check
/// [`blazen_transcription_result_has_cost`] to distinguish missing from zero.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_cost(
    handle: *const BlazenTranscriptionResult,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.unwrap_or(0.0)
}

/// Returns `true` if the provider reported a cost.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_has_cost(
    handle: *const BlazenTranscriptionResult,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.cost.is_some()
}

/// Returns the `metadata` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_metadata_json(
    handle: *const BlazenTranscriptionResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.metadata)
}

/// Returns the request timing as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_timing_json(
    handle: *const BlazenTranscriptionResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match serde_json::to_string(&r.0.timing) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the token-usage record as a JSON-encoded caller-owned C string,
/// or null if not reported.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_usage_json(
    handle: *const BlazenTranscriptionResult,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.usage {
        Some(u) => match serde_json::to_string(u) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenTranscriptionResult`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_free(handle: *mut BlazenTranscriptionResult) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// VoiceHandle
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::VoiceHandle`].
#[repr(C)]
pub struct BlazenVoiceHandle(pub(crate) InnerVoiceHandle);

impl BlazenVoiceHandle {
    pub(crate) fn into_ptr(self) -> *mut BlazenVoiceHandle {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVoiceHandle> for BlazenVoiceHandle {
    fn from(inner: InnerVoiceHandle) -> Self {
        Self(inner)
    }
}

/// Returns the provider-specific voice `id` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_id(handle: *const BlazenVoiceHandle) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let v = unsafe { &*handle };
    alloc_cstring(&v.0.id)
}

/// Returns the human-readable voice `name` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_name(handle: *const BlazenVoiceHandle) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let v = unsafe { &*handle };
    alloc_cstring(&v.0.name)
}

/// Returns the owning provider name as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_provider(
    handle: *const BlazenVoiceHandle,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let v = unsafe { &*handle };
    alloc_cstring(&v.0.provider)
}

/// Returns the optional language code as a caller-owned C string, or null if
/// unset.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_language(
    handle: *const BlazenVoiceHandle,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let v = unsafe { &*handle };
    match &v.0.language {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Returns the optional description as a caller-owned C string, or null if
/// unset.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_description(
    handle: *const BlazenVoiceHandle,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let v = unsafe { &*handle };
    match &v.0.description {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Returns the `metadata` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_metadata_json(
    handle: *const BlazenVoiceHandle,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let v = unsafe { &*handle };
    alloc_json_cstring(&v.0.metadata)
}

/// Frees a `BlazenVoiceHandle`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_free(handle: *mut BlazenVoiceHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// JSON-shim constructors (Wave 3a)
//
// Each `_from_json` parses a NUL-terminated UTF-8 JSON string as the inner
// `blazen_llm::compute::*` record (which derives `serde::Deserialize`) and
// returns a freshly-boxed opaque handle. On any failure — null pointer,
// non-UTF-8, malformed JSON, or deserialization error — writes a fresh
// `BlazenError::Internal { message }` through `out_err` (if non-null) and
// returns null. The returned handle is owned by the caller; release it with
// the matching `*_free`.
// ===========================================================================

/// Constructs a [`BlazenAudioResult`] handle from a JSON-encoded
/// [`blazen_llm::compute::AudioResult`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_audio_result_free`]. On failure returns null and writes a fresh
/// `BlazenError::Internal { message }` into `*out_err` when `out_err` is
/// non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_result_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenAudioResult {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) = (unsafe { read_json_input(json, "blazen_audio_result_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<InnerAudioResult>(s) {
        Ok(inner) => BlazenAudioResult(inner).into_ptr(),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_audio_result_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

/// Constructs a [`BlazenImageResult`] handle from a JSON-encoded
/// [`blazen_llm::compute::ImageResult`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_image_result_free`]. On failure returns null and writes a fresh
/// `BlazenError::Internal { message }` into `*out_err` when `out_err` is
/// non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_result_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenImageResult {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) = (unsafe { read_json_input(json, "blazen_image_result_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<InnerImageResult>(s) {
        Ok(inner) => BlazenImageResult(inner).into_ptr(),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_image_result_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

/// Constructs a [`BlazenVideoResult`] handle from a JSON-encoded
/// [`blazen_llm::compute::VideoResult`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_video_result_free`]. On failure returns null and writes a fresh
/// `BlazenError::Internal { message }` into `*out_err` when `out_err` is
/// non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_result_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenVideoResult {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) = (unsafe { read_json_input(json, "blazen_video_result_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<InnerVideoResult>(s) {
        Ok(inner) => BlazenVideoResult(inner).into_ptr(),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_video_result_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

/// Constructs a [`BlazenThreeDResult`] handle from a JSON-encoded
/// [`blazen_llm::compute::ThreeDResult`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_three_d_result_free`]. On failure returns null and writes a fresh
/// `BlazenError::Internal { message }` into `*out_err` when `out_err` is
/// non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_result_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenThreeDResult {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) = (unsafe { read_json_input(json, "blazen_three_d_result_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<InnerThreeDResult>(s) {
        Ok(inner) => BlazenThreeDResult(inner).into_ptr(),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_three_d_result_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

/// Constructs a [`BlazenTranscriptionResult`] handle from a JSON-encoded
/// [`blazen_llm::compute::TranscriptionResult`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_transcription_result_free`]. On failure returns null and writes a
/// fresh `BlazenError::Internal { message }` into `*out_err` when `out_err`
/// is non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_result_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenTranscriptionResult {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) =
        (unsafe { read_json_input(json, "blazen_transcription_result_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<InnerTranscriptionResult>(s) {
        Ok(inner) => BlazenTranscriptionResult(inner).into_ptr(),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_transcription_result_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

/// Constructs a [`BlazenVoiceHandle`] handle from a JSON-encoded
/// [`blazen_llm::compute::VoiceHandle`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_voice_handle_free`]. On failure returns null and writes a fresh
/// `BlazenError::Internal { message }` into `*out_err` when `out_err` is
/// non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenVoiceHandle {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) = (unsafe { read_json_input(json, "blazen_voice_handle_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<InnerVoiceHandle>(s) {
        Ok(inner) => BlazenVoiceHandle(inner).into_ptr(),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_voice_handle_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

// ===========================================================================
// BlazenVoiceHandleArray (Wave 3a)
//
// `list_voices` in the `CustomProvider` vtable yields a `Vec<VoiceHandle>`.
// To bridge a Ruby `list_voices` override (which may return any iterable of
// `VoiceHandle` JSON blobs) into the vtable's `out_array: *mut *mut *mut
// BlazenVoiceHandle` slot, this opaque array type lets the Ruby trampoline
// parse a JSON array once and then pop entries one-by-one with `_take`.
//
// The vtable's signature is unchanged (Wave 3a is constrained to
// `_from_json` constructors); the Ruby side uses these helpers to construct
// the contiguous pointer block the vtable expects.
// ===========================================================================

/// Opaque wrapper around `Vec<blazen_llm::compute::VoiceHandle>`. Produced by
/// [`blazen_voice_handle_array_from_json`]; entries are removed one-at-a-time
/// with [`blazen_voice_handle_array_take`]. Released with
/// [`blazen_voice_handle_array_free`] (which drops any remaining entries).
#[repr(C)]
pub struct BlazenVoiceHandleArray {
    pub(crate) inner: Vec<InnerVoiceHandle>,
}

/// Parses a JSON array of [`blazen_llm::compute::VoiceHandle`] records into a
/// freshly-boxed [`BlazenVoiceHandleArray`].
///
/// # Ownership
///
/// On success returns a non-null handle owned by the caller — release with
/// [`blazen_voice_handle_array_free`]. On failure returns null and writes a
/// fresh `BlazenError::Internal { message }` into `*out_err` when `out_err`
/// is non-null (caller frees with [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call. `out_err` must be null OR point to a writable
/// `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_array_from_json(
    json: *const c_char,
    out_err: *mut *mut BlazenError,
) -> *mut BlazenVoiceHandleArray {
    // SAFETY: forwarded to `read_json_input`; caller upholds the contract.
    let Some(s) =
        (unsafe { read_json_input(json, "blazen_voice_handle_array_from_json", out_err) })
    else {
        return std::ptr::null_mut();
    };
    match serde_json::from_str::<Vec<InnerVoiceHandle>>(s) {
        Ok(inner) => Box::into_raw(Box::new(BlazenVoiceHandleArray { inner })),
        Err(e) => {
            write_internal_err(
                out_err,
                format!("blazen_voice_handle_array_from_json: deserialize failed: {e}"),
            );
            std::ptr::null_mut()
        }
    }
}

/// Returns the current length of the array. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandleArray`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_array_len(
    handle: *const BlazenVoiceHandleArray,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenVoiceHandleArray`.
    let a = unsafe { &*handle };
    a.inner.len()
}

/// Pops the `idx`-th entry from the array and returns it as a freshly-boxed
/// [`BlazenVoiceHandle`] handle owned by the caller (release with
/// [`blazen_voice_handle_free`]). Returns null if `handle` is null or `idx`
/// is out of range.
///
/// Note: `idx` is interpreted against the array's current length, which
/// shrinks by one after every successful call. Callers should typically iterate
/// from index `0` until [`blazen_voice_handle_array_len`] returns `0`. Calling
/// `_take(0)` repeatedly is the canonical drain pattern; `Vec::remove`
/// semantics apply, so out-of-bounds indices are rejected with null rather
/// than aborting.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceHandleArray` (and not freed
/// concurrently from another thread).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_array_take(
    handle: *mut BlazenVoiceHandleArray,
    idx: usize,
) -> *mut BlazenVoiceHandle {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenVoiceHandleArray`.
    let a = unsafe { &mut *handle };
    if idx >= a.inner.len() {
        return std::ptr::null_mut();
    }
    let inner = a.inner.remove(idx);
    BlazenVoiceHandle(inner).into_ptr()
}

/// Frees a `BlazenVoiceHandleArray` handle, dropping any remaining entries.
/// No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by
/// [`blazen_voice_handle_array_from_json`]. Calling this twice on the same
/// non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_handle_array_free(handle: *mut BlazenVoiceHandleArray) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
