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

use std::ffi::c_char;

use blazen_llm::compute::{
    AudioResult as InnerAudioResult, ImageResult as InnerImageResult,
    ThreeDResult as InnerThreeDResult, TranscriptionResult as InnerTranscriptionResult,
    TranscriptionSegment as InnerTranscriptionSegment, VideoResult as InnerVideoResult,
    VoiceHandle as InnerVoiceHandle,
};

use crate::string::alloc_cstring;

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
