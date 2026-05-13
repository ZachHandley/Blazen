//! Compute-related record marshalling. Opaque handles wrap
//! `blazen_uniffi::compute::*` records (`TtsResult`, `SttResult`, `ImageGenResult`)
//! produced by `TtsModel::synthesize`, `SttModel::transcribe`, and
//! `ImageGenModel::generate` respectively (wired in Phase R3).
//!
//! Ownership: each `*_free` consumes the heap allocation. String getters
//! return caller-owned C strings — free with `blazen_string_free`. Nested
//! handle getters (e.g. `_images_get`) return caller-owned pointers cloned
//! out of the source — free via the matching nested `_free` (e.g.
//! `blazen_media_free`).

#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::compute::{
    ImageGenResult as InnerImageGenResult, SttResult as InnerSttResult, TtsResult as InnerTtsResult,
};

use crate::llm_records::BlazenMedia;
use crate::string::alloc_cstring;

// ---------------------------------------------------------------------------
// TtsResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerTtsResult`].
///
/// Produced by the cabi `TtsModel::synthesize` wrapper (Phase R3). Holds a
/// base64-encoded audio payload (empty when the provider returned a URL
/// only), a MIME type string, and a duration in milliseconds.
pub struct BlazenTtsResult(pub(crate) InnerTtsResult);

impl BlazenTtsResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenTtsResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTtsResult> for BlazenTtsResult {
    fn from(inner: InnerTtsResult) -> Self {
        Self(inner)
    }
}

/// Returns the synthesized audio as a heap-allocated C string of
/// base64-encoded bytes. Caller frees with `blazen_string_free`. Returns
/// null if `result` is null. The returned string is empty when the upstream
/// provider only returned a URL (not raw bytes).
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenTtsResult` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_result_audio_base64(
    result: *const BlazenTtsResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.audio_base64)
}

/// Returns the MIME type of the synthesized audio as a heap-allocated C
/// string. Caller frees with `blazen_string_free`. Returns null if `result`
/// is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenTtsResult` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_result_mime_type(
    result: *const BlazenTtsResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.mime_type)
}

/// Returns the duration of the synthesized audio in milliseconds. Returns
/// `0` if `result` is null or if the provider did not report timing.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenTtsResult` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_result_duration_ms(result: *const BlazenTtsResult) -> u64 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.duration_ms
}

/// Frees a `BlazenTtsResult` produced by the cabi surface. Passing null is
/// a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// TTS-synthesis wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tts_result_free(result: *mut BlazenTtsResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}

// ---------------------------------------------------------------------------
// SttResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerSttResult`].
///
/// Produced by the cabi `SttModel::transcribe` wrapper (Phase R3). Holds
/// the transcript text, an optional detected language (empty string when
/// the provider didn't report one), and a duration in milliseconds.
pub struct BlazenSttResult(pub(crate) InnerSttResult);

impl BlazenSttResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenSttResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerSttResult> for BlazenSttResult {
    fn from(inner: InnerSttResult) -> Self {
        Self(inner)
    }
}

/// Returns the transcript text as a heap-allocated C string. Caller frees
/// with `blazen_string_free`. Returns null if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenSttResult` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_result_transcript(
    result: *const BlazenSttResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.transcript)
}

/// Returns the detected language as a heap-allocated C string (ISO-639-1).
/// Caller frees with `blazen_string_free`. The returned string is empty
/// when the provider did not report a language. Returns null if `result`
/// is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenSttResult` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_result_language(result: *const BlazenSttResult) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.language)
}

/// Returns the duration of the source audio in milliseconds. Returns `0`
/// if `result` is null or the provider did not report timing.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenSttResult` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_result_duration_ms(result: *const BlazenSttResult) -> u64 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.duration_ms
}

/// Frees a `BlazenSttResult` produced by the cabi surface. Passing null is
/// a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// STT-transcribe wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stt_result_free(result: *mut BlazenSttResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}

// ---------------------------------------------------------------------------
// ImageGenResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerImageGenResult`].
///
/// Produced by the cabi `ImageGenModel::generate` wrapper (Phase R3). Holds
/// the rendered image list; each entry's `data_base64` field carries either
/// raw base64 bytes or a URL string per its `mime_type` (see the upstream
/// [`ImageGenResult`](blazen_uniffi::compute::ImageGenResult) docs).
pub struct BlazenImageGenResult(pub(crate) InnerImageGenResult);

impl BlazenImageGenResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenImageGenResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerImageGenResult> for BlazenImageGenResult {
    fn from(inner: InnerImageGenResult) -> Self {
        Self(inner)
    }
}

/// Returns the number of images in the result. Returns `0` if `result` is
/// null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenImageGenResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_result_images_count(
    result: *const BlazenImageGenResult,
) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.images.len()
}

/// Clones the image at index `idx` out of the result and returns a
/// caller-owned `BlazenMedia` handle. Returns null if `result` is null or
/// `idx` is out of bounds. Caller frees the returned handle with
/// `blazen_media_free`.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenImageGenResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_result_images_get(
    result: *const BlazenImageGenResult,
    idx: usize,
) -> *mut BlazenMedia {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    match result.0.images.get(idx) {
        Some(media) => BlazenMedia::from(media.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenImageGenResult` produced by the cabi surface. Passing
/// null is a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// image-generation wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_gen_result_free(result: *mut BlazenImageGenResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}
