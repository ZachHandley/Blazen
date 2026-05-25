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
use blazen_uniffi::compute_music::{
    MusicChunk as InnerMusicChunk, MusicResult as InnerMusicResult,
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

// ---------------------------------------------------------------------------
// MusicChunk
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerMusicChunk`].
///
/// Produced by the music-stream sink trampoline (one allocation per
/// streamed chunk) and handed to the foreign-side `on_chunk` callback. The
/// foreign callback OWNS the chunk and MUST release it via
/// [`blazen_music_chunk_free`] before returning.
///
/// Wire-format: `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the
/// backend's sample rate; `is_final` is a UI hint (the canonical
/// end-of-stream marker is the sink's `on_done` callback);
/// `latency_seconds` is the per-chunk measured latency from call-start
/// (None when unreported).
pub struct BlazenMusicChunk(pub(crate) InnerMusicChunk);

impl BlazenMusicChunk {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenMusicChunk {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerMusicChunk> for BlazenMusicChunk {
    fn from(inner: InnerMusicChunk) -> Self {
        Self(inner)
    }
}

/// Borrows the chunk's PCM sample slice. Writes the slice length (in
/// `f32` elements) into `*out_len` and returns the pointer to the first
/// sample. The returned pointer is valid for the lifetime of the chunk
/// handle (i.e. until [`blazen_music_chunk_free`] is called); callers
/// must NOT free the sample buffer directly.
///
/// Returns null and writes `0` into `*out_len` if `handle` is null.
///
/// # Safety
///
/// `handle` must be null OR a valid pointer to a `BlazenMusicChunk`
/// produced by the cabi surface. `out_len` must be null OR a writable
/// pointer to a single `usize` slot. The returned pointer aliases the
/// chunk's internal `Vec<f32>` — keep the chunk alive for as long as the
/// pointer is in use, and do not mutate the buffer through it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_chunk_samples(
    handle: *const BlazenMusicChunk,
    out_len: *mut usize,
) -> *const f32 {
    if handle.is_null() {
        if !out_len.is_null() {
            // SAFETY: caller upholds the out-pointer contract.
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let handle = unsafe { &*handle };
    let slice = handle.0.samples.as_slice();
    if !out_len.is_null() {
        // SAFETY: caller upholds the out-pointer contract.
        unsafe {
            *out_len = slice.len();
        }
    }
    slice.as_ptr()
}

/// Returns `true` if this is the final emitted chunk of the streaming
/// generation, `false` otherwise. Returns `false` if `handle` is null.
///
/// # Safety
///
/// `handle` must be null OR a valid pointer to a `BlazenMusicChunk`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_chunk_is_final(handle: *const BlazenMusicChunk) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let handle = unsafe { &*handle };
    handle.0.is_final
}

/// Returns the per-chunk latency in seconds, or NaN when the backend did
/// not report a measurement (or when `handle` is null). The NaN sentinel
/// mirrors the encoding used elsewhere in the cabi surface for
/// `Option<f32>` returns.
///
/// # Safety
///
/// `handle` must be null OR a valid pointer to a `BlazenMusicChunk`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_chunk_latency_seconds(
    handle: *const BlazenMusicChunk,
) -> f32 {
    if handle.is_null() {
        return f32::NAN;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let handle = unsafe { &*handle };
    handle.0.latency_seconds.unwrap_or(f32::NAN)
}

/// Frees a `BlazenMusicChunk` previously produced by the cabi surface
/// (typically inside a music-stream sink's `on_chunk` callback). Passing
/// null is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface's music-stream trampoline. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_chunk_free(handle: *mut BlazenMusicChunk) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// MusicResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerMusicResult`].
///
/// Produced by the cabi `MusicModel::generate_music` / `generate_sfx`
/// wrappers and by the typed `blazen_future_take_music_result` taker.
/// Holds the encoded audio payload (empty when the provider only returned
/// a URL), MIME type, sample rate, channel count, duration, and an
/// optional URL.
pub struct BlazenMusicResult(pub(crate) InnerMusicResult);

impl BlazenMusicResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenMusicResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerMusicResult> for BlazenMusicResult {
    fn from(inner: InnerMusicResult) -> Self {
        Self(inner)
    }
}

/// Borrows the result's encoded audio bytes. Writes the slice length
/// into `*out_len` and returns the pointer to the first byte. The
/// returned pointer is valid for the lifetime of the result handle
/// (i.e. until [`blazen_music_result_free`] is called); callers must NOT
/// free the buffer directly.
///
/// Returns null and writes `0` into `*out_len` if `result` is null.
/// The returned slice is empty when the upstream provider only returned
/// a URL (check [`blazen_music_result_url`] in that case).
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenMusicResult`
/// produced by the cabi surface. `out_len` must be null OR a writable
/// pointer to a single `usize` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_bytes(
    result: *const BlazenMusicResult,
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
    let slice = result.0.bytes.as_slice();
    if !out_len.is_null() {
        // SAFETY: caller upholds the out-pointer contract.
        unsafe {
            *out_len = slice.len();
        }
    }
    slice.as_ptr()
}

/// Returns the IANA MIME type of the encoded audio as a heap-allocated
/// C string. Caller frees with `blazen_string_free`. Returns null if
/// `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenMusicResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_mime_type(
    result: *const BlazenMusicResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.mime_type)
}

/// Returns the sample rate in Hz. Returns `0` if `result` is null or
/// the upstream provider did not report a sample rate.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenMusicResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_sample_rate(result: *const BlazenMusicResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.sample_rate
}

/// Returns the channel count (1 = mono, 2 = stereo). Returns `0` if
/// `result` is null or the upstream provider did not report a channel
/// count.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenMusicResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_channels(result: *const BlazenMusicResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.channels
}

/// Returns the duration of the clip in seconds. Returns `0.0` if `result`
/// is null or the upstream provider did not report a duration.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenMusicResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_duration_seconds(
    result: *const BlazenMusicResult,
) -> f32 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.duration_seconds
}

/// Returns the URL of the audio asset as a heap-allocated C string when
/// the upstream provider returned a link rather than inline bytes;
/// returns an empty string for inline-bytes results. Caller frees with
/// `blazen_string_free`. Returns null if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenMusicResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_url(result: *const BlazenMusicResult) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.url)
}

/// Frees a `BlazenMusicResult` produced by the cabi surface. Passing
/// null is a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// music-generation wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_result_free(result: *mut BlazenMusicResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}
