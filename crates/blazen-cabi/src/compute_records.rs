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

#[cfg(feature = "triposr")]
use blazen_uniffi::compute::ThreeDGenerateResult as InnerThreeDGenerateResult;
use blazen_uniffi::compute::{
    ImageGenResult as InnerImageGenResult, SttResult as InnerSttResult, TtsResult as InnerTtsResult,
};
use blazen_uniffi::compute_music::{
    MusicChunk as InnerMusicChunk, MusicResult as InnerMusicResult,
};
use blazen_uniffi::compute_vc::{
    TargetVoice as InnerTargetVoice, VcChunk as InnerVcChunk, VcResult as InnerVcResult,
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

// ---------------------------------------------------------------------------
// VcChunk
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerVcChunk`].
///
/// Produced by the voice-conversion stream sink trampoline (one allocation
/// per streamed chunk) and handed to the foreign-side `on_chunk` callback.
/// The foreign callback OWNS the chunk and MUST release it via
/// [`blazen_vc_chunk_free`] before returning.
///
/// Wire-format: `samples` is 32-bit float PCM in `[-1.0, 1.0]` at the
/// target voice's native sample rate; `is_final` is a UI hint (the
/// canonical end-of-stream marker is the sink's `on_done` callback);
/// `latency_seconds` is the per-chunk measured latency from call-start
/// (None when unreported).
pub struct BlazenVcChunk(pub(crate) InnerVcChunk);

impl BlazenVcChunk {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenVcChunk {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVcChunk> for BlazenVcChunk {
    fn from(inner: InnerVcChunk) -> Self {
        Self(inner)
    }
}

/// Borrows the chunk's PCM sample slice. Writes the slice length (in
/// `f32` elements) into `*out_len` and returns the pointer to the first
/// sample. The returned pointer is valid for the lifetime of the chunk
/// handle (i.e. until [`blazen_vc_chunk_free`] is called); callers must
/// NOT free the sample buffer directly.
///
/// Returns null and writes `0` into `*out_len` if `handle` is null.
///
/// # Safety
///
/// `handle` must be null OR a valid pointer to a `BlazenVcChunk` produced
/// by the cabi surface. `out_len` must be null OR a writable pointer to a
/// single `usize` slot. The returned pointer aliases the chunk's internal
/// `Vec<f32>` — keep the chunk alive for as long as the pointer is in
/// use, and do not mutate the buffer through it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_chunk_samples(
    handle: *const BlazenVcChunk,
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
/// voice-conversion call, `false` otherwise. Returns `false` if `handle`
/// is null.
///
/// # Safety
///
/// `handle` must be null OR a valid pointer to a `BlazenVcChunk` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_chunk_is_final(handle: *const BlazenVcChunk) -> bool {
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
/// `handle` must be null OR a valid pointer to a `BlazenVcChunk` produced
/// by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_chunk_latency_seconds(handle: *const BlazenVcChunk) -> f32 {
    if handle.is_null() {
        return f32::NAN;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let handle = unsafe { &*handle };
    handle.0.latency_seconds.unwrap_or(f32::NAN)
}

/// Frees a `BlazenVcChunk` previously produced by the cabi surface
/// (typically inside a voice-conversion stream sink's `on_chunk`
/// callback). Passing null is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface's voice-conversion stream trampoline. Double-free is undefined
/// behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_chunk_free(handle: *mut BlazenVcChunk) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// VcResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping [`InnerVcResult`].
///
/// Produced by the cabi `VcModel::convert_voice` wrappers and by the typed
/// `blazen_future_take_vc_result` taker. Holds a complete WAV
/// (RIFF/`fmt `/`data`) container with 16-bit signed PCM samples at the
/// target voice's native sample rate, a MIME type string, the sample rate
/// in Hz, and a duration in seconds. There is no `channels` field — the
/// RVC backend renders mono and stamps the rate from the resolved target
/// voice — and no `url` field (voice conversion is always rendered to
/// inline bytes today).
pub struct BlazenVcResult(pub(crate) InnerVcResult);

impl BlazenVcResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenVcResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVcResult> for BlazenVcResult {
    fn from(inner: InnerVcResult) -> Self {
        Self(inner)
    }
}

/// Borrows the result's encoded audio bytes. Writes the slice length
/// into `*out_len` and returns the pointer to the first byte. The
/// returned pointer is valid for the lifetime of the result handle
/// (i.e. until [`blazen_vc_result_free`] is called); callers must NOT
/// free the buffer directly.
///
/// Returns null and writes `0` into `*out_len` if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenVcResult` produced
/// by the cabi surface. `out_len` must be null OR a writable pointer to
/// a single `usize` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_result_bytes(
    result: *const BlazenVcResult,
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
/// `result` must be null OR a valid pointer to a `BlazenVcResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_result_mime_type(result: *const BlazenVcResult) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.mime_type)
}

/// Returns the sample rate in Hz. Returns `0` if `result` is null or
/// the upstream backend did not report a sample rate.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenVcResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_result_sample_rate(result: *const BlazenVcResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.sample_rate
}

/// Returns the duration of the clip in seconds. Returns `0.0` if `result`
/// is null or the upstream backend did not report a duration.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenVcResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_result_duration_seconds(result: *const BlazenVcResult) -> f32 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.duration_seconds
}

/// Frees a `BlazenVcResult` produced by the cabi surface. Passing null is
/// a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// voice-conversion wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_vc_result_free(result: *mut BlazenVcResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}

// ---------------------------------------------------------------------------
// TargetVoice
// ---------------------------------------------------------------------------

/// Opaque snapshot of one registered voice the backend can render — wraps
/// [`InnerTargetVoice`].
///
/// The `label` accessor returns an empty string when the upstream backend
/// did not record a display name (matches the [`BlazenSttResult::language`]
/// convention rather than threading `Option<String>` through C).
pub struct BlazenTargetVoice(pub(crate) InnerTargetVoice);

impl BlazenTargetVoice {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenTargetVoice {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTargetVoice> for BlazenTargetVoice {
    fn from(inner: InnerTargetVoice) -> Self {
        Self(inner)
    }
}

/// Returns the backend-scoped voice identifier as a heap-allocated C
/// string. Caller frees with `blazen_string_free`. Returns null if
/// `voice` is null.
///
/// # Safety
///
/// `voice` must be null OR a valid pointer to a `BlazenTargetVoice`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_id(voice: *const BlazenTargetVoice) -> *mut c_char {
    if voice.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let voice = unsafe { &*voice };
    alloc_cstring(&voice.0.id)
}

/// Returns the human-readable display label as a heap-allocated C string.
/// The returned string is empty when the upstream backend did not record
/// a label (mirrors the `BlazenSttResult::language` empty-string
/// convention). Caller frees with `blazen_string_free`. Returns null if
/// `voice` is null.
///
/// # Safety
///
/// `voice` must be null OR a valid pointer to a `BlazenTargetVoice`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_label(voice: *const BlazenTargetVoice) -> *mut c_char {
    if voice.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let voice = unsafe { &*voice };
    alloc_cstring(voice.0.label.as_deref().unwrap_or(""))
}

/// Returns the native sample rate (Hz) the backend renders this voice at.
/// Returns `0` if `voice` is null.
///
/// # Safety
///
/// `voice` must be null OR a valid pointer to a `BlazenTargetVoice`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_sample_rate_hz(
    voice: *const BlazenTargetVoice,
) -> u32 {
    if voice.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let voice = unsafe { &*voice };
    voice.0.sample_rate_hz
}

/// Frees a `BlazenTargetVoice` produced by the cabi surface. Passing null
/// is a no-op.
///
/// # Safety
///
/// `voice` must be null OR a pointer produced by the cabi surface.
/// Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_free(voice: *mut BlazenTargetVoice) {
    if voice.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(voice) });
}

// ---------------------------------------------------------------------------
// BlazenTargetVoiceList
// ---------------------------------------------------------------------------

/// Opaque list of [`BlazenTargetVoice`] snapshots. Mirrors
/// [`crate::manager_records::BlazenModelStatusList`]: borrow-by-index
/// (`_get`) for read-only iteration; move-by-index (`_take`) to peel off a
/// caller-owned handle.
pub struct BlazenTargetVoiceList {
    pub(crate) inner: Vec<BlazenTargetVoice>,
}

impl BlazenTargetVoiceList {
    /// Heap-allocate the list and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenTargetVoiceList {
        Box::into_raw(Box::new(self))
    }
}

impl From<Vec<InnerTargetVoice>> for BlazenTargetVoiceList {
    fn from(items: Vec<InnerTargetVoice>) -> Self {
        Self {
            inner: items.into_iter().map(BlazenTargetVoice::from).collect(),
        }
    }
}

/// Returns the number of entries in the list. Returns `0` on a null
/// handle.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenTargetVoiceList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_list_len(list: *const BlazenTargetVoiceList) -> usize {
    if list.is_null() {
        return 0;
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner.len()
}

/// Borrows the `idx`-th entry. Returns null if `list` is null or `idx` is
/// out of range. The returned pointer is valid for the lifetime of the
/// list — do NOT call `blazen_target_voice_free` on it.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenTargetVoiceList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_list_get(
    list: *const BlazenTargetVoiceList,
    idx: usize,
) -> *const BlazenTargetVoice {
    if list.is_null() {
        return std::ptr::null();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &*list };
    l.inner
        .get(idx)
        .map_or(std::ptr::null(), std::ptr::from_ref)
}

/// Pops the `idx`-th entry and returns it as a caller-owned handle. Free
/// the returned handle with [`blazen_target_voice_free`]. Returns null if
/// `list` is null or `idx` is out of range.
///
/// # Safety
///
/// `list` must be null OR a live [`BlazenTargetVoiceList`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_list_take(
    list: *mut BlazenTargetVoiceList,
    idx: usize,
) -> *mut BlazenTargetVoice {
    if list.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: live-pointer contract per the per-fn docs.
    let l = unsafe { &mut *list };
    if idx >= l.inner.len() {
        return std::ptr::null_mut();
    }
    l.inner.remove(idx).into_ptr()
}

/// Frees a [`BlazenTargetVoiceList`], dropping any remaining entries.
///
/// # Safety
///
/// `list` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_target_voice_list_free(list: *mut BlazenTargetVoiceList) {
    if list.is_null() {
        return;
    }
    // SAFETY: per the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(list) });
}

// ---------------------------------------------------------------------------
// BlazenThreeDGenerateResult (gated on `triposr` — only the TripoSR per-engine
// blocking factory in `crate::three_d` produces these handles, and that whole
// file is `#[cfg(feature = "triposr")]`. Without the gate, default-features
// cabi build fails because the inner uniffi `ThreeDGenerateResult` doesn't
// exist without `triposr`.)
// ---------------------------------------------------------------------------

/// Opaque handle wrapping
/// [`blazen_uniffi::compute::ThreeDGenerateResult`].
///
/// Produced by per-engine cabi 3D-generation wrappers (e.g.
/// `BlazenTripoSrProvider::generate_from_image`) and by the typed
/// [`blazen_future_take_three_d_generate_result`] taker. Holds the encoded
/// 3D model bytes (typically GLB / gltf-binary) and an IANA MIME type
/// string so foreign callers can dispatch on the format without sniffing
/// the buffer.
#[cfg(feature = "triposr")]
pub struct BlazenThreeDGenerateResult(pub(crate) InnerThreeDGenerateResult);

#[cfg(feature = "triposr")]
impl BlazenThreeDGenerateResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenThreeDGenerateResult {
        Box::into_raw(Box::new(self))
    }
}

#[cfg(feature = "triposr")]
impl From<InnerThreeDGenerateResult> for BlazenThreeDGenerateResult {
    fn from(inner: InnerThreeDGenerateResult) -> Self {
        Self(inner)
    }
}

/// Pops a typed `ThreeDGenerateResult` out of `fut`. On success returns
/// `0` and writes a caller-owned `*mut BlazenThreeDGenerateResult` into
/// `out`; on failure returns `-1` and writes a caller-owned
/// `*mut BlazenError` into `err`.
///
/// `out` / `err` may be null when the caller wants to discard the value.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by a per-engine cabi
/// 3D-generation wrapper (e.g. `blazen_triposr_provider_generate_from_image`),
/// not yet freed, and not concurrently freed from another thread. `out`
/// / `err` must be null OR writable pointers to the appropriate slot.
#[unsafe(no_mangle)]
#[cfg(feature = "triposr")]
pub unsafe extern "C" fn blazen_future_take_three_d_generate_result(
    fut: *mut crate::future::BlazenFuture,
    out: *mut *mut BlazenThreeDGenerateResult,
    err: *mut *mut crate::error::BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { crate::future::BlazenFuture::take_typed::<InnerThreeDGenerateResult>(fut) } {
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
                    *err = crate::error::BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
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
#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
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
#[cfg(feature = "triposr")]
pub unsafe extern "C" fn blazen_three_d_generate_result_free(
    result: *mut BlazenThreeDGenerateResult,
) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}

#[cfg(test)]
#[cfg(feature = "triposr")]
mod three_d_generate_result_tests {
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
