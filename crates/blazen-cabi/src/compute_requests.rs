//! Compute request marshalling. Opaque handles wrap the nine typed request
//! records from `blazen_llm::compute::*`:
//!
//! - [`ImageRequest`](blazen_llm::compute::ImageRequest)
//! - [`UpscaleRequest`](blazen_llm::compute::UpscaleRequest)
//! - [`VideoRequest`](blazen_llm::compute::VideoRequest)
//! - [`SpeechRequest`](blazen_llm::compute::SpeechRequest)
//! - [`VoiceCloneRequest`](blazen_llm::compute::VoiceCloneRequest)
//! - [`MusicRequest`](blazen_llm::compute::MusicRequest)
//! - [`TranscriptionRequest`](blazen_llm::compute::TranscriptionRequest)
//! - [`ThreeDRequest`](blazen_llm::compute::ThreeDRequest)
//! - [`BackgroundRemovalRequest`](blazen_llm::compute::BackgroundRemovalRequest)
//!
//! These will be consumed by the Ruby `FFI::Library` wrapper to build typed
//! compute requests without resorting to opaque JSON blobs. Phase A0.cabi.
//!
//! # Ownership conventions
//!
//! Every `blazen_*_request_new` returns a heap-allocated handle the caller
//! owns; release with the matching `_free`. Setters consume nothing (they
//! just mutate the handle in place). Getters returning `*mut c_char` allocate
//! a fresh C string the caller releases via
//! [`crate::string::blazen_string_free`].
//!
//! # Optional fields
//!
//! Optional input fields use a paired `_set_<field>` / `_clear_<field>` API
//! matching `llm_records.rs`. For optional `String` fields, `_set_<field>`
//! accepts a null pointer as shorthand for clearing.
//!
//! # JSON-blob fields
//!
//! The `parameters` field on every request is a `serde_json::Value`. We do
//! not expose its inner structure across the C boundary — callers serialize
//! their parameters to a JSON string and pass it through
//! `_set_parameters_json` / `_parameters_json`. Invalid JSON in the setter is
//! silently treated as a clear (sets `Value::Null`-equivalent empty object).
//!
//! # `audio_source` on `TranscriptionRequest`
//!
//! The `audio_source` field is `Option<MediaSource>`, where `MediaSource`
//! has five variants. Exposing the full enum across the C boundary is more
//! ergonomic via flat per-variant setters than via a single opaque handle:
//!
//! - [`blazen_transcription_request_set_audio_source_url`]
//! - [`blazen_transcription_request_set_audio_source_base64`]
//! - [`blazen_transcription_request_set_audio_source_file`]
//! - [`blazen_transcription_request_clear_audio_source`]
//!
//! `ProviderFile` and `Handle` are out of scope for the cabi surface (they
//! depend on the Rust `ContentStore` infrastructure which has no FFI hook
//! today) and are best driven from the Rust side until that lands.

#![allow(dead_code)]

use std::ffi::c_char;
use std::path::PathBuf;

use blazen_llm::compute::{
    BackgroundRemovalRequest as InnerBackgroundRemovalRequest, ImageRequest as InnerImageRequest,
    MusicRequest as InnerMusicRequest, SpeechRequest as InnerSpeechRequest,
    ThreeDRequest as InnerThreeDRequest, TranscriptionRequest as InnerTranscriptionRequest,
    UpscaleRequest as InnerUpscaleRequest, VideoRequest as InnerVideoRequest,
    VoiceCloneRequest as InnerVoiceCloneRequest,
};

use crate::string::{alloc_cstring, cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Best-effort parse of a JSON string into a `serde_json::Value`. Returns an
/// empty JSON object on null/invalid input — matches the upstream defaults
/// for every `parameters` field on the compute requests.
///
/// # Safety
///
/// `ptr` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call.
unsafe fn parse_json_value(ptr: *const c_char) -> serde_json::Value {
    // SAFETY: forwarded to `cstr_to_str`; caller upholds the same contract.
    let Some(s) = (unsafe { cstr_to_str(ptr) }) else {
        return serde_json::Value::Object(serde_json::Map::new());
    };
    serde_json::from_str(s).unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
}

/// Returns the JSON-encoded text of a `serde_json::Value` as a caller-owned
/// C string. Returns null only on a malformed string conversion (e.g. interior
/// NUL bytes); in practice the upstream `serde_json::to_string` cannot fail
/// for the value types we round-trip.
fn alloc_json_cstring(value: &serde_json::Value) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

// ===========================================================================
// ImageRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::ImageRequest`].
#[repr(C)]
pub struct BlazenImageRequest(pub(crate) InnerImageRequest);

impl BlazenImageRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenImageRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerImageRequest> for BlazenImageRequest {
    fn from(inner: InnerImageRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `ImageRequest` with the given prompt and every optional
/// field unset. Returns null if `prompt` is null or non-UTF-8.
///
/// # Safety
///
/// `prompt` must be null OR point to a NUL-terminated UTF-8 buffer valid for
/// the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_new(
    prompt: *const c_char,
) -> *mut BlazenImageRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    BlazenImageRequest(InnerImageRequest::new(prompt)).into_ptr()
}

/// Sets the optional `negative_prompt`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_set_negative_prompt(
    handle: *mut BlazenImageRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.negative_prompt = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `width`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_set_width(
    handle: *mut BlazenImageRequest,
    value: u32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.width = Some(value);
}

/// Clears `width` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_clear_width(handle: *mut BlazenImageRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.width = None;
}

/// Sets the optional `height`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_set_height(
    handle: *mut BlazenImageRequest,
    value: u32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.height = Some(value);
}

/// Clears `height` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_clear_height(handle: *mut BlazenImageRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.height = None;
}

/// Sets the optional `num_images`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_set_num_images(
    handle: *mut BlazenImageRequest,
    value: u32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.num_images = Some(value);
}

/// Clears `num_images` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_clear_num_images(handle: *mut BlazenImageRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.num_images = None;
}

/// Sets the optional `model` field. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_set_model(
    handle: *mut BlazenImageRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
/// Null / invalid JSON resets the field to an empty JSON object.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`. `json` must be null
/// OR a NUL-terminated UTF-8 buffer valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_set_parameters_json(
    handle: *mut BlazenImageRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `prompt` field as a caller-owned C string. Returns null on a
/// null handle. Free with `blazen_string_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_prompt(
    handle: *const BlazenImageRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.prompt)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_parameters_json(
    handle: *const BlazenImageRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenImageRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by [`blazen_image_request_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_request_free(handle: *mut BlazenImageRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// UpscaleRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::UpscaleRequest`].
#[repr(C)]
pub struct BlazenUpscaleRequest(pub(crate) InnerUpscaleRequest);

impl BlazenUpscaleRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenUpscaleRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerUpscaleRequest> for BlazenUpscaleRequest {
    fn from(inner: InnerUpscaleRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `UpscaleRequest` from the image URL and scale factor.
/// Returns null if `image_url` is null or non-UTF-8.
///
/// # Safety
///
/// `image_url` must be null OR a NUL-terminated UTF-8 buffer valid for the
/// duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_new(
    image_url: *const c_char,
    scale: f32,
) -> *mut BlazenUpscaleRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(image_url) = (unsafe { cstr_to_str(image_url) }) else {
        return std::ptr::null_mut();
    };
    BlazenUpscaleRequest(InnerUpscaleRequest::new(image_url, scale)).into_ptr()
}

/// Sets the optional `model` field. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenUpscaleRequest`. `value` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_set_model(
    handle: *mut BlazenUpscaleRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenUpscaleRequest`. `json` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_set_parameters_json(
    handle: *mut BlazenUpscaleRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `image_url` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenUpscaleRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_image_url(
    handle: *const BlazenUpscaleRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.image_url)
}

/// Returns the `scale` factor. Returns `0.0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenUpscaleRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_scale(handle: *const BlazenUpscaleRequest) -> f32 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.scale
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenUpscaleRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_parameters_json(
    handle: *const BlazenUpscaleRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenUpscaleRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_upscale_request_free(handle: *mut BlazenUpscaleRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// VideoRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::VideoRequest`].
#[repr(C)]
pub struct BlazenVideoRequest(pub(crate) InnerVideoRequest);

impl BlazenVideoRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenVideoRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVideoRequest> for BlazenVideoRequest {
    fn from(inner: InnerVideoRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new text-to-video request from the prompt. Returns null if
/// `prompt` is null or non-UTF-8.
///
/// # Safety
///
/// `prompt` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_new(
    prompt: *const c_char,
) -> *mut BlazenVideoRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    BlazenVideoRequest(InnerVideoRequest::new(prompt)).into_ptr()
}

/// Sets the optional `image_url` (image-to-video). Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_image_url(
    handle: *mut BlazenVideoRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.image_url = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `duration_seconds`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_duration_seconds(
    handle: *mut BlazenVideoRequest,
    value: f32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.duration_seconds = Some(value);
}

/// Clears `duration_seconds` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_clear_duration_seconds(
    handle: *mut BlazenVideoRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.duration_seconds = None;
}

/// Sets the optional `negative_prompt`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_negative_prompt(
    handle: *mut BlazenVideoRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.negative_prompt = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `width`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_width(
    handle: *mut BlazenVideoRequest,
    value: u32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.width = Some(value);
}

/// Clears `width` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_clear_width(handle: *mut BlazenVideoRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.width = None;
}

/// Sets the optional `height`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_height(
    handle: *mut BlazenVideoRequest,
    value: u32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.height = Some(value);
}

/// Clears `height` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_clear_height(handle: *mut BlazenVideoRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.height = None;
}

/// Sets the optional `model`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_model(
    handle: *mut BlazenVideoRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`. `json` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_set_parameters_json(
    handle: *mut BlazenVideoRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `prompt` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_prompt(
    handle: *const BlazenVideoRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.prompt)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_parameters_json(
    handle: *const BlazenVideoRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenVideoRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_request_free(handle: *mut BlazenVideoRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// SpeechRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::SpeechRequest`].
#[repr(C)]
pub struct BlazenSpeechRequest(pub(crate) InnerSpeechRequest);

impl BlazenSpeechRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenSpeechRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerSpeechRequest> for BlazenSpeechRequest {
    fn from(inner: InnerSpeechRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `SpeechRequest` from the input text. Returns null if
/// `text` is null or non-UTF-8.
///
/// # Safety
///
/// `text` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_new(
    text: *const c_char,
) -> *mut BlazenSpeechRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(text) = (unsafe { cstr_to_str(text) }) else {
        return std::ptr::null_mut();
    };
    BlazenSpeechRequest(InnerSpeechRequest::new(text)).into_ptr()
}

/// Sets the optional `voice`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_set_voice(
    handle: *mut BlazenSpeechRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.voice = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `voice_url` (reference audio for voice cloning). Null
/// `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_set_voice_url(
    handle: *mut BlazenSpeechRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.voice_url = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `language` (ISO-639-1). Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_set_language(
    handle: *mut BlazenSpeechRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.language = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `speed` multiplier (1.0 = normal).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_set_speed(
    handle: *mut BlazenSpeechRequest,
    value: f32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.speed = Some(value);
}

/// Clears `speed` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_clear_speed(handle: *mut BlazenSpeechRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.speed = None;
}

/// Sets the optional `model`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_set_model(
    handle: *mut BlazenSpeechRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`. `json` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_set_parameters_json(
    handle: *mut BlazenSpeechRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `text` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_text(
    handle: *const BlazenSpeechRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.text)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenSpeechRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_parameters_json(
    handle: *const BlazenSpeechRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenSpeechRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_speech_request_free(handle: *mut BlazenSpeechRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// VoiceCloneRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::VoiceCloneRequest`].
#[repr(C)]
pub struct BlazenVoiceCloneRequest(pub(crate) InnerVoiceCloneRequest);

impl BlazenVoiceCloneRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenVoiceCloneRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVoiceCloneRequest> for BlazenVoiceCloneRequest {
    fn from(inner: InnerVoiceCloneRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `VoiceCloneRequest` with the given human-readable name
/// and an empty `reference_urls` list. Returns null if `name` is null or
/// non-UTF-8. Push reference URLs via
/// [`blazen_voice_clone_request_reference_urls_push`] before submitting.
///
/// # Safety
///
/// `name` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_new(
    name: *const c_char,
) -> *mut BlazenVoiceCloneRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        return std::ptr::null_mut();
    };
    BlazenVoiceCloneRequest(InnerVoiceCloneRequest::new(name, Vec::new())).into_ptr()
}

/// Appends `url` (a NUL-terminated UTF-8 string) to the request's
/// `reference_urls` vec. No-op on a null handle or non-UTF-8 `url`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`. `url` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_reference_urls_push(
    handle: *mut BlazenVoiceCloneRequest,
    url: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `url`.
    let Some(url) = (unsafe { cstr_to_str(url) }) else {
        return;
    };
    let url = url.to_owned();
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.reference_urls.push(url);
}

/// Returns the number of entries in `reference_urls`. Returns `0` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_reference_urls_count(
    handle: *const BlazenVoiceCloneRequest,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.reference_urls.len()
}

/// Returns the `idx`-th `reference_urls` entry as a caller-owned C string.
/// Returns null on a null handle or out-of-range index.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_reference_urls_get(
    handle: *const BlazenVoiceCloneRequest,
    idx: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match r.0.reference_urls.get(idx) {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Sets the optional `language`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`. `value` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_set_language(
    handle: *mut BlazenVoiceCloneRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.language = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `description`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`. `value` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_set_description(
    handle: *mut BlazenVoiceCloneRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.description = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`. `json` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_set_parameters_json(
    handle: *mut BlazenVoiceCloneRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `name` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_name(
    handle: *const BlazenVoiceCloneRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.name)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloneRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_parameters_json(
    handle: *const BlazenVoiceCloneRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenVoiceCloneRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_clone_request_free(handle: *mut BlazenVoiceCloneRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// MusicRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::MusicRequest`].
#[repr(C)]
pub struct BlazenMusicRequest(pub(crate) InnerMusicRequest);

impl BlazenMusicRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenMusicRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerMusicRequest> for BlazenMusicRequest {
    fn from(inner: InnerMusicRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `MusicRequest` from the prompt. Returns null on null or
/// non-UTF-8 input.
///
/// # Safety
///
/// `prompt` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_new(
    prompt: *const c_char,
) -> *mut BlazenMusicRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    BlazenMusicRequest(InnerMusicRequest::new(prompt)).into_ptr()
}

/// Sets the optional `duration_seconds`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMusicRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_set_duration_seconds(
    handle: *mut BlazenMusicRequest,
    value: f32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.duration_seconds = Some(value);
}

/// Clears `duration_seconds` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMusicRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_clear_duration_seconds(
    handle: *mut BlazenMusicRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.duration_seconds = None;
}

/// Sets the optional `model`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMusicRequest`. `value` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_set_model(
    handle: *mut BlazenMusicRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMusicRequest`. `json` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_set_parameters_json(
    handle: *mut BlazenMusicRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `prompt` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMusicRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_prompt(
    handle: *const BlazenMusicRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.prompt)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMusicRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_parameters_json(
    handle: *const BlazenMusicRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenMusicRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_music_request_free(handle: *mut BlazenMusicRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// TranscriptionRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::TranscriptionRequest`].
#[repr(C)]
pub struct BlazenTranscriptionRequest(pub(crate) InnerTranscriptionRequest);

impl BlazenTranscriptionRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenTranscriptionRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTranscriptionRequest> for BlazenTranscriptionRequest {
    fn from(inner: InnerTranscriptionRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `TranscriptionRequest` from the source audio URL.
/// Returns null on null or non-UTF-8 input. To transcribe a local file,
/// pass an empty string here and call
/// [`blazen_transcription_request_set_audio_source_file`].
///
/// # Safety
///
/// `audio_url` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_new(
    audio_url: *const c_char,
) -> *mut BlazenTranscriptionRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(audio_url) = (unsafe { cstr_to_str(audio_url) }) else {
        return std::ptr::null_mut();
    };
    BlazenTranscriptionRequest(InnerTranscriptionRequest::new(audio_url)).into_ptr()
}

/// Sets `audio_source` to `MediaSource::Url { url }`. No-op on null/non-UTF-8
/// input.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`. `url` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_audio_source_url(
    handle: *mut BlazenTranscriptionRequest,
    url: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `url`.
    let Some(url) = (unsafe { cstr_to_str(url) }) else {
        return;
    };
    let url = url.to_owned();
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.audio_source = Some(blazen_llm::MediaSource::Url { url });
}

/// Sets `audio_source` to `MediaSource::Base64 { data }`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`. `data` must
/// be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_audio_source_base64(
    handle: *mut BlazenTranscriptionRequest,
    data: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `data`.
    let Some(data) = (unsafe { cstr_to_str(data) }) else {
        return;
    };
    let data = data.to_owned();
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.audio_source = Some(blazen_llm::MediaSource::Base64 { data });
}

/// Sets `audio_source` to `MediaSource::File { path }` so local backends can
/// read audio directly from disk.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`. `path` must
/// be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_audio_source_file(
    handle: *mut BlazenTranscriptionRequest,
    path: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `path`.
    let Some(path) = (unsafe { cstr_to_str(path) }) else {
        return;
    };
    let path = PathBuf::from(path);
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.audio_source = Some(blazen_llm::MediaSource::File { path });
}

/// Clears `audio_source` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_clear_audio_source(
    handle: *mut BlazenTranscriptionRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.audio_source = None;
}

/// Sets the optional `language` hint. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`. `value` must
/// be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_language(
    handle: *mut BlazenTranscriptionRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.language = unsafe { cstr_to_opt_string(value) };
}

/// Sets the `diarize` flag.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_diarize(
    handle: *mut BlazenTranscriptionRequest,
    value: bool,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.diarize = value;
}

/// Sets the optional `model`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`. `value` must
/// be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_model(
    handle: *mut BlazenTranscriptionRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`. `json` must
/// be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_set_parameters_json(
    handle: *mut BlazenTranscriptionRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `audio_url` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_audio_url(
    handle: *const BlazenTranscriptionRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.audio_url)
}

/// Returns the `diarize` flag. Returns `false` on null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_diarize(
    handle: *const BlazenTranscriptionRequest,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.diarize
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_parameters_json(
    handle: *const BlazenTranscriptionRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenTranscriptionRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_request_free(
    handle: *mut BlazenTranscriptionRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// ThreeDRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::ThreeDRequest`].
#[repr(C)]
pub struct BlazenThreeDRequest(pub(crate) InnerThreeDRequest);

impl BlazenThreeDRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenThreeDRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerThreeDRequest> for BlazenThreeDRequest {
    fn from(inner: InnerThreeDRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new text-to-3D request from the prompt. Returns null on
/// null/non-UTF-8 input.
///
/// # Safety
///
/// `prompt` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_new(
    prompt: *const c_char,
) -> *mut BlazenThreeDRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(prompt) = (unsafe { cstr_to_str(prompt) }) else {
        return std::ptr::null_mut();
    };
    BlazenThreeDRequest(InnerThreeDRequest::new(prompt)).into_ptr()
}

/// Sets the optional `image_url` (image-to-3D). Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDRequest`. `value` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_set_image_url(
    handle: *mut BlazenThreeDRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.image_url = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `format` (e.g. `"glb"`, `"obj"`, `"usdz"`). Null `value`
/// clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDRequest`. `value` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_set_format(
    handle: *mut BlazenThreeDRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.format = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `model`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDRequest`. `value` must be
/// null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_set_model(
    handle: *mut BlazenThreeDRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDRequest`. `json` must be null
/// OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_set_parameters_json(
    handle: *mut BlazenThreeDRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `prompt` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_prompt(
    handle: *const BlazenThreeDRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.prompt)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_parameters_json(
    handle: *const BlazenThreeDRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenThreeDRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_request_free(handle: *mut BlazenThreeDRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BackgroundRemovalRequest
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::compute::BackgroundRemovalRequest`].
#[repr(C)]
pub struct BlazenBackgroundRemovalRequest(pub(crate) InnerBackgroundRemovalRequest);

impl BlazenBackgroundRemovalRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenBackgroundRemovalRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerBackgroundRemovalRequest> for BlazenBackgroundRemovalRequest {
    fn from(inner: InnerBackgroundRemovalRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `BackgroundRemovalRequest` from the source image URL.
/// `BackgroundRemovalRequest` has no `::new` constructor upstream; we
/// initialize the struct manually with empty optionals and an empty
/// `parameters` object. Returns null on null or non-UTF-8 input.
///
/// # Safety
///
/// `image_url` must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_request_new(
    image_url: *const c_char,
) -> *mut BlazenBackgroundRemovalRequest {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(image_url) = (unsafe { cstr_to_str(image_url) }) else {
        return std::ptr::null_mut();
    };
    BlazenBackgroundRemovalRequest(InnerBackgroundRemovalRequest {
        image_url: image_url.to_owned(),
        model: None,
        parameters: serde_json::Value::Object(serde_json::Map::new()),
    })
    .into_ptr()
}

/// Sets the optional `model`. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalRequest`. `value`
/// must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_request_set_model(
    handle: *mut BlazenBackgroundRemovalRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `value`.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Replaces the `parameters` field with the JSON object decoded from `json`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalRequest`. `json`
/// must be null OR a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_request_set_parameters_json(
    handle: *mut BlazenBackgroundRemovalRequest,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `json`.
    r.0.parameters = unsafe { parse_json_value(json) };
}

/// Returns the `image_url` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_request_image_url(
    handle: *const BlazenBackgroundRemovalRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.image_url)
}

/// Returns the `parameters` field as a JSON-encoded caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_request_parameters_json(
    handle: *const BlazenBackgroundRemovalRequest,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_json_cstring(&r.0.parameters)
}

/// Frees a `BlazenBackgroundRemovalRequest`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_request_free(
    handle: *mut BlazenBackgroundRemovalRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
