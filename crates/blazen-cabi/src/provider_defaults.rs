//! C ABI wrappers for the provider-defaults hierarchy:
//!
//! - [`BlazenBaseProviderDefaults`]
//! - [`BlazenCompletionProviderDefaults`]
//! - [`BlazenEmbeddingProviderDefaults`]
//! - 9 role-specific defaults:
//!   [`BlazenAudioSpeechProviderDefaults`],
//!   [`BlazenAudioMusicProviderDefaults`],
//!   [`BlazenVoiceCloningProviderDefaults`],
//!   [`BlazenImageGenerationProviderDefaults`],
//!   [`BlazenImageUpscaleProviderDefaults`],
//!   [`BlazenVideoProviderDefaults`],
//!   [`BlazenTranscriptionProviderDefaults`],
//!   [`BlazenThreeDProviderDefaults`],
//!   [`BlazenBackgroundRemovalProviderDefaults`].
//!
//! V1 surface: constructors, free, getters/setters for plain data fields, and
//! base composition (`set_base` / `base` accessors). Hooks (`before_request`,
//! `before_completion`, role-specific `before` hooks) are intentionally not
//! exposed here — those need a callback-vtable mechanism similar to
//! `step_handler.rs` / `stream_sink.rs` and land in Phase C.
//!
//! ## JSON-blob fields
//!
//! `CompletionProviderDefaults::tools` is a `Vec<ToolDefinition>` and
//! `response_format` is `Option<serde_json::Value>`. The cabi exposes these
//! via opaque JSON strings on the boundary — the Ruby wrapper marshals to
//! JSON before set and parses on get. Invalid JSON on set is silently
//! treated as clear.

#![allow(dead_code)]

use std::ffi::c_char;

use blazen_llm::providers::defaults::{
    AudioMusicProviderDefaults as InnerAudioMusicProviderDefaults,
    AudioSpeechProviderDefaults as InnerAudioSpeechProviderDefaults,
    BackgroundRemovalProviderDefaults as InnerBackgroundRemovalProviderDefaults,
    BaseProviderDefaults as InnerBaseProviderDefaults,
    CompletionProviderDefaults as InnerCompletionProviderDefaults,
    EmbeddingProviderDefaults as InnerEmbeddingProviderDefaults,
    ImageGenerationProviderDefaults as InnerImageGenerationProviderDefaults,
    ImageUpscaleProviderDefaults as InnerImageUpscaleProviderDefaults,
    ThreeDProviderDefaults as InnerThreeDProviderDefaults,
    TranscriptionProviderDefaults as InnerTranscriptionProviderDefaults,
    VideoProviderDefaults as InnerVideoProviderDefaults,
    VoiceCloningProviderDefaults as InnerVoiceCloningProviderDefaults,
};
use blazen_llm::types::ToolDefinition;

use crate::string::{alloc_cstring, cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Parse a NUL-terminated JSON string into `Vec<ToolDefinition>`. Returns an
/// empty vec on null / invalid input.
///
/// # Safety
///
/// `ptr` must be null OR a valid NUL-terminated UTF-8 buffer.
unsafe fn parse_tools_json(ptr: *const c_char) -> Vec<ToolDefinition> {
    // SAFETY: forwarded to `cstr_to_str`; caller upholds the same contract.
    let Some(s) = (unsafe { cstr_to_str(ptr) }) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<ToolDefinition>>(s).unwrap_or_default()
}

/// Serialize a `&[ToolDefinition]` as a JSON string into a caller-owned C
/// string. Returns null only on a malformed conversion.
fn alloc_tools_json(tools: &[ToolDefinition]) -> *mut c_char {
    match serde_json::to_string(tools) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Parse a NUL-terminated JSON string into a `serde_json::Value`. Returns
/// `None` on null input. Returns `Some(Value::Null)` on a non-UTF-8 buffer or
/// when the JSON is malformed.
///
/// # Safety
///
/// `ptr` must be null OR a valid NUL-terminated UTF-8 buffer.
unsafe fn parse_opt_json_value(ptr: *const c_char) -> Option<serde_json::Value> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller upholds the borrow contract.
    let Some(s) = (unsafe { cstr_to_str(ptr) }) else {
        return Some(serde_json::Value::Null);
    };
    Some(serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::Value::Null))
}

/// Serialize an optional JSON value to a caller-owned C string. Returns null
/// for `None`.
fn alloc_opt_json(value: Option<&serde_json::Value>) -> *mut c_char {
    let Some(v) = value else {
        return std::ptr::null_mut();
    };
    match serde_json::to_string(v) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

// ===========================================================================
// BlazenBaseProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::providers::BaseProviderDefaults`].
#[repr(C)]
pub struct BlazenBaseProviderDefaults(pub(crate) InnerBaseProviderDefaults);

impl BlazenBaseProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenBaseProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerBaseProviderDefaults> for BlazenBaseProviderDefaults {
    fn from(inner: InnerBaseProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `BaseProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_base_provider_defaults_new() -> *mut BlazenBaseProviderDefaults {
    BlazenBaseProviderDefaults(InnerBaseProviderDefaults::default()).into_ptr()
}

/// Returns whether a `before_request` hook is set. V1 hooks are not yet
/// settable from C — this always returns `false` until Phase C wires the
/// vtable in.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_defaults_has_before_request(
    handle: *const BlazenBaseProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before_request.is_some()
}

/// Frees a `BlazenBaseProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_base_provider_defaults_new`] or a getter that clones it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_defaults_free(
    handle: *mut BlazenBaseProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenCompletionProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::providers::CompletionProviderDefaults`].
#[repr(C)]
pub struct BlazenCompletionProviderDefaults(pub(crate) InnerCompletionProviderDefaults);

impl BlazenCompletionProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenCompletionProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerCompletionProviderDefaults> for BlazenCompletionProviderDefaults {
    fn from(inner: InnerCompletionProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `CompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_completion_provider_defaults_new() -> *mut BlazenCompletionProviderDefaults
{
    BlazenCompletionProviderDefaults(InnerCompletionProviderDefaults::default()).into_ptr()
}

/// Sets the optional `system_prompt`. Null `prompt` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`.
/// `prompt` must be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_set_system_prompt(
    handle: *mut BlazenCompletionProviderDefaults,
    prompt: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on `prompt`.
    r.0.system_prompt = unsafe { cstr_to_opt_string(prompt) };
}

/// Returns the `system_prompt` field as a caller-owned C string, or null when
/// unset.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_system_prompt(
    handle: *const BlazenCompletionProviderDefaults,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.system_prompt
        .as_deref()
        .map_or(std::ptr::null_mut(), alloc_cstring)
}

/// Replaces the `tools` field by parsing `json` as a JSON array of
/// [`ToolDefinition`] objects. Null / invalid JSON clears the list.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`. `json`
/// must be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_set_tools_json(
    handle: *mut BlazenCompletionProviderDefaults,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the borrow contract on `json`.
    r.0.tools = unsafe { parse_tools_json(json) };
}

/// Returns the `tools` field as a JSON-encoded caller-owned C string. Returns
/// `"[]"` when no tools are set.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_tools_json(
    handle: *const BlazenCompletionProviderDefaults,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_tools_json(&r.0.tools)
}

/// Replaces the `response_format` field by parsing `json`. Null clears it;
/// invalid JSON stores `Value::Null`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`. `json`
/// must be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_set_response_format_json(
    handle: *mut BlazenCompletionProviderDefaults,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the borrow contract.
    r.0.response_format = unsafe { parse_opt_json_value(json) };
}

/// Returns the `response_format` field as a JSON-encoded caller-owned C
/// string, or null when unset.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_response_format_json(
    handle: *const BlazenCompletionProviderDefaults,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_opt_json(r.0.response_format.as_ref())
}

/// Replaces the `base` field with a clone of the supplied
/// `BlazenBaseProviderDefaults`. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`. `base`
/// must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_set_base(
    handle: *mut BlazenCompletionProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field as a caller-owned
/// `BlazenBaseProviderDefaults` handle. Free with
/// [`blazen_base_provider_defaults_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_base(
    handle: *const BlazenCompletionProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether a `before_completion` hook is set. V1 always returns
/// `false` (hooks land in Phase C).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_has_before_completion(
    handle: *const BlazenCompletionProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before_completion.is_some()
}

/// Frees a `BlazenCompletionProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_completion_provider_defaults_new`] or a clone getter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_provider_defaults_free(
    handle: *mut BlazenCompletionProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenEmbeddingProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::providers::EmbeddingProviderDefaults`].
#[repr(C)]
pub struct BlazenEmbeddingProviderDefaults(pub(crate) InnerEmbeddingProviderDefaults);

impl BlazenEmbeddingProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenEmbeddingProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerEmbeddingProviderDefaults> for BlazenEmbeddingProviderDefaults {
    fn from(inner: InnerEmbeddingProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `EmbeddingProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_embedding_provider_defaults_new() -> *mut BlazenEmbeddingProviderDefaults {
    BlazenEmbeddingProviderDefaults(InnerEmbeddingProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field with a clone of the supplied handle. Null `base`
/// is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingProviderDefaults`. `base`
/// must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_provider_defaults_set_base(
    handle: *mut BlazenEmbeddingProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_provider_defaults_base(
    handle: *const BlazenEmbeddingProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Frees a `BlazenEmbeddingProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_embedding_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_provider_defaults_free(
    handle: *mut BlazenEmbeddingProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// Role-specific defaults
//
// The 9 role-specific defaults types all share the same shape:
//   { base: BaseProviderDefaults, before: Option<Hook> }
//
// The function bodies are mechanically identical apart from the inner type
// and the per-role function names. We hand-roll each set rather than using a
// `macro_rules!` block because `cbindgen` (running on stable Rust without
// `-Zunpretty=expanded`) cannot see through macro invocations, so a macro
// would silently drop these symbols from the generated header.
// ---------------------------------------------------------------------------

// ===========================================================================
// BlazenAudioSpeechProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerAudioSpeechProviderDefaults`].
#[repr(C)]
pub struct BlazenAudioSpeechProviderDefaults(pub(crate) InnerAudioSpeechProviderDefaults);

impl BlazenAudioSpeechProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenAudioSpeechProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerAudioSpeechProviderDefaults> for BlazenAudioSpeechProviderDefaults {
    fn from(inner: InnerAudioSpeechProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `AudioSpeechProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_audio_speech_provider_defaults_new()
-> *mut BlazenAudioSpeechProviderDefaults {
    BlazenAudioSpeechProviderDefaults(InnerAudioSpeechProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field with a clone of the supplied handle. Null `base`
/// is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioSpeechProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_speech_provider_defaults_set_base(
    handle: *mut BlazenAudioSpeechProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioSpeechProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_speech_provider_defaults_base(
    handle: *const BlazenAudioSpeechProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`
/// (hooks land in Phase C).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioSpeechProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_speech_provider_defaults_has_before(
    handle: *const BlazenAudioSpeechProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenAudioSpeechProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_audio_speech_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_speech_provider_defaults_free(
    handle: *mut BlazenAudioSpeechProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenAudioMusicProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerAudioMusicProviderDefaults`].
#[repr(C)]
pub struct BlazenAudioMusicProviderDefaults(pub(crate) InnerAudioMusicProviderDefaults);

impl BlazenAudioMusicProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenAudioMusicProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerAudioMusicProviderDefaults> for BlazenAudioMusicProviderDefaults {
    fn from(inner: InnerAudioMusicProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `AudioMusicProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_audio_music_provider_defaults_new() -> *mut BlazenAudioMusicProviderDefaults
{
    BlazenAudioMusicProviderDefaults(InnerAudioMusicProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioMusicProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_music_provider_defaults_set_base(
    handle: *mut BlazenAudioMusicProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioMusicProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_music_provider_defaults_base(
    handle: *const BlazenAudioMusicProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenAudioMusicProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_music_provider_defaults_has_before(
    handle: *const BlazenAudioMusicProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenAudioMusicProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_audio_music_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_audio_music_provider_defaults_free(
    handle: *mut BlazenAudioMusicProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenVoiceCloningProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerVoiceCloningProviderDefaults`].
#[repr(C)]
pub struct BlazenVoiceCloningProviderDefaults(pub(crate) InnerVoiceCloningProviderDefaults);

impl BlazenVoiceCloningProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenVoiceCloningProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVoiceCloningProviderDefaults> for BlazenVoiceCloningProviderDefaults {
    fn from(inner: InnerVoiceCloningProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `VoiceCloningProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_voice_cloning_provider_defaults_new()
-> *mut BlazenVoiceCloningProviderDefaults {
    BlazenVoiceCloningProviderDefaults(InnerVoiceCloningProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloningProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_cloning_provider_defaults_set_base(
    handle: *mut BlazenVoiceCloningProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloningProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_cloning_provider_defaults_base(
    handle: *const BlazenVoiceCloningProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVoiceCloningProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_cloning_provider_defaults_has_before(
    handle: *const BlazenVoiceCloningProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenVoiceCloningProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_voice_cloning_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_voice_cloning_provider_defaults_free(
    handle: *mut BlazenVoiceCloningProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenImageGenerationProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerImageGenerationProviderDefaults`].
#[repr(C)]
pub struct BlazenImageGenerationProviderDefaults(pub(crate) InnerImageGenerationProviderDefaults);

impl BlazenImageGenerationProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenImageGenerationProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerImageGenerationProviderDefaults> for BlazenImageGenerationProviderDefaults {
    fn from(inner: InnerImageGenerationProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `ImageGenerationProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_image_generation_provider_defaults_new()
-> *mut BlazenImageGenerationProviderDefaults {
    BlazenImageGenerationProviderDefaults(InnerImageGenerationProviderDefaults::default())
        .into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageGenerationProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_generation_provider_defaults_set_base(
    handle: *mut BlazenImageGenerationProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageGenerationProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_generation_provider_defaults_base(
    handle: *const BlazenImageGenerationProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageGenerationProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_generation_provider_defaults_has_before(
    handle: *const BlazenImageGenerationProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenImageGenerationProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_image_generation_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_generation_provider_defaults_free(
    handle: *mut BlazenImageGenerationProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenImageUpscaleProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerImageUpscaleProviderDefaults`].
#[repr(C)]
pub struct BlazenImageUpscaleProviderDefaults(pub(crate) InnerImageUpscaleProviderDefaults);

impl BlazenImageUpscaleProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenImageUpscaleProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerImageUpscaleProviderDefaults> for BlazenImageUpscaleProviderDefaults {
    fn from(inner: InnerImageUpscaleProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `ImageUpscaleProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_image_upscale_provider_defaults_new()
-> *mut BlazenImageUpscaleProviderDefaults {
    BlazenImageUpscaleProviderDefaults(InnerImageUpscaleProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageUpscaleProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_upscale_provider_defaults_set_base(
    handle: *mut BlazenImageUpscaleProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageUpscaleProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_upscale_provider_defaults_base(
    handle: *const BlazenImageUpscaleProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenImageUpscaleProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_upscale_provider_defaults_has_before(
    handle: *const BlazenImageUpscaleProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenImageUpscaleProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_image_upscale_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_image_upscale_provider_defaults_free(
    handle: *mut BlazenImageUpscaleProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenVideoProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerVideoProviderDefaults`].
#[repr(C)]
pub struct BlazenVideoProviderDefaults(pub(crate) InnerVideoProviderDefaults);

impl BlazenVideoProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenVideoProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerVideoProviderDefaults> for BlazenVideoProviderDefaults {
    fn from(inner: InnerVideoProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `VideoProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_video_provider_defaults_new() -> *mut BlazenVideoProviderDefaults {
    BlazenVideoProviderDefaults(InnerVideoProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_provider_defaults_set_base(
    handle: *mut BlazenVideoProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_provider_defaults_base(
    handle: *const BlazenVideoProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenVideoProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_provider_defaults_has_before(
    handle: *const BlazenVideoProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenVideoProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_video_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_video_provider_defaults_free(
    handle: *mut BlazenVideoProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenTranscriptionProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerTranscriptionProviderDefaults`].
#[repr(C)]
pub struct BlazenTranscriptionProviderDefaults(pub(crate) InnerTranscriptionProviderDefaults);

impl BlazenTranscriptionProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenTranscriptionProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTranscriptionProviderDefaults> for BlazenTranscriptionProviderDefaults {
    fn from(inner: InnerTranscriptionProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `TranscriptionProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_transcription_provider_defaults_new()
-> *mut BlazenTranscriptionProviderDefaults {
    BlazenTranscriptionProviderDefaults(InnerTranscriptionProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_provider_defaults_set_base(
    handle: *mut BlazenTranscriptionProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_provider_defaults_base(
    handle: *const BlazenTranscriptionProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTranscriptionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_provider_defaults_has_before(
    handle: *const BlazenTranscriptionProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenTranscriptionProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_transcription_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_transcription_provider_defaults_free(
    handle: *mut BlazenTranscriptionProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenThreeDProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerThreeDProviderDefaults`].
#[repr(C)]
pub struct BlazenThreeDProviderDefaults(pub(crate) InnerThreeDProviderDefaults);

impl BlazenThreeDProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenThreeDProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerThreeDProviderDefaults> for BlazenThreeDProviderDefaults {
    fn from(inner: InnerThreeDProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `ThreeDProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_three_d_provider_defaults_new() -> *mut BlazenThreeDProviderDefaults {
    BlazenThreeDProviderDefaults(InnerThreeDProviderDefaults::default()).into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_provider_defaults_set_base(
    handle: *mut BlazenThreeDProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_provider_defaults_base(
    handle: *const BlazenThreeDProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenThreeDProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_provider_defaults_has_before(
    handle: *const BlazenThreeDProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenThreeDProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_three_d_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_three_d_provider_defaults_free(
    handle: *mut BlazenThreeDProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenBackgroundRemovalProviderDefaults
// ===========================================================================

/// Opaque wrapper around [`InnerBackgroundRemovalProviderDefaults`].
#[repr(C)]
pub struct BlazenBackgroundRemovalProviderDefaults(
    pub(crate) InnerBackgroundRemovalProviderDefaults,
);

impl BlazenBackgroundRemovalProviderDefaults {
    pub(crate) fn into_ptr(self) -> *mut BlazenBackgroundRemovalProviderDefaults {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerBackgroundRemovalProviderDefaults> for BlazenBackgroundRemovalProviderDefaults {
    fn from(inner: InnerBackgroundRemovalProviderDefaults) -> Self {
        Self(inner)
    }
}

/// Constructs an empty `BackgroundRemovalProviderDefaults`.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_background_removal_provider_defaults_new()
-> *mut BlazenBackgroundRemovalProviderDefaults {
    BlazenBackgroundRemovalProviderDefaults(InnerBackgroundRemovalProviderDefaults::default())
        .into_ptr()
}

/// Replaces the `base` field. Null `base` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalProviderDefaults`.
/// `base` must be null OR a live `BlazenBaseProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_provider_defaults_set_base(
    handle: *mut BlazenBackgroundRemovalProviderDefaults,
    base: *const BlazenBaseProviderDefaults,
) {
    if handle.is_null() || base.is_null() {
        return;
    }
    // SAFETY: caller guarantees live handles.
    let r = unsafe { &mut *handle };
    let b = unsafe { &*base };
    r.0.base = b.0.clone();
}

/// Returns a clone of the inner `base` field.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_provider_defaults_base(
    handle: *const BlazenBackgroundRemovalProviderDefaults,
) -> *mut BlazenBaseProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenBaseProviderDefaults(r.0.base.clone()).into_ptr()
}

/// Returns whether the typed `before` hook is set. V1 always returns `false`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBackgroundRemovalProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_provider_defaults_has_before(
    handle: *const BlazenBackgroundRemovalProviderDefaults,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.before.is_some()
}

/// Frees a `BlazenBackgroundRemovalProviderDefaults`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_background_removal_provider_defaults_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_background_removal_provider_defaults_free(
    handle: *mut BlazenBackgroundRemovalProviderDefaults,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
