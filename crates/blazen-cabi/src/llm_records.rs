//! LLM-related record marshalling. Wraps `Media`, `ChatMessage`, `ToolCall`,
//! `Tool`, `TokenUsage`, `CompletionRequest`, `CompletionResponse`, and
//! `EmbeddingResponse` from `blazen_uniffi::llm` as opaque C handles.
//!
//! # Ownership conventions
//!
//! All `blazen_*_new` constructors return `*mut Blazen<T>` whose ownership
//! transfers to the C caller. The caller must release the handle with the
//! matching `blazen_*_free`. Caller-owned strings returned by getters
//! (`*mut c_char`) are released with [`crate::string::blazen_string_free`].
//!
//! `*_push` setters take ownership of the pushed handle: the inner record is
//! moved into the container's `Vec` and the original `*mut` is consumed
//! (calling `blazen_*_free` on it afterwards is a double-free). `*_get(idx)`
//! getters clone the indexed item into a freshly-allocated handle the caller
//! owns. Out-of-range indices return null.
//!
//! Optional input fields use a paired `_set_<field>` / `_clear_<field>` API:
//! `_set_<field>` writes `Some(value)`, `_clear_<field>` writes `None`. For
//! `Option<String>` fields, `_set_<field>` accepts a null pointer as
//! shorthand for clearing.
//!
//! # Nested `Vec<Vec<f64>>` for embeddings
//!
//! [`BlazenEmbeddingResponse`] is unusual: the natural getter shape is
//! "vector i, index j". We expose both an indexed `_embedding_get(i, j)` for
//! sparse access and `_embedding_to_buffer(i, out, out_len)` for bulk copy
//! into a caller-supplied `f64` buffer — the bulk variant matters for hot
//! paths in embedding-heavy workloads (RAG, semantic search).

// Foundation utility consumed by R3+ wrappers; flat extern fns are kept by
// the linker regardless, but `pub(crate)` helpers fire dead-code without
// this. Once R3 wires up the typed `complete_blocking` etc., this allow can
// shrink.
#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::llm::{
    ChatMessage as InnerChatMessage, CompletionRequest as InnerCompletionRequest,
    CompletionResponse as InnerCompletionResponse, EmbeddingResponse as InnerEmbeddingResponse,
    Media as InnerMedia, TokenUsage as InnerTokenUsage, Tool as InnerTool,
    ToolCall as InnerToolCall,
};

use crate::string::{alloc_cstring, cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// BlazenMedia
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::Media`].
pub struct BlazenMedia(pub(crate) InnerMedia);

impl BlazenMedia {
    pub(crate) fn into_ptr(self) -> *mut BlazenMedia {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerMedia> for BlazenMedia {
    fn from(inner: InnerMedia) -> Self {
        Self(inner)
    }
}

/// Constructs a new `Media` handle from the three required string fields.
///
/// Returns null if any pointer is null or contains non-UTF-8 bytes. Caller
/// owns the returned handle and must free it with [`blazen_media_free`].
///
/// # Safety
///
/// `kind`, `mime_type`, and `data_base64` must each be null OR point to a
/// NUL-terminated UTF-8 buffer that remains valid for the duration of this
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_media_new(
    kind: *const c_char,
    mime_type: *const c_char,
    data_base64: *const c_char,
) -> *mut BlazenMedia {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on each input.
    let kind = match unsafe { cstr_to_str(kind) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let mime_type = match unsafe { cstr_to_str(mime_type) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let data_base64 = match unsafe { cstr_to_str(data_base64) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    BlazenMedia(InnerMedia {
        kind,
        mime_type,
        data_base64,
    })
    .into_ptr()
}

/// Returns the `kind` field as a caller-owned C string. Returns null on a
/// null handle. Caller frees with `blazen_string_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMedia` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_media_kind(handle: *const BlazenMedia) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenMedia`.
    let m = unsafe { &*handle };
    alloc_cstring(&m.0.kind)
}

/// Returns the `mime_type` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMedia` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_media_mime_type(handle: *const BlazenMedia) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenMedia`.
    let m = unsafe { &*handle };
    alloc_cstring(&m.0.mime_type)
}

/// Returns the `data_base64` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenMedia` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_media_data_base64(handle: *const BlazenMedia) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenMedia`.
    let m = unsafe { &*handle };
    alloc_cstring(&m.0.data_base64)
}

/// Frees a `BlazenMedia` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by
/// [`blazen_media_new`] (or by a `*_get` that returned a cloned media). Calling
/// this twice on the same non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_media_free(handle: *mut BlazenMedia) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenToolCall
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::ToolCall`].
pub struct BlazenToolCall(pub(crate) InnerToolCall);

impl BlazenToolCall {
    pub(crate) fn into_ptr(self) -> *mut BlazenToolCall {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerToolCall> for BlazenToolCall {
    fn from(inner: InnerToolCall) -> Self {
        Self(inner)
    }
}

/// Constructs a new `ToolCall` handle. Returns null if any input is null or
/// non-UTF-8.
///
/// # Safety
///
/// `id`, `name`, and `arguments_json` must each be null OR point to a
/// NUL-terminated UTF-8 buffer valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_call_new(
    id: *const c_char,
    name: *const c_char,
    arguments_json: *const c_char,
) -> *mut BlazenToolCall {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on each input.
    let id = match unsafe { cstr_to_str(id) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let name = match unsafe { cstr_to_str(name) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let arguments_json = match unsafe { cstr_to_str(arguments_json) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    BlazenToolCall(InnerToolCall {
        id,
        name,
        arguments_json,
    })
    .into_ptr()
}

/// Returns the `id` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenToolCall` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_call_id(handle: *const BlazenToolCall) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenToolCall`.
    let t = unsafe { &*handle };
    alloc_cstring(&t.0.id)
}

/// Returns the `name` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenToolCall` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_call_name(handle: *const BlazenToolCall) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenToolCall`.
    let t = unsafe { &*handle };
    alloc_cstring(&t.0.name)
}

/// Returns the `arguments_json` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenToolCall` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_call_arguments_json(
    handle: *const BlazenToolCall,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenToolCall`.
    let t = unsafe { &*handle };
    alloc_cstring(&t.0.arguments_json)
}

/// Frees a `BlazenToolCall` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_call_free(handle: *mut BlazenToolCall) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenTool
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::Tool`].
pub struct BlazenTool(pub(crate) InnerTool);

impl BlazenTool {
    pub(crate) fn into_ptr(self) -> *mut BlazenTool {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTool> for BlazenTool {
    fn from(inner: InnerTool) -> Self {
        Self(inner)
    }
}

/// Constructs a new `Tool` handle. Returns null if any input is null or
/// non-UTF-8.
///
/// # Safety
///
/// `name`, `description`, and `parameters_json` must each be null OR point to
/// a NUL-terminated UTF-8 buffer valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_new(
    name: *const c_char,
    description: *const c_char,
    parameters_json: *const c_char,
) -> *mut BlazenTool {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on each input.
    let name = match unsafe { cstr_to_str(name) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let description = match unsafe { cstr_to_str(description) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let parameters_json = match unsafe { cstr_to_str(parameters_json) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    BlazenTool(InnerTool {
        name,
        description,
        parameters_json,
    })
    .into_ptr()
}

/// Returns the `name` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTool` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_name(handle: *const BlazenTool) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTool`.
    let t = unsafe { &*handle };
    alloc_cstring(&t.0.name)
}

/// Returns the `description` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTool` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_description(handle: *const BlazenTool) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTool`.
    let t = unsafe { &*handle };
    alloc_cstring(&t.0.description)
}

/// Returns the `parameters_json` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTool` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_parameters_json(handle: *const BlazenTool) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTool`.
    let t = unsafe { &*handle };
    alloc_cstring(&t.0.parameters_json)
}

/// Frees a `BlazenTool` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_tool_free(handle: *mut BlazenTool) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenTokenUsage
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::TokenUsage`].
pub struct BlazenTokenUsage(pub(crate) InnerTokenUsage);

impl BlazenTokenUsage {
    pub(crate) fn into_ptr(self) -> *mut BlazenTokenUsage {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerTokenUsage> for BlazenTokenUsage {
    fn from(inner: InnerTokenUsage) -> Self {
        Self(inner)
    }
}

/// Constructs a new `TokenUsage` handle from its five `u64` counters.
///
/// Always succeeds (never returns null). Caller owns the returned handle.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_token_usage_new(
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cached_input_tokens: u64,
    reasoning_tokens: u64,
) -> *mut BlazenTokenUsage {
    BlazenTokenUsage(InnerTokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_input_tokens,
        reasoning_tokens,
    })
    .into_ptr()
}

/// Returns `prompt_tokens`. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTokenUsage` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_token_usage_prompt_tokens(handle: *const BlazenTokenUsage) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTokenUsage`.
    let u = unsafe { &*handle };
    u.0.prompt_tokens
}

/// Returns `completion_tokens`. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTokenUsage` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_token_usage_completion_tokens(
    handle: *const BlazenTokenUsage,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTokenUsage`.
    let u = unsafe { &*handle };
    u.0.completion_tokens
}

/// Returns `total_tokens`. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTokenUsage` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_token_usage_total_tokens(handle: *const BlazenTokenUsage) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTokenUsage`.
    let u = unsafe { &*handle };
    u.0.total_tokens
}

/// Returns `cached_input_tokens`. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTokenUsage` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_token_usage_cached_input_tokens(
    handle: *const BlazenTokenUsage,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTokenUsage`.
    let u = unsafe { &*handle };
    u.0.cached_input_tokens
}

/// Returns `reasoning_tokens`. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenTokenUsage` produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_token_usage_reasoning_tokens(
    handle: *const BlazenTokenUsage,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenTokenUsage`.
    let u = unsafe { &*handle };
    u.0.reasoning_tokens
}

/// Frees a `BlazenTokenUsage` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_token_usage_free(handle: *mut BlazenTokenUsage) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenChatMessage
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::ChatMessage`].
pub struct BlazenChatMessage(pub(crate) InnerChatMessage);

impl BlazenChatMessage {
    pub(crate) fn into_ptr(self) -> *mut BlazenChatMessage {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerChatMessage> for BlazenChatMessage {
    fn from(inner: InnerChatMessage) -> Self {
        Self(inner)
    }
}

/// Constructs a new `ChatMessage` with empty media-parts / tool-calls vecs
/// and unset `tool_call_id` / `name` optionals.
///
/// Returns null if either input is null or non-UTF-8.
///
/// # Safety
///
/// `role` and `content` must each be null OR point to a NUL-terminated UTF-8
/// buffer valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_new(
    role: *const c_char,
    content: *const c_char,
) -> *mut BlazenChatMessage {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on each input.
    let role = match unsafe { cstr_to_str(role) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: same as above.
    let content = match unsafe { cstr_to_str(content) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    BlazenChatMessage(InnerChatMessage {
        role,
        content,
        media_parts: Vec::new(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    })
    .into_ptr()
}

/// Returns the `role` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_role(handle: *const BlazenChatMessage) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    alloc_cstring(&m.0.role)
}

/// Returns the `content` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_content(
    handle: *const BlazenChatMessage,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    alloc_cstring(&m.0.content)
}

/// Pushes a `BlazenMedia` onto the message's `media_parts` vec. Consumes
/// the `media` handle — the caller must NOT free it afterwards. No-op if
/// either pointer is null (the `media` allocation is still freed in that case
/// to avoid a leak when only the handle is null).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`. `media` must be null
/// OR a live `BlazenMedia` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_media_parts_push(
    handle: *mut BlazenChatMessage,
    media: *mut BlazenMedia,
) {
    if media.is_null() {
        return;
    }
    // SAFETY: per the contract, `media` came from `Box::into_raw`; reclaiming
    // ownership here moves the inner record so we can either push it or drop
    // it.
    let media_box = unsafe { Box::from_raw(media) };
    if handle.is_null() {
        // Drop the reclaimed Box so we don't leak it. `media_box` falls out of
        // scope here.
        drop(media_box);
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &mut *handle };
    m.0.media_parts.push(media_box.0);
}

/// Returns the number of entries in the message's `media_parts` vec.
/// Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_media_parts_count(
    handle: *const BlazenChatMessage,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    m.0.media_parts.len()
}

/// Clones the `idx`-th entry from `media_parts` into a fresh `BlazenMedia`
/// handle the caller owns. Returns null if `handle` is null or `idx` is out
/// of range.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_media_parts_get(
    handle: *const BlazenChatMessage,
    idx: usize,
) -> *mut BlazenMedia {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    match m.0.media_parts.get(idx) {
        Some(media) => BlazenMedia(media.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Pushes a `BlazenToolCall` onto the message's `tool_calls` vec. Consumes
/// `tool_call`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`. `tool_call` must be
/// null OR a live `BlazenToolCall`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_tool_calls_push(
    handle: *mut BlazenChatMessage,
    tool_call: *mut BlazenToolCall,
) {
    if tool_call.is_null() {
        return;
    }
    // SAFETY: per the contract, `tool_call` came from `Box::into_raw`.
    let tc_box = unsafe { Box::from_raw(tool_call) };
    if handle.is_null() {
        drop(tc_box);
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &mut *handle };
    m.0.tool_calls.push(tc_box.0);
}

/// Returns the number of entries in `tool_calls`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_tool_calls_count(
    handle: *const BlazenChatMessage,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    m.0.tool_calls.len()
}

/// Clones the `idx`-th tool-call entry. Returns null on out-of-range / null
/// handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_tool_calls_get(
    handle: *const BlazenChatMessage,
    idx: usize,
) -> *mut BlazenToolCall {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    match m.0.tool_calls.get(idx) {
        Some(tc) => BlazenToolCall(tc.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Sets the optional `tool_call_id` field. A null `value` clears the field
/// (sets `None`); a non-null pointer sets `Some(<string>)`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`. `value` must be null
/// OR point to a NUL-terminated UTF-8 buffer valid for the duration of this
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_set_tool_call_id(
    handle: *mut BlazenChatMessage,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    m.0.tool_call_id = unsafe { cstr_to_opt_string(value) };
}

/// Returns the optional `tool_call_id` as a caller-owned C string. Returns
/// null if the field is unset or `handle` is null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_tool_call_id(
    handle: *const BlazenChatMessage,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    match &m.0.tool_call_id {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Sets the optional `name` field. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`. `value` must be null
/// OR point to a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_set_name(
    handle: *mut BlazenChatMessage,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    m.0.name = unsafe { cstr_to_opt_string(value) };
}

/// Returns the optional `name` field as a caller-owned C string. Returns null
/// if unset.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_name(handle: *const BlazenChatMessage) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenChatMessage`.
    let m = unsafe { &*handle };
    match &m.0.name {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenChatMessage` handle (and all owned vec / option contents).
/// No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_chat_message_free(handle: *mut BlazenChatMessage) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenCompletionRequest
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::CompletionRequest`].
pub struct BlazenCompletionRequest(pub(crate) InnerCompletionRequest);

impl BlazenCompletionRequest {
    pub(crate) fn into_ptr(self) -> *mut BlazenCompletionRequest {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerCompletionRequest> for BlazenCompletionRequest {
    fn from(inner: InnerCompletionRequest) -> Self {
        Self(inner)
    }
}

/// Constructs a new `CompletionRequest` with empty `messages`/`tools` vecs and
/// every optional field unset. Always succeeds; caller owns the handle.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_completion_request_new() -> *mut BlazenCompletionRequest {
    BlazenCompletionRequest(InnerCompletionRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        top_p: None,
        model: None,
        response_format_json: None,
        system: None,
    })
    .into_ptr()
}

/// Pushes a `BlazenChatMessage` onto the request's `messages` vec. Consumes
/// `message`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`. `message` must
/// be null OR a live `BlazenChatMessage`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_messages_push(
    handle: *mut BlazenCompletionRequest,
    message: *mut BlazenChatMessage,
) {
    if message.is_null() {
        return;
    }
    // SAFETY: per the contract, `message` came from `Box::into_raw`.
    let msg_box = unsafe { Box::from_raw(message) };
    if handle.is_null() {
        drop(msg_box);
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.messages.push(msg_box.0);
}

/// Pushes a `BlazenTool` onto the request's `tools` vec. Consumes `tool`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`. `tool` must be
/// null OR a live `BlazenTool`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_tools_push(
    handle: *mut BlazenCompletionRequest,
    tool: *mut BlazenTool,
) {
    if tool.is_null() {
        return;
    }
    // SAFETY: per the contract, `tool` came from `Box::into_raw`.
    let tool_box = unsafe { Box::from_raw(tool) };
    if handle.is_null() {
        drop(tool_box);
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.tools.push(tool_box.0);
}

/// Sets `temperature` to `Some(value)`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_set_temperature(
    handle: *mut BlazenCompletionRequest,
    value: f64,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.temperature = Some(value);
}

/// Clears `temperature` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_clear_temperature(
    handle: *mut BlazenCompletionRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.temperature = None;
}

/// Sets `max_tokens` to `Some(value)`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_set_max_tokens(
    handle: *mut BlazenCompletionRequest,
    value: u32,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.max_tokens = Some(value);
}

/// Clears `max_tokens` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_clear_max_tokens(
    handle: *mut BlazenCompletionRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.max_tokens = None;
}

/// Sets `top_p` to `Some(value)`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_set_top_p(
    handle: *mut BlazenCompletionRequest,
    value: f64,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.top_p = Some(value);
}

/// Clears `top_p` back to `None`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_clear_top_p(
    handle: *mut BlazenCompletionRequest,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    r.0.top_p = None;
}

/// Sets the optional `model` field. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`. `value` must be
/// null OR point to a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_set_model(
    handle: *mut BlazenCompletionRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    r.0.model = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `response_format_json` field. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`. `value` must be
/// null OR point to a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_set_response_format_json(
    handle: *mut BlazenCompletionRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    r.0.response_format_json = unsafe { cstr_to_opt_string(value) };
}

/// Sets the optional `system` field. Null `value` clears it.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionRequest`. `value` must be
/// null OR point to a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_set_system(
    handle: *mut BlazenCompletionRequest,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionRequest`.
    let r = unsafe { &mut *handle };
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    r.0.system = unsafe { cstr_to_opt_string(value) };
}

/// Frees a `BlazenCompletionRequest` handle and all owned contents. No-op on
/// a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_request_free(handle: *mut BlazenCompletionRequest) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenCompletionResponse (output-only)
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::CompletionResponse`]. Produced
/// by `complete` / `complete_blocking` in Phase R3; no public constructor.
pub struct BlazenCompletionResponse(pub(crate) InnerCompletionResponse);

impl BlazenCompletionResponse {
    pub(crate) fn into_ptr(self) -> *mut BlazenCompletionResponse {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerCompletionResponse> for BlazenCompletionResponse {
    fn from(inner: InnerCompletionResponse) -> Self {
        Self(inner)
    }
}

/// Returns the `content` text as a caller-owned C string. Returns null on a
/// null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_content(
    handle: *const BlazenCompletionResponse,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionResponse`.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.content)
}

/// Returns the `finish_reason` field as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_finish_reason(
    handle: *const BlazenCompletionResponse,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionResponse`.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.finish_reason)
}

/// Returns the `model` identifier as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_model(
    handle: *const BlazenCompletionResponse,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionResponse`.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.model)
}

/// Returns the number of tool-call entries.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_tool_calls_count(
    handle: *const BlazenCompletionResponse,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionResponse`.
    let r = unsafe { &*handle };
    r.0.tool_calls.len()
}

/// Clones the `idx`-th tool-call entry into a fresh handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_tool_calls_get(
    handle: *const BlazenCompletionResponse,
    idx: usize,
) -> *mut BlazenToolCall {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionResponse`.
    let r = unsafe { &*handle };
    match r.0.tool_calls.get(idx) {
        Some(tc) => BlazenToolCall(tc.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Returns a fresh `BlazenTokenUsage` handle cloned from the response's usage
/// counters. Caller frees with `blazen_token_usage_free`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenCompletionResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_usage(
    handle: *const BlazenCompletionResponse,
) -> *mut BlazenTokenUsage {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenCompletionResponse`.
    let r = unsafe { &*handle };
    BlazenTokenUsage(r.0.usage.clone()).into_ptr()
}

/// Frees a `BlazenCompletionResponse` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_response_free(handle: *mut BlazenCompletionResponse) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// BlazenEmbeddingResponse (output-only)
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`blazen_uniffi::llm::EmbeddingResponse`]. Produced
/// by `embed` / `embed_blocking` in Phase R3; no public constructor.
pub struct BlazenEmbeddingResponse(pub(crate) InnerEmbeddingResponse);

impl BlazenEmbeddingResponse {
    pub(crate) fn into_ptr(self) -> *mut BlazenEmbeddingResponse {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerEmbeddingResponse> for BlazenEmbeddingResponse {
    fn from(inner: InnerEmbeddingResponse) -> Self {
        Self(inner)
    }
}

/// Returns the number of embedding vectors. Returns `0` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_embeddings_count(
    handle: *const BlazenEmbeddingResponse,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenEmbeddingResponse`.
    let r = unsafe { &*handle };
    r.0.embeddings.len()
}

/// Returns the dimensionality of the `vec_idx`-th embedding vector. Returns
/// `0` on a null handle or an out-of-range index.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_embedding_dim(
    handle: *const BlazenEmbeddingResponse,
    vec_idx: usize,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenEmbeddingResponse`.
    let r = unsafe { &*handle };
    r.0.embeddings.get(vec_idx).map_or(0, Vec::len)
}

/// Returns the `dim_idx`-th coordinate of the `vec_idx`-th embedding vector.
/// Returns `0.0` on a null handle or any out-of-range index — callers should
/// gate access with `_embeddings_count` and `_embedding_dim` first.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_embedding_get(
    handle: *const BlazenEmbeddingResponse,
    vec_idx: usize,
    dim_idx: usize,
) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenEmbeddingResponse`.
    let r = unsafe { &*handle };
    r.0.embeddings
        .get(vec_idx)
        .and_then(|v| v.get(dim_idx))
        .copied()
        .unwrap_or(0.0)
}

/// Bulk-copy the `vec_idx`-th embedding vector into the caller-supplied
/// buffer. Writes up to `min(vector_len, out_buf_len)` `f64`s starting at
/// `out_buf` and returns the actual number of `f64`s written.
///
/// Returns `0` if `handle` is null, `vec_idx` is out of range, or
/// `out_buf` is null. Designed for hot paths in embedding-heavy workloads
/// where allocating one C string per coordinate is unacceptable.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingResponse`. `out_buf` must
/// be null OR point to a writable buffer of at least `out_buf_len`
/// `f64`-aligned `sizeof(f64)`-spaced slots, valid for the duration of this
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_embedding_to_buffer(
    handle: *const BlazenEmbeddingResponse,
    vec_idx: usize,
    out_buf: *mut f64,
    out_buf_len: usize,
) -> usize {
    if handle.is_null() || out_buf.is_null() || out_buf_len == 0 {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenEmbeddingResponse`.
    let r = unsafe { &*handle };
    let Some(vec) = r.0.embeddings.get(vec_idx) else {
        return 0;
    };
    let n = vec.len().min(out_buf_len);
    // SAFETY: `out_buf` is non-null and the caller has guaranteed it is valid
    // for at least `out_buf_len` writable `f64` slots. `n` is bounded above
    // by `out_buf_len`, so the write stays in-bounds. Source and destination
    // are non-overlapping (the embedding vector lives behind `&*handle` on
    // the Rust heap; the destination is the caller-supplied buffer).
    unsafe {
        std::ptr::copy_nonoverlapping(vec.as_ptr(), out_buf, n);
    }
    n
}

/// Returns the embedding model identifier as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_model(
    handle: *const BlazenEmbeddingResponse,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenEmbeddingResponse`.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.model)
}

/// Returns a fresh `BlazenTokenUsage` handle cloned from the response.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenEmbeddingResponse`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_usage(
    handle: *const BlazenEmbeddingResponse,
) -> *mut BlazenTokenUsage {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenEmbeddingResponse`.
    let r = unsafe { &*handle };
    BlazenTokenUsage(r.0.usage.clone()).into_ptr()
}

/// Frees a `BlazenEmbeddingResponse` handle. No-op on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_response_free(handle: *mut BlazenEmbeddingResponse) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
