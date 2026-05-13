//! Streaming-related record marshalling. Wraps [`blazen_uniffi::streaming::StreamChunk`]
//! as an opaque C handle.
//!
//! # Ownership conventions
//!
//! `blazen_stream_chunk_new` returns a caller-owned `*mut BlazenStreamChunk`,
//! released with [`blazen_stream_chunk_free`]. Tool-call entries pushed via
//! [`blazen_stream_chunk_tool_calls_push`] are consumed (the source
//! `BlazenToolCall*` must not be freed by the caller afterwards). `_get`
//! accessors clone the indexed item into a fresh caller-owned handle.
//!
//! Strings returned by getters (`*mut c_char`) are released with
//! [`crate::string::blazen_string_free`].
//!
//! `StreamChunk` crosses the boundary in *both* directions: Rust → C when
//! Phase R5's `CompletionStreamSink` trampoline delivers chunks to a foreign
//! sink, and C → Rust if the host ever needs to synthesise chunks for tests
//! or replay. The push/getter symmetry lives here for both directions.

// Foundation utility consumed by R3+ wrappers; the public extern fns stay
// linker-resident regardless.
#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::streaming::StreamChunk as InnerStreamChunk;

use crate::llm_records::BlazenToolCall;
use crate::string::{alloc_cstring, cstr_to_str};

/// Opaque wrapper around [`blazen_uniffi::streaming::StreamChunk`].
pub struct BlazenStreamChunk(pub(crate) InnerStreamChunk);

impl BlazenStreamChunk {
    pub(crate) fn into_ptr(self) -> *mut BlazenStreamChunk {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerStreamChunk> for BlazenStreamChunk {
    fn from(inner: InnerStreamChunk) -> Self {
        Self(inner)
    }
}

/// Constructs a new `StreamChunk` with the given `content_delta` and
/// `is_final` flag. `tool_calls` is initialised empty.
///
/// Returns null if `content_delta` is null or non-UTF-8.
///
/// # Safety
///
/// `content_delta` must be null OR point to a NUL-terminated UTF-8 buffer
/// valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_new(
    content_delta: *const c_char,
    is_final: bool,
) -> *mut BlazenStreamChunk {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let content_delta = match unsafe { cstr_to_str(content_delta) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    BlazenStreamChunk(InnerStreamChunk {
        content_delta,
        tool_calls: Vec::new(),
        is_final,
    })
    .into_ptr()
}

/// Pushes a `BlazenToolCall` onto the chunk's `tool_calls` snapshot. Consumes
/// `tool_call` — the caller must NOT free it afterwards.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenStreamChunk`. `tool_call` must be
/// null OR a live `BlazenToolCall` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_tool_calls_push(
    handle: *mut BlazenStreamChunk,
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
    // SAFETY: caller has guaranteed `handle` is a live `BlazenStreamChunk`.
    let c = unsafe { &mut *handle };
    c.0.tool_calls.push(tc_box.0);
}

/// Returns the `content_delta` text as a caller-owned C string. Returns null
/// on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenStreamChunk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_content_delta(
    handle: *const BlazenStreamChunk,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenStreamChunk`.
    let c = unsafe { &*handle };
    alloc_cstring(&c.0.content_delta)
}

/// Returns the chunk's `is_final` flag. Returns `false` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenStreamChunk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_is_final(handle: *const BlazenStreamChunk) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenStreamChunk`.
    let c = unsafe { &*handle };
    c.0.is_final
}

/// Returns the number of tool-call entries on this chunk. Returns `0` on a
/// null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenStreamChunk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_tool_calls_count(
    handle: *const BlazenStreamChunk,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenStreamChunk`.
    let c = unsafe { &*handle };
    c.0.tool_calls.len()
}

/// Clones the `idx`-th tool-call entry into a fresh caller-owned handle.
/// Returns null on a null handle or out-of-range index.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenStreamChunk`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_tool_calls_get(
    handle: *const BlazenStreamChunk,
    idx: usize,
) -> *mut BlazenToolCall {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live `BlazenStreamChunk`.
    let c = unsafe { &*handle };
    match c.0.tool_calls.get(idx) {
        Some(tc) => BlazenToolCall(tc.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenStreamChunk` handle and all owned tool-call entries. No-op
/// on a null pointer.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// surface. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_stream_chunk_free(handle: *mut BlazenStreamChunk) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
