//! C ABI wrapper for [`blazen_llm::providers::BaseProvider`].
//!
//! `BaseProvider` wraps an `Arc<dyn CompletionModel>` and applies a
//! [`CompletionProviderDefaults`](blazen_llm::providers::CompletionProviderDefaults)
//! to every completion call. For V1 we cannot construct a `BaseProvider` from
//! C alone — the `Arc<dyn CompletionModel>` it needs has no FFI representation
//! today. The wrapper therefore exposes only:
//!
//! - Builder-style setters (system prompt / tools / response format / defaults).
//! - Read-only getters for the configured defaults and inner model identity.
//! - `_free`.
//!
//! Constructors will land in Phase B, where the cabi gains `CustomProvider`
//! factories that return `BlazenBaseProvider` handles. Phase C wires hooks
//! through a vtable.
//!
//! ## Builder mutation pattern
//!
//! The inner `BaseProvider::with_*` methods consume `self` and return `Self`.
//! Since the C side owns the handle as a `*mut BlazenBaseProvider`, we mutate
//! in place by using `std::ptr::read` to take ownership of the inner
//! `BaseProvider`, running the builder, and writing the result back with
//! `std::ptr::write`. The pointer remains valid and continues to identify the
//! same handle slot — the caller does not need to re-bind their pointer.

#![allow(dead_code)]

use std::ffi::c_char;

use blazen_llm::providers::base::BaseProvider;
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::provider_defaults::BlazenCompletionProviderDefaults;
use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Opaque wrapper around [`BaseProvider`].
#[repr(C)]
pub struct BlazenBaseProvider(pub(crate) BaseProvider);

impl BlazenBaseProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenBaseProvider {
        Box::into_raw(Box::new(self))
    }
}

impl From<BaseProvider> for BlazenBaseProvider {
    fn from(inner: BaseProvider) -> Self {
        Self(inner)
    }
}

// ---------------------------------------------------------------------------
// Internal builder helper
// ---------------------------------------------------------------------------

/// Run a builder method (`fn(BaseProvider) -> BaseProvider`) against the
/// inner provider behind the supplied handle, mutating the slot in place.
///
/// # Safety
///
/// `handle` must be a non-null pointer to a live `BlazenBaseProvider`.
unsafe fn with_inner<F: FnOnce(BaseProvider) -> BaseProvider>(
    handle: *mut BlazenBaseProvider,
    f: F,
) {
    // SAFETY: caller has guaranteed `handle` is non-null and live. We take
    // the inner `BaseProvider` by value (`ptr::read`), run the builder, then
    // write the new value back into the same slot. Between the read and the
    // write there are no other live references to the slot, so this is a
    // straight-line move that preserves the handle pointer's identity.
    unsafe {
        let bp = std::ptr::read(&raw const (*handle).0);
        let new_bp = f(bp);
        std::ptr::write(&raw mut (*handle).0, new_bp);
    }
}

// ---------------------------------------------------------------------------
// Builder setters (in-place mutation; pointer remains valid)
// ---------------------------------------------------------------------------

/// Sets the `system_prompt` field on the inner defaults. Null `s` is a no-op
/// (use [`blazen_completion_provider_defaults_set_system_prompt`] via
/// `_set_defaults` if you need to clear).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`. `s` must be null OR
/// a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_with_system_prompt(
    handle: *mut BlazenBaseProvider,
    s: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the borrow contract on `s`.
    let Some(s) = (unsafe { cstr_to_str(s) }) else {
        return;
    };
    let owned = s.to_owned();
    // SAFETY: caller guarantees live handle.
    unsafe { with_inner(handle, |bp| bp.with_system_prompt(owned)) }
}

/// Replaces the `tools` list by parsing `json` as a JSON array of
/// [`ToolDefinition`] values. Null `json` is a no-op; invalid JSON clears the
/// list.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`. `json` must be null
/// OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_with_tools_json(
    handle: *mut BlazenBaseProvider,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the borrow contract on `json`.
    let Some(s) = (unsafe { cstr_to_str(json) }) else {
        return;
    };
    let tools = serde_json::from_str::<Vec<ToolDefinition>>(s).unwrap_or_default();
    // SAFETY: caller guarantees live handle.
    unsafe { with_inner(handle, |bp| bp.with_tools(tools)) }
}

/// Replaces the `response_format` field by parsing `json`. Null `json` is a
/// no-op; invalid JSON stores `Value::Null`.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`. `json` must be null
/// OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_with_response_format_json(
    handle: *mut BlazenBaseProvider,
    json: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the borrow contract on `json`.
    let Some(s) = (unsafe { cstr_to_str(json) }) else {
        return;
    };
    let value = serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::Value::Null);
    // SAFETY: caller guarantees live handle.
    unsafe { with_inner(handle, |bp| bp.with_response_format(value)) }
}

/// Replaces the entire `CompletionProviderDefaults` on the provider with a
/// clone of the supplied handle. Null `d` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`. `d` must be null OR
/// a live `BlazenCompletionProviderDefaults`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_with_defaults(
    handle: *mut BlazenBaseProvider,
    d: *const BlazenCompletionProviderDefaults,
) {
    if handle.is_null() || d.is_null() {
        return;
    }
    // SAFETY: caller guarantees live `d`.
    let defaults_clone = unsafe { &*d }.0.clone();
    // SAFETY: caller guarantees live handle.
    unsafe { with_inner(handle, |bp| bp.set_defaults(defaults_clone)) }
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------

/// Returns a clone of the configured `CompletionProviderDefaults`. Caller owns
/// the returned handle and must free with
/// [`crate::provider_defaults::blazen_completion_provider_defaults_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_defaults(
    handle: *const BlazenBaseProvider,
) -> *mut BlazenCompletionProviderDefaults {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    BlazenCompletionProviderDefaults::from(r.0.defaults().clone()).into_ptr()
}

/// Returns the inner model's `model_id` as a caller-owned C string. Free with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_model_id(
    handle: *const BlazenBaseProvider,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(r.0.model_id())
}

/// Returns the inner model's `provider_id` as a caller-owned C string. The
/// `CompletionModel` trait surfaces this through `model_id()` plus the
/// provider's own identification; for V1 we return the same string as
/// [`blazen_base_provider_model_id`]. Free with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_provider_id(
    handle: *const BlazenBaseProvider,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(r.0.model_id())
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

/// Frees a `BlazenBaseProvider`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by a Phase B `CustomProvider`
/// factory that returns `BlazenBaseProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_free(handle: *mut BlazenBaseProvider) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// Typed extract
// ---------------------------------------------------------------------------

/// Spawns an async `extract(schema, messages)` against the wrapped provider.
///
/// `schema_json` must be a JSON Schema string (the same shape a typed
/// language binding would build via Pydantic's `model_json_schema()` or
/// zod's `zodToJsonSchema`). `messages_json` must be a JSON-encoded
/// `Vec<ChatMessage>`.
///
/// The implementation:
///
/// 1. Parses both JSON inputs.
/// 2. Builds a [`CompletionRequest`] with `response_format = schema`.
/// 3. Awaits the provider's `complete(...)`.
/// 4. Returns the response's `content` string (the model's JSON output) as
///    the future's typed result.
///
/// Pop the result with [`blazen_future_take_extract_result`]. The Ruby (or
/// any host-language) caller is then responsible for `JSON.parse`-ing the
/// content and validating against the schema.
///
/// Returns null if any argument is null OR if either JSON blob fails to
/// parse (`schema` must be valid JSON, `messages` must decode into
/// `Vec<ChatMessage>`).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenBaseProvider`. `schema_json` and
/// `messages_json` must each be a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_base_provider_extract(
    handle: *const BlazenBaseProvider,
    schema_json: *const c_char,
    messages_json: *const c_char,
) -> *mut BlazenFuture {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(schema_str) = (unsafe { cstr_to_str(schema_json) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same.
    let Some(messages_str) = (unsafe { cstr_to_str(messages_json) }) else {
        return std::ptr::null_mut();
    };
    let Ok(schema) = serde_json::from_str::<serde_json::Value>(schema_str) else {
        return std::ptr::null_mut();
    };
    let Ok(messages) = serde_json::from_str::<Vec<ChatMessage>>(messages_str) else {
        return std::ptr::null_mut();
    };

    // Clone the BaseProvider out of the handle so the spawned task owns its
    // own copy; the original handle stays usable afterwards.
    // SAFETY: caller guarantees live handle.
    let provider = unsafe { &*handle }.0.clone();

    let request = CompletionRequest::new(messages).with_response_format(schema);

    BlazenFuture::spawn::<String, _>(async move {
        let resp = provider.complete(request).await?;
        // Surface `content` to the host. An empty/None content path is
        // treated as Ok("") rather than an error — the host can decide
        // whether an empty string is a validation failure.
        Ok(resp.content.unwrap_or_default())
    })
}

/// Pops the [`String`] result of [`blazen_base_provider_extract`] out of
/// `fut`. On success returns `0` and writes a caller-owned `*mut c_char`
/// (NUL-terminated UTF-8, JSON-encoded content) into `out`; on failure
/// returns `-1` and writes a caller-owned `*mut BlazenError` into `err`.
///
/// Either out-pointer may be null to discard. The C string is released via
/// [`crate::string::blazen_string_free`]; the error via
/// [`crate::error::blazen_error_free`].
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_base_provider_extract`], not yet freed, and not concurrently
/// freed from another thread. `out` / `err` must be null OR writable
/// pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_extract_result(
    fut: *mut BlazenFuture,
    out: *mut *mut c_char,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<String>(fut) } {
        Ok(s) => {
            if !out.is_null() {
                // SAFETY: caller has guaranteed `out` is writable.
                unsafe {
                    *out = alloc_cstring(&s);
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller has guaranteed `err` is writable.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}
