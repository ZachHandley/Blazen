//! Opaque error handle exposed across the C ABI.
//!
//! Every fallible cabi entry point returns a status code (0 = ok) and writes
//! a `*mut BlazenError` into a caller-supplied out-parameter on failure.
//! Callers free the handle with [`blazen_error_free`].
//!
//! The Rust-side variants live in `blazen_uniffi::errors::BlazenError`; this
//! module wraps them in an opaque struct and exposes flat C accessors so FFI
//! hosts can interrogate the error without needing to know the Rust enum
//! shape. Variant discrimination goes through [`blazen_error_kind`] which
//! returns one of the `BLAZEN_ERROR_KIND_*` constants below.
//!
//! Accessor functions that only apply to a subset of variants document their
//! sentinel for non-applicable variants:
//! - String accessors return `null` when the field is unset / wrong variant.
//! - `blazen_error_retry_after_ms` / `blazen_error_status` return `-1`.
//! - `blazen_error_elapsed_ms` returns `0`.

// Foundation utility: producers of `*mut BlazenError` land in Phase R3+ when
// the fallible wrappers are wired in. `into_ptr` / `from_ptr_take` are
// crate-private helpers those wrappers will reach for; until then the
// public extern functions are the only used surface, and `dead_code` would
// fire on the helpers.
#![allow(dead_code)]

use std::ffi::{CStr, c_char};

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::string::alloc_cstring;

/// Variant tag for the `Auth` error category.
pub const BLAZEN_ERROR_KIND_AUTH: u32 = 1;
/// Variant tag for the `RateLimit` error category.
pub const BLAZEN_ERROR_KIND_RATE_LIMIT: u32 = 2;
/// Variant tag for the `Timeout` error category.
pub const BLAZEN_ERROR_KIND_TIMEOUT: u32 = 3;
/// Variant tag for the `Validation` error category.
pub const BLAZEN_ERROR_KIND_VALIDATION: u32 = 4;
/// Variant tag for the `ContentPolicy` error category.
pub const BLAZEN_ERROR_KIND_CONTENT_POLICY: u32 = 5;
/// Variant tag for the `Unsupported` error category.
pub const BLAZEN_ERROR_KIND_UNSUPPORTED: u32 = 6;
/// Variant tag for the `Compute` error category.
pub const BLAZEN_ERROR_KIND_COMPUTE: u32 = 7;
/// Variant tag for the `Media` error category.
pub const BLAZEN_ERROR_KIND_MEDIA: u32 = 8;
/// Variant tag for the `Provider` error category.
pub const BLAZEN_ERROR_KIND_PROVIDER: u32 = 9;
/// Variant tag for the `Workflow` error category.
pub const BLAZEN_ERROR_KIND_WORKFLOW: u32 = 10;
/// Variant tag for the `Tool` error category.
pub const BLAZEN_ERROR_KIND_TOOL: u32 = 11;
/// Variant tag for the `Peer` error category.
pub const BLAZEN_ERROR_KIND_PEER: u32 = 12;
/// Variant tag for the `Persist` error category.
pub const BLAZEN_ERROR_KIND_PERSIST: u32 = 13;
/// Variant tag for the `Prompt` error category.
pub const BLAZEN_ERROR_KIND_PROMPT: u32 = 14;
/// Variant tag for the `Memory` error category.
pub const BLAZEN_ERROR_KIND_MEMORY: u32 = 15;
/// Variant tag for the `Cache` error category.
pub const BLAZEN_ERROR_KIND_CACHE: u32 = 16;
/// Variant tag for the `Cancelled` error category.
pub const BLAZEN_ERROR_KIND_CANCELLED: u32 = 17;
/// Variant tag for the `Internal` (catch-all) error category.
pub const BLAZEN_ERROR_KIND_INTERNAL: u32 = 18;

/// Opaque error handle owned by the caller. Produced by any fallible cabi
/// function via an out-parameter `*mut *mut BlazenError`. Released with
/// [`blazen_error_free`].
///
/// Deliberately not `#[repr(C)]` so cbindgen emits the C side as a
/// forward-declared `typedef struct BlazenError BlazenError;` opaque type —
/// FFI hosts never inspect the layout directly, they go through the
/// accessor functions below.
pub struct BlazenError {
    pub(crate) inner: InnerError,
}

impl BlazenError {
    /// Wraps an inner `blazen_uniffi` error into a fresh heap-allocated handle
    /// and returns its raw pointer. Used by future fallible wrappers to fill
    /// the `*mut *mut BlazenError` out-parameter on failure.
    pub(crate) fn into_ptr(self) -> *mut BlazenError {
        Box::into_raw(Box::new(self))
    }

    /// Reclaims ownership of a previously-leaked error pointer, returning it
    /// as a `Box` so it gets dropped at end of scope.
    ///
    /// # Safety
    ///
    /// `ptr` must be a non-null pointer previously produced by
    /// [`BlazenError::into_ptr`] (or equivalent `Box::into_raw` over a
    /// `BlazenError`). Calling this twice on the same pointer is a
    /// double-free.
    pub(crate) unsafe fn from_ptr_take(ptr: *mut BlazenError) -> Box<BlazenError> {
        debug_assert!(
            !ptr.is_null(),
            "BlazenError::from_ptr_take called with null"
        );
        // SAFETY: the caller has guaranteed `ptr` came from `Box::into_raw`
        // over a `BlazenError`, so reconstructing the `Box` is sound.
        unsafe { Box::from_raw(ptr) }
    }
}

impl From<InnerError> for BlazenError {
    fn from(inner: InnerError) -> Self {
        Self { inner }
    }
}

/// Returns the variant tag for `err` — one of the `BLAZEN_ERROR_KIND_*`
/// constants. Returns `0` if `err` is null (which is otherwise an invalid
/// state — successful calls never produce an error handle).
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` previously
/// produced by a `blazen_*` function that documents `*mut *mut BlazenError`
/// out-parameter semantics. The pointer must remain valid for the duration
/// of this call (i.e. not freed concurrently from another thread).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_kind(err: *const BlazenError) -> u32 {
    if err.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Auth { .. } => BLAZEN_ERROR_KIND_AUTH,
        InnerError::RateLimit { .. } => BLAZEN_ERROR_KIND_RATE_LIMIT,
        InnerError::Timeout { .. } => BLAZEN_ERROR_KIND_TIMEOUT,
        InnerError::Validation { .. } => BLAZEN_ERROR_KIND_VALIDATION,
        InnerError::ContentPolicy { .. } => BLAZEN_ERROR_KIND_CONTENT_POLICY,
        InnerError::Unsupported { .. } => BLAZEN_ERROR_KIND_UNSUPPORTED,
        InnerError::Compute { .. } => BLAZEN_ERROR_KIND_COMPUTE,
        InnerError::Media { .. } => BLAZEN_ERROR_KIND_MEDIA,
        InnerError::Provider { .. } => BLAZEN_ERROR_KIND_PROVIDER,
        InnerError::Workflow { .. } => BLAZEN_ERROR_KIND_WORKFLOW,
        InnerError::Tool { .. } => BLAZEN_ERROR_KIND_TOOL,
        InnerError::Peer { .. } => BLAZEN_ERROR_KIND_PEER,
        InnerError::Persist { .. } => BLAZEN_ERROR_KIND_PERSIST,
        InnerError::Prompt { .. } => BLAZEN_ERROR_KIND_PROMPT,
        InnerError::Memory { .. } => BLAZEN_ERROR_KIND_MEMORY,
        InnerError::Cache { .. } => BLAZEN_ERROR_KIND_CACHE,
        InnerError::Cancelled => BLAZEN_ERROR_KIND_CANCELLED,
        InnerError::Internal { .. } => BLAZEN_ERROR_KIND_INTERNAL,
    }
}

/// Returns the variant's primary message as a heap-allocated NUL-terminated
/// UTF-8 C string. The caller owns the returned pointer and must free it
/// with `blazen_string_free`. Returns null if `err` is null.
///
/// For the `Cancelled` variant — which has no `message` field — the returned
/// string is its `Display` rendering (`"cancelled"`).
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface. The returned string buffer is independent of `err`'s
/// lifetime; freeing `err` does not invalidate the message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_message(err: *const BlazenError) -> *mut c_char {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    let msg = match &err.inner {
        InnerError::Auth { message }
        | InnerError::RateLimit { message, .. }
        | InnerError::Timeout { message, .. }
        | InnerError::Validation { message }
        | InnerError::ContentPolicy { message }
        | InnerError::Unsupported { message }
        | InnerError::Compute { message }
        | InnerError::Media { message }
        | InnerError::Provider { message, .. }
        | InnerError::Workflow { message }
        | InnerError::Tool { message }
        | InnerError::Peer { message, .. }
        | InnerError::Persist { message }
        | InnerError::Prompt { message, .. }
        | InnerError::Memory { message, .. }
        | InnerError::Cache { message, .. }
        | InnerError::Internal { message } => message.as_str(),
        InnerError::Cancelled => "cancelled",
    };
    alloc_cstring(msg)
}

/// Returns the `retry_after_ms` hint in milliseconds for the `RateLimit` and
/// `Provider` variants. Returns `-1` if `err` is null, the variant doesn't
/// carry this field, or the field is unset on a variant that does.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_retry_after_ms(err: *const BlazenError) -> i64 {
    if err.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    let value = match &err.inner {
        InnerError::RateLimit { retry_after_ms, .. }
        | InnerError::Provider { retry_after_ms, .. } => *retry_after_ms,
        _ => None,
    };
    value.and_then(|v| i64::try_from(v).ok()).unwrap_or(-1)
}

/// Returns `elapsed_ms` for the `Timeout` variant. Returns `0` if `err` is
/// null or the variant doesn't carry this field.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_elapsed_ms(err: *const BlazenError) -> u64 {
    if err.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Timeout { elapsed_ms, .. } => *elapsed_ms,
        _ => 0,
    }
}

/// Returns the HTTP status code carried by the `Provider` variant. Returns
/// `-1` if `err` is null, the variant doesn't carry a status, or the status
/// field is unset.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_status(err: *const BlazenError) -> i32 {
    if err.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Provider {
            status: Some(status),
            ..
        } => i32::try_from(*status).unwrap_or(-1),
        _ => -1,
    }
}

/// Returns the `provider` slug (e.g. `"openai"`, `"anthropic"`) for the
/// `Provider` variant as a heap-allocated C string. Returns null if `err`
/// is null, the variant doesn't carry a provider, or the field is unset.
/// Caller frees with `blazen_string_free`.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_provider(err: *const BlazenError) -> *mut c_char {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Provider {
            provider: Some(provider),
            ..
        } => alloc_cstring(provider),
        _ => std::ptr::null_mut(),
    }
}

/// Returns the `endpoint` URL for the `Provider` variant as a heap-allocated
/// C string. Returns null if `err` is null, the variant doesn't carry an
/// endpoint, or the field is unset. Caller frees with `blazen_string_free`.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_endpoint(err: *const BlazenError) -> *mut c_char {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Provider {
            endpoint: Some(endpoint),
            ..
        } => alloc_cstring(endpoint),
        _ => std::ptr::null_mut(),
    }
}

/// Returns the `request_id` for the `Provider` variant as a heap-allocated
/// C string. Returns null if `err` is null, the variant doesn't carry a
/// request id, or the field is unset. Caller frees with `blazen_string_free`.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_request_id(err: *const BlazenError) -> *mut c_char {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Provider {
            request_id: Some(request_id),
            ..
        } => alloc_cstring(request_id),
        _ => std::ptr::null_mut(),
    }
}

/// Returns the `detail` payload for the `Provider` variant as a heap-allocated
/// C string. Returns null if `err` is null, the variant doesn't carry a
/// detail, or the field is unset. Caller frees with `blazen_string_free`.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_detail(err: *const BlazenError) -> *mut c_char {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Provider {
            detail: Some(detail),
            ..
        } => alloc_cstring(detail),
        _ => std::ptr::null_mut(),
    }
}

/// Returns the sub-kind discriminator (the inner sub-enum's `Display`
/// rendering) for the variants that carry one: `Provider`, `Peer`,
/// `Prompt`, `Memory`, and `Cache`. Returns null for all other variants
/// (and for null `err`). Caller frees with `blazen_string_free`.
///
/// The strings come straight from the `kind` field stored on the inner
/// `BlazenError` variant — they're stable identifiers like `"Http"`,
/// `"Transport"`, `"MissingVariable"`, `"NoEmbedder"`, `"Download"`,
/// matching the documented values in `blazen_uniffi::errors::BlazenError`.
///
/// # Safety
///
/// `err` must be null OR a valid pointer to a `BlazenError` produced by the
/// cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_subkind(err: *const BlazenError) -> *mut c_char {
    if err.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `err` is a live `BlazenError` pointer.
    let err = unsafe { &*err };
    match &err.inner {
        InnerError::Provider { kind, .. }
        | InnerError::Peer { kind, .. }
        | InnerError::Prompt { kind, .. }
        | InnerError::Memory { kind, .. }
        | InnerError::Cache { kind, .. } => alloc_cstring(kind),
        _ => std::ptr::null_mut(),
    }
}

/// Frees an error handle previously produced by the cabi surface. No-op on
/// a null pointer.
///
/// # Safety
///
/// `err` must be null OR a pointer previously produced by a `blazen_*`
/// function that documents `*mut *mut BlazenError` out-parameter
/// semantics. Calling this twice on the same non-null pointer is a
/// double-free; reading any of the accessors on `err` after this call is
/// a use-after-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_free(err: *mut BlazenError) {
    if err.is_null() {
        return;
    }
    // SAFETY: per the contract above, `err` was produced by
    // `Box::into_raw` over a `BlazenError`, so reconstructing the `Box`
    // here is sound and `drop` releases the original allocation.
    drop(unsafe { Box::from_raw(err) });
}

/// Constructs a fresh `BlazenError` handle from a JSON object describing the
/// variant and its message. Used by FFI hosts (notably the Ruby binding) to
/// materialise a typed error from a foreign-language exception so it can be
/// handed back through a fallible cabi callback.
///
/// The JSON must be an object of shape `{ "kind": "<Variant>", "message": "..." }`
/// where `<Variant>` is one of (case-sensitive, mirroring the
/// `blazen_uniffi::errors::BlazenError` variants):
/// `Auth`, `RateLimit`, `Timeout`, `Validation`, `ContentPolicy`,
/// `Unsupported`, `Compute`, `Media`, `Provider`, `Workflow`, `Tool`, `Peer`,
/// `Persist`, `Prompt`, `Memory`, `Cache`, `Cancelled`, `Internal`.
///
/// Variants that carry extra structured fields (`Provider`, `Peer`, `Prompt`,
/// `Memory`, `Cache`, `RateLimit`, `Timeout`) accept the same field names as
/// their Rust counterparts; missing optional fields default sensibly
/// (`Provider.kind` defaults to `"Other"`, all optional fields default to
/// `None`/`0`).
///
/// On any failure — null input, non-UTF-8 input, missing `kind`, unknown
/// `kind`, or malformed JSON — falls back to `BlazenError::Internal` with a
/// best-effort message and returns a non-null handle. This function never
/// returns null for a non-null input pointer.
///
/// # Ownership
///
/// The returned handle is owned by the caller and must be released with
/// [`blazen_error_free`]. Returns null only if `json` is null.
///
/// # Safety
///
/// `json` must be null OR point to a NUL-terminated byte buffer (any
/// encoding — non-UTF-8 input is handled gracefully) that remains valid for
/// the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_error_from_json(json: *const c_char) -> *mut BlazenError {
    if json.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: per the contract, `json` is a live NUL-terminated buffer.
    let cstr = unsafe { CStr::from_ptr(json) };
    let Ok(s) = cstr.to_str() else {
        return BlazenError::from(InnerError::Internal {
            message: "blazen_error_from_json: input is not valid UTF-8".to_string(),
        })
        .into_ptr();
    };
    let inner = parse_error_json(s);
    BlazenError::from(inner).into_ptr()
}

/// Parses a `{kind, message, ...}` JSON object into the matching
/// [`InnerError`] variant. Any failure (malformed JSON, missing/unknown
/// `kind`, missing required field on a structured variant) collapses to
/// `InnerError::Internal { message }` where `message` is a best-effort
/// description.
fn parse_error_json(s: &str) -> InnerError {
    let value: serde_json::Value = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(e) => {
            return InnerError::Internal {
                message: format!("blazen_error_from_json: malformed JSON: {e}"),
            };
        }
    };
    let Some(obj) = value.as_object() else {
        return InnerError::Internal {
            message: format!("blazen_error_from_json: expected JSON object, got {value}"),
        };
    };
    let kind = obj.get("kind").and_then(serde_json::Value::as_str);
    let message = obj
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();

    let get_str = |key: &str| -> Option<String> {
        obj.get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    let get_u32 = |key: &str| -> Option<u32> {
        obj.get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
    };
    let get_u64 = |key: &str| -> Option<u64> { obj.get(key).and_then(serde_json::Value::as_u64) };

    match kind {
        Some("Auth") => InnerError::Auth { message },
        Some("RateLimit") => InnerError::RateLimit {
            message,
            retry_after_ms: get_u64("retry_after_ms"),
        },
        Some("Timeout") => InnerError::Timeout {
            message,
            elapsed_ms: get_u64("elapsed_ms").unwrap_or(0),
        },
        Some("Validation") => InnerError::Validation { message },
        Some("ContentPolicy") => InnerError::ContentPolicy { message },
        Some("Unsupported") => InnerError::Unsupported { message },
        Some("Compute") => InnerError::Compute { message },
        Some("Media") => InnerError::Media { message },
        Some("Provider") => InnerError::Provider {
            kind: get_str("subkind")
                .or_else(|| get_str("provider_kind"))
                .unwrap_or_else(|| "Other".to_string()),
            message,
            provider: get_str("provider"),
            status: get_u32("status"),
            endpoint: get_str("endpoint"),
            request_id: get_str("request_id"),
            detail: get_str("detail"),
            retry_after_ms: get_u64("retry_after_ms"),
        },
        Some("Workflow") => InnerError::Workflow { message },
        Some("Tool") => InnerError::Tool { message },
        Some("Peer") => InnerError::Peer {
            kind: get_str("subkind")
                .or_else(|| get_str("peer_kind"))
                .unwrap_or_else(|| "Transport".to_string()),
            message,
        },
        Some("Persist") => InnerError::Persist { message },
        Some("Prompt") => InnerError::Prompt {
            kind: get_str("subkind")
                .or_else(|| get_str("prompt_kind"))
                .unwrap_or_else(|| "Validation".to_string()),
            message,
        },
        Some("Memory") => InnerError::Memory {
            kind: get_str("subkind")
                .or_else(|| get_str("memory_kind"))
                .unwrap_or_else(|| "Backend".to_string()),
            message,
        },
        Some("Cache") => InnerError::Cache {
            kind: get_str("subkind")
                .or_else(|| get_str("cache_kind"))
                .unwrap_or_else(|| "Io".to_string()),
            message,
        },
        Some("Cancelled") => InnerError::Cancelled,
        Some("Internal") => InnerError::Internal { message },
        Some(other) => InnerError::Internal {
            message: format!(
                "blazen_error_from_json: unknown error kind {other:?}; original message: {message}"
            ),
        },
        None => InnerError::Internal {
            message: format!(
                "blazen_error_from_json: missing 'kind' field; original message: {message}"
            ),
        },
    }
}
