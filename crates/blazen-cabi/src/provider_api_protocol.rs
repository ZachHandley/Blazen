//! C ABI wrappers for [`blazen_llm::providers::ApiProtocol`] and its supporting
//! [`OpenAiCompatConfig`] payload. These let Ruby (and any future cbindgen
//! host) describe the wire-protocol selector that drives `CustomProvider` in
//! later phases.
//!
//! ## Type map
//!
//! - [`BlazenApiProtocol`] — opaque handle around [`ApiProtocol`].
//! - [`BlazenOpenAiCompatConfig`] — opaque handle around [`OpenAiCompatConfig`].
//!
//! ## Ownership conventions
//!
//! Mirrors the rest of the cabi surface:
//!
//! - Every `_new` returns a heap-allocated handle the caller owns; free with
//!   the matching `_free`.
//! - String inputs (`*const c_char`) are borrowed for the duration of the
//!   call; the wrapper copies into owned `String`s before storing.
//! - Optional strings map null → `None`.
//! - String returns (`*mut c_char`) are caller-owned; release via
//!   [`crate::string::blazen_string_free`].
//! - Auth method is encoded as a small integer enum across the boundary
//!   (`0 = Bearer`, `1 = ApiKeyHeader`, `2 = AzureApiKey`, `3 = KeyPrefix`)
//!   with the optional header name argument used only for variant `1`.

#![allow(dead_code)]

use std::ffi::c_char;

use blazen_llm::providers::custom::ApiProtocol;
use blazen_llm::providers::openai_compat::{AuthMethod, OpenAiCompatConfig};

use crate::string::{alloc_cstring, cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Auth method encoding
// ---------------------------------------------------------------------------

/// Integer encoding of [`AuthMethod`] across the C boundary.
///
/// - `0` → [`AuthMethod::Bearer`]
/// - `1` → [`AuthMethod::ApiKeyHeader(header_name)`]
/// - `2` → [`AuthMethod::AzureApiKey`]
/// - `3` → [`AuthMethod::KeyPrefix`]
///
/// Any other value falls back to `Bearer`.
#[inline]
fn auth_method_from_code(code: u32, header_name: Option<String>) -> AuthMethod {
    match code {
        1 => AuthMethod::ApiKeyHeader(header_name.unwrap_or_default()),
        2 => AuthMethod::AzureApiKey,
        3 => AuthMethod::KeyPrefix,
        _ => AuthMethod::Bearer,
    }
}

/// Inverse of [`auth_method_from_code`]. Returns the integer code only; the
/// optional header-name payload is exposed through a separate getter.
#[inline]
fn auth_method_to_code(m: &AuthMethod) -> u32 {
    match m {
        AuthMethod::Bearer => 0,
        AuthMethod::ApiKeyHeader(_) => 1,
        AuthMethod::AzureApiKey => 2,
        AuthMethod::KeyPrefix => 3,
    }
}

// ===========================================================================
// BlazenOpenAiCompatConfig
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::providers::openai_compat::OpenAiCompatConfig`].
#[repr(C)]
pub struct BlazenOpenAiCompatConfig(pub(crate) OpenAiCompatConfig);

impl BlazenOpenAiCompatConfig {
    pub(crate) fn into_ptr(self) -> *mut BlazenOpenAiCompatConfig {
        Box::into_raw(Box::new(self))
    }
}

impl From<OpenAiCompatConfig> for BlazenOpenAiCompatConfig {
    fn from(inner: OpenAiCompatConfig) -> Self {
        Self(inner)
    }
}

/// Constructs a new `OpenAiCompatConfig` with the four required string fields.
/// `auth_code` selects the [`AuthMethod`] variant (see the module docs); for
/// `auth_code == 1` (`ApiKeyHeader`) the supplied `auth_header_name` becomes
/// the header. `supports_model_listing` defaults to the upstream value; pass
/// `true` to opt in.
///
/// Returns null if any of the four required strings is null or non-UTF-8.
///
/// # Safety
///
/// `provider_name`, `base_url`, `api_key`, and `default_model` must each be
/// a valid NUL-terminated UTF-8 buffer for the duration of the call.
/// `auth_header_name` must be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_new(
    provider_name: *const c_char,
    base_url: *const c_char,
    api_key: *const c_char,
    default_model: *const c_char,
    auth_code: u32,
    auth_header_name: *const c_char,
    supports_model_listing: bool,
) -> *mut BlazenOpenAiCompatConfig {
    // SAFETY: caller upholds the NUL-terminated UTF-8 contract.
    let Some(provider_name) = (unsafe { cstr_to_str(provider_name) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same.
    let Some(base_url) = (unsafe { cstr_to_str(base_url) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same.
    let Some(default_model) = (unsafe { cstr_to_str(default_model) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the optional NUL-terminated UTF-8 contract.
    let auth_header_name = unsafe { cstr_to_opt_string(auth_header_name) };
    let cfg = OpenAiCompatConfig {
        provider_name: provider_name.to_owned(),
        base_url: base_url.to_owned(),
        api_key: api_key.to_owned(),
        default_model: default_model.to_owned(),
        auth_method: auth_method_from_code(auth_code, auth_header_name),
        extra_headers: Vec::new(),
        query_params: Vec::new(),
        supports_model_listing,
    };
    BlazenOpenAiCompatConfig(cfg).into_ptr()
}

/// Appends an `extra_headers` entry. Null `name` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`. `name` and
/// `value` must each be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_push_extra_header(
    handle: *mut BlazenOpenAiCompatConfig,
    name: *const c_char,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the borrow contract.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        return;
    };
    // SAFETY: same.
    let value = unsafe { cstr_to_opt_string(value) }.unwrap_or_default();
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.extra_headers.push((name.to_owned(), value));
}

/// Appends a `query_params` entry. Null `name` is a no-op.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`. `name` and
/// `value` must each be null OR a valid NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_push_query_param(
    handle: *mut BlazenOpenAiCompatConfig,
    name: *const c_char,
    value: *const c_char,
) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the borrow contract.
    let Some(name) = (unsafe { cstr_to_str(name) }) else {
        return;
    };
    // SAFETY: same.
    let value = unsafe { cstr_to_opt_string(value) }.unwrap_or_default();
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &mut *handle };
    r.0.query_params.push((name.to_owned(), value));
}

/// Returns `provider_name` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_provider_name(
    handle: *const BlazenOpenAiCompatConfig,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.provider_name)
}

/// Returns `base_url` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_base_url(
    handle: *const BlazenOpenAiCompatConfig,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.base_url)
}

/// Returns `api_key` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_api_key(
    handle: *const BlazenOpenAiCompatConfig,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.api_key)
}

/// Returns `default_model` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_default_model(
    handle: *const BlazenOpenAiCompatConfig,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    alloc_cstring(&r.0.default_model)
}

/// Returns the [`AuthMethod`] code (see module docs).
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`. Returns `0`
/// (Bearer) on a null handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_auth_code(
    handle: *const BlazenOpenAiCompatConfig,
) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    auth_method_to_code(&r.0.auth_method)
}

/// Returns the header name for the `ApiKeyHeader` variant. Returns null for
/// any other variant or a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_auth_header_name(
    handle: *const BlazenOpenAiCompatConfig,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0.auth_method {
        AuthMethod::ApiKeyHeader(s) => alloc_cstring(s),
        _ => std::ptr::null_mut(),
    }
}

/// Returns `supports_model_listing`. Returns `false` on a null handle.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_supports_model_listing(
    handle: *const BlazenOpenAiCompatConfig,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.supports_model_listing
}

/// Returns the number of `extra_headers` entries.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_extra_headers_len(
    handle: *const BlazenOpenAiCompatConfig,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.extra_headers.len()
}

/// Returns the name at `index` in `extra_headers` as a caller-owned C string.
/// Out-of-bounds indices return null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_extra_header_name(
    handle: *const BlazenOpenAiCompatConfig,
    index: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.extra_headers
        .get(index)
        .map_or(std::ptr::null_mut(), |(k, _)| alloc_cstring(k))
}

/// Returns the value at `index` in `extra_headers` as a caller-owned C string.
/// Out-of-bounds indices return null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_extra_header_value(
    handle: *const BlazenOpenAiCompatConfig,
    index: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.extra_headers
        .get(index)
        .map_or(std::ptr::null_mut(), |(_, v)| alloc_cstring(v))
}

/// Returns the number of `query_params` entries.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_query_params_len(
    handle: *const BlazenOpenAiCompatConfig,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.query_params.len()
}

/// Returns the name at `index` in `query_params` as a caller-owned C string.
/// Out-of-bounds indices return null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_query_param_name(
    handle: *const BlazenOpenAiCompatConfig,
    index: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.query_params
        .get(index)
        .map_or(std::ptr::null_mut(), |(k, _)| alloc_cstring(k))
}

/// Returns the value at `index` in `query_params` as a caller-owned C string.
/// Out-of-bounds indices return null.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_query_param_value(
    handle: *const BlazenOpenAiCompatConfig,
    index: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    r.0.query_params
        .get(index)
        .map_or(std::ptr::null_mut(), |(_, v)| alloc_cstring(v))
}

/// Frees a `BlazenOpenAiCompatConfig`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by
/// [`blazen_openai_compat_config_new`] or [`blazen_api_protocol_config`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_config_free(handle: *mut BlazenOpenAiCompatConfig) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}

// ===========================================================================
// BlazenApiProtocol
// ===========================================================================

/// Opaque wrapper around [`blazen_llm::providers::ApiProtocol`].
#[repr(C)]
pub struct BlazenApiProtocol(pub(crate) ApiProtocol);

impl BlazenApiProtocol {
    pub(crate) fn into_ptr(self) -> *mut BlazenApiProtocol {
        Box::into_raw(Box::new(self))
    }
}

impl From<ApiProtocol> for BlazenApiProtocol {
    fn from(inner: ApiProtocol) -> Self {
        Self(inner)
    }
}

/// Constructs an `ApiProtocol::OpenAi(config)` variant. Clones the inner
/// `OpenAiCompatConfig` from the passed-in handle (caller retains ownership of
/// the original config handle). Returns null on a null `config` argument.
///
/// # Safety
///
/// `config` must be null OR a live `BlazenOpenAiCompatConfig`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_api_protocol_openai(
    config: *const BlazenOpenAiCompatConfig,
) -> *mut BlazenApiProtocol {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let cfg = unsafe { &*config };
    BlazenApiProtocol(ApiProtocol::OpenAi(cfg.0.clone())).into_ptr()
}

/// Constructs an `ApiProtocol::Custom` variant.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_api_protocol_custom() -> *mut BlazenApiProtocol {
    BlazenApiProtocol(ApiProtocol::Custom).into_ptr()
}

/// Returns `"openai"` or `"custom"` as a caller-owned C string.
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenApiProtocol`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_api_protocol_kind(handle: *const BlazenApiProtocol) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    let kind = match r.0 {
        ApiProtocol::OpenAi(_) => "openai",
        ApiProtocol::Custom => "custom",
    };
    alloc_cstring(kind)
}

/// Returns a clone of the inner `OpenAiCompatConfig` for the `OpenAi` variant,
/// or null for the `Custom` variant / null handle. Caller owns the returned
/// handle and must free with [`blazen_openai_compat_config_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenApiProtocol`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_api_protocol_config(
    handle: *const BlazenApiProtocol,
) -> *mut BlazenOpenAiCompatConfig {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees live handle.
    let r = unsafe { &*handle };
    match &r.0 {
        ApiProtocol::OpenAi(cfg) => BlazenOpenAiCompatConfig(cfg.clone()).into_ptr(),
        ApiProtocol::Custom => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenApiProtocol`. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer produced by [`blazen_api_protocol_openai`]
/// or [`blazen_api_protocol_custom`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_api_protocol_free(handle: *mut BlazenApiProtocol) {
    if handle.is_null() {
        return;
    }
    // SAFETY: per the contract, `handle` came from `Box::into_raw`.
    drop(unsafe { Box::from_raw(handle) });
}
