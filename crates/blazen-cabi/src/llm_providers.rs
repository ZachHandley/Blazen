//! Per-engine LLM provider opaques + C ABI factories.
//!
//! Each engine wraps an `Arc<blazen_uniffi::concrete::llm::<Engine>Provider>`
//! in a `Blazen<Engine>Provider` opaque, exposing:
//!
//! - `blazen_<engine>_provider_new` -- factory
//! - `blazen_<engine>_provider_complete` -- async future returning a
//!   `*mut BlazenFuture` whose typed result is popped with
//!   [`crate::llm::blazen_future_take_model_response`]
//! - `blazen_<engine>_provider_complete_blocking` -- synchronous variant
//! - `blazen_<engine>_provider_free` -- destructor (no-op on null)
//!
//! ## Ownership conventions
//!
//! - Provider handles are heap-allocated `Box<Blazen<Engine>Provider>` returned
//!   by the factory functions. Callers free with the matching `*_free`.
//!   Double-free is undefined behavior.
//! - Constructor string inputs (`api_key`, `resource_name`, `deployment_name`,
//!   `region`) are required and copied; `model` / `base_url` are
//!   `*const c_char` and treat null as `None`.
//! - The `request` pointer passed to `*_complete[_blocking]` is CONSUMED:
//!   internally `Box::from_raw` reclaims ownership and moves the inner record
//!   out. Calling [`crate::llm_records::blazen_model_request_free`] on the
//!   same pointer afterwards is a double-free. This matches the existing
//!   central [`crate::llm::blazen_model_complete`] / `_blocking` contract.
//! - The futures and `BlazenModelResponse`s produced here share the existing
//!   [`crate::llm::blazen_future_take_model_response`] /
//!   [`crate::llm_records::blazen_model_response_free`] pipeline used by the
//!   central [`crate::llm::BlazenModel`] — there is no per-engine result
//!   type.
//!
//! ## Relationship to the central [`crate::llm::BlazenModel`]
//!
//! The central `BlazenModel` + `blazen_model_*` entry points in
//! [`crate::llm`] remain in place — this module is purely additive. Foreign
//! hosts (Ruby today, future Dart / Crystal / Lua / PHP) can migrate to the
//! per-engine surface incrementally without breaking the existing Ruby gem
//! entry points.
//!
//! ## Why hand-expanded (no `macro_rules!`)?
//!
//! cbindgen does not expand declarative macros by default, and enabling
//! `[parse.expand]` requires a full `cargo rustc -Zunpretty=expanded` run as
//! part of the cbindgen build step (slow and brittle in CI). Each engine is
//! therefore written out longhand below; the structural shape is identical so
//! the boilerplate is mechanical.

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::llm::ModelResponse as InnerModelResponse;

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::llm_records::{BlazenModelRequest, BlazenModelResponse};
use crate::runtime::runtime;
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// Writes a caller-owned `BlazenError` into `out_err` (if non-null).
fn write_error(out_err: *mut *mut BlazenError, err: InnerError) {
    if !out_err.is_null() {
        // SAFETY: `out_err` is non-null per the branch above. The caller has
        // guaranteed it points to a writable `*mut BlazenError` slot.
        unsafe {
            *out_err = BlazenError::from(err).into_ptr();
        }
    }
}

/// Builds + writes a `BlazenError::Internal { message }` into `out_err`. Used
/// for argument-shape errors (null required pointers, non-UTF-8 strings) that
/// don't originate from a `blazen_uniffi` call.
fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    );
}

// ===========================================================================
// OpenAiProvider — chat / completion against api.openai.com (or compat)
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::OpenAiProvider>`.
///
/// Free with [`blazen_openai_provider_free`].
pub struct BlazenOpenAiProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::OpenAiProvider>);

impl BlazenOpenAiProvider {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    pub(crate) fn into_ptr(self) -> *mut BlazenOpenAiProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct an `OpenAI` provider. `model` / `base_url` may be null to defer
/// to the upstream default chat model / `https://api.openai.com/v1` endpoint.
///
/// On success returns `0` and writes a fresh handle into `*out_model`. On
/// failure returns `-1` and writes a fresh `BlazenError*` into `*out_err`.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` / `base_url` must be null OR valid NUL-terminated UTF-8 buffers.
/// - `out_model` / `out_err` must each be null OR point to a writable slot of
///   the matching pointer type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenOpenAiProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_openai_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };

    let arc = blazen_uniffi::concrete::llm::OpenAiProvider::new(api_key, model, base_url);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenOpenAiProvider(arc).into_ptr();
        }
    }
    0
}

/// Spawns a chat completion onto the cabi tokio runtime and returns an opaque
/// future handle. Pop the typed result with
/// [`crate::llm::blazen_future_take_model_response`]; free the future with
/// `blazen_future_free`.
///
/// Returns null if either `model` or `request` is null (in which case the
/// `request`, if non-null, is still consumed and freed to avoid a leak).
///
/// **The `request` pointer is consumed.** See
/// [`crate::llm::blazen_model_complete`] for the same contract on the central
/// `BlazenModel` surface.
///
/// # Safety
///
/// `model` must be null OR a live `BlazenOpenAiProvider`. `request` must be
/// null OR a live `BlazenModelRequest`; ownership transfers to this function
/// regardless of whether the call returns null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_complete(
    model: *const BlazenOpenAiProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronously runs a chat completion on the cabi tokio runtime.
///
/// On success returns `0` and writes a fresh `BlazenModelResponse*` into
/// `*out_response`. On failure returns `-1` and writes a fresh `BlazenError*`
/// into `*out_err`.
///
/// **The `request` pointer is consumed.** See
/// [`crate::llm::blazen_model_complete_blocking`].
///
/// # Safety
///
/// `model` must be null OR a live `BlazenOpenAiProvider`. `request` must be
/// null OR a live `BlazenModelRequest`; ownership transfers to this function.
/// `out_response` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_complete_blocking(
    model: *const BlazenOpenAiProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_openai_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_openai_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenOpenAiProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_openai_provider_new`]. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_free(model: *mut BlazenOpenAiProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// AnthropicProvider — Claude family via api.anthropic.com
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::AnthropicProvider>`.
pub struct BlazenAnthropicProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::AnthropicProvider>);

impl BlazenAnthropicProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenAnthropicProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct an Anthropic provider. `model` / `base_url` may be null.
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenAnthropicProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_anthropic_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };

    let arc = blazen_uniffi::concrete::llm::AnthropicProvider::new(api_key, model, base_url);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenAnthropicProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_complete(
    model: *const BlazenAnthropicProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_complete_blocking(
    model: *const BlazenAnthropicProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_anthropic_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_anthropic_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenAnthropicProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_anthropic_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_free(model: *mut BlazenAnthropicProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// GeminiProvider — Google Gemini via generativelanguage.googleapis.com
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::GeminiProvider>`.
pub struct BlazenGeminiProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::GeminiProvider>);

impl BlazenGeminiProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenGeminiProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Gemini provider. `model` / `base_url` may be null.
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenGeminiProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_gemini_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };

    let arc = blazen_uniffi::concrete::llm::GeminiProvider::new(api_key, model, base_url);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenGeminiProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_complete(
    model: *const BlazenGeminiProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_complete_blocking(
    model: *const BlazenGeminiProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_gemini_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_gemini_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenGeminiProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_gemini_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_free(model: *mut BlazenGeminiProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// AzureOpenAiProvider — Azure OpenAI deployment (resource + deployment name)
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::AzureOpenAiProvider>`.
pub struct BlazenAzureOpenAiProvider(
    pub(crate) Arc<blazen_uniffi::concrete::llm::AzureOpenAiProvider>,
);

impl BlazenAzureOpenAiProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenAzureOpenAiProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct an Azure `OpenAI` provider. `resource_name` forms the URL host
/// (`<resource>.openai.azure.com`); `deployment_name` is the Azure deployment
/// id and doubles as the model selector. There is no separate model argument
/// — Azure routes by deployment.
///
/// # Safety
///
/// - `api_key`, `resource_name`, `deployment_name` must each be valid
///   NUL-terminated UTF-8 buffers (non-null).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_new(
    api_key: *const c_char,
    resource_name: *const c_char,
    deployment_name: *const c_char,
    out_model: *mut *mut BlazenAzureOpenAiProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_azure_openai_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(resource_name) = (unsafe { cstr_to_str(resource_name) }) else {
        write_internal_error(
            out_err,
            "blazen_azure_openai_provider_new: resource_name must not be null",
        );
        return -1;
    };
    let resource_name = resource_name.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(deployment_name) = (unsafe { cstr_to_str(deployment_name) }) else {
        write_internal_error(
            out_err,
            "blazen_azure_openai_provider_new: deployment_name must not be null",
        );
        return -1;
    };
    let deployment_name = deployment_name.to_owned();

    let arc = blazen_uniffi::concrete::llm::AzureOpenAiProvider::new(
        api_key,
        resource_name,
        deployment_name,
    );
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenAzureOpenAiProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_complete(
    model: *const BlazenAzureOpenAiProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_complete_blocking(
    model: *const BlazenAzureOpenAiProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_azure_openai_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_azure_openai_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenAzureOpenAiProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_azure_openai_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_free(model: *mut BlazenAzureOpenAiProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// BedrockProvider — AWS Bedrock (region-scoped)
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::BedrockProvider>`.
pub struct BlazenBedrockProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::BedrockProvider>);

impl BlazenBedrockProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenBedrockProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Bedrock provider. `region` is required (e.g. `"us-east-1"`);
/// `model` may be null.
///
/// # Safety
///
/// - `api_key` and `region` must each be valid NUL-terminated UTF-8 buffers
///   (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_new(
    api_key: *const c_char,
    region: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenBedrockProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_bedrock_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(region) = (unsafe { cstr_to_str(region) }) else {
        write_internal_error(
            out_err,
            "blazen_bedrock_provider_new: region must not be null",
        );
        return -1;
    };
    let region = region.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::BedrockProvider::new(api_key, region, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenBedrockProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_complete(
    model: *const BlazenBedrockProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_complete_blocking(
    model: *const BlazenBedrockProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_bedrock_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_bedrock_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenBedrockProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_bedrock_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_free(model: *mut BlazenBedrockProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// FalLlmProvider — fal.ai LLM endpoints
// ===========================================================================
//
// Named `FalLlmProvider` (not `FalProvider`) at the binding-surface layer to
// disambiguate from the per-capability `FalTtsProvider` / `FalSttProvider` /
// `FalMusicProvider` / `FalVcProvider` / `FalImageGenProvider` concretes.
// The Rust-side upstream type retains its original name
// (`blazen_llm::providers::fal::FalProvider`).

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::FalLlmProvider>`.
pub struct BlazenFalLlmProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::FalLlmProvider>);

impl BlazenFalLlmProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFalLlmProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a fal.ai LLM provider. `model` is the underlying LLM model id
/// sent in the request body (e.g. `"anthropic/claude-sonnet-4.5"`,
/// `"openai/gpt-4o"`). `base_url` overrides the default queue URL.
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenFalLlmProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_fal_llm_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };

    let arc = blazen_uniffi::concrete::llm::FalLlmProvider::new(api_key, model, base_url);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFalLlmProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_complete(
    model: *const BlazenFalLlmProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_complete_blocking(
    model: *const BlazenFalLlmProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_fal_llm_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_fal_llm_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenFalLlmProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_fal_llm_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_free(model: *mut BlazenFalLlmProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}
// ===========================================================================
// MistralProvider — Mistral AI
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::MistralProvider>`.
pub struct BlazenMistralProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::MistralProvider>);

impl BlazenMistralProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenMistralProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Mistral AI provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenMistralProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_mistral_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::MistralProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenMistralProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_complete(
    model: *const BlazenMistralProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_complete_blocking(
    model: *const BlazenMistralProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_mistral_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_mistral_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenMistralProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_mistral_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_free(model: *mut BlazenMistralProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// FireworksProvider — Fireworks AI
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::FireworksProvider>`.
pub struct BlazenFireworksProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::FireworksProvider>);

impl BlazenFireworksProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenFireworksProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Fireworks AI provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenFireworksProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_fireworks_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::FireworksProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenFireworksProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_complete(
    model: *const BlazenFireworksProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_complete_blocking(
    model: *const BlazenFireworksProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_fireworks_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_fireworks_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenFireworksProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_fireworks_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_free(model: *mut BlazenFireworksProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// DeepSeekProvider — DeepSeek
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::DeepSeekProvider>`.
pub struct BlazenDeepSeekProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::DeepSeekProvider>);

impl BlazenDeepSeekProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenDeepSeekProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `DeepSeek` provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenDeepSeekProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_deepseek_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::DeepSeekProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenDeepSeekProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_complete(
    model: *const BlazenDeepSeekProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_complete_blocking(
    model: *const BlazenDeepSeekProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_deepseek_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_deepseek_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenDeepSeekProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_deepseek_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_free(model: *mut BlazenDeepSeekProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// PerplexityProvider — Perplexity
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::PerplexityProvider>`.
pub struct BlazenPerplexityProvider(
    pub(crate) Arc<blazen_uniffi::concrete::llm::PerplexityProvider>,
);

impl BlazenPerplexityProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenPerplexityProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Perplexity provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenPerplexityProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_perplexity_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::PerplexityProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenPerplexityProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_complete(
    model: *const BlazenPerplexityProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_complete_blocking(
    model: *const BlazenPerplexityProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_perplexity_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_perplexity_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenPerplexityProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_perplexity_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_free(model: *mut BlazenPerplexityProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// TogetherProvider — Together AI
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::TogetherProvider>`.
pub struct BlazenTogetherProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::TogetherProvider>);

impl BlazenTogetherProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenTogetherProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Together AI provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenTogetherProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_together_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::TogetherProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenTogetherProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_complete(
    model: *const BlazenTogetherProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_complete_blocking(
    model: *const BlazenTogetherProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_together_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_together_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenTogetherProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_together_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_free(model: *mut BlazenTogetherProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// GroqProvider — Groq
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::GroqProvider>`.
pub struct BlazenGroqProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::GroqProvider>);

impl BlazenGroqProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenGroqProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Groq provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenGroqProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_groq_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::GroqProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenGroqProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_complete(
    model: *const BlazenGroqProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_complete_blocking(
    model: *const BlazenGroqProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_groq_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_groq_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenGroqProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_groq_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_free(model: *mut BlazenGroqProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// OpenRouterProvider — OpenRouter
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::OpenRouterProvider>`.
pub struct BlazenOpenRouterProvider(
    pub(crate) Arc<blazen_uniffi::concrete::llm::OpenRouterProvider>,
);

impl BlazenOpenRouterProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenOpenRouterProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a `OpenRouter` provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenOpenRouterProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_openrouter_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::OpenRouterProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenOpenRouterProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_complete(
    model: *const BlazenOpenRouterProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_complete_blocking(
    model: *const BlazenOpenRouterProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_openrouter_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_openrouter_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenOpenRouterProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_openrouter_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_free(model: *mut BlazenOpenRouterProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// CohereProvider — Cohere
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::CohereProvider>`.
pub struct BlazenCohereProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::CohereProvider>);

impl BlazenCohereProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenCohereProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a Cohere provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenCohereProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_cohere_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::CohereProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenCohereProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_complete(
    model: *const BlazenCohereProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_complete_blocking(
    model: *const BlazenCohereProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_cohere_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_cohere_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenCohereProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_cohere_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_free(model: *mut BlazenCohereProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// XaiProvider — xAI (Grok)
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::XaiProvider>`.
pub struct BlazenXaiProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::XaiProvider>);

impl BlazenXaiProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenXaiProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a xAI (Grok) provider. `model` may be null to defer to the
/// upstream default.
///
/// # Safety
///
/// - `api_key` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `model` must be null OR a valid NUL-terminated UTF-8 buffer.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_new(
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenXaiProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(out_err, "blazen_xai_provider_new: api_key must not be null");
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };

    let arc = blazen_uniffi::concrete::llm::XaiProvider::new(api_key, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenXaiProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_complete(
    model: *const BlazenXaiProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_complete_blocking(
    model: *const BlazenXaiProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(out_err, "blazen_xai_provider_complete_blocking: null model");
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_xai_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenXaiProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_xai_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_free(model: *mut BlazenXaiProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// OpenAiCompatProvider — generic OpenAI Chat Completions-compatible server
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::OpenAiCompatProvider>`.
pub struct BlazenOpenAiCompatProvider(
    pub(crate) Arc<blazen_uniffi::concrete::llm::OpenAiCompatProvider>,
);

impl BlazenOpenAiCompatProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenOpenAiCompatProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a generic `OpenAI` Chat Completions-compatible provider. All four
/// string arguments are required: `provider_name` is the logical id reported
/// by the provider, `base_url` is the server root (e.g.
/// `http://localhost:8000/v1`), `api_key` is the bearer token, and `model` is
/// the default chat model.
///
/// # Safety
///
/// - `provider_name`, `base_url`, `api_key`, `model` must each be valid
///   NUL-terminated UTF-8 buffers (non-null).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_new(
    provider_name: *const c_char,
    base_url: *const c_char,
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenOpenAiCompatProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(provider_name) = (unsafe { cstr_to_str(provider_name) }) else {
        write_internal_error(
            out_err,
            "blazen_openai_compat_provider_new: provider_name must not be null",
        );
        return -1;
    };
    let provider_name = provider_name.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(base_url) = (unsafe { cstr_to_str(base_url) }) else {
        write_internal_error(
            out_err,
            "blazen_openai_compat_provider_new: base_url must not be null",
        );
        return -1;
    };
    let base_url = base_url.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        write_internal_error(
            out_err,
            "blazen_openai_compat_provider_new: api_key must not be null",
        );
        return -1;
    };
    let api_key = api_key.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        write_internal_error(
            out_err,
            "blazen_openai_compat_provider_new: model must not be null",
        );
        return -1;
    };
    let model = model.to_owned();

    let arc = blazen_uniffi::concrete::llm::OpenAiCompatProvider::new(
        provider_name,
        base_url,
        api_key,
        model,
    );
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenOpenAiCompatProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_complete(
    model: *const BlazenOpenAiCompatProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_complete_blocking(
    model: *const BlazenOpenAiCompatProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_openai_compat_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_openai_compat_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenOpenAiCompatProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_openai_compat_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_free(
    model: *mut BlazenOpenAiCompatProvider,
) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// OllamaProvider — local Ollama server (OpenAI-compatible protocol)
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::OllamaProvider>`.
pub struct BlazenOllamaProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::OllamaProvider>);

impl BlazenOllamaProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenOllamaProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct an Ollama provider targeting `http://{host}:{port}/v1`. `host`
/// and `model` are required; `port` is taken by value.
///
/// # Safety
///
/// - `host`, `model` must each be valid NUL-terminated UTF-8 buffers
///   (non-null).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_new(
    host: *const c_char,
    port: u16,
    model: *const c_char,
    out_model: *mut *mut BlazenOllamaProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(host) = (unsafe { cstr_to_str(host) }) else {
        write_internal_error(out_err, "blazen_ollama_provider_new: host must not be null");
        return -1;
    };
    let host = host.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        write_internal_error(
            out_err,
            "blazen_ollama_provider_new: model must not be null",
        );
        return -1;
    };
    let model = model.to_owned();

    let arc = blazen_uniffi::concrete::llm::OllamaProvider::new(host, port, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenOllamaProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_complete(
    model: *const BlazenOllamaProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_complete_blocking(
    model: *const BlazenOllamaProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_ollama_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_ollama_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenOllamaProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_ollama_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_free(model: *mut BlazenOllamaProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// LmStudioProvider — local LM Studio server (OpenAI-compatible protocol)
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::LmStudioProvider>`.
pub struct BlazenLmStudioProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::LmStudioProvider>);

impl BlazenLmStudioProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenLmStudioProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct an LM Studio provider targeting `http://{host}:{port}/v1`. `host`
/// and `model` are required; `port` is taken by value.
///
/// # Safety
///
/// - `host`, `model` must each be valid NUL-terminated UTF-8 buffers
///   (non-null).
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_new(
    host: *const c_char,
    port: u16,
    model: *const c_char,
    out_model: *mut *mut BlazenLmStudioProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(host) = (unsafe { cstr_to_str(host) }) else {
        write_internal_error(
            out_err,
            "blazen_lm_studio_provider_new: host must not be null",
        );
        return -1;
    };
    let host = host.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        write_internal_error(
            out_err,
            "blazen_lm_studio_provider_new: model must not be null",
        );
        return -1;
    };
    let model = model.to_owned();

    let arc = blazen_uniffi::concrete::llm::LmStudioProvider::new(host, port, model);
    if !out_model.is_null() {
        // SAFETY: caller has guaranteed `out_model` is writable.
        unsafe {
            *out_model = BlazenLmStudioProvider(arc).into_ptr();
        }
    }
    0
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_complete(
    model: *const BlazenLmStudioProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_complete_blocking(
    model: *const BlazenLmStudioProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_lm_studio_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_lm_studio_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenLmStudioProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_lm_studio_provider_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_free(model: *mut BlazenLmStudioProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// MistralRsProvider — local mistral.rs chat-completion backend
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::MistralRsProvider>`.
#[cfg(feature = "mistralrs")]
pub struct BlazenMistralRsProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::MistralRsProvider>);

#[cfg(feature = "mistralrs")]
impl BlazenMistralRsProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenMistralRsProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a local mistral.rs provider. `model_id` is required (a
/// `HuggingFace` repo id or local GGUF path); `device` / `quantization` are
/// optional (null treated as `None`); `context_length` is a nullable pointer
/// to a `u32` (null treated as `None`); `vision` enables multimodal models.
///
/// The upstream constructor is fallible (model load can fail) — on failure
/// returns `-1` and writes a fresh `BlazenError*` into `*out_err`.
///
/// # Safety
///
/// - `model_id` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `device` / `quantization` must each be null OR valid NUL-terminated
///   UTF-8 buffers.
/// - `context_length` must be null OR point to a readable `u32`.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_new(
    model_id: *const c_char,
    device: *const c_char,
    quantization: *const c_char,
    context_length: *const u32,
    vision: bool,
    out_model: *mut *mut BlazenMistralRsProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model_id) = (unsafe { cstr_to_str(model_id) }) else {
        write_internal_error(
            out_err,
            "blazen_mistralrs_provider_new: model_id must not be null",
        );
        return -1;
    };
    let model_id = model_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let quantization = unsafe { cstr_to_opt_string(quantization) };
    let context_length = if context_length.is_null() {
        None
    } else {
        // SAFETY: caller has guaranteed `context_length` points to a readable `u32`.
        Some(unsafe { *context_length })
    };

    match blazen_uniffi::concrete::llm::MistralRsProvider::new(
        model_id,
        device,
        quantization,
        context_length,
        vision,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenMistralRsProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_complete(
    model: *const BlazenMistralRsProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_complete_blocking(
    model: *const BlazenMistralRsProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_mistralrs_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_mistralrs_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenMistralRsProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_mistralrs_provider_new`].
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_free(model: *mut BlazenMistralRsProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// LlamaCppProvider — local llama.cpp chat-completion backend
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::LlamaCppProvider>`.
#[cfg(feature = "llamacpp")]
pub struct BlazenLlamaCppProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::LlamaCppProvider>);

#[cfg(feature = "llamacpp")]
impl BlazenLlamaCppProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenLlamaCppProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a local llama.cpp provider. `model_path` is required (a local
/// GGUF file path or a `HuggingFace` repo id); `device` / `quantization` are
/// optional (null treated as `None`); `context_length` and `n_gpu_layers` are
/// nullable pointers to `u32` (null treated as `None`).
///
/// The upstream constructor is fallible (model load can fail) — on failure
/// returns `-1` and writes a fresh `BlazenError*` into `*out_err`.
///
/// # Safety
///
/// - `model_path` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `device` / `quantization` must each be null OR valid NUL-terminated
///   UTF-8 buffers.
/// - `context_length` / `n_gpu_layers` must each be null OR point to a
///   readable `u32`.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_new(
    model_path: *const c_char,
    device: *const c_char,
    quantization: *const c_char,
    context_length: *const u32,
    n_gpu_layers: *const u32,
    out_model: *mut *mut BlazenLlamaCppProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model_path) = (unsafe { cstr_to_str(model_path) }) else {
        write_internal_error(
            out_err,
            "blazen_llamacpp_provider_new: model_path must not be null",
        );
        return -1;
    };
    let model_path = model_path.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let quantization = unsafe { cstr_to_opt_string(quantization) };
    let context_length = if context_length.is_null() {
        None
    } else {
        // SAFETY: caller has guaranteed `context_length` points to a readable `u32`.
        Some(unsafe { *context_length })
    };
    let n_gpu_layers = if n_gpu_layers.is_null() {
        None
    } else {
        // SAFETY: caller has guaranteed `n_gpu_layers` points to a readable `u32`.
        Some(unsafe { *n_gpu_layers })
    };

    match blazen_uniffi::concrete::llm::LlamaCppProvider::new(
        model_path,
        device,
        quantization,
        context_length,
        n_gpu_layers,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenLlamaCppProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_complete(
    model: *const BlazenLlamaCppProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_complete_blocking(
    model: *const BlazenLlamaCppProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_llamacpp_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_llamacpp_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenLlamaCppProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_llamacpp_provider_new`].
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_free(model: *mut BlazenLlamaCppProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// CandleLlmProvider — local candle chat-completion backend
// ===========================================================================

/// Opaque handle wrapping
/// `Arc<blazen_uniffi::concrete::llm::CandleLlmProvider>`.
#[cfg(feature = "candle-llm")]
pub struct BlazenCandleLlmProvider(pub(crate) Arc<blazen_uniffi::concrete::llm::CandleLlmProvider>);

#[cfg(feature = "candle-llm")]
impl BlazenCandleLlmProvider {
    pub(crate) fn into_ptr(self) -> *mut BlazenCandleLlmProvider {
        Box::into_raw(Box::new(self))
    }
}

/// Construct a local candle provider. `model_id` is required (a `HuggingFace`
/// repo id); `device` / `quantization` / `revision` are optional (null treated
/// as `None`); `context_length` is a nullable pointer to a `u32` (null treated
/// as `None`).
///
/// The upstream constructor is fallible (model load can fail) — on failure
/// returns `-1` and writes a fresh `BlazenError*` into `*out_err`.
///
/// # Safety
///
/// - `model_id` must be a valid NUL-terminated UTF-8 buffer (non-null).
/// - `device` / `quantization` / `revision` must each be null OR valid
///   NUL-terminated UTF-8 buffers.
/// - `context_length` must be null OR point to a readable `u32`.
/// - `out_model` / `out_err` must each be null OR point to a writable slot.
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_new(
    model_id: *const c_char,
    device: *const c_char,
    quantization: *const c_char,
    revision: *const c_char,
    context_length: *const u32,
    out_model: *mut *mut BlazenCandleLlmProvider,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model_id) = (unsafe { cstr_to_str(model_id) }) else {
        write_internal_error(
            out_err,
            "blazen_candle_provider_new: model_id must not be null",
        );
        return -1;
    };
    let model_id = model_id.to_owned();
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let quantization = unsafe { cstr_to_opt_string(quantization) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let revision = unsafe { cstr_to_opt_string(revision) };
    let context_length = if context_length.is_null() {
        None
    } else {
        // SAFETY: caller has guaranteed `context_length` points to a readable `u32`.
        Some(unsafe { *context_length })
    };

    match blazen_uniffi::concrete::llm::CandleLlmProvider::new(
        model_id,
        device,
        quantization,
        revision,
        context_length,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller has guaranteed `out_model` is writable.
                unsafe {
                    *out_model = BlazenCandleLlmProvider(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Async completion. See [`blazen_openai_provider_complete`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete`].
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_complete(
    model: *const BlazenCandleLlmProvider,
    request: *mut BlazenModelRequest,
) -> *mut BlazenFuture {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        return std::ptr::null_mut();
    }
    if request.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    BlazenFuture::spawn::<InnerModelResponse, _>(async move { inner.complete(inner_request).await })
}

/// Synchronous completion. See [`blazen_openai_provider_complete_blocking`].
///
/// # Safety
///
/// Same contracts as [`blazen_openai_provider_complete_blocking`].
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_complete_blocking(
    model: *const BlazenCandleLlmProvider,
    request: *mut BlazenModelRequest,
    out_response: *mut *mut BlazenModelResponse,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if model.is_null() {
        if !request.is_null() {
            // SAFETY: caller transferred ownership; drop to avoid a leak.
            drop(unsafe { Box::from_raw(request) });
        }
        write_internal_error(
            out_err,
            "blazen_candle_provider_complete_blocking: null model",
        );
        return -1;
    }
    if request.is_null() {
        write_internal_error(
            out_err,
            "blazen_candle_provider_complete_blocking: null request",
        );
        return -1;
    }
    // SAFETY: caller has guaranteed `model` is a live handle.
    let m = unsafe { &*model };
    let inner = Arc::clone(&m.0);
    // SAFETY: caller has transferred ownership of `request`.
    let request_box = unsafe { Box::from_raw(request) };
    let inner_request = request_box.0;

    let result = runtime().block_on(async move { inner.complete(inner_request).await });
    match result {
        Ok(resp) => {
            if !out_response.is_null() {
                // SAFETY: `out_response` is non-null per the branch above.
                unsafe {
                    *out_response = BlazenModelResponse::from(resp).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            write_error(out_err, e);
            -1
        }
    }
}

/// Frees a `BlazenCandleLlmProvider`. No-op on null.
///
/// # Safety
///
/// `model` must be null OR a pointer previously produced by
/// [`blazen_candle_provider_new`].
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_free(model: *mut BlazenCandleLlmProvider) {
    if model.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(model) });
}

// ===========================================================================
// Polymorphic `as_llm_provider` conversions
// ===========================================================================
//
// One C function per engine that clones the inner per-engine `Arc<...Provider>`,
// coerces it to `Arc<dyn blazen_uniffi::concrete::bases::LlmProvider>`, and
// boxes the result into a [`BlazenLlmProvider`] handle. Used by callers that
// need to pass a polymorphic provider into a surface like
// [`crate::tool_handler::blazen_agent_new`] or [`crate::batch`].
//
// The original per-engine handle is BORROWED and remains valid after the
// conversion — both handles clean up independently.

use crate::llm_provider::BlazenLlmProvider;

/// Returns a fresh [`BlazenLlmProvider`] cloned from this per-engine handle.
/// Returns null on a null input. Caller frees with
/// [`crate::llm_provider::blazen_llm_provider_free`].
///
/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_provider_as_llm_provider(
    handle: *const BlazenOpenAiProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenAnthropicProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_anthropic_provider_as_llm_provider(
    handle: *const BlazenAnthropicProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenGeminiProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_gemini_provider_as_llm_provider(
    handle: *const BlazenGeminiProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenAzureOpenAiProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_azure_openai_provider_as_llm_provider(
    handle: *const BlazenAzureOpenAiProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenBedrockProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_bedrock_provider_as_llm_provider(
    handle: *const BlazenBedrockProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenMistralProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistral_provider_as_llm_provider(
    handle: *const BlazenMistralProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenFireworksProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fireworks_provider_as_llm_provider(
    handle: *const BlazenFireworksProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenDeepSeekProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_deepseek_provider_as_llm_provider(
    handle: *const BlazenDeepSeekProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenPerplexityProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_perplexity_provider_as_llm_provider(
    handle: *const BlazenPerplexityProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenTogetherProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_together_provider_as_llm_provider(
    handle: *const BlazenTogetherProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenGroqProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_groq_provider_as_llm_provider(
    handle: *const BlazenGroqProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenRouterProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openrouter_provider_as_llm_provider(
    handle: *const BlazenOpenRouterProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenCohereProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_cohere_provider_as_llm_provider(
    handle: *const BlazenCohereProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenXaiProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_xai_provider_as_llm_provider(
    handle: *const BlazenXaiProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenFalLlmProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_fal_llm_provider_as_llm_provider(
    handle: *const BlazenFalLlmProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenOpenAiCompatProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_openai_compat_provider_as_llm_provider(
    handle: *const BlazenOpenAiCompatProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenOllamaProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_ollama_provider_as_llm_provider(
    handle: *const BlazenOllamaProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenLmStudioProvider`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_lm_studio_provider_as_llm_provider(
    handle: *const BlazenLmStudioProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenMistralRsProvider`.
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_mistralrs_provider_as_llm_provider(
    handle: *const BlazenMistralRsProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenLlamaCppProvider`.
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_llamacpp_provider_as_llm_provider(
    handle: *const BlazenLlamaCppProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}

/// # Safety
///
/// `handle` must be null OR a live `BlazenCandleLlmProvider`.
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_candle_provider_as_llm_provider(
    handle: *const BlazenCandleLlmProvider,
) -> *mut BlazenLlmProvider {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `handle` is a live handle.
    let h = unsafe { &*handle };
    BlazenLlmProvider(h.0.clone()).into_ptr()
}
