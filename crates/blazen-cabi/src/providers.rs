//! Provider factory functions for [`BlazenCompletionModel`] and
//! [`BlazenEmbeddingModel`].
//!
//! Each factory is a thin shim over the matching
//! `blazen_uniffi::providers::new_<provider>_<kind>_model(...)` fn, returning a
//! caller-owned model handle via an out-parameter. Feature-gated factories are
//! gated on the same feature flag the underlying `blazen-uniffi` symbol uses
//! — `blazen-cabi`'s features in `Cargo.toml` mirror `blazen-uniffi`'s
//! one-to-one so the gates line up automatically.
//!
//! ## Ownership conventions
//!
//! - **Input strings** (`*const c_char`) are BORROWED — callers retain the
//!   underlying buffer. Each factory clones the contents into an owned
//!   `String` before handing them to the inner factory.
//! - **Optional strings** map null → `None` and otherwise clone into
//!   `Some(String)`. The two `_compat` and Azure `resource_name` /
//!   `deployment_name` arguments are REQUIRED — a null pointer there yields a
//!   `BlazenError::Internal` and a `-1` return.
//! - **Optional `u32`** values (`context_length`, `n_gpu_layers`,
//!   `max_batch_size`, `dimensions`) come across as `i32` with `-1` meaning
//!   "not set" — see [`opt_u32_from_i32`].
//! - **On success**, `*out_model` receives a caller-owned
//!   `*mut BlazenCompletionModel` (or `*mut BlazenEmbeddingModel`). Free with
//!   [`crate::llm::blazen_completion_model_free`] /
//!   [`crate::llm::blazen_embedding_model_free`].
//! - **On failure**, `*out_err` receives a caller-owned `*mut BlazenError`.
//!   Free with [`crate::error::blazen_error_free`].
//! - Either out-parameter may be null to discard that side of the result.
//!
//! Phase R4 Agent A.

use std::ffi::c_char;

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::llm::{BlazenCompletionModel, BlazenEmbeddingModel};
use crate::string::{cstr_to_opt_string, cstr_to_str};

// ---------------------------------------------------------------------------
// Shared error-out helpers
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`. Centralises the
/// fallible-call epilogue so the per-factory bodies stay focused on the happy
/// path.
///
/// # Safety
///
/// `out_err` must be null OR a valid destination for a single
/// `*mut BlazenError` write.
unsafe fn write_error(out_err: *mut *mut BlazenError, e: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: caller-supplied out-param; per the function-level contract
        // it's either null (handled above) or a valid destination for a
        // single pointer-sized write.
        unsafe {
            *out_err = BlazenError::from(e).into_ptr();
        }
    }
    -1
}

/// Writes a synthesised [`InnerError::Internal`] to the out-param and returns
/// `-1`. Used for null-pointer / non-UTF-8 input failures where there isn't an
/// originating inner error.
///
/// # Safety
///
/// Same contract as [`write_error`].
unsafe fn write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded to `write_error`; caller upholds the same contract.
    unsafe {
        write_error(
            out_err,
            InnerError::Internal {
                message: msg.to_owned(),
            },
        )
    }
}

/// Convert a C `i32` Option-as-sentinel into the equivalent `Option<u32>`.
///
/// Any negative value (canonically `-1`) collapses to `None`; non-negative
/// values pass through as `Some(v as u32)`. Used for the local-backend
/// `context_length` / `n_gpu_layers` / `max_batch_size` / `dimensions`
/// arguments that the inner `UniFFI` factory takes as `Option<u32>`.
#[inline]
fn opt_u32_from_i32(v: i32) -> Option<u32> {
    if v < 0 {
        None
    } else {
        Some(u32::try_from(v).unwrap_or(0))
    }
}

// ---------------------------------------------------------------------------
// Cloud completion factories — common `(api_key, model, base_url)` shape
// ---------------------------------------------------------------------------

/// Constructs an `OpenAI` chat-completion model.
///
/// `api_key` is required (NUL-terminated UTF-8). `model` and `base_url` are
/// optional — pass null to use the upstream default.
///
/// On success returns `0` and writes a caller-owned `*mut BlazenCompletionModel`
/// into `*out_model`. On error returns `-1` and writes a `*mut BlazenError`
/// into `*out_err`. Either out-parameter may be null to discard.
///
/// # Safety
///
/// `api_key` must be a valid NUL-terminated UTF-8 buffer. `model` and
/// `base_url` must each be null OR a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_openai(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_openai_completion_model(api_key.to_owned(), model, base_url)
    {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs an Anthropic Messages-API chat-completion model.
///
/// See [`blazen_completion_model_new_openai`] for the argument and ownership
/// conventions — identical shape.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_anthropic(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_anthropic_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Google Gemini chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_gemini(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_gemini_completion_model(api_key.to_owned(), model, base_url)
    {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs an `OpenRouter` chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_openrouter(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_openrouter_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Groq chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_groq(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_groq_completion_model(api_key.to_owned(), model, base_url) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Together AI chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_together(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_together_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Mistral chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_mistral(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_mistral_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a `DeepSeek` chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_deepseek(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_deepseek_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Fireworks AI chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_fireworks(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_fireworks_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Perplexity chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_perplexity(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_perplexity_completion_model(
        api_key.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs an xAI (Grok) chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_xai(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_xai_completion_model(api_key.to_owned(), model, base_url) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a Cohere chat-completion model.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_openai`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_cohere(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_cohere_completion_model(api_key.to_owned(), model, base_url)
    {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// Cloud completion factories — provider-specific argument shapes
// ---------------------------------------------------------------------------

/// Constructs an Azure `OpenAI` chat-completion model.
///
/// Azure derives its endpoint from `resource_name` + `deployment_name` and its
/// model id from `deployment_name`, so there's no `model` / `base_url` knob.
/// `api_version` is optional — pass null to use the provider's pinned API
/// version.
///
/// # Safety
///
/// `api_key`, `resource_name`, and `deployment_name` must each be a valid
/// NUL-terminated UTF-8 buffer. `api_version` must be null OR a valid
/// NUL-terminated UTF-8 buffer. `out_model` / `out_err` must each be null OR
/// a valid destination for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_azure(
    api_key: *const c_char,
    resource_name: *const c_char,
    deployment_name: *const c_char,
    api_version: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(resource_name) = (unsafe { cstr_to_str(resource_name) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "resource_name must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(deployment_name) = (unsafe { cstr_to_str(deployment_name) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "deployment_name must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let api_version = unsafe { cstr_to_opt_string(api_version) };
    match blazen_uniffi::providers::new_azure_completion_model(
        api_key.to_owned(),
        resource_name.to_owned(),
        deployment_name.to_owned(),
        api_version,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs an AWS Bedrock chat-completion model.
///
/// `region` is the AWS region (e.g. `"us-east-1"`); `api_key` is the Bedrock
/// API key (pass an empty string to resolve from `AWS_BEARER_TOKEN_BEDROCK`).
///
/// # Safety
///
/// `api_key` and `region` must each be a valid NUL-terminated UTF-8 buffer.
/// `model` and `base_url` must each be null OR a valid NUL-terminated UTF-8
/// buffer. `out_model` / `out_err` must each be null OR a valid destination
/// for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_bedrock(
    api_key: *const c_char,
    region: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(region) = (unsafe { cstr_to_str(region) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "region must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_bedrock_completion_model(
        api_key.to_owned(),
        region.to_owned(),
        model,
        base_url,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a fal.ai chat-completion model.
///
/// `endpoint` selects the fal endpoint family — one of `"openai_chat"`
/// (default when null), `"openai_responses"`, `"openai_embeddings"`,
/// `"openrouter"`, `"any_llm"`. Unrecognised values fall back to
/// `OpenAiChat`. `enterprise` promotes the endpoint to its enterprise
/// variant; `auto_route_modality` toggles automatic routing to a
/// vision/audio/video endpoint when the request carries media.
///
/// # Safety
///
/// `api_key` must be a valid NUL-terminated UTF-8 buffer. `model`,
/// `endpoint`, and `base_url` must each be null OR a valid NUL-terminated
/// UTF-8 buffer. `out_model` / `out_err` must each be null OR a valid
/// destination for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_fal(
    api_key: *const c_char,
    model: *const c_char,
    endpoint: *const c_char,
    enterprise: bool,
    auto_route_modality: bool,
    base_url: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let endpoint = unsafe { cstr_to_opt_string(endpoint) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_fal_completion_model(
        api_key.to_owned(),
        model,
        base_url,
        endpoint,
        enterprise,
        auto_route_modality,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a generic OpenAI-compatible chat-completion model.
///
/// Targets any service that speaks the official `OpenAI` Chat Completions wire
/// format (vLLM, llama-server, LM Studio, local proxies, ...). All four
/// string arguments are REQUIRED.
///
/// # Safety
///
/// `provider_name`, `base_url`, `api_key`, and `model` must each be a valid
/// NUL-terminated UTF-8 buffer. `out_model` / `out_err` must each be null OR
/// a valid destination for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_openai_compat(
    provider_name: *const c_char,
    base_url: *const c_char,
    api_key: *const c_char,
    model: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(provider_name) = (unsafe { cstr_to_str(provider_name) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "provider_name must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(base_url) = (unsafe { cstr_to_str(base_url) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "base_url must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model must not be null") };
    };
    match blazen_uniffi::providers::new_openai_compat_completion_model(
        provider_name.to_owned(),
        base_url.to_owned(),
        api_key.to_owned(),
        model.to_owned(),
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a `CompletionModel` for an Ollama server.
///
/// Convenience wrapper around the OpenAI-compatible factory with
/// `base_url = format!("http://{host}:{port}/v1")` and no API key.
///
/// On success returns `0` and writes a caller-owned `*mut BlazenCompletionModel`
/// into `*out_model`. On error returns `-1` and writes a `*mut BlazenError`
/// into `*out_err`. Either out-parameter may be null to discard.
///
/// # Safety
///
/// `host` and `model` must each be a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_ollama(
    host: *const c_char,
    port: u16,
    model: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(host) = (unsafe { cstr_to_str(host) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "host must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model must not be null") };
    };
    match blazen_uniffi::providers::new_ollama_completion_model(
        host.to_owned(),
        port,
        model.to_owned(),
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a `CompletionModel` for an LM Studio server.
///
/// See [`blazen_completion_model_new_ollama`] for argument and ownership
/// conventions — identical shape.
///
/// # Safety
///
/// Same contracts as [`blazen_completion_model_new_ollama`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_lm_studio(
    host: *const c_char,
    port: u16,
    model: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(host) = (unsafe { cstr_to_str(host) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "host must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model must not be null") };
    };
    match blazen_uniffi::providers::new_lm_studio_completion_model(
        host.to_owned(),
        port,
        model.to_owned(),
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a `CompletionModel` wrapping an arbitrary OpenAI-compatible
/// server via the universal `CustomProvider`.
///
/// Pass `api_key = null` if the server does not require authentication
/// (typical for local LLM servers).
///
/// # Safety
///
/// `provider_id`, `base_url`, and `model` must each be a valid
/// NUL-terminated UTF-8 buffer. `api_key` must be null OR a valid
/// NUL-terminated UTF-8 buffer. `out_model` / `out_err` must each be null
/// OR a valid destination for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_custom_with_openai_protocol(
    provider_id: *const c_char,
    base_url: *const c_char,
    model: *const c_char,
    api_key: *const c_char,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(provider_id) = (unsafe { cstr_to_str(provider_id) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "provider_id must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(base_url) = (unsafe { cstr_to_str(base_url) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "base_url must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model) = (unsafe { cstr_to_str(model) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let api_key = unsafe { cstr_to_opt_string(api_key) };
    match blazen_uniffi::providers::new_custom_completion_model_with_openai_protocol(
        provider_id.to_owned(),
        base_url.to_owned(),
        model.to_owned(),
        api_key,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// Local completion factories — feature-gated
// ---------------------------------------------------------------------------

/// Constructs a local mistral.rs chat-completion model.
///
/// `model_id` is the `HuggingFace` repo id (e.g.
/// `"mistralai/Mistral-7B-Instruct-v0.3"`) or a local GGUF path. `device` and
/// `quantization` follow Blazen's parser format (`"cpu"`, `"cuda:0"`,
/// `"metal"`, `"q4_k_m"`, ...). `context_length` of `-1` means "use the
/// model's default"; non-negative values pass through. Set `vision = true`
/// for multimodal models like `LLaVA` / Qwen2-VL.
///
/// Feature-gated on `mistralrs`.
///
/// # Safety
///
/// `model_id` must be a valid NUL-terminated UTF-8 buffer. `device` and
/// `quantization` must each be null OR a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[cfg(feature = "mistralrs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_mistralrs(
    model_id: *const c_char,
    device: *const c_char,
    quantization: *const c_char,
    context_length: i32,
    vision: bool,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model_id) = (unsafe { cstr_to_str(model_id) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model_id must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let quantization = unsafe { cstr_to_opt_string(quantization) };
    let context_length = opt_u32_from_i32(context_length);
    match blazen_uniffi::providers::new_mistralrs_completion_model(
        model_id.to_owned(),
        device,
        quantization,
        context_length,
        vision,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a local llama.cpp chat-completion model.
///
/// `model_path` is either a local GGUF file path or a `HuggingFace` repo id;
/// `n_gpu_layers` offloads the given number of layers to the GPU when the
/// device supports it. Both `context_length` and `n_gpu_layers` use `-1` to
/// mean "not set" (i.e. use the upstream default); non-negative values pass
/// through as `Some(value as u32)`.
///
/// Feature-gated on `llamacpp`.
///
/// # Safety
///
/// `model_path` must be a valid NUL-terminated UTF-8 buffer. `device` and
/// `quantization` must each be null OR a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[cfg(feature = "llamacpp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_llamacpp(
    model_path: *const c_char,
    device: *const c_char,
    quantization: *const c_char,
    context_length: i32,
    n_gpu_layers: i32,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model_path) = (unsafe { cstr_to_str(model_path) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model_path must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let quantization = unsafe { cstr_to_opt_string(quantization) };
    let context_length = opt_u32_from_i32(context_length);
    let n_gpu_layers = opt_u32_from_i32(n_gpu_layers);
    match blazen_uniffi::providers::new_llamacpp_completion_model(
        model_path.to_owned(),
        device,
        quantization,
        context_length,
        n_gpu_layers,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a local candle chat-completion model.
///
/// Wraps `CandleLlmProvider` through the `CandleLlmCompletionModel` trait
/// bridge so it satisfies the same `CompletionModel` trait as remote
/// providers. `context_length` of `-1` means "use the model's default".
///
/// Feature-gated on `candle-llm`.
///
/// # Safety
///
/// `model_id` must be a valid NUL-terminated UTF-8 buffer. `device`,
/// `quantization`, and `revision` must each be null OR a valid
/// NUL-terminated UTF-8 buffer. `out_model` / `out_err` must each be null OR
/// a valid destination for one pointer write.
#[cfg(feature = "candle-llm")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_completion_model_new_candle(
    model_id: *const c_char,
    device: *const c_char,
    quantization: *const c_char,
    revision: *const c_char,
    context_length: i32,
    out_model: *mut *mut BlazenCompletionModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(model_id) = (unsafe { cstr_to_str(model_id) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "model_id must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let quantization = unsafe { cstr_to_opt_string(quantization) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let revision = unsafe { cstr_to_opt_string(revision) };
    let context_length = opt_u32_from_i32(context_length);
    match blazen_uniffi::providers::new_candle_completion_model(
        model_id.to_owned(),
        device,
        quantization,
        revision,
        context_length,
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenCompletionModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// Embedding factories
// ---------------------------------------------------------------------------

/// Constructs an `OpenAI` embedding model.
///
/// Defaults to `text-embedding-3-small` (1536 dimensions) when `model` is
/// null. `base_url` overrides the `OpenAI` base URL — pass null to use the
/// default.
///
/// # Safety
///
/// `api_key` must be a valid NUL-terminated UTF-8 buffer. `model` and
/// `base_url` must each be null OR a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_new_openai(
    api_key: *const c_char,
    model: *const c_char,
    base_url: *const c_char,
    out_model: *mut *mut BlazenEmbeddingModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let base_url = unsafe { cstr_to_opt_string(base_url) };
    match blazen_uniffi::providers::new_openai_embedding_model(api_key.to_owned(), model, base_url)
    {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenEmbeddingModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a fal.ai embedding model.
///
/// Routes through fal's OpenAI-compatible embeddings endpoint. `model`
/// defaults to `"openai/text-embedding-3-small"` (1536 dims) when null;
/// `dimensions` of `-1` keeps the model's default vector size, non-negative
/// values override it to the supplied dimensionality (must match an
/// upstream-supported size).
///
/// # Safety
///
/// `api_key` must be a valid NUL-terminated UTF-8 buffer. `model` must be
/// null OR a valid NUL-terminated UTF-8 buffer. `out_model` / `out_err` must
/// each be null OR a valid destination for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_new_fal(
    api_key: *const c_char,
    model: *const c_char,
    dimensions: i32,
    out_model: *mut *mut BlazenEmbeddingModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let Some(api_key) = (unsafe { cstr_to_str(api_key) }) else {
        // SAFETY: caller upholds the out-param contract on `out_err`.
        return unsafe { write_internal_error(out_err, "api_key must not be null") };
    };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model = unsafe { cstr_to_opt_string(model) };
    let dimensions = opt_u32_from_i32(dimensions);
    match blazen_uniffi::providers::new_fal_embedding_model(api_key.to_owned(), model, dimensions) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenEmbeddingModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a local fastembed (ONNX Runtime) embedding model.
///
/// `model_name` selects a variant from fastembed's catalog (case-insensitive
/// debug spelling: `"BGESmallENV15"`, `"AllMiniLML6V2"`, ...); when null,
/// defaults to `BGESmallENV15`. `max_batch_size` of `-1` keeps the upstream
/// default. `show_download_progress` is always applied (`UniFFI`'s
/// `Option<bool>` collapses to a plain bool here).
///
/// Feature-gated on `embed`.
///
/// # Safety
///
/// `model_name` must be null OR a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[cfg(feature = "embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_new_fastembed(
    model_name: *const c_char,
    max_batch_size: i32,
    show_download_progress: bool,
    out_model: *mut *mut BlazenEmbeddingModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_name = unsafe { cstr_to_opt_string(model_name) };
    let max_batch_size = opt_u32_from_i32(max_batch_size);
    match blazen_uniffi::providers::new_fastembed_embedding_model(
        model_name,
        max_batch_size,
        Some(show_download_progress),
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenEmbeddingModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a local candle text-embedding model.
///
/// Loads weights from `HuggingFace` and runs inference on-device. Defaults
/// to `"sentence-transformers/all-MiniLM-L6-v2"` when `model_id` is null.
///
/// Feature-gated on `candle-embed`.
///
/// # Safety
///
/// `model_id`, `device`, and `revision` must each be null OR a valid
/// NUL-terminated UTF-8 buffer. `out_model` / `out_err` must each be null OR
/// a valid destination for one pointer write.
#[cfg(feature = "candle-embed")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_new_candle(
    model_id: *const c_char,
    device: *const c_char,
    revision: *const c_char,
    out_model: *mut *mut BlazenEmbeddingModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_id = unsafe { cstr_to_opt_string(model_id) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let device = unsafe { cstr_to_opt_string(device) };
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let revision = unsafe { cstr_to_opt_string(revision) };
    match blazen_uniffi::providers::new_candle_embedding_model(model_id, device, revision) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenEmbeddingModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Constructs a local tract (pure-Rust ONNX) embedding model.
///
/// Drop-in replacement for [`blazen_embedding_model_new_fastembed`] for
/// targets where the prebuilt ONNX Runtime binaries can't link (musl-libc,
/// some sandboxed environments). Loads the same fastembed model catalog via
/// `tract_onnx`. `model_name` of null defaults to `BGESmallENV15`;
/// `max_batch_size` of `-1` keeps the upstream default.
///
/// Feature-gated on `tract`.
///
/// # Safety
///
/// `model_name` must be null OR a valid NUL-terminated UTF-8 buffer.
/// `out_model` / `out_err` must each be null OR a valid destination for one
/// pointer write.
#[cfg(feature = "tract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_embedding_model_new_tract(
    model_name: *const c_char,
    max_batch_size: i32,
    show_download_progress: bool,
    out_model: *mut *mut BlazenEmbeddingModel,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + UTF-8 contract.
    let model_name = unsafe { cstr_to_opt_string(model_name) };
    let max_batch_size = opt_u32_from_i32(max_batch_size);
    match blazen_uniffi::providers::new_tract_embedding_model(
        model_name,
        max_batch_size,
        Some(show_download_progress),
    ) {
        Ok(arc) => {
            if !out_model.is_null() {
                // SAFETY: caller upholds the out-param contract on `out_model`.
                unsafe {
                    *out_model = BlazenEmbeddingModel::from(arc).into_ptr();
                }
            }
            0
        }
        // SAFETY: caller upholds the out-param contract on `out_err`.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}
