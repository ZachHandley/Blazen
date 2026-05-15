//! C ABI for the runtime pricing-refresh path.
//!
//! Surface intentionally minimal: only the *blocking* refresh is exposed.
//! Refresh is meant to be called once at app startup; using cabi's
//! `BlazenFuture` machinery for a one-shot startup call would be more
//! complexity than payoff. Hosts that want async (e.g. Ruby with
//! `Fiber.scheduler`) can wrap this call in a thread / fiber themselves.
//!
//! Cost lookup itself flows through the existing
//! `BlazenCompletionResponse.usage.cost_usd` field which is computed from
//! the global pricing registry — calling this function populates that
//! registry with the latest catalog from blazen.dev.

use std::ffi::c_char;

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::runtime::runtime;
use crate::string::cstr_to_str;

fn write_error(out_err: *mut *mut BlazenError, err: InnerError) {
    if !out_err.is_null() {
        // SAFETY: `out_err` is non-null per the branch above; caller has
        // guaranteed it points to a writable `*mut BlazenError` slot.
        unsafe {
            *out_err = BlazenError::from(err).into_ptr();
        }
    }
}

fn write_internal_error(out_err: *mut *mut BlazenError, message: &str) -> i32 {
    write_error(
        out_err,
        InnerError::Internal {
            message: message.to_owned(),
        },
    );
    -1
}

/// Refresh the pricing registry from a remote catalog. `url` may be null
/// to use the default (`https://blazen.dev/api/pricing.json`); otherwise it
/// must be a NUL-terminated UTF-8 string.
///
/// On success returns `0` and writes the number of registered entries into
/// `*out_count` (when non-null). On failure returns `-1` and writes a
/// `BlazenError*` into `*out_err` (when non-null) — the caller owns the
/// returned error and must free it via `blazen_error_free`.
///
/// Blocks the calling thread until the HTTP fetch completes; intended to
/// be called once at application startup.
///
/// # Safety
///
/// - `url` must be null OR a NUL-terminated UTF-8 buffer valid for the
///   duration of this call.
/// - `out_count` must be null OR point to a writable `u32` slot.
/// - `out_err` must be null OR point to a writable `*mut BlazenError` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_refresh_pricing_blocking(
    url: *const c_char,
    out_count: *mut u32,
    out_err: *mut *mut BlazenError,
) -> i32 {
    let target = if url.is_null() {
        blazen_llm::DEFAULT_PRICING_URL.to_owned()
    } else {
        // SAFETY: caller has guaranteed `url` is a valid NUL-terminated UTF-8
        // buffer valid for this call.
        match unsafe { cstr_to_str(url) } {
            Some(s) => s.to_owned(),
            None => {
                return write_internal_error(out_err, "blazen_refresh_pricing: invalid utf-8 url");
            }
        }
    };

    let result =
        runtime().block_on(async move { blazen_llm::refresh_default_with_url(&target).await });

    match result {
        Ok(count) => {
            if !out_count.is_null() {
                let n = u32::try_from(count).unwrap_or(u32::MAX);
                // SAFETY: caller has guaranteed `out_count` is writable.
                unsafe {
                    *out_count = n;
                }
            }
            0
        }
        Err(e) => {
            write_error(
                out_err,
                InnerError::Internal {
                    message: e.to_string(),
                },
            );
            -1
        }
    }
}
