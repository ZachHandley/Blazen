//! `CheckpointStore` factory functions (redb + valkey). Phase R4 Agent C.
//!
//! Constructs a [`crate::persist::BlazenCheckpointStore`] handle from one of
//! the upstream persistence backends. These are the only entry points
//! foreign callers have for minting a store — every method on
//! [`crate::persist::BlazenCheckpointStore`] consumes a pointer produced
//! here.
//!
//! # Ownership conventions
//!
//! - All input strings are BORROWED — caller keeps them alive only for the
//!   duration of the cabi call. The factories internally copy whatever they
//!   need to retain.
//! - On success, `*out_store` receives a caller-owned
//!   `*mut BlazenCheckpointStore`. Free with
//!   [`crate::persist::blazen_checkpoint_store_free`].
//! - On failure, `*out_err` receives a caller-owned `*mut BlazenError`. Free
//!   with [`crate::error::blazen_error_free`].

use std::ffi::c_char;

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::error::BlazenError;
use crate::persist::BlazenCheckpointStore;
use crate::string::cstr_to_str;

// ---------------------------------------------------------------------------
// Shared error-out helpers
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`.
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

/// Writes a synthesised `Internal` error to the out-param and returns `-1`.
/// Used for null-pointer / UTF-8 input failures where there isn't an
/// originating `InnerError`.
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
                message: msg.into(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// redb
// ---------------------------------------------------------------------------

/// Build an embedded redb-backed checkpoint store rooted at `path`.
///
/// The database file is created if it does not exist. Re-opening an
/// existing file is safe and preserves prior checkpoints.
///
/// Returns `0` on success and writes a fresh
/// `*mut BlazenCheckpointStore` into `*out_store`. Returns `-1` on backend
/// failure (writing the inner error to `*out_err`), or `-2` when `path` is
/// null or not valid UTF-8 (also written to `*out_err` as an `Internal`
/// variant).
///
/// # Safety
///
/// - `path` must be null OR a valid NUL-terminated UTF-8 buffer that
///   remains live for the duration of the call.
/// - `out_store` must be null OR a writable pointer to a
///   `*mut BlazenCheckpointStore` slot. When null the freshly built store
///   is dropped immediately to avoid a leak (the call still reports the
///   success status).
/// - `out_err` must be null OR a writable pointer to a `*mut BlazenError`
///   slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_new_redb(
    path: *const c_char,
    out_store: *mut *mut BlazenCheckpointStore,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `path`.
    let Some(path_str) = (unsafe { cstr_to_str(path) }) else {
        return unsafe { write_internal_error(out_err, "path must not be null or non-UTF-8") };
    };

    match blazen_uniffi::persist::new_redb_checkpoint_store(path_str.to_owned()) {
        Ok(arc) => {
            if out_store.is_null() {
                // Caller doesn't want the handle — drop the Arc immediately.
                drop(arc);
            } else {
                // SAFETY: caller has guaranteed `out_store` is writable.
                unsafe {
                    *out_store = BlazenCheckpointStore::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// valkey
// ---------------------------------------------------------------------------

/// Build a Redis/ValKey-backed checkpoint store connected to `url`.
///
/// `url` is in the form `redis://host:port/db` (or `rediss://` for TLS).
/// When `ttl_seconds >= 0` every saved checkpoint will auto-expire after
/// that many seconds; pass `-1` for no TTL.
///
/// The initial connection is established eagerly on the shared Tokio
/// runtime (the underlying constructor is async); subsequent reconnections
/// are handled automatically by the connection manager.
///
/// Returns `0` on success and writes a fresh
/// `*mut BlazenCheckpointStore` into `*out_store`. Returns `-1` on backend
/// failure (writing the inner error to `*out_err`), or `-2` when `url` is
/// null or not valid UTF-8.
///
/// # Safety
///
/// - `url` must be null OR a valid NUL-terminated UTF-8 buffer that
///   remains live for the duration of the call.
/// - `out_store` must be null OR a writable pointer to a
///   `*mut BlazenCheckpointStore` slot. When null the freshly built store
///   is dropped immediately to avoid a leak.
/// - `out_err` must be null OR a writable pointer to a `*mut BlazenError`
///   slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_checkpoint_store_new_valkey(
    url: *const c_char,
    ttl_seconds: i64,
    out_store: *mut *mut BlazenCheckpointStore,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `url`.
    let Some(url_str) = (unsafe { cstr_to_str(url) }) else {
        return unsafe { write_internal_error(out_err, "url must not be null or non-UTF-8") };
    };

    let ttl = if ttl_seconds < 0 {
        None
    } else {
        // `ttl_seconds >= 0` so the cast is well-defined and lossless on i64.
        #[allow(clippy::cast_sign_loss)]
        Some(ttl_seconds as u64)
    };

    match blazen_uniffi::persist::new_valkey_checkpoint_store(url_str.to_owned(), ttl) {
        Ok(arc) => {
            if out_store.is_null() {
                // Caller doesn't want the handle — drop the Arc immediately.
                drop(arc);
            } else {
                // SAFETY: caller has guaranteed `out_store` is writable.
                unsafe {
                    *out_store = BlazenCheckpointStore::from(arc).into_ptr();
                }
            }
            0
        }
        Err(e) => unsafe { write_error(out_err, e) },
    }
}
