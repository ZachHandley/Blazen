//! C ABI over `blazen-uniffi` for FFI hosts (Ruby `ffi` gem today, future
//! Dart/Crystal/Lua/PHP). Every Rust `async fn` here exposes two C entry
//! points: a `*_blocking` variant that synchronously blocks on the cabi
//! tokio runtime, and a future-returning variant that yields a `BlazenFuture*`
//! handle with a pipe read-fd suitable for `Fiber.scheduler`-aware Ruby
//! consumers (and any host that can wait on a file descriptor).
//!
//! See `crates/blazen-cabi/cbindgen.toml` and `build.rs` for header emission.
//! The committed header is at `bindings/ruby/ext/blazen/blazen.h`.

#![allow(unsafe_code)] // entire crate is an unsafe FFI surface; safety contracts documented per-fn

use std::ffi::{CString, c_char};

/// Returns the blazen-cabi crate version as a heap-allocated UTF-8 C string.
///
/// # Ownership
///
/// Caller must free with [`blazen_string_free`].
///
/// # Safety
///
/// The returned pointer is non-null and points to a NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_version() -> *mut c_char {
    // Use the workspace package version. Mirrors `blazen_uniffi::version()`.
    let v = env!("CARGO_PKG_VERSION");
    match CString::new(v) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees a heap-allocated C string produced by any `blazen_*` function that
/// returns `*mut c_char`. Passing a null pointer is a no-op.
///
/// # Safety
///
/// `ptr` must have been produced by a `blazen_*` function that documents
/// caller-owned-string semantics. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` was produced by `CString::into_raw` per the contract above.
    drop(unsafe { CString::from_raw(ptr) });
}
