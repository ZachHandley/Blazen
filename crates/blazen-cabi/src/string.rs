//! String marshalling helpers shared across every C entry point.
//!
//! All caller-owned strings the cabi surface returns are produced by
//! [`alloc_cstring`] and freed by [`blazen_string_free`]. Borrowed inputs go
//! through [`cstr_to_str`] or [`cstr_to_opt_string`] depending on whether
//! null is a valid sentinel for the call site.

// The borrow helpers are crate-private foundations for Phase R2+ wrappers;
// the only public symbol is `blazen_string_free`, which the linker keeps
// regardless. Suppress `dead_code` until the wrappers wire in.
#![allow(dead_code)]

use std::ffi::{CStr, CString, c_char};

/// Allocates a heap NUL-terminated UTF-8 C string from a Rust `&str`.
///
/// Returns `std::ptr::null_mut()` if `s` contains an interior NUL byte
/// (which `CString::new` rejects). The caller owns the returned pointer
/// and must release it with [`blazen_string_free`].
pub(crate) fn alloc_cstring(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Heap-allocates a fresh NUL-terminated UTF-8 C string from a borrowed
/// caller-supplied C string. The returned pointer is owned by the
/// caller (and the cabi runtime, when handed back across a vtable
/// boundary) and must eventually be released with
/// [`blazen_string_free`].
///
/// FFI-host vtables that report string output via an out-param (e.g.
/// the control-plane `AssignmentHandler`'s `handle` callback) MUST use
/// this helper to mint the output buffer so the cabi can free it on
/// the matching allocator. Foreign-allocated buffers (Ruby
/// `MemoryPointer.from_string`, etc.) are not wire-compatible with the
/// cabi's allocator and would invoke undefined behaviour if reclaimed
/// via `CString::from_raw` on the Rust side.
///
/// Returns `std::ptr::null_mut()` if `input` is null, contains non-
/// UTF-8 bytes, or has an interior NUL byte.
///
/// # Safety
///
/// `input` must be null OR a NUL-terminated buffer that remains valid
/// for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_string_alloc(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract.
    let cstr = unsafe { CStr::from_ptr(input) };
    let Ok(s) = cstr.to_str() else {
        return std::ptr::null_mut();
    };
    alloc_cstring(s)
}

/// Frees a heap-allocated C string produced by any `blazen_*` function that
/// returns `*mut c_char`. Passing a null pointer is a no-op.
///
/// # Safety
///
/// `ptr` must have been produced by a `blazen_*` function that documents
/// caller-owned-string semantics (i.e. went through [`alloc_cstring`] or
/// `CString::into_raw`). Double-free is undefined behavior. Calling this
/// on a pointer borrowed from a different allocator (e.g. `malloc`) is
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: per the contract above, `ptr` was produced by `CString::into_raw`,
    // so reconstructing the `CString` here is sound and `drop` releases the
    // original allocation.
    drop(unsafe { CString::from_raw(ptr) });
}

/// Borrows a C string as `&str`. Returns `None` on a null pointer or on
/// non-UTF-8 contents. The caller is responsible for keeping the source
/// buffer alive for the duration of the returned borrow's lifetime `'a`.
///
/// # Safety
///
/// `ptr` must be null OR point to a NUL-terminated buffer that remains
/// valid (not freed, not mutated, not reallocated) for the entirety of
/// lifetime `'a`. The buffer's contents must not be modified by another
/// thread while the borrow is live.
pub(crate) unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is non-null and the caller has guaranteed it points to a
    // valid NUL-terminated buffer that outlives `'a`.
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().ok()
}

/// Borrows a C string as an owned `Option<String>`. Convenience for surfaces
/// that take `Option<String>` arguments — a null pointer becomes `None`, a
/// non-UTF-8 pointer also becomes `None`, and a valid pointer becomes
/// `Some(String)` (independent of the source buffer's lifetime).
///
/// # Safety
///
/// Same as [`cstr_to_str`]: `ptr` must be null OR point to a NUL-terminated
/// buffer that remains valid for the duration of this call.
pub(crate) unsafe fn cstr_to_opt_string(ptr: *const c_char) -> Option<String> {
    // SAFETY: forwarded to `cstr_to_str`; caller upholds the same contract.
    unsafe { cstr_to_str(ptr) }.map(std::string::ToString::to_string)
}
