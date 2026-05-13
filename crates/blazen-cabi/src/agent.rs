//! Agent opaque object: wraps `blazen_uniffi::agent::Agent` for the C ABI.
//!
//! Phase R3 Agent D.
//!
//! ## Ownership conventions
//!
//! - The constructor (`Agent::new` in `blazen-uniffi`) takes a `ToolHandler`
//!   foreign callback, which depends on the Phase R5 trampoline. The cabi
//!   surface therefore does **not** expose `blazen_agent_new` yet — only the
//!   `run` / `run_blocking` entry points are wired here. See the deferred-
//!   surface note at the bottom of this file.
//! - The `*_blocking` and future-returning `_run` wrappers both produce
//!   caller-owned values. `_blocking` writes a `*mut BlazenAgentResult` into
//!   `out_result`; the future variant hands the result through
//!   [`blazen_future_take_agent_result`].
//! - Errors flow through `*mut *mut BlazenError` out-params on fallible
//!   sync wrappers; the future-returning variant funnels errors through
//!   `blazen_future_take_agent_result`'s `err` out-param.
//! - [`blazen_agent_free`] releases the opaque handle; the inner `Arc` ref
//!   is dropped (the agent's model and tool definitions live behind shared
//!   refs, so freeing the handle doesn't necessarily tear those down).

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::agent::{Agent as InnerAgent, AgentResult as InnerAgentResult};
use blazen_uniffi::errors::BlazenError as InnerError;

use crate::agent_records::BlazenAgentResult;
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::cstr_to_str;

// ---------------------------------------------------------------------------
// Shared error-out helpers
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`. Mirrors the
/// helper in `workflow.rs` so per-method bodies stay focused on the happy
/// path.
///
/// # Safety
///
/// `out_err` must be null OR a valid destination for a single
/// `*mut BlazenError` write (typically a caller's stack-local
/// `*mut BlazenError` initialised to null).
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
/// Same contract as [`write_error`]: `out_err` is null OR points at a single
/// writable `*mut BlazenError` slot.
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
// BlazenAgent
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::agent::Agent`.
///
/// The inner `Arc` matches the `self: Arc<Self>` shape of the underlying
/// async methods — each C entry point clones the ref so the spawned task
/// can keep the agent alive for the duration of the run without the cabi
/// handle being pinned to the foreign caller's call stack.
pub struct BlazenAgent(pub(crate) Arc<InnerAgent>);

impl BlazenAgent {
    /// Heap-allocates the handle and returns its raw pointer. Used by the
    /// Phase R5 constructor wrapper once the `ToolHandler` trampoline lands;
    /// the helper is kept crate-private rather than removed so the R5 wiring
    /// doesn't have to re-derive the boxing dance.
    #[allow(dead_code)]
    pub(crate) fn into_ptr(self) -> *mut BlazenAgent {
        Box::into_raw(Box::new(self))
    }
}

impl From<Arc<InnerAgent>> for BlazenAgent {
    fn from(inner: Arc<InnerAgent>) -> Self {
        Self(inner)
    }
}

/// Synchronously runs the agent loop with `user_input` as the initial user
/// message. Blocks the calling thread on the cabi tokio runtime. Returns `0`
/// on success (writing a caller-owned `*mut BlazenAgentResult` to
/// `out_result`) or `-1` on failure (writing the inner error to `out_err`).
///
/// # Safety
///
/// `agent` must be a valid pointer to a `BlazenAgent` previously produced by
/// the cabi surface. `user_input` must be a valid NUL-terminated UTF-8
/// buffer that remains live for the duration of the call. `out_result` is
/// null OR a valid destination for one `*mut BlazenAgentResult` write.
/// `out_err` is null OR a valid destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_run_blocking(
    agent: *const BlazenAgent,
    user_input: *const c_char,
    out_result: *mut *mut BlazenAgentResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if agent.is_null() || user_input.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `agent` is a live pointer.
    let agent = unsafe { &*agent };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `user_input`.
    let input = match unsafe { cstr_to_str(user_input) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => return unsafe { write_internal_error(out_err, "user_input not valid UTF-8") },
    };
    let inner = Arc::clone(&agent.0);
    match runtime().block_on(async move { inner.run(input).await }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: caller-supplied out-param; per the function-level
                // contract it's either null (handled above) or a valid
                // destination for a single pointer-sized write.
                unsafe {
                    *out_result = BlazenAgentResult::from(result).into_ptr();
                }
            }
            0
        }
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Runs the agent loop asynchronously, returning an opaque future handle
/// immediately. The caller waits via `blazen_future_wait` / `blazen_future_fd`
/// / `blazen_future_poll`, then takes the result via
/// [`blazen_future_take_agent_result`].
///
/// Returns null if `agent` or `user_input` is null, or if `user_input` is
/// not valid UTF-8. Errors that surface during the async run are delivered
/// through `blazen_future_take_agent_result`'s `err` out-param.
///
/// # Safety
///
/// `agent` must be a valid pointer to a `BlazenAgent` previously produced by
/// the cabi surface. `user_input` must be a valid NUL-terminated UTF-8
/// buffer that remains valid for the duration of this call (the buffer is
/// copied before this function returns).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_run(
    agent: *const BlazenAgent,
    user_input: *const c_char,
) -> *mut BlazenFuture {
    if agent.is_null() || user_input.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `agent` is a live pointer.
    let agent = unsafe { &*agent };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `user_input`.
    let input = match unsafe { cstr_to_str(user_input) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    let inner = Arc::clone(&agent.0);
    BlazenFuture::spawn(async move { inner.run(input).await })
}

/// Frees a `BlazenAgent` handle previously produced by the cabi surface
/// (Phase R5 constructor wrapper). No-op on a null pointer.
///
/// # Safety
///
/// `agent` must be null OR a pointer previously produced by the cabi
/// surface as a `BlazenAgent`. Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_free(agent: *mut BlazenAgent) {
    if agent.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(agent) });
}

// ---------------------------------------------------------------------------
// Typed future-take for AgentResult
// ---------------------------------------------------------------------------

/// Pops the `AgentResult` out of a future produced by [`blazen_agent_run`].
/// Returns `0` on success or `-1` on error.
///
/// On success, `out` receives a caller-owned `*mut BlazenAgentResult` (free
/// with [`crate::agent_records::blazen_agent_result_free`]). On error, `err`
/// receives a caller-owned `*mut BlazenError` (free with
/// [`crate::error::blazen_error_free`]).
///
/// # Safety
///
/// All three pointers must follow the cabi-future contract: `fut` is a live
/// future produced by [`blazen_agent_run`], observed completed via
/// `blazen_future_poll` / `_wait` / `_fd`. `out` and `err` are valid
/// destinations for a single `*mut` write (typically stack `*mut BlazenX`
/// locals — can be null to discard).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_agent_result(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenAgentResult,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the future-pointer contract documented above.
    match unsafe { BlazenFuture::take_typed::<InnerAgentResult>(fut) } {
        Ok(v) => {
            if !out.is_null() {
                // SAFETY: caller-supplied out-param; per the contract above
                // it's either null (handled) or a valid destination for a
                // single pointer-sized write.
                unsafe {
                    *out = BlazenAgentResult::from(v).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: caller-supplied out-param; same contract as `out`.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Deferred surface
// ---------------------------------------------------------------------------

// `blazen_agent_new` is intentionally **not** exposed in Phase R3. Building
// an `Agent` requires a foreign-language `ToolHandler` trampoline — a
// callback that the Rust agent loop can invoke from arbitrary tokio worker
// threads, with the foreign-language method returning a JSON-encoded tool
// result string. That trampoline lands in Phase R5 alongside the
// `blazen_workflow_builder_step` bridge; until then, the cabi surface is
// run-only against agents constructed via UniFFI or future R5 wiring.
