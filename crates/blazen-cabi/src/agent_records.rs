//! Agent-related record marshalling. Opaque handles wrap
//! `blazen_uniffi::agent::AgentResult` — the value returned from
//! `Agent::run` (wired in Phase R3).
//!
//! Ownership: each `*_free` consumes the heap allocation. String getters
//! return caller-owned C strings — free with `blazen_string_free`. Nested
//! handle getters (`_total_usage`) return caller-owned pointers cloned out
//! of the source — free via the matching nested `_free` (e.g.
//! `blazen_token_usage_free`).

#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::agent::AgentResult as InnerAgentResult;

use crate::llm_records::BlazenTokenUsage;
use crate::string::alloc_cstring;

/// Opaque handle wrapping [`InnerAgentResult`].
///
/// Produced by the cabi `Agent::run` wrapper (Phase R3). The wrapped value
/// is the canonical wire format from `blazen_uniffi::agent`; field
/// accessors below clone individual pieces out so FFI hosts can read them
/// without exposing the Rust enum layout.
pub struct BlazenAgentResult(pub(crate) InnerAgentResult);

impl BlazenAgentResult {
    /// Heap-allocate the handle and return the raw pointer the caller owns.
    ///
    /// Used by `Agent::run` (Phase R3); not exposed across the FFI directly.
    pub(crate) fn into_ptr(self) -> *mut BlazenAgentResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerAgentResult> for BlazenAgentResult {
    fn from(inner: InnerAgentResult) -> Self {
        Self(inner)
    }
}

/// Returns the agent's final message as a heap-allocated C string. Caller
/// frees with `blazen_string_free`. Returns null if `result` is null or if
/// the message contains an interior NUL byte.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenAgentResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_result_final_message(
    result: *const BlazenAgentResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    alloc_cstring(&result.0.final_message)
}

/// Returns the number of iterations (LLM round-trips) the agent loop
/// executed before terminating. Returns `0` if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenAgentResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_result_iterations(result: *const BlazenAgentResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.iterations
}

/// Returns the total number of tool calls executed across all iterations.
/// Returns `0` if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenAgentResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_result_tool_call_count(
    result: *const BlazenAgentResult,
) -> u32 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.tool_call_count
}

/// Clones the aggregated [`TokenUsage`](blazen_uniffi::llm::TokenUsage) out
/// of the result and returns a caller-owned handle. Returns null if
/// `result` is null. Caller frees with `blazen_token_usage_free`.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenAgentResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_result_total_usage(
    result: *const BlazenAgentResult,
) -> *mut BlazenTokenUsage {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    BlazenTokenUsage::from(result.0.total_usage.clone()).into_ptr()
}

/// Returns the aggregated USD cost across every completion call in the
/// loop. Returns `0.0` if `result` is null or the provider did not report
/// cost data (the wire format does not distinguish "zero" from "unknown").
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenAgentResult`
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_result_total_cost_usd(
    result: *const BlazenAgentResult,
) -> f64 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller upholds the live-pointer contract documented above.
    let result = unsafe { &*result };
    result.0.total_cost_usd
}

/// Frees a `BlazenAgentResult` produced by the cabi surface. Passing null
/// is a no-op.
///
/// # Safety
///
/// `result` must be null OR a pointer produced by the cabi surface's
/// agent-run wrapper. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_result_free(result: *mut BlazenAgentResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the unique-ownership contract documented above.
    drop(unsafe { Box::from_raw(result) });
}
