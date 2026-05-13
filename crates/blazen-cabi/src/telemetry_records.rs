//! Telemetry-related record marshalling. Opaque handles wrap
//! `blazen_uniffi::telemetry::WorkflowHistoryEntry` with accessor + free
//! entry points.
//!
//! ## Ownership conventions
//!
//! `WorkflowHistoryEntry` is output-only — values come from parsing a
//! JSON-serialised `blazen_telemetry::WorkflowHistory` on the Rust side
//! (see `blazen_uniffi::telemetry::parse_workflow_history`), never
//! constructed from C. The Phase R3 wrapper around
//! `parse_workflow_history` will hand out heap-allocated
//! `BlazenWorkflowHistoryEntry` handles using
//! [`BlazenWorkflowHistoryEntry::into_ptr`]; this module exposes only
//! the readers plus a destructor.
//!
//! - String getters (`*_workflow_id`, `*_step_name`, `*_event_type`,
//!   `*_event_data_json`, `*_error`) return caller-owned heap C strings.
//!   Free with [`crate::string::blazen_string_free`].
//! - The `*_error` accessor returns null on the `error: None` case
//!   (matching the inner `Option<String>`).
//! - `*_timestamp_ms` returns the value by `u64` directly.
//! - `*_duration_ms` returns `i64`, with `-1` as the sentinel for the
//!   inner `duration_ms: None` case (mirrors
//!   `blazen_error_retry_after_ms`'s `-1`-on-absent convention).

// `into_ptr` and the `From<Inner>` impl are crate-private factories used
// by the Phase R3 `parse_workflow_history` wrapper to mint handles out
// of parsed history entries. Allowing `dead_code` until that wrapper
// lands keeps clippy quiet without affecting the public extern surface.
#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::telemetry::WorkflowHistoryEntry as InnerWorkflowHistoryEntry;

use crate::string::alloc_cstring;

/// Opaque handle wrapping a `blazen_uniffi::telemetry::WorkflowHistoryEntry`.
///
/// Produced exclusively by the Rust side (the cabi `parse_workflow_history`
/// wrapper landing in Phase R3 will allocate a heap array of these).
/// Foreign callers read individual fields via the getters below and
/// release with [`blazen_workflow_history_entry_free`].
pub struct BlazenWorkflowHistoryEntry(pub(crate) InnerWorkflowHistoryEntry);

impl BlazenWorkflowHistoryEntry {
    /// Heap-allocates the handle and returns its raw pointer.
    pub(crate) fn into_ptr(self) -> *mut BlazenWorkflowHistoryEntry {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerWorkflowHistoryEntry> for BlazenWorkflowHistoryEntry {
    fn from(inner: InnerWorkflowHistoryEntry) -> Self {
        Self(inner)
    }
}

/// Returns the history entry's `workflow_id` (UUID string of the
/// enclosing run) as a heap-allocated C string. Returns null if `entry`
/// is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_workflow_id(
    entry: *const BlazenWorkflowHistoryEntry,
) -> *mut c_char {
    if entry.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    alloc_cstring(&entry.0.workflow_id)
}

/// Returns the history entry's `step_name` as a heap-allocated C string.
/// The value is the step name for step- or LLM-call-scoped events and
/// the empty string for workflow-level events (mirroring the inner
/// record's plain-string field — there is no `Option<String>` here, only
/// `""`).
///
/// Returns null if `entry` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_step_name(
    entry: *const BlazenWorkflowHistoryEntry,
) -> *mut c_char {
    if entry.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    alloc_cstring(&entry.0.step_name)
}

/// Returns the history entry's `event_type` variant tag (e.g.
/// `"WorkflowStarted"`, `"StepCompleted"`, `"LlmCallFailed"`) as a
/// heap-allocated C string. Returns null if `entry` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_event_type(
    entry: *const BlazenWorkflowHistoryEntry,
) -> *mut c_char {
    if entry.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    alloc_cstring(&entry.0.event_type)
}

/// Returns the history entry's full `event_data_json` payload (the serde
/// JSON of the upstream `HistoryEventKind` variant) as a heap-allocated
/// C string. Returns null if `entry` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_event_data_json(
    entry: *const BlazenWorkflowHistoryEntry,
) -> *mut c_char {
    if entry.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    alloc_cstring(&entry.0.event_data_json)
}

/// Returns the history entry's `timestamp_ms` (Unix-epoch milliseconds),
/// or `0` if `entry` is null.
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_timestamp_ms(
    entry: *const BlazenWorkflowHistoryEntry,
) -> u64 {
    if entry.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    entry.0.timestamp_ms
}

/// Returns the history entry's `duration_ms` as an `i64`. The sentinel
/// `-1` covers the `entry == null` case AND the inner
/// `duration_ms: None` case (which the upstream record uses for events
/// that have no duration concept, e.g. `WorkflowStarted`,
/// `StepDispatched`, `LlmCallStarted`).
///
/// If the inner `duration_ms: Some(value)` exceeds `i64::MAX`, the
/// result saturates to `i64::MAX` rather than overflowing — in practice
/// no real-world duration crosses that threshold (it would represent
/// hundreds of millions of years).
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_duration_ms(
    entry: *const BlazenWorkflowHistoryEntry,
) -> i64 {
    if entry.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    match entry.0.duration_ms {
        Some(value) => i64::try_from(value).unwrap_or(i64::MAX),
        None => -1,
    }
}

/// Returns the history entry's `error` message as a heap-allocated C
/// string, or null if `entry` is null OR the inner `error: None` (i.e.
/// the event is not a failure variant).
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `entry` must be null OR a valid pointer to a
/// `BlazenWorkflowHistoryEntry` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_error(
    entry: *const BlazenWorkflowHistoryEntry,
) -> *mut c_char {
    if entry.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `entry` is a live pointer.
    let entry = unsafe { &*entry };
    match &entry.0.error {
        Some(message) => alloc_cstring(message),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenWorkflowHistoryEntry` handle previously produced by
/// the cabi surface. No-op on a null pointer.
///
/// # Safety
///
/// `entry` must be null OR a pointer previously produced by the cabi
/// surface as a `BlazenWorkflowHistoryEntry`. Calling this twice on the
/// same non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_free(
    entry: *mut BlazenWorkflowHistoryEntry,
) {
    if entry.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(entry) });
}
