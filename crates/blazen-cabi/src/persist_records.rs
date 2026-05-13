//! Persistence-related record marshalling. Opaque handles wrap
//! `blazen_uniffi::persist::{PersistedEvent, WorkflowCheckpoint}` records
//! with constructor + accessor + free entry points.
//!
//! ## Ownership conventions
//!
//! - `*_new` returns a caller-owned pointer freed with the matching
//!   `*_free`. Null inputs / non-UTF-8 strings → null output.
//! - String getters return caller-owned heap C strings; free with
//!   [`crate::string::blazen_string_free`].
//! - Numeric getters return by value.
//! - `*_pending_events_get` clones the indexed element into a fresh
//!   handle the caller owns (because borrowing into a Rust Vec from C
//!   isn't safe across the FFI boundary).
//! - `*_pending_events_push` **takes ownership** of the pushed
//!   `BlazenPersistedEvent`. Callers MUST NOT separately free the input
//!   after a successful push.

// Crate-private helpers (`into_ptr`, `From<Inner>` impls) wire in during
// Phase R3 when the CheckpointStore method wrappers land — until then
// only the public extern symbols are reachable directly. Allowing
// `dead_code` keeps clippy quiet without weakening the public surface.
#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::persist::{
    PersistedEvent as InnerPersistedEvent, WorkflowCheckpoint as InnerWorkflowCheckpoint,
};

use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// BlazenPersistedEvent
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a `blazen_uniffi::persist::PersistedEvent`.
///
/// Same wire shape as a workflow [`crate::workflow_records::BlazenEvent`] —
/// `{ event_type, data_json }` — but a distinct opaque type because
/// `PersistedEvent` carries the additional contract that `data_json` is
/// derived from a `serde_json::Value` (see
/// `blazen_uniffi::persist::PersistedEvent::TryFrom<SerializedEvent>`).
pub struct BlazenPersistedEvent(pub(crate) InnerPersistedEvent);

impl BlazenPersistedEvent {
    /// Heap-allocates the handle and returns its raw pointer.
    pub(crate) fn into_ptr(self) -> *mut BlazenPersistedEvent {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerPersistedEvent> for BlazenPersistedEvent {
    fn from(inner: InnerPersistedEvent) -> Self {
        Self(inner)
    }
}

/// Constructs a new persisted event with the given `event_type` (e.g.
/// `"blazen::StartEvent"`) and JSON-encoded `data_json` payload. Returns
/// null if either pointer is null or contains non-UTF-8 bytes.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with
/// [`blazen_persisted_event_free`].
///
/// # Safety
///
/// Both pointers must be null OR valid NUL-terminated UTF-8 buffers that
/// remain live for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_persisted_event_new(
    event_type: *const c_char,
    data_json: *const c_char,
) -> *mut BlazenPersistedEvent {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(event_type) = (unsafe { cstr_to_str(event_type) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let Some(data_json) = (unsafe { cstr_to_str(data_json) }) else {
        return std::ptr::null_mut();
    };
    BlazenPersistedEvent::from(InnerPersistedEvent {
        event_type: event_type.to_owned(),
        data_json: data_json.to_owned(),
    })
    .into_ptr()
}

/// Returns the persisted event's `event_type` as a heap-allocated C string.
/// Returns null if `event` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenPersistedEvent`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_persisted_event_event_type(
    event: *const BlazenPersistedEvent,
) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live pointer.
    let event = unsafe { &*event };
    alloc_cstring(&event.0.event_type)
}

/// Returns the persisted event's `data_json` as a heap-allocated C string.
/// Returns null if `event` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenPersistedEvent`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_persisted_event_data_json(
    event: *const BlazenPersistedEvent,
) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live pointer.
    let event = unsafe { &*event };
    alloc_cstring(&event.0.data_json)
}

/// Frees a `BlazenPersistedEvent` handle previously produced by the cabi
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `event` must be null OR a pointer previously produced by
/// [`blazen_persisted_event_new`] (or any other cabi function producing
/// a `BlazenPersistedEvent`). Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_persisted_event_free(event: *mut BlazenPersistedEvent) {
    if event.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(event) });
}

// ---------------------------------------------------------------------------
// BlazenWorkflowCheckpoint
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a `blazen_uniffi::persist::WorkflowCheckpoint`.
///
/// `pending_events` is exposed via the `_pending_events_count` /
/// `_pending_events_get(idx)` getter pair (cloning on read) plus a
/// `_pending_events_push` adder (consuming on write). All other fields
/// are simple scalars or strings with direct getters; the constructor
/// takes the four input strings and `timestamp_ms` upfront.
pub struct BlazenWorkflowCheckpoint(pub(crate) InnerWorkflowCheckpoint);

impl BlazenWorkflowCheckpoint {
    /// Heap-allocates the handle and returns its raw pointer.
    pub(crate) fn into_ptr(self) -> *mut BlazenWorkflowCheckpoint {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerWorkflowCheckpoint> for BlazenWorkflowCheckpoint {
    fn from(inner: InnerWorkflowCheckpoint) -> Self {
        Self(inner)
    }
}

/// Constructs a new workflow checkpoint.
///
/// `workflow_name` is the human-readable workflow identifier. `run_id` is
/// a UUID string (or empty — the inner `try_into::<CoreWorkflowCheckpoint>`
/// generates a fresh UUID when persisted with an empty `run_id`).
/// `state_json` and `metadata_json` are JSON-encoded objects (or empty
/// strings to mean "no state" / "no metadata"). `timestamp_ms` is
/// Unix-epoch milliseconds.
///
/// Returns null if any of the four string pointers is null or contains
/// non-UTF-8 bytes. The pending-events list starts empty — use
/// [`blazen_workflow_checkpoint_pending_events_push`] to append entries.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with
/// [`blazen_workflow_checkpoint_free`].
///
/// # Safety
///
/// All four string pointers must be null OR valid NUL-terminated UTF-8
/// buffers that remain live for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_new(
    workflow_name: *const c_char,
    run_id: *const c_char,
    state_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_ms: u64,
) -> *mut BlazenWorkflowCheckpoint {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(workflow_name) = (unsafe { cstr_to_str(workflow_name) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let Some(run_id) = (unsafe { cstr_to_str(run_id) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let Some(state_json) = (unsafe { cstr_to_str(state_json) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let Some(metadata_json) = (unsafe { cstr_to_str(metadata_json) }) else {
        return std::ptr::null_mut();
    };
    BlazenWorkflowCheckpoint::from(InnerWorkflowCheckpoint {
        workflow_name: workflow_name.to_owned(),
        run_id: run_id.to_owned(),
        timestamp_ms,
        state_json: state_json.to_owned(),
        pending_events: Vec::new(),
        metadata_json: metadata_json.to_owned(),
    })
    .into_ptr()
}

/// Appends `event` to the checkpoint's `pending_events` vec. Consumes
/// ownership of `event` — callers MUST NOT separately free it after a
/// successful push.
///
/// Returns `0` on success, `-1` if either pointer is null. On the
/// null-checkpoint path the function will still reclaim and free `event`
/// (if non-null) so the caller-allocated input doesn't leak.
///
/// # Safety
///
/// `checkpoint` must be a valid pointer to a `BlazenWorkflowCheckpoint`
/// previously produced by the cabi surface. `event` must be null OR a
/// pointer previously produced by [`blazen_persisted_event_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_pending_events_push(
    checkpoint: *mut BlazenWorkflowCheckpoint,
    event: *mut BlazenPersistedEvent,
) -> i32 {
    if checkpoint.is_null() {
        if !event.is_null() {
            // SAFETY: caller upholds the `Box::into_raw` contract on `event`.
            drop(unsafe { Box::from_raw(event) });
        }
        return -1;
    }
    if event.is_null() {
        return -1;
    }
    // SAFETY: caller upholds the `Box::into_raw` contract.
    let event = unsafe { Box::from_raw(event) }.0;
    // SAFETY: caller upholds the `Box::into_raw` contract.
    let checkpoint = unsafe { &mut *checkpoint };
    checkpoint.0.pending_events.push(event);
    0
}

/// Returns the checkpoint's `workflow_name` as a heap-allocated C string.
/// Returns null if `checkpoint` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_workflow_name(
    checkpoint: *const BlazenWorkflowCheckpoint,
) -> *mut c_char {
    if checkpoint.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    alloc_cstring(&checkpoint.0.workflow_name)
}

/// Returns the checkpoint's `run_id` (UUID string) as a heap-allocated C
/// string. Returns null if `checkpoint` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_run_id(
    checkpoint: *const BlazenWorkflowCheckpoint,
) -> *mut c_char {
    if checkpoint.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    alloc_cstring(&checkpoint.0.run_id)
}

/// Returns the checkpoint's `timestamp_ms` (Unix-epoch milliseconds), or
/// `0` if `checkpoint` is null.
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_timestamp_ms(
    checkpoint: *const BlazenWorkflowCheckpoint,
) -> u64 {
    if checkpoint.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    checkpoint.0.timestamp_ms
}

/// Returns the checkpoint's `state_json` as a heap-allocated C string.
/// Returns null if `checkpoint` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_state_json(
    checkpoint: *const BlazenWorkflowCheckpoint,
) -> *mut c_char {
    if checkpoint.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    alloc_cstring(&checkpoint.0.state_json)
}

/// Returns the checkpoint's `metadata_json` as a heap-allocated C string.
/// Returns null if `checkpoint` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_metadata_json(
    checkpoint: *const BlazenWorkflowCheckpoint,
) -> *mut c_char {
    if checkpoint.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    alloc_cstring(&checkpoint.0.metadata_json)
}

/// Returns the number of entries in the checkpoint's `pending_events`
/// vec, or `0` if `checkpoint` is null.
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_pending_events_count(
    checkpoint: *const BlazenWorkflowCheckpoint,
) -> usize {
    if checkpoint.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    checkpoint.0.pending_events.len()
}

/// Returns a freshly-cloned `BlazenPersistedEvent` handle holding the
/// event at position `idx` of the checkpoint's `pending_events` vec.
/// Returns null if `checkpoint` is null or `idx` is out of bounds.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with
/// [`blazen_persisted_event_free`].
///
/// # Safety
///
/// `checkpoint` must be null OR a valid pointer to a
/// `BlazenWorkflowCheckpoint` previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_pending_events_get(
    checkpoint: *const BlazenWorkflowCheckpoint,
    idx: usize,
) -> *mut BlazenPersistedEvent {
    if checkpoint.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `checkpoint` is a live pointer.
    let checkpoint = unsafe { &*checkpoint };
    checkpoint
        .0
        .pending_events
        .get(idx)
        .map_or(std::ptr::null_mut(), |event| {
            BlazenPersistedEvent::from(event.clone()).into_ptr()
        })
}

/// Frees a `BlazenWorkflowCheckpoint` handle previously produced by the
/// cabi surface. No-op on a null pointer. Releasing the checkpoint also
/// drops every inner `PersistedEvent` it owns — callers must NOT
/// separately free events handed off via
/// [`blazen_workflow_checkpoint_pending_events_push`].
///
/// # Safety
///
/// `checkpoint` must be null OR a pointer previously produced by
/// [`blazen_workflow_checkpoint_new`] (or any other cabi function
/// producing a `BlazenWorkflowCheckpoint`). Calling this twice on the
/// same non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_checkpoint_free(
    checkpoint: *mut BlazenWorkflowCheckpoint,
) {
    if checkpoint.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(checkpoint) });
}
