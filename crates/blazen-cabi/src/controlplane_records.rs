//! Opaque record handles for the control-plane C ABI: snapshots, worker
//! info, and run events.
//!
//! These mirror the wire-data structs in [`blazen_core::distributed`] and
//! cross the C boundary as opaque pointers with flat accessor functions.
//! Field strings are cloned out via [`crate::string::alloc_cstring`] and
//! freed by [`crate::string::blazen_string_free`].
//!
//! Gated on the `distributed` feature, matching the rest of the
//! control-plane surface in [`crate::controlplane`].

#![cfg(feature = "distributed")]

use std::ffi::c_char;

use blazen_core::distributed::{
    RunEvent as InnerRunEvent, RunStateSnapshot as InnerRunStateSnapshot,
    RunStatus as InnerRunStatus, WorkerInfo as InnerWorkerInfo,
};

use crate::string::alloc_cstring;

// ---------------------------------------------------------------------------
// RunStatus constants
// ---------------------------------------------------------------------------

/// Run is queued but not yet assigned to a worker.
pub const BLAZEN_RUN_STATUS_PENDING: u32 = 0;
/// Run is in flight on a worker.
pub const BLAZEN_RUN_STATUS_RUNNING: u32 = 1;
/// Run finished successfully and produced an output.
pub const BLAZEN_RUN_STATUS_COMPLETED: u32 = 2;
/// Run finished with an error.
pub const BLAZEN_RUN_STATUS_FAILED: u32 = 3;
/// Run was cancelled by the orchestrator or operator.
pub const BLAZEN_RUN_STATUS_CANCELLED: u32 = 4;

fn status_to_u32(status: InnerRunStatus) -> u32 {
    match status {
        InnerRunStatus::Pending => BLAZEN_RUN_STATUS_PENDING,
        InnerRunStatus::Running => BLAZEN_RUN_STATUS_RUNNING,
        InnerRunStatus::Completed => BLAZEN_RUN_STATUS_COMPLETED,
        InnerRunStatus::Failed => BLAZEN_RUN_STATUS_FAILED,
        InnerRunStatus::Cancelled => BLAZEN_RUN_STATUS_CANCELLED,
    }
}

// ---------------------------------------------------------------------------
// BlazenRunStateSnapshot
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a [`blazen_core::distributed::RunStateSnapshot`].
///
/// Snapshots are output-only — produced by Rust as the result of submit /
/// cancel / describe RPCs — so no public constructor is exposed across the
/// FFI. The crate-private `into_ptr` mints fresh handles.
pub struct BlazenRunStateSnapshot(pub(crate) InnerRunStateSnapshot);

impl BlazenRunStateSnapshot {
    /// Heap-allocates the handle and returns its raw pointer.
    pub(crate) fn into_ptr(self) -> *mut BlazenRunStateSnapshot {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerRunStateSnapshot> for BlazenRunStateSnapshot {
    fn from(inner: InnerRunStateSnapshot) -> Self {
        Self(inner)
    }
}

/// Returns the run id as a heap-allocated NUL-terminated UTF-8 C string
/// (the lowercase-hyphenated UUID rendering). Returns null if `snap` is
/// null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_run_id(
    snap: *const BlazenRunStateSnapshot,
) -> *mut c_char {
    if snap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    alloc_cstring(&snap.0.run_id.to_string())
}

/// Returns the run's current status as one of the `BLAZEN_RUN_STATUS_*`
/// constants. Returns `BLAZEN_RUN_STATUS_PENDING` (0) if `snap` is null —
/// callers should pre-check for null when the distinction matters.
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_status(
    snap: *const BlazenRunStateSnapshot,
) -> u32 {
    if snap.is_null() {
        return BLAZEN_RUN_STATUS_PENDING;
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    status_to_u32(snap.0.status)
}

/// Returns the submit-time wall-clock timestamp in milliseconds since the
/// Unix epoch. Returns `0` if `snap` is null.
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_started_at_ms(
    snap: *const BlazenRunStateSnapshot,
) -> u64 {
    if snap.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    snap.0.started_at_ms
}

/// Returns the terminal-state timestamp in milliseconds since the Unix
/// epoch via the `out_ms` out-param, with a `has_value` indicator
/// (`0` = unset, `1` = set). Returns `-1` if `snap` is null, `0` otherwise.
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`.
/// `out_ms` and `has_value` must each be null OR a writable destination
/// for one `u64` / `i32` respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_completed_at_ms(
    snap: *const BlazenRunStateSnapshot,
    out_ms: *mut u64,
    has_value: *mut i32,
) -> i32 {
    if snap.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    let (value, present) = match snap.0.completed_at_ms {
        Some(v) => (v, 1),
        None => (0, 0),
    };
    if !out_ms.is_null() {
        // SAFETY: caller-supplied writable slot per the function contract.
        unsafe {
            *out_ms = value;
        }
    }
    if !has_value.is_null() {
        // SAFETY: caller-supplied writable slot per the function contract.
        unsafe {
            *has_value = present;
        }
    }
    0
}

/// Returns the assigned worker's `node_id` (heap-allocated UTF-8 string),
/// or null if the run is unassigned, the field is unset, or `snap` is
/// null. Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_assigned_to(
    snap: *const BlazenRunStateSnapshot,
) -> *mut c_char {
    if snap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    match &snap.0.assigned_to {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Returns the most-recent event timestamp in milliseconds via the
/// `out_ms` and `has_value` out-params. Returns `-1` if `snap` is null,
/// `0` otherwise.
///
/// # Safety
///
/// Same as [`blazen_run_state_snapshot_completed_at_ms`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_last_event_at_ms(
    snap: *const BlazenRunStateSnapshot,
    out_ms: *mut u64,
    has_value: *mut i32,
) -> i32 {
    if snap.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    let (value, present) = match snap.0.last_event_at_ms {
        Some(v) => (v, 1),
        None => (0, 0),
    };
    if !out_ms.is_null() {
        // SAFETY: caller-supplied writable slot.
        unsafe {
            *out_ms = value;
        }
    }
    if !has_value.is_null() {
        // SAFETY: caller-supplied writable slot.
        unsafe {
            *has_value = present;
        }
    }
    0
}

/// Returns the terminal `output` as a JSON-encoded UTF-8 string when
/// `status == Completed`. Returns null if `snap` is null, `output` is
/// unset, or JSON serialisation fails. Caller frees with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_output_json(
    snap: *const BlazenRunStateSnapshot,
) -> *mut c_char {
    if snap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    match snap.0.output.as_ref() {
        Some(v) => match serde_json::to_string(v) {
            Ok(s) => alloc_cstring(&s),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Returns the run's `error` message as a heap-allocated UTF-8 string
/// when `status == Failed`. Returns null if `snap` is null or the field
/// is unset. Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `snap` must be null OR a valid pointer to a `BlazenRunStateSnapshot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_error(
    snap: *const BlazenRunStateSnapshot,
) -> *mut c_char {
    if snap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `snap` is a live pointer.
    let snap = unsafe { &*snap };
    match &snap.0.error {
        Some(s) => alloc_cstring(s),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenRunStateSnapshot` handle previously produced by the
/// cabi surface. No-op on a null pointer.
///
/// # Safety
///
/// `snap` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice on the same non-null pointer
/// is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_state_snapshot_free(snap: *mut BlazenRunStateSnapshot) {
    if snap.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(snap) });
}

// ---------------------------------------------------------------------------
// BlazenWorkerInfo
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a [`blazen_core::distributed::WorkerInfo`].
///
/// Surfaced through [`BlazenWorkerInfoList`] from `list_workers`. Tags,
/// capabilities, and the admission snapshot are exposed as JSON via the
/// accessor functions to keep the C surface flat (no nested opaque types).
pub struct BlazenWorkerInfo(pub(crate) InnerWorkerInfo);

impl BlazenWorkerInfo {
    pub(crate) fn into_ptr(self) -> *mut BlazenWorkerInfo {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerWorkerInfo> for BlazenWorkerInfo {
    fn from(inner: InnerWorkerInfo) -> Self {
        Self(inner)
    }
}

/// Returns the worker's `node_id` as a heap-allocated UTF-8 string.
/// Returns null if `info` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `info` must be null OR a valid pointer to a `BlazenWorkerInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_node_id(info: *const BlazenWorkerInfo) -> *mut c_char {
    if info.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `info` is a live pointer.
    let info = unsafe { &*info };
    alloc_cstring(&info.0.node_id)
}

/// Returns the worker's capabilities as a JSON array of
/// `{ "kind": "...", "version": <u32> }` objects. Returns null if `info`
/// is null or JSON serialisation fails (which should be impossible for
/// these well-typed fields). Caller frees with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `info` must be null OR a valid pointer to a `BlazenWorkerInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_capabilities_json(
    info: *const BlazenWorkerInfo,
) -> *mut c_char {
    if info.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `info` is a live pointer.
    let info = unsafe { &*info };
    let json: Vec<serde_json::Value> = info
        .0
        .capabilities
        .iter()
        .map(|c| {
            serde_json::json!({
                "kind": c.kind,
                "version": c.version,
            })
        })
        .collect();
    match serde_json::to_string(&json) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the worker's tags as a JSON object (`{"key": "value", ...}`).
/// Returns null if `info` is null. Caller frees with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `info` must be null OR a valid pointer to a `BlazenWorkerInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_tags_json(
    info: *const BlazenWorkerInfo,
) -> *mut c_char {
    if info.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `info` is a live pointer.
    let info = unsafe { &*info };
    match serde_json::to_string(&info.0.tags) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns a JSON encoding of the worker's `AdmissionMode`. Shapes:
/// - `{"mode":"Fixed","max_in_flight":<u32>}`
/// - `{"mode":"VramBudget","max_vram_mb":<u64>}`
/// - `{"mode":"Reactive"}`
///
/// Returns null if `info` is null. Caller frees with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `info` must be null OR a valid pointer to a `BlazenWorkerInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_admission_json(
    info: *const BlazenWorkerInfo,
) -> *mut c_char {
    if info.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `info` is a live pointer.
    let info = unsafe { &*info };
    let value = match &info.0.admission {
        blazen_core::distributed::AdmissionMode::Fixed { max_in_flight } => {
            serde_json::json!({ "mode": "Fixed", "max_in_flight": max_in_flight })
        }
        blazen_core::distributed::AdmissionMode::VramBudget { max_vram_mb } => {
            serde_json::json!({ "mode": "VramBudget", "max_vram_mb": max_vram_mb })
        }
        blazen_core::distributed::AdmissionMode::Reactive => {
            serde_json::json!({ "mode": "Reactive" })
        }
    };
    match serde_json::to_string(&value) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the latest reported in-flight count for the worker. Returns
/// `0` if `info` is null.
///
/// # Safety
///
/// `info` must be null OR a valid pointer to a `BlazenWorkerInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_in_flight(info: *const BlazenWorkerInfo) -> u32 {
    if info.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `info` is a live pointer.
    let info = unsafe { &*info };
    info.0.in_flight
}

/// Returns the timestamp (ms since the Unix epoch) when the worker
/// connected. Returns `0` if `info` is null.
///
/// # Safety
///
/// `info` must be null OR a valid pointer to a `BlazenWorkerInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_connected_at_ms(info: *const BlazenWorkerInfo) -> u64 {
    if info.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `info` is a live pointer.
    let info = unsafe { &*info };
    info.0.connected_at_ms
}

/// Frees a `BlazenWorkerInfo` handle. No-op on a null pointer.
///
/// # Safety
///
/// `info` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_free(info: *mut BlazenWorkerInfo) {
    if info.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(info) });
}

// ---------------------------------------------------------------------------
// BlazenWorkerInfoList
// ---------------------------------------------------------------------------

/// Owned list of `BlazenWorkerInfo` handles. Hands them out one at a time
/// via [`blazen_worker_info_list_get`] — the list retains ownership until
/// freed, so the per-entry pointers stay valid for the list's lifetime.
pub struct BlazenWorkerInfoList(pub(crate) Vec<InnerWorkerInfo>);

impl BlazenWorkerInfoList {
    pub(crate) fn into_ptr(self) -> *mut BlazenWorkerInfoList {
        Box::into_raw(Box::new(self))
    }
}

impl From<Vec<InnerWorkerInfo>> for BlazenWorkerInfoList {
    fn from(workers: Vec<InnerWorkerInfo>) -> Self {
        Self(workers)
    }
}

/// Returns the number of entries in the list. Returns `0` if `list` is
/// null.
///
/// # Safety
///
/// `list` must be null OR a valid pointer to a `BlazenWorkerInfoList`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_list_count(list: *const BlazenWorkerInfoList) -> usize {
    if list.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `list` is a live pointer.
    let list = unsafe { &*list };
    list.0.len()
}

/// Returns a fresh `BlazenWorkerInfo` clone for the entry at `index`,
/// owned by the caller. Returns null if `list` is null or `index >=
/// count`. Each call clones the underlying record so the returned handle
/// outlives the list.
///
/// # Ownership
///
/// Caller frees with [`blazen_worker_info_free`].
///
/// # Safety
///
/// `list` must be null OR a valid pointer to a `BlazenWorkerInfoList`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_list_get(
    list: *const BlazenWorkerInfoList,
    index: usize,
) -> *mut BlazenWorkerInfo {
    if list.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `list` is a live pointer.
    let list = unsafe { &*list };
    match list.0.get(index) {
        Some(info) => BlazenWorkerInfo(info.clone()).into_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenWorkerInfoList` previously produced by the cabi
/// control-plane surface. No-op on a null pointer. Frees the underlying
/// records but does NOT invalidate handles previously returned by
/// [`blazen_worker_info_list_get`] — those are independent clones.
///
/// # Safety
///
/// `list` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_worker_info_list_free(list: *mut BlazenWorkerInfoList) {
    if list.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(list) });
}

// ---------------------------------------------------------------------------
// BlazenRunEvent (purely Rust-side helper; surfaced via the subscription
// callback rather than as a standalone allocator).
// ---------------------------------------------------------------------------

/// Opaque wrapper around a [`blazen_core::distributed::RunEvent`].
///
/// The subscription dispatcher in [`crate::controlplane`] decodes each
/// streamed event into one of these and invokes the consumer's sink
/// callback with a borrowed view. Standalone construction across the FFI
/// isn't supported — events are output-only.
pub struct BlazenRunEvent(pub(crate) InnerRunEvent);

impl From<InnerRunEvent> for BlazenRunEvent {
    fn from(inner: InnerRunEvent) -> Self {
        Self(inner)
    }
}

// `BlazenRunEvent` is constructed only by the subscription dispatcher
// (deferred in this phase). When subscriptions ship, reintroduce
// `into_ptr` / `inner` accessors here for the dispatcher to mint
// handles and project field-level views into the sink callback.

/// Returns the run id as a heap-allocated NUL-terminated UTF-8 C string
/// (the lowercase-hyphenated UUID rendering). Returns null if `event` is
/// null. Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenRunEvent`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_event_run_id(event: *const BlazenRunEvent) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live pointer.
    let event = unsafe { &*event };
    alloc_cstring(&event.0.run_id.to_string())
}

/// Returns the event type tag as a heap-allocated UTF-8 string. Returns
/// null if `event` is null. Caller frees with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenRunEvent`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_event_event_type(event: *const BlazenRunEvent) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live pointer.
    let event = unsafe { &*event };
    alloc_cstring(&event.0.event_type)
}

/// Returns the event payload as JSON. Returns null if `event` is null or
/// JSON encoding fails. Caller frees with
/// [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenRunEvent`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_event_data_json(event: *const BlazenRunEvent) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live pointer.
    let event = unsafe { &*event };
    match serde_json::to_string(&event.0.data) {
        Ok(s) => alloc_cstring(&s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Returns the event timestamp (ms since the Unix epoch). Returns `0` if
/// `event` is null.
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenRunEvent`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_event_timestamp_ms(event: *const BlazenRunEvent) -> u64 {
    if event.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `event` is a live pointer.
    let event = unsafe { &*event };
    event.0.timestamp_ms
}

/// Frees a `BlazenRunEvent` previously produced by the cabi control-plane
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `event` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_run_event_free(event: *mut BlazenRunEvent) {
    if event.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(event) });
}
