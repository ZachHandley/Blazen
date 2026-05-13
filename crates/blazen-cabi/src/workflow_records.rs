//! Workflow-related record marshalling. Opaque handles wrap
//! `blazen_uniffi::workflow::{Event, StepOutput, WorkflowResult}` records
//! with constructor + accessor + free entry points.
//!
//! ## Ownership conventions
//!
//! - Every `*_new*` returns a caller-owned pointer that must be released
//!   with the matching `*_free`. Returning null signals an allocation /
//!   validation failure (typically a null input pointer or non-UTF-8
//!   string content).
//! - String accessors (`*_event_type`, `*_data_json`) return
//!   caller-owned heap C strings. Free with
//!   [`crate::string::blazen_string_free`].
//! - Numeric accessors return by value.
//! - Vec / enum accessors that produce sub-records (e.g.
//!   [`blazen_workflow_result_event`], [`blazen_step_output_single_event`],
//!   [`blazen_step_output_multiple_get`]) **clone** the underlying value
//!   into a fresh handle the caller owns.
//! - `*_push` adders (e.g. [`blazen_step_output_multiple_push`]) **take
//!   ownership** of the pushed handle. Callers MUST NOT separately free
//!   the input handle after a successful push; the record now owns it.
//!
//! The cabi surface is built on top of `blazen_uniffi`'s `uniffi::Record`
//! types, which are plain-data Rust structs/enums — there's no shared
//! state to thread through and every operation here is a copy / move
//! against a single owning Box.

// `into_ptr` (and the inner-struct constructors below) are crate-private
// helpers used by Phase R3+ wrappers that surface real `Event` /
// `WorkflowResult` values out of fallible cabi entry points. Their public
// extern siblings are exported via `#[unsafe(no_mangle)]` which keeps the
// linker happy regardless of dead-code warnings.
#![allow(dead_code)]

use std::ffi::c_char;

use blazen_uniffi::workflow::{
    Event as InnerEvent, StepOutput as InnerStepOutput, WorkflowResult as InnerWorkflowResult,
};

use crate::string::{alloc_cstring, cstr_to_str};

// ---------------------------------------------------------------------------
// Tag constants for the `StepOutput` enum.
// ---------------------------------------------------------------------------

/// `StepOutput` variant tag for the `None` case — the step performed work
/// but produced no event.
pub const BLAZEN_STEP_OUTPUT_NONE: u32 = 0;
/// `StepOutput` variant tag for the `Single` case — the step produced
/// exactly one event.
pub const BLAZEN_STEP_OUTPUT_SINGLE: u32 = 1;
/// `StepOutput` variant tag for the `Multiple` case — the step produced
/// zero, one, or many events (fan-out).
pub const BLAZEN_STEP_OUTPUT_MULTIPLE: u32 = 2;

// ---------------------------------------------------------------------------
// BlazenEvent
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a `blazen_uniffi::workflow::Event` value.
///
/// Deliberately not `#[repr(C)]` — cbindgen emits this as a forward-
/// declared opaque struct on the C side.
pub struct BlazenEvent(pub(crate) InnerEvent);

impl BlazenEvent {
    /// Heap-allocates the handle and returns its raw pointer. Used by
    /// other cabi modules that need to hand out an event pointer (e.g.
    /// the workflow run wrapper landing in Phase R3).
    pub(crate) fn into_ptr(self) -> *mut BlazenEvent {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerEvent> for BlazenEvent {
    fn from(inner: InnerEvent) -> Self {
        Self(inner)
    }
}

/// Constructs a new event with the given `event_type` (e.g. `"StartEvent"`)
/// and JSON-encoded `data_json` payload. Returns null if either pointer is
/// null or refers to non-UTF-8 bytes.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_event_free`].
///
/// # Safety
///
/// Both pointers must be null OR valid NUL-terminated UTF-8 buffers that
/// remain live for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_event_new(
    event_type: *const c_char,
    data_json: *const c_char,
) -> *mut BlazenEvent {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(event_type) = (unsafe { cstr_to_str(event_type) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: same contract.
    let Some(data_json) = (unsafe { cstr_to_str(data_json) }) else {
        return std::ptr::null_mut();
    };
    BlazenEvent::from(InnerEvent {
        event_type: event_type.to_owned(),
        data_json: data_json.to_owned(),
    })
    .into_ptr()
}

/// Returns the event's `event_type` field as a heap-allocated NUL-terminated
/// UTF-8 C string. Returns null if `event` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenEvent` previously
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_event_event_type(event: *const BlazenEvent) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live `BlazenEvent` pointer.
    let event = unsafe { &*event };
    alloc_cstring(&event.0.event_type)
}

/// Returns the event's `data_json` field as a heap-allocated NUL-terminated
/// UTF-8 C string. Returns null if `event` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `event` must be null OR a valid pointer to a `BlazenEvent` previously
/// produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_event_data_json(event: *const BlazenEvent) -> *mut c_char {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `event` is a live `BlazenEvent` pointer.
    let event = unsafe { &*event };
    alloc_cstring(&event.0.data_json)
}

/// Frees a `BlazenEvent` handle previously produced by the cabi surface.
/// No-op on a null pointer.
///
/// # Safety
///
/// `event` must be null OR a pointer previously produced by
/// [`blazen_event_new`] (or any other cabi function documenting
/// `BlazenEvent` ownership-transfer-to-caller semantics). Calling this
/// twice on the same non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_event_free(event: *mut BlazenEvent) {
    if event.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(event) });
}

// ---------------------------------------------------------------------------
// BlazenWorkflowResult
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a `blazen_uniffi::workflow::WorkflowResult`.
///
/// Workflow results are output-only — produced by Rust at the end of a
/// run — so no constructor is exposed across the FFI. The crate-private
/// `into_ptr` helper is the only way to mint one (used by Phase R3+
/// `blazen_workflow_run` wrappers).
pub struct BlazenWorkflowResult(pub(crate) InnerWorkflowResult);

impl BlazenWorkflowResult {
    /// Heap-allocates the handle and returns its raw pointer.
    pub(crate) fn into_ptr(self) -> *mut BlazenWorkflowResult {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerWorkflowResult> for BlazenWorkflowResult {
    fn from(inner: InnerWorkflowResult) -> Self {
        Self(inner)
    }
}

/// Returns a freshly-cloned `BlazenEvent` handle holding the terminal event
/// of this workflow run (typically a `"StopEvent"`). Returns null if
/// `result` is null.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_event_free`].
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenWorkflowResult`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_result_event(
    result: *const BlazenWorkflowResult,
) -> *mut BlazenEvent {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `result` is a live pointer.
    let result = unsafe { &*result };
    BlazenEvent::from(result.0.event.clone()).into_ptr()
}

/// Returns the total LLM input-token count accumulated across the run.
/// Returns `0` if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenWorkflowResult`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_result_total_input_tokens(
    result: *const BlazenWorkflowResult,
) -> u64 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `result` is a live pointer.
    let result = unsafe { &*result };
    result.0.total_input_tokens
}

/// Returns the total LLM output-token count accumulated across the run.
/// Returns `0` if `result` is null.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenWorkflowResult`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_result_total_output_tokens(
    result: *const BlazenWorkflowResult,
) -> u64 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `result` is a live pointer.
    let result = unsafe { &*result };
    result.0.total_output_tokens
}

/// Returns the total USD cost accumulated across the run. Returns `0.0` if
/// `result` is null or if no pricing data was available for the providers
/// involved.
///
/// # Safety
///
/// `result` must be null OR a valid pointer to a `BlazenWorkflowResult`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_result_total_cost_usd(
    result: *const BlazenWorkflowResult,
) -> f64 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller has guaranteed `result` is a live pointer.
    let result = unsafe { &*result };
    result.0.total_cost_usd
}

/// Frees a `BlazenWorkflowResult` handle previously produced by the cabi
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `result` must be null OR a pointer previously produced by the cabi
/// surface as a `BlazenWorkflowResult`. Calling this twice on the same
/// non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_result_free(result: *mut BlazenWorkflowResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(result) });
}

// ---------------------------------------------------------------------------
// BlazenStepOutput
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a `blazen_uniffi::workflow::StepOutput` enum.
///
/// `StepOutput` is what a foreign step handler returns to the Rust
/// workflow engine, so this surface needs both constructors (for foreign
/// step handlers being invoked from Rust via Phase R5 trampolines) and
/// readers (for Rust producing values consumed by foreign code).
///
/// Variant discrimination goes through [`blazen_step_output_kind`] which
/// returns one of the `BLAZEN_STEP_OUTPUT_*` constants.
pub struct BlazenStepOutput(pub(crate) InnerStepOutput);

impl BlazenStepOutput {
    /// Heap-allocates the handle and returns its raw pointer.
    pub(crate) fn into_ptr(self) -> *mut BlazenStepOutput {
        Box::into_raw(Box::new(self))
    }
}

impl From<InnerStepOutput> for BlazenStepOutput {
    fn from(inner: InnerStepOutput) -> Self {
        Self(inner)
    }
}

/// Constructs a `StepOutput::None` value — the step performed work but
/// produced no event.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_step_output_free`].
#[unsafe(no_mangle)]
pub extern "C" fn blazen_step_output_new_none() -> *mut BlazenStepOutput {
    BlazenStepOutput::from(InnerStepOutput::None).into_ptr()
}

/// Constructs a `StepOutput::Single { event }` value. Consumes ownership
/// of `event` — callers MUST NOT separately free it after this call
/// returns a non-null handle.
///
/// Returns null if `event` is null. On the null-input path the function is
/// a no-op (there's nothing to free).
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_step_output_free`].
///
/// # Safety
///
/// `event` must be null OR a pointer previously produced by
/// [`blazen_event_new`] (or any cabi function producing a `BlazenEvent`).
/// Calling this twice with the same non-null `event` is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_new_single(
    event: *mut BlazenEvent,
) -> *mut BlazenStepOutput {
    if event.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract on `event`.
    let event = unsafe { Box::from_raw(event) };
    BlazenStepOutput::from(InnerStepOutput::Single { event: event.0 }).into_ptr()
}

/// Constructs an empty `StepOutput::Multiple { events: [] }` value. Use
/// [`blazen_step_output_multiple_push`] to append events to it.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_step_output_free`].
#[unsafe(no_mangle)]
pub extern "C" fn blazen_step_output_new_multiple() -> *mut BlazenStepOutput {
    BlazenStepOutput::from(InnerStepOutput::Multiple { events: Vec::new() }).into_ptr()
}

/// Appends `event` to a `StepOutput::Multiple` value. Consumes ownership
/// of `event` — callers MUST NOT separately free it after a successful
/// push.
///
/// If `output` currently holds `None`, it transitions to
/// `Multiple { events: [event] }`. If it holds `Single { event: prior }`,
/// it transitions to `Multiple { events: [prior, event] }`. If it already
/// holds `Multiple`, the event is appended.
///
/// Returns the previous variant tag (one of the `BLAZEN_STEP_OUTPUT_*`
/// constants) on success, or `u32::MAX` if either pointer is null. On the
/// `u32::MAX` path the function is a no-op and `event` (if non-null) is
/// freed to avoid leaking caller-allocated input on the failure path.
///
/// # Safety
///
/// `output` must be a valid pointer to a `BlazenStepOutput` previously
/// produced by the cabi surface. `event` must be null OR a pointer
/// previously produced by [`blazen_event_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_multiple_push(
    output: *mut BlazenStepOutput,
    event: *mut BlazenEvent,
) -> u32 {
    if output.is_null() {
        if !event.is_null() {
            // SAFETY: caller upholds the `Box::into_raw` contract on `event`.
            drop(unsafe { Box::from_raw(event) });
        }
        return u32::MAX;
    }
    if event.is_null() {
        return u32::MAX;
    }
    // SAFETY: caller upholds the `Box::into_raw` contract on `event`.
    let event = unsafe { Box::from_raw(event) }.0;
    // SAFETY: caller upholds the `Box::into_raw` contract on `output`.
    let output = unsafe { &mut *output };

    // `std::mem::replace` lets us move out of the enum without an extra
    // clone — the placeholder `None` is overwritten below.
    let prior = std::mem::replace(&mut output.0, InnerStepOutput::None);
    let (prior_tag, next) = match prior {
        InnerStepOutput::None => (
            BLAZEN_STEP_OUTPUT_NONE,
            InnerStepOutput::Multiple {
                events: vec![event],
            },
        ),
        InnerStepOutput::Single { event: prior_event } => (
            BLAZEN_STEP_OUTPUT_SINGLE,
            InnerStepOutput::Multiple {
                events: vec![prior_event, event],
            },
        ),
        InnerStepOutput::Multiple { mut events } => {
            events.push(event);
            (
                BLAZEN_STEP_OUTPUT_MULTIPLE,
                InnerStepOutput::Multiple { events },
            )
        }
    };
    output.0 = next;
    prior_tag
}

/// Returns the variant tag of `output` — one of the `BLAZEN_STEP_OUTPUT_*`
/// constants. Returns `u32::MAX` if `output` is null.
///
/// # Safety
///
/// `output` must be null OR a valid pointer to a `BlazenStepOutput`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_kind(output: *const BlazenStepOutput) -> u32 {
    if output.is_null() {
        return u32::MAX;
    }
    // SAFETY: caller has guaranteed `output` is a live pointer.
    let output = unsafe { &*output };
    match &output.0 {
        InnerStepOutput::None => BLAZEN_STEP_OUTPUT_NONE,
        InnerStepOutput::Single { .. } => BLAZEN_STEP_OUTPUT_SINGLE,
        InnerStepOutput::Multiple { .. } => BLAZEN_STEP_OUTPUT_MULTIPLE,
    }
}

/// Returns a freshly-cloned `BlazenEvent` handle holding the inner event
/// of a `StepOutput::Single` value. Returns null if `output` is null or
/// the variant is not `Single`.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_event_free`].
///
/// # Safety
///
/// `output` must be null OR a valid pointer to a `BlazenStepOutput`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_single_event(
    output: *const BlazenStepOutput,
) -> *mut BlazenEvent {
    if output.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `output` is a live pointer.
    let output = unsafe { &*output };
    match &output.0 {
        InnerStepOutput::Single { event } => BlazenEvent::from(event.clone()).into_ptr(),
        _ => std::ptr::null_mut(),
    }
}

/// Returns the number of events in a `StepOutput::Multiple` value, or `0`
/// if `output` is null or the variant is not `Multiple`.
///
/// # Safety
///
/// `output` must be null OR a valid pointer to a `BlazenStepOutput`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_multiple_count(
    output: *const BlazenStepOutput,
) -> usize {
    if output.is_null() {
        return 0;
    }
    // SAFETY: caller has guaranteed `output` is a live pointer.
    let output = unsafe { &*output };
    match &output.0 {
        InnerStepOutput::Multiple { events } => events.len(),
        _ => 0,
    }
}

/// Returns a freshly-cloned `BlazenEvent` handle holding the event at
/// position `idx` of a `StepOutput::Multiple` value. Returns null if
/// `output` is null, the variant is not `Multiple`, or `idx` is out of
/// bounds.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_event_free`].
///
/// # Safety
///
/// `output` must be null OR a valid pointer to a `BlazenStepOutput`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_multiple_get(
    output: *const BlazenStepOutput,
    idx: usize,
) -> *mut BlazenEvent {
    if output.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `output` is a live pointer.
    let output = unsafe { &*output };
    match &output.0 {
        InnerStepOutput::Multiple { events } => {
            events.get(idx).map_or(std::ptr::null_mut(), |event| {
                BlazenEvent::from(event.clone()).into_ptr()
            })
        }
        _ => std::ptr::null_mut(),
    }
}

/// Frees a `BlazenStepOutput` handle previously produced by the cabi
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `output` must be null OR a pointer previously produced by one of the
/// `blazen_step_output_new_*` functions. Calling this twice on the same
/// non-null pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_step_output_free(output: *mut BlazenStepOutput) {
    if output.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(output) });
}
