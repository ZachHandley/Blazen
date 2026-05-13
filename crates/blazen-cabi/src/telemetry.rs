//! Telemetry init/shutdown + workflow history parsing. Phase R4 Agent C.
//!
//! Feature-gated exporters mirror the uniffi surface one-to-one:
//!
//! - [`blazen_init_langfuse`] (`feature = "langfuse"`)
//! - [`blazen_init_otlp`]     (`feature = "otlp"`)
//! - [`blazen_init_prometheus`] (`feature = "prometheus"`)
//!
//! The always-available pieces are [`blazen_shutdown_telemetry`] and
//! [`blazen_parse_workflow_history`] (plus its companion array-free
//! helper).
//!
//! # Ownership conventions
//!
//! - All input strings are BORROWED — caller keeps them alive only for the
//!   duration of the cabi call.
//! - On failure, `*out_err` receives a caller-owned `*mut BlazenError`. Free
//!   with [`crate::error::blazen_error_free`].
//! - [`blazen_parse_workflow_history`] writes a caller-owned heap array of
//!   `*mut BlazenWorkflowHistoryEntry` plus its length. Release the array
//!   with [`blazen_workflow_history_entry_array_free`].

use std::ffi::c_char;

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::telemetry::WorkflowHistoryEntry as InnerWorkflowHistoryEntry;

use crate::error::BlazenError;
use crate::string::cstr_to_str;
use crate::telemetry_records::BlazenWorkflowHistoryEntry;

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
// Langfuse
// ---------------------------------------------------------------------------

/// Initialize the Langfuse LLM-observability exporter.
///
/// Spawns a background tokio task that periodically flushes buffered LLM
/// call traces, token usage, and latency data to the Langfuse ingestion
/// API. Call once at process startup, before any traced work.
///
/// Arguments:
///   - `public_key`: Langfuse public API key (HTTP Basic-auth username).
///   - `secret_key`: Langfuse secret API key (HTTP Basic-auth password).
///   - `host`: optional Langfuse host URL; null defaults to
///     `https://cloud.langfuse.com`.
///
/// Returns `0` on success, `-1` on failure (writing the inner error to
/// `*out_err`), or `-2` when `public_key` or `secret_key` is null / not
/// valid UTF-8.
///
/// # Safety
///
/// - `public_key` and `secret_key` must be valid NUL-terminated UTF-8
///   buffers that remain live for the duration of the call.
/// - `host` must be null OR a valid NUL-terminated UTF-8 buffer that
///   remains live for the duration of the call.
/// - `out_err` must be null OR a writable pointer to a `*mut BlazenError`
///   slot.
#[cfg(feature = "langfuse")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_init_langfuse(
    public_key: *const c_char,
    secret_key: *const c_char,
    host: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `public_key`.
    let Some(public_key_str) = (unsafe { cstr_to_str(public_key) }) else {
        return unsafe {
            write_internal_error(out_err, "public_key must not be null or non-UTF-8")
        };
    };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `secret_key`.
    let Some(secret_key_str) = (unsafe { cstr_to_str(secret_key) }) else {
        return unsafe {
            write_internal_error(out_err, "secret_key must not be null or non-UTF-8")
        };
    };
    let host_opt = if host.is_null() {
        None
    } else {
        // SAFETY: caller upholds the NUL-termination + lifetime contract on `host`.
        match unsafe { cstr_to_str(host) } {
            Some(s) => Some(s.to_owned()),
            None => return unsafe { write_internal_error(out_err, "host not valid UTF-8") },
        }
    };

    match blazen_uniffi::telemetry::init_langfuse(
        public_key_str.to_owned(),
        secret_key_str.to_owned(),
        host_opt,
    ) {
        Ok(()) => 0,
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// OTLP
// ---------------------------------------------------------------------------

/// Initialize the OpenTelemetry OTLP (gRPC) trace exporter.
///
/// Installs an `opentelemetry-otlp` exporter as the global tracing
/// subscriber.
///
/// Arguments:
///   - `endpoint`: OTLP gRPC endpoint URL (e.g. `"http://localhost:4317"`).
///   - `service_name`: optional service name reported to the backend; null
///     defaults to `"blazen"`.
///
/// Returns `0` on success, `-1` on failure (writing the inner error to
/// `*out_err`), or `-2` when `endpoint` is null / not valid UTF-8.
///
/// # Safety
///
/// - `endpoint` must be a valid NUL-terminated UTF-8 buffer that remains
///   live for the duration of the call.
/// - `service_name` must be null OR a valid NUL-terminated UTF-8 buffer
///   that remains live for the duration of the call.
/// - `out_err` must be null OR a writable pointer to a `*mut BlazenError`
///   slot.
#[cfg(feature = "otlp")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_init_otlp(
    endpoint: *const c_char,
    service_name: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `endpoint`.
    let Some(endpoint_str) = (unsafe { cstr_to_str(endpoint) }) else {
        return unsafe { write_internal_error(out_err, "endpoint must not be null or non-UTF-8") };
    };
    let service_name_opt = if service_name.is_null() {
        None
    } else {
        // SAFETY: caller upholds the NUL-termination + lifetime contract on `service_name`.
        match unsafe { cstr_to_str(service_name) } {
            Some(s) => Some(s.to_owned()),
            None => {
                return unsafe { write_internal_error(out_err, "service_name not valid UTF-8") };
            }
        }
    };

    match blazen_uniffi::telemetry::init_otlp(endpoint_str.to_owned(), service_name_opt) {
        Ok(()) => 0,
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// Prometheus
// ---------------------------------------------------------------------------

/// Initialize the Prometheus metrics exporter and start its HTTP listener.
///
/// Installs a global `metrics` recorder backed by Prometheus and starts an
/// HTTP server serving the `/metrics` endpoint.
///
/// `listen_address` accepts a `host:port` string (e.g. `"0.0.0.0:9100"`)
/// or a bare port (e.g. `"9100"`). The upstream listener always binds
/// `0.0.0.0`; only the port portion is honored.
///
/// Returns `0` on success, `-1` on failure (writing the inner error to
/// `*out_err`), or `-2` when `listen_address` is null / not valid UTF-8.
///
/// # Safety
///
/// - `listen_address` must be a valid NUL-terminated UTF-8 buffer that
///   remains live for the duration of the call.
/// - `out_err` must be null OR a writable pointer to a `*mut BlazenError`
///   slot.
#[cfg(feature = "prometheus")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_init_prometheus(
    listen_address: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(listen_str) = (unsafe { cstr_to_str(listen_address) }) else {
        return unsafe {
            write_internal_error(out_err, "listen_address must not be null or non-UTF-8")
        };
    };

    match blazen_uniffi::telemetry::init_prometheus(listen_str.to_owned()) {
        Ok(()) => 0,
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

/// Best-effort flush + shutdown of any initialised telemetry exporters.
///
/// Safe to call even if no exporter was initialised. Currently a no-op on
/// the upstream side — see `blazen_uniffi::telemetry::shutdown_telemetry`
/// for context — but exposed so foreign callers can wire a single
/// "shutdown" hook into their app lifecycle without conditional branching
/// on features.
#[unsafe(no_mangle)]
pub extern "C" fn blazen_shutdown_telemetry() {
    blazen_uniffi::telemetry::shutdown_telemetry();
}

// ---------------------------------------------------------------------------
// Workflow history
// ---------------------------------------------------------------------------

/// Convert a `Vec<WorkflowHistoryEntry>` into a heap-allocated array of
/// `*mut BlazenWorkflowHistoryEntry` plus its length. Mirrors the
/// `checkpoints_to_c_array` helper in `persist.rs`.
fn entries_to_c_array(
    entries: Vec<InnerWorkflowHistoryEntry>,
) -> (*mut *mut BlazenWorkflowHistoryEntry, usize) {
    let boxed: Box<[*mut BlazenWorkflowHistoryEntry]> = entries
        .into_iter()
        .map(|e| BlazenWorkflowHistoryEntry::from(e).into_ptr())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let len = boxed.len();
    let raw = Box::into_raw(boxed);
    // `*mut [T]` -> `*mut T` via `.cast()` keeps provenance correct as long
    // as the matching free reconstructs the same `[T]` shape via
    // `slice::from_raw_parts_mut(base, len)` + `Box::from_raw`.
    (raw.cast::<*mut BlazenWorkflowHistoryEntry>(), len)
}

/// Decode a JSON-serialised upstream `blazen_telemetry::WorkflowHistory`
/// into a flat array of [`BlazenWorkflowHistoryEntry`] handles.
///
/// The expected input is the exact format produced by
/// `serde_json::to_string(&history)` on a `blazen_telemetry::WorkflowHistory`
/// (i.e. an object with `run_id`, `workflow_name`, and `events`).
///
/// On success returns `0`, writes the array base pointer into `*out_array`,
/// and writes its length into `*out_count`. The array contains a heap
/// pointer per event; release the whole thing with
/// [`blazen_workflow_history_entry_array_free`].
///
/// Returns `-1` on parse failure (writing the inner error to `*out_err`),
/// or `-2` when `history_json` is null or not valid UTF-8.
///
/// When `out_array` is null the freshly built array is freed immediately
/// to avoid a leak — `*out_count` is still populated.
///
/// # Safety
///
/// - `history_json` must be a valid NUL-terminated UTF-8 buffer that
///   remains live for the duration of the call.
/// - `out_array` / `out_count` / `out_err` must be null OR writable
///   pointers to the appropriate slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_parse_workflow_history(
    history_json: *const c_char,
    out_array: *mut *mut *mut BlazenWorkflowHistoryEntry,
    out_count: *mut usize,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(json_str) = (unsafe { cstr_to_str(history_json) }) else {
        return unsafe {
            write_internal_error(out_err, "history_json must not be null or non-UTF-8")
        };
    };

    match blazen_uniffi::telemetry::parse_workflow_history(json_str.to_owned()) {
        Ok(entries) => {
            let (base, count) = entries_to_c_array(entries);
            if out_array.is_null() {
                // Caller doesn't want the array — release it immediately.
                // SAFETY: `base` + `count` were just produced by
                // `entries_to_c_array`; reconstructing the boxed slice is
                // sound. The `Box<[_]>` drop releases the slice; each
                // element pointer is released via `Box::from_raw` below.
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(base, count);
                    let owned = Box::from_raw(slice);
                    for &ptr in &owned {
                        if !ptr.is_null() {
                            drop(Box::from_raw(ptr));
                        }
                    }
                    drop(owned);
                }
            } else {
                // SAFETY: caller has guaranteed `out_array` is writable.
                unsafe {
                    *out_array = base;
                }
            }
            if !out_count.is_null() {
                // SAFETY: caller has guaranteed `out_count` is writable.
                unsafe {
                    *out_count = count;
                }
            }
            0
        }
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Frees an array of `*mut BlazenWorkflowHistoryEntry` previously produced
/// by [`blazen_parse_workflow_history`].
///
/// Releases each entry handle AND the backing slice in one call. No-op on
/// a null `arr` (regardless of `count`).
///
/// # Safety
///
/// `arr` must be null OR a pointer previously produced by
/// [`blazen_parse_workflow_history`], with `count` matching its length.
/// Double-free is undefined behavior; modifying or freeing individual
/// element pointers before this call is also undefined.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_workflow_history_entry_array_free(
    arr: *mut *mut BlazenWorkflowHistoryEntry,
    count: usize,
) {
    if arr.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `arr` + `count` describe a live
    // `Box<[*mut BlazenWorkflowHistoryEntry]>` allocation. Reconstructing
    // the slice and reboxing releases the backing storage; each non-null
    // element pointer is then `Box::from_raw`'d back to release its
    // entry allocation.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(arr, count);
        let owned = Box::from_raw(slice);
        for &ptr in &owned {
            if !ptr.is_null() {
                drop(Box::from_raw(ptr));
            }
        }
        drop(owned);
    }
}
