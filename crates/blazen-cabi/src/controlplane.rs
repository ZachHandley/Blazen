//! Control-plane C ABI: orchestrator-side [`Client`] and worker-side
//! [`Worker`]. Mirrors the layout of [`crate::peer`] but for the
//! central-server topology in [`blazen_controlplane`].
//!
//! ## Surface
//!
//! Every async cabi entry point exposes a `*_blocking` variant (drives
//! the future on the cabi tokio runtime via `block_on`) and a
//! future-returning variant (`spawn`s onto the runtime and hands back a
//! `*mut BlazenFuture`). Typed `*_take_*` helpers pop the typed result
//! out of the future once it resolves.
//!
//! - [`blazen_controlplane_client_connect`] -> opens a connection.
//! - [`blazen_controlplane_client_submit_workflow`] -> enqueues a run
//!   and resolves with the initial [`BlazenRunStateSnapshot`].
//! - [`blazen_controlplane_client_cancel_workflow`] /
//!   [`blazen_controlplane_client_describe_workflow`] -> snapshot-
//!   producing observers.
//! - [`blazen_controlplane_client_list_workers`] ->
//!   [`BlazenWorkerInfoList`].
//! - [`blazen_controlplane_client_drain_worker`] -> unit result.
//! - [`blazen_controlplane_worker_new`] / `run` / `shutdown` /
//!   `free` -> worker lifecycle. The handler is supplied via a
//!   [`BlazenAssignmentHandlerVTable`] populated by the foreign side
//!   (e.g. Ruby `FFI::Function` ptrs).
//! - [`blazen_controlplane_client_subscribe_run_events`] /
//!   [`blazen_controlplane_client_subscribe_all`] -> register a
//!   [`BlazenRunEventSinkVTable`] callback sink and return a
//!   [`BlazenControlPlaneSubscription`] handle that can be cancelled
//!   or freed.
//!
//! ## mTLS
//!
//! [`blazen_controlplane_client_connect_with_mtls_blocking`] /
//! [`blazen_controlplane_client_connect_with_mtls`] /
//! [`blazen_controlplane_worker_new_with_mtls_blocking`] accept PEM
//! file paths for cert/key/CA and route through
//! [`InnerClient::with_mtls`] / [`InnerWorkerConfig::with_mtls`].
//!
//! Gated on the `distributed` feature, matching the rest of the
//! control-plane surface (see [`crate::controlplane_records`]).

#![cfg(feature = "distributed")]

use std::ffi::{CString, c_char};
use std::os::raw::c_void;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_controlplane::protocol::Assignment;
use blazen_controlplane::worker::{AssignmentContext, AssignmentFailure};
use blazen_controlplane::{
    AssignmentHandler, Client as InnerClient, Worker as InnerWorker,
    WorkerConfig as InnerWorkerConfig,
};
use blazen_core::distributed::{
    AdmissionMode, OrchestratorClient, SubmitWorkflowRequest, WorkerCapability,
};
use futures_util::StreamExt;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use blazen_uniffi::errors::BlazenError as InnerError;

use crate::controlplane_records::{BlazenRunStateSnapshot, BlazenWorkerInfoList};
use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::cstr_to_str;

// ===========================================================================
// Shared error-out helpers (mirror peer.rs)
// ===========================================================================

/// Writes `e` to the out-param if non-null and returns `-1`.
///
/// # Safety
///
/// `out_err` must be null OR a valid destination for a single
/// `*mut BlazenError` write.
unsafe fn write_error(out_err: *mut *mut BlazenError, e: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: caller upholds the out-param contract.
        unsafe {
            *out_err = BlazenError::from(e).into_ptr();
        }
    }
    -1
}

/// Writes a synthesised `Internal` error and returns `-1`.
///
/// # Safety
///
/// Same as [`write_error`].
unsafe fn write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded.
    unsafe {
        write_error(
            out_err,
            InnerError::Internal {
                message: msg.into(),
            },
        )
    }
}

// ===========================================================================
// Helpers — parse JSON inputs into typed values
// ===========================================================================

/// Decode a NUL-terminated UTF-8 JSON array into `Vec<String>`. Returns
/// `None` on null, non-UTF-8, malformed JSON, or non-array shapes.
///
/// # Safety
///
/// `ptr` must be null OR a NUL-terminated UTF-8 buffer that remains
/// valid for the duration of the call.
unsafe fn parse_json_string_array(ptr: *const c_char) -> Option<Vec<String>> {
    if ptr.is_null() {
        return Some(Vec::new());
    }
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let s = unsafe { cstr_to_str(ptr) }?;
    let value: serde_json::Value = serde_json::from_str(s).ok()?;
    match value {
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let serde_json::Value::String(s) = item else {
                    return None;
                };
                out.push(s);
            }
            Some(out)
        }
        serde_json::Value::Null => Some(Vec::new()),
        _ => None,
    }
}

/// Decode a NUL-terminated UTF-8 JSON array of
/// `{ "kind": "...", "version": <u32> }` objects into
/// `Vec<WorkerCapability>`. Treats null / empty / null-value as an
/// empty list.
///
/// # Safety
///
/// Same as [`parse_json_string_array`].
unsafe fn parse_json_capabilities(ptr: *const c_char) -> Option<Vec<WorkerCapability>> {
    if ptr.is_null() {
        return Some(Vec::new());
    }
    // SAFETY: caller upholds the contract.
    let s = unsafe { cstr_to_str(ptr) }?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Some(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(s).ok()?;
    let items = match value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Null => return Some(Vec::new()),
        _ => return None,
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let obj = item.as_object()?;
        let kind = obj.get("kind")?.as_str()?.to_owned();
        let version = obj.get("version").and_then(serde_json::Value::as_u64)?;
        let version = u32::try_from(version).ok()?;
        out.push(WorkerCapability { kind, version });
    }
    Some(out)
}

/// Parse a UUID-shaped string. Returns `None` on null, non-UTF-8, or
/// malformed UUID.
///
/// # Safety
///
/// Same as [`parse_json_string_array`].
unsafe fn parse_uuid(ptr: *const c_char) -> Option<Uuid> {
    // SAFETY: forwarded.
    let s = unsafe { cstr_to_str(ptr) }?;
    Uuid::parse_str(s).ok()
}

// ===========================================================================
// BlazenControlPlaneClient
// ===========================================================================

/// Opaque wrapper around [`blazen_controlplane::Client`]. The inner
/// [`Client`] is cheaply cloneable (it holds an `Arc<Mutex<...>>`
/// internally) so multiple cabi calls on the same handle can run
/// concurrently.
pub struct BlazenControlPlaneClient {
    inner: InnerClient,
}

/// Synchronously open a connection to the control plane at `endpoint`.
/// Blocks the calling thread on the cabi tokio runtime. Returns `0` on
/// success (writing a caller-owned `*mut BlazenControlPlaneClient` to
/// `out_client`) or `-1` on failure (writing the inner error to
/// `out_err`).
///
/// `endpoint` is a gRPC URI such as `"http://cp.example.com:7445"` or
/// `"https://cp.example.com"`. TLS is selected automatically by the
/// underlying tonic Endpoint based on the URI scheme; the cabi surface
/// does not currently expose explicit TLS configuration — that landed
/// only in the higher-level `UniFFI` / `PyO3` / `napi` bindings.
///
/// # Safety
///
/// `endpoint` must be a valid NUL-terminated UTF-8 buffer that remains
/// live for the duration of the call. `out_client` is null OR a
/// destination for one `*mut BlazenControlPlaneClient` write. `out_err`
/// is null OR a destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_connect_blocking(
    endpoint: *const c_char,
    out_client: *mut *mut BlazenControlPlaneClient,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if endpoint.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `endpoint`.
    let endpoint = match unsafe { cstr_to_str(endpoint) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "endpoint not valid UTF-8") },
    };
    match runtime().block_on(async move { InnerClient::connect(endpoint, None).await }) {
        Ok(client) => {
            if !out_client.is_null() {
                // SAFETY: out_client upholds the function contract.
                unsafe {
                    *out_client =
                        Box::into_raw(Box::new(BlazenControlPlaneClient { inner: client }));
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Opens a connection asynchronously, returning an opaque future
/// handle. Resolves to `*mut BlazenControlPlaneClient` — pop with
/// [`blazen_future_take_controlplane_client`].
///
/// Returns null on null input or non-UTF-8 endpoint.
///
/// # Safety
///
/// `endpoint` must be a valid NUL-terminated UTF-8 buffer that remains
/// live for the duration of the call (its contents are copied before
/// this function returns).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_connect(
    endpoint: *const c_char,
) -> *mut BlazenFuture {
    if endpoint.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `endpoint`.
    let endpoint = match unsafe { cstr_to_str(endpoint) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    BlazenFuture::spawn(async move {
        InnerClient::connect(endpoint, None)
            .await
            .map(|client| BlazenControlPlaneClient { inner: client })
            .map_err(InnerError::from)
    })
}

/// Frees a `BlazenControlPlaneClient` handle. No-op on null.
///
/// # Safety
///
/// `client` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_free(client: *mut BlazenControlPlaneClient) {
    if client.is_null() {
        return;
    }
    // SAFETY: caller upholds the Box::into_raw provenance contract.
    drop(unsafe { Box::from_raw(client) });
}

// ---------------------------------------------------------------------------
// submit_workflow
// ---------------------------------------------------------------------------

/// Build a [`SubmitWorkflowRequest`] from the cabi inputs. Returns
/// the request on success, or a string describing the JSON parse
/// failure. Callers wrap the error in `InnerError::Internal` on the
/// failure path — keeping `String` here (instead of `InnerError`)
/// avoids the `result_large_err` clippy lint without forcing a
/// `Box<InnerError>` indirection that the rest of the cabi avoids.
fn build_submit_request(
    workflow_name: String,
    input_json: Option<&str>,
    required_tags: Vec<String>,
    wait_for_worker: bool,
) -> Result<SubmitWorkflowRequest, String> {
    let input = match input_json {
        Some(s) => {
            serde_json::from_str(s).map_err(|e| format!("input_json not valid JSON: {e}"))?
        }
        None => serde_json::Value::Null,
    };
    Ok(SubmitWorkflowRequest {
        workflow_name,
        workflow_version: None,
        input,
        required_tags,
        idempotency_key: None,
        deadline_ms: None,
        wait_for_worker,
        resource_hint: None,
    })
}

/// Synchronously submit a workflow run. Returns `0` on success
/// (writing a caller-owned `*mut BlazenRunStateSnapshot` to
/// `out_snapshot`) or `-1` on failure.
///
/// `required_tags_json` is a JSON array of `key=value` strings; pass
/// null or `"[]"` for no tag requirements.
///
/// # Safety
///
/// `client` must be a valid pointer to a `BlazenControlPlaneClient`.
/// `workflow_name` must be a valid NUL-terminated UTF-8 buffer.
/// `input_json` and `required_tags_json` must each be null OR a valid
/// NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_submit_workflow_blocking(
    client: *const BlazenControlPlaneClient,
    workflow_name: *const c_char,
    input_json: *const c_char,
    required_tags_json: *const c_char,
    wait_for_worker: bool,
    out_snapshot: *mut *mut BlazenRunStateSnapshot,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if client.is_null() || workflow_name.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `workflow_name`.
    let workflow_name = match unsafe { cstr_to_str(workflow_name) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => {
            return unsafe { write_internal_error(out_err, "workflow_name not valid UTF-8") };
        }
    };
    // SAFETY: caller upholds NUL + lifetime on `input_json` (may be null).
    let input_borrow = unsafe { cstr_to_str(input_json) };
    // SAFETY: caller upholds NUL + lifetime on `required_tags_json` (may be null).
    let Some(required_tags) = (unsafe { parse_json_string_array(required_tags_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(out_err, "required_tags_json is not a JSON string array")
        };
    };
    let request =
        match build_submit_request(workflow_name, input_borrow, required_tags, wait_for_worker) {
            Ok(r) => r,
            // SAFETY: out_err upholds the function contract.
            Err(msg) => return unsafe { write_internal_error(out_err, &msg) },
        };
    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.submit_workflow(request).await }) {
        Ok(snap) => {
            if !out_snapshot.is_null() {
                // SAFETY: caller-supplied out-param.
                unsafe {
                    *out_snapshot = BlazenRunStateSnapshot::from(snap).into_ptr();
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async submit. Returns a future that resolves to
/// `*mut BlazenRunStateSnapshot`. Pop with
/// [`blazen_future_take_run_state_snapshot`].
///
/// # Safety
///
/// Same as [`blazen_controlplane_client_submit_workflow_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_submit_workflow(
    client: *const BlazenControlPlaneClient,
    workflow_name: *const c_char,
    input_json: *const c_char,
    required_tags_json: *const c_char,
    wait_for_worker: bool,
) -> *mut BlazenFuture {
    if client.is_null() || workflow_name.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `workflow_name`.
    let workflow_name = match unsafe { cstr_to_str(workflow_name) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: caller upholds NUL + lifetime on `input_json` (may be null).
    let input_owned = unsafe { cstr_to_str(input_json) }.map(str::to_owned);
    // SAFETY: caller upholds NUL + lifetime on `required_tags_json` (may be null).
    let Some(required_tags) = (unsafe { parse_json_string_array(required_tags_json) }) else {
        return std::ptr::null_mut();
    };
    let inner = client.inner.clone();
    BlazenFuture::spawn(async move {
        let request = build_submit_request(
            workflow_name,
            input_owned.as_deref(),
            required_tags,
            wait_for_worker,
        )
        .map_err(|message| InnerError::Internal { message })?;
        let snap = inner
            .submit_workflow(request)
            .await
            .map_err(InnerError::from)?;
        Ok(snap)
    })
}

// ---------------------------------------------------------------------------
// cancel_workflow
// ---------------------------------------------------------------------------

/// Synchronously cancel an in-flight run. Returns `0` on success and a
/// fresh `BlazenRunStateSnapshot`; `-1` on failure with `out_err`.
///
/// # Safety
///
/// `client` must be a valid pointer. `run_id` must be a valid
/// NUL-terminated UTF-8 buffer (a hyphenated UUID rendering).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_cancel_workflow_blocking(
    client: *const BlazenControlPlaneClient,
    run_id: *const c_char,
    out_snapshot: *mut *mut BlazenRunStateSnapshot,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if client.is_null() || run_id.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `run_id`.
    let Some(uuid) = (unsafe { parse_uuid(run_id) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "run_id is not a valid UUID") };
    };
    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.cancel_workflow(uuid).await }) {
        Ok(snap) => {
            if !out_snapshot.is_null() {
                // SAFETY: out_snapshot upholds the function contract.
                unsafe {
                    *out_snapshot = BlazenRunStateSnapshot::from(snap).into_ptr();
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async cancel. Resolves to `*mut BlazenRunStateSnapshot`; pop with
/// [`blazen_future_take_run_state_snapshot`].
///
/// # Safety
///
/// Same as [`blazen_controlplane_client_cancel_workflow_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_cancel_workflow(
    client: *const BlazenControlPlaneClient,
    run_id: *const c_char,
) -> *mut BlazenFuture {
    if client.is_null() || run_id.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `run_id`.
    let Some(uuid) = (unsafe { parse_uuid(run_id) }) else {
        return std::ptr::null_mut();
    };
    let inner = client.inner.clone();
    BlazenFuture::spawn(async move { inner.cancel_workflow(uuid).await.map_err(InnerError::from) })
}

// ---------------------------------------------------------------------------
// describe_workflow
// ---------------------------------------------------------------------------

/// Synchronously describe a run. Same shape as cancel.
///
/// # Safety
///
/// Same as [`blazen_controlplane_client_cancel_workflow_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_describe_workflow_blocking(
    client: *const BlazenControlPlaneClient,
    run_id: *const c_char,
    out_snapshot: *mut *mut BlazenRunStateSnapshot,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if client.is_null() || run_id.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `run_id`.
    let Some(uuid) = (unsafe { parse_uuid(run_id) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "run_id is not a valid UUID") };
    };
    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.describe_workflow(uuid).await }) {
        Ok(snap) => {
            if !out_snapshot.is_null() {
                // SAFETY: out_snapshot upholds the function contract.
                unsafe {
                    *out_snapshot = BlazenRunStateSnapshot::from(snap).into_ptr();
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async describe. Resolves to `*mut BlazenRunStateSnapshot`.
///
/// # Safety
///
/// Same as [`blazen_controlplane_client_describe_workflow_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_describe_workflow(
    client: *const BlazenControlPlaneClient,
    run_id: *const c_char,
) -> *mut BlazenFuture {
    if client.is_null() || run_id.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `run_id`.
    let Some(uuid) = (unsafe { parse_uuid(run_id) }) else {
        return std::ptr::null_mut();
    };
    let inner = client.inner.clone();
    BlazenFuture::spawn(async move {
        inner
            .describe_workflow(uuid)
            .await
            .map_err(InnerError::from)
    })
}

// ---------------------------------------------------------------------------
// list_workers
// ---------------------------------------------------------------------------

/// Synchronously list connected workers. Returns `0` on success
/// (writing a caller-owned `*mut BlazenWorkerInfoList` to `out_list`).
///
/// # Safety
///
/// `client` must be a valid pointer. `out_list` is null OR a valid
/// destination for one `*mut BlazenWorkerInfoList` write. `out_err`
/// upholds the standard out-pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_list_workers_blocking(
    client: *const BlazenControlPlaneClient,
    out_list: *mut *mut BlazenWorkerInfoList,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if client.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.list_workers().await }) {
        Ok(workers) => {
            if !out_list.is_null() {
                // SAFETY: out_list upholds the function contract.
                unsafe {
                    *out_list = BlazenWorkerInfoList::from(workers).into_ptr();
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async list. Resolves to `*mut BlazenWorkerInfoList`; pop with
/// [`blazen_future_take_worker_info_list`].
///
/// # Safety
///
/// `client` must be a valid pointer to a `BlazenControlPlaneClient`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_list_workers(
    client: *const BlazenControlPlaneClient,
) -> *mut BlazenFuture {
    if client.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    let inner = client.inner.clone();
    BlazenFuture::spawn(async move { inner.list_workers().await.map_err(InnerError::from) })
}

// ---------------------------------------------------------------------------
// drain_worker
// ---------------------------------------------------------------------------

/// Synchronously drain a worker. Returns `0` on success / `-1` on
/// failure.
///
/// # Safety
///
/// `client` must be a valid pointer. `node_id` must be a valid
/// NUL-terminated UTF-8 buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_drain_worker_blocking(
    client: *const BlazenControlPlaneClient,
    node_id: *const c_char,
    immediate: bool,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if client.is_null() || node_id.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `node_id`.
    let node_id = match unsafe { cstr_to_str(node_id) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "node_id not valid UTF-8") },
    };
    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.drain_worker(node_id, immediate).await }) {
        Ok(()) => 0,
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async drain. Resolves to unit; pop with `blazen_future_take_unit`
/// (defined in `persist.rs`).
///
/// # Safety
///
/// Same as [`blazen_controlplane_client_drain_worker_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_drain_worker(
    client: *const BlazenControlPlaneClient,
    node_id: *const c_char,
    immediate: bool,
) -> *mut BlazenFuture {
    if client.is_null() || node_id.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `node_id`.
    let node_id = match unsafe { cstr_to_str(node_id) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    let inner = client.inner.clone();
    BlazenFuture::spawn(async move {
        inner
            .drain_worker(node_id, immediate)
            .await
            .map_err(InnerError::from)
    })
}

// ===========================================================================
// AssignmentHandler vtable
// ===========================================================================

/// Vtable a foreign caller fills in to implement an assignment handler.
///
/// Every function pointer is invoked from inside the cabi tokio
/// runtime. The foreign side is responsible for thread-safety: Ruby's
/// `ffi` gem automatically reacquires the GVL for declared callback
/// signatures, so a single Ruby instance can safely back multiple
/// concurrent invocations.
///
/// ## Ownership
///
/// - `user_data` is owned by the vtable. The cabi takes responsibility
///   for releasing it via `drop_user_data` exactly once, when the
///   wrapping `CAssignmentHandler` drops.
/// - The `handle` callback receives three caller-owned NUL-terminated
///   UTF-8 strings (`run_id`, `workflow_name`, `input_json`). The
///   callback MUST NOT free them — the cabi frees them after the
///   callback returns.
/// - On success (`return == 0`), the callback writes a caller-owned
///   `*mut c_char` (heap-allocated UTF-8 JSON, freeable via
///   [`crate::string::blazen_string_free`]) into `out_json`. Pass
///   `null` to report the JSON `null` value.
/// - On failure (`return != 0`), the callback writes a caller-owned
///   `*mut BlazenError` into `out_err`. The cabi reclaims it.
#[repr(C)]
pub struct BlazenAssignmentHandlerVTable {
    /// Foreign-side context pointer handed back to every callback.
    pub user_data: *mut c_void,
    /// Release `user_data`. Called exactly once on drop.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),
    /// Run an assignment. See struct-level docs for ownership.
    pub handle: extern "C" fn(
        user_data: *mut c_void,
        run_id: *const c_char,
        workflow_name: *const c_char,
        input_json: *const c_char,
        out_json: *mut *mut c_char,
        out_err: *mut *mut BlazenError,
    ) -> i32,
    /// Notify the handler that the named run has been cancelled.
    /// `run_id` is borrowed for the call's duration.
    pub on_cancel: extern "C" fn(user_data: *mut c_void, run_id: *const c_char),
    /// Notify the handler that the worker has been drained. `immediate`
    /// = `true` aborts in-flight assignments; `false` waits for them.
    pub on_drain: extern "C" fn(user_data: *mut c_void, immediate: bool),
}

// SAFETY: the foreign side guarantees thread-safe access to `user_data`
// and the function pointers (Ruby's `ffi` gem reacquires the GVL). The
// fn pointers themselves are `extern "C" fn`, which is `Copy + Send +
// Sync`.
unsafe impl Send for BlazenAssignmentHandlerVTable {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for BlazenAssignmentHandlerVTable {}

/// Rust-side trampoline wrapping a foreign
/// [`BlazenAssignmentHandlerVTable`]. Implements
/// [`AssignmentHandler`] by calling into the vtable's function
/// pointers from a `spawn_blocking` task (so the callback never blocks
/// the tokio runtime).
struct CAssignmentHandler {
    vtable: BlazenAssignmentHandlerVTable,
}

impl Drop for CAssignmentHandler {
    fn drop(&mut self) {
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

#[async_trait]
impl AssignmentHandler for CAssignmentHandler {
    async fn handle(
        &self,
        assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        // Decode the assignment's binary input_json into a UTF-8 string
        // so the foreign callback can parse it. Any error here surfaces
        // as `AssignmentFailure` since the worker can't run an
        // assignment whose input isn't valid JSON.
        let run_id = match CString::new(assignment.run_id.to_string()) {
            Ok(c) => c,
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "run_id has interior NUL: {e}"
                )));
            }
        };
        let workflow_name = match CString::new(assignment.workflow_name.clone()) {
            Ok(c) => c,
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "workflow_name has interior NUL: {e}"
                )));
            }
        };
        // Re-encode the wire `Vec<u8>` JSON payload to a UTF-8 string.
        // Failed UTF-8 means the wire payload is malformed; treat as
        // assignment failure.
        let input_str = match std::str::from_utf8(&assignment.input_json) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "input_json not valid UTF-8: {e}"
                )));
            }
        };
        let input_json = match CString::new(input_str) {
            Ok(c) => c,
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "input_json has interior NUL: {e}"
                )));
            }
        };

        // Copy the vtable fields out so the spawn_blocking closure is
        // 'static + Send. Cast through usize for the pointer fields.
        let user_data_addr = self.vtable.user_data as usize;
        let handle_fn = self.vtable.handle;

        // SAFETY: foreign side guarantees thread-safe access to
        // user_data (see vtable docs). The handle function pointer is
        // `extern "C" fn`, which is `Copy + Send + Sync`.
        //
        // The result tuple is shuttled back through usize because raw
        // `*mut c_char` / `*mut BlazenError` are `!Send`. Caller
        // re-casts after the spawn_blocking returns; the pointer
        // provenance round-trip through `usize` is sound for FFI
        // pointers minted by the foreign side (no aliased Rust
        // references exist).
        let join = tokio::task::spawn_blocking(move || -> (i32, usize, usize) {
            let user_data = user_data_addr as *mut c_void;
            let mut out_json: *mut c_char = std::ptr::null_mut();
            let mut out_err: *mut BlazenError = std::ptr::null_mut();
            let rc = handle_fn(
                user_data,
                run_id.as_ptr(),
                workflow_name.as_ptr(),
                input_json.as_ptr(),
                &raw mut out_json,
                &raw mut out_err,
            );
            (rc, out_json as usize, out_err as usize)
        })
        .await;

        let (rc, out_json, out_err) = match join {
            Ok((rc, j, e)) => (rc, j as *mut c_char, e as *mut BlazenError),
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "assignment handler task panicked: {e}"
                )));
            }
        };

        if rc != 0 {
            let msg = if out_err.is_null() {
                "assignment handler returned non-zero without setting out_err".to_string()
            } else {
                // SAFETY: per the vtable contract on a failure return
                // (rc != 0), the foreign side wrote a valid pointer
                // previously produced by the cabi error surface.
                let owned = unsafe { Box::from_raw(out_err) };
                owned.inner.to_string()
            };
            return Err(AssignmentFailure::new(msg));
        }

        if out_json.is_null() {
            return Ok(serde_json::Value::Null);
        }

        // SAFETY: per the vtable contract on a success return, the
        // foreign side wrote a heap-allocated NUL-terminated UTF-8
        // buffer previously produced by `alloc_cstring` /
        // `CString::into_raw`. Reclaim it via `CString::from_raw` so
        // the allocation is released after we parse the JSON.
        let owned = unsafe { CString::from_raw(out_json) };
        let s = match owned.into_string() {
            Ok(s) => s,
            Err(e) => {
                return Err(AssignmentFailure::new(format!(
                    "assignment handler returned non-UTF-8 output: {e}"
                )));
            }
        };
        serde_json::from_str(&s).map_err(|e| {
            AssignmentFailure::new(format!("assignment handler returned non-JSON output: {e}"))
        })
    }

    async fn on_cancel(&self, run_id: Uuid) {
        let cstr = match CString::new(run_id.to_string()) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, %run_id, "on_cancel: run_id has interior NUL");
                return;
            }
        };
        let user_data_addr = self.vtable.user_data as usize;
        let on_cancel_fn = self.vtable.on_cancel;
        // SAFETY: see CAssignmentHandler::handle.
        let _ = tokio::task::spawn_blocking(move || {
            let user_data = user_data_addr as *mut c_void;
            on_cancel_fn(user_data, cstr.as_ptr());
        })
        .await;
    }

    async fn on_drain(&self, immediate: bool) {
        let user_data_addr = self.vtable.user_data as usize;
        let on_drain_fn = self.vtable.on_drain;
        // SAFETY: see CAssignmentHandler::handle.
        let _ = tokio::task::spawn_blocking(move || {
            let user_data = user_data_addr as *mut c_void;
            on_drain_fn(user_data, immediate);
        })
        .await;
    }
}

// ===========================================================================
// BlazenControlPlaneWorker
// ===========================================================================

/// Opaque wrapper around [`blazen_controlplane::Worker`].
///
/// `Worker::run` consumes the value by move, so the inner slot is
/// `Option<InnerHolder>` and goes `None` on the first successful `run`
/// call. The `Mutex` lets `shutdown` inspect the slot without
/// requiring `&mut`.
///
/// Once `run` has consumed the worker, `shutdown` becomes a no-op —
/// the standard way to terminate the worker after `run` is in flight
/// is to drop the surrounding future (the bidi session loop honours
/// cancellation at every iteration).
pub struct BlazenControlPlaneWorker {
    inner: Arc<Mutex<Option<InnerHolder>>>,
}

/// Synchronously construct (and validate) a new worker. Does NOT open
/// a network connection — that happens inside
/// [`blazen_controlplane_worker_run_blocking`] /
/// [`blazen_controlplane_worker_run`], which lets the retry policy
/// cover the initial connect and reconnects uniformly.
///
/// `capabilities_json` is a JSON array of `{ "kind": "<str>",
/// "version": <u32> }` objects. Pass null, `"null"`, or `"[]"` to
/// advertise no capabilities. `admission_mode` selects the worker's
/// admission strategy:
/// - `0` = `Fixed` with `max_in_flight = admission_param` (or 1 if
///   `admission_param == 0`).
/// - `1` = `VramBudget` with `max_vram_mb = admission_param`.
/// - `2` = `Reactive`.
///
/// Any other value falls back to `Fixed { max_in_flight: 1 }`.
///
/// `tags_json` is an optional JSON object mapping `key` -> `value`
/// strings. Pass null for no tags.
///
/// On success (`return == 0`), writes a caller-owned
/// `*mut BlazenControlPlaneWorker` to `out_worker`. On failure,
/// writes an error to `out_err` AND releases `vtable.user_data` via
/// `vtable.drop_user_data` — the foreign caller MUST NOT free it
/// themselves on either path.
///
/// # Safety
///
/// `endpoint`, `node_id` must be valid NUL-terminated UTF-8 buffers.
/// `capabilities_json` and `tags_json` are null OR valid NUL-terminated
/// UTF-8 buffers. `vtable.user_data` and `vtable.drop_user_data` /
/// `vtable.handle` / `vtable.on_cancel` / `vtable.on_drain` must form a
/// coherent thread-safe vtable (see [`BlazenAssignmentHandlerVTable`]
/// docs).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_worker_new_blocking(
    endpoint: *const c_char,
    node_id: *const c_char,
    capabilities_json: *const c_char,
    tags_json: *const c_char,
    admission_mode: u32,
    admission_param: u64,
    vtable: BlazenAssignmentHandlerVTable,
    out_worker: *mut *mut BlazenControlPlaneWorker,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // Helper: release vtable's user_data on early-return paths so the
    // ownership-transfer contract documented above is honored.
    let release_vtable = || {
        (vtable.drop_user_data)(vtable.user_data);
    };

    if endpoint.is_null() || node_id.is_null() {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }

    // SAFETY: caller upholds NUL + lifetime on `endpoint`.
    let Some(endpoint) = (unsafe { cstr_to_str(endpoint) }).map(str::to_owned) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "endpoint not valid UTF-8") };
    };
    // SAFETY: caller upholds NUL + lifetime on `node_id`.
    let Some(node_id) = (unsafe { cstr_to_str(node_id) }).map(str::to_owned) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "node_id not valid UTF-8") };
    };
    // SAFETY: caller upholds NUL + lifetime on `capabilities_json` (may be null).
    let Some(capabilities) = (unsafe { parse_json_capabilities(capabilities_json) }) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(
                out_err,
                "capabilities_json must be a JSON array of {kind, version}",
            )
        };
    };
    // SAFETY: caller upholds NUL + lifetime on `tags_json` (may be null).
    let Some(tags) = (unsafe { parse_tags_json(tags_json) }) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(out_err, "tags_json must be a JSON object of string->string")
        };
    };

    let admission = build_admission(admission_mode, admission_param);

    let mut config = InnerWorkerConfig::new(endpoint, node_id);
    for cap in capabilities {
        config = config.with_capability(cap);
    }
    for (k, v) in tags {
        config = config.with_tag(k, v);
    }
    config = config.with_admission(admission);

    let worker_inner = match InnerWorker::connect(config) {
        Ok(w) => w,
        Err(e) => {
            release_vtable();
            // SAFETY: out_err upholds the function contract.
            return unsafe { write_error(out_err, InnerError::from(e)) };
        }
    };

    // Worker accepted the config — store both the worker and the
    // handler (which owns the vtable) so `run` can move them out
    // together.
    let handler = Arc::new(CAssignmentHandler { vtable });
    let worker = BlazenControlPlaneWorker {
        inner: Arc::new(Mutex::new(Some(InnerHolder {
            worker: worker_inner,
            handler,
        }))),
    };

    if !out_worker.is_null() {
        // SAFETY: out_worker upholds the function contract.
        unsafe {
            *out_worker = Box::into_raw(Box::new(worker));
        }
    }
    0
}

/// Pairs a `Worker` with the handler it will run with. Sits inside the
/// `BlazenControlPlaneWorker`'s `Option` slot so `run` can take both
/// out atomically.
struct InnerHolder {
    worker: InnerWorker,
    handler: Arc<CAssignmentHandler>,
}

/// Synchronously drive the worker until shutdown / drain / retry
/// exhaustion. Consumes the inner worker; calling `run` twice returns
/// `-1` with an Internal error.
///
/// # Safety
///
/// `worker` must be a valid pointer to a `BlazenControlPlaneWorker`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_worker_run_blocking(
    worker: *const BlazenControlPlaneWorker,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if worker.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null worker pointer") };
    }
    // SAFETY: caller has guaranteed `worker` is a live pointer.
    let worker = unsafe { &*worker };
    let taken = worker.inner.lock().take();
    let Some(holder) = taken else {
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(out_err, "worker.run already called; create a fresh worker")
        };
    };
    let InnerHolder { worker: w, handler } = holder;
    match runtime().block_on(async move { w.run(HandlerArc(handler)).await }) {
        Ok(()) => 0,
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async variant of `run_blocking`. Resolves to unit; pop with
/// `blazen_future_take_unit`. Returns null on null input or if `run`
/// has already been called.
///
/// # Safety
///
/// Same as [`blazen_controlplane_worker_run_blocking`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_worker_run(
    worker: *const BlazenControlPlaneWorker,
) -> *mut BlazenFuture {
    if worker.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `worker` is a live pointer.
    let worker = unsafe { &*worker };
    let Some(holder) = worker.inner.lock().take() else {
        return std::ptr::null_mut();
    };
    let InnerHolder { worker: w, handler } = holder;
    BlazenFuture::spawn(async move { w.run(HandlerArc(handler)).await.map_err(InnerError::from) })
}

/// Signal the worker to stop. No-op if `run` hasn't been called yet
/// (the worker's shutdown hook is bound to the inner `Worker` which
/// is consumed on the first `run` call, so post-`run` shutdown flows
/// through the worker's own internal cancellation tokens).
///
/// Idempotent.
///
/// # Safety
///
/// `worker` must be null OR a valid pointer to a
/// `BlazenControlPlaneWorker`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_worker_shutdown(
    worker: *const BlazenControlPlaneWorker,
) {
    if worker.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `worker` is a live pointer.
    let worker = unsafe { &*worker };
    let guard = worker.inner.lock();
    if let Some(holder) = guard.as_ref() {
        holder.worker.shutdown();
    }
}

/// Frees a `BlazenControlPlaneWorker` handle. If `run` was never
/// called, the contained worker config is dropped here (which in
/// turn releases the assignment-handler vtable's `user_data`).
///
/// # Safety
///
/// `worker` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_worker_free(worker: *mut BlazenControlPlaneWorker) {
    if worker.is_null() {
        return;
    }
    // SAFETY: caller upholds Box::into_raw provenance.
    drop(unsafe { Box::from_raw(worker) });
}

// ---------------------------------------------------------------------------
// AssignmentHandler trait pass-through for Arc<CAssignmentHandler>.
// `Worker::run<H>` requires `H: AssignmentHandler`, where the trait is
// not implemented for `Arc<H>` upstream. Wrap the Arc in a newtype so
// we can implement the trait once and dispatch through to the inner
// handler. The inner handler holds the vtable's user_data via Drop —
// dropping the last Arc reclaims it through `CAssignmentHandler::drop`.
// ---------------------------------------------------------------------------

struct HandlerArc(Arc<CAssignmentHandler>);

#[async_trait]
impl AssignmentHandler for HandlerArc {
    async fn handle(
        &self,
        assignment: Assignment,
        ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        self.0.handle(assignment, ctx).await
    }

    async fn on_cancel(&self, run_id: Uuid) {
        self.0.on_cancel(run_id).await;
    }

    async fn on_drain(&self, immediate: bool) {
        self.0.on_drain(immediate).await;
    }
}

// ---------------------------------------------------------------------------
// Helpers for new_blocking
// ---------------------------------------------------------------------------

/// Decode an optional JSON object string into a `Vec<(key, value)>` of
/// tags. Returns `Some(empty)` for null / null-literal / "{}" /
/// missing input; `None` only on a malformed object or a non-object
/// JSON value.
///
/// # Safety
///
/// Same as [`parse_json_string_array`].
unsafe fn parse_tags_json(ptr: *const c_char) -> Option<Vec<(String, String)>> {
    if ptr.is_null() {
        return Some(Vec::new());
    }
    // SAFETY: caller upholds the contract.
    let s = unsafe { cstr_to_str(ptr) }?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Some(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(s).ok()?;
    let obj = match value {
        serde_json::Value::Object(obj) => obj,
        serde_json::Value::Null => return Some(Vec::new()),
        _ => return None,
    };
    let mut out = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let s = v.as_str()?.to_owned();
        out.push((k, s));
    }
    Some(out)
}

/// Build an [`AdmissionMode`] from the cabi enum + param pair.
fn build_admission(mode: u32, param: u64) -> AdmissionMode {
    match mode {
        1 => AdmissionMode::VramBudget { max_vram_mb: param },
        2 => AdmissionMode::Reactive,
        _ => {
            let max_in_flight = if param == 0 {
                1
            } else {
                u32::try_from(param).unwrap_or(u32::MAX)
            };
            AdmissionMode::Fixed { max_in_flight }
        }
    }
}

// ===========================================================================
// mTLS — Client / Worker constructors that load PEM files from disk.
// ===========================================================================

/// Read three NUL-terminated UTF-8 PEM file path arguments into owned
/// [`PathBuf`]s, surfacing UTF-8 / null errors via `out_err`. Returns
/// `Some((cert, key, ca))` on success.
///
/// # Safety
///
/// `cert_path`, `key_path`, and `ca_path` must each be a valid
/// NUL-terminated UTF-8 buffer that remains live for the duration of
/// the call.
unsafe fn parse_mtls_paths(
    cert_path: *const c_char,
    key_path: *const c_char,
    ca_path: *const c_char,
    out_err: *mut *mut BlazenError,
) -> Option<(PathBuf, PathBuf, PathBuf)> {
    if cert_path.is_null() || key_path.is_null() || ca_path.is_null() {
        // SAFETY: out_err upholds the function contract.
        unsafe { write_internal_error(out_err, "null pointer argument") };
        return None;
    }
    // SAFETY: caller upholds NUL + lifetime on each path arg.
    let Some(cert) = (unsafe { cstr_to_str(cert_path) }).map(PathBuf::from) else {
        // SAFETY: out_err upholds the function contract.
        unsafe { write_internal_error(out_err, "cert_path not valid UTF-8") };
        return None;
    };
    // SAFETY: caller upholds NUL + lifetime on `key_path`.
    let Some(key) = (unsafe { cstr_to_str(key_path) }).map(PathBuf::from) else {
        // SAFETY: out_err upholds the function contract.
        unsafe { write_internal_error(out_err, "key_path not valid UTF-8") };
        return None;
    };
    // SAFETY: caller upholds NUL + lifetime on `ca_path`.
    let Some(ca) = (unsafe { cstr_to_str(ca_path) }).map(PathBuf::from) else {
        // SAFETY: out_err upholds the function contract.
        unsafe { write_internal_error(out_err, "ca_path not valid UTF-8") };
        return None;
    };
    Some((cert, key, ca))
}

/// Synchronously open an mTLS connection to the control plane at
/// `endpoint`, loading the client identity + CA bundle from PEM files
/// on disk. Same shape as
/// [`blazen_controlplane_client_connect_blocking`].
///
/// # Safety
///
/// `endpoint`, `cert_path`, `key_path`, and `ca_path` must each be a
/// valid NUL-terminated UTF-8 buffer that remains live for the
/// duration of the call. `out_client` is null OR a destination for one
/// `*mut BlazenControlPlaneClient` write. `out_err` is null OR a
/// destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_connect_with_mtls_blocking(
    endpoint: *const c_char,
    cert_path: *const c_char,
    key_path: *const c_char,
    ca_path: *const c_char,
    out_client: *mut *mut BlazenControlPlaneClient,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if endpoint.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller upholds NUL + lifetime on `endpoint`.
    let endpoint = match unsafe { cstr_to_str(endpoint) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "endpoint not valid UTF-8") },
    };
    // SAFETY: caller upholds NUL + lifetime on each PEM-path arg.
    let Some((cert, key, ca)) =
        (unsafe { parse_mtls_paths(cert_path, key_path, ca_path, out_err) })
    else {
        return -1;
    };
    match runtime()
        .block_on(async move { InnerClient::with_mtls(endpoint, &cert, &key, &ca).await })
    {
        Ok(client) => {
            if !out_client.is_null() {
                // SAFETY: out_client upholds the function contract.
                unsafe {
                    *out_client =
                        Box::into_raw(Box::new(BlazenControlPlaneClient { inner: client }));
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Async variant of
/// [`blazen_controlplane_client_connect_with_mtls_blocking`]. Returns a
/// `*mut BlazenFuture` resolving to
/// `*mut BlazenControlPlaneClient` — pop with
/// [`blazen_future_take_controlplane_client`]. Returns null on null
/// input or non-UTF-8 paths.
///
/// # Safety
///
/// Same as
/// [`blazen_controlplane_client_connect_with_mtls_blocking`]; the
/// contents of each path buffer are copied before this function
/// returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_connect_with_mtls(
    endpoint: *const c_char,
    cert_path: *const c_char,
    key_path: *const c_char,
    ca_path: *const c_char,
) -> *mut BlazenFuture {
    if endpoint.is_null() || cert_path.is_null() || key_path.is_null() || ca_path.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller upholds NUL + lifetime on `endpoint`.
    let Some(endpoint) = (unsafe { cstr_to_str(endpoint) }).map(str::to_owned) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds NUL + lifetime on each PEM-path arg.
    let Some(cert) = (unsafe { cstr_to_str(cert_path) }).map(PathBuf::from) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds NUL + lifetime on `key_path`.
    let Some(key) = (unsafe { cstr_to_str(key_path) }).map(PathBuf::from) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds NUL + lifetime on `ca_path`.
    let Some(ca) = (unsafe { cstr_to_str(ca_path) }).map(PathBuf::from) else {
        return std::ptr::null_mut();
    };
    BlazenFuture::spawn(async move {
        InnerClient::with_mtls(endpoint, &cert, &key, &ca)
            .await
            .map(|client| BlazenControlPlaneClient { inner: client })
            .map_err(InnerError::from)
    })
}

/// Synchronously construct (and validate) a new worker with mTLS
/// loaded from PEM files on disk. Same semantics as
/// [`blazen_controlplane_worker_new_blocking`], plus three additional
/// path arguments. The vtable's ownership-transfer contract is
/// unchanged — `vtable.user_data` is released via
/// `vtable.drop_user_data` on every failure path.
///
/// # Safety
///
/// `endpoint`, `node_id`, `cert_path`, `key_path`, and `ca_path` must
/// each be valid NUL-terminated UTF-8 buffers. `capabilities_json` and
/// `tags_json` are null OR valid NUL-terminated UTF-8 buffers. The
/// vtable must satisfy the contracts documented on
/// [`BlazenAssignmentHandlerVTable`].
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_worker_new_with_mtls_blocking(
    endpoint: *const c_char,
    node_id: *const c_char,
    capabilities_json: *const c_char,
    tags_json: *const c_char,
    admission_mode: u32,
    admission_param: u64,
    cert_path: *const c_char,
    key_path: *const c_char,
    ca_path: *const c_char,
    vtable: BlazenAssignmentHandlerVTable,
    out_worker: *mut *mut BlazenControlPlaneWorker,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // Helper: release vtable's user_data on early-return paths so the
    // ownership-transfer contract documented above is honored.
    let release_vtable = || {
        (vtable.drop_user_data)(vtable.user_data);
    };

    if endpoint.is_null() || node_id.is_null() {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }

    // SAFETY: caller upholds NUL + lifetime on `endpoint`.
    let Some(endpoint) = (unsafe { cstr_to_str(endpoint) }).map(str::to_owned) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "endpoint not valid UTF-8") };
    };
    // SAFETY: caller upholds NUL + lifetime on `node_id`.
    let Some(node_id) = (unsafe { cstr_to_str(node_id) }).map(str::to_owned) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "node_id not valid UTF-8") };
    };
    // SAFETY: caller upholds NUL + lifetime on `capabilities_json` (may be null).
    let Some(capabilities) = (unsafe { parse_json_capabilities(capabilities_json) }) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(
                out_err,
                "capabilities_json must be a JSON array of {kind, version}",
            )
        };
    };
    // SAFETY: caller upholds NUL + lifetime on `tags_json` (may be null).
    let Some(tags) = (unsafe { parse_tags_json(tags_json) }) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(out_err, "tags_json must be a JSON object of string->string")
        };
    };
    // SAFETY: caller upholds NUL + lifetime on each PEM-path arg.
    let Some((cert, key, ca)) =
        (unsafe { parse_mtls_paths(cert_path, key_path, ca_path, out_err) })
    else {
        release_vtable();
        return -1;
    };

    let admission = build_admission(admission_mode, admission_param);

    let mut config = InnerWorkerConfig::new(endpoint, node_id);
    for cap in capabilities {
        config = config.with_capability(cap);
    }
    for (k, v) in tags {
        config = config.with_tag(k, v);
    }
    config = config.with_admission(admission);
    config = match config.with_mtls(&cert, &key, &ca) {
        Ok(c) => c,
        Err(e) => {
            release_vtable();
            // SAFETY: out_err upholds the function contract.
            return unsafe { write_error(out_err, InnerError::from(e)) };
        }
    };

    let worker_inner = match InnerWorker::connect(config) {
        Ok(w) => w,
        Err(e) => {
            release_vtable();
            // SAFETY: out_err upholds the function contract.
            return unsafe { write_error(out_err, InnerError::from(e)) };
        }
    };

    let handler = Arc::new(CAssignmentHandler { vtable });
    let worker = BlazenControlPlaneWorker {
        inner: Arc::new(Mutex::new(Some(InnerHolder {
            worker: worker_inner,
            handler,
        }))),
    };

    if !out_worker.is_null() {
        // SAFETY: out_worker upholds the function contract.
        unsafe {
            *out_worker = Box::into_raw(Box::new(worker));
        }
    }
    0
}

// ===========================================================================
// Subscriptions — vtable-based event sink + per-run / fan-out subscribers.
// ===========================================================================

/// Vtable a foreign caller fills in to receive run-event streams.
///
/// Each subscription owns one of these. The subscription pump task
/// invokes the callbacks from a `spawn_blocking` thread, so foreign
/// hosts that need GVL reacquisition (Ruby) get it for free, and
/// hosts that pin to a particular thread (Dart) marshal back through
/// their own runtime adapter.
///
/// ## Ownership
///
/// - `user_data` is owned by the vtable. The subscription releases it
///   via `drop_user_data` exactly once when the wrapping
///   [`BlazenControlPlaneSubscription`] drops.
/// - `run_id`, `event_type`, and `data_json` passed to `on_event` are
///   BORROWED — the callback MUST NOT free them. They remain live for
///   the duration of the call.
/// - `error` passed to `on_error` is BORROWED — same rules.
/// - `on_close` is invoked exactly once when the stream terminates
///   cleanly (server end-of-stream). `on_error` is invoked at most
///   once and replaces `on_close` on the error path. Cancelling /
///   freeing the subscription suppresses both terminal callbacks.
#[repr(C)]
pub struct BlazenRunEventSinkVTable {
    /// Foreign-side context pointer handed back to every callback.
    pub user_data: *mut c_void,
    /// Release `user_data`. Called exactly once when the wrapping
    /// subscription drops.
    pub drop_user_data: unsafe extern "C" fn(user_data: *mut c_void),
    /// Receive one event. All three string arguments are borrowed,
    /// NUL-terminated UTF-8 buffers valid for the duration of the call.
    /// `data_json` is the serialised JSON payload of the event's
    /// `data` field.
    pub on_event: unsafe extern "C" fn(
        user_data: *mut c_void,
        run_id: *const c_char,
        event_type: *const c_char,
        data_json: *const c_char,
        timestamp_ms: u64,
    ),
    /// Terminal callback invoked once on clean stream end.
    pub on_close: unsafe extern "C" fn(user_data: *mut c_void),
    /// Terminal callback invoked once on a stream error. `error` is a
    /// borrowed NUL-terminated UTF-8 buffer.
    pub on_error: unsafe extern "C" fn(user_data: *mut c_void, error: *const c_char),
}

// SAFETY: the foreign side guarantees thread-safe access to
// `user_data` and the function pointers (Ruby's `ffi` gem reacquires
// the GVL, Dart's `NativeCallable.listener` marshals through the
// isolate event loop, native hosts opt in via their own runtime).
unsafe impl Send for BlazenRunEventSinkVTable {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for BlazenRunEventSinkVTable {}

/// Opaque handle returned by
/// [`blazen_controlplane_client_subscribe_run_events`] /
/// [`blazen_controlplane_client_subscribe_all`]. Holds the
/// cancellation token tied to the background pump task and keeps the
/// foreign-supplied vtable alive (so callbacks remain valid until the
/// caller frees the subscription).
pub struct BlazenControlPlaneSubscription {
    cancel: CancellationToken,
}

/// Trampoline owning a foreign [`BlazenRunEventSinkVTable`]. Drops
/// the vtable's `user_data` via `drop_user_data` exactly once when the
/// pump task ends (cleanly, via error, or via cancellation).
struct CRunEventSink {
    vtable: BlazenRunEventSinkVTable,
}

impl Drop for CRunEventSink {
    fn drop(&mut self) {
        // SAFETY: by the vtable contract, `drop_user_data` is the
        // foreign side's release thunk for `user_data` and is safe to
        // call exactly once when the wrapper is destroyed. Each pump
        // task constructs exactly one `CRunEventSink`; this Drop is
        // the only invocation site.
        unsafe {
            (self.vtable.drop_user_data)(self.vtable.user_data);
        }
    }
}

/// Identifies which subscribe RPC the pump task should open. The
/// task holds the [`InnerClient`] clone for the duration of the
/// subscription so the stream's borrow on the gRPC channel is
/// satisfied.
enum SubscribeKind {
    PerRun(Uuid),
    All(Vec<String>),
}

/// Spawn the per-subscription pump task. Owns `client` (so the
/// stream's borrow on its gRPC channel is satisfied for the task's
/// lifetime), opens the subscribe RPC, then pumps events to the sink.
///
/// The initial subscribe error / success status is delivered via
/// `ready_tx` so the caller can surface a failed handshake to the
/// foreign side synchronously. After a successful handshake, terminal
/// events (`on_close` / `on_error`) are dispatched through the sink
/// vtable unless the cancellation token fires first.
fn spawn_event_pump(
    sink: CRunEventSink,
    client: InnerClient,
    kind: SubscribeKind,
    cancel: CancellationToken,
    ready_tx: tokio::sync::oneshot::Sender<Result<(), InnerError>>,
) -> tokio::task::JoinHandle<()> {
    runtime().spawn(async move {
        run_event_pump_inner(sink, client, kind, cancel, ready_tx).await;
    })
}

/// Body of the per-subscription pump task. Split out so the borrow
/// of `client` by the stream is confined to a single scope that
/// ends with the stream's drop — after which `client` may be moved
/// or dropped without conflicting with the (now-dead) borrow.
async fn run_event_pump_inner(
    sink: CRunEventSink,
    client: InnerClient,
    kind: SubscribeKind,
    cancel: CancellationToken,
    ready_tx: tokio::sync::oneshot::Sender<Result<(), InnerError>>,
) {
    let sink = Arc::new(sink);
    let outcome = {
        // Open the upstream stream first. If it fails, surface the
        // error and exit — the sink wrapper drops, releasing
        // `user_data`.
        let mut stream = match open_stream(&client, kind).await {
            Ok(s) => {
                let _ = ready_tx.send(Ok(()));
                s
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        };

        let mut result: Result<(), String> = Ok(());
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => break,
                next = stream.next() => match next {
                    None => break,
                    Some(Ok(event)) => {
                        dispatch_event(Arc::clone(&sink), event).await;
                    }
                    Some(Err(e)) => {
                        result = Err(e.to_string());
                        break;
                    }
                },
            }
        }
        // `stream` (and its borrow on `client`) drops at scope exit.
        result
    };

    // Keep `client` alive until after the borrow has ended; dropping
    // it here releases the gRPC channel.
    drop(client);

    if cancel.is_cancelled() {
        return;
    }
    match outcome {
        Ok(()) => {
            fire_on_close(Arc::clone(&sink)).await;
        }
        Err(msg) => {
            fire_on_error(Arc::clone(&sink), msg).await;
        }
    }
}

/// Open the upstream subscribe RPC, mapping any error into an
/// [`InnerError`] so the pump task can forward it on `ready_tx`.
async fn open_stream(
    client: &InnerClient,
    kind: SubscribeKind,
) -> Result<blazen_core::distributed::RunEventStream<'_>, InnerError> {
    match kind {
        SubscribeKind::PerRun(uuid) => {
            client
                .subscribe_run_events(uuid)
                .await
                .map_err(|e| InnerError::Internal {
                    message: format!("subscribe_run_events failed: {e}"),
                })
        }
        SubscribeKind::All(tags) => {
            client
                .subscribe_all(tags)
                .await
                .map_err(|e| InnerError::Internal {
                    message: format!("subscribe_all failed: {e}"),
                })
        }
    }
}

/// Marshal one `RunEvent` to the foreign `on_event` callback via
/// `spawn_blocking` so a slow callback doesn't block the runtime.
/// Encoding failures (interior NULs, JSON serialisation) are logged
/// but do not abort the stream.
async fn dispatch_event(sink: Arc<CRunEventSink>, event: blazen_core::distributed::RunEvent) {
    let run_id = match CString::new(event.run_id.to_string()) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "subscription on_event: run_id has interior NUL");
            return;
        }
    };
    let event_type = match CString::new(event.event_type.clone()) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "subscription on_event: event_type has interior NUL");
            return;
        }
    };
    let data_str = match serde_json::to_string(&event.data) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "subscription on_event: data JSON encode failed");
            return;
        }
    };
    let data_json = match CString::new(data_str) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "subscription on_event: data_json has interior NUL");
            return;
        }
    };
    let timestamp_ms = event.timestamp_ms;

    let user_data_addr = sink.vtable.user_data as usize;
    let on_event_fn = sink.vtable.on_event;

    // SAFETY: the foreign side guarantees thread-safe access to
    // `user_data` (see vtable docs). The three CStrings live for the
    // duration of the spawn_blocking closure (moved into it). The
    // function pointer is `extern "C" fn`, which is `Copy + Send +
    // Sync`.
    let _ = tokio::task::spawn_blocking(move || {
        let user_data = user_data_addr as *mut c_void;
        // SAFETY: vtable contract — borrowed NUL-terminated UTF-8
        // pointers valid for the duration of the call.
        unsafe {
            on_event_fn(
                user_data,
                run_id.as_ptr(),
                event_type.as_ptr(),
                data_json.as_ptr(),
                timestamp_ms,
            );
        }
    })
    .await;
}

/// Fire the foreign-side `on_close` terminal callback via
/// `spawn_blocking`.
async fn fire_on_close(sink: Arc<CRunEventSink>) {
    let user_data_addr = sink.vtable.user_data as usize;
    let on_close_fn = sink.vtable.on_close;
    // SAFETY: same justification as `dispatch_event`.
    let _ = tokio::task::spawn_blocking(move || {
        let user_data = user_data_addr as *mut c_void;
        // SAFETY: vtable contract.
        unsafe {
            on_close_fn(user_data);
        }
    })
    .await;
}

/// Fire the foreign-side `on_error` terminal callback via
/// `spawn_blocking`, passing a NUL-terminated UTF-8 message.
async fn fire_on_error(sink: Arc<CRunEventSink>, msg: String) {
    let cstr = match CString::new(msg.clone()) {
        Ok(c) => c,
        Err(_) => {
            // Strip interior NULs and try again. If even that fails
            // we substitute a placeholder so the foreign side still
            // gets a terminal signal.
            CString::new(msg.replace('\0', "?"))
                .unwrap_or_else(|_| CString::new("<unrenderable error>").unwrap_or_default())
        }
    };
    let user_data_addr = sink.vtable.user_data as usize;
    let on_error_fn = sink.vtable.on_error;
    // SAFETY: same justification as `dispatch_event`.
    let _ = tokio::task::spawn_blocking(move || {
        let user_data = user_data_addr as *mut c_void;
        // SAFETY: vtable contract.
        unsafe {
            on_error_fn(user_data, cstr.as_ptr());
        }
    })
    .await;
}

/// Subscribe to the event stream for a single run. On success returns
/// `0` and writes a caller-owned `*mut BlazenControlPlaneSubscription`
/// into `out_sub`. The vtable's `user_data` is consumed on every
/// path — released via `vtable.drop_user_data` on any early-return
/// failure, otherwise released when the resulting subscription is
/// freed.
///
/// # Safety
///
/// `client` must be a valid pointer to a `BlazenControlPlaneClient`.
/// `run_id` must be a valid NUL-terminated UTF-8 buffer (a hyphenated
/// UUID rendering). `vtable.user_data` plus the four function
/// pointers must form a coherent thread-safe vtable (see
/// [`BlazenRunEventSinkVTable`] docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_subscribe_run_events(
    client: *const BlazenControlPlaneClient,
    run_id: *const c_char,
    sink: BlazenRunEventSinkVTable,
    out_sub: *mut *mut BlazenControlPlaneSubscription,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // Helper to release `user_data` on every early-return path.
    let release_vtable = || {
        // SAFETY: vtable contract — `drop_user_data` is safe to invoke
        // exactly once on the owned `user_data`.
        unsafe {
            (sink.drop_user_data)(sink.user_data);
        }
    };

    if client.is_null() || run_id.is_null() {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client_ref = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `run_id`.
    let Some(uuid) = (unsafe { parse_uuid(run_id) }) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "run_id is not a valid UUID") };
    };

    // Sink wrapper now owns `user_data`. From here on, drop runs the
    // foreign release thunk.
    let csink = CRunEventSink { vtable: sink };
    let cancel = CancellationToken::new();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

    let inner = client_ref.inner.clone();
    spawn_event_pump(
        csink,
        inner,
        SubscribeKind::PerRun(uuid),
        cancel.clone(),
        ready_tx,
    );

    // Block until the pump task reports the initial subscribe RPC's
    // outcome. The sink wrapper is owned by the spawned task at this
    // point, so even on a handshake failure the foreign `user_data`
    // is released by the task's drop chain.
    match runtime().block_on(ready_rx) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            cancel.cancel();
            // SAFETY: out_err upholds the function contract.
            return unsafe { write_error(out_err, e) };
        }
        Err(_) => {
            cancel.cancel();
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, "subscription pump task dropped before signalling")
            };
        }
    }

    if out_sub.is_null() {
        // Caller didn't provide an out-slot: cancel immediately so the
        // pump task winds down and the sink wrapper drops, releasing
        // `user_data`. This honours the "no leak" contract on degenerate
        // input.
        cancel.cancel();
    } else {
        let sub = BlazenControlPlaneSubscription { cancel };
        // SAFETY: out_sub upholds the function contract.
        unsafe {
            *out_sub = Box::into_raw(Box::new(sub));
        }
    }
    0
}

/// Subscribe to the fan-out event stream across all runs, optionally
/// filtered by tag predicates. On success returns `0` and writes a
/// caller-owned `*mut BlazenControlPlaneSubscription` into `out_sub`.
/// `required_tags_json` is a JSON array of `key=value` strings (pass
/// null or `"[]"` for no filtering).
///
/// # Safety
///
/// `client` must be a valid pointer to a `BlazenControlPlaneClient`.
/// `required_tags_json` is null OR a valid NUL-terminated UTF-8
/// buffer. `vtable.user_data` plus the four function pointers must
/// form a coherent thread-safe vtable (see
/// [`BlazenRunEventSinkVTable`] docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_client_subscribe_all(
    client: *const BlazenControlPlaneClient,
    required_tags_json: *const c_char,
    sink: BlazenRunEventSinkVTable,
    out_sub: *mut *mut BlazenControlPlaneSubscription,
    out_err: *mut *mut BlazenError,
) -> i32 {
    let release_vtable = || {
        // SAFETY: vtable contract.
        unsafe {
            (sink.drop_user_data)(sink.user_data);
        }
    };

    if client.is_null() {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null client pointer") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client_ref = unsafe { &*client };
    // SAFETY: caller upholds NUL + lifetime on `required_tags_json` (may be null).
    let Some(required_tags) = (unsafe { parse_json_string_array(required_tags_json) }) else {
        release_vtable();
        // SAFETY: out_err upholds the function contract.
        return unsafe {
            write_internal_error(out_err, "required_tags_json is not a JSON string array")
        };
    };

    // Sink wrapper now owns `user_data`. From here on, drop runs the
    // foreign release thunk.
    let csink = CRunEventSink { vtable: sink };
    let cancel = CancellationToken::new();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

    let inner = client_ref.inner.clone();
    spawn_event_pump(
        csink,
        inner,
        SubscribeKind::All(required_tags),
        cancel.clone(),
        ready_tx,
    );

    match runtime().block_on(ready_rx) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            cancel.cancel();
            // SAFETY: out_err upholds the function contract.
            return unsafe { write_error(out_err, e) };
        }
        Err(_) => {
            cancel.cancel();
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, "subscription pump task dropped before signalling")
            };
        }
    }

    if out_sub.is_null() {
        cancel.cancel();
    } else {
        let sub = BlazenControlPlaneSubscription { cancel };
        // SAFETY: out_sub upholds the function contract.
        unsafe {
            *out_sub = Box::into_raw(Box::new(sub));
        }
    }
    0
}

/// Cancel an in-flight subscription. Fires the internal cancellation
/// token; the pump task observes it on the next stream poll, exits
/// without firing terminal callbacks, and releases `user_data` via
/// the vtable's `drop_user_data` thunk. Idempotent.
///
/// # Safety
///
/// `sub` must be null OR a valid pointer to a
/// `BlazenControlPlaneSubscription` produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_subscription_cancel(
    sub: *mut BlazenControlPlaneSubscription,
) {
    if sub.is_null() {
        return;
    }
    // SAFETY: caller has guaranteed `sub` is a live pointer.
    let sub = unsafe { &*sub };
    sub.cancel.cancel();
}

/// Free a subscription handle. Cancels the underlying pump task as a
/// side effect (same semantics as
/// [`blazen_controlplane_subscription_cancel`]). No-op on null.
///
/// # Safety
///
/// `sub` must be null OR a pointer previously produced by the cabi
/// control-plane surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_controlplane_subscription_free(
    sub: *mut BlazenControlPlaneSubscription,
) {
    if sub.is_null() {
        return;
    }
    // SAFETY: caller upholds the Box::into_raw provenance contract.
    let owned = unsafe { Box::from_raw(sub) };
    // Make sure the pump task winds down and the vtable's user_data
    // is released even if the caller never cancelled explicitly.
    owned.cancel.cancel();
}

// ===========================================================================
// Typed future-take entry points
// ===========================================================================

/// Pop a `BlazenControlPlaneClient` from `fut`. On success returns `0`
/// and writes a caller-owned `*mut BlazenControlPlaneClient` to `out`;
/// on failure returns `-1` and writes the error to `err`.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_controlplane_client_connect`], not yet freed. `out` / `err`
/// are null OR valid destinations for their respective slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_controlplane_client(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenControlPlaneClient,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<BlazenControlPlaneClient>(fut) } {
        Ok(client) => {
            if !out.is_null() {
                // SAFETY: out upholds the function contract.
                unsafe {
                    *out = Box::into_raw(Box::new(client));
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: err upholds the function contract.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Pop a `BlazenRunStateSnapshot` from `fut`. Same shape as the client
/// take helper.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by one of
/// [`blazen_controlplane_client_submit_workflow`],
/// [`blazen_controlplane_client_cancel_workflow`], or
/// [`blazen_controlplane_client_describe_workflow`], not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_run_state_snapshot(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenRunStateSnapshot,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<blazen_core::distributed::RunStateSnapshot>(fut) } {
        Ok(snap) => {
            if !out.is_null() {
                // SAFETY: out upholds the function contract.
                unsafe {
                    *out = BlazenRunStateSnapshot::from(snap).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: err upholds the function contract.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}

/// Pop a `BlazenWorkerInfoList` from `fut`. Same shape as the snapshot
/// take helper.
///
/// # Safety
///
/// `fut` must be a non-null pointer produced by
/// [`blazen_controlplane_client_list_workers`], not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_take_worker_info_list(
    fut: *mut BlazenFuture,
    out: *mut *mut BlazenWorkerInfoList,
    err: *mut *mut BlazenError,
) -> i32 {
    // SAFETY: caller upholds the live-future-pointer contract.
    match unsafe { BlazenFuture::take_typed::<Vec<blazen_core::distributed::WorkerInfo>>(fut) } {
        Ok(workers) => {
            if !out.is_null() {
                // SAFETY: out upholds the function contract.
                unsafe {
                    *out = BlazenWorkerInfoList::from(workers).into_ptr();
                }
            }
            0
        }
        Err(e) => {
            if !err.is_null() {
                // SAFETY: err upholds the function contract.
                unsafe {
                    *err = BlazenError::from(e).into_ptr();
                }
            }
            -1
        }
    }
}
