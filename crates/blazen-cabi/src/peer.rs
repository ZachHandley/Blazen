//! Distributed-peer opaque objects: `PeerServer` and `PeerClient`. Entire
//! module is gated on the `distributed` feature. Phase R3 Agent D.
//!
//! ## Ownership conventions
//!
//! - [`blazen_peer_server_new`] returns a caller-owned `*mut BlazenPeerServer`.
//!   Release with [`blazen_peer_server_free`]. The handle wraps an `Arc` so
//!   freeing it merely drops the cabi-side ref; if a `serve` task is still
//!   running it keeps its own ref and shuts down on its own terms.
//! - [`blazen_peer_client_connect`] writes a caller-owned `*mut BlazenPeerClient`
//!   into `out_client` on success. Release with [`blazen_peer_client_free`].
//! - The `*_blocking` variants return `0` on success / `-1` on failure with
//!   errors written through `*mut *mut BlazenError` out-params. The
//!   future-returning variants funnel errors and (for `run_remote_workflow`)
//!   the `WorkflowResult` through the typed future-take functions defined in
//!   sibling modules:
//!     - `blazen_peer_server_serve` -> `blazen_future_take_unit`
//!       (defined in `persist.rs`).
//!     - `blazen_peer_client_run_remote_workflow` ->
//!       `blazen_future_take_workflow_result` (defined in `workflow.rs`).
//! - String accessors (`blazen_peer_client_node_id`) return caller-owned heap
//!   strings; free with [`crate::string::blazen_string_free`].

#![cfg(feature = "distributed")]

use std::ffi::c_char;
use std::sync::Arc;

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::peer::{PeerClient as InnerPeerClient, PeerServer as InnerPeerServer};

use crate::error::BlazenError;
use crate::future::BlazenFuture;
use crate::runtime::runtime;
use crate::string::{alloc_cstring, cstr_to_str};
use crate::workflow_records::BlazenWorkflowResult;

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
// String-array helper
// ---------------------------------------------------------------------------

/// Materialise a C `(const char *const *ptrs, size_t count)` pair into a
/// `Vec<String>`. Returns `None` if `ptrs` is null while `count > 0`, or if
/// any entry pointer is null or contains non-UTF-8 bytes.
///
/// # Safety
///
/// `ptrs` must be null (only valid when `count == 0`) OR point to a buffer
/// of at least `count` consecutive `*const c_char` slots, each of which is
/// null OR points to a NUL-terminated UTF-8 buffer that remains valid for
/// the duration of this call.
unsafe fn ptr_array_to_strings(ptrs: *const *const c_char, count: usize) -> Option<Vec<String>> {
    if ptrs.is_null() && count > 0 {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: caller has guaranteed the buffer holds at least `count`
        // entries, so `ptrs.add(i)` is in-bounds. `ptrs.add(i)` itself is a
        // valid pointer-sized read; the *dereferenced* per-entry pointer
        // either is null (handled by `cstr_to_str`) or upholds the
        // NUL-terminated-UTF-8 contract documented above.
        let p = unsafe { *ptrs.add(i) };
        // SAFETY: per-entry pointer satisfies the `cstr_to_str` contract
        // (null OR NUL-terminated UTF-8 buffer live for this call).
        let s = unsafe { cstr_to_str(p) }?;
        out.push(s.to_owned());
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// BlazenPeerServer
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::peer::PeerServer`.
///
/// The inner `Arc` matches the `self: Arc<Self>` shape of the underlying
/// async `serve` method.
pub struct BlazenPeerServer(pub(crate) Arc<InnerPeerServer>);

/// Construct a new peer server with the given UTF-8 `node_id`. Returns null
/// on a null pointer or non-UTF-8 input.
///
/// `node_id` is the stable identifier this server stamps onto every
/// `RemoteRefDescriptor` it returns. Typical values are the hostname or a
/// UUID picked at process startup.
///
/// # Ownership
///
/// Returned pointer is caller-owned. Free with [`blazen_peer_server_free`].
///
/// # Safety
///
/// `node_id` must be null OR a valid NUL-terminated UTF-8 buffer that
/// remains live for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_server_new(node_id: *const c_char) -> *mut BlazenPeerServer {
    // SAFETY: caller upholds the NUL-termination + lifetime contract.
    let Some(node_id) = (unsafe { cstr_to_str(node_id) }) else {
        return std::ptr::null_mut();
    };
    let inner = InnerPeerServer::new(node_id.to_owned());
    Box::into_raw(Box::new(BlazenPeerServer(inner)))
}

/// Synchronously binds the gRPC server to `listen_address` and serves until
/// the listener errors or the call is interrupted. Blocks the calling thread
/// on the cabi tokio runtime. Returns `0` on success / `-1` on failure
/// (writing the inner error to `out_err`).
///
/// `listen_address` must parse as a `std::net::SocketAddr` (for example
/// `"0.0.0.0:50051"` or `"127.0.0.1:7443"`). The underlying server is
/// consumed by `serve`; calling this twice on the same `BlazenPeerServer`
/// returns a `Validation` error.
///
/// # Safety
///
/// `server` must be a valid pointer to a `BlazenPeerServer` previously
/// produced by the cabi surface. `listen_address` must be a valid
/// NUL-terminated UTF-8 buffer. `out_err` is null OR a valid destination for
/// one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_server_serve_blocking(
    server: *const BlazenPeerServer,
    listen_address: *const c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if server.is_null() || listen_address.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `server` is a live pointer.
    let server = unsafe { &*server };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `listen_address`.
    let addr = match unsafe { cstr_to_str(listen_address) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => {
            return unsafe { write_internal_error(out_err, "listen_address not valid UTF-8") };
        }
    };
    let inner = Arc::clone(&server.0);
    match runtime().block_on(async move { inner.serve(addr).await }) {
        Ok(()) => 0,
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Spawns the peer server onto the cabi tokio runtime and returns an opaque
/// future handle immediately. The caller waits via `blazen_future_wait` /
/// `blazen_future_fd` / `blazen_future_poll`, then takes the (unit) result
/// via `blazen_future_take_unit` (defined in `persist.rs`).
///
/// Returns null if `server` or `listen_address` is null, or if
/// `listen_address` is not valid UTF-8.
///
/// # Safety
///
/// `server` must be a valid pointer to a `BlazenPeerServer` previously
/// produced by the cabi surface. `listen_address` must be a valid
/// NUL-terminated UTF-8 buffer that remains valid for the duration of this
/// call (the buffer is copied before this function returns).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_server_serve(
    server: *const BlazenPeerServer,
    listen_address: *const c_char,
) -> *mut BlazenFuture {
    if server.is_null() || listen_address.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `server` is a live pointer.
    let server = unsafe { &*server };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `listen_address`.
    let addr = match unsafe { cstr_to_str(listen_address) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    let inner = Arc::clone(&server.0);
    BlazenFuture::spawn(async move { inner.serve(addr).await })
}

/// Frees a `BlazenPeerServer` handle previously produced by the cabi
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `server` must be null OR a pointer previously produced by
/// [`blazen_peer_server_new`]. Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_server_free(server: *mut BlazenPeerServer) {
    if server.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(server) });
}

// ---------------------------------------------------------------------------
// BlazenPeerClient
// ---------------------------------------------------------------------------

/// Opaque wrapper around `blazen_uniffi::peer::PeerClient`.
///
/// The inner `Arc` matches the `self: Arc<Self>` shape of the underlying
/// async `run_remote_workflow` method.
pub struct BlazenPeerClient(pub(crate) Arc<InnerPeerClient>);

/// Opens a connection to the peer at `address`, blocking the calling thread
/// on the cabi tokio runtime for the TCP / HTTP/2 handshake. Returns `0` on
/// success (writing a caller-owned `*mut BlazenPeerClient` to `out_client`)
/// or `-1` on failure (writing the inner error to `out_err`).
///
/// `address` must be a valid gRPC endpoint URI such as
/// `"http://node-a.local:7443"`. `client_node_id` identifies this end of
/// the connection in trace logs on both sides and is typically the local
/// hostname or a process-startup UUID.
///
/// # Safety
///
/// `address` and `client_node_id` must be valid NUL-terminated UTF-8 buffers
/// that remain live for the duration of the call. `out_client` is null OR a
/// valid destination for one `*mut BlazenPeerClient` write. `out_err` is
/// null OR a valid destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_client_connect(
    address: *const c_char,
    client_node_id: *const c_char,
    out_client: *mut *mut BlazenPeerClient,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if address.is_null() || client_node_id.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `address`.
    let address = match unsafe { cstr_to_str(address) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => return unsafe { write_internal_error(out_err, "address not valid UTF-8") },
    };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `client_node_id`.
    let node_id = match unsafe { cstr_to_str(client_node_id) } {
        Some(s) => s.to_owned(),
        None => {
            // SAFETY: `out_err` upholds the function-level contract.
            return unsafe { write_internal_error(out_err, "client_node_id not valid UTF-8") };
        }
    };
    match InnerPeerClient::connect(address, node_id) {
        Ok(client) => {
            if !out_client.is_null() {
                // SAFETY: caller-supplied out-param; per the function-level
                // contract it's either null (handled above) or a valid
                // destination for a single pointer-sized write.
                unsafe {
                    *out_client = Box::into_raw(Box::new(BlazenPeerClient(client)));
                }
            }
            0
        }
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Returns the client's `node_id` (the value passed as `client_node_id` to
/// [`blazen_peer_client_connect`]) as a heap-allocated NUL-terminated UTF-8
/// C string. Returns null if `client` is null.
///
/// # Ownership
///
/// Caller frees with [`crate::string::blazen_string_free`].
///
/// # Safety
///
/// `client` must be null OR a valid pointer to a `BlazenPeerClient`
/// previously produced by the cabi surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_client_node_id(
    client: *const BlazenPeerClient,
) -> *mut c_char {
    if client.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    let node_id = Arc::clone(&client.0).node_id();
    alloc_cstring(&node_id)
}

/// Synchronously invokes a workflow on the connected peer and waits for its
/// terminal result. Blocks the calling thread on the cabi tokio runtime.
/// Returns `0` on success (writing a caller-owned `*mut BlazenWorkflowResult`
/// to `out_result`) or `-1` on failure (writing the inner error to
/// `out_err`).
///
/// - `workflow_name` is the symbolic name the remote peer's step registry
///   knows the workflow as.
/// - `step_ids` is an array of `step_ids_count` NUL-terminated UTF-8 C
///   strings, each identifying a step to execute. Pass null + `0` for an
///   empty list.
/// - `input_json` is the JSON-encoded payload fed into the workflow's entry
///   step.
/// - `timeout_secs` bounds the remote workflow's wall-clock execution.
///   Pass `-1` to defer to the server's default deadline; any non-negative
///   value is converted to `u64` seconds.
///
/// # Safety
///
/// `client` must be a valid pointer to a `BlazenPeerClient`. `workflow_name`
/// and `input_json` must be valid NUL-terminated UTF-8 buffers. `step_ids`
/// must satisfy the `(ptrs, count)` contract documented on
/// [`ptr_array_to_strings`]. `out_result` is null OR a valid destination for
/// one `*mut BlazenWorkflowResult` write. `out_err` is null OR a valid
/// destination for one `*mut BlazenError` write.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_client_run_remote_workflow_blocking(
    client: *const BlazenPeerClient,
    workflow_name: *const c_char,
    step_ids: *const *const c_char,
    step_ids_count: usize,
    input_json: *const c_char,
    timeout_secs: i64,
    out_result: *mut *mut BlazenWorkflowResult,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if client.is_null() || workflow_name.is_null() || input_json.is_null() {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `workflow_name`.
    let workflow_name = match unsafe { cstr_to_str(workflow_name) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => {
            return unsafe { write_internal_error(out_err, "workflow_name not valid UTF-8") };
        }
    };
    // SAFETY: caller upholds the `(ptrs, count)` contract on `step_ids`.
    let Some(step_ids) = (unsafe { ptr_array_to_strings(step_ids, step_ids_count) }) else {
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe {
            write_internal_error(out_err, "step_ids contained null or non-UTF-8 entry")
        };
    };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `input_json`.
    let input = match unsafe { cstr_to_str(input_json) } {
        Some(s) => s.to_owned(),
        // SAFETY: `out_err` upholds the function-level contract.
        None => return unsafe { write_internal_error(out_err, "input_json not valid UTF-8") },
    };
    let timeout = if timeout_secs < 0 {
        None
    } else {
        Some(u64::try_from(timeout_secs).unwrap_or(u64::MAX))
    };
    let inner = Arc::clone(&client.0);
    match runtime().block_on(async move {
        inner
            .run_remote_workflow(workflow_name, step_ids, input, timeout)
            .await
    }) {
        Ok(result) => {
            if !out_result.is_null() {
                // SAFETY: caller-supplied out-param; per the function-level
                // contract it's either null (handled above) or a valid
                // destination for a single pointer-sized write.
                unsafe {
                    *out_result = BlazenWorkflowResult::from(result).into_ptr();
                }
            }
            0
        }
        // SAFETY: `out_err` upholds the function-level contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Invokes a workflow on the connected peer asynchronously, returning an
/// opaque future handle immediately. The caller waits via
/// `blazen_future_wait` / `blazen_future_fd` / `blazen_future_poll`, then
/// takes the result via `blazen_future_take_workflow_result` (defined in
/// `workflow.rs`).
///
/// Returns null if any input pointer is null (other than `step_ids` when
/// `step_ids_count == 0`) or if any string argument is not valid UTF-8.
/// Errors that surface during the async run are delivered through
/// `blazen_future_take_workflow_result`'s `err` out-param.
///
/// Argument semantics match
/// [`blazen_peer_client_run_remote_workflow_blocking`] — `timeout_secs < 0`
/// maps to `None` (server default), `>= 0` maps to `Some(u64)` seconds.
///
/// # Safety
///
/// `client` must be a valid pointer to a `BlazenPeerClient`. `workflow_name`
/// and `input_json` must be valid NUL-terminated UTF-8 buffers that remain
/// valid for the duration of this call (their contents are copied before
/// this function returns). `step_ids` must satisfy the `(ptrs, count)`
/// contract documented on [`ptr_array_to_strings`], with every per-entry
/// buffer remaining valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_client_run_remote_workflow(
    client: *const BlazenPeerClient,
    workflow_name: *const c_char,
    step_ids: *const *const c_char,
    step_ids_count: usize,
    input_json: *const c_char,
    timeout_secs: i64,
) -> *mut BlazenFuture {
    if client.is_null() || workflow_name.is_null() || input_json.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller has guaranteed `client` is a live pointer.
    let client = unsafe { &*client };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `workflow_name`.
    let workflow_name = match unsafe { cstr_to_str(workflow_name) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    // SAFETY: caller upholds the `(ptrs, count)` contract on `step_ids`.
    let Some(step_ids) = (unsafe { ptr_array_to_strings(step_ids, step_ids_count) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: caller upholds the NUL-termination + lifetime contract on `input_json`.
    let input = match unsafe { cstr_to_str(input_json) } {
        Some(s) => s.to_owned(),
        None => return std::ptr::null_mut(),
    };
    let timeout = if timeout_secs < 0 {
        None
    } else {
        Some(u64::try_from(timeout_secs).unwrap_or(u64::MAX))
    };
    let inner = Arc::clone(&client.0);
    BlazenFuture::spawn(async move {
        inner
            .run_remote_workflow(workflow_name, step_ids, input, timeout)
            .await
    })
}

/// Frees a `BlazenPeerClient` handle previously produced by the cabi
/// surface. No-op on a null pointer.
///
/// # Safety
///
/// `client` must be null OR a pointer previously produced by
/// [`blazen_peer_client_connect`]. Calling this twice on the same non-null
/// pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_peer_client_free(client: *mut BlazenPeerClient) {
    if client.is_null() {
        return;
    }
    // SAFETY: caller upholds the `Box::into_raw` provenance contract.
    drop(unsafe { Box::from_raw(client) });
}

// The peer module deliberately does NOT define typed future-take wrappers.
// Consumers of `blazen_peer_server_serve`'s future use
// `blazen_future_take_unit` (defined in `persist.rs`); consumers of
// `blazen_peer_client_run_remote_workflow` use
// `blazen_future_take_workflow_result` (defined in `workflow.rs`).
