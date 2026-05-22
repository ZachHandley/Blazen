//! Model-plane C ABI: client-side wrapper around
//! [`blazen_controlplane::ModelClient`].
//!
//! Mirrors the layout of [`crate::controlplane`] but targets the
//! `BlazenModelServer` gRPC service (model lifecycle, inference,
//! adapters, blobs). This first wave only exposes the four blocking
//! lifecycle/status entry points that the Ruby binding's
//! `Blazen::Distributed::ModelClient` needs to bring up a smoke test —
//! adapters, inference, and streaming land in later waves.
//!
//! ## Surface
//!
//! - [`blazen_modelclient_connect_blocking`] — plaintext gRPC.
//! - [`blazen_modelclient_connect_with_tls_blocking`] — TLS / mTLS.
//!   The CA bundle PEM is required; the client identity (cert + key)
//!   PEM pair is optional, enabling either server-auth-only TLS or
//!   full mTLS depending on which pair is supplied.
//! - [`blazen_modelclient_status_blocking`] — server-wide model
//!   snapshot serialized as a JSON string (caller frees via
//!   [`crate::string::blazen_string_free`]).
//! - [`blazen_modelclient_is_loaded_blocking`] — boolean liveness
//!   probe for a single model id.
//! - [`blazen_modelclient_load_blocking`] — load a previously-registered
//!   model. Request/response are JSON strings mirroring
//!   `LoadRequest` / `LoadResponse`.
//! - [`blazen_modelclient_unload_blocking`] — drop a loaded model from
//!   memory. Same JSON-in/JSON-out shape.
//! - [`blazen_modelclient_load_from_hf_blocking`] — register-and-load a
//!   model from a Hugging Face repo; response carries the chosen
//!   backend.
//! - [`blazen_modelclient_free`] — drops the client handle.
//!
//! Gated on the `distributed` feature, matching the rest of the
//! control-plane surface (see [`crate::controlplane`]).

use std::ffi::c_char;
use std::sync::Arc;

use blazen_controlplane::ModelClient as InnerModelClient;
use blazen_controlplane::model_protocol::{
    CompleteRequest, EmbedRequest, FetchBlobChunk, FetchBlobRequest, GenerateImageRequest,
    GenerateMusicRequest, IsLoadedRequest, ListAdaptersRequest, LoadAdapterRequest,
    LoadFromHfRequest, LoadRequest, MODEL_ENVELOPE_VERSION, StatusRequest, StreamCompleteChunk,
    TextToSpeechRequest, TranscribeRequest, UnloadAdapterRequest, UnloadRequest, UploadBlobChunk,
};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tonic::transport::{Certificate, ClientTlsConfig, Identity};

use blazen_uniffi::errors::BlazenError as InnerError;
use blazen_uniffi::llm::TokenUsage as InnerTokenUsage;
use blazen_uniffi::streaming::{CompletionStreamSink, StreamChunk as InnerStreamChunk};

use crate::error::BlazenError;
use crate::runtime::runtime;
use crate::stream_sink::{BlazenCompletionStreamSinkVTable, CStreamSink};
use crate::string::{alloc_cstring, cstr_to_str};

// ===========================================================================
// Shared error-out helpers (mirror controlplane.rs)
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
// BlazenModelClient
// ===========================================================================

/// Opaque wrapper around [`blazen_controlplane::ModelClient`]. The
/// inner [`InnerModelClient`] is cheaply cloneable (it holds an
/// `Arc<Mutex<...>>` internally) so multiple cabi calls on the same
/// handle can run concurrently.
pub struct BlazenModelClient {
    inner: InnerModelClient,
}

/// Synchronously open a plaintext gRPC connection to a model server at
/// `endpoint`. Blocks the calling thread on the cabi tokio runtime.
/// Returns `0` on success (writing a caller-owned `*mut BlazenModelClient`
/// to `out_handle`) or `-1` on failure (writing the inner error to
/// `out_err`).
///
/// `endpoint` is a gRPC URI such as `"http://models.example.com:7446"`.
/// For TLS, use [`blazen_modelclient_connect_with_tls_blocking`].
///
/// # Safety
///
/// `endpoint` must be a valid NUL-terminated UTF-8 buffer that remains
/// live for the duration of the call. `out_handle` is null OR a
/// destination for one `*mut BlazenModelClient` write. `out_err` is
/// null OR a destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_connect_blocking(
    endpoint: *const c_char,
    out_handle: *mut *mut BlazenModelClient,
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
    match runtime().block_on(async move { InnerModelClient::connect(endpoint).await }) {
        Ok(client) => {
            if !out_handle.is_null() {
                // SAFETY: out_handle upholds the function contract.
                unsafe {
                    *out_handle = Box::into_raw(Box::new(BlazenModelClient { inner: client }));
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously open a TLS (or mTLS) gRPC connection to a model
/// server at `endpoint`, supplying PEM credentials as in-memory UTF-8
/// strings. Same return shape as
/// [`blazen_modelclient_connect_blocking`].
///
/// `ca_cert_pem` is required and pins the server's identity. If both
/// `client_cert_pem` and `client_key_pem` are supplied (non-null), the
/// connection upgrades to mutual TLS using that client identity;
/// otherwise the connection is server-auth-only TLS. Supplying only
/// one of the pair returns a validation error.
///
/// # Safety
///
/// `endpoint` and `ca_cert_pem` must be valid NUL-terminated UTF-8
/// buffers that remain live for the duration of the call.
/// `client_cert_pem` and `client_key_pem` must each be null OR a valid
/// NUL-terminated UTF-8 buffer. `out_handle` is null OR a destination
/// for one `*mut BlazenModelClient` write. `out_err` is null OR a
/// destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_connect_with_tls_blocking(
    endpoint: *const c_char,
    ca_cert_pem: *const c_char,
    client_cert_pem: *const c_char,
    client_key_pem: *const c_char,
    out_handle: *mut *mut BlazenModelClient,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if endpoint.is_null() || ca_cert_pem.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller upholds the NUL + lifetime contract on `endpoint`.
    let endpoint = match unsafe { cstr_to_str(endpoint) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "endpoint not valid UTF-8") },
    };
    // SAFETY: caller upholds the NUL + lifetime contract on `ca_cert_pem`.
    let ca_pem = match unsafe { cstr_to_str(ca_cert_pem) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "ca_cert_pem not valid UTF-8") },
    };
    // SAFETY: caller upholds the NUL + lifetime contract on `client_cert_pem` (may be null).
    let client_cert = unsafe { cstr_to_str(client_cert_pem) }.map(str::to_owned);
    // SAFETY: caller upholds the NUL + lifetime contract on `client_key_pem` (may be null).
    let client_key = unsafe { cstr_to_str(client_key_pem) }.map(str::to_owned);

    let identity = match (client_cert, client_key) {
        (Some(c), Some(k)) => Some(Identity::from_pem(c, k)),
        (None, None) => None,
        _ => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(
                    out_err,
                    "client_cert_pem and client_key_pem must both be set or both null",
                )
            };
        }
    };

    let mut tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_pem));
    if let Some(id) = identity {
        tls = tls.identity(id);
    }

    match runtime()
        .block_on(async move { InnerModelClient::connect_with_tls(endpoint, Some(tls)).await })
    {
        Ok(client) => {
            if !out_handle.is_null() {
                // SAFETY: out_handle upholds the function contract.
                unsafe {
                    *out_handle = Box::into_raw(Box::new(BlazenModelClient { inner: client }));
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously fetch the server's model status snapshot. Returns `0`
/// on success (writing a caller-owned `*mut c_char` JSON string to
/// `out_status_json`; free via [`crate::string::blazen_string_free`])
/// or `-1` on failure.
///
/// `model_id` is currently ignored on the wire (the server returns the
/// full registry snapshot), but is accepted for forward compatibility.
/// Pass null to request the full snapshot explicitly.
///
/// The JSON payload mirrors
/// [`blazen_controlplane::model_protocol::StatusResponse`] —
/// `{ "envelope_version": u32, "models": [...] }`.
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `model_id` is null OR a valid NUL-terminated UTF-8 buffer.
/// `out_status_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_status_blocking(
    handle: *mut BlazenModelClient,
    model_id: *const c_char,
    out_status_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null handle argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // Validate model_id UTF-8 even though the wire request currently
    // ignores it — surfaces malformed callers eagerly rather than
    // silently swallowing them.
    if !model_id.is_null() {
        // SAFETY: caller upholds the NUL + lifetime contract on `model_id` (may be null).
        if unsafe { cstr_to_str(model_id) }.is_none() {
            // SAFETY: out_err upholds the function contract.
            return unsafe { write_internal_error(out_err, "model_id not valid UTF-8") };
        }
    }

    let inner = client.inner.clone();
    let req = StatusRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
    };
    match runtime().block_on(async move { inner.status(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize StatusResponse: {e}"),
                        )
                    };
                }
            };
            if !out_status_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(out_err, "status JSON contains interior NUL byte")
                    };
                }
                // SAFETY: out_status_json upholds the function contract.
                unsafe {
                    *out_status_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously probe whether the named model is currently loaded on
/// the server. Returns `0` on success (writing the boolean to
/// `out_loaded`) or `-1` on failure.
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `model_id` must be a valid NUL-terminated UTF-8 buffer. `out_loaded`
/// is null OR a destination for one `bool` write. `out_err` is null OR
/// a destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_is_loaded_blocking(
    handle: *mut BlazenModelClient,
    model_id: *const c_char,
    out_loaded: *mut bool,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || model_id.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `model_id`.
    let model_id = match unsafe { cstr_to_str(model_id) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "model_id not valid UTF-8") },
    };

    let inner = client.inner.clone();
    let req = IsLoadedRequest {
        envelope_version: MODEL_ENVELOPE_VERSION,
        model_id,
    };
    match runtime().block_on(async move { inner.is_loaded(req).await }) {
        Ok(resp) => {
            if !out_loaded.is_null() {
                // SAFETY: out_loaded upholds the function contract.
                unsafe {
                    *out_loaded = resp.loaded;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously load a previously-registered model on the server.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `LoadResponse` to `out_response_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::LoadRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_load_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let req: LoadRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse LoadRequest: {e}"))
            };
        }
    };

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.load(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize LoadResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "load response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously unload a previously-loaded model on the server.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `UnloadResponse` to `out_response_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::UnloadRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_unload_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let req: UnloadRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse UnloadRequest: {e}"))
            };
        }
    };

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.unload(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize UnloadResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "unload response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously register-and-load a model from a Hugging Face Hub repo.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `LoadFromHfResponse` to `out_response_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::LoadFromHfRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_load_from_hf_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let req: LoadFromHfRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse LoadFromHfRequest: {e}"))
            };
        }
    };

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.load_from_hf(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize LoadFromHfResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "load_from_hf response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

// ===========================================================================
// Adapters (3)
// ===========================================================================

/// Synchronously load an adapter onto a previously-loaded base model.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `LoadAdapterResponse` to `out_response_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::LoadAdapterRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_load_adapter_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: LoadAdapterRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse LoadAdapterRequest: {e}"))
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.load_adapter(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize LoadAdapterResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "load_adapter response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously unload a previously-mounted adapter.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `UnloadAdapterResponse` to `out_response_json`; free
/// via [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::UnloadAdapterRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_unload_adapter_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: UnloadAdapterRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(
                    out_err,
                    &format!("failed to parse UnloadAdapterRequest: {e}"),
                )
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.unload_adapter(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize UnloadAdapterResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "unload_adapter response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously list the adapters currently mounted on a model.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `ListAdaptersResponse` to `out_response_json`; free
/// via [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::ListAdaptersRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_list_adapters_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: ListAdaptersRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(
                    out_err,
                    &format!("failed to parse ListAdaptersRequest: {e}"),
                )
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.list_adapters(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize ListAdaptersResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "list_adapters response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

// ===========================================================================
// Inference (2 — non-streaming)
// ===========================================================================

/// Synchronously issue a `Complete` RPC against a loaded model.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `CompleteResponse` to `out_response_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::CompleteRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_complete_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: CompleteRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse CompleteRequest: {e}"))
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.complete(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize CompleteResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "complete response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously issue an `Embed` RPC.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `EmbedResponse` to `out_response_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::EmbedRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_embed_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: EmbedRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse EmbedRequest: {e}"))
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.embed(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize EmbedResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "embed response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

// ===========================================================================
// Multimodal (4)
// ===========================================================================

/// Synchronously issue a `GenerateImage` RPC.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `GenerateImageResponse` to `out_response_json`; free
/// via [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::GenerateImageRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_generate_image_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: GenerateImageRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(
                    out_err,
                    &format!("failed to parse GenerateImageRequest: {e}"),
                )
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.generate_image(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize GenerateImageResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "generate_image response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously issue a `TextToSpeech` RPC.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `TextToSpeechResponse` to `out_response_json`; free
/// via [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::TextToSpeechRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_text_to_speech_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: TextToSpeechRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(
                    out_err,
                    &format!("failed to parse TextToSpeechRequest: {e}"),
                )
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.text_to_speech(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize TextToSpeechResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "text_to_speech response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously issue a `GenerateMusic` RPC.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `GenerateMusicResponse` to `out_response_json`; free
/// via [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::GenerateMusicRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_generate_music_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: GenerateMusicRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(
                    out_err,
                    &format!("failed to parse GenerateMusicRequest: {e}"),
                )
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.generate_music(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize GenerateMusicResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "generate_music response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

/// Synchronously issue a `Transcribe` RPC.
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `TranscribeResponse` to `out_response_json`; free
/// via [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::TranscribeRequest`].
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_response_json` is null OR a destination for one `*mut c_char`
/// write. `out_err` is null OR a destination for one `*mut BlazenError`
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_transcribe_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_response_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: TranscribeRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse TranscribeRequest: {e}"))
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move { inner.transcribe(req).await }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize TranscribeResponse: {e}"),
                        )
                    };
                }
            };
            if !out_response_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "transcribe response JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_response_json upholds the function contract.
                unsafe {
                    *out_response_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, InnerError::from(e)) },
    }
}

// ===========================================================================
// Inference (streaming)
// ===========================================================================

/// Synchronously issue a `StreamComplete` server-streaming RPC against a
/// loaded model, dispatching each token-delta to the supplied vtable. Blocks
/// the calling thread on the cabi tokio runtime until the stream terminates.
/// Returns `0` on success (the sink's `on_done` fired) or `-1` on
/// initial-stream-start failure (writing a fresh `*mut BlazenError` to
/// `out_err`).
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of [`blazen_controlplane::model_protocol::CompleteRequest`];
/// the `envelope_version` field is overwritten with
/// [`MODEL_ENVELOPE_VERSION`] before dispatch.
///
/// ## Mid-stream errors
///
/// Errors observed mid-stream are delivered to the sink via `on_error`, and
/// the function still returns `0`. The only way to receive `-1` is to fail
/// before the first frame: invalid JSON, transport handshake failure on
/// `stream_complete`, or a null pointer argument. This mirrors the
/// uniffi-side [`blazen_uniffi::model_client::ModelClient::stream_complete`]
/// surface so all bindings observe a uniform happy/error split.
///
/// ## Text-only simplification
///
/// The wire-level [`StreamCompleteChunk`] carries only `text` payloads
/// (no per-frame tool-call deltas, citations, or reasoning trace). Each
/// `Delta` frame's `text` is forwarded to `on_chunk` as the
/// `BlazenStreamChunk`'s `content_delta`; `tool_calls` is empty. The
/// terminal `Done` frame's `finish_reason` (or `""` if absent) and the
/// reported `prompt_tokens`/`completion_tokens` are delivered through
/// `on_done` as a `BlazenTokenUsage`. The penultimate-vs-final flagging
/// mirrors the uniffi side: one chunk is buffered so the last
/// content-bearing chunk lands with `is_final = true`.
///
/// ## Ownership transfer
///
/// - `handle` is BORROWED — the inner `ModelClient` is cloned (cheap; it's
///   an `Arc<Mutex<...>>` internally) into the streaming call.
/// - `request_json` is BORROWED — copied into an owned `String` before
///   the runtime block; caller retains its buffer.
/// - `vtable` is CONSUMED — ownership of `user_data` transfers to the
///   wrapping `CStreamSink`, which releases it via `drop_user_data` on drop.
///   On every early-return failure path that aborts BEFORE constructing
///   `CStreamSink`, this function explicitly invokes
///   `(vtable.drop_user_data)(vtable.user_data)` to honour the same contract.
///
/// # Safety
///
/// `handle` must be null OR a valid pointer to a `BlazenModelClient`.
/// `request_json` must be null OR a valid NUL-terminated UTF-8 buffer.
/// `vtable.user_data` and its four function pointers must satisfy the
/// contracts documented on [`BlazenCompletionStreamSinkVTable`]. `out_err`
/// must be null OR a writable slot for a single `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_stream_complete_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    vtable: BlazenCompletionStreamSinkVTable,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        (vtable.drop_user_data)(vtable.user_data);
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: CompleteRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            (vtable.drop_user_data)(vtable.user_data);
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse CompleteRequest: {e}"))
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();

    // From here on, `CStreamSink::drop` is responsible for invoking
    // `drop_user_data` exactly once.
    let sink: Arc<dyn CompletionStreamSink> = Arc::new(CStreamSink::from_vtable(vtable));

    runtime().block_on(async move {
        let stream = match inner.stream_complete(req).await {
            Ok(s) => s,
            // SAFETY: out_err upholds the function contract.
            Err(e) => return unsafe { write_error(out_err, InnerError::from(e)) },
        };

        let mut stream = std::pin::pin!(stream);
        let mut last_finish_reason = String::new();
        let mut usage = InnerTokenUsage::default();
        let mut pending: Option<InnerStreamChunk> = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamCompleteChunk::Delta { text, .. }) => {
                    let wire = InnerStreamChunk {
                        content_delta: text,
                        tool_calls: Vec::new(),
                        is_final: false,
                    };
                    // Why: defer dispatch by one step so the last
                    // content-bearing chunk can be flagged `is_final = true`
                    // when we observe the terminal `Done` frame (or stream
                    // end) without having to peek ahead.
                    if let Some(prev) = pending.take()
                        && let Err(sink_err) = sink.on_chunk(prev).await
                    {
                        let _ = sink.on_error(sink_err).await;
                        return 0;
                    }
                    pending = Some(wire);
                }
                Ok(StreamCompleteChunk::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                }) => {
                    if let Some(reason) = finish_reason {
                        last_finish_reason = reason;
                    }
                    let prompt = u64::from(prompt_tokens.unwrap_or(0));
                    let completion = u64::from(completion_tokens.unwrap_or(0));
                    usage = InnerTokenUsage {
                        prompt_tokens: prompt,
                        completion_tokens: completion,
                        total_tokens: prompt + completion,
                        ..InnerTokenUsage::default()
                    };
                }
                Err(err) => {
                    if let Some(prev) = pending.take() {
                        let _ = sink.on_chunk(prev).await;
                    }
                    let _ = sink.on_error(InnerError::from(err)).await;
                    return 0;
                }
            }
        }

        if let Some(mut last) = pending.take() {
            last.is_final = true;
            if let Err(sink_err) = sink.on_chunk(last).await {
                let _ = sink.on_error(sink_err).await;
                return 0;
            }
        }

        if let Err(sink_err) = sink.on_done(last_finish_reason, usage).await {
            let _ = sink.on_error(sink_err).await;
        }
        0
    })
}

// ===========================================================================
// Blobs (2)
// ===========================================================================

/// Synchronously issue an `UploadBlob` client-streaming RPC. Sends a
/// single `Start` frame naming the blob (with optional `mime`), a single
/// `Data` frame carrying `data[..data_len]`, then `End`, and returns the
/// server's acknowledgement once the stream is closed.
///
/// Returns `0` on success (writing a caller-owned `*mut c_char` JSON
/// string for the `UploadBlobResponse` to `out_ack_json`; free via
/// [`crate::string::blazen_string_free`]) or `-1` on failure.
///
/// `blob_id` is required. `mime` may be null (omits the content-type
/// hint). `data` may be null only when `data_len == 0`.
///
/// ## Buffered ownership
///
/// The cabi pre-buffers the entire payload into a single in-memory
/// `Vec<u8>` before opening the stream — matching the `UniFFI` sibling
/// surface's synchronous-buffered semantics. For multi-gigabyte blobs
/// this materialises the full payload twice (caller buffer + chunk
/// vec); chunked streaming is reserved for a future wave once the
/// foreign-language sinks land.
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`. `blob_id`
/// must be a valid NUL-terminated UTF-8 buffer. `mime` is null OR a
/// valid NUL-terminated UTF-8 buffer. `data` is null OR points to at
/// least `data_len` readable bytes. `out_ack_json` is null OR a
/// destination for one `*mut c_char` write. `out_err` is null OR a
/// destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_upload_blob_blocking(
    handle: *mut BlazenModelClient,
    blob_id: *const c_char,
    mime: *const c_char,
    data: *const u8,
    data_len: usize,
    out_ack_json: *mut *mut c_char,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || blob_id.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    if data.is_null() && data_len > 0 {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "data is null but data_len > 0") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `blob_id`.
    let blob_id = match unsafe { cstr_to_str(blob_id) } {
        Some(s) => s.to_owned(),
        // SAFETY: out_err upholds the function contract.
        None => return unsafe { write_internal_error(out_err, "blob_id not valid UTF-8") },
    };
    // SAFETY: caller upholds the NUL + lifetime contract on `mime` (may be null).
    let mime = if mime.is_null() {
        None
    } else {
        match unsafe { cstr_to_str(mime) } {
            Some(s) => Some(s.to_owned()),
            // SAFETY: out_err upholds the function contract.
            None => return unsafe { write_internal_error(out_err, "mime not valid UTF-8") },
        }
    };
    // SAFETY: caller has guaranteed `data` points to at least `data_len`
    // readable bytes (or `data_len == 0`, in which case we substitute an
    // empty slice without dereferencing the null pointer).
    let payload: Vec<u8> = if data_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data, data_len) }.to_vec()
    };

    let total_bytes = u64::try_from(payload.len()).ok();
    let inner = client.inner.clone();
    match runtime().block_on(async move {
        let (tx, rx) = mpsc::channel::<UploadBlobChunk>(4);
        tx.send(UploadBlobChunk::Start {
            envelope_version: MODEL_ENVELOPE_VERSION,
            blob_id,
            total_bytes,
            content_type: mime,
        })
        .await
        .map_err(|e| InnerError::Internal {
            message: format!("upload_blob send Start failed: {e}"),
        })?;
        if !payload.is_empty() {
            tx.send(UploadBlobChunk::Data {
                envelope_version: MODEL_ENVELOPE_VERSION,
                bytes: payload,
            })
            .await
            .map_err(|e| InnerError::Internal {
                message: format!("upload_blob send Data failed: {e}"),
            })?;
        }
        tx.send(UploadBlobChunk::End {
            envelope_version: MODEL_ENVELOPE_VERSION,
        })
        .await
        .map_err(|e| InnerError::Internal {
            message: format!("upload_blob send End failed: {e}"),
        })?;
        drop(tx);
        inner.upload_blob(rx).await.map_err(InnerError::from)
    }) {
        Ok(resp) => {
            let json = match serde_json::to_string(&resp) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            &format!("failed to serialize UploadBlobResponse: {e}"),
                        )
                    };
                }
            };
            if !out_ack_json.is_null() {
                let ptr = alloc_cstring(&json);
                if ptr.is_null() {
                    // SAFETY: out_err upholds the function contract.
                    return unsafe {
                        write_internal_error(
                            out_err,
                            "upload_blob ack JSON contains interior NUL byte",
                        )
                    };
                }
                // SAFETY: out_ack_json upholds the function contract.
                unsafe {
                    *out_ack_json = ptr;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Synchronously issue a `FetchBlob` server-streaming RPC, draining the
/// stream into a single heap-allocated byte buffer.
///
/// Returns `0` on success (writing a caller-owned `*mut u8` to
/// `out_data` and its length to `out_data_len`; free via
/// [`blazen_modelclient_bytes_free`]) or `-1` on failure.
///
/// `request_json` is a NUL-terminated UTF-8 buffer holding a JSON
/// serialisation of
/// [`blazen_controlplane::model_protocol::FetchBlobRequest`]; the
/// `envelope_version` field is overwritten with
/// [`MODEL_ENVELOPE_VERSION`] before dispatch.
///
/// ## Buffered ownership
///
/// All `FetchBlobChunk::Data` frames' bytes are concatenated into a
/// single contiguous buffer. The `Start` and `End` frames carry no
/// payload bytes and are consumed silently. For multi-gigabyte blobs
/// this materialises the full payload in memory; chunked streaming is
/// reserved for a future wave.
///
/// # Safety
///
/// `handle` must be a valid pointer to a `BlazenModelClient`.
/// `request_json` must be a valid NUL-terminated UTF-8 buffer.
/// `out_data` is null OR a destination for one `*mut u8` write.
/// `out_data_len` is null OR a destination for one `usize` write.
/// `out_err` is null OR a destination for one `*mut BlazenError` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_fetch_blob_blocking(
    handle: *mut BlazenModelClient,
    request_json: *const c_char,
    out_data: *mut *mut u8,
    out_data_len: *mut usize,
    out_err: *mut *mut BlazenError,
) -> i32 {
    if handle.is_null() || request_json.is_null() {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "null pointer argument") };
    }
    // SAFETY: caller has guaranteed `handle` is a live pointer.
    let client = unsafe { &*handle };
    // SAFETY: caller upholds the NUL + lifetime contract on `request_json`.
    let Some(req_str) = (unsafe { cstr_to_str(request_json) }) else {
        // SAFETY: out_err upholds the function contract.
        return unsafe { write_internal_error(out_err, "request_json not valid UTF-8") };
    };
    let mut req: FetchBlobRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            // SAFETY: out_err upholds the function contract.
            return unsafe {
                write_internal_error(out_err, &format!("failed to parse FetchBlobRequest: {e}"))
            };
        }
    };
    req.envelope_version = MODEL_ENVELOPE_VERSION;

    let inner = client.inner.clone();
    match runtime().block_on(async move {
        let stream = inner.fetch_blob(req).await.map_err(InnerError::from)?;
        let mut stream = std::pin::pin!(stream);
        let mut buf: Vec<u8> = Vec::new();
        while let Some(frame) = stream.next().await {
            match frame.map_err(InnerError::from)? {
                FetchBlobChunk::Data { bytes, .. } => buf.extend_from_slice(&bytes),
                FetchBlobChunk::Start { .. } | FetchBlobChunk::End { .. } => {}
            }
        }
        Ok::<Vec<u8>, InnerError>(buf)
    }) {
        Ok(buf) => {
            let len = buf.len();
            let boxed: Box<[u8]> = buf.into_boxed_slice();
            // `Box::into_raw(Box<[u8]>)` yields `*mut [u8]`; cast to `*mut u8`
            // for the C-side caller. The matching free reconstructs the slice
            // via `slice::from_raw_parts_mut(ptr, len)` + `Box::from_raw`.
            let raw: *mut [u8] = Box::into_raw(boxed);
            let data_ptr: *mut u8 = raw.cast::<u8>();
            if out_data.is_null() {
                // Caller declined the buffer; release it immediately rather
                // than leak. SAFETY: we just produced this allocation above.
                unsafe {
                    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                        data_ptr, len,
                    )));
                }
            } else {
                // SAFETY: out_data upholds the function contract.
                unsafe {
                    *out_data = data_ptr;
                }
            }
            if !out_data_len.is_null() {
                // SAFETY: out_data_len upholds the function contract.
                unsafe {
                    *out_data_len = len;
                }
            }
            0
        }
        // SAFETY: out_err upholds the function contract.
        Err(e) => unsafe { write_error(out_err, e) },
    }
}

/// Frees a byte buffer previously produced by
/// [`blazen_modelclient_fetch_blob_blocking`]. Pairs the `*mut u8` /
/// `usize` (length) out-params produced by that call.
///
/// No-op when `ptr` is null. `len` must match the length originally
/// returned via `out_data_len`; passing a mismatched length is undefined
/// behaviour (the underlying allocation is reconstructed as
/// `Box<[u8]>` of exactly `len` bytes).
///
/// # Safety
///
/// `ptr` must be null OR a pointer previously produced by
/// [`blazen_modelclient_fetch_blob_blocking`]. `len` must equal the
/// length originally written to that call's `out_data_len`. Calling this
/// twice on the same pointer is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_bytes_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: per the contract above, `ptr` + `len` together describe an
    // owned `Box<[u8]>` allocation produced by
    // `blazen_modelclient_fetch_blob_blocking`. Reconstructing the slice
    // and dropping its `Box` releases the original allocation.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
    }
}

/// Frees a `BlazenModelClient` handle. No-op on null.
///
/// # Safety
///
/// `handle` must be null OR a pointer previously produced by the cabi
/// model-client surface. Calling this twice is a double-free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_modelclient_free(handle: *mut BlazenModelClient) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller upholds the Box::into_raw provenance contract.
    drop(unsafe { Box::from_raw(handle) });
}
