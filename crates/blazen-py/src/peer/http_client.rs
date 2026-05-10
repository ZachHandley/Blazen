//! HTTP/JSON peer client Python binding.
//!
//! Mirrors [`crate::peer::client::PyBlazenPeerClient`]'s method surface but
//! uses [`blazen_peer::HttpPeerClient`] under the hood — pure HTTP/JSON over
//! `Arc<dyn blazen_llm::http::HttpClient>`. Use this on hosts where the
//! native gRPC transport is unavailable, or when you simply want pure
//! HTTP/JSON peer transport on native targets.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use blazen_core::session_ref::RegistryKey;
use blazen_peer::{DerefResponse, ENVELOPE_VERSION, HttpPeerClient, ReleaseResponse};

use crate::peer::error::peer_err;
use crate::peer::types::{
    PyDerefRequest, PyDerefResponse, PyReleaseRequest, PyReleaseResponse, PySubWorkflowRequest,
    PySubWorkflowResponse,
};

/// HTTP/JSON peer client. Mirrors
/// [`crate::peer::client::PyBlazenPeerClient`]'s method surface but speaks
/// pure HTTP/JSON to a peer (or peer-shim) at `base_url`.
///
/// Construct with `HttpPeerClient.new_http(base_url, node_id)`. All RPC
/// methods are async and return Python coroutines.
#[gen_stub_pyclass]
#[pyclass(name = "HttpPeerClient")]
pub struct PyHttpPeerClient {
    inner: Arc<Mutex<HttpPeerClient>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHttpPeerClient {
    /// Build a new HTTP/JSON peer client.
    ///
    /// Args:
    ///     base_url: Peer HTTP root, e.g. ``"https://peer.example.com"``.
    ///         A trailing slash is tolerated and trimmed before each
    ///         request.
    ///     node_id: Identifier of *this* client used for trace logs.
    ///         Sent to the peer as the ``X-Blazen-Peer-Node-Id`` header.
    ///
    /// Returns:
    ///     A configured `HttpPeerClient`. No network I/O happens until
    ///     the first RPC call.
    #[staticmethod]
    #[pyo3(name = "new_http")]
    fn new_http(base_url: String, node_id: String) -> Self {
        let http = default_http_client();
        let client = HttpPeerClient::new(base_url, http, node_id);
        Self {
            inner: Arc::new(Mutex::new(client)),
        }
    }

    /// Invoke a sub-workflow on the connected peer.
    ///
    /// Args:
    ///     request: A `SubWorkflowRequest`.
    ///
    /// Returns:
    ///     A `SubWorkflowResponse`.
    ///
    /// Raises:
    ///     PeerError: If the request cannot be encoded, the HTTP call
    ///         fails, the peer returns a non-2xx status, or the
    ///         response cannot be decoded.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, SubWorkflowResponse]",
        imports = ("typing",)
    ))]
    fn invoke_sub_workflow<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PySubWorkflowRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let resp = guard.invoke_sub_workflow(req).await.map_err(peer_err)?;
            Ok(PySubWorkflowResponse { inner: resp })
        })
    }

    /// Dereference a remote session ref by its UUID. Returns a typed
    /// `DerefResponse` carrying the raw bytes returned by the origin
    /// node.
    ///
    /// Args:
    ///     request: A `DerefRequest` carrying the registry-entry UUID.
    ///
    /// Returns:
    ///     A `DerefResponse` whose `payload` is the raw bytes of the
    ///     underlying serialized value.
    ///
    /// Raises:
    ///     PeerError: On JSON encode/decode errors, transport errors,
    ///         or a non-2xx response (including the remote-side
    ///         ``NOT_FOUND`` analogue).
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, DerefResponse]",
        imports = ("typing",)
    ))]
    fn deref_session_ref<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PyDerefRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let key = RegistryKey(request.inner.ref_uuid);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let payload = guard.deref_session_ref(key).await.map_err(peer_err)?;
            Ok(PyDerefResponse {
                inner: DerefResponse {
                    envelope_version: ENVELOPE_VERSION,
                    payload,
                },
            })
        })
    }

    /// Release (drop) a remote session ref. Returns a typed
    /// `ReleaseResponse` carrying whether the ref was found and
    /// released on the origin node.
    ///
    /// Args:
    ///     request: A `ReleaseRequest` carrying the registry-entry
    ///         UUID to drop on the origin node.
    ///
    /// Returns:
    ///     A `ReleaseResponse` whose `released` is ``True`` if the
    ///     ref was found and dropped, ``False`` if it was already
    ///     gone.
    ///
    /// Raises:
    ///     PeerError: On JSON encode/decode errors, transport errors,
    ///         or a non-2xx response.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, ReleaseResponse]",
        imports = ("typing",)
    ))]
    fn release_session_ref<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, PyReleaseRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let key = RegistryKey(request.inner.ref_uuid);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let released = guard.release_session_ref(key).await.map_err(peer_err)?;
            Ok(PyReleaseResponse {
                inner: ReleaseResponse {
                    envelope_version: ENVELOPE_VERSION,
                    released,
                },
            })
        })
    }

    fn __repr__(&self) -> String {
        "HttpPeerClient(...)".to_owned()
    }
}

/// Resolve the platform-appropriate default HTTP client. Mirrors
/// `blazen_llm::default_http_client` (which is `pub(crate)`): on wasi we
/// build a [`blazen_llm::http_napi_wasi::LazyHttpClient`] (which defers to
/// `setDefaultHttpClient`), on native we build a stock
/// [`blazen_llm::ReqwestHttpClient`].
#[cfg(target_os = "wasi")]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::http_napi_wasi::LazyHttpClient::new().into_arc()
}

#[cfg(not(target_os = "wasi"))]
fn default_http_client() -> Arc<dyn blazen_llm::http::HttpClient> {
    blazen_llm::ReqwestHttpClient::new().into_arc()
}
