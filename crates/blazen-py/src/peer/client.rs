//! Python wrapper for [`blazen_peer::BlazenPeerClient`].

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;
use uuid::Uuid;

use blazen_core::session_ref::RegistryKey;
use blazen_peer::BlazenPeerClient;

use crate::peer::error::{PeerException, peer_err};
use crate::peer::types::{PySubWorkflowRequest, PySubWorkflowResponse};

/// Client handle for talking to a remote `BlazenPeerServer`.
///
/// Construct with `BlazenPeerClient.connect(endpoint, node_id)`. All RPC
/// methods are async and return Python coroutines.
#[gen_stub_pyclass]
#[pyclass(name = "BlazenPeerClient")]
pub struct PyBlazenPeerClient {
    inner: Arc<Mutex<BlazenPeerClient>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlazenPeerClient {
    /// Open a connection to the peer at `endpoint`.
    ///
    /// Args:
    ///     endpoint: A gRPC URI, e.g. ``"http://node-a.local:7443"``.
    ///     node_id: Identifier of *this* client used for trace logs.
    ///
    /// Returns:
    ///     A connected `BlazenPeerClient`.
    ///
    /// Raises:
    ///     PeerError: If the endpoint URI is invalid or the connection
    ///         cannot be established.
    #[staticmethod]
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, BlazenPeerClient]",
        imports = ("typing",)
    ))]
    #[allow(clippy::elidable_lifetime_names)]
    fn connect<'py>(
        py: Python<'py>,
        endpoint: String,
        node_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = BlazenPeerClient::connect(endpoint, node_id)
                .await
                .map_err(peer_err)?;
            Ok(Self {
                inner: Arc::new(Mutex::new(client)),
            })
        })
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
    ///     PeerError: If the request cannot be encoded, the RPC fails,
    ///         or the response cannot be decoded.
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

    /// Dereference a remote session ref by its UUID. Returns the raw
    /// serialized bytes of the underlying value.
    ///
    /// Args:
    ///     ref_uuid: A UUID string identifying the registry entry on the
    ///         origin node.
    ///
    /// Returns:
    ///     The raw bytes returned by the origin node's serializer.
    ///
    /// Raises:
    ///     PeerError: On postcard or transport errors, including remote
    ///         ``NOT_FOUND``.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, bytes]",
        imports = ("typing",)
    ))]
    fn deref_session_ref<'py>(
        &self,
        py: Python<'py>,
        ref_uuid: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uuid = Uuid::parse_str(ref_uuid)
            .map_err(|e| PeerException::new_err(format!("invalid UUID: {e}")))?;
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let bytes = guard
                .deref_session_ref(RegistryKey(uuid))
                .await
                .map_err(peer_err)?;
            Ok(bytes)
        })
    }

    /// Release (drop) a remote session ref. Returns ``True`` if the ref
    /// was found and released, ``False`` if it was already gone.
    ///
    /// Args:
    ///     ref_uuid: A UUID string identifying the registry entry on the
    ///         origin node.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, bool]",
        imports = ("typing",)
    ))]
    fn release_session_ref<'py>(
        &self,
        py: Python<'py>,
        ref_uuid: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uuid = Uuid::parse_str(ref_uuid)
            .map_err(|e| PeerException::new_err(format!("invalid UUID: {e}")))?;
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let released = guard
                .release_session_ref(RegistryKey(uuid))
                .await
                .map_err(peer_err)?;
            Ok(released)
        })
    }

    fn __repr__(&self) -> String {
        "BlazenPeerClient(...)".to_owned()
    }
}
