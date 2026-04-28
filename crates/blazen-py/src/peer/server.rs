//! Python wrapper for [`blazen_peer::BlazenPeerServer`].

use std::net::SocketAddr;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::Mutex;

use blazen_peer::BlazenPeerServer;

use crate::peer::error::{PeerException, peer_err};

/// A node-local Blazen peer gRPC server.
///
/// Owns a stable `node_id` and an in-process session-ref registry.
/// Call `serve(addr)` to bind and serve forever.
#[gen_stub_pyclass]
#[pyclass(name = "BlazenPeerServer")]
pub struct PyBlazenPeerServer {
    /// Wrapped in a `Mutex<Option<...>>` so `serve` can take ownership of
    /// the inner server (its `serve` method is `self`-by-value) without
    /// requiring `&mut self` on the Python side.
    inner: Arc<Mutex<Option<BlazenPeerServer>>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBlazenPeerServer {
    /// Create a new peer server with a fresh, empty session-ref registry.
    ///
    /// Args:
    ///     node_id: Stable identifier embedded in `RemoteRefDescriptor`s
    ///         this server hands out.
    #[new]
    fn new(node_id: &str) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(BlazenPeerServer::new(node_id)))),
        }
    }

    /// Replace the per-server session-ref registry with one supplied by the
    /// embedder. Lets a Blazen process share its registry between in-process
    /// workflows and remote peers.
    ///
    /// Args:
    ///     registry: A `_SessionRegistryHandle` obtained from the workflow
    ///         layer (typically via the Python contextvar bridge).
    fn with_session_refs(
        &self,
        registry: PyRef<'_, crate::workflow::session_ref::PySessionRegistryHandle>,
    ) -> PyResult<()> {
        let mut guard = self
            .inner
            .try_lock()
            .map_err(|_| PeerException::new_err("server is currently being served"))?;
        let server = guard
            .take()
            .ok_or_else(|| PeerException::new_err("server already consumed by serve()"))?;
        let updated = server.with_session_refs(Arc::clone(&registry.inner));
        *guard = Some(updated);
        Ok(())
    }

    /// Bind the gRPC server to `addr` and serve forever.
    ///
    /// Args:
    ///     addr: A socket address string, e.g. ``"0.0.0.0:7443"``.
    ///
    /// Raises:
    ///     PeerError: If the server cannot bind or fails while serving.
    #[gen_stub(override_return_type(
        type_repr = "typing.Coroutine[typing.Any, typing.Any, None]",
        imports = ("typing",)
    ))]
    fn serve<'py>(&self, py: Python<'py>, addr: &str) -> PyResult<Bound<'py, PyAny>> {
        let socket_addr: SocketAddr = addr
            .parse()
            .map_err(|e| PeerException::new_err(format!("invalid socket address {addr:?}: {e}")))?;
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let server = {
                let mut guard = inner.lock().await;
                guard
                    .take()
                    .ok_or_else(|| PeerException::new_err("server already consumed by serve()"))?
            };
            server.serve(socket_addr).await.map_err(peer_err)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        "BlazenPeerServer(...)".to_owned()
    }
}
