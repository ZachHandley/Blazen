//! Node bindings for [`blazen_peer::BlazenPeerServer`].
//!
//! Exposes a `BlazenPeerServer` JS class with a factory for setting the
//! local `node_id` and an async `serve(addr)` method that runs the gRPC
//! server until the process exits or the future is dropped.

use std::net::SocketAddr;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_peer::BlazenPeerServer;

use crate::error::peer_error_to_napi;

/// A Blazen peer gRPC server.
///
/// Each instance owns a stable `node_id` used to stamp
/// `RemoteRefDescriptor::origin_node_id` on session refs handed out by
/// this node.
///
/// ```typescript
/// const server = BlazenPeerServer.create("node-a");
/// await server.serve("127.0.0.1:7443");
/// ```
#[napi(js_name = "BlazenPeerServer")]
pub struct JsBlazenPeerServer {
    /// The native server. Wrapped in `Mutex<Option<...>>` because
    /// `BlazenPeerServer::serve` consumes `self`; we swap the value
    /// out via `Option::take` when `serve` is called.
    inner: Mutex<Option<BlazenPeerServer>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsBlazenPeerServer {
    /// Create a new peer server with the given stable `nodeId`.
    #[napi(factory)]
    pub fn create(node_id: String) -> Self {
        Self {
            inner: Mutex::new(Some(BlazenPeerServer::new(node_id))),
        }
    }

    /// Bind the gRPC server to `addr` and serve forever.
    ///
    /// `addr` must be a valid `SocketAddr`-style string such as
    /// `"127.0.0.1:7443"` or `"[::1]:7443"`. Consumes the server: a
    /// second call to `serve` on the same instance throws.
    #[napi]
    pub async fn serve(&self, addr: String) -> Result<()> {
        let server = {
            let mut guard = self.inner.lock().expect("poisoned");
            guard.take().ok_or_else(|| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    "BlazenPeerServer already consumed (serve was already called)",
                )
            })?
        };
        let socket: SocketAddr = addr
            .parse()
            .map_err(|e: std::net::AddrParseError| napi::Error::from_reason(e.to_string()))?;
        server.serve(socket).await.map_err(peer_error_to_napi)
    }
}
