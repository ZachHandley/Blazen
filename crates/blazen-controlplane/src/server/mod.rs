//! Tonic server implementation for the `BlazenControlPlane` gRPC service.
//!
//! See [`ControlPlaneServer`] for the entry point. Submodules:
//!
//! - [`registry`] — connected-worker registry with capability index.
//! - [`queue`] — assignment queue (per-capability FIFO).
//! - [`admission`] — server-side admission enforcement.
//! - [`session`] — bidi `WorkerSession` stream handler.
//! - [`rpc`] — unary submit/cancel/describe/list/drain handlers.
//! - [`subscribe`] — server-streaming event subscribers.
//! - [`interceptor`] — bearer-token auth interceptor.

pub mod admission;
pub mod interceptor;
pub mod queue;
pub mod registry;
pub mod rpc;
pub mod service;
pub mod session;
pub mod subscribe;

pub use service::ControlPlaneService;

use std::net::SocketAddr;
use std::sync::Arc;

use crate::error::ControlPlaneError;

/// The Blazen control-plane gRPC server.
///
/// Build with [`ControlPlaneServer::new`], optionally configure TLS via
/// [`ControlPlaneServer::with_tls`], then call [`ControlPlaneServer::serve`]
/// to bind a socket and run until shutdown.
pub struct ControlPlaneServer {
    // `node_id` is surfaced to workers via `shared.node_id` in the
    // `Welcome` handshake; the field is retained on the struct for
    // future use (e.g. self-identifying server logs / metrics) and to
    // keep the constructor signature stable across phases.
    #[allow(dead_code)]
    pub(crate) node_id: String,
    pub(crate) shared: Arc<SharedState>,
    pub(crate) tls: Option<tonic::transport::ServerTlsConfig>,
    #[cfg(feature = "http-transport")]
    pub(crate) http_addr: Option<SocketAddr>,
}

/// Shared state owned by the server and shared with every handler /
/// session task. All fields are `Arc`s or contain interior mutability so
/// the struct itself does not need to be wrapped.
pub struct SharedState {
    pub registry: registry::WorkerRegistry,
    pub queue: queue::AssignmentQueue,
    pub admission: admission::Admission,
    /// Local node identifier surfaced to workers in `Welcome`.
    pub node_id: String,
}

impl ControlPlaneServer {
    /// Build a fresh server with an empty registry and queue.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        let node_id = node_id.into();
        let shared = Arc::new(SharedState {
            registry: registry::WorkerRegistry::new(),
            queue: queue::AssignmentQueue::new(),
            admission: admission::Admission::new(),
            node_id: node_id.clone(),
        });
        Self {
            node_id,
            shared,
            tls: None,
            #[cfg(feature = "http-transport")]
            http_addr: None,
        }
    }

    /// Attach a TLS configuration. See [`crate::tls`].
    #[must_use]
    pub fn with_tls(mut self, tls: tonic::transport::ServerTlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Additionally serve the HTTP/SSE worker tier on `addr`. Requires
    /// the `http-transport` feature.
    #[cfg(feature = "http-transport")]
    #[must_use]
    pub fn with_http(mut self, addr: SocketAddr) -> Self {
        self.http_addr = Some(addr);
        self
    }

    /// Bind the gRPC service to `addr` and serve until the future is
    /// dropped. Auth via [`crate::server::interceptor::BearerAuthInterceptor`]
    /// is always installed.
    ///
    /// To additionally serve the HTTP/SSE worker tier (for browsers / wasi),
    /// build the server with the `http-transport` feature enabled and call
    /// [`ControlPlaneServer::with_http`] before `serve`.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] for any bind / TLS / accept
    /// failure.
    pub async fn serve(self, addr: SocketAddr) -> Result<(), ControlPlaneError> {
        use crate::pb::blazen_control_plane_server::BlazenControlPlaneServer;
        use crate::server::interceptor::BearerAuthInterceptor;
        use crate::server::service::ControlPlaneService;

        let service = ControlPlaneService::new(self.shared.clone());
        let interceptor = BearerAuthInterceptor::new();
        let svc = BlazenControlPlaneServer::with_interceptor(service, interceptor);

        let mut builder = tonic::transport::Server::builder();
        if let Some(tls) = self.tls {
            builder = builder
                .tls_config(tls)
                .map_err(|e| ControlPlaneError::Tls(e.to_string()))?;
        }

        #[cfg(feature = "http-transport")]
        let http_handle: Option<tokio::task::JoinHandle<Result<(), ControlPlaneError>>> = {
            if let Some(http_addr) = self.http_addr {
                let router = crate::http::router(self.shared.clone());
                let handle = tokio::spawn(async move {
                    let listener = tokio::net::TcpListener::bind(http_addr)
                        .await
                        .map_err(|e| ControlPlaneError::Transport(format!("bind http: {e}")))?;
                    axum::serve(listener, router)
                        .await
                        .map_err(|e| ControlPlaneError::Transport(format!("axum serve: {e}")))
                });
                Some(handle)
            } else {
                None
            }
        };

        let grpc_result = builder
            .add_service(svc)
            .serve(addr)
            .await
            .map_err(|e| ControlPlaneError::Transport(format!("grpc serve: {e}")));

        #[cfg(feature = "http-transport")]
        if let Some(handle) = http_handle {
            handle.abort();
            // Ignore JoinError — abort is the normal shutdown path.
            let _ = handle.await;
        }

        grpc_result
    }
}
