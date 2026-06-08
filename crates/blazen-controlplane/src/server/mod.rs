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
pub mod key_store;
pub mod queue;
pub mod registry;
pub mod rpc;
pub mod service;
pub mod session;
pub mod store;
pub mod subscribe;
#[cfg(feature = "valkey-store")]
pub mod valkey_store;

// PR5: remote-mode ModelManager server side. Gated behind the
// `model-server` feature so the host can keep the workflow control
// plane without pulling in the model-server symbols (and vice versa).
#[cfg(feature = "model-server")]
pub mod model_manager;
#[cfg(feature = "model-server")]
pub mod model_service;

pub use key_store::{EnvFileKeyStore, KeyStore, SharedKey};
pub use service::ControlPlaneService;
pub use store::{AssignmentStore, MemoryAssignmentStore};
#[cfg(feature = "valkey-store")]
pub use valkey_store::ValkeyAssignmentStore;

#[cfg(feature = "model-server")]
pub use model_manager::{ManagerHandle, ModelServerState};
#[cfg(feature = "model-server")]
pub use model_service::ModelService;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::broadcast;

use blazen_core::distributed::RunEvent;

use crate::error::ControlPlaneError;

/// Broadcast capacity for the per-server [`RunEvent`] fan-out bus.
/// Subscribers that fall this far behind receive `BroadcastStreamRecvError::Lagged`
/// (silently dropped by the subscribe handlers).
const EVENT_BUS_CAPACITY: usize = 1024;

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
    /// Per-server broadcast bus for [`RunEvent`]s. Producers: worker
    /// `Event` frames (via [`session::handle_worker_session`]) plus
    /// queue status transitions (via [`queue::AssignmentQueue`]'s
    /// `mark_*` / `update_run_to_running` mutators). Consumers: the
    /// `SubscribeRunEvents` / `SubscribeAll` server-streaming handlers
    /// and (when `http-transport` is on) the matching SSE routes.
    pub events: broadcast::Sender<RunEvent>,
    /// Per-place provider-key store. The session handler consults this on
    /// a worker `KeyRequest`, scoping the lookup to the worker's
    /// server-authenticated place. Defaults to
    /// [`key_store::EnvFileKeyStore`]; override with
    /// [`ControlPlaneServer::with_key_store`]. Keys resolved here NEVER
    /// touch an [`Assignment`](crate::protocol::Assignment) and are never
    /// logged.
    pub key_store: Arc<dyn key_store::KeyStore>,
    /// Per-session state for the HTTP/SSE worker tier. Keyed by
    /// `session_id` returned to the worker by `worker_register`. Each
    /// entry's `outbound_rx` is taken exactly once by the matching
    /// `worker_stream` handler.
    #[cfg(feature = "http-transport")]
    pub http_sessions: dashmap::DashMap<uuid::Uuid, Arc<crate::http::HttpWorkerState>>,
}

impl ControlPlaneServer {
    /// Build a fresh server with an empty registry and queue.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        let node_id = node_id.into();
        let (events_tx, _) = broadcast::channel(EVENT_BUS_CAPACITY);
        let shared = Arc::new(SharedState {
            registry: registry::WorkerRegistry::new(),
            queue: queue::AssignmentQueue::with_events(events_tx.clone()),
            admission: admission::Admission::new(),
            node_id: node_id.clone(),
            events: events_tx,
            key_store: key_store::default_key_store(),
            #[cfg(feature = "http-transport")]
            http_sessions: dashmap::DashMap::new(),
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

    /// Replace the default in-memory [`AssignmentStore`] with a
    /// caller-supplied implementation. Use this to wire in a
    /// Valkey-backed store (or any other durable backing) so queue
    /// mutations survive a control-plane restart.
    ///
    /// Builder-style: must be called BEFORE [`Self::serve`]. Internally
    /// rebuilds the [`SharedState`] so the queue is constructed against
    /// the supplied store. The previous `SharedState` is dropped; any
    /// references handed out before this call (none, in normal builder
    /// flow) would be orphaned.
    #[must_use]
    pub fn with_store(mut self, store: Arc<dyn store::AssignmentStore>) -> Self {
        let (events_tx, _) = broadcast::channel(EVENT_BUS_CAPACITY);
        // Preserve any previously-configured key store across the rebuild —
        // builder-method ordering must not silently drop a key store the
        // caller set with `with_key_store` first.
        let key_store = self.shared.key_store.clone();
        let new_shared = Arc::new(SharedState {
            registry: registry::WorkerRegistry::new(),
            queue: queue::AssignmentQueue::with_events_and_store(events_tx.clone(), store),
            admission: admission::Admission::new(),
            node_id: self.node_id.clone(),
            events: events_tx,
            key_store,
            #[cfg(feature = "http-transport")]
            http_sessions: dashmap::DashMap::new(),
        });
        self.shared = new_shared;
        self
    }

    /// Replace the default [`key_store::EnvFileKeyStore`] with a
    /// caller-supplied [`key_store::KeyStore`]. Use this to wire in a
    /// tenant-aware secret manager so the control plane can serve per-place
    /// provider keys to workers over the authenticated session.
    ///
    /// Builder-style: must be called BEFORE [`Self::serve`]. Mutates the
    /// existing [`SharedState`] in place via `Arc::make_mut`-free swap — the
    /// field is the only thing replaced, so any sibling builder
    /// (`with_store`) ordering is preserved.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `SharedState` `Arc` is unexpectedly
    /// shared at build time (it is not in the normal builder flow, where
    /// the server holds the sole reference until `serve`).
    #[must_use]
    pub fn with_key_store(mut self, key_store: Arc<dyn key_store::KeyStore>) -> Self {
        let shared = Arc::get_mut(&mut self.shared)
            .expect("SharedState must be uniquely held during builder configuration");
        shared.key_store = key_store;
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
    /// Downstream services that need to attach their own auth / context
    /// (e.g. JWT validation, per-caller `CallerCtx` in request extensions)
    /// should call [`Self::serve_with_interceptor`] instead — this method
    /// is a thin wrapper that supplies the default
    /// [`crate::server::interceptor::BearerAuthInterceptor`].
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] for any bind / TLS / accept
    /// failure.
    pub async fn serve(self, addr: SocketAddr) -> Result<(), ControlPlaneError> {
        use crate::server::interceptor::BearerAuthInterceptor;
        self.serve_with_interceptor(addr, BearerAuthInterceptor::new())
            .await
    }

    /// Like [`Self::serve`], but allows the caller to supply a custom tonic
    /// [`tonic::service::Interceptor`] instead of the default
    /// [`crate::server::interceptor::BearerAuthInterceptor`].
    ///
    /// Useful for downstream services that have their own auth layer
    /// (e.g. JWT validation) and want to attach a `CallerCtx` to each
    /// gRPC request's extensions before the handler runs.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] for any bind / TLS / accept
    /// failure, identical to [`Self::serve`].
    pub async fn serve_with_interceptor<I>(
        self,
        addr: SocketAddr,
        interceptor: I,
    ) -> Result<(), ControlPlaneError>
    where
        I: tonic::service::Interceptor + Clone + Send + Sync + 'static,
    {
        use crate::pb::blazen_control_plane_server::BlazenControlPlaneServer;
        use crate::server::service::ControlPlaneService;

        // Cold-start recovery: re-hydrate the queue's in-memory cache
        // from the persisted [`AssignmentStore`] before accepting any
        // gRPC connections. With the default in-memory store this is a
        // no-op; with a durable store (e.g. Valkey) it surrenders work
        // owned by the previous process back into the pending pool.
        self.shared.queue.recover_from_store().await?;

        let service = ControlPlaneService::new(self.shared.clone());
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
