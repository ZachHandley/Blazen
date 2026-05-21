//! HTTP surface for `blazen-controlplane`.
//!
//! This module hosts two independent HTTP servers that happen to share a
//! crate:
//!
//! - [`transport`] — the workflow control plane's HTTP/SSE bridge for
//!   environments that cannot speak HTTP/2 bidi gRPC (browsers, wasi
//!   runtimes, some serverless platforms). Gated by `http-transport`.
//! - PR5 phase 2 — OpenAI-compatible REST + Blazen-admin endpoints on
//!   top of the [`crate::server::model_manager::ManagerHandle`] trait.
//!   Gated by `http-rest`. Files under this module:
//!   - [`openai_compat`] — `/v1/chat/completions`, `/v1/completions`,
//!     `/v1/embeddings`, `/v1/models`, `/v1/audio/speech`,
//!     `/v1/audio/transcriptions`, `/v1/images/generations`.
//!   - [`blazen_admin`] — `/v1/blazen/adapters/...`,
//!     `/v1/blazen/models/...`, `/v1/blazen/health`,
//!     `/v1/blazen/metrics`.
//!   - [`sse`] — server-sent-events helpers (data framing + `[DONE]`
//!     terminator).
//!   - [`uploads`] — chunked in-memory `ContentStore` for adapter blobs
//!     and audio payloads that exceed the per-request body limit.
//!   - [`error`] — central `HttpError` enum and its `IntoResponse`
//!     mapping into OpenAI-style JSON.
//!
//! The two surfaces are intentionally siloed: the workflow CP uses
//! postcard-in-base64 JSON envelopes, while the REST surface is plain
//! `application/json` with the `OpenAI` schema. Both can be mounted on the
//! same axum router if a deployment wants a single port, or on separate
//! routers if not.

#[cfg(feature = "http-transport")]
pub mod transport;

#[cfg(feature = "http-transport")]
pub use transport::{HttpError as TransportHttpError, HttpWorkerState, PostcardEnvelope, router};

#[cfg(feature = "http-rest")]
pub mod error;
#[cfg(feature = "http-rest")]
pub mod sse;
#[cfg(feature = "http-rest")]
pub mod uploads;

#[cfg(feature = "http-rest")]
pub mod blazen_admin;
#[cfg(feature = "http-rest")]
pub mod openai_compat;

#[cfg(feature = "http-rest")]
mod rest_state;

#[cfg(feature = "http-rest")]
pub use error::HttpError;
#[cfg(feature = "http-rest")]
pub use rest_state::{RestMetrics, RestState};
#[cfg(feature = "http-rest")]
pub use uploads::{ContentStore, StoredBlob};

#[cfg(feature = "http-rest")]
use std::sync::Arc;

#[cfg(feature = "http-rest")]
use axum::Router;

#[cfg(feature = "http-rest")]
use crate::server::model_manager::ManagerHandle;

/// Build the combined OpenAI-compat + Blazen-admin axum router.
///
/// The returned router does NOT install any authentication middleware —
/// PR5 phase 3 owns that. Mount it under whatever bearer / mTLS guard
/// the deployment requires.
#[cfg(feature = "http-rest")]
pub fn build_router(handle: Arc<dyn ManagerHandle>) -> Router {
    let state = Arc::new(RestState::new(handle));
    Router::new()
        .merge(openai_compat::router(state.clone()))
        .merge(blazen_admin::router(state))
}

/// Build a router around an already-constructed [`RestState`]. Useful
/// for tests that want to inspect the metrics counter or the content
/// store after a request.
#[cfg(feature = "http-rest")]
pub fn build_router_with_state(state: Arc<RestState>) -> Router {
    Router::new()
        .merge(openai_compat::router(state.clone()))
        .merge(blazen_admin::router(state))
}
