//! Node bindings for the [`blazen_peer`] crate.
//!
//! Exposes the peer gRPC server and client to JS, alongside the
//! postcard wire types and helper functions for env-var auth and TLS
//! configuration.
//!
//! The `blazen-peer` dependency is pulled in with both its `server`
//! and `client` features enabled in this crate's `Cargo.toml`, so the
//! corresponding submodules are always available here.
//!
//! `types` is ungated — its `#[napi(object)]` wrappers are pure data
//! shapes shared with [`crate::peer_http`]'s wasi-compatible HTTP/JSON
//! peer client. The remaining submodules (`client`, `server`, `funcs`)
//! depend on tonic / rustls and are native-only.

#[cfg(not(target_os = "wasi"))]
pub mod client;
#[cfg(not(target_os = "wasi"))]
pub mod funcs;
#[cfg(not(target_os = "wasi"))]
pub mod server;
pub mod types;

#[cfg(not(target_os = "wasi"))]
pub use client::JsBlazenPeerClient;
#[cfg(not(target_os = "wasi"))]
pub use funcs::{
    load_client_tls, load_server_tls, peer_envelope_version, peer_token_env, resolve_beer_token,
};
#[cfg(not(target_os = "wasi"))]
pub use server::JsBlazenPeerServer;
pub use types::{
    JsDerefRequest, JsDerefResponse, JsPeerRemoteRefDescriptor, JsReleaseRequest,
    JsReleaseResponse, JsSubWorkflowRequest, JsSubWorkflowResponse,
};
