//! Node bindings for the [`blazen_peer`] crate.
//!
//! Exposes the peer gRPC server and client to JS, alongside the
//! postcard wire types and helper functions for env-var auth and TLS
//! configuration.
//!
//! The `blazen-peer` dependency is pulled in with both its `server`
//! and `client` features enabled in this crate's `Cargo.toml`, so the
//! corresponding submodules are always available here.

pub mod client;
pub mod funcs;
pub mod server;
pub mod types;

pub use client::JsBlazenPeerClient;
pub use funcs::{
    load_client_tls, load_server_tls, peer_envelope_version, peer_token_env, resolve_beer_token,
};
pub use server::JsBlazenPeerServer;
pub use types::{
    JsDerefRequest, JsDerefResponse, JsPeerRemoteRefDescriptor, JsReleaseRequest,
    JsReleaseResponse, JsSubWorkflowRequest, JsSubWorkflowResponse,
};
