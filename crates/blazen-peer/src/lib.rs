//! # Blazen Peer
//!
//! Distributed workflow execution layer for Blazen. This crate exposes a
//! gRPC service ([`server`]) that lets one Blazen process invoke
//! sub-workflows on another, plus a matching client ([`client`]) for
//! making those calls. Together they form the foundation for running a
//! Blazen pipeline across multiple processes or machines.
//!
//! ## Architecture
//!
//! - **Transport.** [`tonic`] over HTTP/2, with optional mTLS via
//!   [`rustls`] + `aws-lc-rs`. This gives us multiplexed streams,
//!   built-in flow control, and a battle-tested wire format without
//!   having to roll our own framing.
//!
//! - **Wire format.** The gRPC schema (see [`pb`]) is intentionally
//!   minimal: every RPC takes a single
//!   `PostcardRequest { bytes postcard_payload }` and returns a
//!   `PostcardResponse { bytes postcard_payload }`. The actual payload
//!   types live in [`protocol`] and are serialized with [`postcard`].
//!
//!   Why postcard-in-bytes instead of richer proto messages?
//!
//!   1. **Schema evolution lives in one place.** Adding a field to
//!      [`protocol::SubWorkflowRequest`] does not require regenerating
//!      proto bindings on every consumer. Versioning is handled by the
//!      [`protocol::ENVELOPE_VERSION`] constant on each payload, not by
//!      the proto schema.
//!   2. **Reuse Rust types.** The structs we move across the wire are
//!      the same ones blazen-core already speaks; postcard means we do
//!      not have to maintain a parallel proto-IDL definition for every
//!      Rust type that a workflow can pass around.
//!   3. **`bincode` is not an option.** As of December 2025 the bincode
//!      crate is unmaintained. postcard is the actively-maintained,
//!      no-std-friendly alternative we've standardized on.
//!
//! - **Session refs.** When a remote sub-workflow returns a value that
//!   the parent cannot easily serialize (an open file handle, a model
//!   weight cache, …), the peer registers it in its local
//!   [`blazen_core::SessionRefRegistry`] and returns a
//!   [`protocol::RemoteRefDescriptor`]. The parent can then dereference
//!   the ref over the same gRPC channel via `DerefSessionRef`, and
//!   release it via `ReleaseSessionRef` when done.
//!
//! ## Phase status
//!
//! This crate is the **scaffold** for Phase 12.3 of the distributed
//! workflow roadmap. The proto compiles and the module tree builds
//! cleanly, but [`server`] and [`client`] are stubs — Phases 12.6 and
//! 12.7 will wire up the tonic service trait and the client stub. The
//! TLS helpers in [`tls`] are similarly deferred to Phase 12.9.
//!
//! ## Feature flags
//!
//! - `server` (default): build the tonic server side. Pulls in the
//!   generated `..._server` module from [`pb`].
//! - `client` (default): build the tonic client side. Pulls in the
//!   generated `..._client` module from [`pb`].
//!
//! Both features are independent; a node that only ever invokes remote
//! workflows can disable `server`, and a worker node that only ever
//! receives invocations can disable `client`.

#[allow(clippy::all, clippy::pedantic)]
pub mod pb {
    //! Generated tonic/prost types for the `blazen.peer.v1` service.
    //!
    //! This module is produced by [`tonic_prost_build`] from
    //! `proto/blazen_peer.proto` at build time. The contents are
    //! intentionally not part of the public API — consumers should use
    //! [`super::protocol`] for the postcard-encoded payload types and
    //! the [`super::server`] / [`super::client`] modules for the
    //! transport.
    tonic::include_proto!("blazen.peer.v1");
}

pub mod auth;
pub mod error;
pub mod protocol;
pub mod tls;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "client")]
pub mod client;

pub use error::PeerError;
pub use protocol::{
    DerefRequest, DerefResponse, ENVELOPE_VERSION, ReleaseRequest, ReleaseResponse,
    RemoteRefDescriptor, SubWorkflowRequest, SubWorkflowResponse,
};

#[cfg(feature = "server")]
pub use server::BlazenPeerServer;

#[cfg(feature = "client")]
pub use client::BlazenPeerClient;
