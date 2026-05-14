//! # Blazen Control Plane
//!
//! Distributed workflow control plane for Blazen. Unlike [`blazen_peer`],
//! which models a flat mesh of peer processes that dial each other
//! directly, this crate provides a **central server that workers connect
//! into**. The control plane owns the authoritative view of running
//! workflows, the worker registry, and the assignment queue; workers and
//! orchestrators are clients of it.
//!
//! ## Topology
//!
//! ```text
//!   orchestrator ──┐
//!                  │  unary + server-stream RPCs
//!                  ▼
//!         BlazenControlPlane (server)
//!                  ▲
//!                  │  bidi `WorkerSession` stream
//!   worker ────────┘  (worker opens the connection)
//! ```
//!
//! Workers always open the connection outbound. This inverts the
//! [`blazen_peer`] model (where peers are equal and either side can dial)
//! and makes the system NAT-friendly: only the control plane needs to be
//! reachable on a public address.
//!
//! ## Admission modes
//!
//! Three admission modes will be supported (see [`server`]):
//!
//! 1. **Open** — any client with the shared bearer token can register.
//! 2. **Allowlist** — workers must present an identity that matches a
//!    configured allowlist (mTLS subject DN, JWT claim, …).
//! 3. **Signed-handshake** — workers present a signed enrollment token
//!    issued out-of-band; the control plane verifies the signature.
//!
//! ## Wire format
//!
//! The gRPC schema (see [`pb`]) is intentionally minimal: every RPC takes
//! a single `PostcardRequest { bytes postcard_payload }` and returns a
//! `PostcardResponse { bytes postcard_payload }`. The actual payload
//! types live in [`protocol`] and are serialized with [`postcard`].
//! Per-message versioning is carried by an `envelope_version` field on
//! each payload struct (see [`protocol::ENVELOPE_VERSION`]) — the proto
//! schema itself never has to change to add a new field.
//!
//! ## Feature flags
//!
//! - `server` (default): build the tonic server-side service.
//! - `client` (default): build the tonic client stubs used by both
//!   workers and orchestrators.
//! - `http-transport`: build an [`axum`]-based HTTP/SSE bridge for
//!   environments that cannot speak HTTP/2 (browsers, some serverless
//!   platforms). Mirrors the trick [`blazen_peer`] uses for the
//!   wasi-http fallback.

// The generated tonic/prost types and the gRPC server / client that consume
// them are unavailable on wasm32-wasi* (tonic does not compile there). The
// `http` module under the `http-transport` feature is the wasm-friendly
// alternative for those targets.
#[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
#[allow(clippy::all, clippy::pedantic)]
pub mod pb {
    //! Generated tonic/prost types for the `blazen.controlplane.v1`
    //! service.
    //!
    //! This module is produced by [`tonic_prost_build`] from
    //! `proto/blazen_controlplane.proto` at build time. The contents are
    //! intentionally not part of the public API — consumers should use
    //! [`super::protocol`] for the postcard-encoded payload types and the
    //! [`super::server`] / [`super::client`] / [`super::worker`] modules
    //! for the transport.
    tonic::include_proto!("blazen.controlplane.v1");
}

pub mod auth;
pub mod error;
pub mod protocol;

#[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
pub mod tls;

#[cfg(all(
    feature = "server",
    not(any(target_os = "wasi", target_arch = "wasm32"))
))]
pub mod server;

#[cfg(feature = "client")]
pub mod worker;

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "http-transport")]
pub mod http;

pub use error::ControlPlaneError;
