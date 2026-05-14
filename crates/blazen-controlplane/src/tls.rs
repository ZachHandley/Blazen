//! mTLS configuration helpers for control-plane connections.
//!
//! Re-exports [`blazen_peer::tls::load_server_tls`] and
//! [`blazen_peer::tls::load_client_tls`]. The control plane uses the
//! same rustls-backed PEM-loading code as the peer crate; there is no
//! TLS surface specific to the control plane.
//!
//! See [`blazen_peer::tls`] for the underlying implementation and PEM
//! file format expectations.

pub use blazen_peer::tls::{load_client_tls, load_server_tls};
