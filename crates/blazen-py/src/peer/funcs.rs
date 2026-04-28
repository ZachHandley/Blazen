//! Free functions exposed from `blazen_peer` to Python.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::peer::error::peer_err;

/// Read the peer authentication token from the environment.
///
/// Returns the value of ``BLAZEN_PEER_TOKEN`` when it is set to a
/// non-empty string, ``None`` otherwise.
#[gen_stub_pyfunction]
#[pyfunction]
#[must_use]
pub fn resolve_peer_token() -> Option<String> {
    blazen_peer::auth::resolve_peer_token()
}

/// Load a server-side TLS configuration from PEM files.
///
/// All three files are read eagerly; the returned value is an opaque
/// success indicator (the underlying `tonic::transport::ServerTlsConfig`
/// is not exposed to Python). Use this from a wrapper that wires the
/// result into a server builder, or as a smoke check that the PEM files
/// exist and are readable.
///
/// Args:
///     cert_pem_path: Path to the server certificate chain (PEM).
///     key_pem_path: Path to the server private key (PEM).
///     ca_pem_path: Path to the CA certificate used to verify clients (PEM).
///
/// Raises:
///     PeerError: If any of the PEM files cannot be read.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn load_server_tls(
    cert_pem_path: PathBuf,
    key_pem_path: PathBuf,
    ca_pem_path: PathBuf,
) -> PyResult<()> {
    blazen_peer::tls::load_server_tls(&cert_pem_path, &key_pem_path, &ca_pem_path)
        .map(|_| ())
        .map_err(peer_err)
}

/// Load a client-side TLS configuration from PEM files.
///
/// Args:
///     cert_pem_path: Path to the client certificate chain (PEM).
///     key_pem_path: Path to the client private key (PEM).
///     ca_pem_path: Path to the CA certificate used to verify the server (PEM).
///
/// Raises:
///     PeerError: If any of the PEM files cannot be read.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn load_client_tls(
    cert_pem_path: PathBuf,
    key_pem_path: PathBuf,
    ca_pem_path: PathBuf,
) -> PyResult<()> {
    blazen_peer::tls::load_client_tls(&cert_pem_path, &key_pem_path, &ca_pem_path)
        .map(|_| ())
        .map_err(peer_err)
}
