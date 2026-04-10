//! mTLS configuration helpers for Blazen peer connections.
//!
//! Provides [`load_server_tls`] and [`load_client_tls`] which read PEM-encoded
//! certificate, key, and CA files from disk and produce the corresponding
//! [`tonic::transport::ServerTlsConfig`] / [`ClientTlsConfig`] values. These
//! are thin wrappers around tonic's [`Identity`] and [`Certificate`] types --
//! all the heavy rustls wiring happens inside tonic itself.
//!
//! [`Identity`]: tonic::transport::Identity
//! [`Certificate`]: tonic::transport::Certificate

use std::path::Path;

use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

use crate::error::PeerError;

/// Load a server-side TLS configuration from PEM files.
///
/// - `cert_pem_path` -- the server's certificate chain (one or more PEM-encoded
///   certificates, leaf first).
/// - `key_pem_path` -- the server's private key in PEM format.
/// - `ca_pem_path` -- the CA certificate used to verify client certificates.
///   Setting this enables mutual TLS: any connecting client must present a
///   certificate signed by this CA.
///
/// Both the server and any connecting clients must present certificates signed
/// by the same CA for mutual TLS to succeed.
///
/// # Errors
///
/// Returns [`PeerError::Tls`] if any of the PEM files cannot be read.
pub fn load_server_tls(
    cert_pem_path: &Path,
    key_pem_path: &Path,
    ca_pem_path: &Path,
) -> Result<ServerTlsConfig, PeerError> {
    let cert_pem = std::fs::read(cert_pem_path).map_err(|e| {
        PeerError::Tls(format!(
            "failed to read server cert {}: {e}",
            cert_pem_path.display()
        ))
    })?;
    let key_pem = std::fs::read(key_pem_path).map_err(|e| {
        PeerError::Tls(format!(
            "failed to read server key {}: {e}",
            key_pem_path.display()
        ))
    })?;
    let ca_pem = std::fs::read(ca_pem_path).map_err(|e| {
        PeerError::Tls(format!(
            "failed to read CA cert {}: {e}",
            ca_pem_path.display()
        ))
    })?;

    let identity = Identity::from_pem(cert_pem, key_pem);
    let ca_cert = Certificate::from_pem(ca_pem);

    Ok(ServerTlsConfig::new()
        .identity(identity)
        .client_ca_root(ca_cert))
}

/// Load a client-side TLS configuration from PEM files.
///
/// - `cert_pem_path` -- the client's certificate chain (one or more
///   PEM-encoded certificates, leaf first).
/// - `key_pem_path` -- the client's private key in PEM format.
/// - `ca_pem_path` -- the CA certificate used to verify the server's
///   certificate.
///
/// The resulting [`ClientTlsConfig`] presents the client identity for mutual
/// TLS and trusts the supplied CA for server verification.
///
/// # Errors
///
/// Returns [`PeerError::Tls`] if any of the PEM files cannot be read.
pub fn load_client_tls(
    cert_pem_path: &Path,
    key_pem_path: &Path,
    ca_pem_path: &Path,
) -> Result<ClientTlsConfig, PeerError> {
    let cert_pem = std::fs::read(cert_pem_path).map_err(|e| {
        PeerError::Tls(format!(
            "failed to read client cert {}: {e}",
            cert_pem_path.display()
        ))
    })?;
    let key_pem = std::fs::read(key_pem_path).map_err(|e| {
        PeerError::Tls(format!(
            "failed to read client key {}: {e}",
            key_pem_path.display()
        ))
    })?;
    let ca_pem = std::fs::read(ca_pem_path).map_err(|e| {
        PeerError::Tls(format!(
            "failed to read CA cert {}: {e}",
            ca_pem_path.display()
        ))
    })?;

    let identity = Identity::from_pem(cert_pem, key_pem);
    let ca_cert = Certificate::from_pem(ca_pem);

    Ok(ClientTlsConfig::new()
        .identity(identity)
        .ca_certificate(ca_cert))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Verify that reading a missing file produces a `PeerError::Tls`.
    #[test]
    fn load_server_tls_missing_file() {
        let missing = Path::new("/tmp/blazen_nonexistent_cert.pem");
        let err = load_server_tls(missing, missing, missing);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("failed to read"), "unexpected error: {msg}");
    }

    /// Verify that reading a missing file produces a `PeerError::Tls`.
    #[test]
    fn load_client_tls_missing_file() {
        let missing = Path::new("/tmp/blazen_nonexistent_cert.pem");
        let err = load_client_tls(missing, missing, missing);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("failed to read"), "unexpected error: {msg}");
    }

    /// Verify that readable (though not necessarily valid) PEM files
    /// produce `Ok(...)` from the loaders. The tonic types accept any
    /// bytes -- validation happens at TLS handshake time, not at config
    /// construction time.
    #[test]
    fn load_tls_with_dummy_pem_files() {
        let mut cert = NamedTempFile::new().unwrap();
        let mut key = NamedTempFile::new().unwrap();
        let mut ca = NamedTempFile::new().unwrap();

        // Dummy PEM content -- not real certs, but the loaders only read
        // bytes and hand them to tonic::Identity / tonic::Certificate.
        write!(
            cert,
            "-----BEGIN CERTIFICATE-----\nZHVtbXk=\n-----END CERTIFICATE-----"
        )
        .unwrap();
        write!(
            key,
            "-----BEGIN PRIVATE KEY-----\nZHVtbXk=\n-----END PRIVATE KEY-----"
        )
        .unwrap();
        write!(
            ca,
            "-----BEGIN CERTIFICATE-----\nZHVtbXk=\n-----END CERTIFICATE-----"
        )
        .unwrap();

        assert!(load_server_tls(cert.path(), key.path(), ca.path()).is_ok());
        assert!(load_client_tls(cert.path(), key.path(), ca.path()).is_ok());
    }
}
