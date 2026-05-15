//! Free functions and module-level constants exposed to JS for the
//! peer transport.
//!
//! `napi-rs` does not have a direct equivalent to `PyO3`'s module-level
//! constants, so versioned values are exposed as zero-argument
//! `#[napi]` functions that return the constant. The auth and TLS
//! helpers are exposed as plain `#[napi]` free functions.

use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_peer::auth;
use blazen_peer::protocol::ENVELOPE_VERSION;
use blazen_peer::tls;

use crate::error::peer_error_to_napi;

// ---------------------------------------------------------------------------
// Constants exposed as getter functions
// ---------------------------------------------------------------------------

/// Current envelope version spoken by this build of `blazen-peer`.
#[napi(js_name = "peerEnvelopeVersion")]
#[must_use]
pub fn peer_envelope_version() -> u32 {
    ENVELOPE_VERSION
}

/// Environment variable name used to carry the peer auth token.
#[napi(js_name = "peerTokenEnv")]
#[must_use]
pub fn peer_token_env() -> String {
    auth::PEER_TOKEN_ENV.to_string()
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

/// Read the peer authentication token from the process environment.
///
/// Returns `null` when the env var (see [`peer_token_env`]) is unset
/// or empty.
#[napi(js_name = "resolvePeerToken")]
#[must_use]
pub fn resolve_peer_token() -> Option<String> {
    auth::resolve_peer_token()
}

// ---------------------------------------------------------------------------
// TLS
// ---------------------------------------------------------------------------

/// Validate that the supplied PEM files exist and can be loaded into a
/// server-side TLS configuration.
///
/// The native `ServerTlsConfig` is not directly representable across
/// the JS boundary, so this function only surfaces the load result --
/// returning `true` on success and throwing `peerErrorToNapi`-style on
/// failure. Wire the actual config into a server in Rust, not in JS.
#[napi(js_name = "loadServerTls")]
#[allow(clippy::missing_errors_doc)]
pub fn load_server_tls(
    cert_pem_path: String,
    key_pem_path: String,
    ca_pem_path: String,
) -> Result<bool> {
    let cert: PathBuf = cert_pem_path.into();
    let key: PathBuf = key_pem_path.into();
    let ca: PathBuf = ca_pem_path.into();
    tls::load_server_tls(&cert, &key, &ca)
        .map(|_| true)
        .map_err(peer_error_to_napi)
}

/// Validate that the supplied PEM files exist and can be loaded into a
/// client-side TLS configuration.
///
/// See [`load_server_tls`] for the same caveats about the native
/// config not crossing the JS boundary.
#[napi(js_name = "loadClientTls")]
#[allow(clippy::missing_errors_doc)]
pub fn load_client_tls(
    cert_pem_path: String,
    key_pem_path: String,
    ca_pem_path: String,
) -> Result<bool> {
    let cert: PathBuf = cert_pem_path.into();
    let key: PathBuf = key_pem_path.into();
    let ca: PathBuf = ca_pem_path.into();
    tls::load_client_tls(&cert, &key, &ca)
        .map(|_| true)
        .map_err(peer_error_to_napi)
}
