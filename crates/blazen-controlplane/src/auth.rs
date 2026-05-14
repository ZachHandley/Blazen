//! Authentication for control-plane connections.
//!
//! Reuses the same shared-secret + mTLS surface as [`blazen_peer::auth`].
//! Workers and orchestrators may authenticate via:
//!
//! 1. mTLS (recommended) — server is configured with a CA cert; clients
//!    present an identity signed by that CA. See [`super::tls`].
//! 2. Shared-secret bearer token in the `BLAZEN_PEER_TOKEN` env var,
//!    carried as the `authorization: Bearer <token>` metadata on every
//!    request.
//!
//! This module exposes:
//!
//! - The env-var constant and reader, re-exported from
//!   [`blazen_peer::auth`].
//! - [`bearer_metadata_value`], for building the `authorization` header
//!   on the client side.
//! - [`validate_bearer`], for verifying that header on the server side.
//!
//! ## Server-side wiring
//!
//! The control-plane server installs a tonic [`tonic::service::Interceptor`]
//! (or the axum equivalent for the HTTP/SSE tier) that calls
//! [`validate_bearer`] on every incoming request and returns
//! `Status::unauthenticated` on failure.

pub use blazen_peer::auth::{PEER_TOKEN_ENV, resolve_peer_token};

/// Build the `Bearer <token>` value for the `authorization` metadata
/// header. Returns `None` if no token is configured in the environment.
#[must_use]
pub fn bearer_metadata_value() -> Option<String> {
    resolve_peer_token().map(|t| format!("Bearer {t}"))
}

/// Validate a bearer header value against the configured token.
///
/// Returns `Ok(())` when:
/// - the server has no `BLAZEN_PEER_TOKEN` configured (auth is
///   effectively off — useful for dev / loopback deployments), OR
/// - the supplied `header_value` is `Some("Bearer <token>")` and the
///   token matches.
///
/// Returns `Err(reason)` when the server has a token configured but the
/// header is missing or wrong. The error string is suitable for
/// surfacing as a `tonic::Status::unauthenticated` message.
///
/// # Errors
///
/// Returns an error message string when the header is missing, the
/// scheme is wrong, or the token does not match.
pub fn validate_bearer(header_value: Option<&str>) -> Result<(), String> {
    let Some(expected) = resolve_peer_token() else {
        // No token configured server-side — auth is off.
        return Ok(());
    };
    let Some(header) = header_value else {
        return Err("missing authorization header".into());
    };
    let Some(presented) = header.strip_prefix("Bearer ") else {
        return Err("authorization header must use Bearer scheme".into());
    };
    if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
        Ok(())
    } else {
        Err("token mismatch".into())
    }
}

/// Constant-time byte-slice comparison. Avoids leaking byte-position of
/// the first mismatch via timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    // Same env-var safety pattern as blazen-peer's auth tests: we
    // serialize all env-var manipulation into a single test to avoid
    // races with parallel-test runs.
    #[test]
    fn validate_bearer_branches() {
        let saved = std::env::var(PEER_TOKEN_ENV).ok();

        // No token configured — anything is accepted.
        // SAFETY: see comment.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        assert!(validate_bearer(None).is_ok(), "no token => accept");
        assert!(validate_bearer(Some("Bearer anything")).is_ok());

        // Token configured.
        // SAFETY: see comment.
        unsafe { std::env::set_var(PEER_TOKEN_ENV, "s3cret") };
        assert!(validate_bearer(None).is_err(), "missing header => deny");
        assert!(
            validate_bearer(Some("Token s3cret")).is_err(),
            "wrong scheme => deny"
        );
        assert!(
            validate_bearer(Some("Bearer wrong")).is_err(),
            "wrong token => deny"
        );
        assert!(
            validate_bearer(Some("Bearer s3cret")).is_ok(),
            "match => accept"
        );
        assert_eq!(bearer_metadata_value().as_deref(), Some("Bearer s3cret"));

        // Restore.
        // SAFETY: see comment.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        if let Some(val) = saved {
            // SAFETY: see comment.
            unsafe { std::env::set_var(PEER_TOKEN_ENV, val) };
        }
    }

    #[test]
    fn constant_time_eq_truth_table() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
