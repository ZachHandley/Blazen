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

use std::sync::{Arc, OnceLock};

pub use blazen_peer::auth::{PEER_TOKEN_ENV, resolve_peer_token};

/// What kind of peer authenticated. Surfaced on [`PeerIdentity`] so
/// handlers can apply role-specific policy later (e.g. admin-only RPCs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerKind {
    /// A worker node connecting to run assignments.
    Worker,
    /// An orchestrator submitting / observing workflows.
    Orchestrator,
    /// A privileged admin caller.
    Admin,
}

/// The server-resolved identity of an authenticated peer.
///
/// Produced by the installed [`PlaceAuthenticator`] from the inbound
/// bearer token and inserted into the tonic request extensions by the
/// interceptor. Handlers read it back to determine the tenant/place a
/// request operates in — the server-side `place` here WINS over any
/// client-set `place` on the wire (anti-spoof).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerIdentity {
    /// Tenant/place this peer belongs to. `"__default__"` for
    /// single-tenant / token-less deployments.
    pub place: String,
    /// Role of the peer.
    pub kind: PeerKind,
}

/// Maps an already-validated bearer token to a [`PeerIdentity`].
///
/// The interceptor validates the bearer with [`validate_bearer`] FIRST,
/// then calls the installed authenticator to derive identity. A custom
/// authenticator (e.g. one that decodes a JWT's `place` claim) can be
/// installed via [`install_place_authenticator`]; the [`DefaultPlaceAuthenticator`]
/// is used otherwise and maps every accepted peer to the default place.
pub trait PlaceAuthenticator: Send + Sync {
    /// Derive a [`PeerIdentity`] from the (already-validated) bearer
    /// header value, or return an error string to reject the request.
    ///
    /// # Errors
    ///
    /// Returns an error message string when the bearer cannot be mapped
    /// to a valid identity (the interceptor surfaces it as
    /// `Status::unauthenticated`).
    fn authenticate(&self, bearer: Option<&str>) -> Result<PeerIdentity, String>;
}

/// Blanket impl so a bare closure can serve as a [`PlaceAuthenticator`].
impl<F> PlaceAuthenticator for F
where
    F: Fn(Option<&str>) -> Result<PeerIdentity, String> + Send + Sync,
{
    fn authenticate(&self, bearer: Option<&str>) -> Result<PeerIdentity, String> {
        self(bearer)
    }
}

/// The default authenticator: maps every (already-validated) bearer to
/// the default place as an [`PeerKind::Orchestrator`]. Preserves
/// pre-tenancy behaviour — a token-less or shared-token deployment keeps
/// working exactly as before.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPlaceAuthenticator;

impl PlaceAuthenticator for DefaultPlaceAuthenticator {
    fn authenticate(&self, _bearer: Option<&str>) -> Result<PeerIdentity, String> {
        Ok(PeerIdentity {
            place: crate::protocol::DEFAULT_PLACE.to_string(),
            kind: PeerKind::Orchestrator,
        })
    }
}

/// Process-global authenticator slot. `None` until installed; the
/// accessor falls back to [`DefaultPlaceAuthenticator`].
static PLACE_AUTHENTICATOR: OnceLock<Arc<dyn PlaceAuthenticator>> = OnceLock::new();

/// Install a custom [`PlaceAuthenticator`]. Set-once (mirrors a
/// `OnceLock` slot): the first call wins and a deployment wires its
/// authenticator once at startup.
///
/// Returns `true` if this call installed the authenticator, or `false`
/// if one was already installed (in which case the existing one is kept).
pub fn install_place_authenticator(auth: Arc<dyn PlaceAuthenticator>) -> bool {
    PLACE_AUTHENTICATOR.set(auth).is_ok()
}

/// The installed [`PlaceAuthenticator`], or the
/// [`DefaultPlaceAuthenticator`] when none was installed.
#[must_use]
pub fn place_authenticator() -> Arc<dyn PlaceAuthenticator> {
    PLACE_AUTHENTICATOR
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(DefaultPlaceAuthenticator))
}

/// Build the `Bearer <token>` value for the `authorization` metadata
/// header. Returns `None` if no token is configured in the environment.
#[must_use]
pub fn bearer_metadata_value() -> Option<String> {
    resolve_peer_token().map(|t| format!("Bearer {t}"))
}

/// Build the `Bearer <token>` value, preferring an explicit `token` and
/// falling back to `BLAZEN_PEER_TOKEN` from the environment. Returns
/// `None` if neither is available.
///
/// Use this when a caller can supply a token directly (e.g. a JWT minted
/// per-connection) rather than relying on the process-global env var.
#[must_use]
pub fn bearer_metadata_value_with(token: Option<&str>) -> Option<String> {
    token
        .map(ToString::to_string)
        .or_else(resolve_peer_token)
        .map(|t| format!("Bearer {t}"))
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

        // `bearer_metadata_value_with`: an explicit token wins over the
        // env var.
        assert_eq!(
            bearer_metadata_value_with(Some("explicit")).as_deref(),
            Some("Bearer explicit"),
            "explicit token should take precedence over env"
        );
        // With no explicit token it falls back to the env var.
        assert_eq!(
            bearer_metadata_value_with(None).as_deref(),
            Some("Bearer s3cret"),
            "None should fall back to BLAZEN_PEER_TOKEN"
        );

        // With the env var unset: explicit still works, None => None.
        // SAFETY: see comment.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        assert_eq!(
            bearer_metadata_value_with(Some("only-explicit")).as_deref(),
            Some("Bearer only-explicit"),
            "explicit token works with no env"
        );
        assert_eq!(
            bearer_metadata_value_with(None),
            None,
            "no explicit + no env => None"
        );

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

    #[test]
    fn default_authenticator_maps_to_default_place() {
        let id = DefaultPlaceAuthenticator
            .authenticate(Some("Bearer whatever"))
            .expect("default authenticator never rejects");
        assert_eq!(id.place, crate::protocol::DEFAULT_PLACE);
        assert_eq!(id.kind, PeerKind::Orchestrator);
        // Token-less path too.
        let id_none = DefaultPlaceAuthenticator.authenticate(None).unwrap();
        assert_eq!(id_none.place, crate::protocol::DEFAULT_PLACE);
    }

    #[test]
    fn closure_authenticator_maps_token_to_place() {
        // A custom authenticator: the bearer's trailing word is the place.
        let auth = |bearer: Option<&str>| -> Result<PeerIdentity, String> {
            let header = bearer.ok_or_else(|| "missing bearer".to_string())?;
            let token = header
                .strip_prefix("Bearer ")
                .ok_or_else(|| "bad scheme".to_string())?;
            Ok(PeerIdentity {
                place: token.to_string(),
                kind: PeerKind::Worker,
            })
        };
        let id = PlaceAuthenticator::authenticate(&auth, Some("Bearer acme")).unwrap();
        assert_eq!(id.place, "acme");
        assert_eq!(id.kind, PeerKind::Worker);
        assert!(PlaceAuthenticator::authenticate(&auth, None).is_err());
    }
}
