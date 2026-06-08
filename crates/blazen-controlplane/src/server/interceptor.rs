//! Tonic auth interceptor for the control-plane gRPC service.
//!
//! Reads the `authorization` metadata header from every incoming request
//! and delegates validation to [`crate::auth::validate_bearer`]. Returns
//! [`tonic::Status::unauthenticated`] when the header is missing, uses
//! the wrong scheme, or carries the wrong token.
//!
//! When `BLAZEN_PEER_TOKEN` is not set in the server's environment,
//! `validate_bearer` returns `Ok(())` for any input, effectively
//! disabling auth — useful for dev / loopback deployments.

use tonic::Status;
use tonic::metadata::MetadataMap;
use tonic::service::Interceptor;

use crate::auth::{place_authenticator, validate_bearer};

/// Interceptor that validates the bearer token on every gRPC call.
#[derive(Debug, Default, Clone)]
pub struct BearerAuthInterceptor;

impl BearerAuthInterceptor {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Interceptor for BearerAuthInterceptor {
    fn call(&mut self, mut request: tonic::Request<()>) -> Result<tonic::Request<()>, Status> {
        let header = extract_bearer(request.metadata());
        // Validate the bearer first (unchanged behaviour), then derive the
        // peer identity from it and stash it in the request extensions for
        // handlers to read. The server-side identity is the source of
        // truth for a request's place.
        validate_bearer(header.as_deref()).map_err(Status::unauthenticated)?;
        let identity = place_authenticator()
            .authenticate(header.as_deref())
            .map_err(Status::unauthenticated)?;
        request.extensions_mut().insert(identity);
        Ok(request)
    }
}

/// Extract the raw `authorization` metadata value as a String.
///
/// Returns `None` if the header is absent or contains invalid ASCII.
/// Binary metadata (`-bin` suffixed keys) is not consulted.
fn extract_bearer(meta: &MetadataMap) -> Option<String> {
    meta.get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::auth::PEER_TOKEN_ENV;

    fn mk_request(headers: &[(&str, &str)]) -> tonic::Request<()> {
        let mut req = tonic::Request::new(());
        for (k, v) in headers {
            let key: tonic::metadata::MetadataKey<tonic::metadata::Ascii> = (*k).parse().unwrap();
            let value = tonic::metadata::MetadataValue::try_from(*v).unwrap();
            req.metadata_mut().insert(key, value);
        }
        req
    }

    /// Consolidated env-var test — both "no token" and "token configured"
    /// branches run sequentially in one `#[test]` to avoid races with
    /// parallel test execution on the process-global `BLAZEN_PEER_TOKEN`.
    /// Mirrors the pattern in `crate::auth::tests::resolve_peer_token_env_var_branches`.
    #[test]
    fn interceptor_bearer_validation_branches() {
        let saved = std::env::var(PEER_TOKEN_ENV).ok();
        let mut interceptor = BearerAuthInterceptor::new();

        // ----- no token configured: accept anything -----
        // SAFETY: env vars are process-global; consolidate in one test.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        assert!(interceptor.call(mk_request(&[])).is_ok());
        assert!(
            interceptor
                .call(mk_request(&[("authorization", "Bearer whatever")]))
                .is_ok()
        );

        // ----- token configured: validate header -----
        // SAFETY: see comment.
        unsafe { std::env::set_var(PEER_TOKEN_ENV, "s3cret") };
        assert_eq!(
            interceptor.call(mk_request(&[])).unwrap_err().code(),
            tonic::Code::Unauthenticated
        );
        assert_eq!(
            interceptor
                .call(mk_request(&[("authorization", "Token s3cret")]))
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated
        );
        assert_eq!(
            interceptor
                .call(mk_request(&[("authorization", "Bearer wrong")]))
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated
        );
        assert!(
            interceptor
                .call(mk_request(&[("authorization", "Bearer s3cret")]))
                .is_ok()
        );

        // ----- restore -----
        // SAFETY: see comment.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        if let Some(v) = saved {
            // SAFETY: see comment.
            unsafe { std::env::set_var(PEER_TOKEN_ENV, v) };
        }
    }

    #[test]
    fn interceptor_inserts_default_identity_extension() {
        let saved = std::env::var(PEER_TOKEN_ENV).ok();
        let mut interceptor = BearerAuthInterceptor::new();

        // No token configured: the request is accepted and carries a
        // PeerIdentity for the default place.
        // SAFETY: env vars are process-global; consolidate in one test.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        let req = interceptor
            .call(mk_request(&[("authorization", "Bearer anything")]))
            .expect("accepted with no token configured");
        let id = req
            .extensions()
            .get::<crate::auth::PeerIdentity>()
            .expect("identity extension present");
        assert_eq!(id.place, crate::protocol::DEFAULT_PLACE);

        // SAFETY: see comment.
        if let Some(v) = saved {
            // SAFETY: see comment.
            unsafe { std::env::set_var(PEER_TOKEN_ENV, v) };
        }
    }
}
