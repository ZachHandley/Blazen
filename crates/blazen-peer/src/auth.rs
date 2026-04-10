//! Simple env-var-based peer authentication for non-mTLS deployments.
//!
//! When mutual TLS is not configured, peers can authenticate by presenting a
//! shared secret token carried in the `BLAZEN_PEER_TOKEN` environment variable.
//! This module provides [`resolve_peer_token`] which reads and validates the
//! env var at call time, and a constant for the env var name.

/// Environment variable name holding the peer-to-peer authentication token.
pub const PEER_TOKEN_ENV: &str = "BLAZEN_PEER_TOKEN";

/// Read the peer authentication token from the environment.
///
/// Returns `Some(token)` when `BLAZEN_PEER_TOKEN` is set to a non-empty
/// string, `None` otherwise.
#[must_use]
pub fn resolve_peer_token() -> Option<String> {
    std::env::var(PEER_TOKEN_ENV).ok().filter(|s| !s.is_empty())
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    // SAFETY note for all `unsafe` blocks in this module:
    // `set_var` and `remove_var` are unsafe in edition 2024 because env
    // vars are process-global shared mutable state. We consolidate all
    // env-var manipulation into a single `#[test]` to avoid races with
    // other tests that the harness may run in parallel. The single test
    // saves and restores the original value, so it is safe provided no
    // other code in this process touches BLAZEN_PEER_TOKEN concurrently
    // -- which nothing else does.

    /// Exercises all three branches of `resolve_peer_token`:
    /// unset, empty, and a real value. Runs as one test to avoid
    /// parallel-test races on the process-global env var.
    #[test]
    fn resolve_peer_token_env_var_branches() {
        // Save whatever was in the env before we started.
        let saved = std::env::var(PEER_TOKEN_ENV).ok();

        // --- unset ---
        // SAFETY: see module-level comment.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        assert!(
            resolve_peer_token().is_none(),
            "expected None when env var is unset"
        );

        // --- empty ---
        // SAFETY: see module-level comment.
        unsafe { std::env::set_var(PEER_TOKEN_ENV, "") };
        assert!(
            resolve_peer_token().is_none(),
            "expected None when env var is empty"
        );

        // --- non-empty ---
        // SAFETY: see module-level comment.
        unsafe { std::env::set_var(PEER_TOKEN_ENV, "test-secret-42") };
        assert_eq!(
            resolve_peer_token().as_deref(),
            Some("test-secret-42"),
            "expected the token value"
        );

        // --- restore ---
        // SAFETY: see module-level comment.
        unsafe { std::env::remove_var(PEER_TOKEN_ENV) };
        if let Some(val) = saved {
            // SAFETY: see module-level comment.
            unsafe { std::env::set_var(PEER_TOKEN_ENV, val) };
        }
    }
}
