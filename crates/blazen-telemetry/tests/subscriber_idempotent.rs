//! Integration test for the reload-handle subscriber seam.
//!
//! The original `init_otlp` path would panic on a tokio worker thread
//! the second time a global subscriber was installed, which trips an
//! uncatchable SIGABRT in Python and Node hosts. After the
//! reload-handle rewrite, `init_otlp` should:
//!
//! 1. Succeed on the first call (`Ok(())`),
//! 2. Succeed on the second call without panicking,
//! 3. Survive even when `install_global_subscriber` has already run
//!    (the standard binding-import sequence).
//!
//! This test only runs when the `otlp-http` feature is enabled (which
//! is the public-HTTPS / Phoenix path; gRPC also goes through
//! `install_provider` and is exercised by the same code path).

#![cfg(feature = "otlp-http")]

use blazen_telemetry::{OtlpConfig, OtlpProtocol, init_otlp, install_global_subscriber};

#[tokio::test]
async fn init_otlp_twice_is_idempotent_and_panic_free() {
    // Mirror the binding-import sequence: install the shared subscriber
    // FIRST, then call `init_otlp` twice. Pre-fix, the second call would
    // panic; post-fix, both succeed and the second one just replaces the
    // first exporter in the reload slot.
    install_global_subscriber().expect("subscriber install must succeed on first call");

    let cfg = OtlpConfig::new("http://127.0.0.1:4318/v1/traces", "blazen-test")
        .with_protocol(OtlpProtocol::HttpProto);

    let first = init_otlp(cfg.clone());
    assert!(first.is_ok(), "first init_otlp should succeed: {first:?}");

    let second = init_otlp(cfg);
    assert!(
        second.is_ok(),
        "second init_otlp must not panic; got: {second:?}"
    );

    // A second call to install_global_subscriber must be a no-op too.
    install_global_subscriber().expect("idempotent installer must succeed when handle exists");
}
