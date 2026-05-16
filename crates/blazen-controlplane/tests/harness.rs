//! Shared `AssignmentHandler` implementations + test plumbing used by
//! the integration tests.
//!
//! Each `tests/*.rs` file in Cargo is its own binary, so this module is
//! pulled in via `mod harness;` from each test entry point. Items used
//! from a single test would emit dead-code warnings — that's expected
//! and silenced with `allow(dead_code)` on definitions that aren't used
//! everywhere.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use blazen_controlplane::protocol::{Assignment, Offer, OfferOutcome};
use blazen_controlplane::worker::{AssignmentContext, AssignmentFailure, AssignmentHandler};

/// Recorder for assignments seen by `RecordingHandler`. Tests poll via
/// [`Captured::wait_for_one`] which resolves once at least one
/// assignment lands.
#[derive(Clone, Default)]
pub struct Captured {
    inner: Arc<Mutex<Option<Assignment>>>,
}

impl Captured {
    pub async fn wait_for_one(&self) -> Assignment {
        loop {
            if let Some(a) = self.inner.lock().await.clone() {
                return a;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Captures the first assignment, then completes it with an echoing
/// JSON payload.
pub struct RecordingHandler {
    pub captured: Captured,
}

impl RecordingHandler {
    #[must_use]
    pub fn new(captured: Captured) -> Self {
        Self { captured }
    }
}

#[async_trait]
impl AssignmentHandler for RecordingHandler {
    async fn handle(
        &self,
        assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        // Stash a copy for the test assertion.
        *self.captured.inner.lock().await = Some(assignment.clone());
        // Echo the input back so the test can check the round-trip.
        let input: serde_json::Value =
            serde_json::from_slice(&assignment.input_json).unwrap_or(serde_json::Value::Null);
        Ok(serde_json::json!({ "echo": input }))
    }
}

/// Trivial echo handler — returns `{"echo": <input>}`.
pub struct EchoHandler;

#[async_trait]
impl AssignmentHandler for EchoHandler {
    async fn handle(
        &self,
        assignment: Assignment,
        _ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        let input: serde_json::Value =
            serde_json::from_slice(&assignment.input_json).unwrap_or(serde_json::Value::Null);
        Ok(serde_json::json!({ "echo": input }))
    }
}

/// Slow handler that blocks forever (until cancelled). Used by the
/// cancel test. Records whether `on_cancel` was invoked.
pub struct SlowHandler {
    pub cancel_seen: Arc<AtomicBool>,
}

#[async_trait]
impl AssignmentHandler for SlowHandler {
    async fn handle(
        &self,
        _assignment: Assignment,
        ctx: AssignmentContext,
    ) -> Result<serde_json::Value, AssignmentFailure> {
        // Park until our own cancellation token fires. The Worker will
        // fire it when the server sends Cancel.
        ctx.cancellation_token().cancelled().await;
        // Returning here means the handler's future has completed
        // post-cancel — the Worker's tokio::select! winner is the
        // token, not this branch, so this Ok value is discarded. We
        // still need a valid Result for the type system.
        Ok(serde_json::Value::Null)
    }

    async fn on_cancel(&self, _run_id: uuid::Uuid) {
        self.cancel_seen.store(true, Ordering::Relaxed);
    }

    async fn evaluate_offer(&self, _offer: &Offer) -> OfferOutcome {
        OfferOutcome::Claim
    }
}
