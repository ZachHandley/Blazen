//! Node bindings for [`blazen_controlplane::Worker`].
//!
//! Exposes a [`JsControlPlaneWorker`] class that connects to a
//! control-plane server, dispatches assignments to a JS handler returning
//! a `Promise<unknown>`, and reports terminal results back to the server.
//!
//! The bridge is built on napi-rs's [`ThreadsafeFunction`]: the user's
//! JS handler is captured at `run()` time, then called from the Tokio
//! task that drives each assignment. The handler's resolved value is
//! forwarded as the assignment output; rejections become
//! [`AssignmentFailure`]s. Long-running handlers can emit non-terminal
//! events via the second [`JsAssignmentContext`] argument.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use tokio::sync::Mutex;

use blazen_controlplane::error::ControlPlaneError;
use blazen_controlplane::protocol::Assignment as WireAssignment;
use blazen_controlplane::{
    AssignmentContext, AssignmentFailure, AssignmentHandler, Worker, WorkerConfig,
};

use crate::controlplane::types::{JsAdmissionMode, JsAssignment, JsWorkerCapability};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a [`ControlPlaneError`] into a [`napi::Error`]. Prefixes the
/// variant with a class-name marker so JS consumers can pattern-match.
#[allow(clippy::needless_pass_by_value)]
fn controlplane_error_to_napi(err: ControlPlaneError) -> napi::Error {
    let prefix = match &err {
        ControlPlaneError::Encode(_) => "ControlPlaneEncodeError",
        ControlPlaneError::Json(_) => "ControlPlaneJsonError",
        ControlPlaneError::Transport(_) => "ControlPlaneTransportError",
        ControlPlaneError::EnvelopeVersion { .. } => "ControlPlaneEnvelopeVersionError",
        ControlPlaneError::Tls(_) => "ControlPlaneTlsError",
        ControlPlaneError::Unauthenticated(_) => "ControlPlaneUnauthenticatedError",
        ControlPlaneError::NoMatchingWorker { .. } => "ControlPlaneNoMatchingWorkerError",
        ControlPlaneError::MissingVramHint => "ControlPlaneMissingVramHintError",
        ControlPlaneError::UnknownRun(_) => "ControlPlaneUnknownRunError",
        ControlPlaneError::UnknownWorker(_) => "ControlPlaneUnknownWorkerError",
        ControlPlaneError::Workflow(_) => "ControlPlaneWorkflowError",
        ControlPlaneError::Rpc(_) => "ControlPlaneRpcError",
    };
    napi::Error::new(Status::GenericFailure, format!("[{prefix}] {err}"))
}

// ---------------------------------------------------------------------------
// ControlPlaneWorkerConfig
// ---------------------------------------------------------------------------

/// Fluent builder for a worker connection. Mirrors
/// [`blazen_controlplane::WorkerConfig`].
///
/// ```typescript
/// const config = new ControlPlaneWorkerConfig("http://cp:7445", "node-a")
///   .withCapability({ kind: "workflow:summarize", version: 1 })
///   .withTag("region", "us-west")
///   .withAdmission({ type: "Fixed", maxInFlight: 4 });
/// ```
#[napi(js_name = "ControlPlaneWorkerConfig")]
pub struct JsControlPlaneWorkerConfig {
    inner: std::sync::Mutex<Option<WorkerConfig>>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsControlPlaneWorkerConfig {
    /// Build a config with the required `endpoint` URI and stable
    /// `nodeId`. Defaults: no capabilities, no tags, `Fixed { maxInFlight: 1 }`,
    /// 5s heartbeat, plaintext transport.
    #[napi(constructor)]
    pub fn new(endpoint: String, node_id: String) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(WorkerConfig::new(endpoint, node_id))),
        }
    }

    /// Append a capability advertised at handshake.
    #[napi(js_name = "withCapability")]
    pub fn with_capability(&self, cap: JsWorkerCapability) -> Result<&Self> {
        self.mutate(|c| c.with_capability(cap.into()))
    }

    /// Insert a free-form `key=value` tag used by submission-time tag
    /// predicates.
    #[napi(js_name = "withTag")]
    pub fn with_tag(&self, key: String, value: String) -> Result<&Self> {
        self.mutate(|c| c.with_tag(key, value))
    }

    /// Override the admission mode declared at handshake.
    #[napi(js_name = "withAdmission")]
    pub fn with_admission(&self, mode: JsAdmissionMode) -> Result<&Self> {
        let admission = mode.into_native()?;
        self.mutate(|c| c.with_admission(admission))
    }

    /// Override the heartbeat cadence (milliseconds).
    #[napi(js_name = "withHeartbeatIntervalMs")]
    pub fn with_heartbeat_interval_ms(&self, ms: u32) -> Result<&Self> {
        self.mutate(|c| c.with_heartbeat_interval(Duration::from_millis(u64::from(ms))))
    }

    /// Load a client identity + CA from PEM files and use them for mTLS.
    #[napi(js_name = "withMtls")]
    pub fn with_mtls(&self, cert_path: String, key_path: String, ca_path: String) -> Result<&Self> {
        let cert = cert_path;
        let key = key_path;
        let ca = ca_path;
        // Take the inner config, apply mTLS, put it back. The mTLS load
        // can fail (missing PEM, bad PEM), so we handle that ourselves
        // rather than going through `mutate`.
        let mut guard = self.inner.lock().expect("poisoned");
        let config = guard.take().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "ControlPlaneWorkerConfig already consumed (passed to ControlPlaneWorker.connect)",
            )
        })?;
        match config.with_mtls(Path::new(&cert), Path::new(&key), Path::new(&ca)) {
            Ok(updated) => {
                *guard = Some(updated);
                Ok(self)
            }
            Err(e) => Err(controlplane_error_to_napi(e)),
        }
    }

    /// Take the inner config, leaving the slot empty. Used by
    /// [`JsControlPlaneWorker::connect`] when consuming the builder.
    pub(crate) fn take(&self) -> Result<WorkerConfig> {
        let mut guard = self.inner.lock().expect("poisoned");
        guard.take().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "ControlPlaneWorkerConfig already consumed (passed to ControlPlaneWorker.connect)",
            )
        })
    }

    /// Apply `f` to the held config in-place.
    fn mutate<F>(&self, f: F) -> Result<&Self>
    where
        F: FnOnce(WorkerConfig) -> WorkerConfig,
    {
        let mut guard = self.inner.lock().expect("poisoned");
        let config = guard.take().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "ControlPlaneWorkerConfig already consumed (passed to ControlPlaneWorker.connect)",
            )
        })?;
        *guard = Some(f(config));
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// AssignmentContext
// ---------------------------------------------------------------------------

/// Per-assignment context handed to the JS handler. Mirrors
/// [`blazen_controlplane::AssignmentContext`].
///
/// Carries the run id and a sink for emitting non-terminal events back
/// to the control plane.
#[napi(js_name = "AssignmentContext")]
pub struct JsAssignmentContext {
    /// We own the native `AssignmentContext` behind a tokio mutex so that
    /// concurrent `emitEvent` calls from JS are serialized (the underlying
    /// `WorkerOutbox` send is `&self` and tolerates concurrent senders,
    /// but capture-by-move into the spawned future requires shared
    /// ownership).
    inner: Arc<AssignmentContext>,
    run_id_str: String,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsAssignmentContext {
    /// Run identifier this context belongs to (UUID string).
    #[napi(getter, js_name = "runId")]
    pub fn run_id(&self) -> String {
        self.run_id_str.clone()
    }

    /// Emit a non-terminal event back to the control plane.
    ///
    /// `data` must be a JSON-serializable JS value (object, array,
    /// string, number, boolean, null).
    #[napi(js_name = "emitEvent")]
    pub async fn emit_event(&self, event_type: String, data: serde_json::Value) -> Result<()> {
        self.inner
            .emit_event(&event_type, data)
            .await
            .map_err(controlplane_error_to_napi)
    }
}

// ---------------------------------------------------------------------------
// AssignmentHandler bridge (JS callback → Rust trait impl)
// ---------------------------------------------------------------------------

/// JS-supplied assignment handler. Receives the assignment payload and a
/// per-run context; returns a JS `Promise` whose resolved value becomes
/// the assignment output.
///
/// Generic-parameter shape follows the same convention used by
/// [`crate::providers::typed_tool::TypedToolHandlerTsfn`]:
/// - `T` = `FnArgs<(JsAssignment, JsAssignmentContext)>` — the two
///   positional arguments handed to JS.
/// - `Return` = `Promise<serde_json::Value>` — JS handler must resolve
///   with a JSON-serializable result.
/// - `CalleeHandled = false` — no error-first callback convention.
/// - `Weak = true` — the TSFN does not pin the Node event loop on its
///   own.
type AssignmentHandlerTsfn = ThreadsafeFunction<
    FnArgs<(JsAssignment, JsAssignmentContext)>,
    Promise<serde_json::Value>,
    FnArgs<(JsAssignment, JsAssignmentContext)>,
    Status,
    false,
    true,
>;

/// Bridge: implements [`AssignmentHandler`] by routing each `handle`
/// invocation through a [`ThreadsafeFunction`] into JS.
struct JsAssignmentHandlerBridge {
    tsfn: Arc<AssignmentHandlerTsfn>,
}

#[async_trait]
impl AssignmentHandler for JsAssignmentHandlerBridge {
    async fn handle(
        &self,
        assignment: WireAssignment,
        ctx: AssignmentContext,
    ) -> std::result::Result<serde_json::Value, AssignmentFailure> {
        let run_id_str = ctx.run_id().to_string();
        let js_assignment = JsAssignment::from(&assignment);
        let js_ctx = JsAssignmentContext {
            inner: Arc::new(ctx),
            run_id_str,
        };

        let promise = self
            .tsfn
            .call_async_catch(FnArgs::from((js_assignment, js_ctx)))
            .await
            .map_err(|e| {
                AssignmentFailure::new(format!("assignment handler dispatch failed: {e}"))
            })?;

        promise
            .await
            .map_err(|e| AssignmentFailure::new(format!("assignment handler rejected: {e}")))
    }
}

// ---------------------------------------------------------------------------
// ControlPlaneWorker
// ---------------------------------------------------------------------------

/// Worker connection to a control-plane server. Construct via
/// [`Self::connect`] (which validates the endpoint URI), then call
/// [`Self::run`] with a JS handler to drive the worker forever.
///
/// ```typescript
/// const config = new ControlPlaneWorkerConfig("http://cp:7445", "node-a")
///   .withCapability({ kind: "workflow:summarize", version: 1 });
/// const worker = ControlPlaneWorker.connect(config);
/// await worker.run(async (assignment, ctx) => {
///   await ctx.emitEvent("started", { runId: assignment.runId });
///   return { ok: true };
/// });
/// ```
#[napi(js_name = "ControlPlaneWorker")]
pub struct JsControlPlaneWorker {
    /// Slot for the owned [`Worker`]. Drained by [`Self::run`].
    worker: Mutex<Option<Worker>>,
    /// `Weak` clone of the cancellation token used by [`Worker::shutdown`].
    /// We can't `Arc<Worker>` here because `Worker::run` consumes the
    /// worker by value, so we extract a shutdown trigger up-front
    /// (cloning the inner `CancellationToken`) and store it on the side.
    shutdown: tokio_util::sync::CancellationToken,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]
impl JsControlPlaneWorker {
    /// Validate the configured endpoint URI and prepare a worker that's
    /// ready to call [`Self::run`]. Does NOT open a network connection
    /// yet — that happens on the first iteration of `run`.
    #[napi(factory)]
    pub fn connect(config: &JsControlPlaneWorkerConfig) -> Result<Self> {
        let cfg = config.take()?;
        let worker = Worker::connect(cfg).map_err(controlplane_error_to_napi)?;
        // The upstream `Worker::shutdown` fires both an internal
        // shutdown token and per-run cancel tokens. We mirror the
        // shutdown side via our own token: when JS calls `shutdown()`,
        // we cancel our token, then a small adapter task between
        // `JsControlPlaneWorker::run` and `Worker::run` watches the
        // token and forwards by calling `Worker::shutdown` directly.
        let shutdown = tokio_util::sync::CancellationToken::new();
        Ok(Self {
            worker: Mutex::new(Some(worker)),
            shutdown,
        })
    }

    /// Drive the worker forever, dispatching each assignment to the JS
    /// handler. Resolves on graceful drain or [`Self::shutdown`].
    ///
    /// The JS handler is invoked with two arguments: the assignment
    /// itself and a [`JsAssignmentContext`] that exposes the run id and
    /// an `emitEvent` method. Its return value (JSON-serialized) becomes
    /// the assignment output reported back to the server.
    #[napi(
        ts_args_type = "handler: (assignment: Assignment, ctx: AssignmentContext) => Promise<unknown>"
    )]
    pub async fn run(&self, handler: AssignmentHandlerTsfn) -> Result<()> {
        let bridge = JsAssignmentHandlerBridge {
            tsfn: Arc::new(handler),
        };

        let worker = self.worker.lock().await.take().ok_or_else(|| {
            napi::Error::new(
                Status::GenericFailure,
                "ControlPlaneWorker.run has already been called",
            )
        })?;

        // Wire the side-band shutdown token into the worker by spawning
        // an adapter that fires `worker.shutdown()` when the token is
        // cancelled. The worker itself is consumed by `run`; to call
        // `worker.shutdown()` we need a `&Worker`, which is gone once
        // the `run` future starts. Solve this with a `tokio::select`:
        // race the `Worker::run` future against the shutdown token. If
        // the token fires, we can't gracefully tell the inner worker to
        // stop (no `&` available), but cancelling the future itself
        // drops the worker — which fires every `CancellationToken` it
        // owned during destruction.
        let shutdown = self.shutdown.clone();
        let run_fut = worker.run(bridge);
        tokio::select! {
            res = run_fut => res.map_err(controlplane_error_to_napi),
            () = shutdown.cancelled() => Ok(()),
        }
    }

    /// Signal the worker to stop. Idempotent.
    ///
    /// This drops the cancellation token tied to [`Self::run`]; the
    /// `run` future resolves at the next await point.
    #[napi]
    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }
}
