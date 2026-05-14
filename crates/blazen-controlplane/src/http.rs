//! HTTP/SSE bridge for the `BlazenControlPlane`.
//!
//! Mirrors the gRPC tier in [`crate::server`] but framed as plain HTTP
//! plus Server-Sent Events, so that environments unable to speak
//! HTTP/2 bidi gRPC (browsers, wasi runtimes, some serverless
//! platforms) can still act as workers or orchestrators. The two tiers
//! share the same [`SharedState`] (registry, queue, admission), so a
//! worker connected via gRPC bidi and a worker connected via SSE are
//! indistinguishable from the orchestrator's perspective.
//!
//! ## Routes
//!
//! ### Worker tier
//! - `POST /v1/cp/worker/register` — body: postcard-encoded
//!   [`crate::protocol::WorkerHello`] (base64 in JSON). Returns
//!   `{session_id, welcome}`.
//! - `GET  /v1/cp/worker/{session_id}/stream` — SSE; each event is a
//!   base64 postcard-encoded [`crate::protocol::ServerToWorker`].
//! - `POST /v1/cp/worker/{session_id}/result` — body:
//!   [`crate::protocol::AssignmentResult`].
//! - `POST /v1/cp/worker/{session_id}/event` — body:
//!   [`crate::protocol::AssignmentEvent`].
//! - `POST /v1/cp/worker/{session_id}/heartbeat` — body:
//!   [`crate::protocol::WorkerHeartbeat`].
//! - `POST /v1/cp/worker/{session_id}/release` — graceful disconnect.
//!
//! ### Orchestrator tier
//! - `POST /v1/cp/submit` — body:
//!   [`crate::protocol::SubmitRequest`].
//! - `POST /v1/cp/cancel` — body:
//!   [`crate::protocol::CancelRequest`].
//! - `GET  /v1/cp/describe/{run_id}` — JSON response with
//!   [`crate::protocol::RunStateSnapshotWire`].
//! - `GET  /v1/cp/events/{run_id}` — SSE stream of events for a run
//!   (placeholder — empty until Phase 10 event bus).
//! - `GET  /v1/cp/events` — SSE stream of all runs (placeholder).
//!
//! ## Authentication
//!
//! Every route validates the `Authorization: Bearer <token>` header
//! against `BLAZEN_PEER_TOKEN`. When the env var is unset, auth is
//! disabled.

use std::convert::Infallible;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::routing::{get, post};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use dashmap::DashMap;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use blazen_core::distributed::{RunStatus, WorkerCapability};

use crate::auth::validate_bearer;
use crate::protocol::{
    self, AssignmentEvent, AssignmentResult, AssignmentStatus, CancelRequest, ENVELOPE_VERSION,
    ServerToWorker, SubmitRequest, Welcome, WorkerHeartbeat, WorkerHello,
    validate_envelope_version,
};
use crate::server::SharedState;
use crate::server::queue::SubmitOutcome;

/// JSON envelope carrying base64-encoded postcard bytes.
///
/// Used by every HTTP route in this module so the wire format mirrors
/// the gRPC tier's postcard payloads while staying ergonomic to
/// construct from JavaScript (`fetch` + `JSON.stringify`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostcardEnvelope {
    /// Base64 (STANDARD alphabet, with padding) of the postcard-encoded
    /// payload.
    pub postcard_b64: String,
}

impl PostcardEnvelope {
    /// Encode `value` with postcard and wrap it in a base64 envelope.
    ///
    /// # Errors
    ///
    /// Returns [`postcard::Error`] if `value` cannot be serialized.
    pub fn encode<T: Serialize>(value: &T) -> Result<Self, postcard::Error> {
        let bytes = postcard::to_allocvec(value)?;
        Ok(Self {
            postcard_b64: BASE64.encode(&bytes),
        })
    }

    /// Decode the wrapped postcard payload back into a `T`.
    ///
    /// # Errors
    ///
    /// Returns [`HttpError::BadRequest`] if either base64 decoding or
    /// postcard decoding fails.
    pub fn decode<T: serde::de::DeserializeOwned>(&self) -> Result<T, HttpError> {
        let bytes = BASE64
            .decode(self.postcard_b64.as_bytes())
            .map_err(|e| HttpError::BadRequest(format!("base64 decode: {e}")))?;
        postcard::from_bytes(&bytes)
            .map_err(|e| HttpError::BadRequest(format!("postcard decode: {e}")))
    }
}

/// All errors surfaceable by an HTTP route in this module. Each variant
/// maps 1:1 onto an HTTP status code via [`IntoResponse`].
#[derive(Debug)]
pub enum HttpError {
    /// Malformed body, bad base64, bad postcard, or bad JSON.
    BadRequest(String),
    /// Bearer token missing, wrong scheme, or wrong value.
    Unauthenticated(String),
    /// Session id / run id not known to the control plane.
    NotFound(String),
    /// Pre-condition for the request was not met (envelope-version
    /// mismatch, session stream already consumed, …).
    FailedPrecondition(String),
    /// Server-side bug — encode failed, run state vanished mid-call, …
    Internal(String),
}

impl IntoResponse for HttpError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match self {
            HttpError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            HttpError::Unauthenticated(m) => (StatusCode::UNAUTHORIZED, m),
            HttpError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            HttpError::FailedPrecondition(m) => (StatusCode::PRECONDITION_FAILED, m),
            HttpError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}

/// Build the axum router. Caller wires it into a server via
/// `axum::serve(listener, router).await`.
pub fn router(shared: Arc<SharedState>) -> Router {
    Router::new()
        // Worker tier
        .route("/v1/cp/worker/register", post(worker_register))
        .route("/v1/cp/worker/{session_id}/stream", get(worker_stream))
        .route("/v1/cp/worker/{session_id}/result", post(worker_result))
        .route("/v1/cp/worker/{session_id}/event", post(worker_event))
        .route(
            "/v1/cp/worker/{session_id}/heartbeat",
            post(worker_heartbeat),
        )
        .route("/v1/cp/worker/{session_id}/release", post(worker_release))
        // Orchestrator tier
        .route("/v1/cp/submit", post(orchestrator_submit))
        .route("/v1/cp/cancel", post(orchestrator_cancel))
        .route("/v1/cp/describe/{run_id}", get(orchestrator_describe))
        .route("/v1/cp/events/{run_id}", get(orchestrator_events_one))
        .route("/v1/cp/events", get(orchestrator_events_all))
        .with_state(shared)
}

fn check_auth(headers: &HeaderMap) -> Result<(), HttpError> {
    let raw = headers.get("authorization").and_then(|v| v.to_str().ok());
    validate_bearer(raw).map_err(HttpError::Unauthenticated)
}

/// State held per-session for the SSE worker tier. The outbound channel
/// receives [`ServerToWorker`] frames just like a bidi session would; the
/// SSE handler reads from it.
struct HttpWorkerState {
    /// Wrapped in a `Mutex<Option<...>>` because the SSE handler takes
    /// the receiver exactly once on first GET — any subsequent stream
    /// open against the same session must return
    /// [`HttpError::FailedPrecondition`] rather than steal frames from
    /// the existing reader.
    outbound_rx: tokio::sync::Mutex<Option<mpsc::Receiver<ServerToWorker>>>,
}

// HACK: rather than threading another field through `SharedState` (and
// keeping its constructor stable across phases), we use a
// process-global `DashMap` initialised via `OnceLock`. This keeps the
// http tier self-contained until Phase 1h figures out a cleaner
// state-sharing pattern. Entries here have the same lifetime as a bidi
// session in [`crate::server::registry::WorkerRegistry`]: created by
// `worker_register`, consumed by `worker_stream`, removed by
// `worker_release` (or implicitly when the bidi-tier evicts the node id).
static HTTP_SESSIONS: OnceLock<DashMap<Uuid, Arc<HttpWorkerState>>> = OnceLock::new();

fn http_sessions() -> &'static DashMap<Uuid, Arc<HttpWorkerState>> {
    HTTP_SESSIONS.get_or_init(DashMap::new)
}

// ===== Worker tier =====

async fn worker_register(
    State(shared): State<Arc<SharedState>>,
    headers: HeaderMap,
    Json(envelope): Json<PostcardEnvelope>,
) -> Result<Json<serde_json::Value>, HttpError> {
    check_auth(&headers)?;
    let hello: WorkerHello = envelope.decode()?;
    validate_envelope_version(hello.envelope_version)
        .map_err(|e| HttpError::FailedPrecondition(e.to_string()))?;

    let (tx, rx) = mpsc::channel::<ServerToWorker>(64);
    let capabilities = hello.capabilities.iter().map(Into::into).collect();
    let session_id = shared.registry.register(
        hello.node_id.clone(),
        capabilities,
        hello.tags.clone(),
        (&hello.admission).into(),
        tx.clone(),
    );

    // Stash the receiver so the SSE handler can pick it up on the
    // subsequent GET /stream.
    http_sessions().insert(
        session_id,
        Arc::new(HttpWorkerState {
            outbound_rx: tokio::sync::Mutex::new(Some(rx)),
        }),
    );

    let welcome = Welcome {
        envelope_version: ENVELOPE_VERSION,
        session_id,
        negotiated_envelope_version: hello
            .supported_envelope_versions
            .iter()
            .copied()
            .filter(|v| *v <= ENVELOPE_VERSION)
            .max()
            .unwrap_or(ENVELOPE_VERSION),
    };
    // Push the Welcome into the outbound channel so the SSE consumer
    // sees it as the first frame, matching the gRPC bidi handshake.
    // Failure to send here just means the worker disconnected
    // immediately — not an error worth surfacing.
    let _ = tx.send(ServerToWorker::Welcome(welcome.clone())).await;

    // Drain any pending assignments for the capabilities this worker
    // advertised — same trick the bidi session uses on accept.
    let cap_vec: Vec<WorkerCapability> = hello.capabilities.iter().map(Into::into).collect();
    shared
        .queue
        .try_drain_for(&cap_vec, &shared.registry, &shared.admission)
        .await;

    let welcome_envelope = PostcardEnvelope::encode(&welcome)
        .map_err(|e| HttpError::Internal(format!("encode Welcome: {e}")))?;
    Ok(Json(serde_json::json!({
        "session_id": session_id.to_string(),
        "welcome": welcome_envelope,
    })))
}

async fn worker_stream(
    State(_shared): State<Arc<SharedState>>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, HttpError> {
    check_auth(&headers)?;

    let entry = http_sessions()
        .get(&session_id)
        .ok_or_else(|| HttpError::NotFound(format!("unknown session {session_id}")))?
        .clone();
    let rx =
        entry.outbound_rx.lock().await.take().ok_or_else(|| {
            HttpError::FailedPrecondition("session stream already consumed".into())
        })?;

    let stream = ReceiverStream::new(rx).map(|frame| {
        let envelope = PostcardEnvelope::encode(&frame)
            .ok()
            .and_then(|e| serde_json::to_string(&e).ok())
            .unwrap_or_else(|| "{}".to_string());
        Ok(SseEvent::default().data(envelope))
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

async fn worker_result(
    State(shared): State<Arc<SharedState>>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
    Json(envelope): Json<PostcardEnvelope>,
) -> Result<StatusCode, HttpError> {
    check_auth(&headers)?;
    // `session_id` is accepted for symmetry with the bidi-tier and to
    // give clients a stable URL shape, but the queue's mark_* fns key
    // by run id alone.
    let _ = session_id;
    let result: AssignmentResult = envelope.decode()?;
    validate_envelope_version(result.envelope_version)
        .map_err(|e| HttpError::FailedPrecondition(e.to_string()))?;
    match result.status {
        AssignmentStatus::Completed => {
            let output: serde_json::Value =
                serde_json::from_slice(&result.output_json).unwrap_or(serde_json::Value::Null);
            shared.queue.mark_completed(result.run_id, output);
        }
        AssignmentStatus::Failed => shared.queue.mark_failed(
            result.run_id,
            result.error.unwrap_or_else(|| "unknown".into()),
        ),
        AssignmentStatus::Cancelled => shared.queue.mark_cancelled(result.run_id),
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn worker_event(
    State(shared): State<Arc<SharedState>>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
    Json(envelope): Json<PostcardEnvelope>,
) -> Result<StatusCode, HttpError> {
    check_auth(&headers)?;
    let _ = session_id;
    let event: AssignmentEvent = envelope.decode()?;
    validate_envelope_version(event.envelope_version)
        .map_err(|e| HttpError::FailedPrecondition(e.to_string()))?;
    shared.queue.record_event(event.run_id);
    Ok(StatusCode::NO_CONTENT)
}

async fn worker_heartbeat(
    State(shared): State<Arc<SharedState>>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
    Json(envelope): Json<PostcardEnvelope>,
) -> Result<StatusCode, HttpError> {
    check_auth(&headers)?;
    let hb: WorkerHeartbeat = envelope.decode()?;
    validate_envelope_version(hb.envelope_version)
        .map_err(|e| HttpError::FailedPrecondition(e.to_string()))?;
    let snap = hb.admission_snapshot.as_ref().map(Into::into);
    shared
        .registry
        .record_heartbeat(session_id, hb.in_flight, snap);
    Ok(StatusCode::NO_CONTENT)
}

async fn worker_release(
    State(shared): State<Arc<SharedState>>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<StatusCode, HttpError> {
    check_auth(&headers)?;
    shared.registry.unregister(session_id);
    shared.queue.surrender_session(session_id).await;
    http_sessions().remove(&session_id);
    Ok(StatusCode::NO_CONTENT)
}

// ===== Orchestrator tier =====

async fn orchestrator_submit(
    State(shared): State<Arc<SharedState>>,
    headers: HeaderMap,
    Json(envelope): Json<PostcardEnvelope>,
) -> Result<Json<PostcardEnvelope>, HttpError> {
    check_auth(&headers)?;
    let req: SubmitRequest = envelope.decode()?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| HttpError::FailedPrecondition(e.to_string()))?;
    let core = req
        .to_core()
        .map_err(|e| HttpError::BadRequest(format!("decode input_json: {e}")))?;

    let run_id = Uuid::new_v4();
    let assignment = protocol::Assignment {
        envelope_version: ENVELOPE_VERSION,
        run_id,
        parent_run_id: None,
        workflow_name: core.workflow_name.clone(),
        workflow_version: core.workflow_version,
        input_json: req.input_json.clone(),
        deadline_ms: core.deadline_ms,
        attempt: 0,
        resource_hint: core.resource_hint.as_ref().map(Into::into),
    };
    let cap = WorkerCapability {
        kind: format!("workflow:{}", core.workflow_name),
        version: core.workflow_version.unwrap_or(0),
    };
    let outcome = shared
        .queue
        .submit(
            run_id,
            assignment,
            cap,
            core.required_tags.clone(),
            core.wait_for_worker,
            &shared.registry,
            &shared.admission,
        )
        .await;

    match outcome {
        SubmitOutcome::Pushed { .. }
        | SubmitOutcome::Offered { .. }
        | SubmitOutcome::Queued { .. } => {
            let snap = shared
                .queue
                .describe(run_id)
                .ok_or_else(|| HttpError::Internal("run state missing".into()))?;
            let wire = run_state_to_wire(&snap)
                .map_err(|e| HttpError::Internal(format!("encode run state: {e}")))?;
            PostcardEnvelope::encode(&wire)
                .map(Json)
                .map_err(|e| HttpError::Internal(format!("encode envelope: {e}")))
        }
        SubmitOutcome::Rejected { reason } => Err(match reason {
            crate::error::ControlPlaneError::NoMatchingWorker { .. } => {
                HttpError::FailedPrecondition(reason.to_string())
            }
            crate::error::ControlPlaneError::MissingVramHint => {
                HttpError::BadRequest(reason.to_string())
            }
            other => HttpError::Internal(other.to_string()),
        }),
    }
}

async fn orchestrator_cancel(
    State(shared): State<Arc<SharedState>>,
    headers: HeaderMap,
    Json(envelope): Json<PostcardEnvelope>,
) -> Result<Json<PostcardEnvelope>, HttpError> {
    check_auth(&headers)?;
    let req: CancelRequest = envelope.decode()?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| HttpError::FailedPrecondition(e.to_string()))?;
    let Some(snap) = shared.queue.describe(req.run_id) else {
        return Err(HttpError::NotFound(format!("unknown run {}", req.run_id)));
    };
    if let Some(ref node_id) = snap.assigned_to
        && let Some(session_id) = shared.registry.session_for_node(node_id)
        && let Some(handle) = shared.registry.get(session_id)
    {
        let cancel = protocol::CancelInstruction {
            envelope_version: ENVELOPE_VERSION,
            run_id: req.run_id,
        };
        let _ = handle.outbound.send(ServerToWorker::Cancel(cancel)).await;
    }
    shared.queue.mark_cancelled(req.run_id);

    let snap = shared
        .queue
        .describe(req.run_id)
        .ok_or_else(|| HttpError::Internal("run state missing".into()))?;
    let wire = run_state_to_wire(&snap)
        .map_err(|e| HttpError::Internal(format!("encode run state: {e}")))?;
    PostcardEnvelope::encode(&wire)
        .map(Json)
        .map_err(|e| HttpError::Internal(format!("encode envelope: {e}")))
}

async fn orchestrator_describe(
    State(shared): State<Arc<SharedState>>,
    Path(run_id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Json<PostcardEnvelope>, HttpError> {
    check_auth(&headers)?;
    let Some(snap) = shared.queue.describe(run_id) else {
        return Err(HttpError::NotFound(format!("unknown run {run_id}")));
    };
    let wire = run_state_to_wire(&snap)
        .map_err(|e| HttpError::Internal(format!("encode run state: {e}")))?;
    PostcardEnvelope::encode(&wire)
        .map(Json)
        .map_err(|e| HttpError::Internal(format!("encode envelope: {e}")))
}

async fn orchestrator_events_one(
    State(_shared): State<Arc<SharedState>>,
    Path(_run_id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, HttpError> {
    check_auth(&headers)?;
    // Placeholder — empty stream until the Phase 10 event bus lands.
    // Clients can poll `GET /describe/{run_id}` in the meantime.
    let s = futures_util::stream::empty();
    Ok(Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

async fn orchestrator_events_all(
    State(_shared): State<Arc<SharedState>>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, HttpError> {
    check_auth(&headers)?;
    let s = futures_util::stream::empty();
    Ok(Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

fn run_state_to_wire(
    snap: &blazen_core::distributed::RunStateSnapshot,
) -> Result<protocol::RunStateSnapshotWire, serde_json::Error> {
    let output_json = match &snap.output {
        Some(v) => serde_json::to_vec(v)?,
        None => Vec::new(),
    };
    Ok(protocol::RunStateSnapshotWire {
        envelope_version: ENVELOPE_VERSION,
        run_id: snap.run_id,
        status: match snap.status {
            RunStatus::Pending => protocol::RunStatusWire::Pending,
            RunStatus::Running => protocol::RunStatusWire::Running,
            RunStatus::Completed => protocol::RunStatusWire::Completed,
            RunStatus::Failed => protocol::RunStatusWire::Failed,
            RunStatus::Cancelled => protocol::RunStatusWire::Cancelled,
        },
        started_at_ms: snap.started_at_ms,
        completed_at_ms: snap.completed_at_ms,
        assigned_to: snap.assigned_to.clone(),
        last_event_at_ms: snap.last_event_at_ms,
        output_json,
        error: snap.error.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip() {
        let hello = WorkerHello {
            envelope_version: ENVELOPE_VERSION,
            node_id: "test".into(),
            capabilities: vec![],
            tags: std::collections::BTreeMap::new(),
            admission: protocol::AdmissionModeWire::Reactive,
            supported_envelope_versions: vec![1],
        };
        let env = PostcardEnvelope::encode(&hello).expect("encode WorkerHello");
        let decoded: WorkerHello = env.decode().expect("decode WorkerHello");
        assert_eq!(decoded.node_id, "test");
        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
        assert!(matches!(
            decoded.admission,
            protocol::AdmissionModeWire::Reactive
        ));
    }
}
