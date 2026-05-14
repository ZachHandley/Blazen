//! Unary RPC handlers for the control plane.
//!
//! All handlers follow the same shape:
//!
//! 1. Decode the postcard payload into the typed request.
//! 2. Validate the envelope version.
//! 3. Call into shared state (registry / queue / admission).
//! 4. Encode the response into a `PostcardResponse`.

use std::sync::Arc;

use tonic::{Response, Status};

use crate::pb::{PostcardRequest, PostcardResponse};
use crate::protocol::{
    self, CancelRequest, DescribeRequest, DrainWorkerRequest, ENVELOPE_VERSION, ListWorkersRequest,
    ListWorkersResponse, RunStateSnapshotWire, RunStatusWire, ServerToWorker, SubmitRequest,
    WorkerInfoWire, validate_envelope_version,
};

use super::SharedState;
use super::queue::SubmitOutcome;
use blazen_core::distributed::{RunStatus, WorkerCapability};

/// Handle a `SubmitWorkflow` unary RPC.
///
/// Generates a fresh `run_id`, builds an [`protocol::Assignment`] from
/// the request, and pushes it through the queue. The response is a
/// [`RunStateSnapshotWire`] that reflects the run's state immediately
/// after the submit (`Pending` if queued, `Running` if pushed).
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported, or
///   no worker matches and the caller did not opt into
///   `wait_for_worker`.
/// - `INVALID_ARGUMENT` if a `VramBudget` worker is targeted without a
///   `resource_hint.vram_mb` estimate.
/// - `INTERNAL` for any other failure (encoding, missing run state).
pub async fn handle_submit_workflow(
    shared: &Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: SubmitRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode SubmitRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let core_req = req
        .to_core()
        .map_err(|e| Status::invalid_argument(format!("decode input_json: {e}")))?;

    let run_id = uuid::Uuid::new_v4();
    let assignment = protocol::Assignment {
        envelope_version: ENVELOPE_VERSION,
        run_id,
        parent_run_id: None,
        workflow_name: core_req.workflow_name.clone(),
        workflow_version: core_req.workflow_version,
        input_json: req.input_json.clone(),
        deadline_ms: core_req.deadline_ms,
        attempt: 0,
        resource_hint: core_req.resource_hint.as_ref().map(Into::into),
    };

    let required_capability = WorkerCapability {
        kind: format!("workflow:{}", core_req.workflow_name),
        version: core_req.workflow_version.unwrap_or(0),
    };

    let outcome = shared
        .queue
        .submit(
            run_id,
            assignment,
            required_capability,
            core_req.required_tags.clone(),
            core_req.wait_for_worker,
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
                .ok_or_else(|| Status::internal("run state missing after submit"))?;
            let wire = run_state_to_wire(&snap)
                .map_err(|e| Status::internal(format!("encode run state: {e}")))?;
            encode_resp(&wire)
        }
        SubmitOutcome::Rejected { reason } => Err(match reason {
            crate::error::ControlPlaneError::NoMatchingWorker { .. } => {
                Status::failed_precondition(reason.to_string())
            }
            crate::error::ControlPlaneError::MissingVramHint => {
                Status::invalid_argument(reason.to_string())
            }
            other => Status::internal(other.to_string()),
        }),
    }
}

/// Handle a `CancelWorkflow` unary RPC.
///
/// Looks up the run, pushes a `Cancel` instruction to the currently
/// assigned worker (if any), and marks the run as `Cancelled` in the
/// queue. Returns the updated snapshot.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
/// - `NOT_FOUND` if no run with this `run_id` is tracked.
/// - `INTERNAL` for encoding failures.
pub async fn handle_cancel_workflow(
    shared: &Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: CancelRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode CancelRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let Some(snap) = shared.queue.describe(req.run_id) else {
        return Err(Status::not_found(format!("unknown run {}", req.run_id)));
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
        .ok_or_else(|| Status::internal("run state missing after cancel"))?;
    let wire =
        run_state_to_wire(&snap).map_err(|e| Status::internal(format!("encode run state: {e}")))?;
    encode_resp(&wire)
}

/// Handle a `DescribeWorkflow` unary RPC.
///
/// Returns the current [`RunStateSnapshotWire`] for the requested run.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
/// - `NOT_FOUND` if no run with this `run_id` is tracked.
/// - `INTERNAL` for encoding failures.
#[allow(clippy::unused_async)] // kept async to match the trait-impl shape and future event-bus integration.
pub async fn handle_describe_workflow(
    shared: &Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: DescribeRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode DescribeRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let Some(snap) = shared.queue.describe(req.run_id) else {
        return Err(Status::not_found(format!("unknown run {}", req.run_id)));
    };
    let wire =
        run_state_to_wire(&snap).map_err(|e| Status::internal(format!("encode run state: {e}")))?;
    encode_resp(&wire)
}

/// Handle a `ListWorkers` unary RPC.
///
/// Snapshots every currently-connected worker via
/// [`super::registry::WorkerRegistry::list`] and returns it as a
/// [`ListWorkersResponse`].
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
/// - `INTERNAL` for encoding failures.
#[allow(clippy::unused_async)] // kept async to match the trait-impl shape and future async work.
pub async fn handle_list_workers(
    shared: &Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: ListWorkersRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode ListWorkersRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let workers_core = shared.registry.list();
    let workers: Vec<WorkerInfoWire> = workers_core.iter().map(Into::into).collect();
    let resp = ListWorkersResponse {
        envelope_version: ENVELOPE_VERSION,
        workers,
    };
    encode_resp(&resp)
}

/// Handle a `DrainWorker` unary RPC.
///
/// Pushes a [`protocol::DrainInstruction`] to the targeted worker's
/// outbound channel. Returns an empty postcard payload on success.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
/// - `NOT_FOUND` if no worker with this `node_id` is currently
///   connected.
/// - `INTERNAL` for encoding failures.
pub async fn handle_drain_worker(
    shared: &Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: DrainWorkerRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode DrainWorkerRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let Some(session_id) = shared.registry.session_for_node(&req.node_id) else {
        return Err(Status::not_found(format!("unknown worker {}", req.node_id)));
    };
    let Some(handle) = shared.registry.get(session_id) else {
        return Err(Status::not_found(format!("unknown worker {}", req.node_id)));
    };
    let drain = protocol::DrainInstruction {
        envelope_version: ENVELOPE_VERSION,
        immediate: req.immediate,
    };
    let _ = handle.outbound.send(ServerToWorker::Drain(drain)).await;

    // Echo back an empty postcard response.
    encode_resp(&())
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes(bytes)
}

fn encode_resp<T: serde::Serialize>(value: &T) -> Result<Response<PostcardResponse>, Status> {
    let bytes = postcard::to_allocvec(value)
        .map_err(|e| Status::internal(format!("encode response: {e}")))?;
    Ok(Response::new(PostcardResponse {
        postcard_payload: bytes,
    }))
}

fn run_state_to_wire(
    snap: &blazen_core::distributed::RunStateSnapshot,
) -> Result<RunStateSnapshotWire, serde_json::Error> {
    let output_json = match &snap.output {
        Some(v) => serde_json::to_vec(v)?,
        None => Vec::new(),
    };
    Ok(RunStateSnapshotWire {
        envelope_version: ENVELOPE_VERSION,
        run_id: snap.run_id,
        status: match snap.status {
            RunStatus::Pending => RunStatusWire::Pending,
            RunStatus::Running => RunStatusWire::Running,
            RunStatus::Completed => RunStatusWire::Completed,
            RunStatus::Failed => RunStatusWire::Failed,
            RunStatus::Cancelled => RunStatusWire::Cancelled,
        },
        started_at_ms: snap.started_at_ms,
        completed_at_ms: snap.completed_at_ms,
        assigned_to: snap.assigned_to.clone(),
        last_event_at_ms: snap.last_event_at_ms,
        output_json,
        error: snap.error.clone(),
    })
}
