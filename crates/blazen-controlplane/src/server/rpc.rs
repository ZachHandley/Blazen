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
    ListWorkersResponse, RespondToInputRequest, RunStateSnapshotWire, RunStatusWire,
    ServerToWorker, SubmitRequest, WorkerInfoWire, validate_envelope_version,
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
    identity: Option<crate::auth::PeerIdentity>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: SubmitRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode SubmitRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    // Resolve the place: the server-derived identity WINS over any
    // client-set `place` on the wire (anti-spoof). Fall back to the
    // request's self-reported place, then to the default place.
    let place = identity.as_ref().map_or_else(
        || {
            req.place
                .clone()
                .unwrap_or_else(|| crate::protocol::DEFAULT_PLACE.to_string())
        },
        |id| id.place.clone(),
    );

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
        priority: blazen_core::distributed::DEFAULT_PRIORITY,
        selector: protocol::NodeSelectorWire::default(),
        tolerations: Vec::new(),
    };

    let required_capability = WorkerCapability {
        kind: format!("workflow:{}", core_req.workflow_name),
        version: core_req.workflow_version.unwrap_or(0),
    };

    let outcome = shared
        .queue
        .submit(
            run_id,
            place.clone(),
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
            let mut wire = run_state_to_wire(&snap)
                .map_err(|e| Status::internal(format!("encode run state: {e}")))?;
            wire.place = place;
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
    shared.queue.mark_cancelled(req.run_id).await;

    let snap = shared
        .queue
        .describe(req.run_id)
        .ok_or_else(|| Status::internal("run state missing after cancel"))?;
    let wire =
        run_state_to_wire(&snap).map_err(|e| Status::internal(format!("encode run state: {e}")))?;
    encode_resp(&wire)
}

/// Handle a `RespondToInput` unary RPC.
///
/// Looks up the run, resolves the worker currently assigned to it, and
/// pushes a [`ServerToWorker::InputResponse`] frame down that worker's
/// outbound channel (which feeds both the gRPC bidi stream and the HTTP
/// SSE stream). Acks with an empty payload.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported, or the
///   run is not currently assigned to a live worker.
/// - `NOT_FOUND` if no run with this `run_id` is tracked.
pub async fn handle_respond_to_input(
    shared: &Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: RespondToInputRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode RespondToInputRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let Some(snap) = shared.queue.describe(req.run_id) else {
        return Err(Status::not_found(format!("unknown run {}", req.run_id)));
    };

    if let Some(ref node_id) = snap.assigned_to
        && let Some(session_id) = shared.registry.session_for_node(node_id)
        && let Some(handle) = shared.registry.get(session_id)
    {
        let response = protocol::InputResponse {
            envelope_version: ENVELOPE_VERSION,
            run_id: req.run_id,
            request_id: req.request_id,
            response_json: req.response_json,
        };
        handle
            .outbound
            .send(ServerToWorker::InputResponse(response))
            .await
            .map_err(|_| Status::failed_precondition("run not assigned to a live worker"))?;
        encode_resp(&())
    } else {
        Err(Status::failed_precondition(
            "run not assigned to a live worker",
        ))
    }
}

/// Handle a `DescribeWorkflow` unary RPC.
///
/// Returns the current [`RunStateSnapshotWire`] for the requested run.
///
/// Place isolation: a run is only describable by a caller in the SAME place.
/// A run that belongs to a different place is reported as `NOT_FOUND` (NOT
/// `PERMISSION_DENIED`) so the run's existence is not leaked across tenants.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
/// - `NOT_FOUND` if no run with this `run_id` is tracked, or the run belongs
///   to a different place than the caller.
/// - `INTERNAL` for encoding failures.
#[allow(clippy::unused_async)] // kept async to match the trait-impl shape and future event-bus integration.
pub async fn handle_describe_workflow(
    shared: &Arc<SharedState>,
    identity: Option<crate::auth::PeerIdentity>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: DescribeRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode DescribeRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    // The caller's server-derived place. Unauthenticated callers fall back to
    // the default place — they can only see default-place runs.
    let caller_place = identity.as_ref().map_or_else(
        || crate::protocol::DEFAULT_PLACE.to_string(),
        |id| id.place.clone(),
    );

    let Some(snap) = shared.queue.describe(req.run_id) else {
        return Err(Status::not_found(format!("unknown run {}", req.run_id)));
    };

    // Place check: a run in another place is indistinguishable from a
    // non-existent run (no existence leakage). A run with no recorded place
    // is treated as the default place.
    let run_place = shared
        .queue
        .place_for_run(req.run_id)
        .unwrap_or_else(|| crate::protocol::DEFAULT_PLACE.to_string());
    if run_place != caller_place {
        return Err(Status::not_found(format!("unknown run {}", req.run_id)));
    }

    let mut wire =
        run_state_to_wire(&snap).map_err(|e| Status::internal(format!("encode run state: {e}")))?;
    wire.place = run_place;
    encode_resp(&wire)
}

/// Handle a `ListWorkers` unary RPC.
///
/// Snapshots the currently-connected workers serving the CALLER's place via
/// [`super::registry::WorkerRegistry::list_by_place_with_place`] and returns
/// them as a [`ListWorkersResponse`]. Workers in other places are never
/// disclosed — tenancy isolation holds at the list surface too.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
/// - `INTERNAL` for encoding failures.
#[allow(clippy::unused_async)] // kept async to match the trait-impl shape and future async work.
pub async fn handle_list_workers(
    shared: &Arc<SharedState>,
    identity: Option<crate::auth::PeerIdentity>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponse>, Status> {
    let req: ListWorkersRequest = decode(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode ListWorkersRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    // The caller's server-derived place. Unauthenticated callers see only the
    // default place's workers.
    let caller_place = identity.as_ref().map_or_else(
        || crate::protocol::DEFAULT_PLACE.to_string(),
        |id| id.place.clone(),
    );

    let workers: Vec<WorkerInfoWire> = shared
        .registry
        .list_by_place_with_place(&caller_place)
        .iter()
        .map(|(info, place)| {
            let mut wire: WorkerInfoWire = info.into();
            wire.place.clone_from(place);
            wire
        })
        .collect();
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
        // Default place; callers that know the run's place overwrite it.
        place: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use blazen_core::distributed::{AdmissionMode, WorkerCapability};
    use uuid::Uuid;

    use super::*;
    use crate::auth::{PeerIdentity, PeerKind};
    use crate::protocol::{
        Assignment, DescribeRequest, ListWorkersRequest, ListWorkersResponse, NodeSelectorWire,
    };

    /// Build a fresh in-process `SharedState` (registry + queue + admission)
    /// via the server constructor.
    fn fresh_shared() -> Arc<SharedState> {
        crate::server::ControlPlaneServer::new("cp-test")
            .shared
            .clone()
    }

    fn identity_for(place: &str) -> PeerIdentity {
        PeerIdentity {
            place: place.to_owned(),
            kind: PeerKind::Orchestrator,
        }
    }

    fn cap(kind: &str, version: u32) -> WorkerCapability {
        WorkerCapability {
            kind: kind.to_owned(),
            version,
        }
    }

    /// Register a reactive worker advertising `workflow:x` in `place`. Returns
    /// the outbound receiver, which the caller must keep alive for the test's
    /// duration so the registry handle's channel isn't observed as closed.
    #[must_use]
    fn register_worker(
        shared: &Arc<SharedState>,
        node_id: &str,
        place: &str,
    ) -> tokio::sync::mpsc::Receiver<ServerToWorker> {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let mut caps = HashSet::new();
        caps.insert(cap("workflow:x", 0));
        let _sid = shared.registry.register(
            node_id.to_owned(),
            place.to_owned(),
            caps,
            BTreeMap::new(),
            AdmissionMode::Reactive,
            tx,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );
        rx
    }

    /// Submit a queued run scoped to `place`, returning its `run_id`. No worker
    /// match is required — `wait_for_worker = true` parks it Pending.
    async fn submit_run(shared: &Arc<SharedState>, place: &str) -> Uuid {
        let run_id = Uuid::new_v4();
        let assignment = Assignment {
            envelope_version: ENVELOPE_VERSION,
            run_id,
            parent_run_id: None,
            workflow_name: "x".to_owned(),
            workflow_version: None,
            input_json: Vec::new(),
            deadline_ms: None,
            attempt: 0,
            resource_hint: None,
            priority: blazen_core::distributed::DEFAULT_PRIORITY,
            selector: NodeSelectorWire::default(),
            tolerations: Vec::new(),
        };
        let outcome = shared
            .queue
            .submit(
                run_id,
                place.to_owned(),
                assignment,
                cap("workflow:x", 0),
                Vec::new(),
                true, // wait_for_worker: park it Pending regardless of matches
                &shared.registry,
                &shared.admission,
            )
            .await;
        assert!(
            matches!(
                outcome,
                SubmitOutcome::Queued { .. } | SubmitOutcome::Pushed { .. }
            ),
            "submit should queue or push, got {outcome:?}"
        );
        run_id
    }

    fn describe_req(run_id: Uuid) -> PostcardRequest {
        PostcardRequest {
            postcard_payload: postcard::to_allocvec(&DescribeRequest {
                envelope_version: ENVELOPE_VERSION,
                run_id,
            })
            .unwrap(),
        }
    }

    fn list_req() -> PostcardRequest {
        PostcardRequest {
            postcard_payload: postcard::to_allocvec(&ListWorkersRequest {
                envelope_version: ENVELOPE_VERSION,
            })
            .unwrap(),
        }
    }

    fn decode_list(resp: &Response<PostcardResponse>) -> ListWorkersResponse {
        postcard::from_bytes(&resp.get_ref().postcard_payload).unwrap()
    }

    fn decode_snapshot(resp: &Response<PostcardResponse>) -> RunStateSnapshotWire {
        postcard::from_bytes(&resp.get_ref().postcard_payload).unwrap()
    }

    #[tokio::test]
    async fn list_workers_is_place_scoped() {
        let shared = fresh_shared();
        let _rx_a = register_worker(&shared, "node-a", "place-a");
        let _rx_b = register_worker(&shared, "node-b", "place-b");

        // Place A sees only its own worker.
        let resp = handle_list_workers(&shared, Some(identity_for("place-a")), list_req())
            .await
            .expect("list ok");
        let list = decode_list(&resp);
        assert_eq!(list.workers.len(), 1, "place A must see exactly one worker");
        assert_eq!(list.workers[0].node_id, "node-a");
        assert_eq!(list.workers[0].place, "place-a");

        // Place B sees only its own worker.
        let resp_b = handle_list_workers(&shared, Some(identity_for("place-b")), list_req())
            .await
            .unwrap();
        let list_b = decode_list(&resp_b);
        assert_eq!(list_b.workers.len(), 1);
        assert_eq!(list_b.workers[0].node_id, "node-b");

        // A third place sees nothing.
        let resp_c = handle_list_workers(&shared, Some(identity_for("place-c")), list_req())
            .await
            .unwrap();
        assert!(decode_list(&resp_c).workers.is_empty());
    }

    #[tokio::test]
    async fn describe_same_place_succeeds() {
        let shared = fresh_shared();
        let run_id = submit_run(&shared, "place-a").await;

        let resp =
            handle_describe_workflow(&shared, Some(identity_for("place-a")), describe_req(run_id))
                .await
                .expect("describe ok");
        let snap = decode_snapshot(&resp);
        assert_eq!(snap.run_id, run_id);
        assert_eq!(snap.place, "place-a");
    }

    #[tokio::test]
    async fn describe_cross_place_is_not_found() {
        let shared = fresh_shared();
        let run_id = submit_run(&shared, "place-b").await;

        // A place-A caller describing a place-B run must get NOT_FOUND (no
        // existence leakage), NOT PermissionDenied.
        let err =
            handle_describe_workflow(&shared, Some(identity_for("place-a")), describe_req(run_id))
                .await
                .expect_err("cross-place describe must fail");
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn describe_unknown_run_is_not_found() {
        let shared = fresh_shared();
        let err = handle_describe_workflow(
            &shared,
            Some(identity_for("place-a")),
            describe_req(Uuid::new_v4()),
        )
        .await
        .expect_err("unknown run must fail");
        assert_eq!(err.code(), tonic::Code::NotFound);
    }
}
