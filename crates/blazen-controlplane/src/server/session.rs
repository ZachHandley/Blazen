//! Bidi `WorkerSession` stream handler.
//!
//! Implements the `WorkerSession` gRPC method on
//! [`super::service::ControlPlaneService`]. On Hello, registers the
//! worker in the registry and sends Welcome. Multiplexes incoming
//! `WorkerToServer` frames and outgoing `ServerToWorker` frames over
//! the bidi gRPC stream.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status, Streaming};
use uuid::Uuid;

use crate::pb::{PostcardRequest, PostcardResponse};
use crate::protocol::{
    ENVELOPE_VERSION, ServerToWorker, Welcome, WorkerToServer, validate_envelope_version,
};

use super::SharedState;
use super::service::PostcardResponseStream;

/// Default per-worker outbound channel buffer. Each slot holds one
/// `ServerToWorker` frame waiting to be encoded and pushed onto the
/// gRPC stream.
const OUTBOUND_BUFFER: usize = 64;

/// Handle a worker bidi session.
///
/// Consumes the incoming `PostcardRequest` stream and returns a
/// matching outgoing `PostcardResponse` stream as the RPC response. The
/// caller (the service trait impl) hands the response back to tonic
/// which threads it onto the wire.
///
/// # Errors
///
/// Returns [`Status::failed_precondition`] when the worker closes the
/// stream before sending a `Hello`, sends a non-`Hello` first frame, or
/// announces an envelope version this build cannot decode.
/// Returns [`Status::invalid_argument`] when the first frame is not a
/// decodable `WorkerToServer`. Returns [`Status::internal`] when the
/// underlying transport reports a recv error before the handshake.
pub async fn handle_worker_session(
    shared: Arc<SharedState>,
    mut incoming: Streaming<PostcardRequest>,
) -> Result<Response<PostcardResponseStream>, Status> {
    // Peek the first frame — must be a Hello.
    let first = incoming
        .next()
        .await
        .ok_or_else(|| Status::failed_precondition("worker session closed before Hello"))?
        .map_err(|e| Status::internal(format!("recv failed: {e}")))?;

    let hello_frame: WorkerToServer = decode_postcard(&first.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode WorkerToServer: {e}")))?;

    let WorkerToServer::Hello(hello) = hello_frame else {
        return Err(Status::failed_precondition(
            "first worker frame must be Hello",
        ));
    };

    validate_envelope_version(hello.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    // Build outbound channel and register.
    let (outbound_tx, outbound_rx) = mpsc::channel::<ServerToWorker>(OUTBOUND_BUFFER);
    let capabilities = hello.capabilities.iter().map(Into::into).collect();
    let session_id = shared.registry.register(
        hello.node_id.clone(),
        capabilities,
        hello.tags.clone(),
        (&hello.admission).into(),
        outbound_tx.clone(),
    );

    let negotiated = hello
        .supported_envelope_versions
        .iter()
        .copied()
        .filter(|v| *v <= ENVELOPE_VERSION)
        .max()
        .unwrap_or(ENVELOPE_VERSION);

    // Send Welcome.
    let welcome = Welcome {
        envelope_version: ENVELOPE_VERSION,
        session_id,
        negotiated_envelope_version: negotiated,
    };
    let _ = outbound_tx.send(ServerToWorker::Welcome(welcome)).await;

    // Drain any pending assignments for capabilities this worker advertises.
    let cap_vec: Vec<_> = hello.capabilities.iter().map(Into::into).collect();
    shared
        .queue
        .try_drain_for(&cap_vec, &shared.registry, &shared.admission)
        .await;

    // Spawn inbound pump: read WorkerToServer frames, update registry / queue.
    let shared_for_inbound = shared.clone();
    tokio::spawn(async move {
        while let Some(msg) = incoming.next().await {
            match msg {
                Ok(req) => {
                    let Ok(frame) = decode_postcard::<WorkerToServer>(&req.postcard_payload) else {
                        tracing::warn!(
                            session_id = %session_id,
                            "discarding malformed WorkerToServer frame"
                        );
                        continue;
                    };
                    handle_worker_frame(&shared_for_inbound, session_id, frame).await;
                }
                Err(e) => {
                    tracing::info!(session_id = %session_id, error = %e, "worker stream ended");
                    break;
                }
            }
        }
        // Stream closed — unregister + surrender in-flight assignments.
        shared_for_inbound.registry.unregister(session_id);
        shared_for_inbound.queue.surrender_session(session_id).await;
    });

    // Build outgoing stream from outbound_rx, encoding each frame as postcard.
    let out_stream = ReceiverStream::new(outbound_rx).map(|frame| {
        encode_postcard(&frame)
            .map(|bytes| PostcardResponse {
                postcard_payload: bytes,
            })
            .map_err(|e| Status::internal(format!("encode ServerToWorker: {e}")))
    });

    let boxed: PostcardResponseStream = Box::pin(out_stream);
    Ok(Response::new(boxed))
}

/// Dispatch a single inbound `WorkerToServer` frame into the
/// registry / queue. Frames whose handler is not yet wired up are
/// logged at debug level and dropped.
#[allow(clippy::unused_async)] // becomes properly async when the offer/claim retry loop and event-bus land.
async fn handle_worker_frame(shared: &Arc<SharedState>, session_id: Uuid, frame: WorkerToServer) {
    match frame {
        WorkerToServer::Hello(_) => {
            tracing::warn!(session_id = %session_id, "ignoring duplicate Hello");
        }
        WorkerToServer::Heartbeat(hb) => {
            let snap = hb.admission_snapshot.as_ref().map(Into::into);
            shared
                .registry
                .record_heartbeat(session_id, hb.in_flight, snap);
        }
        WorkerToServer::Result(r) => match r.status {
            crate::protocol::AssignmentStatus::Completed => {
                let output =
                    serde_json::from_slice(&r.output_json).unwrap_or(serde_json::Value::Null);
                shared.queue.mark_completed(r.run_id, output);
            }
            crate::protocol::AssignmentStatus::Failed => {
                shared
                    .queue
                    .mark_failed(r.run_id, r.error.unwrap_or_else(|| "unknown".into()));
            }
            crate::protocol::AssignmentStatus::Cancelled => {
                shared.queue.mark_cancelled(r.run_id);
            }
        },
        WorkerToServer::Event(e) => {
            shared.queue.record_event(e.run_id);
            // TODO Phase 1e: forward to SubscribeRunEvents / SubscribeAll
            // subscribers. For now the queue just tracks the timestamp so
            // `describe()` can surface `last_event_at_ms`.
        }
        WorkerToServer::OfferDecision(_) => {
            // TODO Phase 1b retry loop: tie into the offer/claim/decline
            // retry path. The queue's `submit()` currently treats Offer as
            // Offered immediately without awaiting a response — when the
            // retry loop lands, this is where the decision gets delivered.
        }
    }
}

fn decode_postcard<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes(bytes)
}

fn encode_postcard<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(value)
}
