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

use blazen_core::distributed::RunEvent;

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
    identity: Option<crate::auth::PeerIdentity>,
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

    // Resolve the worker's place. The server-derived identity WINS over
    // the worker's self-reported `hello.place` (anti-spoof). Fall back to
    // the hello's place, then to the default place.
    let place = identity.as_ref().map_or_else(
        || {
            hello
                .place
                .clone()
                .unwrap_or_else(|| crate::protocol::DEFAULT_PLACE.to_string())
        },
        |id| id.place.clone(),
    );

    // Build outbound channel and register.
    let (outbound_tx, outbound_rx) = mpsc::channel::<ServerToWorker>(OUTBOUND_BUFFER);
    let capabilities = hello.capabilities.iter().map(Into::into).collect();
    let taints = hello
        .taints
        .iter()
        .cloned()
        .map(blazen_core::distributed::WorkerTaint::from)
        .collect();
    let session_id = shared.registry.register(
        hello.node_id.clone(),
        place.clone(),
        capabilities,
        hello.tags.clone(),
        (&hello.admission).into(),
        outbound_tx.clone(),
        hello.labels.clone(),
        taints,
        hello.descriptors.clone(),
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
        .try_drain_for(&place, &cap_vec, &shared.registry, &shared.admission)
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
                shared.queue.mark_completed(r.run_id, output).await;
            }
            crate::protocol::AssignmentStatus::Failed => {
                shared
                    .queue
                    .mark_failed(r.run_id, r.error.unwrap_or_else(|| "unknown".into()))
                    .await;
            }
            crate::protocol::AssignmentStatus::Cancelled => {
                shared.queue.mark_cancelled(r.run_id).await;
            }
        },
        WorkerToServer::Event(e) => {
            shared.queue.record_event(e.run_id).await;
            let data = serde_json::from_slice(&e.data_json).unwrap_or(serde_json::Value::Null);
            let event = RunEvent {
                run_id: e.run_id,
                event_type: e.event_type,
                data,
                timestamp_ms: e.timestamp_ms,
            };
            // `broadcast::Sender::send` returns Err when there are zero live
            // subscribers — expected and harmless; discard.
            let _ = shared.events.send(event);
        }
        WorkerToServer::OfferDecision(_) => {
            // TODO Phase 1b retry loop: tie into the offer/claim/decline
            // retry path. The queue's `submit()` currently treats Offer as
            // Offered immediately without awaiting a response — when the
            // retry loop lands, this is where the decision gets delivered.
        }
        WorkerToServer::KeyRequest(req) => {
            handle_key_request(shared, session_id, req).await;
        }
    }
}

/// Resolve a worker's [`crate::protocol::KeyRequest`] against the
/// per-place [`crate::server::key_store::KeyStore`] and reply with a
/// [`ServerToWorker::KeyResponse`] on the worker's outbound channel.
///
/// The place is ALWAYS the worker session's server-authenticated place
/// (read from the registry handle) — never a place the worker named — so
/// one tenant can never fetch another's key. An unknown session, an
/// unbrokered provider, or a store error all resolve to `key: None`; the
/// worker then falls through to the next link in its resolver chain. The
/// key value is never logged (the [`crate::protocol::KeyResponse`] frame
/// redacts it in `Debug`).
async fn handle_key_request(
    shared: &Arc<SharedState>,
    session_id: Uuid,
    req: crate::protocol::KeyRequest,
) {
    // Look up the worker's authenticated place + outbound channel by
    // session id. A vanished session (worker raced a disconnect) → drop.
    let Some(handle) = shared.registry.get(session_id) else {
        tracing::debug!(
            session_id = %session_id,
            "KeyRequest for unknown session; dropping"
        );
        return;
    };

    let key = match shared.key_store.get_key(&handle.place, &req.provider).await {
        Ok(found) => found,
        Err(e) => {
            // Never include the key (there is none on this path) or place
            // secrets in the log; surface only the provider + error.
            tracing::warn!(
                session_id = %session_id,
                provider = %req.provider,
                error = %e,
                "key store lookup failed; replying with no key"
            );
            None
        }
    };

    // Split SharedKey into the wire fields. `None` (unbrokered/unknown
    // provider) yields `key: None` with default cache metadata.
    let (key_value, ttl_secs, version) = match key {
        Some(sk) => (Some(sk.value), sk.ttl_secs, sk.version),
        None => (None, None, 0),
    };

    let response = ServerToWorker::KeyResponse(crate::protocol::KeyResponse {
        envelope_version: ENVELOPE_VERSION,
        request_id: req.request_id,
        key: key_value,
        ttl_secs,
        version,
    });

    // Best-effort send — a full/closed outbound channel means the worker
    // is gone or backed up; the worker will re-request on its next
    // pre-warm. Never log the response (it carries the key).
    if handle.outbound.send(response).await.is_err() {
        tracing::debug!(
            session_id = %session_id,
            provider = %req.provider,
            "worker outbound closed before KeyResponse could be sent"
        );
    }
}

fn decode_postcard<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes(bytes)
}

fn encode_postcard<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(value)
}
