//! Server-streaming event subscribers.
//!
//! Two RPCs:
//!
//! - `SubscribeRunEvents` — events for a single `run_id`.
//! - `SubscribeAll` — events for every run, optionally tag-filtered.
//!
//! Current implementation is **minimal-viable**: we return an empty
//! stream that closes immediately. Real event fan-out from the
//! `WorkerToServer::Event` frames into per-subscriber channels lands
//! in a later Phase 10 (telemetry) wave when we wire up the
//! event-broadcast bus.

use std::sync::Arc;

use tonic::{Response, Status};

use crate::pb::PostcardRequest;
use crate::protocol::{SubscribeAllRequest, SubscribeRunRequest, validate_envelope_version};

use super::SharedState;
use super::service::PostcardResponseStream;

/// Handle a `SubscribeRunEvents` server-streaming RPC.
///
/// Currently returns an empty stream that closes immediately — the
/// event-broadcast bus is wired up in a later phase.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
#[allow(clippy::unused_async)] // becomes properly async when the event-broadcast bus lands.
pub async fn handle_subscribe_run_events(
    _shared: Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponseStream>, Status> {
    let req: SubscribeRunRequest = postcard::from_bytes(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode SubscribeRunRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    // Empty stream — placeholder until the event-broadcast bus lands.
    let s = tokio_stream::empty();
    let boxed: PostcardResponseStream = Box::pin(s);
    Ok(Response::new(boxed))
}

/// Handle a `SubscribeAll` server-streaming RPC.
///
/// Currently returns an empty stream that closes immediately — the
/// event-broadcast bus is wired up in a later phase.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
#[allow(clippy::unused_async)] // becomes properly async when the event-broadcast bus lands.
pub async fn handle_subscribe_all(
    _shared: Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponseStream>, Status> {
    let req: SubscribeAllRequest = postcard::from_bytes(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode SubscribeAllRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let s = tokio_stream::empty();
    let boxed: PostcardResponseStream = Box::pin(s);
    Ok(Response::new(boxed))
}
