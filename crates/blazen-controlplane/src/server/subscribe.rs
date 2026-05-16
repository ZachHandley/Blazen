//! Server-streaming event subscribers.
//!
//! Two RPCs:
//!
//! - `SubscribeRunEvents` — events for a single `run_id`.
//! - `SubscribeAll` — events for every run, optionally tag-filtered.
//!
//! Both handlers attach to the per-server [`tokio::sync::broadcast`] bus
//! held on [`SharedState::events`]. Producers (worker `Event` frames and
//! queue status transitions) publish [`blazen_core::distributed::RunEvent`]s
//! onto that bus; each subscriber is a [`tokio_stream::wrappers::BroadcastStream`]
//! that filters down to the events it cares about, re-encodes them via
//! [`crate::protocol::RunEventWire`] + postcard, and yields them to tonic.
//! Malformed and lagged broadcast items are dropped silently.

use std::sync::Arc;

use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tonic::{Response, Status};

use crate::pb::{PostcardRequest, PostcardResponse};
use crate::protocol::{
    RunEventWire, SubscribeAllRequest, SubscribeRunRequest, validate_envelope_version,
};

use super::SharedState;
use super::service::PostcardResponseStream;

/// Handle a `SubscribeRunEvents` server-streaming RPC.
///
/// Subscribes to the per-server event bus and yields every
/// [`blazen_core::distributed::RunEvent`] whose `run_id` matches the
/// request. Events are re-encoded as [`RunEventWire`] (postcard) for the
/// wire. Malformed and lagged broadcast items are dropped silently.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
// tonic handler shape requires `async fn` even though this function only
// constructs a stream and never awaits; clippy's `unused_async` lint is
// a false positive here.
#[allow(clippy::unused_async)]
pub async fn handle_subscribe_run_events(
    shared: Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponseStream>, Status> {
    let req: SubscribeRunRequest = postcard::from_bytes(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode SubscribeRunRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let target_run_id = req.run_id;
    let rx = shared.events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |res| {
        let event = res.ok()?;
        if event.run_id != target_run_id {
            return None;
        }
        let wire = RunEventWire::from_core(&event).ok()?;
        let bytes = postcard::to_allocvec(&wire).ok()?;
        Some(Ok(PostcardResponse {
            postcard_payload: bytes,
        }))
    });

    let boxed: PostcardResponseStream = Box::pin(stream);
    Ok(Response::new(boxed))
}

/// Handle a `SubscribeAll` server-streaming RPC.
///
/// Subscribes to the per-server event bus and yields every
/// [`blazen_core::distributed::RunEvent`] whose assigned worker's tags
/// satisfy the AND-conjunction tag predicate in `required_tags`. An
/// empty predicate matches everything. Events for unassigned runs (or
/// runs whose assigned node has since been forgotten by the registry)
/// are dropped when a non-empty predicate is supplied, since we cannot
/// verify the predicate without the worker handle.
///
/// Events are re-encoded as [`RunEventWire`] (postcard) for the wire.
/// Malformed and lagged broadcast items are dropped silently.
///
/// # Errors
///
/// - `INVALID_ARGUMENT` if the request fails to decode.
/// - `FAILED_PRECONDITION` if the envelope version is unsupported.
// tonic handler shape requires `async fn` even though this function only
// constructs a stream and never awaits; clippy's `unused_async` lint is
// a false positive here.
#[allow(clippy::unused_async)]
pub async fn handle_subscribe_all(
    shared: Arc<SharedState>,
    request: PostcardRequest,
) -> Result<Response<PostcardResponseStream>, Status> {
    let req: SubscribeAllRequest = postcard::from_bytes(&request.postcard_payload)
        .map_err(|e| Status::invalid_argument(format!("decode SubscribeAllRequest: {e}")))?;
    validate_envelope_version(req.envelope_version)
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

    let required_tags = req.required_tags.clone();
    let shared_for_filter = shared.clone();
    let rx = shared.events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |res| {
        let event = res.ok()?;
        if !required_tags.is_empty()
            && !run_tags_match(&shared_for_filter, event.run_id, &required_tags)
        {
            return None;
        }
        let wire = RunEventWire::from_core(&event).ok()?;
        let bytes = postcard::to_allocvec(&wire).ok()?;
        Some(Ok(PostcardResponse {
            postcard_payload: bytes,
        }))
    });

    let boxed: PostcardResponseStream = Box::pin(stream);
    Ok(Response::new(boxed))
}

/// AND-conjunction tag-predicate check against the worker currently
/// assigned to `run_id`. Returns `false` if the run is unknown, has no
/// assigned node, the node has no live session, or the session handle
/// no longer matches every requirement.
///
/// Tag format mirrors the admission module exactly: each entry in
/// `required_tags` is either `key=value` (exact match) or `key=*`
/// (presence-only wildcard). Malformed entries (missing `=`) fail
/// closed. Keep this in sync with `server::admission::tags_match` —
/// that function is private to its module, so the predicate is
/// replicated literally here.
fn run_tags_match(shared: &SharedState, run_id: uuid::Uuid, required_tags: &[String]) -> bool {
    let Some(snap) = shared.queue.describe(run_id) else {
        return false;
    };
    let Some(node_id) = snap.assigned_to else {
        return false;
    };
    let Some(session_id) = shared.registry.session_for_node(&node_id) else {
        return false;
    };
    let Some(handle) = shared.registry.get(session_id) else {
        return false;
    };
    required_tags
        .iter()
        .all(|entry| match entry.split_once('=') {
            Some((k, "*")) => handle.tags.contains_key(k),
            Some((k, v)) => handle.tags.get(k).map(String::as_str) == Some(v),
            None => false,
        })
}
