//! Blazen peer server -- accepts remote sub-workflow invocations.
//!
//! Implements the [`crate::pb::blazen_peer_server::BlazenPeer`] tonic
//! service trait on top of [`BlazenPeerServer`] and wires up the
//! dispatch path into [`blazen_core`].

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use blazen_core::Workflow;
use blazen_core::session_ref::{RegistryKey, SessionRefRegistry};
use blazen_core::step_registry::lookup_step_builder;
use blazen_events::StopEvent;
use tracing::{debug, error, info, warn};

use crate::error::PeerError;
use crate::pb;
use crate::protocol::{
    DerefRequest, DerefResponse, ENVELOPE_VERSION, ReleaseRequest, ReleaseResponse,
    RemoteRefDescriptor, SubWorkflowRequest, SubWorkflowResponse,
};

/// A node-local Blazen peer server.
///
/// Each instance owns:
///
/// - a stable `node_id` used in
///   [`crate::protocol::RemoteRefDescriptor::origin_node_id`] when
///   handing out session-ref proxies, and
/// - a [`SessionRefRegistry`] that holds the live values backing every
///   ref this node has handed out.
pub struct BlazenPeerServer {
    node_id: String,
    session_refs: Arc<SessionRefRegistry>,
}

impl BlazenPeerServer {
    /// Create a new peer server with a fresh, empty
    /// [`SessionRefRegistry`]. Use [`Self::with_session_refs`] to
    /// share an existing registry with the rest of the host process.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            session_refs: Arc::new(SessionRefRegistry::default()),
        }
    }

    /// Replace the per-server session-ref registry with one supplied
    /// by the embedder. Lets a Blazen process share its registry
    /// between in-process workflows and remote peers.
    #[must_use]
    pub fn with_session_refs(mut self, refs: Arc<SessionRefRegistry>) -> Self {
        self.session_refs = refs;
        self
    }

    /// Bind the gRPC server to `addr` and serve forever.
    ///
    /// # Errors
    /// Returns [`PeerError::Transport`] if the tonic server fails to
    /// start or encounters a fatal I/O error while serving.
    pub async fn serve(self, addr: SocketAddr) -> Result<(), PeerError> {
        info!(node_id = %self.node_id, %addr, "starting blazen peer gRPC server");
        let svc = pb::blazen_peer_server::BlazenPeerServer::new(self);
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve(addr)
            .await
            .map_err(|e| PeerError::Transport(e.to_string()))
    }
}

/// Decode a postcard payload from a `PostcardRequest`, producing a
/// tonic `Status` on failure.
fn decode_postcard<T: serde::de::DeserializeOwned>(
    request: &tonic::Request<pb::PostcardRequest>,
) -> Result<T, tonic::Status> {
    postcard::from_bytes(&request.get_ref().postcard_payload)
        .map_err(|e| tonic::Status::invalid_argument(format!("postcard decode failed: {e}")))
}

/// Encode a value into a `PostcardResponse`, producing a tonic
/// `Status` on failure.
fn encode_postcard<T: serde::Serialize>(value: &T) -> Result<pb::PostcardResponse, tonic::Status> {
    let payload = postcard::to_allocvec(value)
        .map_err(|e| tonic::Status::internal(format!("postcard encode failed: {e}")))?;
    Ok(pb::PostcardResponse {
        postcard_payload: payload,
    })
}

/// Validate the envelope version on a decoded payload. Returns
/// `Ok(())` if the version is supported, or an appropriate tonic
/// `Status` if it is too new.
fn validate_envelope_version(got: u32) -> Result<(), tonic::Status> {
    if got > ENVELOPE_VERSION {
        return Err(tonic::Status::failed_precondition(format!(
            "envelope version {got} is newer than supported {ENVELOPE_VERSION}"
        )));
    }
    Ok(())
}

#[tonic::async_trait]
impl pb::blazen_peer_server::BlazenPeer for BlazenPeerServer {
    /// Placeholder type for the streaming RPC (unimplemented).
    type StreamSubWorkflowEventsStream =
        tokio_stream::Empty<Result<pb::PostcardResponse, tonic::Status>>;

    async fn invoke_sub_workflow(
        &self,
        request: tonic::Request<pb::PostcardRequest>,
    ) -> Result<tonic::Response<pb::PostcardResponse>, tonic::Status> {
        let req: SubWorkflowRequest = decode_postcard(&request)?;
        validate_envelope_version(req.envelope_version)?;

        debug!(
            workflow = %req.workflow_name,
            steps = ?req.step_ids,
            "received invoke_sub_workflow"
        );

        // Validate that every requested step is registered on this node.
        for step_id in &req.step_ids {
            if lookup_step_builder(step_id).is_none() {
                warn!(step_id, "unknown step in invoke_sub_workflow");
                return Err(tonic::Status::not_found(format!(
                    "unknown step id `{step_id}`"
                )));
            }
        }

        // Build the workflow from the registered steps.
        let step_id_refs: Vec<&str> = req.step_ids.iter().map(String::as_str).collect();
        let workflow = Workflow::new_from_registered_steps(&req.workflow_name, step_id_refs)
            .map_err(|e| tonic::Status::internal(format!("workflow build failed: {e}")))?;

        // Decode the input JSON.
        let input = req
            .input_value()
            .map_err(|e| tonic::Status::invalid_argument(format!("invalid input JSON: {e}")))?;

        // Run the workflow with the server's session-ref registry.
        let registry = Arc::clone(&self.session_refs);
        let handler = workflow
            .run_with_registry(input, Arc::clone(&registry))
            .await
            .map_err(|e| tonic::Status::internal(format!("workflow run failed: {e}")))?;

        // Optionally apply a timeout around waiting for the result.
        let result_event = if let Some(secs) = req.timeout_secs {
            match tokio::time::timeout(Duration::from_secs(secs), handler.result()).await {
                Ok(Ok(workflow_result)) => workflow_result.event,
                Ok(Err(e)) => {
                    error!(?e, "workflow execution error");
                    let resp = SubWorkflowResponse::err(format!("workflow error: {e}"));
                    return Ok(tonic::Response::new(encode_postcard(&resp)?));
                }
                Err(_elapsed) => {
                    let resp =
                        SubWorkflowResponse::err(format!("workflow timed out after {secs}s"));
                    return Ok(tonic::Response::new(encode_postcard(&resp)?));
                }
            }
        } else {
            match handler.result().await {
                Ok(workflow_result) => workflow_result.event,
                Err(e) => {
                    error!(?e, "workflow execution error");
                    let resp = SubWorkflowResponse::err(format!("workflow error: {e}"));
                    return Ok(tonic::Response::new(encode_postcard(&resp)?));
                }
            }
        };

        // Extract the StopEvent result.
        let result_json = result_event
            .downcast_ref::<StopEvent>()
            .map(|stop| &stop.result);

        // Collect serializable session refs to send back as remote
        // descriptors so the caller can dereference them later.
        let serializable = registry.serializable_entries().await;
        let mut remote_refs = HashMap::with_capacity(serializable.len());
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        #[allow(clippy::cast_possible_truncation)]
        let now_ms = now_ms as u64;
        for (key, ser_ref) in &serializable {
            remote_refs.insert(
                key.0,
                RemoteRefDescriptor {
                    origin_node_id: self.node_id.clone(),
                    type_tag: ser_ref.blazen_type_tag().to_string(),
                    created_at_epoch_ms: now_ms,
                },
            );
        }

        let response = SubWorkflowResponse::ok(&HashMap::new(), result_json, remote_refs)
            .map_err(|e| tonic::Status::internal(format!("response encoding failed: {e}")))?;

        debug!(
            workflow = %req.workflow_name,
            refs = response.remote_refs.len(),
            "invoke_sub_workflow completed"
        );

        Ok(tonic::Response::new(encode_postcard(&response)?))
    }

    async fn stream_sub_workflow_events(
        &self,
        _request: tonic::Request<pb::PostcardRequest>,
    ) -> Result<tonic::Response<Self::StreamSubWorkflowEventsStream>, tonic::Status> {
        Err(tonic::Status::unimplemented(
            "StreamSubWorkflowEvents is not yet implemented (Phase 12.8)",
        ))
    }

    async fn deref_session_ref(
        &self,
        request: tonic::Request<pb::PostcardRequest>,
    ) -> Result<tonic::Response<pb::PostcardResponse>, tonic::Status> {
        let req: DerefRequest = decode_postcard(&request)?;
        validate_envelope_version(req.envelope_version)?;

        let key = RegistryKey(req.ref_uuid);
        debug!(?key, "received deref_session_ref");

        let ser_ref = self
            .session_refs
            .get_serializable(key)
            .await
            .ok_or_else(|| tonic::Status::not_found(format!("session ref {key} not found")))?;

        let payload = ser_ref.blazen_serialize().map_err(|e| {
            tonic::Status::internal(format!("serialization of session ref {key} failed: {e}"))
        })?;

        let response = DerefResponse {
            envelope_version: ENVELOPE_VERSION,
            payload,
        };

        Ok(tonic::Response::new(encode_postcard(&response)?))
    }

    async fn release_session_ref(
        &self,
        request: tonic::Request<pb::PostcardRequest>,
    ) -> Result<tonic::Response<pb::PostcardResponse>, tonic::Status> {
        let req: ReleaseRequest = decode_postcard(&request)?;
        validate_envelope_version(req.envelope_version)?;

        let key = RegistryKey(req.ref_uuid);
        debug!(?key, "received release_session_ref");

        let was_some = self.session_refs.remove(key).await.is_some();

        let response = ReleaseResponse {
            envelope_version: ENVELOPE_VERSION,
            released: was_some,
        };

        Ok(tonic::Response::new(encode_postcard(&response)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postcard_encode_decode_sub_workflow_request() {
        let input = serde_json::json!({"key": "value"});
        let req = SubWorkflowRequest::new(
            "test-wf",
            vec!["step-a".to_string(), "step-b".to_string()],
            &input,
            Some(30),
        )
        .unwrap();

        let encoded = postcard::to_allocvec(&req).unwrap();
        let decoded: SubWorkflowRequest = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.workflow_name, "test-wf");
        assert_eq!(decoded.step_ids, vec!["step-a", "step-b"]);
        assert_eq!(decoded.timeout_secs, Some(30));
        assert_eq!(decoded.input_value().unwrap(), input);
    }

    #[test]
    fn postcard_encode_decode_deref_request() {
        let uuid = uuid::Uuid::new_v4();
        let req = DerefRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: uuid,
        };

        let encoded = postcard::to_allocvec(&req).unwrap();
        let decoded: DerefRequest = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
        assert_eq!(decoded.ref_uuid, uuid);
    }

    #[test]
    fn postcard_encode_decode_release_request() {
        let uuid = uuid::Uuid::new_v4();
        let req = ReleaseRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: uuid,
        };

        let encoded = postcard::to_allocvec(&req).unwrap();
        let decoded: ReleaseRequest = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
        assert_eq!(decoded.ref_uuid, uuid);
    }

    #[test]
    fn validate_envelope_version_accepts_current() {
        assert!(validate_envelope_version(ENVELOPE_VERSION).is_ok());
    }

    #[test]
    fn validate_envelope_version_accepts_older() {
        if ENVELOPE_VERSION > 0 {
            assert!(validate_envelope_version(ENVELOPE_VERSION - 1).is_ok());
        }
    }

    #[test]
    fn validate_envelope_version_rejects_newer() {
        let result = validate_envelope_version(ENVELOPE_VERSION + 1);
        assert!(result.is_err());
    }
}
