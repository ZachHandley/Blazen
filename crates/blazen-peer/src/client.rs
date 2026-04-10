//! Blazen peer client -- invokes remote sub-workflows.
//!
//! Wraps a tonic gRPC channel to communicate with a remote
//! [`super::server::BlazenPeerServer`].

use blazen_core::session_ref::RegistryKey;
use tracing::debug;

use crate::error::PeerError;
use crate::pb;
use crate::protocol::{
    DerefRequest, DerefResponse, ENVELOPE_VERSION, ReleaseRequest, ReleaseResponse,
    SubWorkflowRequest, SubWorkflowResponse,
};

/// Client handle for talking to a remote [`super::server::BlazenPeerServer`].
///
/// Wraps a tonic gRPC channel together with a `node_id` that
/// identifies this end of the connection for tracing purposes.
pub struct BlazenPeerClient {
    client: pb::blazen_peer_client::BlazenPeerClient<tonic::transport::Channel>,
    node_id: String,
}

impl BlazenPeerClient {
    /// Open a connection to the peer at `endpoint`.
    ///
    /// `endpoint` must be a valid gRPC URI, e.g.
    /// `http://node-a.local:7443`.
    ///
    /// `node_id` identifies this client in trace logs. Typically the
    /// hostname or a UUID set at process startup.
    ///
    /// # Errors
    /// Returns [`PeerError::Transport`] if the endpoint URI is invalid
    /// or the TCP connection cannot be established.
    pub async fn connect(
        endpoint: impl Into<String>,
        node_id: impl Into<String>,
    ) -> Result<Self, PeerError> {
        let endpoint_str = endpoint.into();
        let node_id = node_id.into();
        debug!(endpoint = %endpoint_str, node_id = %node_id, "connecting to blazen peer");

        let channel = tonic::transport::Channel::from_shared(endpoint_str)
            .map_err(|e| PeerError::Transport(e.to_string()))?
            .connect()
            .await
            .map_err(|e| PeerError::Transport(e.to_string()))?;

        let client = pb::blazen_peer_client::BlazenPeerClient::new(channel);
        Ok(Self { client, node_id })
    }

    /// Invoke a sub-workflow on the connected peer.
    ///
    /// # Errors
    /// Returns [`PeerError::Encode`] if the request cannot be
    /// postcard-encoded, [`PeerError::Transport`] if the RPC fails,
    /// or [`PeerError::Encode`] if the response cannot be decoded.
    pub async fn invoke_sub_workflow(
        &mut self,
        request: SubWorkflowRequest,
    ) -> Result<SubWorkflowResponse, PeerError> {
        debug!(
            node_id = %self.node_id,
            workflow = %request.workflow_name,
            "invoking remote sub-workflow"
        );

        let postcard_req = pb::PostcardRequest {
            postcard_payload: postcard::to_allocvec(&request)?,
        };

        let response = self
            .client
            .invoke_sub_workflow(tonic::Request::new(postcard_req))
            .await
            .map_err(|s| PeerError::Transport(s.to_string()))?;

        let resp: SubWorkflowResponse =
            postcard::from_bytes(&response.into_inner().postcard_payload)?;

        Ok(resp)
    }

    /// Dereference a remote session ref by its registry key. Returns
    /// the raw serialized bytes of the underlying value.
    ///
    /// # Errors
    /// Returns [`PeerError::Encode`] on postcard errors or
    /// [`PeerError::Transport`] if the RPC fails (including the
    /// remote returning `NOT_FOUND`).
    pub async fn deref_session_ref(&mut self, key: RegistryKey) -> Result<Vec<u8>, PeerError> {
        debug!(node_id = %self.node_id, ?key, "dereferencing remote session ref");

        let req = DerefRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: key.0,
        };
        let postcard_req = pb::PostcardRequest {
            postcard_payload: postcard::to_allocvec(&req)?,
        };

        let response = self
            .client
            .deref_session_ref(tonic::Request::new(postcard_req))
            .await
            .map_err(|s| PeerError::Transport(s.to_string()))?;

        let resp: DerefResponse = postcard::from_bytes(&response.into_inner().postcard_payload)?;

        Ok(resp.payload)
    }

    /// Release (drop) a remote session ref. Returns `true` if the
    /// ref was found and released, `false` if it was already gone.
    ///
    /// # Errors
    /// Returns [`PeerError::Encode`] on postcard errors or
    /// [`PeerError::Transport`] if the RPC fails.
    pub async fn release_session_ref(&mut self, key: RegistryKey) -> Result<bool, PeerError> {
        debug!(node_id = %self.node_id, ?key, "releasing remote session ref");

        let req = ReleaseRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: key.0,
        };
        let postcard_req = pb::PostcardRequest {
            postcard_payload: postcard::to_allocvec(&req)?,
        };

        let response = self
            .client
            .release_session_ref(tonic::Request::new(postcard_req))
            .await
            .map_err(|s| PeerError::Transport(s.to_string()))?;

        let resp: ReleaseResponse = postcard::from_bytes(&response.into_inner().postcard_payload)?;

        Ok(resp.released)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postcard_roundtrip_invoke_request() {
        let input = serde_json::json!({"url": "https://example.com"});
        let req = SubWorkflowRequest::new(
            "summarize",
            vec!["fetch".to_string(), "summarize".to_string()],
            &input,
            Some(60),
        )
        .unwrap();

        let encoded = postcard::to_allocvec(&req).unwrap();
        let decoded: SubWorkflowRequest = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.workflow_name, req.workflow_name);
        assert_eq!(decoded.step_ids, req.step_ids);
        assert_eq!(decoded.timeout_secs, req.timeout_secs);
        assert_eq!(decoded.input_value().unwrap(), input);
    }

    #[test]
    fn postcard_roundtrip_deref_request() {
        let key = RegistryKey::new();
        let req = DerefRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: key.0,
        };

        let encoded = postcard::to_allocvec(&req).unwrap();
        let decoded: DerefRequest = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.ref_uuid, key.0);
        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
    }

    #[test]
    fn postcard_roundtrip_release_request() {
        let key = RegistryKey::new();
        let req = ReleaseRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: key.0,
        };

        let encoded = postcard::to_allocvec(&req).unwrap();
        let decoded: ReleaseRequest = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.ref_uuid, key.0);
        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
    }

    #[test]
    fn postcard_roundtrip_deref_response() {
        let resp = DerefResponse {
            envelope_version: ENVELOPE_VERSION,
            payload: vec![1, 2, 3, 4, 5],
        };

        let encoded = postcard::to_allocvec(&resp).unwrap();
        let decoded: DerefResponse = postcard::from_bytes(&encoded).unwrap();

        assert_eq!(decoded.payload, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn postcard_roundtrip_release_response() {
        let resp = ReleaseResponse {
            envelope_version: ENVELOPE_VERSION,
            released: true,
        };

        let encoded = postcard::to_allocvec(&resp).unwrap();
        let decoded: ReleaseResponse = postcard::from_bytes(&encoded).unwrap();

        assert!(decoded.released);
    }
}
