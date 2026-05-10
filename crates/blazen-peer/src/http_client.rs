//! HTTP/JSON transport for blazen-peer. Wasi-compatible alternative to
//! the tonic gRPC [`crate::client::BlazenPeerClient`].
//!
//! The native gRPC client speaks a postcard-in-bytes wire format over
//! HTTP/2 + tonic. That stack does not compile for `wasm32-wasi*` (no
//! `mio`, no `aws-lc-rs`), so this module provides a pure-HTTP/JSON
//! variant that runs on top of any [`HttpClient`] implementation —
//! including the lazy proxy used in Cloudflare Workers / Deno.
//!
//! # Wire format
//!
//! Every gRPC method maps to a single HTTP `POST` with a JSON body
//! matching the corresponding [`crate::protocol`] request type. The
//! server returns a JSON body matching the response type, or a non-2xx
//! status with a JSON error body of the shape `{"error": "..."}`.
//!
//! # Routes
//!
//! - `POST /v1/peer/invoke_sub_workflow`  → [`SubWorkflowResponse`]
//! - `POST /v1/peer/deref_session_ref`    → [`DerefResponse`]
//! - `POST /v1/peer/release_session_ref`  → [`ReleaseResponse`]
//!
//! # Server side
//!
//! There is no server-side counterpart in this crate today. Workers and
//! similar wasi hosts cannot host listeners, so the only server-side
//! path remains the native tonic gRPC server. This client is therefore
//! a one-way bridge: the wasi caller speaks HTTP/JSON and an external
//! adapter (or a future HTTP-native peer server) translates to gRPC at
//! the receiving end.

use std::sync::Arc;

use blazen_core::session_ref::RegistryKey;
use blazen_llm::http::{HttpClient, HttpMethod, HttpRequest};
use tracing::debug;

use crate::error::PeerError;
use crate::protocol::{
    DerefRequest, DerefResponse, ENVELOPE_VERSION, ReleaseRequest, ReleaseResponse,
    SubWorkflowRequest, SubWorkflowResponse,
};

/// Path component of [`HttpPeerClient::invoke_sub_workflow`].
const PATH_INVOKE_SUB_WORKFLOW: &str = "/v1/peer/invoke_sub_workflow";
/// Path component of [`HttpPeerClient::deref_session_ref`].
const PATH_DEREF_SESSION_REF: &str = "/v1/peer/deref_session_ref";
/// Path component of [`HttpPeerClient::release_session_ref`].
const PATH_RELEASE_SESSION_REF: &str = "/v1/peer/release_session_ref";

/// HTTP header used to carry the calling node's identity to the peer.
///
/// The native gRPC client carries this implicitly through tonic
/// metadata; over plain HTTP we surface it as a request header so the
/// receiving adapter can log / authenticate the caller.
const NODE_ID_HEADER: &str = "X-Blazen-Peer-Node-Id";

/// HTTP/JSON peer client. Wasi-compatible alternative to
/// [`crate::client::BlazenPeerClient`] with a mirroring method surface.
///
/// Construct with [`HttpPeerClient::new`], passing a base URL (e.g.
/// `https://peer.example.com`), an [`Arc<dyn HttpClient>`], and a
/// `node_id` that identifies this caller in trace logs.
pub struct HttpPeerClient {
    base_url: String,
    http: Arc<dyn HttpClient>,
    node_id: String,
}

impl std::fmt::Debug for HttpPeerClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpPeerClient")
            .field("base_url", &self.base_url)
            .field("node_id", &self.node_id)
            .finish_non_exhaustive()
    }
}

impl HttpPeerClient {
    /// Build a new client.
    ///
    /// `base_url` should not include a trailing slash, but a trailing
    /// slash is tolerated and trimmed before each request.
    ///
    /// `http` is any [`HttpClient`] implementation — the lazy proxy
    /// installed by `setDefaultHttpClient` on a Worker is the typical
    /// caller.
    ///
    /// `node_id` identifies this client in trace logs and is sent to
    /// the peer as the [`X-Blazen-Peer-Node-Id`](self) header.
    pub fn new(
        base_url: impl Into<String>,
        http: Arc<dyn HttpClient>,
        node_id: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            http,
            node_id: node_id.into(),
        }
    }

    /// The configured trace identifier for this client.
    #[must_use]
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// The configured base URL (with any trailing slash preserved).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Invoke a sub-workflow on the connected peer.
    ///
    /// Mirrors [`crate::client::BlazenPeerClient::invoke_sub_workflow`].
    ///
    /// # Errors
    /// Returns [`PeerError::Encode`] if the request cannot be
    /// JSON-encoded or the response cannot be JSON-decoded, and
    /// [`PeerError::Transport`] if the underlying HTTP call fails or
    /// the peer returns a non-2xx status.
    pub async fn invoke_sub_workflow(
        &mut self,
        request: SubWorkflowRequest,
    ) -> Result<SubWorkflowResponse, PeerError> {
        debug!(
            node_id = %self.node_id,
            workflow = %request.workflow_name,
            "invoking remote sub-workflow over http/json"
        );
        self.post_json(PATH_INVOKE_SUB_WORKFLOW, &request).await
    }

    /// Dereference a remote session ref by its registry key.
    ///
    /// Mirrors [`crate::client::BlazenPeerClient::deref_session_ref`].
    ///
    /// # Errors
    /// Returns [`PeerError::Encode`] on JSON encode/decode failure and
    /// [`PeerError::Transport`] on HTTP failure or non-2xx status
    /// (including the remote-side `NOT_FOUND` analogue).
    pub async fn deref_session_ref(&mut self, key: RegistryKey) -> Result<Vec<u8>, PeerError> {
        debug!(node_id = %self.node_id, ?key, "dereferencing remote session ref over http/json");
        let req = DerefRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: key.0,
        };
        let resp: DerefResponse = self.post_json(PATH_DEREF_SESSION_REF, &req).await?;
        Ok(resp.payload)
    }

    /// Release (drop) a remote session ref. Returns `true` if the ref
    /// was found and released, `false` if it was already gone.
    ///
    /// Mirrors [`crate::client::BlazenPeerClient::release_session_ref`].
    ///
    /// # Errors
    /// Returns [`PeerError::Encode`] on JSON encode/decode failure and
    /// [`PeerError::Transport`] on HTTP failure or non-2xx status.
    pub async fn release_session_ref(&mut self, key: RegistryKey) -> Result<bool, PeerError> {
        debug!(node_id = %self.node_id, ?key, "releasing remote session ref over http/json");
        let req = ReleaseRequest {
            envelope_version: ENVELOPE_VERSION,
            ref_uuid: key.0,
        };
        let resp: ReleaseResponse = self.post_json(PATH_RELEASE_SESSION_REF, &req).await?;
        Ok(resp.released)
    }

    /// JSON-encode `body`, POST it to `{base_url}{path}`, and decode
    /// the JSON response into `Resp`. Maps every failure mode onto
    /// the appropriate [`PeerError`] variant.
    async fn post_json<Req, Resp>(&self, path: &str, body: &Req) -> Result<Resp, PeerError>
    where
        Req: serde::Serialize,
        Resp: serde::de::DeserializeOwned,
    {
        let url = format!("{}{path}", self.base_url.trim_end_matches('/'));
        let body_bytes = serde_json::to_vec(body)
            .map_err(|e| PeerError::Transport(format!("json serialize: {e}")))?;

        let request = HttpRequest {
            method: HttpMethod::Post,
            url,
            headers: vec![
                ("Content-Type".to_owned(), "application/json".to_owned()),
                (NODE_ID_HEADER.to_owned(), self.node_id.clone()),
            ],
            body: Some(body_bytes),
            query_params: Vec::new(),
        };

        let response = self
            .http
            .send(request)
            .await
            .map_err(|e| PeerError::Transport(format!("http transport: {e}")))?;

        if !response.is_success() {
            // Try to surface a structured `{"error": "..."}` body when the
            // peer provides one; fall back to lossy UTF-8 of whatever bytes
            // we got. The status code is always included so the caller can
            // distinguish 4xx from 5xx without parsing the message.
            let detail = match serde_json::from_slice::<ErrorBody>(&response.body) {
                Ok(eb) => eb.error,
                Err(_) => response.text(),
            };
            return Err(PeerError::Transport(format!(
                "HTTP {} from {}: {}",
                response.status, path, detail
            )));
        }

        serde_json::from_slice(&response.body)
            .map_err(|e| PeerError::Transport(format!("json deserialize: {e}")))
    }
}

/// Minimal shape we attempt to parse out of a non-2xx response body
/// before falling back to the raw text.
#[derive(serde::Deserialize)]
struct ErrorBody {
    error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_invoke_request() {
        let input = serde_json::json!({"url": "https://example.com"});
        let req = SubWorkflowRequest::new(
            "summarize",
            vec!["fetch".to_string(), "summarize".to_string()],
            &input,
            Some(60),
        )
        .unwrap();

        let encoded = serde_json::to_vec(&req).unwrap();
        let decoded: SubWorkflowRequest = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded.workflow_name, req.workflow_name);
        assert_eq!(decoded.step_ids, req.step_ids);
        assert_eq!(decoded.timeout_secs, req.timeout_secs);
        assert_eq!(decoded.input_value().unwrap(), input);
    }

    #[test]
    fn json_roundtrip_deref_response() {
        let resp = DerefResponse {
            envelope_version: ENVELOPE_VERSION,
            payload: vec![1, 2, 3, 4, 5],
        };

        let encoded = serde_json::to_vec(&resp).unwrap();
        let decoded: DerefResponse = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded.payload, vec![1, 2, 3, 4, 5]);
        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
    }

    #[test]
    fn json_roundtrip_release_response() {
        let resp = ReleaseResponse {
            envelope_version: ENVELOPE_VERSION,
            released: true,
        };

        let encoded = serde_json::to_vec(&resp).unwrap();
        let decoded: ReleaseResponse = serde_json::from_slice(&encoded).unwrap();

        assert!(decoded.released);
        assert_eq!(decoded.envelope_version, ENVELOPE_VERSION);
    }

    #[test]
    fn debug_redacts_http_client() {
        // Smoke-test that the manual Debug impl doesn't try to print the
        // Arc<dyn HttpClient> directly (which has no useful Debug repr).
        // This needs a concrete HttpClient — use the trait object via a
        // tiny stub.
        #[derive(Debug)]
        struct Stub;

        #[async_trait::async_trait]
        impl HttpClient for Stub {
            async fn send(
                &self,
                _request: HttpRequest,
            ) -> Result<blazen_llm::http::HttpResponse, blazen_llm::error::BlazenError>
            {
                unreachable!("not called in this test")
            }

            async fn send_streaming(
                &self,
                _request: HttpRequest,
            ) -> Result<
                (u16, Vec<(String, String)>, blazen_llm::http::ByteStream),
                blazen_llm::error::BlazenError,
            > {
                unreachable!("not called in this test")
            }
        }

        let client = HttpPeerClient::new("https://peer.example.com/", Arc::new(Stub), "node-1");
        let rendered = format!("{client:?}");
        assert!(rendered.contains("HttpPeerClient"));
        assert!(rendered.contains("node-1"));
        assert!(rendered.contains("peer.example.com"));
    }
}
