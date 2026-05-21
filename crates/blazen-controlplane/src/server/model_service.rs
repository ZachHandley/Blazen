//! Tonic `BlazenModelServer` service implementation.
//!
//! Thin shim — each generated trait method delegates to the matching
//! [`super::model_manager`] handler. Built from a [`ModelServerState`]
//! so the same `ManagerHandle` is shared across every concurrent RPC.

use tonic::{Request, Response, Status, Streaming};

use crate::model_pb::blazen_model_server_server::BlazenModelServer;
use crate::model_pb::{PostcardRequest, PostcardResponse};

use super::model_manager::{
    self, ModelServerState, PostcardOutStream, handle_complete, handle_embed, handle_fetch_blob,
    handle_generate_image, handle_generate_music, handle_is_loaded, handle_list_adapters,
    handle_load, handle_load_adapter, handle_load_from_hf, handle_status, handle_stream_complete,
    handle_text_to_speech, handle_transcribe, handle_unload, handle_unload_adapter,
    handle_upload_blob,
};

/// Concrete tonic service for `BlazenModelServer`.
#[derive(Clone)]
pub struct ModelService {
    state: ModelServerState,
}

impl ModelService {
    /// Build a service from the supplied state.
    #[must_use]
    pub fn new(state: ModelServerState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl BlazenModelServer for ModelService {
    type StreamCompleteStream = PostcardOutStream;
    type FetchBlobStream = PostcardOutStream;

    async fn load(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_load(&self.state, request.into_inner()).await
    }

    async fn unload(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_unload(&self.state, request.into_inner()).await
    }

    async fn is_loaded(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_is_loaded(&self.state, request.into_inner()).await
    }

    async fn status(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_status(&self.state, request.into_inner()).await
    }

    async fn load_from_hf(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_load_from_hf(&self.state, request.into_inner()).await
    }

    async fn load_adapter(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_load_adapter(&self.state, request.into_inner()).await
    }

    async fn unload_adapter(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_unload_adapter(&self.state, request.into_inner()).await
    }

    async fn list_adapters(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_list_adapters(&self.state, request.into_inner()).await
    }

    async fn complete(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_complete(&self.state, request.into_inner()).await
    }

    async fn stream_complete(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<Self::StreamCompleteStream>, Status> {
        handle_stream_complete(self.state.clone(), request.into_inner()).await
    }

    async fn embed(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_embed(&self.state, request.into_inner()).await
    }

    async fn generate_image(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_generate_image(&self.state, request.into_inner()).await
    }

    async fn text_to_speech(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_text_to_speech(&self.state, request.into_inner()).await
    }

    async fn generate_music(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_generate_music(&self.state, request.into_inner()).await
    }

    async fn transcribe(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_transcribe(&self.state, request.into_inner()).await
    }

    async fn upload_blob(
        &self,
        request: Request<Streaming<PostcardRequest>>,
    ) -> Result<Response<PostcardResponse>, Status> {
        handle_upload_blob(self.state.clone(), request.into_inner()).await
    }

    async fn fetch_blob(
        &self,
        request: Request<PostcardRequest>,
    ) -> Result<Response<Self::FetchBlobStream>, Status> {
        handle_fetch_blob(self.state.clone(), request.into_inner()).await
    }
}

// Re-export so the binding layer / lib.rs can `pub use` from a single
// location.
pub use model_manager::ManagerHandle;

// ---------------------------------------------------------------------------
// Integration test — in-process server + client over an ephemeral port
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::model_client::ModelClient;
    use crate::model_pb::blazen_model_server_server::BlazenModelServerServer;
    use crate::model_protocol::{
        IsLoadedRequest, LoadRequest, MODEL_ENVELOPE_VERSION, StatusRequest, UnloadRequest,
    };
    use crate::server::model_manager::test_support::MockManagerHandle;
    use std::time::Duration;
    use tokio::net::TcpListener;

    async fn spawn_test_server() -> (String, std::sync::Arc<MockManagerHandle>) {
        let mock = MockManagerHandle::new();
        let state = ModelServerState::new(mock.clone());
        // Bind-then-drop to discover a free port; tonic re-binds when we
        // call `serve(addr)` below. Mirrors the pattern used by the
        // workflow control-plane integration tests.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let svc = BlazenModelServerServer::new(ModelService::new(state));
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(svc)
                .serve(addr)
                .await;
        });
        // Give the server a moment to start accepting.
        tokio::time::sleep(Duration::from_millis(150)).await;
        (format!("http://{addr}"), mock)
    }

    #[tokio::test]
    async fn load_status_unload_round_trip() {
        let (endpoint, mock) = spawn_test_server().await;
        let client = ModelClient::connect(endpoint).await.unwrap();

        let load = LoadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
        };
        client.load(load).await.unwrap();

        let is_loaded = IsLoadedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
        };
        let resp = client.is_loaded(is_loaded).await.unwrap();
        assert!(resp.loaded);

        let status = StatusRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
        };
        client.status(status).await.unwrap();

        let unload = UnloadRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id: "qwen".to_owned(),
        };
        client.unload(unload).await.unwrap();

        let calls = mock.calls.lock().unwrap().clone();
        assert_eq!(calls.as_slice(), &["load", "is_loaded", "status", "unload"]);
    }
}
