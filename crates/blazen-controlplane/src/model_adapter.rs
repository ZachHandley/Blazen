//! Bridge from `blazen-llm`'s local-model fallback seam onto the control
//! plane's in-process [`ManagerHandle`].
//!
//! P5 gave `blazen-llm` a provider-agnostic fallback mechanism in
//! [`build_model`](blazen_llm::build_model): on an [`Auth`](blazen_llm::BlazenError::Auth)
//! miss with a permitting [`FallbackPolicy`](blazen_llm::FallbackPolicy), the
//! factory consults a [`LocalModelProbe`](blazen_llm::LocalModelProbe) and, when
//! the model is servable, a [`LocalModelFactory`](blazen_llm::LocalModelFactory).
//! Standalone `blazen-llm` installs
//! [`NoLocalModels`](blazen_llm::NoLocalModels) and never routes local.
//!
//! This module is the control-plane *implementation* of that seam. A worker
//! that holds a [`ManagerHandle`] (the host's in-process model manager, reached
//! over the [`model_protocol`](crate::model_protocol) wire structs) installs:
//!
//! - [`ManagerProbe`] — answers "is this model loaded / locally servable?" by
//!   calling [`ManagerHandle::is_loaded`].
//! - [`ManagerFactory`] — builds a [`ManagerBridgeModel`], a
//!   [`blazen_llm::Model`] whose [`complete`](blazen_llm::Model::complete)
//!   translates the request to a
//!   [`CompleteRequest`](crate::model_protocol::CompleteRequest) and routes it
//!   through [`ManagerHandle::complete`].
//!
//! Streaming via the manager bridge is intentionally **not** wired:
//! [`ManagerBridgeModel::stream`] returns a real
//! [`unsupported`](blazen_llm::BlazenError::unsupported) error rather than a
//! pretend stub. The fallback path that uses this bridge is non-streaming
//! completion; callers that need streaming should reach the manager's
//! `stream_complete` surface directly.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_llm::{
    BlazenError, LocalModelFactory, LocalModelProbe, Model, ModelRequest, ModelResponse,
    StreamChunk, TokenUsage,
};
use futures_util::Stream;

use crate::model_protocol::{
    ChatMessageWire, CompleteRequest, IsLoadedRequest, MODEL_ENVELOPE_VERSION, RpcError,
};
use crate::server::model_manager::ManagerHandle;

/// Default model id handed to [`ManagerHandle`] verbs when the caller did not
/// pin one. The control plane's model manager resolves this to whatever it has
/// registered under that id.
const DEFAULT_MANAGER_MODEL: &str = "default";

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

/// [`LocalModelProbe`] backed by a [`ManagerHandle`].
///
/// A `(provider, model)` pair is considered locally servable when the manager
/// reports the model id as loaded. When the caller does not pin a model, the
/// probe asks about [`DEFAULT_MANAGER_MODEL`].
#[derive(Clone)]
pub struct ManagerProbe {
    handle: Arc<dyn ManagerHandle>,
}

impl ManagerProbe {
    /// Wrap a manager handle in a probe.
    #[must_use]
    pub fn new(handle: Arc<dyn ManagerHandle>) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl LocalModelProbe for ManagerProbe {
    async fn is_locally_servable(&self, _provider: &str, model: Option<&str>) -> bool {
        let model_id = model.unwrap_or(DEFAULT_MANAGER_MODEL).to_owned();
        let req = IsLoadedRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id,
        };
        // A wire/manager error is treated as "not servable" — the probe is a
        // hint on the hot path, not an authority; the factory only builds local
        // when this returns true, so erring towards `false` keeps the remote
        // key-resolution error as the surfaced failure.
        matches!(self.handle.is_loaded(req).await, Ok(resp) if resp.loaded)
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// [`LocalModelFactory`] backed by a [`ManagerHandle`].
///
/// [`build_local`](LocalModelFactory::build_local) hands back a
/// [`ManagerBridgeModel`] bound to the same handle and the requested model id.
#[derive(Clone)]
pub struct ManagerFactory {
    handle: Arc<dyn ManagerHandle>,
}

impl ManagerFactory {
    /// Wrap a manager handle in a factory.
    #[must_use]
    pub fn new(handle: Arc<dyn ManagerHandle>) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl LocalModelFactory for ManagerFactory {
    async fn build_local(
        &self,
        _provider: &str,
        model: &str,
    ) -> Result<Box<dyn Model>, BlazenError> {
        Ok(Box::new(ManagerBridgeModel {
            handle: Arc::clone(&self.handle),
            model_id: model.to_owned(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Bridge model
// ---------------------------------------------------------------------------

/// A [`blazen_llm::Model`] that satisfies completions by calling a
/// [`ManagerHandle`] over the [`model_protocol`](crate::model_protocol) wire
/// structs.
///
/// Only [`complete`](Model::complete) is supported; see the module docs for why
/// [`stream`](Model::stream) deliberately returns an error.
pub struct ManagerBridgeModel {
    handle: Arc<dyn ManagerHandle>,
    model_id: String,
}

impl ManagerBridgeModel {
    /// Translate a [`ModelRequest`] into a wire [`CompleteRequest`].
    fn to_complete_request(&self, request: &ModelRequest) -> CompleteRequest {
        let messages = request
            .messages
            .iter()
            .map(message_to_wire)
            .collect::<Vec<_>>();
        // Honor a per-request model override; otherwise use the id this bridge
        // was constructed for.
        let model_id = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_id.clone());
        let response_format_json = match &request.response_format {
            Some(value) => serde_json::to_vec(value).unwrap_or_default(),
            None => Vec::new(),
        };
        CompleteRequest {
            envelope_version: MODEL_ENVELOPE_VERSION,
            model_id,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: Vec::new(),
            response_format_json,
            extra_json: Vec::new(),
            tags: BTreeMap::new(),
        }
    }
}

#[async_trait]
impl Model for ManagerBridgeModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        let wire = self.to_complete_request(&request);
        let model_id = wire.model_id.clone();
        let resp = self
            .handle
            .complete(wire)
            .await
            .map_err(rpc_error_to_blazen)?;

        let usage = match (resp.prompt_tokens, resp.completion_tokens) {
            (None, None) => None,
            (prompt, completion) => {
                let prompt_tokens = prompt.unwrap_or(0);
                let completion_tokens = completion.unwrap_or(0);
                Some(TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens.saturating_add(completion_tokens),
                    ..TokenUsage::default()
                })
            }
        };

        Ok(ModelResponse {
            content: Some(resp.text),
            tool_calls: Vec::new(),
            reasoning: None,
            citations: Vec::new(),
            artifacts: Vec::new(),
            usage,
            model: model_id,
            finish_reason: resp.finish_reason,
            cost: None,
            timing: None,
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            metadata: serde_json::Value::Null,
        })
    }

    async fn stream(
        &self,
        _request: ModelRequest,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>,
        BlazenError,
    > {
        // The local-model fallback path that installs this bridge is
        // non-streaming. Rather than fabricate a single-frame pseudo-stream that
        // pretends to stream, report honestly that streaming through the manager
        // bridge is unsupported; the manager's own `stream_complete` surface is
        // the streaming-capable path.
        Err(BlazenError::unsupported(
            "streaming via the control-plane ManagerHandle bridge is unsupported; \
             use the manager's stream_complete surface directly",
        ))
    }
}

/// Translate a [`blazen_llm::ChatMessage`] into the wire
/// [`ChatMessageWire`]. Plain-text content travels in `text`; non-text content
/// (images, multi-part) is serialised into `content_json` so nothing is
/// silently dropped.
fn message_to_wire(message: &blazen_llm::ChatMessage) -> ChatMessageWire {
    let role = role_str(&message.role).to_owned();
    if let Some(text) = message.content.as_text() {
        ChatMessageWire {
            role,
            text: text.to_owned(),
            content_json: Vec::new(),
        }
    } else {
        let content_json = serde_json::to_vec(&message.content).unwrap_or_default();
        ChatMessageWire {
            role,
            text: String::new(),
            content_json,
        }
    }
}

/// Map a [`blazen_llm::Role`] to its canonical wire string.
fn role_str(role: &blazen_llm::Role) -> &'static str {
    match role {
        blazen_llm::Role::System => "system",
        blazen_llm::Role::User => "user",
        blazen_llm::Role::Assistant => "assistant",
        blazen_llm::Role::Tool => "tool",
    }
}

/// Translate an [`RpcError`] from the manager into a [`BlazenError`].
fn rpc_error_to_blazen(err: RpcError) -> BlazenError {
    use crate::model_protocol::{
        RPC_ERR_INCOMPATIBLE, RPC_ERR_NOT_FOUND, RPC_ERR_QUOTA, RPC_ERR_UNSUPPORTED,
    };
    match err.code {
        RPC_ERR_NOT_FOUND => {
            BlazenError::internal(format!("local model not found: {}", err.message))
        }
        RPC_ERR_UNSUPPORTED => BlazenError::unsupported(err.message),
        RPC_ERR_QUOTA => {
            BlazenError::internal(format!("local model quota exceeded: {}", err.message))
        }
        RPC_ERR_INCOMPATIBLE => {
            BlazenError::internal(format!("local model protocol mismatch: {}", err.message))
        }
        _ => BlazenError::internal(err.message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_protocol::{CompleteResponse, IsLoadedResponse};
    use crate::server::model_manager::test_support::MockManagerHandle;
    use blazen_llm::{ChatMessage, FallbackPolicy};

    /// A [`ManagerHandle`] mock whose `is_loaded` answer is configurable and
    /// whose `complete` echoes a deterministic body so the bridge round-trip can
    /// be asserted. Delegates every other verb to [`MockManagerHandle`].
    struct ConfigurableHandle {
        inner: Arc<MockManagerHandle>,
        loaded: bool,
    }

    impl ConfigurableHandle {
        fn new(loaded: bool) -> Arc<Self> {
            Arc::new(Self {
                inner: MockManagerHandle::new(),
                loaded,
            })
        }
    }

    #[async_trait]
    impl ManagerHandle for ConfigurableHandle {
        async fn load(
            &self,
            req: crate::model_protocol::LoadRequest,
        ) -> Result<crate::model_protocol::LoadResponse, RpcError> {
            self.inner.load(req).await
        }
        async fn unload(
            &self,
            req: crate::model_protocol::UnloadRequest,
        ) -> Result<crate::model_protocol::UnloadResponse, RpcError> {
            self.inner.unload(req).await
        }
        async fn is_loaded(&self, _req: IsLoadedRequest) -> Result<IsLoadedResponse, RpcError> {
            Ok(IsLoadedResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                loaded: self.loaded,
            })
        }
        async fn status(
            &self,
            req: crate::model_protocol::StatusRequest,
        ) -> Result<crate::model_protocol::StatusResponse, RpcError> {
            self.inner.status(req).await
        }
        async fn load_from_hf(
            &self,
            req: crate::model_protocol::LoadFromHfRequest,
        ) -> Result<crate::model_protocol::LoadFromHfResponse, RpcError> {
            self.inner.load_from_hf(req).await
        }
        async fn load_adapter(
            &self,
            req: crate::model_protocol::LoadAdapterRequest,
        ) -> Result<crate::model_protocol::LoadAdapterResponse, RpcError> {
            self.inner.load_adapter(req).await
        }
        async fn unload_adapter(
            &self,
            req: crate::model_protocol::UnloadAdapterRequest,
        ) -> Result<crate::model_protocol::UnloadAdapterResponse, RpcError> {
            self.inner.unload_adapter(req).await
        }
        async fn list_adapters(
            &self,
            req: crate::model_protocol::ListAdaptersRequest,
        ) -> Result<crate::model_protocol::ListAdaptersResponse, RpcError> {
            self.inner.list_adapters(req).await
        }
        async fn complete(&self, req: CompleteRequest) -> Result<CompleteResponse, RpcError> {
            // Echo the message count + first role so the bridge mapping is
            // observable in the response text.
            let first_role = req
                .messages
                .first()
                .map_or_else(String::new, |m| m.role.clone());
            Ok(CompleteResponse {
                envelope_version: MODEL_ENVELOPE_VERSION,
                text: format!("bridge:{}:{first_role}", req.messages.len()),
                prompt_tokens: Some(7),
                completion_tokens: Some(11),
                finish_reason: Some("stop".to_owned()),
                tool_calls_json: Vec::new(),
            })
        }
        async fn stream_complete(
            &self,
            req: CompleteRequest,
        ) -> Result<crate::server::model_manager::StreamCompleteStream, RpcError> {
            self.inner.stream_complete(req).await
        }
        async fn embed(
            &self,
            req: crate::model_protocol::EmbedRequest,
        ) -> Result<crate::model_protocol::EmbedResponse, RpcError> {
            self.inner.embed(req).await
        }
        async fn generate_image(
            &self,
            req: crate::model_protocol::GenerateImageRequest,
        ) -> Result<crate::model_protocol::GenerateImageResponse, RpcError> {
            self.inner.generate_image(req).await
        }
        async fn text_to_speech(
            &self,
            req: crate::model_protocol::TextToSpeechRequest,
        ) -> Result<crate::model_protocol::TextToSpeechResponse, RpcError> {
            self.inner.text_to_speech(req).await
        }
        async fn generate_music(
            &self,
            req: crate::model_protocol::GenerateMusicRequest,
        ) -> Result<crate::model_protocol::GenerateMusicResponse, RpcError> {
            self.inner.generate_music(req).await
        }
        async fn transcribe(
            &self,
            req: crate::model_protocol::TranscribeRequest,
        ) -> Result<crate::model_protocol::TranscribeResponse, RpcError> {
            self.inner.transcribe(req).await
        }
        async fn upload_blob(
            &self,
            chunks: tokio::sync::mpsc::Receiver<crate::model_protocol::UploadBlobChunk>,
        ) -> Result<crate::model_protocol::UploadBlobResponse, RpcError> {
            self.inner.upload_blob(chunks).await
        }
        async fn fetch_blob(
            &self,
            req: crate::model_protocol::FetchBlobRequest,
        ) -> Result<crate::server::model_manager::FetchBlobStream, RpcError> {
            self.inner.fetch_blob(req).await
        }
    }

    #[tokio::test]
    async fn probe_reports_servable_when_loaded() {
        let handle = ConfigurableHandle::new(true);
        let probe = ManagerProbe::new(handle);
        assert!(
            probe
                .is_locally_servable("anything", Some("llama-3-8b"))
                .await
        );
        // model = None falls through to the default id and is still servable.
        assert!(probe.is_locally_servable("anything", None).await);
    }

    #[tokio::test]
    async fn probe_reports_not_servable_when_unloaded() {
        let handle = ConfigurableHandle::new(false);
        let probe = ManagerProbe::new(handle);
        assert!(
            !probe
                .is_locally_servable("anything", Some("llama-3-8b"))
                .await
        );
    }

    #[tokio::test]
    async fn build_local_roundtrips_complete_through_manager() {
        let handle = ConfigurableHandle::new(true);
        let factory = ManagerFactory::new(handle);
        let model = factory
            .build_local("openai", "local-llama")
            .await
            .expect("build_local");
        assert_eq!(model.model_id(), "local-llama");

        let request = ModelRequest::new(vec![
            ChatMessage::system("you are terse"),
            ChatMessage::user("hi"),
        ]);
        let resp = model.complete(request).await.expect("complete");
        // Two messages, first role "system" — proves the wire mapping ran.
        assert_eq!(resp.content.as_deref(), Some("bridge:2:system"));
        assert_eq!(resp.model, "local-llama");
        let usage = resp.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 11);
        assert_eq!(usage.total_tokens, 18);
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn per_request_model_override_wins() {
        let handle = ConfigurableHandle::new(true);
        let factory = ManagerFactory::new(handle);
        let model = factory.build_local("openai", "base").await.unwrap();
        let request = ModelRequest::new(vec![ChatMessage::user("hi")]).with_model("override-model");
        let resp = model.complete(request).await.unwrap();
        assert_eq!(resp.model, "override-model");
    }

    #[tokio::test]
    async fn stream_is_unsupported_not_a_stub() {
        let handle = ConfigurableHandle::new(true);
        let factory = ManagerFactory::new(handle);
        let model = factory.build_local("openai", "local").await.unwrap();
        let request = ModelRequest::new(vec![ChatMessage::user("hi")]);
        // The stream future resolves to `Err`; the Ok variant isn't `Debug`, so
        // match the result rather than calling `expect_err`.
        match model.stream(request).await {
            Err(BlazenError::Unsupported { .. }) => {}
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("stream must error, not return a stream"),
        }
    }

    /// End-to-end through `build_model`: no key + servable probe +
    /// `WhenNoKey` ⇒ the manager bridge is selected.
    #[tokio::test]
    async fn build_model_routes_to_manager_bridge() {
        // Ensure no env key resolves for this synthetic provider.
        let provider = "cp_adapter_test_provider";
        let handle = ConfigurableHandle::new(true);
        let probe = ManagerProbe::new(Arc::clone(&handle) as Arc<dyn ManagerHandle>);
        let factory = ManagerFactory::new(handle);

        let opts = blazen_llm::types::provider_options::ProviderOptions {
            model: Some("local-llama".into()),
            ..Default::default()
        };
        let model = blazen_llm::build_model(
            provider,
            opts,
            FallbackPolicy::WhenNoKey,
            &probe,
            &factory,
            |_key, _opts| {
                panic!("remote builder must not run when no key resolves");
            },
        )
        .await
        .expect("build_model should route local");
        assert_eq!(model.model_id(), "local-llama");
        let resp = model
            .complete(ModelRequest::new(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        assert_eq!(resp.content.as_deref(), Some("bridge:1:user"));
    }
}
