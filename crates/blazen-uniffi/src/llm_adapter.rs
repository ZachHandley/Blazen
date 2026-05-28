//! Adapter bridging `Arc<dyn LlmProvider>` (uniffi capability trait) onto
//! the blazen-llm domain `Model` trait, so the existing
//! `core_run_agent` / `core_complete_batch` functions can drive any
//! per-engine uniffi LLM provider without going through the deleted
//! central `Model` opaque.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use blazen_llm::error::BlazenError as CoreBlazenError;
use blazen_llm::traits::Model as CoreModel;
use blazen_llm::types::{
    ModelRequest as CoreModelRequest, ModelResponse as CoreModelResponse,
    StreamChunk as CoreStreamChunk,
};

use crate::concrete::bases::LlmProvider;
use crate::llm::ModelRequest;

/// Wraps an `Arc<dyn LlmProvider>` so it satisfies the blazen-llm domain
/// `Model` trait. Used internally by `Agent` and `complete_batch` so foreign
/// callers can pass any per-engine LLM provider class (`OpenAiProvider`,
/// `AnthropicProvider`, ...) without depending on the central `Model` opaque.
///
/// The bridge round-trips requests/responses through the uniffi wire format,
/// which loses some core-only fields (`modalities`, `image_config`,
/// `audio_config` on the request; `cost`, `timing`, `metadata`, `reasoning`,
/// `citations`, `artifacts`, `images`, `audio`, `videos` on the response).
/// This is acceptable for the agent loop and batch completion — both consume
/// only `content`, `tool_calls`, and `usage`.
pub(crate) struct UniffiLlmAdapter {
    inner: Arc<dyn LlmProvider>,
    /// Cached so `model_id(&self) -> &str` can return a borrow. Populated
    /// from the wrapped provider's `provider_id()` at construction time —
    /// the uniffi `LlmProvider` trait doesn't carry a per-request model id
    /// today, so the engine identifier (`"openai"`, `"anthropic"`, ...)
    /// stands in. Telemetry surfacing this string will show the engine
    /// rather than the model variant; the actual per-call model id flows
    /// through `ModelResponse.model` and is unaffected.
    model_id: String,
}

impl UniffiLlmAdapter {
    pub(crate) fn from_provider(inner: Arc<dyn LlmProvider>) -> Self {
        let model_id = inner.provider_id();
        Self { inner, model_id }
    }

    /// Build an `Arc<dyn CoreModel>` ready to hand to `core_run_agent` /
    /// `core_complete_batch`.
    pub(crate) fn into_core_model(inner: Arc<dyn LlmProvider>) -> Arc<dyn CoreModel> {
        Arc::new(Self::from_provider(inner))
    }
}

#[async_trait]
impl CoreModel for UniffiLlmAdapter {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        request: CoreModelRequest,
    ) -> Result<CoreModelResponse, CoreBlazenError> {
        let uniffi_req = ModelRequest::from(request);
        let uniffi_resp = self
            .inner
            .complete(uniffi_req)
            .await
            .map_err(uniffi_err_to_core)?;
        CoreModelResponse::try_from(uniffi_resp).map_err(uniffi_err_to_core)
    }

    async fn stream(
        &self,
        _request: CoreModelRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CoreStreamChunk, CoreBlazenError>> + Send>>,
        CoreBlazenError,
    > {
        Err(CoreBlazenError::unsupported(
            "streaming via Arc<dyn LlmProvider> is not yet supported in Agent/Batch — \
             use the per-engine `<engine>_provider_complete_streaming` free functions instead",
        ))
    }
}

/// Flatten a uniffi `BlazenError` into a core `BlazenError`. The agent loop
/// and batch path don't need to preserve typed variants here (only
/// `tool_handler` paths surface `CallerError`, and that lives elsewhere) —
/// the message round-trips through `Display`.
fn uniffi_err_to_core(err: crate::errors::BlazenError) -> CoreBlazenError {
    CoreBlazenError::tool_error(err.to_string())
}
