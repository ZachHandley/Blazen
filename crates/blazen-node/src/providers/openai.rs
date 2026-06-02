//! JavaScript bindings for the `OpenAI` provider.
//!
//! Exposes [`JsOpenAiProvider`] as a NAPI class with a factory constructor,
//! chat-completion / streaming methods, and a `textToSpeech` async method
//! backed by `OpenAI`'s `/v1/audio/speech` endpoint. This is the standalone
//! class form of [`Model.openai`](crate::providers::model::JsModel::openai);
//! both surfaces wrap the same Rust provider.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::Model;
use blazen_llm::compute::AudioGeneration;
use blazen_llm::providers::openai::OpenAiProvider;
use blazen_llm::types::provider_options::ProviderOptions;
use blazen_llm::types::{ChatMessage, ModelRequest, ToolDefinition};

use crate::error::{blazen_error_to_napi, llm_error_to_napi};
use crate::generated::{JsAudioResult, JsProviderOptions, JsSpeechRequest};
use crate::providers::model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsModelOptions, JsModelResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// OpenAiProvider NAPI class
// ---------------------------------------------------------------------------

/// An `OpenAI` provider exposing chat completion, streaming, and text-to-speech.
///
/// This is the standalone class form of
/// [`Model.openai`](crate::providers::model::JsModel::openai); both surfaces
/// wrap the same Rust provider.
///
/// ```typescript
/// const openai = OpenAiProvider.create({ apiKey: "sk-..." });
/// const response = await openai.complete([ChatMessage.user("Hi")]);
/// const audio = await openai.textToSpeech({
///     text: "Hello, world!",
///     voice: "alloy",
/// });
/// ```
#[napi(js_name = "OpenAiProvider")]
pub struct JsOpenAiProvider {
    inner: Arc<OpenAiProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsOpenAiProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new `OpenAI` provider.
    ///
    /// `options` optionally overrides the API key, model, and base URL.
    /// When omitted, the API key is read from the `OPENAI_API_KEY`
    /// environment variable and the defaults from
    /// [`OpenAiProvider::from_options`] are applied.
    #[napi(factory)]
    pub fn create(options: Option<JsProviderOptions>) -> Result<Self> {
        let opts: ProviderOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(OpenAiProvider::from_options(opts).map_err(blazen_error_to_napi)?),
        })
    }

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Chat completion
    // -----------------------------------------------------------------

    /// Perform a chat completion.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsModelResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = ModelRequest::new(chat_messages);
        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(build_response(response))
    }

    /// Perform a chat completion with additional options.
    #[napi(js_name = "completeWithOptions")]
    pub async fn complete_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        options: JsModelOptions,
    ) -> Result<JsModelResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = apply_options(ModelRequest::new(chat_messages), options);
        let response = self
            .inner
            .complete(request)
            .await
            .map_err(llm_error_to_napi)?;
        Ok(build_response(response))
    }

    /// Stream a chat completion.
    #[napi]
    pub async fn stream(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = ModelRequest::new(chat_messages);
        self.run_stream(request, on_chunk).await
    }

    /// Stream a chat completion with additional options.
    #[napi(js_name = "streamWithOptions")]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsModelOptions,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = apply_options(ModelRequest::new(chat_messages), options);
        self.run_stream(request, on_chunk).await
    }

    // -----------------------------------------------------------------
    // Compute methods -- Audio
    // -----------------------------------------------------------------

    /// Synthesize speech from text via `OpenAI`'s `/v1/audio/speech`.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsSpeechRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::SpeechRequest = request.into();
        let inner = Arc::clone(&self.inner);
        let result = AudioGeneration::text_to_speech(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }
}

impl JsOpenAiProvider {
    /// Drive a streaming completion, forwarding each chunk to `on_chunk`.
    async fn run_stream(
        &self,
        request: ModelRequest,
        on_chunk: StreamChunkCallbackTsfn,
    ) -> Result<()> {
        let stream = self
            .inner
            .stream(request)
            .await
            .map_err(llm_error_to_napi)?;
        let mut stream = std::pin::pin!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    on_chunk.call(
                        build_stream_chunk(chunk),
                        ThreadsafeFunctionCallMode::Blocking,
                    );
                }
                Err(e) => return Err(napi::Error::from_reason(e.to_string())),
            }
        }
        Ok(())
    }
}

/// Apply optional sampling parameters, model override, tools, and response
/// format onto a [`ModelRequest`].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn apply_options(mut request: ModelRequest, options: JsModelOptions) -> ModelRequest {
    if let Some(temp) = options.temperature {
        request.temperature = Some(temp as f32);
    }
    if let Some(max) = options.max_tokens {
        request.max_tokens = Some(max as u32);
    }
    if let Some(top_p) = options.top_p {
        request.top_p = Some(top_p as f32);
    }
    if let Some(model) = options.model {
        request.model = Some(model);
    }
    if let Some(tools) = options.tools {
        request.tools = tools
            .into_iter()
            .map(|t| ToolDefinition {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
            })
            .collect();
    }
    if let Some(fmt) = options.response_format {
        request = request.with_response_format(fmt);
    }
    request
}
