//! JavaScript bindings for the Perplexity provider.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::CompletionModel;
use blazen_llm::providers::perplexity::PerplexityProvider;
use blazen_llm::types::provider_options::ProviderOptions;
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};

use crate::error::{blazen_error_to_napi, llm_error_to_napi};
use crate::generated::JsProviderOptions;
use crate::providers::completion_model::StreamChunkCallbackTsfn;
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

/// A Perplexity chat completion provider.
#[napi(js_name = "PerplexityProvider")]
pub struct JsPerplexityProvider {
    inner: Arc<PerplexityProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl JsPerplexityProvider {
    /// Create a new Perplexity provider.
    #[napi(factory)]
    pub fn create(options: Option<JsProviderOptions>) -> Result<Self> {
        let opts: ProviderOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(PerplexityProvider::from_options(opts).map_err(blazen_error_to_napi)?),
        })
    }

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Perform a chat completion.
    #[napi]
    pub async fn complete(&self, messages: Vec<&JsChatMessage>) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let request = CompletionRequest::new(chat_messages);
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
        options: JsCompletionOptions,
    ) -> Result<JsCompletionResponse> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);
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
        let request = CompletionRequest::new(chat_messages);
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

    /// Stream a chat completion with additional options.
    #[napi(js_name = "streamWithOptions")]
    pub async fn stream_with_options(
        &self,
        messages: Vec<&JsChatMessage>,
        on_chunk: StreamChunkCallbackTsfn,
        options: JsCompletionOptions,
    ) -> Result<()> {
        let chat_messages: Vec<ChatMessage> = messages.iter().map(|m| m.inner.clone()).collect();
        let mut request = CompletionRequest::new(chat_messages);
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
