//! JavaScript bindings for the local mistral.rs LLM provider.
//!
//! Exposes [`JsMistralRsProvider`] as a NAPI class with a factory
//! constructor, async completion / streaming methods, and `LocalModel`
//! lifecycle controls (`load`, `unload`, `isLoaded`, `vramBytes`).
//!
//! Runs LLM inference entirely on-device using the mistral.rs engine.
//! No API key is required.

#![cfg(feature = "mistralrs")]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tokio_stream::StreamExt;

use blazen_llm::traits::{CompletionModel, LocalModel};
use blazen_llm::types::{ChatMessage, CompletionRequest, ToolDefinition};
use blazen_llm::{MistralRsOptions, MistralRsProvider};

use crate::error::{llm_error_to_napi, mistralrs_error_to_napi, to_napi_error};
use crate::providers::completion_model::{JsMistralRsOptions, StreamChunkCallbackTsfn};
use crate::types::{
    JsChatMessage, JsCompletionOptions, JsCompletionResponse, build_response, build_stream_chunk,
};

// ---------------------------------------------------------------------------
// JsMistralRsProvider NAPI class
// ---------------------------------------------------------------------------

/// A local mistral.rs LLM provider with completion and streaming.
///
/// ```javascript
/// const provider = MistralRsProvider.create({
///   modelId: "mistralai/Mistral-7B-Instruct-v0.3",
/// });
/// await provider.load();
/// const response = await provider.complete([
///   ChatMessage.user("What is 2+2?"),
/// ]);
/// ```
#[napi(js_name = "MistralRsProvider")]
pub struct JsMistralRsProvider {
    inner: Arc<MistralRsProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
impl JsMistralRsProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new mistral.rs provider.
    #[napi(factory)]
    pub fn create(options: JsMistralRsOptions) -> Result<Self> {
        let opts: MistralRsOptions = options.into();
        Ok(Self {
            inner: Arc::new(
                MistralRsProvider::from_options(opts).map_err(mistralrs_error_to_napi)?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Model info
    // -----------------------------------------------------------------

    /// Get the model ID.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Completion
    // -----------------------------------------------------------------

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
                Err(e) => {
                    return Err(napi::Error::from_reason(e.to_string()));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // LocalModel lifecycle
    // -----------------------------------------------------------------

    /// Explicitly load the model weights into memory / `VRAM`.
    #[napi]
    pub async fn load(&self) -> Result<()> {
        self.inner.load().await.map_err(to_napi_error)
    }

    /// Drop the loaded model and free its memory / `VRAM`.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        self.inner.unload().await.map_err(to_napi_error)
    }

    /// Whether the model is currently loaded in memory / `VRAM`.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> bool {
        self.inner.is_loaded().await
    }

    /// Approximate `VRAM` footprint in bytes.
    #[napi(js_name = "vramBytes")]
    pub async fn vram_bytes(&self) -> Option<i64> {
        self.inner
            .vram_bytes()
            .await
            .map(|b| i64::try_from(b).unwrap_or(i64::MAX))
    }
}
