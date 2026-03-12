//! Core traits defining the LLM provider abstraction layer.
//!
//! [`CompletionModel`] is the central trait that every provider must implement.
//! [`StructuredOutput`] and [`EmbeddingModel`] extend the surface area with
//! schema-constrained extraction and vector embeddings respectively.
//! [`ModelRegistry`] allows providers to advertise their available models.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::types::{
    ChatMessage, CompletionRequest, CompletionResponse, StreamChunk, ToolDefinition,
};

// ---------------------------------------------------------------------------
// CompletionModel
// ---------------------------------------------------------------------------

/// A chat completion model capable of generating text and invoking tools.
///
/// Implementors must handle both one-shot and streaming completions.
#[async_trait]
pub trait CompletionModel: Send + Sync {
    /// The identifier of the default model used by this provider.
    fn model_id(&self) -> &str;

    /// Perform a non-streaming chat completion.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    /// Perform a streaming chat completion, returning an async stream of chunks.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError>;
}

// ---------------------------------------------------------------------------
// StructuredOutput
// ---------------------------------------------------------------------------

/// Extract structured data from a model by providing a JSON Schema.
///
/// This trait has a blanket implementation for every [`CompletionModel`], so
/// providers do not need to implement it explicitly. It works by injecting the
/// schema derived from `T` via `schemars` into the `response_format` field of
/// the completion request.
#[async_trait]
pub trait StructuredOutput: CompletionModel {
    /// Extract a value of type `T` from the model's response.
    ///
    /// The conversation in `messages` is sent to the model with a JSON Schema
    /// constraint derived from `T`. The response is then deserialized into `T`.
    async fn extract<T: JsonSchema + DeserializeOwned + Send>(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<T, LlmError> {
        let schema = schemars::schema_for!(T);
        let schema_json = serde_json::to_value(&schema)?;

        let request = CompletionRequest {
            messages,
            tools: vec![],
            temperature: Some(0.0),
            max_tokens: None,
            top_p: None,
            response_format: Some(schema_json),
            model: None,
        };

        let response = self.complete(request).await?;
        let content = response.content.ok_or(LlmError::NoContent)?;
        let parsed: T = serde_json::from_str(&content)?;
        Ok(parsed)
    }
}

/// Blanket implementation: every [`CompletionModel`] automatically supports
/// structured output extraction.
impl<M: CompletionModel> StructuredOutput for M {}

// ---------------------------------------------------------------------------
// EmbeddingModel
// ---------------------------------------------------------------------------

/// A model that produces vector embeddings for text inputs.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// The identifier of the embedding model.
    fn model_id(&self) -> &str;

    /// The dimensionality of the vectors produced by this model.
    fn dimensions(&self) -> usize;

    /// Embed one or more texts, returning one vector per input text.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError>;
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// A callable tool that can be invoked by an LLM during a conversation.
///
/// Implementations describe their schema via [`Tool::definition`] and handle
/// invocations via [`Tool::execute`].
#[async_trait]
pub trait Tool: Send + Sync {
    /// Return the JSON Schema definition of this tool.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given arguments and return the result.
    async fn execute(&self, arguments: serde_json::Value) -> Result<serde_json::Value, LlmError>;
}

// ---------------------------------------------------------------------------
// Model information and registry
// ---------------------------------------------------------------------------

/// Information about a model offered by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The model identifier used in API requests (e.g. `"gpt-4o"`).
    pub id: String,
    /// A human-readable display name, if different from the id.
    pub name: Option<String>,
    /// The provider that serves this model.
    pub provider: String,
    /// Maximum context window length in tokens.
    pub context_length: Option<u64>,
    /// Pricing information, if available.
    pub pricing: Option<ModelPricing>,
    /// What this model can do.
    pub capabilities: ModelCapabilities,
}

/// Pricing information for a model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelPricing {
    /// Cost per million input tokens in USD.
    pub input_per_million: Option<f64>,
    /// Cost per million output tokens in USD.
    pub output_per_million: Option<f64>,
    /// Cost per image (for image generation models).
    pub per_image: Option<f64>,
    /// Cost per second of compute (for fal.ai style pricing).
    pub per_second: Option<f64>,
}

/// Capabilities that a model may support.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// Supports chat completions.
    pub chat: bool,
    /// Supports streaming responses.
    pub streaming: bool,
    /// Supports tool/function calling.
    pub tool_use: bool,
    /// Supports structured output (JSON schema constraints).
    pub structured_output: bool,
    /// Supports vision / image inputs.
    pub vision: bool,
    /// Supports image generation.
    pub image_generation: bool,
    /// Supports text embeddings.
    pub embeddings: bool,
}

/// A provider that can list its available models.
#[async_trait]
pub trait ModelRegistry: Send + Sync {
    /// List all models available from this provider.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Look up a specific model by its identifier.
    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>, LlmError>;
}
