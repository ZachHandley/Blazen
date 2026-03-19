//! Embedding model and response types for the Node.js bindings.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::traits::EmbeddingModel;

use super::usage::{JsRequestTiming, JsTokenUsage};
use crate::error::llm_error_to_napi;

// ---------------------------------------------------------------------------
// EmbeddingResponse
// ---------------------------------------------------------------------------

/// The result of an embedding operation.
#[napi(object)]
pub struct JsEmbeddingResponse {
    /// The embedding vectors (one per input text).
    pub embeddings: Vec<Vec<f64>>,
    /// The model that produced these embeddings.
    pub model: String,
    /// Token usage statistics, if available.
    pub usage: Option<JsTokenUsage>,
    /// Estimated cost in USD, if available.
    pub cost: Option<f64>,
    /// Request timing breakdown, if available.
    pub timing: Option<JsRequestTiming>,
    /// Provider-specific metadata.
    pub metadata: serde_json::Value,
}

// ---------------------------------------------------------------------------
// EmbeddingModel wrapper
// ---------------------------------------------------------------------------

/// An embedding model that produces vector representations of text.
///
/// Use the static factory methods to create an instance for your provider:
///
/// ```javascript
/// const model = EmbeddingModel.openai("sk-...");
/// const response = await model.embed(["Hello", "World"]);
/// console.log(response.embeddings); // [[0.1, ...], [0.3, ...]]
/// ```
#[napi(js_name = "EmbeddingModel")]
pub struct JsEmbeddingModel {
    inner: Arc<dyn EmbeddingModel>,
}

impl JsEmbeddingModel {
    /// Access the inner embedding model (used by memory bindings).
    pub(crate) fn inner_arc(&self) -> Arc<dyn EmbeddingModel> {
        Arc::clone(&self.inner)
    }
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsEmbeddingModel {
    // -----------------------------------------------------------------
    // Provider factory methods
    // -----------------------------------------------------------------

    /// Create an `OpenAI` embedding model.
    ///
    /// Defaults to `text-embedding-3-small` (1536 dimensions).
    #[napi(factory)]
    pub fn openai(api_key: String) -> Self {
        Self {
            inner: Arc::new(blazen_llm::providers::openai::OpenAiEmbeddingModel::new(
                api_key,
            )),
        }
    }

    /// Create a Together AI embedding model.
    ///
    /// Defaults to `togethercomputer/m2-bert-80M-8k-retrieval` (768 dimensions).
    #[napi(factory)]
    pub fn together(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::together(api_key),
            ),
        }
    }

    /// Create a Cohere embedding model.
    ///
    /// Defaults to `embed-v4.0` (1024 dimensions).
    #[napi(factory)]
    pub fn cohere(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::cohere(api_key),
            ),
        }
    }

    /// Create a Fireworks AI embedding model.
    ///
    /// Defaults to `nomic-ai/nomic-embed-text-v1.5` (768 dimensions).
    #[napi(factory)]
    pub fn fireworks(api_key: String) -> Self {
        Self {
            inner: Arc::new(
                blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel::fireworks(
                    api_key,
                ),
            ),
        }
    }

    // -----------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------

    /// The model identifier.
    #[napi(js_name = "modelId", getter)]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// The dimensionality of the embedding vectors produced by this model.
    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.dimensions() as u32
        }
    }

    // -----------------------------------------------------------------
    // Embed
    // -----------------------------------------------------------------

    /// Embed one or more texts, returning one vector per input text.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<JsEmbeddingResponse> {
        let response = self.inner.embed(&texts).await.map_err(llm_error_to_napi)?;

        Ok(JsEmbeddingResponse {
            // napi does not support Vec<Vec<f32>>; convert to f64 for JS.
            embeddings: response
                .embeddings
                .into_iter()
                .map(|v| v.into_iter().map(f64::from).collect())
                .collect(),
            model: response.model,
            usage: response.usage.map(|u| JsTokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            cost: response.cost,
            #[allow(clippy::cast_possible_wrap)]
            timing: response.timing.map(|t| JsRequestTiming {
                queue_ms: t.queue_ms.map(|v| v as i64),
                execution_ms: t.execution_ms.map(|v| v as i64),
                total_ms: t.total_ms.map(|v| v as i64),
            }),
            metadata: response.metadata,
        })
    }
}
