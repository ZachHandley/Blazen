//! Shared embedding model wrappers used by multiple cloud providers.
//!
//! This module exposes [`JsOpenAiEmbeddingModel`] and
//! [`JsOpenAiCompatEmbeddingModel`] as standalone NAPI classes so callers
//! can construct embedding models directly from any provider that backs
//! its embedding endpoint with the OpenAI-compatible wire format
//! (Together, Cohere, Fireworks, etc.).

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::providers::openai::OpenAiEmbeddingModel;
use blazen_llm::providers::openai_compat::OpenAiCompatEmbeddingModel;
use blazen_llm::traits::EmbeddingModel;

use crate::error::blazen_error_to_napi;
use crate::generated::JsProviderOptions;

/// An `OpenAI` embedding model.
///
/// ```typescript
/// const em = OpenAiEmbeddingModel.create({ apiKey: "sk-..." });
/// const vectors = await em.embed(["hello", "world"]);
/// ```
#[napi(js_name = "OpenAiEmbeddingModel")]
pub struct JsOpenAiEmbeddingModel {
    inner: Arc<OpenAiEmbeddingModel>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsOpenAiEmbeddingModel {
    /// Create a new `OpenAI` embedding model.
    ///
    /// Defaults to `text-embedding-3-small` (1536 dimensions).
    #[napi(factory)]
    pub fn create(options: Option<JsProviderOptions>) -> Result<Self> {
        let api_key = blazen_llm::keys::resolve_api_key("openai", options.and_then(|o| o.api_key))
            .map_err(blazen_error_to_napi)?;
        Ok(Self {
            inner: Arc::new(OpenAiEmbeddingModel::new(api_key)),
        })
    }

    /// The underlying embedding model id.
    #[napi(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// The dimensionality of the produced embedding vectors.
    #[napi(getter)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    /// Embed one or more texts.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        let inner = Arc::clone(&self.inner);
        let response = inner
            .embed(&texts)
            .await
            .map_err(|e| napi::Error::from_reason(format!("openai embed: {e}")))?;
        Ok(response
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect())
            .collect())
    }
}

/// An `OpenAI`-compatible embedding model (Together, Cohere, Fireworks).
///
/// Constructed via the per-provider `embeddingModel(...)` factory on
/// each provider class.
#[napi(js_name = "OpenAiCompatEmbeddingModel")]
pub struct JsOpenAiCompatEmbeddingModel {
    inner: Arc<OpenAiCompatEmbeddingModel>,
}

impl JsOpenAiCompatEmbeddingModel {
    /// Wrap a Rust `OpenAiCompatEmbeddingModel` for exposure to JavaScript.
    #[must_use]
    pub fn wrap(inner: Arc<OpenAiCompatEmbeddingModel>) -> Self {
        Self { inner }
    }
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsOpenAiCompatEmbeddingModel {
    /// The underlying embedding model id.
    #[napi(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// The dimensionality of the produced embedding vectors.
    #[napi(getter)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    /// Embed one or more texts.
    #[napi]
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        let inner = Arc::clone(&self.inner);
        let response = inner
            .embed(&texts)
            .await
            .map_err(|e| napi::Error::from_reason(format!("openai-compat embed: {e}")))?;
        Ok(response
            .embeddings
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect())
            .collect())
    }
}
