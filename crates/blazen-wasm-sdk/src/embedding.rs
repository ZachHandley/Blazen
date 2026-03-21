//! `wasm-bindgen` wrapper for [`blazen_llm::EmbeddingModel`].
//!
//! Exposes an `EmbeddingModel` class to JavaScript with factory methods for
//! OpenAI-compatible embedding providers and an async `embed()` method that
//! returns embedding vectors as a nested JS array.

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::http::HttpClient;
use blazen_llm::providers::openai_compat::{
    AuthMethod, OpenAiCompatConfig, OpenAiCompatEmbeddingModel,
};
use blazen_llm::traits::EmbeddingModel;

use blazen_llm::FetchHttpClient;

// ---------------------------------------------------------------------------
// WasmEmbeddingModel
// ---------------------------------------------------------------------------

/// A provider-agnostic text embedding model.
///
/// Use the static factory methods to create a model for a specific provider:
///
/// ```js
/// const embedder = EmbeddingModel.openai('sk-...');
/// const result = await embedder.embed(['Hello', 'World']);
/// console.log(result); // [[0.1, 0.2, ...], [0.3, 0.4, ...]]
/// ```
#[wasm_bindgen(js_name = "EmbeddingModel")]
pub struct WasmEmbeddingModel {
    inner: Arc<dyn EmbeddingModel>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmEmbeddingModel {}
unsafe impl Sync for WasmEmbeddingModel {}

/// Create an `OpenAiCompatEmbeddingModel` backed by the fetch HTTP client.
fn compat_embedding_with_fetch(model: OpenAiCompatEmbeddingModel) -> OpenAiCompatEmbeddingModel {
    let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
    model.with_http_client(client)
}

#[wasm_bindgen(js_class = "EmbeddingModel")]
impl WasmEmbeddingModel {
    // -----------------------------------------------------------------------
    // Factory methods
    // -----------------------------------------------------------------------

    /// OpenAI (`text-embedding-3-small`, 1536 dimensions).
    #[wasm_bindgen]
    pub fn openai(api_key: &str) -> Self {
        let model = compat_embedding_with_fetch(OpenAiCompatEmbeddingModel::new(
            OpenAiCompatConfig {
                provider_name: "openai".into(),
                base_url: "https://api.openai.com/v1".into(),
                api_key: api_key.into(),
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "text-embedding-3-small",
            1536,
        ));
        Self {
            inner: Arc::new(model),
        }
    }

    /// Together AI (`togethercomputer/m2-bert-80M-8k-retrieval`, 768 dimensions).
    #[wasm_bindgen]
    pub fn together(api_key: &str) -> Self {
        let model =
            compat_embedding_with_fetch(OpenAiCompatEmbeddingModel::together(api_key));
        Self {
            inner: Arc::new(model),
        }
    }

    /// Cohere (`embed-v4.0`, 1024 dimensions).
    #[wasm_bindgen]
    pub fn cohere(api_key: &str) -> Self {
        let model =
            compat_embedding_with_fetch(OpenAiCompatEmbeddingModel::cohere(api_key));
        Self {
            inner: Arc::new(model),
        }
    }

    /// Fireworks AI (`nomic-ai/nomic-embed-text-v1.5`, 768 dimensions).
    #[wasm_bindgen]
    pub fn fireworks(api_key: &str) -> Self {
        let model =
            compat_embedding_with_fetch(OpenAiCompatEmbeddingModel::fireworks(api_key));
        Self {
            inner: Arc::new(model),
        }
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// The model identifier used by this embedding provider.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// The dimensionality of vectors produced by this model.
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.inner.dimensions() as u32
        }
    }

    // -----------------------------------------------------------------------
    // Embed
    // -----------------------------------------------------------------------

    /// Embed one or more texts, returning a nested array of float vectors.
    ///
    /// ```js
    /// const result = await embedder.embed(['Hello world', 'Goodbye']);
    /// // result is an array of arrays: [[0.1, -0.2, ...], [0.3, 0.4, ...]]
    /// ```
    ///
    /// Returns a `Promise<number[][]>`.
    #[wasm_bindgen]
    pub fn embed(&self, texts: Vec<String>) -> js_sys::Promise {
        let model = Arc::clone(&self.inner);
        future_to_promise(async move {
            let response = model
                .embed(&texts)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Convert Vec<Vec<f32>> to a JS nested array.
            let outer = js_sys::Array::new_with_length(response.embeddings.len() as u32);
            for (i, embedding) in response.embeddings.iter().enumerate() {
                let inner = js_sys::Float32Array::new_with_length(embedding.len() as u32);
                inner.copy_from(embedding);
                outer.set(i as u32, inner.into());
            }

            Ok(outer.into())
        })
    }
}
