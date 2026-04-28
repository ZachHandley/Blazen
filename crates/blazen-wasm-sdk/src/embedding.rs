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

use crate::js_embedding::JsEmbeddingHandler;

// ---------------------------------------------------------------------------
// TransformersJsOptions
// ---------------------------------------------------------------------------

/// Options for the `EmbeddingModel.transformersJs()` convenience factory.
///
/// Controls quantisation and declared embedding dimensions for models loaded
/// via `@huggingface/transformers` (formerly `@xenova/transformers`).
#[wasm_bindgen(js_name = "TransformersJsOptions")]
pub struct TransformersJsOptions {
    /// Whether to load the quantized (ONNX int8) variant of the model.
    /// Defaults to `true`.
    pub quantized: bool,

    /// The dimensionality of the embedding vectors produced by the model.
    /// Defaults to 384 (correct for `all-MiniLM-L6-v2` and similar small
    /// sentence-transformer models).
    pub dimensions: u32,
}

#[wasm_bindgen(js_class = "TransformersJsOptions")]
impl TransformersJsOptions {
    /// Create default options (`quantized: true`, `dimensions: 384`).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            quantized: true,
            dimensions: 384,
        }
    }
}

impl Default for TransformersJsOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WasmEmbeddingModel
// ---------------------------------------------------------------------------

/// A provider-agnostic text embedding model.
///
/// Use the static factory methods to create a model for a specific provider:
///
/// ```js
/// const embedder = EmbeddingModel.openai();
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

impl WasmEmbeddingModel {
    /// Access the inner `Arc<dyn EmbeddingModel>` for use by other crate modules.
    pub(crate) fn inner_arc(&self) -> Arc<dyn EmbeddingModel> {
        Arc::clone(&self.inner)
    }

    /// Construct a [`WasmEmbeddingModel`] from any `EmbeddingModel`
    /// implementation. Used by sibling factory modules (e.g. `embed_tract`)
    /// to build a JS-facing wrapper around their adapter type.
    pub(crate) fn from_inner(inner: Arc<dyn EmbeddingModel>) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(typescript_custom_section)]
const TS_EMBED_HANDLER: &str = r#"
/** Callback for local/in-browser embedding via `EmbeddingModel.fromJsHandler`. */
export type EmbedHandler = (texts: string[]) => Promise<Float32Array[]> | Float32Array[];
"#;

/// Create an `OpenAiCompatEmbeddingModel` backed by the fetch HTTP client.
fn compat_embedding_with_fetch(
    config: OpenAiCompatConfig,
    model: &str,
    dimensions: usize,
) -> OpenAiCompatEmbeddingModel {
    let client: Arc<dyn HttpClient> = FetchHttpClient::new().into_arc();
    OpenAiCompatEmbeddingModel::new_with_client(config, model, dimensions, client)
}

#[wasm_bindgen(js_class = "EmbeddingModel")]
impl WasmEmbeddingModel {
    // -----------------------------------------------------------------------
    // Factory methods
    // -----------------------------------------------------------------------

    /// OpenAI (`text-embedding-3-small`, 1536 dimensions).
    #[wasm_bindgen]
    pub fn openai() -> Result<WasmEmbeddingModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("openai", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = compat_embedding_with_fetch(
            OpenAiCompatConfig {
                provider_name: "openai".into(),
                base_url: "https://api.openai.com/v1".into(),
                api_key,
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "text-embedding-3-small",
            1536,
        );
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Together AI (`togethercomputer/m2-bert-80M-8k-retrieval`, 768 dimensions).
    #[wasm_bindgen]
    pub fn together() -> Result<WasmEmbeddingModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("together", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = compat_embedding_with_fetch(
            OpenAiCompatConfig {
                provider_name: "together".into(),
                base_url: "https://api.together.xyz/v1".into(),
                api_key,
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "togethercomputer/m2-bert-80M-8k-retrieval",
            768,
        );
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Cohere (`embed-v4.0`, 1024 dimensions).
    #[wasm_bindgen]
    pub fn cohere() -> Result<WasmEmbeddingModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("cohere", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = compat_embedding_with_fetch(
            OpenAiCompatConfig {
                provider_name: "cohere".into(),
                base_url: "https://api.cohere.ai/compatibility/v1".into(),
                api_key,
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "embed-v4.0",
            1024,
        );
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Fireworks AI (`nomic-ai/nomic-embed-text-v1.5`, 768 dimensions).
    #[wasm_bindgen]
    pub fn fireworks() -> Result<WasmEmbeddingModel, JsValue> {
        let api_key = blazen_llm::keys::resolve_api_key("fireworks", None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = compat_embedding_with_fetch(
            OpenAiCompatConfig {
                provider_name: "fireworks".into(),
                base_url: "https://api.fireworks.ai/inference/v1".into(),
                api_key,
                default_model: String::new(),
                auth_method: AuthMethod::Bearer,
                extra_headers: Vec::new(),
                query_params: Vec::new(),
                supports_model_listing: false,
            },
            "nomic-ai/nomic-embed-text-v1.5",
            768,
        );
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    /// Create an embedding model backed by a JavaScript callback function.
    ///
    /// The handler receives a `string[]` and must return
    /// `Promise<Float32Array[]> | Float32Array[]`.
    ///
    /// This lets you wrap local inference libraries like `transformers.js` or
    /// ONNX Runtime Web:
    ///
    /// ```js
    /// const embedder = EmbeddingModel.fromJsHandler(
    ///   'all-MiniLM-L6-v2',
    ///   384,
    ///   async (texts) => {
    ///     const results = await pipe(texts, { pooling: 'mean', normalize: true });
    ///     return Array.from({ length: texts.length }, (_, i) => results[i].data);
    ///   },
    /// );
    /// ```
    #[must_use]
    #[wasm_bindgen(js_name = "fromJsHandler")]
    pub fn from_js_handler(
        model_id: String,
        dimensions: u32,
        handler: js_sys::Function,
    ) -> Self {
        let wrapper = JsEmbeddingHandler::new(model_id, dimensions as usize, handler);
        Self {
            inner: Arc::new(wrapper),
        }
    }

    /// Create an embedding model backed by `@huggingface/transformers`.
    ///
    /// The library is **dynamically imported** on the first `embed()` call,
    /// so there is no top-level `await` and the WASM init path stays
    /// synchronous.  If the package is not installed the first call fails
    /// with a clear import error.
    ///
    /// **Requires:** `npm install @huggingface/transformers`
    ///
    /// # Errors
    ///
    /// This factory itself is infallible in practice; errors from the dynamic
    /// import surface at `embed()` call time, not here.
    ///
    /// ```js
    /// const embedder = EmbeddingModel.transformersJs('Xenova/all-MiniLM-L6-v2');
    /// const vecs = await embedder.embed(['Hello world']);
    /// ```
    #[allow(clippy::needless_pass_by_value)] // wasm_bindgen requires owned args
    #[wasm_bindgen(js_name = "transformersJs")]
    pub fn transformers_js(
        model_id: String,
        options: Option<TransformersJsOptions>,
    ) -> Result<WasmEmbeddingModel, JsValue> {
        let quantized = options.as_ref().is_none_or(|o| o.quantized);
        let dimensions = options.as_ref().map_or(384, |o| o.dimensions);

        // Build a JS function body that lazily imports @huggingface/transformers,
        // creates a feature-extraction pipeline (cached on globalThis), and
        // returns Float32Array[] for each input text.
        //
        // The cache key includes the model id so different models can coexist.
        let cache_key = format!(
            "__blazen_embed_pipeline_{}",
            model_id.replace(['/', '-', '.'], "_")
        );

        // Note: we use named `format!` args rather than inline captures because
        // the `{{` / `}}` escaping for JS object literals conflicts with inline
        // capture syntax and makes the template less readable.
        #[allow(clippy::uninlined_format_args)]
        let handler_body = format!(
            r"
            const {{ pipeline }} = await import('@huggingface/transformers');
            if (!globalThis['{cache_key}']) {{
                globalThis['{cache_key}'] = await pipeline(
                    'feature-extraction',
                    '{model_id}',
                    {{ quantized: {quantized} }}
                );
            }}
            const pipe = globalThis['{cache_key}'];
            const results = [];
            for (const text of texts) {{
                const output = await pipe(text, {{ pooling: 'mean', normalize: true }});
                results.push(new Float32Array(output.data));
            }}
            return results;
            ",
            cache_key = cache_key,
            model_id = model_id,
            quantized = quantized,
        );

        // `new Function("texts", body)` creates `function anonymous(texts) { body }`.
        let handler = js_sys::Function::new_with_args("texts", &handler_body);

        let wrapper =
            JsEmbeddingHandler::new(model_id, dimensions as usize, handler);
        Ok(Self {
            inner: Arc::new(wrapper),
        })
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
