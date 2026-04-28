//! Subclassable abstract bases for traits that JS users may extend.
//!
//! Each class follows the
//! [`crate::types::memory::JsMemoryBackend`] pattern: a constructor that
//! does no work and stub methods that throw "subclass must override".
//! JavaScript subclasses extend the class and override the methods to
//! plug their own implementation into Blazen.
//!
//! These bases exist so the TypeScript surface advertises the expected
//! shape (method names, argument types, return types) for user-defined
//! providers, separately from the concrete provider classes such as
//! [`crate::providers::JsCompletionModel`].

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::completion_request::JsStructuredResponse;
use super::message::JsChatMessage;
use crate::generated::JsToolDefinition;

// ---------------------------------------------------------------------------
// JsTool
// ---------------------------------------------------------------------------

/// Base class for custom tools.
///
/// Extend and override `definition()` and `execute(args)` to implement a
/// tool that the agent loop can invoke. For most use cases the
/// [`crate::providers::JsTypedTool`] class is more ergonomic; this base
/// exists for callers who prefer the subclass-style API.
///
/// ```javascript
/// class MyTool extends Tool {
///   definition() {
///     return {
///       name: "myTool",
///       description: "Do a thing",
///       parameters: { type: "object", properties: {} },
///     };
///   }
///   async execute(args) { return { ok: true }; }
/// }
/// ```
#[napi(js_name = "Tool")]
pub struct JsTool {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsTool {
    /// Create a new tool base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Return the tool's JSON Schema definition. Subclasses **must**
    /// override this method.
    #[napi]
    pub fn definition(&self) -> Result<JsToolDefinition> {
        Err(napi::Error::from_reason(
            "subclass must override definition()",
        ))
    }

    /// Execute the tool. Subclasses **must** override this method.
    #[napi]
    pub async fn execute(&self, _arguments: serde_json::Value) -> Result<serde_json::Value> {
        Err(napi::Error::from_reason("subclass must override execute()"))
    }
}

// ---------------------------------------------------------------------------
// JsLocalModel
// ---------------------------------------------------------------------------

/// Base class for in-process model providers that load weights into
/// memory / VRAM.
///
/// Mirrors [`blazen_llm::traits::LocalModel`]. Subclasses must override
/// `load`, `unload`, and `isLoaded` (and may optionally override
/// `vramBytes`).
///
/// ```javascript
/// class MyLocalModel extends LocalModel {
///   async load() { /* ... */ }
///   async unload() { /* ... */ }
///   async isLoaded() { return false; }
/// }
/// ```
#[napi(js_name = "LocalModel")]
pub struct JsLocalModel {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async
)]
impl JsLocalModel {
    /// Create a new local model base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Load the model into memory / VRAM. Subclasses **must** override.
    #[napi]
    pub async fn load(&self) -> Result<()> {
        Err(napi::Error::from_reason("subclass must override load()"))
    }

    /// Drop the loaded model and free its memory / VRAM. Subclasses
    /// **must** override.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        Err(napi::Error::from_reason("subclass must override unload()"))
    }

    /// Whether the model is currently loaded. Subclasses **must** override.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> Result<bool> {
        Err(napi::Error::from_reason(
            "subclass must override isLoaded()",
        ))
    }

    /// Approximate memory footprint in bytes. Default implementation
    /// returns `null`.
    #[napi(js_name = "vramBytes")]
    pub async fn vram_bytes(&self) -> Option<i64> {
        None
    }
}

// ---------------------------------------------------------------------------
// JsModelRegistry
// ---------------------------------------------------------------------------

/// Base class for providers that can list their available models.
///
/// Mirrors [`blazen_llm::traits::ModelRegistry`]. Subclasses override
/// `listModels()` and `getModel(modelId)` to expose their catalog to
/// Blazen's discovery code.
///
/// ```javascript
/// class MyRegistry extends ModelRegistry {
///   async listModels() { return [{ id: "m1", provider: "x", capabilities: {} }]; }
///   async getModel(id) { return id === "m1" ? { id, provider: "x", capabilities: {} } : null; }
/// }
/// ```
#[napi(js_name = "ModelRegistry")]
pub struct JsModelRegistry {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsModelRegistry {
    /// Create a new model registry base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// List all models available from this provider. Subclasses **must**
    /// override.
    ///
    /// Each entry should be a plain object matching the `ModelInfo`
    /// shape (see [`crate::types::provider_info::JsModelInfo`]).
    #[napi(js_name = "listModels")]
    pub async fn list_models(&self) -> Result<Vec<serde_json::Value>> {
        Err(napi::Error::from_reason(
            "subclass must override listModels()",
        ))
    }

    /// Look up a specific model by its identifier. Subclasses **must**
    /// override.
    #[napi(js_name = "getModel")]
    pub async fn get_model(&self, _model_id: String) -> Result<Option<serde_json::Value>> {
        Err(napi::Error::from_reason(
            "subclass must override getModel()",
        ))
    }
}

// ---------------------------------------------------------------------------
// JsStructuredOutput
// ---------------------------------------------------------------------------

/// Base class for the structured-output extraction surface.
///
/// Mirrors [`blazen_llm::traits::StructuredOutput`]. Most callers should
/// use [`crate::providers::JsCompletionModel`]'s built-in structured
/// output (every completion model supports it via the blanket impl);
/// this class exists so users can write a custom `extract` that does
/// something different (e.g. multi-pass extraction, retries, custom
/// schema injection).
#[napi(js_name = "StructuredOutput")]
pub struct JsStructuredOutput {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsStructuredOutput {
    /// Create a new structured output base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Extract structured data from the given messages, constrained by
    /// `schema` (a JSON Schema object). Subclasses **must** override.
    #[napi]
    pub async fn extract(
        &self,
        _messages: Vec<&JsChatMessage>,
        _schema: serde_json::Value,
    ) -> Result<JsStructuredResponse> {
        Err(napi::Error::from_reason("subclass must override extract()"))
    }
}

// ---------------------------------------------------------------------------
// JsHostDispatch
// ---------------------------------------------------------------------------

/// Base class for user-extendable host dispatchers.
///
/// Mirrors [`blazen_llm::HostDispatch`]. Subclasses override `call(method,
/// request)` to plug a JS-side capability table into a Blazen
/// [`CustomProvider`](crate::providers::JsCustomProvider). Most users
/// will reach for [`crate::providers::JsCustomProvider`] directly with a
/// plain JS object; this class exists so users can subclass with shared
/// state and override `hasMethod` independently from `call`.
#[napi(js_name = "HostDispatch")]
pub struct JsHostDispatch {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsHostDispatch {
    /// Create a new host dispatch base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Dispatch a Rust-side capability call to the JavaScript host.
    /// Subclasses **must** override.
    #[napi]
    pub async fn call(
        &self,
        _method: String,
        _request: serde_json::Value,
    ) -> Result<serde_json::Value> {
        Err(napi::Error::from_reason("subclass must override call()"))
    }

    /// Whether this dispatcher implements `method`. Subclasses **must**
    /// override.
    #[napi(js_name = "hasMethod")]
    pub fn has_method(&self, _method: String) -> Result<bool> {
        Err(napi::Error::from_reason(
            "subclass must override hasMethod()",
        ))
    }
}

// ---------------------------------------------------------------------------
// JsImageModel
// ---------------------------------------------------------------------------

/// Base class for custom image-generation providers.
///
/// Mirrors [`blazen_llm::traits::ImageModel`] (and the underlying
/// [`blazen_llm::compute::traits::ImageGeneration`] capability). Most
/// callers route image generation through one of the concrete provider
/// classes ([`crate::providers::JsFalProvider`],
/// [`crate::providers::JsImageProvider`], etc.); this base exists so
/// users can subclass when they need to plug a JS-side image backend
/// into Blazen's compute surface.
///
/// Subclasses **must** override `generateImage(request)` and
/// `upscaleImage(request)`. Requests and results are exchanged as
/// JSON objects matching the
/// [`blazen_llm::types::ImageRequest`](blazen_llm::types::ImageRequest) /
/// [`blazen_llm::types::ImageResult`](blazen_llm::types::ImageResult)
/// shapes.
///
/// ```javascript
/// class MyImageBackend extends ImageModel {
///   async generateImage(request) { /* ... */ }
///   async upscaleImage(request) { /* ... */ }
/// }
/// ```
#[napi(js_name = "ImageModel")]
pub struct JsImageModel {}

#[napi]
#[allow(
    clippy::new_without_default,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::unused_async,
    clippy::needless_pass_by_value
)]
impl JsImageModel {
    /// Create a new image model base instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Generate one or more images from a text prompt. Subclasses
    /// **must** override this method.
    #[napi(js_name = "generateImage")]
    pub async fn generate_image(&self, _request: serde_json::Value) -> Result<serde_json::Value> {
        Err(napi::Error::from_reason(
            "subclass must override generateImage()",
        ))
    }

    /// Upscale an existing image. Subclasses **must** override this
    /// method.
    #[napi(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, _request: serde_json::Value) -> Result<serde_json::Value> {
        Err(napi::Error::from_reason(
            "subclass must override upscaleImage()",
        ))
    }
}
