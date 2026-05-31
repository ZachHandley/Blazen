//! Foreign-implementable provider ABCs for the WASM SDK.
//!
//! Mirrors the [`CustomProvider`](super::custom::WasmCustomProvider)
//! `fromJsObject` pattern: each class wraps a JS object whose methods are
//! invoked on dispatch, producing a real `Arc<dyn …Provider>` trait object on
//! the Rust side. The capability traits all extend
//! [`blazen_llm::BaseProvider`], so every adapter carries a
//! [`ProviderMetadata`] describing its provider id + capability kind.
//!
//! Bound types:
//! - [`WasmCapabilityKind`] (`CapabilityKind`) — capability tag enum.
//! - [`WasmProviderMetadata`] (`ProviderMetadata`) — provider id + capability.
//! - [`WasmBaseProviderAbc`] (`BaseProvider`) — the polymorphic root ABC.
//! - [`WasmLLMProvider`] (`LLMProvider`) — chat / completion / streaming.
//! - [`WasmEmbeddingProvider`] (`EmbeddingProvider`) — vector embeddings.
//! - [`WasmImageGenProvider`] (`ImageGenProvider`) — image generation.
//! - [`WasmVcProvider`] (`VcProvider`) — voice conversion.
//!
//! JS usage:
//!
//! ```js
//! class MyLlm extends LLMProvider {
//!   async complete(req) { return await callMyApi(req); }
//! }
//! const provider = LLMProvider.fromJsObject(
//!   new ProviderMetadata("my-llm", CapabilityKind.Llm),
//!   new MyLlm(),
//! );
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use js_sys::{Function, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use blazen_llm::providers::{
    BaseProvider, CapabilityKind, EmbeddingProvider, ImageGenProvider, LLMProvider,
    ProviderMetadata,
};
use blazen_llm::types::{ModelRequest, ModelResponse, StreamChunk};
use blazen_llm::BlazenError;

// ---------------------------------------------------------------------------
// SendFuture — same single-threaded wasm pattern as byo_backend / custom.
// ---------------------------------------------------------------------------

struct SendFuture<F>(F);

// SAFETY: wasm32 is single-threaded.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: structural projection; `F` is never moved out.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// CapabilityKind
// ---------------------------------------------------------------------------

/// Coarse categorization of what a provider does.
///
/// Mirrors [`blazen_llm::CapabilityKind`].
#[wasm_bindgen(js_name = "CapabilityKind")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmCapabilityKind {
    /// Large language model — chat / completion / streaming.
    Llm,
    /// Text-to-speech audio synthesis.
    Tts,
    /// Speech-to-text transcription.
    Stt,
    /// Text-to-music / text-to-sfx audio generation.
    Music,
    /// Voice conversion.
    Vc,
    /// 3D mesh generation.
    ThreeD,
    /// 2D image generation.
    ImageGen,
    /// Vector embedding generation.
    Embedding,
    /// Neural audio codec.
    Codec,
    /// Background removal on existing images.
    BackgroundRemoval,
    /// Video generation.
    Video,
}

impl From<WasmCapabilityKind> for CapabilityKind {
    fn from(value: WasmCapabilityKind) -> Self {
        match value {
            WasmCapabilityKind::Llm => CapabilityKind::Llm,
            WasmCapabilityKind::Tts => CapabilityKind::Tts,
            WasmCapabilityKind::Stt => CapabilityKind::Stt,
            WasmCapabilityKind::Music => CapabilityKind::Music,
            WasmCapabilityKind::Vc => CapabilityKind::Vc,
            WasmCapabilityKind::ThreeD => CapabilityKind::ThreeD,
            WasmCapabilityKind::ImageGen => CapabilityKind::ImageGen,
            WasmCapabilityKind::Embedding => CapabilityKind::Embedding,
            WasmCapabilityKind::Codec => CapabilityKind::Codec,
            WasmCapabilityKind::BackgroundRemoval => CapabilityKind::BackgroundRemoval,
            WasmCapabilityKind::Video => CapabilityKind::Video,
        }
    }
}

impl From<CapabilityKind> for WasmCapabilityKind {
    fn from(value: CapabilityKind) -> Self {
        match value {
            CapabilityKind::Llm => WasmCapabilityKind::Llm,
            CapabilityKind::Tts => WasmCapabilityKind::Tts,
            CapabilityKind::Stt => WasmCapabilityKind::Stt,
            CapabilityKind::Music => WasmCapabilityKind::Music,
            CapabilityKind::Vc => WasmCapabilityKind::Vc,
            CapabilityKind::ThreeD => WasmCapabilityKind::ThreeD,
            CapabilityKind::ImageGen => WasmCapabilityKind::ImageGen,
            CapabilityKind::Embedding => WasmCapabilityKind::Embedding,
            CapabilityKind::Codec => WasmCapabilityKind::Codec,
            CapabilityKind::BackgroundRemoval => WasmCapabilityKind::BackgroundRemoval,
            CapabilityKind::Video => WasmCapabilityKind::Video,
        }
    }
}

// ---------------------------------------------------------------------------
// ProviderMetadata
// ---------------------------------------------------------------------------

/// Static metadata describing a provider instance.
///
/// Mirrors [`blazen_llm::ProviderMetadata`].
#[wasm_bindgen(js_name = "ProviderMetadata")]
#[derive(Debug, Clone)]
pub struct WasmProviderMetadata {
    inner: ProviderMetadata,
}

#[wasm_bindgen(js_class = "ProviderMetadata")]
impl WasmProviderMetadata {
    /// Build a minimal metadata record from a provider id + capability.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(provider_id: String, capability: WasmCapabilityKind) -> Self {
        Self {
            inner: ProviderMetadata::new(provider_id, capability.into()),
        }
    }

    /// The canonical provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.inner.provider_id.clone()
    }

    /// The capability kind this provider serves.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn capability(&self) -> WasmCapabilityKind {
        self.inner.capability.into()
    }

    /// The optional human-readable display name (falls back to the provider
    /// id when unset).
    #[wasm_bindgen(getter, js_name = "displayName")]
    #[must_use]
    pub fn display_name(&self) -> String {
        self.inner.display().to_owned()
    }

    /// The optional version pin, if any.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn version(&self) -> Option<String> {
        self.inner.version.clone()
    }

    /// Attach a human-readable display name (chainable).
    #[wasm_bindgen(js_name = "withDisplayName")]
    pub fn with_display_name(&mut self, name: String) {
        self.inner = self.inner.clone().with_display_name(name);
    }

    /// Attach a version pin (chainable).
    #[wasm_bindgen(js_name = "withVersion")]
    pub fn with_version(&mut self, version: String) {
        self.inner = self.inner.clone().with_version(version);
    }
}

impl WasmProviderMetadata {
    fn inner(&self) -> ProviderMetadata {
        self.inner.clone()
    }
}

// ---------------------------------------------------------------------------
// JS dispatch helpers (shared by every ABC adapter).
// ---------------------------------------------------------------------------

/// `Send + Sync` wrapper around the JS host object + its metadata.
///
/// SAFETY: wasm32 is single-threaded.
struct JsProviderHost {
    obj: JsValue,
    metadata: ProviderMetadata,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for JsProviderHost {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for JsProviderHost {}

impl std::fmt::Debug for JsProviderHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsProviderHost")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl JsProviderHost {
    fn new(obj: JsValue, metadata: ProviderMetadata) -> Result<Self, JsValue> {
        if !obj.is_object() {
            return Err(JsValue::from_str("provider host must be a JS object"));
        }
        Ok(Self { obj, metadata })
    }

    fn lookup(&self, method: &str) -> Option<Function> {
        let key = JsValue::from_str(method);
        if !Reflect::has(&self.obj, &key).unwrap_or(false) {
            return None;
        }
        let candidate = Reflect::get(&self.obj, &key).ok()?;
        candidate.is_function().then(|| candidate.unchecked_into())
    }

    /// Call `method(arg)` on the host, awaiting any returned promise.
    async fn call1(&self, method: &str, arg: JsValue) -> Result<JsValue, BlazenError> {
        let func = self.lookup(method).ok_or_else(|| {
            BlazenError::unsupported(format!("provider host has no method `{method}`"))
        })?;
        let raw = func.call1(&self.obj, &arg).map_err(|e| {
            BlazenError::provider("abc", format!("host method `{method}` threw: {e:?}"))
        })?;
        resolve(method, raw).await
    }
}

/// Await a JS return value if it is a promise.
async fn resolve(method: &str, raw: JsValue) -> Result<JsValue, BlazenError> {
    if raw.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = raw.unchecked_into();
        JsFuture::from(promise).await.map_err(|e| {
            BlazenError::provider("abc", format!("host method `{method}` rejected: {e:?}"))
        })
    } else {
        Ok(raw)
    }
}

fn to_js<T: serde::Serialize>(method: &str, value: &T) -> Result<JsValue, BlazenError> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| BlazenError::Serialization(format!("marshal `{method}` arg: {e}")))
}

fn from_js<T: for<'de> serde::Deserialize<'de>>(
    method: &str,
    value: JsValue,
) -> Result<T, BlazenError> {
    serde_wasm_bindgen::from_value(value)
        .map_err(|e| BlazenError::Serialization(format!("deserialize `{method}` result: {e}")))
}

impl BaseProvider for JsProviderHost {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

// ---------------------------------------------------------------------------
// BaseProvider ABC
// ---------------------------------------------------------------------------

/// The polymorphic root ABC for every Blazen provider.
///
/// Wraps a JS object plus its [`ProviderMetadata`], producing a real
/// `Arc<dyn BaseProvider>`. Capability subclasses
/// ([`WasmLLMProvider`] etc.) extend this.
#[wasm_bindgen(js_name = "BaseProvider")]
pub struct WasmBaseProviderAbc {
    inner: Arc<JsProviderHost>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmBaseProviderAbc {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmBaseProviderAbc {}

#[wasm_bindgen(js_class = "BaseProvider")]
impl WasmBaseProviderAbc {
    /// Wrap a JS object as a `BaseProvider` carrying `metadata`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `obj` is not a JS object.
    #[wasm_bindgen(js_name = "fromJsObject")]
    pub fn from_js_object(
        metadata: &WasmProviderMetadata,
        obj: JsValue,
    ) -> Result<WasmBaseProviderAbc, JsValue> {
        Ok(Self {
            inner: Arc::new(JsProviderHost::new(obj, metadata.inner())?),
        })
    }

    /// The canonical provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.inner.provider_id().to_owned()
    }

    /// The provider metadata.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn metadata(&self) -> WasmProviderMetadata {
        WasmProviderMetadata {
            inner: BaseProvider::metadata(&*self.inner).clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// LLMProvider ABC
// ---------------------------------------------------------------------------

/// LLM provider ABC — chat completion + token streaming.
#[derive(Debug)]
struct LLMAdapterImpl(Arc<JsProviderHost>);

#[async_trait]
impl BaseProvider for LLMAdapterImpl {
    fn metadata(&self) -> &ProviderMetadata {
        BaseProvider::metadata(&*self.0)
    }
}

#[async_trait]
impl LLMProvider for LLMAdapterImpl {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, BlazenError> {
        let host = Arc::clone(&self.0);
        SendFuture(async move {
            let arg = to_js("complete", &request)?;
            let raw = host.call1("complete", arg).await?;
            from_js("complete", raw)
        })
        .await
    }

    async fn stream(
        &self,
        request: ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // No native streaming bridge for the JS ABC — fall back to a single
        // chunk built from `complete()` (same degradation the BYO backend
        // uses when `streamComplete` is absent).
        let response = self.complete(request).await?;
        let chunk = StreamChunk {
            delta: response.content,
            tool_calls: response.tool_calls,
            finish_reason: response.finish_reason,
            reasoning_delta: response.reasoning.map(|r| r.text),
            citations: response.citations,
            artifacts: response.artifacts,
        };
        Ok(Box::pin(futures_util::stream::once(async { Ok(chunk) })))
    }
}

/// Foreign-implementable LLM provider.
#[wasm_bindgen(js_name = "LLMProvider")]
pub struct WasmLLMProvider {
    inner: Arc<dyn LLMProvider>,
    host: Arc<JsProviderHost>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmLLMProvider {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmLLMProvider {}

#[wasm_bindgen(js_class = "LLMProvider")]
impl WasmLLMProvider {
    /// Wrap a JS object exposing an async `complete(request)` method.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `obj` is not a JS object.
    #[wasm_bindgen(js_name = "fromJsObject")]
    pub fn from_js_object(
        metadata: &WasmProviderMetadata,
        obj: JsValue,
    ) -> Result<WasmLLMProvider, JsValue> {
        let host = Arc::new(JsProviderHost::new(obj, metadata.inner())?);
        Ok(Self {
            inner: Arc::new(LLMAdapterImpl(Arc::clone(&host))),
            host,
        })
    }

    /// The canonical provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.host.provider_id().to_owned()
    }

    /// Run a non-streaming completion through the wrapped provider.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JS host rejects or returns a value
    /// that does not deserialize to a `ModelResponse`.
    #[wasm_bindgen]
    pub async fn complete(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: ModelRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = self
            .inner
            .complete(req)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProvider ABC
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct EmbeddingAdapterImpl(Arc<JsProviderHost>);

#[async_trait]
impl BaseProvider for EmbeddingAdapterImpl {
    fn metadata(&self) -> &ProviderMetadata {
        BaseProvider::metadata(&*self.0)
    }
}

#[async_trait]
impl EmbeddingProvider for EmbeddingAdapterImpl {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BlazenError> {
        let host = Arc::clone(&self.0);
        SendFuture(async move {
            let arg = to_js("embed", &texts)?;
            let raw = host.call1("embed", arg).await?;
            from_js("embed", raw)
        })
        .await
    }

    fn dimensions(&self) -> usize {
        // The host advertises dimensionality via a `dimensions` field on the
        // metadata version pin convention is not used here; read the optional
        // `dimensions` property synchronously off the JS object.
        self.0
            .lookup("dimensions")
            .and_then(|f| f.call0(&self.0.obj).ok())
            .and_then(|v| v.as_f64())
            .map_or(0, |n| n as usize)
    }
}

/// Foreign-implementable embedding provider.
#[wasm_bindgen(js_name = "EmbeddingProvider")]
pub struct WasmEmbeddingProvider {
    inner: Arc<dyn EmbeddingProvider>,
    host: Arc<JsProviderHost>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmEmbeddingProvider {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmEmbeddingProvider {}

#[wasm_bindgen(js_class = "EmbeddingProvider")]
impl WasmEmbeddingProvider {
    /// Wrap a JS object exposing an async `embed(texts)` method and an
    /// optional `dimensions()` method.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `obj` is not a JS object.
    #[wasm_bindgen(js_name = "fromJsObject")]
    pub fn from_js_object(
        metadata: &WasmProviderMetadata,
        obj: JsValue,
    ) -> Result<WasmEmbeddingProvider, JsValue> {
        let host = Arc::new(JsProviderHost::new(obj, metadata.inner())?);
        Ok(Self {
            inner: Arc::new(EmbeddingAdapterImpl(Arc::clone(&host))),
            host,
        })
    }

    /// The canonical provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.host.provider_id().to_owned()
    }

    /// The embedding vector dimensionality.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn dimensions(&self) -> u32 {
        u32::try_from(self.inner.dimensions()).unwrap_or(0)
    }

    /// Embed a batch of texts, returning one `number[]` per input.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JS host rejects or returns a value
    /// that does not deserialize to `number[][]`.
    #[wasm_bindgen]
    pub async fn embed(&self, texts: Vec<String>) -> Result<JsValue, JsValue> {
        let result = self
            .inner
            .embed(texts)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// ImageGenProvider ABC
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ImageGenAdapterImpl(Arc<JsProviderHost>);

#[async_trait]
impl BaseProvider for ImageGenAdapterImpl {
    fn metadata(&self) -> &ProviderMetadata {
        BaseProvider::metadata(&*self.0)
    }
}

#[async_trait]
impl ImageGenProvider for ImageGenAdapterImpl {
    async fn generate_image(
        &self,
        request: blazen_llm::compute::requests::ImageRequest,
    ) -> Result<blazen_llm::compute::results::ImageResult, BlazenError> {
        let host = Arc::clone(&self.0);
        SendFuture(async move {
            let arg = to_js("generateImage", &request)?;
            let raw = host.call1("generateImage", arg).await?;
            from_js("generateImage", raw)
        })
        .await
    }

    async fn upscale_image(
        &self,
        request: blazen_llm::compute::requests::UpscaleRequest,
    ) -> Result<blazen_llm::compute::results::ImageResult, BlazenError> {
        let host = Arc::clone(&self.0);
        SendFuture(async move {
            let arg = to_js("upscaleImage", &request)?;
            let raw = host.call1("upscaleImage", arg).await?;
            from_js("upscaleImage", raw)
        })
        .await
    }
}

/// Foreign-implementable image-generation provider.
#[wasm_bindgen(js_name = "ImageGenProvider")]
pub struct WasmImageGenProvider {
    inner: Arc<dyn ImageGenProvider>,
    host: Arc<JsProviderHost>,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmImageGenProvider {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmImageGenProvider {}

#[wasm_bindgen(js_class = "ImageGenProvider")]
impl WasmImageGenProvider {
    /// Wrap a JS object exposing an async `generateImage(request)` method
    /// (and optionally `upscaleImage(request)`).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `obj` is not a JS object.
    #[wasm_bindgen(js_name = "fromJsObject")]
    pub fn from_js_object(
        metadata: &WasmProviderMetadata,
        obj: JsValue,
    ) -> Result<WasmImageGenProvider, JsValue> {
        let host = Arc::new(JsProviderHost::new(obj, metadata.inner())?);
        Ok(Self {
            inner: Arc::new(ImageGenAdapterImpl(Arc::clone(&host))),
            host,
        })
    }

    /// The canonical provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.host.provider_id().to_owned()
    }

    /// Generate an image from the given request.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JS host rejects or returns a value
    /// that does not deserialize to an `ImageResult`.
    #[wasm_bindgen(js_name = "generateImage")]
    pub async fn generate_image(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = self
            .inner
            .generate_image(req)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Upscale an existing image.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the JS host rejects or returns a value
    /// that does not deserialize to an `ImageResult`.
    #[wasm_bindgen(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = self
            .inner
            .upscale_image(req)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// NOTE: `VcProvider` is intentionally NOT re-bound here — the WASM SDK
// already exposes a JS-implementable `VcProvider` class in
// `crate::capability_providers`. Binding it a second time would collide on
// the `VcProvider` wasm-bindgen export name.
