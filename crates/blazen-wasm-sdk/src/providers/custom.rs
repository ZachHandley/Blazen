//! `wasm-bindgen` wrapper for the [`blazen_llm::CustomProvider`] trait and
//! [`blazen_llm::CustomProviderHandle`] concrete wrapper.
//!
//! Composes [`WasmBaseProvider`] (since WASM has no class-extends semantic,
//! the base sits as an internal field rather than as a Rust supertype) and
//! exposes four canonical factories plus the 13 typed compute methods.
//!
//! ## Factories
//!
//! - [`WasmCustomProvider::ollama`] — convenience wrapper around the
//!   `OpenAI`-compatible Ollama endpoint.
//! - [`WasmCustomProvider::lm_studio`] — convenience wrapper around LM Studio.
//! - [`WasmCustomProvider::openai_compat`] — accepts a
//!   [`WasmOpenAiCompatConfig`] for arbitrary `OpenAI`-compatible servers.
//! - [`WasmCustomProvider::from_js_object`] — takes a JS object/class
//!   instance whose own methods (or those on its prototype chain) implement
//!   any subset of the 16 typed capability methods on the Rust
//!   [`blazen_llm::CustomProvider`] trait. Each Rust call introspects the JS
//!   object via [`js_sys::Reflect`] and either dispatches to the matching JS
//!   method (awaiting any returned `Promise`) or returns
//!   [`blazen_llm::BlazenError::Unsupported`] when the method is absent.
//!
//! ## Subclassing strategy
//!
//! `wasm-bindgen`-generated classes cannot be `extends`-ed cleanly from JS,
//! so we ship a `class CustomProvider` in the JS shim that users extend.
//! Once the user has an instance, they hand it to
//! [`WasmCustomProvider::from_js_object`] (exposed as `CustomProvider.fromJsObject`
//! on the JS side) which wraps the instance into a `WasmCustomProvider`
//! powered by [`WasmCustomProviderAdapter`]. The adapter inspects the JS
//! object's prototype chain for typed methods (`textToSpeech`, `complete`,
//! `embed`, etc.) on every call, falling back to `Unsupported` when a
//! method is not defined.
//!
//! ## Compute methods
//!
//! All 13 capability methods accept a typed request (serde-decoded from
//! the JS object) and return a typed response. They are `Promise`-returning
//! on the JS side because the underlying `async_trait` impl drives an
//! async HTTP or JS-dispatch call.

use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use js_sys::Reflect;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use blazen_llm::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use blazen_llm::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ImageGeneration, ThreeDGeneration, Transcription,
    VideoGeneration, VoiceCloning,
};
use blazen_llm::error::BlazenError;
use blazen_llm::providers::custom::{self as core_custom, CustomProvider, CustomProviderHandle};
use blazen_llm::traits::CompletionModel;
use blazen_llm::types::{CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk};

use super::base::WasmBaseProvider;
use super::defaults::WasmCompletionProviderDefaults;
use super::openai_compat::WasmOpenAiCompatConfig;

// ---------------------------------------------------------------------------
// SendFuture — wraps a non-Send future so it satisfies the `Send` bound the
// async_trait machinery generates on trait methods.
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
///
/// SAFETY: wasm32 is single-threaded; no other thread can observe the
/// wrapped future. The wrapper exists purely to satisfy the
/// `Pin<Box<dyn Future + Send>>` return type produced by `#[async_trait]`.
struct SendFuture<F>(F);

// SAFETY: see SendFuture docs.
unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: we are not moving `F`, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// WasmCustomProviderAdapter — implements `CustomProvider` by dispatching to
// matching JS methods on a held `js_sys::Object`.
// ---------------------------------------------------------------------------

/// Adapter that implements [`CustomProvider`] by dispatching every typed
/// method to a JS-side method of the same (camelCased) name on a held
/// `js_sys::Object`.
///
/// The wrapped object is typically a JS class instance whose class
/// `extends CustomProvider` on the JS side; the adapter inspects the
/// prototype chain via [`Reflect::has`] on each call. Missing methods cause
/// the corresponding Rust method to return [`BlazenError::Unsupported`].
///
/// JS methods may be `async` (returning a `Promise`) or synchronous. Promises
/// are awaited via [`JsFuture`] before the resolved value is deserialized
/// back to a typed Rust response.
///
/// SAFETY: wasm32 is single-threaded so the `unsafe impl Send + Sync` is
/// vacuously safe — no other thread can observe the wrapped JS object.
pub(crate) struct WasmCustomProviderAdapter {
    provider_id: String,
    obj: js_sys::Object,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmCustomProviderAdapter {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmCustomProviderAdapter {}

impl WasmCustomProviderAdapter {
    fn new(provider_id: String, obj: js_sys::Object) -> Self {
        Self { provider_id, obj }
    }

    /// Look up a function on the wrapped JS object by name. Returns `None`
    /// when the property is absent or not a function. Walks the prototype
    /// chain implicitly via JS `[[Get]]`.
    fn lookup_function(&self, method: &str) -> Option<js_sys::Function> {
        let key = JsValue::from_str(method);
        // `Reflect::has` includes inherited properties (prototype chain), so
        // class-method overrides are visible here.
        if !Reflect::has(&self.obj, &key).unwrap_or(false) {
            return None;
        }
        let candidate = Reflect::get(&self.obj, &key).ok()?;
        if candidate.is_function() {
            Some(candidate.unchecked_into::<js_sys::Function>())
        } else {
            None
        }
    }

    /// Returns `true` when the JS object exposes a callable method named
    /// `method`. Used by trait methods to short-circuit to `Unsupported`
    /// before performing any serde marshaling.
    fn has(&self, method: &str) -> bool {
        self.lookup_function(method).is_some()
    }

    /// Call `method` on the JS object with a single argument `arg`, awaiting
    /// any returned `Promise`. Returns the resolved [`JsValue`].
    async fn call_js(&self, method: &str, arg: JsValue) -> Result<JsValue, BlazenError> {
        let func = self.lookup_function(method).ok_or_else(|| {
            BlazenError::unsupported(format!(
                "CustomProvider host has no method `{method}` (or it is not a function)"
            ))
        })?;
        let raw = func.call1(&self.obj, &arg).map_err(|e| {
            BlazenError::provider("custom", format!("host method `{method}` threw: {e:?}"))
        })?;
        if raw.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = raw.unchecked_into();
            JsFuture::from(promise).await.map_err(|e| {
                BlazenError::provider("custom", format!("host method `{method}` rejected: {e:?}"))
            })
        } else {
            Ok(raw)
        }
    }

    /// Call `method` on the JS object with no arguments, awaiting any
    /// returned `Promise`. Returns the resolved [`JsValue`].
    async fn call_js_noarg(&self, method: &str) -> Result<JsValue, BlazenError> {
        let func = self.lookup_function(method).ok_or_else(|| {
            BlazenError::unsupported(format!(
                "CustomProvider host has no method `{method}` (or it is not a function)"
            ))
        })?;
        let raw = func.call0(&self.obj).map_err(|e| {
            BlazenError::provider("custom", format!("host method `{method}` threw: {e:?}"))
        })?;
        if raw.has_type::<js_sys::Promise>() {
            let promise: js_sys::Promise = raw.unchecked_into();
            JsFuture::from(promise).await.map_err(|e| {
                BlazenError::provider("custom", format!("host method `{method}` rejected: {e:?}"))
            })
        } else {
            Ok(raw)
        }
    }

    /// Marshal a serde-serializable value into a `JsValue` for the JS-side
    /// call argument.
    fn to_js<T: serde::Serialize>(method: &str, value: &T) -> Result<JsValue, BlazenError> {
        serde_wasm_bindgen::to_value(value).map_err(|e| {
            BlazenError::Serialization(format!(
                "failed to marshal request for `{method}` to JS: {e}"
            ))
        })
    }

    /// Deserialize the JS-side return value into the typed Rust response.
    fn from_js<T: for<'de> serde::Deserialize<'de>>(
        method: &str,
        value: JsValue,
    ) -> Result<T, BlazenError> {
        serde_wasm_bindgen::from_value(value).map_err(|e| {
            BlazenError::Serialization(format!(
                "failed to deserialize host result of `{method}`: {e}"
            ))
        })
    }
}

#[async_trait]
impl CustomProvider for WasmCustomProviderAdapter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        if !self.has("complete") {
            return Err(BlazenError::unsupported(
                "CustomProvider::complete not implemented",
            ));
        }
        let arg = Self::to_js("complete", &request)?;
        let result = SendFuture(self.call_js("complete", arg)).await?;
        Self::from_js("complete", result)
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        // Streaming through a JS callback is non-trivial — the host JS
        // method has no natural way to yield chunks across the FFI boundary
        // without a callback parameter, which the typed trait does not
        // expose. Wasm SDK users who need streaming should subclass the JS
        // `CustomProvider`, override `complete` only, and let the framework
        // synthesize a single-chunk stream. We surface that as
        // `Unsupported` so callers get a clear error.
        Err(BlazenError::unsupported(
            "CustomProvider::stream not implemented (JS host objects cannot yield chunks; override `complete` instead)",
        ))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, BlazenError> {
        if !self.has("embed") {
            return Err(BlazenError::unsupported(
                "CustomProvider::embed not implemented",
            ));
        }
        let arg = Self::to_js("embed", &texts)?;
        let result = SendFuture(self.call_js("embed", arg)).await?;
        Self::from_js("embed", result)
    }

    async fn text_to_speech(&self, req: SpeechRequest) -> Result<AudioResult, BlazenError> {
        if !self.has("textToSpeech") {
            return Err(BlazenError::unsupported(
                "CustomProvider::text_to_speech not implemented",
            ));
        }
        let arg = Self::to_js("textToSpeech", &req)?;
        let result = SendFuture(self.call_js("textToSpeech", arg)).await?;
        Self::from_js("textToSpeech", result)
    }

    async fn generate_music(&self, req: MusicRequest) -> Result<AudioResult, BlazenError> {
        if !self.has("generateMusic") {
            return Err(BlazenError::unsupported(
                "CustomProvider::generate_music not implemented",
            ));
        }
        let arg = Self::to_js("generateMusic", &req)?;
        let result = SendFuture(self.call_js("generateMusic", arg)).await?;
        Self::from_js("generateMusic", result)
    }

    async fn generate_sfx(&self, req: MusicRequest) -> Result<AudioResult, BlazenError> {
        if !self.has("generateSfx") {
            return Err(BlazenError::unsupported(
                "CustomProvider::generate_sfx not implemented",
            ));
        }
        let arg = Self::to_js("generateSfx", &req)?;
        let result = SendFuture(self.call_js("generateSfx", arg)).await?;
        Self::from_js("generateSfx", result)
    }

    async fn clone_voice(&self, req: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        if !self.has("cloneVoice") {
            return Err(BlazenError::unsupported(
                "CustomProvider::clone_voice not implemented",
            ));
        }
        let arg = Self::to_js("cloneVoice", &req)?;
        let result = SendFuture(self.call_js("cloneVoice", arg)).await?;
        Self::from_js("cloneVoice", result)
    }

    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        if !self.has("listVoices") {
            return Err(BlazenError::unsupported(
                "CustomProvider::list_voices not implemented",
            ));
        }
        let result = SendFuture(self.call_js_noarg("listVoices")).await?;
        Self::from_js("listVoices", result)
    }

    async fn delete_voice(&self, voice: VoiceHandle) -> Result<(), BlazenError> {
        if !self.has("deleteVoice") {
            return Err(BlazenError::unsupported(
                "CustomProvider::delete_voice not implemented",
            ));
        }
        let arg = Self::to_js("deleteVoice", &voice)?;
        let _ = SendFuture(self.call_js("deleteVoice", arg)).await?;
        Ok(())
    }

    async fn generate_image(&self, req: ImageRequest) -> Result<ImageResult, BlazenError> {
        if !self.has("generateImage") {
            return Err(BlazenError::unsupported(
                "CustomProvider::generate_image not implemented",
            ));
        }
        let arg = Self::to_js("generateImage", &req)?;
        let result = SendFuture(self.call_js("generateImage", arg)).await?;
        Self::from_js("generateImage", result)
    }

    async fn upscale_image(&self, req: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        if !self.has("upscaleImage") {
            return Err(BlazenError::unsupported(
                "CustomProvider::upscale_image not implemented",
            ));
        }
        let arg = Self::to_js("upscaleImage", &req)?;
        let result = SendFuture(self.call_js("upscaleImage", arg)).await?;
        Self::from_js("upscaleImage", result)
    }

    async fn text_to_video(&self, req: VideoRequest) -> Result<VideoResult, BlazenError> {
        if !self.has("textToVideo") {
            return Err(BlazenError::unsupported(
                "CustomProvider::text_to_video not implemented",
            ));
        }
        let arg = Self::to_js("textToVideo", &req)?;
        let result = SendFuture(self.call_js("textToVideo", arg)).await?;
        Self::from_js("textToVideo", result)
    }

    async fn image_to_video(&self, req: VideoRequest) -> Result<VideoResult, BlazenError> {
        if !self.has("imageToVideo") {
            return Err(BlazenError::unsupported(
                "CustomProvider::image_to_video not implemented",
            ));
        }
        let arg = Self::to_js("imageToVideo", &req)?;
        let result = SendFuture(self.call_js("imageToVideo", arg)).await?;
        Self::from_js("imageToVideo", result)
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        if !self.has("transcribe") {
            return Err(BlazenError::unsupported(
                "CustomProvider::transcribe not implemented",
            ));
        }
        let arg = Self::to_js("transcribe", &req)?;
        let result = SendFuture(self.call_js("transcribe", arg)).await?;
        Self::from_js("transcribe", result)
    }

    async fn generate_3d(&self, req: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        if !self.has("generate3d") {
            return Err(BlazenError::unsupported(
                "CustomProvider::generate_3d not implemented",
            ));
        }
        let arg = Self::to_js("generate3d", &req)?;
        let result = SendFuture(self.call_js("generate3d", arg)).await?;
        Self::from_js("generate3d", result)
    }

    async fn remove_background(
        &self,
        req: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        if !self.has("removeBackground") {
            return Err(BlazenError::unsupported(
                "CustomProvider::remove_background not implemented",
            ));
        }
        let arg = Self::to_js("removeBackground", &req)?;
        let result = SendFuture(self.call_js("removeBackground", arg)).await?;
        Self::from_js("removeBackground", result)
    }
}

// ---------------------------------------------------------------------------
// WasmCustomProvider
// ---------------------------------------------------------------------------

/// User-extensible Blazen provider exposed to JS as `CustomProvider`.
///
/// Composes a [`WasmBaseProvider`] (since WASM has no class-extends, the
/// base lives as an internal field) and wraps a
/// [`blazen_llm::CustomProviderHandle`]. Use the four static factories below
/// to construct one:
///
/// - `CustomProvider.ollama(model, host?, port?)`
/// - `CustomProvider.lmStudio(model, host?, port?)`
/// - `CustomProvider.openaiCompat(providerId, config)` (config is a
///   `WasmOpenAiCompatConfig`)
/// - `CustomProvider.fromJsObject(providerId, jsInstance)` — wraps a JS
///   class instance (or any object) whose own/prototype methods implement
///   any subset of the typed capability methods. Missing methods surface as
///   `Unsupported` errors.
#[wasm_bindgen(js_name = "CustomProvider")]
pub struct WasmCustomProvider {
    base: WasmBaseProvider,
    inner: Rc<CustomProviderHandle>,
    provider_id: String,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmCustomProvider {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmCustomProvider {}

impl Clone for WasmCustomProvider {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            inner: Rc::clone(&self.inner),
            provider_id: self.provider_id.clone(),
        }
    }
}

impl WasmCustomProvider {
    /// Build a fresh `WasmCustomProvider` from a fully-constructed
    /// [`CustomProviderHandle`], wiring it into a new [`WasmBaseProvider`]
    /// handle so that [`WasmBaseProvider::extract`] works on the composed
    /// `base` field.
    fn from_handle(provider_id: String, model_id: String, handle: CustomProviderHandle) -> Self {
        let inner_rc = Rc::new(handle);
        // `CustomProviderHandle` implements `CompletionModel`. Clone it
        // (cheap — internal Arcs) and box it as a trait object so the base
        // provider can drive `extract()` and any other CompletionModel-only
        // entry points.
        let completion: Arc<dyn CompletionModel> = Arc::new((*inner_rc).clone());
        let base = WasmBaseProvider::new_with_inner(
            WasmCompletionProviderDefaults::default(),
            model_id,
            provider_id.clone(),
            completion,
        );
        Self {
            base,
            inner: inner_rc,
            provider_id,
        }
    }
}

#[wasm_bindgen(js_class = "CustomProvider")]
impl WasmCustomProvider {
    /// Convenience factory for a local or networked Ollama server.
    ///
    /// `host` defaults to `"localhost"` and `port` defaults to `11434`.
    /// `model` is required and becomes the default completion model.
    #[wasm_bindgen]
    #[must_use]
    pub fn ollama(model: String, host: Option<String>, port: Option<u16>) -> WasmCustomProvider {
        let host_str = host.unwrap_or_else(|| "localhost".to_owned());
        let port_num = port.unwrap_or(11434);
        let handle = core_custom::ollama(host_str, port_num, model.clone());
        Self::from_handle("ollama".to_owned(), model, handle)
    }

    /// Convenience factory for an LM Studio server.
    ///
    /// `host` defaults to `"localhost"` and `port` defaults to `1234`.
    /// `model` is required and becomes the default completion model.
    #[wasm_bindgen(js_name = "lmStudio")]
    #[must_use]
    pub fn lm_studio(model: String, host: Option<String>, port: Option<u16>) -> WasmCustomProvider {
        let host_str = host.unwrap_or_else(|| "localhost".to_owned());
        let port_num = port.unwrap_or(1234);
        let handle = core_custom::lm_studio(host_str, port_num, model.clone());
        Self::from_handle("lm_studio".to_owned(), model, handle)
    }

    /// Build a provider that speaks the `OpenAI` Chat Completions wire
    /// format. `config` is a [`WasmOpenAiCompatConfig`] (the same value
    /// passed to [`super::openai_compat::WasmOpenAiCompatProvider::new`]).
    #[wasm_bindgen(js_name = "openaiCompat")]
    #[must_use]
    pub fn openai_compat(
        provider_id: String,
        config: &WasmOpenAiCompatConfig,
    ) -> WasmCustomProvider {
        let cfg = config.inner_config();
        let model_id = cfg.default_model.clone();
        let handle = core_custom::openai_compat(provider_id.clone(), cfg);
        Self::from_handle(provider_id, model_id, handle)
    }

    /// Wrap a JS class instance (or any object) into a `CustomProvider`.
    ///
    /// On every typed call the adapter inspects the JS object's prototype
    /// chain via `Reflect.has` for a method of the matching camelCased name
    /// (`textToSpeech`, `generateImage`, `cloneVoice`, …). Present methods
    /// are invoked with the request as a single JS object argument; absent
    /// methods cause the Rust call to return `Unsupported`. JS methods may
    /// be `async` (`Promise`-returning) or synchronous.
    ///
    /// JS-side usage:
    ///
    /// ```js
    /// class MyTtsProvider extends CustomProvider {
    ///   async textToSpeech(req) { return await myTtsImpl(req); }
    /// }
    /// const provider = CustomProvider.fromJsObject('my-tts', new MyTtsProvider());
    /// ```
    ///
    /// The `providerId` argument is exposed via `provider.providerId` and
    /// used for logs and routing.
    #[wasm_bindgen(js_name = "fromJsObject")]
    #[must_use]
    pub fn from_js_object(provider_id: String, js_instance: js_sys::Object) -> WasmCustomProvider {
        let adapter: Arc<dyn CustomProvider> = Arc::new(WasmCustomProviderAdapter::new(
            provider_id.clone(),
            js_instance,
        ));
        let handle = CustomProviderHandle::new(adapter);
        // Use provider_id as a sensible default model_id — the JS side has
        // no reliable model identifier for arbitrary user-extended objects.
        Self::from_handle(provider_id.clone(), provider_id, handle)
    }

    /// Return the embedded [`WasmBaseProvider`] handle.
    ///
    /// Use this to configure provider-wide defaults (system prompt,
    /// tools, response format, `before_request`/`before_completion`
    /// hooks) and to invoke [`WasmBaseProvider::extract`].
    #[wasm_bindgen(getter, js_name = "base")]
    #[must_use]
    pub fn base(&self) -> WasmBaseProvider {
        self.base.clone()
    }

    /// The provider id supplied at construction time.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    // -----------------------------------------------------------------
    // Audio generation
    // -----------------------------------------------------------------

    /// Synthesize speech from text. Dispatches to the inner provider's
    /// `text_to_speech` implementation — for `fromJsObject`-built providers
    /// this hits the JS `textToSpeech` method; for HTTP-only
    /// `openaiCompat`/`ollama` providers it returns `Unsupported` (speech is
    /// not part of the `OpenAI` Chat Completions protocol).
    #[wasm_bindgen(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: SpeechRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid SpeechRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { AudioGeneration::text_to_speech(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Generate background music. See [`Self::text_to_speech`] for the
    /// dispatch semantics.
    #[wasm_bindgen(js_name = "generateMusic")]
    pub async fn generate_music(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: MusicRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid MusicRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { AudioGeneration::generate_music(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Generate sound effects. See [`Self::text_to_speech`] for the
    /// dispatch semantics.
    #[wasm_bindgen(js_name = "generateSfx")]
    pub async fn generate_sfx(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: MusicRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid MusicRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { AudioGeneration::generate_sfx(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // -----------------------------------------------------------------
    // Voice cloning
    // -----------------------------------------------------------------

    /// Clone a voice from reference audio. Returns a `VoiceHandle`
    /// usable as the `voice` field on subsequent
    /// [`Self::text_to_speech`] requests.
    #[wasm_bindgen(js_name = "cloneVoice")]
    pub async fn clone_voice(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: VoiceCloneRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid VoiceCloneRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { VoiceCloning::clone_voice(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// List all voices known to the provider.
    #[wasm_bindgen(js_name = "listVoices")]
    pub async fn list_voices(&self) -> Result<JsValue, JsValue> {
        let inner = Rc::clone(&self.inner);
        let result = SendFuture(async move { VoiceCloning::list_voices(inner.as_ref()).await })
            .await
            .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Delete a previously cloned voice. Resolves to `true` on success.
    #[wasm_bindgen(js_name = "deleteVoice")]
    pub async fn delete_voice(&self, voice: JsValue) -> Result<bool, JsValue> {
        let handle: VoiceHandle = serde_wasm_bindgen::from_value(voice)
            .map_err(|e| JsValue::from_str(&format!("invalid VoiceHandle: {e}")))?;
        let inner = Rc::clone(&self.inner);
        SendFuture(async move { VoiceCloning::delete_voice(inner.as_ref(), &handle).await })
            .await
            .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        Ok(true)
    }

    // -----------------------------------------------------------------
    // Image generation
    // -----------------------------------------------------------------

    /// Generate an image from a prompt.
    #[wasm_bindgen(js_name = "generateImage")]
    pub async fn generate_image(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: ImageRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid ImageRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { ImageGeneration::generate_image(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Upscale an existing image.
    #[wasm_bindgen(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: UpscaleRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid UpscaleRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { ImageGeneration::upscale_image(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // -----------------------------------------------------------------
    // Video generation
    // -----------------------------------------------------------------

    /// Generate a video from text.
    #[wasm_bindgen(js_name = "textToVideo")]
    pub async fn text_to_video(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: VideoRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid VideoRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { VideoGeneration::text_to_video(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Generate a video from a reference image.
    #[wasm_bindgen(js_name = "imageToVideo")]
    pub async fn image_to_video(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: VideoRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid VideoRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { VideoGeneration::image_to_video(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe an audio clip.
    #[wasm_bindgen]
    pub async fn transcribe(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: TranscriptionRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid TranscriptionRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { Transcription::transcribe(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // -----------------------------------------------------------------
    // 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D asset (mesh / gaussian splat / etc.) from text.
    #[wasm_bindgen(js_name = "generate3d")]
    pub async fn generate_3d(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: ThreeDRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid ThreeDRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(async move { ThreeDGeneration::generate_3d(inner.as_ref(), req).await })
                .await
                .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // -----------------------------------------------------------------
    // Background removal
    // -----------------------------------------------------------------

    /// Remove the background from an image.
    #[wasm_bindgen(js_name = "removeBackground")]
    pub async fn remove_background(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let req: BackgroundRemovalRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&format!("invalid BackgroundRemovalRequest: {e}")))?;
        let inner = Rc::clone(&self.inner);
        let result =
            SendFuture(
                async move { BackgroundRemoval::remove_background(inner.as_ref(), req).await },
            )
            .await
            .map_err(|e: BlazenError| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
