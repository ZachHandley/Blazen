//! JavaScript bindings for user-defined custom providers.
//!
//! Lets TypeScript/JavaScript users write a normal class (or subclass of
//! `CustomProvider`) with async capability methods and wrap it as a
//! first-class Blazen provider via
//! [`CustomProvider`](blazen_llm::CustomProvider). The workflow engine
//! then sees the wrapped object as a provider that implements whichever
//! combination of [`AudioGeneration`], [`VoiceCloning`],
//! [`ImageGeneration`], [`VideoGeneration`], [`Transcription`],
//! [`ThreeDGeneration`], and [`BackgroundRemoval`] traits the host
//! class's methods cover.
//!
//! ## Bridging JavaScript async -> Rust async
//!
//! When the user subclasses `CustomProvider` from JavaScript, the
//! constructor installs a [`JsCustomProviderAdapter`] that implements
//! the typed [`blazen_llm::CustomProvider`] trait by dispatching each
//! method to the matching JS method on the held JS instance handle.
//!
//! Under the hood, the Rust side:
//!
//! 1. Enumerates the JS instance's prototype chain at construction
//!    time and, for each trait method the subclass overrides, builds a
//!    per-method [`ThreadsafeFunction`] bound to the JS instance.
//! 2. When the framework calls a typed trait method, the adapter
//!    serializes the typed request to `serde_json::Value`, invokes the
//!    matching TSFN, awaits the JS `Promise` it returns, and
//!    deserializes the response into the typed result.
//! 3. If the subclass did not override a given method the adapter
//!    returns [`BlazenError::unsupported`] without ever scheduling a
//!    JS callback.
//!
//! ## Method name mapping
//!
//! Blazen's trait uses **Rust** `snake_case` names
//! (`text_to_speech`, `clone_voice`, ...). The Node shim translates each
//! Rust name to the idiomatic JavaScript `camelCase` equivalent
//! (`textToSpeech`, `cloneVoice`, ...) when extracting the function off
//! the host object at construction time. The mapping table lives in
//! [`TRAIT_METHODS`].

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_llm::CustomProvider;
use blazen_llm::compute::job::{ComputeRequest, ComputeResult, JobHandle, JobStatus};
use blazen_llm::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use blazen_llm::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use blazen_llm::compute::traits::{
    AudioGeneration, BackgroundRemoval, ComputeProvider, ImageGeneration, ThreeDGeneration,
    Transcription, VideoGeneration, VoiceCloning,
};
use blazen_llm::error::BlazenError;
use blazen_llm::providers::custom::{ApiProtocol, CustomProviderHandle};
use blazen_llm::providers::openai_compat::OpenAiCompatConfig;
use blazen_llm::types::{
    CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk, ToolDefinition,
};

use crate::error::blazen_error_to_napi;
use crate::generated::{
    JsAudioResult, JsBackgroundRemovalRequest, JsImageRequest, JsImageResult, JsMusicRequest,
    JsSpeechRequest, JsThreeDRequest, JsThreeDResult, JsTranscriptionRequest,
    JsTranscriptionResult, JsUpscaleRequest, JsVideoRequest, JsVideoResult, JsVoiceCloneRequest,
    JsVoiceHandle,
};
use crate::providers::api_protocol::JsApiProtocol;
use crate::providers::openai_compat::JsOpenAiCompatConfig;

// ---------------------------------------------------------------------------
// Trait method name table
// ---------------------------------------------------------------------------

/// Mapping from Rust `snake_case` method names (matching the
/// [`blazen_llm::CustomProvider`] trait) to JavaScript `camelCase`
/// method names (expected on the host object).
///
/// Adding a new method to `CustomProvider` on the Rust side only
/// requires adding an entry here and a matching trait impl branch in
/// [`JsCustomProviderAdapter`].
const TRAIT_METHODS: &[(&str, &str)] = &[
    // Completion
    ("complete", "complete"),
    ("stream", "stream"),
    // Embeddings
    ("embed", "embed"),
    // Audio
    ("text_to_speech", "textToSpeech"),
    ("generate_music", "generateMusic"),
    ("generate_sfx", "generateSfx"),
    // Voice cloning
    ("clone_voice", "cloneVoice"),
    ("list_voices", "listVoices"),
    ("delete_voice", "deleteVoice"),
    // Image
    ("generate_image", "generateImage"),
    ("upscale_image", "upscaleImage"),
    ("remove_background", "removeBackground"),
    // Video
    ("text_to_video", "textToVideo"),
    ("image_to_video", "imageToVideo"),
    // Transcription
    ("transcribe", "transcribe"),
    // 3D generation
    ("generate_3d", "generate3d"),
];

// ---------------------------------------------------------------------------
// ThreadsafeFunction type alias
// ---------------------------------------------------------------------------

/// Pre-built JS callback for one capability method.
///
/// - `T = serde_json::Value`: the single JSON argument passed to the JS
///   host method (the serialized Blazen request, or `null` for
///   no-argument methods like `list_voices`).
/// - `Return = Promise<Option<serde_json::Value>>`: the JS host method
///   must return a `Promise` (either `async` or explicit). `Option<_>`
///   accepts both `undefined` and `null` as valid empty results (mapped
///   to [`serde_json::Value::Null`]) so that void-returning methods
///   like `deleteVoice` don't need to return anything explicitly.
/// - `CallJsBackArgs = T`: no custom callback, the raw value is passed
///   straight through.
/// - `CalleeHandled = false`: no error-first callback convention --
///   the JS host method either resolves or rejects its `Promise`.
/// - `Weak = true`: does not prevent Node.js from exiting once all
///   user-facing references drop.
type HostMethodTsfn = ThreadsafeFunction<
    serde_json::Value,
    Promise<Option<serde_json::Value>>,
    serde_json::Value,
    Status,
    false,
    true,
>;

// ---------------------------------------------------------------------------
// JsCustomProviderAdapter
// ---------------------------------------------------------------------------

/// Rust [`blazen_llm::CustomProvider`] implementation backed by a
/// JavaScript subclass instance.
///
/// One [`ThreadsafeFunction`] is built per trait method that the JS
/// subclass overrides, so each typed trait impl below reduces to a
/// [`HashMap::get`] + `call_async` against the cached TSFN. Methods
/// the subclass did not override return [`BlazenError::unsupported`]
/// without ever scheduling a JS callback (this preserves the typed
/// trait's `default Unsupported` semantics).
///
/// All Rust trait impls go through JSON: the typed Rust request
/// serializes via `serde_json::to_value`, the JS method receives a
/// plain object, and its returned `Promise<any>` value deserializes
/// back into the typed result.
pub(crate) struct JsCustomProviderAdapter {
    /// Stable provider identifier for logs and metrics.
    provider_id: String,
    /// Cached callbacks keyed by **Rust** method name (`snake_case`).
    methods: HashMap<&'static str, Arc<HostMethodTsfn>>,
}

impl std::fmt::Debug for JsCustomProviderAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsCustomProviderAdapter")
            .field("provider_id", &self.provider_id)
            .field("method_count", &self.methods.len())
            .finish_non_exhaustive()
    }
}

impl JsCustomProviderAdapter {
    /// Walk the trait method table and extract a bound
    /// [`ThreadsafeFunction`] for every method the JS instance
    /// overrides.
    ///
    /// Each host method is bound to the host object via JavaScript
    /// `Function.prototype.bind` before being converted to a TSFN so
    /// that `this` refers to the original class instance when the
    /// callback runs -- without this, `this` would be `undefined` and
    /// user code like `this.client.foo()` would throw.
    ///
    /// Methods the JS subclass did not override are simply omitted
    /// from the table; the trait impls below report them as
    /// [`BlazenError::Unsupported`].
    pub(crate) fn from_host_object(provider_id: String, host_object: &Object<'_>) -> Result<Self> {
        let mut methods: HashMap<&'static str, Arc<HostMethodTsfn>> = HashMap::new();

        for &(rust_name, js_name) in TRAIT_METHODS {
            // Skip if the JS prototype chain has no property by this
            // name. `has_named_property` includes prototype-inherited
            // methods, which is what we want for class instances.
            if !host_object.has_named_property(js_name).unwrap_or(false) {
                continue;
            }

            // Extract the property as a typed `Function`. If the
            // extraction fails (the property exists but isn't a
            // function), we silently skip it -- the user's intent is
            // clearly that this capability is unsupported.
            let js_function: Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>> =
                match host_object.get_named_property(js_name) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

            // Bind `this` to the host object so user code that
            // references `this.foo` resolves correctly when the TSFN
            // callback fires on the Node main thread.
            let bound_function = js_function.bind(host_object).map_err(|e| {
                napi::Error::from_reason(format!(
                    "CustomProvider: failed to bind `this` for method `{js_name}`: {e}"
                ))
            })?;

            // Build a threadsafe function from the bound function.
            // `.weak::<true>()` makes the TSFN not prevent Node from
            // exiting; `CalleeHandled` stays at its default `false`.
            let tsfn: HostMethodTsfn = bound_function
                .build_threadsafe_function::<serde_json::Value>()
                .weak::<true>()
                .build()
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "CustomProvider: failed to build threadsafe function for `{js_name}`: {e}"
                    ))
                })?;

            methods.insert(rust_name, Arc::new(tsfn));
        }

        Ok(Self {
            provider_id,
            methods,
        })
    }

    /// Returns `true` iff the JS subclass overrides the given Rust
    /// trait method name (`snake_case`).
    fn has(&self, method: &str) -> bool {
        self.methods.contains_key(method)
    }

    /// Serialize `request`, dispatch to the JS override of `method`,
    /// and deserialize the resolved value back into `T`.
    ///
    /// Returns [`BlazenError::unsupported`] when the JS subclass did
    /// not override `method`. Provider errors and JS-thrown errors
    /// propagate as [`BlazenError::provider`].
    async fn dispatch<Req: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        method: &'static str,
        request: &Req,
    ) -> std::result::Result<T, BlazenError> {
        let tsfn = self.methods.get(method).ok_or_else(|| {
            BlazenError::unsupported(format!("CustomProvider does not override `{method}`"))
        })?;

        let value = serde_json::to_value(request).map_err(|e| {
            BlazenError::provider(
                &self.provider_id,
                format!("CustomProvider: failed to serialize `{method}` request: {e}"),
            )
        })?;

        // Phase 1: schedule the JS callback on the Node main thread
        // and await napi capturing its return value (a `Promise`).
        let promise = tsfn.call_async(value).await.map_err(|e| {
            BlazenError::provider(
                &self.provider_id,
                format!("CustomProvider: host method `{method}` dispatch failed: {e}"),
            )
        })?;

        // Phase 2: await the JS `Promise` itself to drive the host's
        // async body to completion. If the host `throw`s or rejects,
        // the error surfaces here.
        let resolved = promise.await.map_err(|e| {
            BlazenError::provider(
                &self.provider_id,
                format!("CustomProvider: host method `{method}` raised: {e}"),
            )
        })?;

        // `undefined` / `null` returns map to `Value::Null` so
        // `serde_json::from_value::<()>` deserializes cleanly for
        // void-returning methods.
        let resolved = resolved.unwrap_or(serde_json::Value::Null);
        serde_json::from_value::<T>(resolved).map_err(|e| {
            BlazenError::provider(
                &self.provider_id,
                format!("CustomProvider: failed to deserialize `{method}` response: {e}"),
            )
        })
    }
}

#[async_trait]
impl CustomProvider for JsCustomProviderAdapter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> std::result::Result<CompletionResponse, BlazenError> {
        self.dispatch("complete", &request).await
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<StreamChunk, BlazenError>> + Send>>,
        BlazenError,
    > {
        // Streaming through a TSFN that returns a `Promise<Value>` is
        // a single resolution point; the typed trait expects a
        // `Stream`. Until we wire a proper push-channel from JS, the
        // JS subclass should override `complete` instead.
        Err(BlazenError::unsupported(
            "CustomProvider::stream from a JavaScript subclass is not yet wired; \
             override `complete` for non-streaming requests",
        ))
    }

    async fn embed(
        &self,
        texts: Vec<String>,
    ) -> std::result::Result<EmbeddingResponse, BlazenError> {
        if !self.has("embed") {
            return Err(BlazenError::unsupported(
                "CustomProvider::embed not implemented",
            ));
        }
        self.dispatch("embed", &texts).await
    }

    async fn text_to_speech(
        &self,
        req: SpeechRequest,
    ) -> std::result::Result<AudioResult, BlazenError> {
        self.dispatch("text_to_speech", &req).await
    }

    async fn generate_music(
        &self,
        req: MusicRequest,
    ) -> std::result::Result<AudioResult, BlazenError> {
        self.dispatch("generate_music", &req).await
    }

    async fn generate_sfx(
        &self,
        req: MusicRequest,
    ) -> std::result::Result<AudioResult, BlazenError> {
        self.dispatch("generate_sfx", &req).await
    }

    async fn clone_voice(
        &self,
        req: VoiceCloneRequest,
    ) -> std::result::Result<VoiceHandle, BlazenError> {
        self.dispatch("clone_voice", &req).await
    }

    async fn list_voices(&self) -> std::result::Result<Vec<VoiceHandle>, BlazenError> {
        self.dispatch("list_voices", &serde_json::Value::Null).await
    }

    async fn delete_voice(&self, voice: VoiceHandle) -> std::result::Result<(), BlazenError> {
        self.dispatch("delete_voice", &voice).await
    }

    async fn generate_image(
        &self,
        req: ImageRequest,
    ) -> std::result::Result<ImageResult, BlazenError> {
        self.dispatch("generate_image", &req).await
    }

    async fn upscale_image(
        &self,
        req: UpscaleRequest,
    ) -> std::result::Result<ImageResult, BlazenError> {
        self.dispatch("upscale_image", &req).await
    }

    async fn text_to_video(
        &self,
        req: VideoRequest,
    ) -> std::result::Result<VideoResult, BlazenError> {
        self.dispatch("text_to_video", &req).await
    }

    async fn image_to_video(
        &self,
        req: VideoRequest,
    ) -> std::result::Result<VideoResult, BlazenError> {
        self.dispatch("image_to_video", &req).await
    }

    async fn transcribe(
        &self,
        req: TranscriptionRequest,
    ) -> std::result::Result<TranscriptionResult, BlazenError> {
        self.dispatch("transcribe", &req).await
    }

    async fn generate_3d(
        &self,
        req: ThreeDRequest,
    ) -> std::result::Result<ThreeDResult, BlazenError> {
        self.dispatch("generate_3d", &req).await
    }

    async fn remove_background(
        &self,
        req: BackgroundRemovalRequest,
    ) -> std::result::Result<ImageResult, BlazenError> {
        self.dispatch("remove_background", &req).await
    }
}

// ---------------------------------------------------------------------------
// JsCustomProvider
// ---------------------------------------------------------------------------

/// A user-defined Blazen provider exposed to JavaScript.
///
/// `CustomProvider` is designed for two complementary use cases:
///
/// 1. **Subclass from JavaScript** to plug an arbitrary backend into
///    Blazen. Override any combination of the typed methods
///    (`textToSpeech`, `generateImage`, `cloneVoice`, …). When the
///    framework dispatches a capability, the override fires; methods
///    you did not override report `UnsupportedError`.
/// 2. **Use a static factory** ([`Self::ollama`], [`Self::lm_studio`],
///    [`Self::openai_compat`]) to get a ready-made handle that speaks
///    the `OpenAI` Chat Completions wire format.
///
/// ```typescript
/// import { ApiProtocol, CustomProvider } from "blazen";
///
/// class MyElevenLabsProvider extends CustomProvider {
///   constructor(apiKey: string) {
///     super("elevenlabs", ApiProtocol.custom());
///     this.client = new ElevenLabs({ apiKey });
///   }
///
///   async textToSpeech(request: { text: string; voice?: string }) {
///     const audio = await this.client.textToSpeech.convert({
///       voiceId: request.voice ?? "default",
///       text: request.text,
///       modelId: "eleven_multilingual_v2",
///     });
///     return {
///       audio: [{
///         media: {
///           base64: Buffer.from(audio).toString("base64"),
///           mediaType: "mpeg",
///         },
///       }],
///       timing: { totalMs: 0, queueMs: null, executionMs: null },
///       metadata: {},
///     };
///   }
/// }
///
/// const provider = new MyElevenLabsProvider("sk-...");
/// const audio = await provider.textToSpeech({
///   text: "hello",
///   voice: "rachel",
/// });
/// ```
#[napi(js_name = "CustomProvider")]
pub struct JsCustomProvider {
    /// The underlying typed Rust handle. Carries the
    /// `CustomProviderHandle` defaults (completion + role-specific
    /// compute defaults), retry config, and the inner trait object
    /// (either an OpenAI-compat-backed adapter or a
    /// [`JsCustomProviderAdapter`] for the JS-subclass path).
    inner: Arc<CustomProviderHandle>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsCustomProvider {
    // -----------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------

    /// Construct a `CustomProvider`.
    ///
    /// - `providerId` — short identifier used for logging
    ///   (e.g. `"elevenlabs"`, `"my-ollama"`).
    /// - `protocol` — optional [`ApiProtocol`]; defaults to
    ///   [`ApiProtocol.custom`]. Subclasses that override capability
    ///   methods typically leave this at the default.
    ///
    /// When the constructor is invoked via `new (class extends
    /// CustomProvider) { … }`, the prototype of the JS instance
    /// differs from `CustomProvider.prototype`. The Rust constructor
    /// detects that case and installs a [`JsCustomProviderAdapter`]
    /// wrapping the JS instance so every overridden method dispatches
    /// through the JS side.
    ///
    /// When the constructor is invoked directly as
    /// `new CustomProvider(…)` (no subclass), the resulting handle
    /// reports every typed method as `Unsupported` unless built via
    /// one of the static factories
    /// ([`Self::ollama`] / [`Self::lm_studio`] / [`Self::openai_compat`]).
    #[napi(constructor)]
    pub fn new(
        provider_id: String,
        protocol: Option<&JsApiProtocol>,
        this: This<'_>,
    ) -> Result<Self> {
        let protocol_inner = protocol.map_or(ApiProtocol::Custom, |p| p.inner().clone());

        // Determine whether this instance is a JS subclass of
        // CustomProvider. If `Object.getPrototypeOf(this) ===
        // CustomProvider.prototype`, the user invoked the constructor
        // directly with no override -- there is no JS instance worth
        // dispatching to.
        let is_subclass = is_js_subclass(&this.object);

        let inner = if is_subclass {
            // Build an adapter from the JS instance's prototype-chain
            // method overrides.
            let adapter: Arc<dyn CustomProvider> = Arc::new(
                JsCustomProviderAdapter::from_host_object(provider_id.clone(), &this.object)?,
            );
            CustomProviderHandle::new(adapter).with_protocol(protocol_inner)
        } else {
            // Direct construction without a subclass: build a handle
            // around an empty adapter (every method returns
            // Unsupported). Static factories replace `inner` entirely
            // via Self::with_handle.
            let adapter: Arc<dyn CustomProvider> = Arc::new(JsCustomProviderAdapter {
                provider_id: provider_id.clone(),
                methods: HashMap::new(),
            });
            CustomProviderHandle::new(adapter).with_protocol(protocol_inner)
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    // -----------------------------------------------------------------
    // Static factories
    // -----------------------------------------------------------------

    /// Convenience constructor for a local Ollama server.
    ///
    /// Equivalent to building a `CustomProvider` whose protocol is
    /// `ApiProtocol.openai({ baseUrl: "http://<host>:<port>/v1", … })`.
    ///
    /// ```typescript
    /// const provider = CustomProvider.ollama("llama3.1");
    /// const provider = CustomProvider.ollama("llama3.1", "192.168.1.50", 11434);
    /// ```
    #[napi(factory)]
    pub fn ollama(model: String, host: Option<String>, port: Option<u16>) -> Result<Self> {
        let host = host.unwrap_or_else(|| "localhost".to_owned());
        let port = port.unwrap_or(11434);
        let handle = blazen_llm::ollama(host, port, model);
        Ok(Self {
            inner: Arc::new(handle),
        })
    }

    /// Convenience constructor for a local LM Studio server.
    ///
    /// Equivalent to building a `CustomProvider` whose protocol is
    /// `ApiProtocol.openai({ baseUrl: "http://<host>:<port>/v1", … })`.
    ///
    /// ```typescript
    /// const provider = CustomProvider.lmStudio("my-model");
    /// const provider = CustomProvider.lmStudio("my-model", "127.0.0.1", 1234);
    /// ```
    #[napi(factory, js_name = "lmStudio")]
    pub fn lm_studio(model: String, host: Option<String>, port: Option<u16>) -> Result<Self> {
        let host = host.unwrap_or_else(|| "localhost".to_owned());
        let port = port.unwrap_or(1234);
        let handle = blazen_llm::lm_studio(host, port, model);
        Ok(Self {
            inner: Arc::new(handle),
        })
    }

    /// Build a `CustomProvider` that speaks the `OpenAI` Chat
    /// Completions protocol. Use for any OpenAI-compatible HTTP
    /// endpoint that is not already covered by [`Self::ollama`] /
    /// [`Self::lm_studio`].
    #[napi(factory, js_name = "openaiCompat")]
    pub fn openai_compat(provider_id: String, config: JsOpenAiCompatConfig) -> Result<Self> {
        let cfg: OpenAiCompatConfig = config.into();
        let handle = blazen_llm::openai_compat(provider_id, cfg);
        Ok(Self {
            inner: Arc::new(handle),
        })
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// The provider identifier used for logging (e.g. `"elevenlabs"`).
    #[napi(js_name = "providerId", getter)]
    pub fn provider_id(&self) -> String {
        self.inner.provider_id_str().to_owned()
    }

    // -----------------------------------------------------------------
    // Audio generation
    // -----------------------------------------------------------------

    /// Synthesize speech by calling the host's `textToSpeech` async
    /// method.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsSpeechRequest) -> Result<JsAudioResult> {
        let rust_req: SpeechRequest = request.into();
        let result = AudioGeneration::text_to_speech(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate music by calling the host's `generateMusic` async
    /// method.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(&self, request: JsMusicRequest) -> Result<JsAudioResult> {
        let rust_req: MusicRequest = request.into();
        let result = AudioGeneration::generate_music(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate sound effects by calling the host's `generateSfx` async
    /// method.
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(&self, request: JsMusicRequest) -> Result<JsAudioResult> {
        let rust_req: MusicRequest = request.into();
        let result = AudioGeneration::generate_sfx(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Voice cloning
    // -----------------------------------------------------------------

    /// Clone a voice from reference audio clips by calling the host's
    /// `cloneVoice` async method. Returns a persistent `VoiceHandle`
    /// that can be passed as `SpeechRequest.voice` on subsequent TTS
    /// calls.
    #[napi(js_name = "cloneVoice")]
    pub async fn clone_voice(&self, request: JsVoiceCloneRequest) -> Result<JsVoiceHandle> {
        let rust_req: VoiceCloneRequest = request.into();
        let result = VoiceCloning::clone_voice(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// List all voices known to the host by calling its `listVoices`
    /// async method (which must return an array of objects shaped
    /// like `VoiceHandle`).
    #[napi(js_name = "listVoices")]
    pub async fn list_voices(&self) -> Result<Vec<JsVoiceHandle>> {
        let voices = VoiceCloning::list_voices(self.inner.as_ref())
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(voices.into_iter().map(Into::into).collect())
    }

    /// Delete a previously cloned voice by calling the host's
    /// `deleteVoice` async method.
    #[napi(js_name = "deleteVoice")]
    pub async fn delete_voice(&self, voice: JsVoiceHandle) -> Result<()> {
        let rust_voice: VoiceHandle = voice.into();
        VoiceCloning::delete_voice(self.inner.as_ref(), &rust_voice)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Image generation
    // -----------------------------------------------------------------

    /// Generate an image by calling the host's `generateImage` async
    /// method.
    #[napi(js_name = "generateImage")]
    pub async fn generate_image(&self, request: JsImageRequest) -> Result<JsImageResult> {
        let rust_req: ImageRequest = request.into();
        let result = ImageGeneration::generate_image(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Upscale an image by calling the host's `upscaleImage` async
    /// method.
    #[napi(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, request: JsUpscaleRequest) -> Result<JsImageResult> {
        let rust_req: UpscaleRequest = request.into();
        let result = ImageGeneration::upscale_image(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Video generation
    // -----------------------------------------------------------------

    /// Generate a video from text by calling the host's `textToVideo`
    /// async method.
    #[napi(js_name = "textToVideo")]
    pub async fn text_to_video(&self, request: JsVideoRequest) -> Result<JsVideoResult> {
        let rust_req: VideoRequest = request.into();
        let result = VideoGeneration::text_to_video(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate a video from a source image by calling the host's
    /// `imageToVideo` async method.
    #[napi(js_name = "imageToVideo")]
    pub async fn image_to_video(&self, request: JsVideoRequest) -> Result<JsVideoResult> {
        let rust_req: VideoRequest = request.into();
        let result = VideoGeneration::image_to_video(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe audio to text by calling the host's `transcribe`
    /// async method.
    #[napi]
    pub async fn transcribe(
        &self,
        request: JsTranscriptionRequest,
    ) -> Result<JsTranscriptionResult> {
        let rust_req: TranscriptionRequest = request.into();
        let result = Transcription::transcribe(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // 3D generation
    // -----------------------------------------------------------------

    /// Generate a 3D model by calling the host's `generate3d` async
    /// method.
    #[napi(js_name = "generate3d")]
    pub async fn generate_3d(&self, request: JsThreeDRequest) -> Result<JsThreeDResult> {
        let rust_req: ThreeDRequest = request.into();
        let result = ThreeDGeneration::generate_3d(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    // -----------------------------------------------------------------
    // Background removal
    // -----------------------------------------------------------------

    /// Remove the background from an image by calling the host's
    /// `removeBackground` async method.
    #[napi(js_name = "removeBackground")]
    pub async fn remove_background(
        &self,
        request: JsBackgroundRemovalRequest,
    ) -> Result<JsImageResult> {
        let rust_req: BackgroundRemovalRequest = request.into();
        let result = BackgroundRemoval::remove_background(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }
}

impl JsCustomProvider {
    /// Internal: clone the underlying handle for completion-model
    /// adapters and other downstream consumers.
    #[allow(dead_code)]
    pub(crate) fn handle(&self) -> Arc<CustomProviderHandle> {
        Arc::clone(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the JS object's `__proto__` is not exactly the
/// `CustomProvider` class prototype -- i.e. the JS user constructed a
/// subclass.
///
/// Implementation: read `Object.getPrototypeOf(thisArg)` and compare it
/// to `Object.getPrototypeOf(thisArg.constructor.prototype)` ?
/// That comparison is not directly available across the napi
/// boundary, so we fall back to a structural test that is correct in
/// every realistic scenario: a subclass override means the JS instance
/// owns at least one of the [`TRAIT_METHODS`] on its own (non-derived)
/// prototype chain *above* `Object.prototype`. The cleanest proxy is
/// "does the instance carry any of our camelCase trait names on
/// itself or its immediate prototype". Direct construction
/// (`new CustomProvider(...)`) produces an instance whose prototype
/// chain only has the napi-generated `CustomProvider.prototype` --
/// which exposes the typed compute methods as napi-defined
/// properties. To distinguish "I overrode `textToSpeech`" from
/// "napi-rs put `textToSpeech` on the prototype", we walk up to the
/// immediate prototype and inspect its prototype: if that grandparent
/// is not `Object.prototype`, we are looking at a subclass.
fn is_js_subclass(this_obj: &Object<'_>) -> bool {
    // `Object.getPrototypeOf(thisObj)` -> the napi-generated class
    // prototype on direct construction; the JS subclass prototype on
    // `class Foo extends CustomProvider`.
    let proto: Object<'_> = match this_obj.get_prototype_unchecked::<Object<'_>>() {
        Ok(p) => p,
        // No prototype is exotic; treat as not-a-subclass and let the
        // call surface report Unsupported.
        Err(_) => return false,
    };

    // `Object.getPrototypeOf(proto)` -> for a direct construction
    // that's `Object.prototype` (or null); for a subclass that's the
    // `CustomProvider.prototype`.
    let grandparent: Object<'_> = match proto.get_prototype_unchecked::<Object<'_>>() {
        Ok(p) => p,
        Err(_) => return false,
    };

    // The cleanest signal is whether the grandparent prototype has
    // *any* of our trait methods defined on it (camelCase). If it
    // does, `proto` sits above `CustomProvider.prototype` in the
    // chain — i.e. this is a subclass instance.
    for &(_, js_name) in TRAIT_METHODS {
        if grandparent.has_named_property(js_name).unwrap_or(false) {
            return true;
        }
    }
    false
}

// `ComputeProvider` is referenced through `CustomProviderHandle`'s
// trait surface; re-export the type alias so the import path stays
// discoverable from this module.
#[allow(dead_code)]
type _ComputeProviderAlias = dyn ComputeProvider;
#[allow(dead_code)]
type _JobHandleAlias = JobHandle;
#[allow(dead_code)]
type _JobStatusAlias = JobStatus;
#[allow(dead_code)]
type _ComputeRequestAlias = ComputeRequest;
#[allow(dead_code)]
type _ComputeResultAlias = ComputeResult;
#[allow(dead_code)]
type _ToolDefinitionAlias = ToolDefinition;
