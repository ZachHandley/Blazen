//! JavaScript bindings for user-defined custom providers.
//!
//! Lets TypeScript/JavaScript users write a normal class with async
//! capability methods and wrap it as a first-class Blazen provider via
//! [`CustomProvider`](blazen_llm::CustomProvider). The workflow engine
//! then sees the wrapped object as a provider that implements whichever
//! combination of [`AudioGeneration`], [`VoiceCloning`],
//! [`ImageGeneration`], [`VideoGeneration`], [`Transcription`],
//! [`ThreeDGeneration`], and [`BackgroundRemoval`] traits the host
//! class's methods cover.
//!
//! ## Bridging JavaScript async -> Rust async
//!
//! Each capability method on [`JsCustomProvider`] looks like a standard
//! async method from JavaScript's perspective: it returns a `Promise`.
//! Under the hood, the Rust side:
//!
//! 1. Serializes the typed request (`JsSpeechRequest`, etc.) into
//!    `serde_json::Value` via the `CustomProvider::call_typed` helper.
//! 2. Hands the JSON to [`NodeHostDispatch::call`], which:
//!    - Looks up a per-method cached [`ThreadsafeFunction`] built at
//!      construction time from the host class's prototype method.
//!    - Awaits `ThreadsafeFunction::call_async`, which schedules a JS
//!      callback on Node's main thread and returns a `Promise` handle.
//!    - Awaits the returned `Promise` to drive the host-language
//!      coroutine to completion.
//! 3. Deserializes the JSON response into the capability method's
//!    return type.
//!
//! The `has_method` fast-path uses the pre-built method map, so missing
//! capabilities short-circuit to [`BlazenError::unsupported`] without
//! ever scheduling a JS callback.
//!
//! ## Method name mapping
//!
//! Blazen's [`HostDispatch`] trait calls methods using **Rust**
//! `snake_case` names (`text_to_speech`, `clone_voice`, ...). The Node
//! shim translates each Rust name to the idiomatic JavaScript
//! `camelCase` equivalent (`textToSpeech`, `cloneVoice`, ...) when
//! extracting the function off the host object at construction time.
//! The mapping table lives in [`CAPABILITY_METHODS`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ComputeProvider, ImageGeneration, ThreeDGeneration,
    Transcription, VideoGeneration, VoiceCloning,
};
use blazen_llm::error::BlazenError;
use blazen_llm::{CustomProvider, HostDispatch};

use crate::error::blazen_error_to_napi;
use crate::generated::{
    JsAudioResult, JsBackgroundRemovalRequest, JsImageRequest, JsImageResult, JsMusicRequest,
    JsSpeechRequest, JsThreeDRequest, JsThreeDResult, JsTranscriptionRequest,
    JsTranscriptionResult, JsUpscaleRequest, JsVideoRequest, JsVideoResult, JsVoiceCloneRequest,
    JsVoiceHandle,
};

// ---------------------------------------------------------------------------
// Capability method name table
// ---------------------------------------------------------------------------

/// Mapping from Rust `snake_case` method names (used by [`HostDispatch`])
/// to JavaScript `camelCase` method names (expected on the host object).
///
/// Adding a new capability to `CustomProvider<D>` on the Rust side only
/// requires adding an entry here and a matching `#[napi]` async method
/// on [`JsCustomProvider`] -- no other wiring is needed because
/// [`CustomProvider::call_typed`] dispatches through JSON.
const CAPABILITY_METHODS: &[(&str, &str)] = &[
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
    // Raw compute (ComputeProvider trait)
    ("submit", "submit"),
    ("status", "status"),
    ("result", "result"),
    ("cancel", "cancel"),
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
// NodeHostDispatch
// ---------------------------------------------------------------------------

/// [`HostDispatch`] implementation backed by a table of pre-built
/// [`ThreadsafeFunction`]s, one per capability method that the host
/// object provides.
///
/// All method lookups happen at construction time (see
/// [`NodeHostDispatch::from_host_object`]) so [`HostDispatch::call`]
/// reduces to a `HashMap::get` and a `call_async` on the cached TSF.
/// `has_method` is a pure `HashMap::contains_key` check with zero
/// cross-thread work.
pub struct NodeHostDispatch {
    /// Cached callbacks keyed by **Rust** method name (`snake_case`).
    methods: HashMap<&'static str, Arc<HostMethodTsfn>>,
}

impl NodeHostDispatch {
    /// Walk the capability method table and extract a bound
    /// [`ThreadsafeFunction`] for every method the host object provides.
    ///
    /// Each host method is bound to the host object via JavaScript
    /// `Function.prototype.bind` before being converted to a TSF so
    /// that `this` refers to the original class instance when the
    /// callback runs -- without this, `this` would be `undefined` and
    /// user code like `this.client.foo()` would throw.
    ///
    /// Missing methods are simply omitted from the table; they are
    /// reported as [`BlazenError::Unsupported`] on access.
    pub(crate) fn from_host_object(host_object: &Object<'_>) -> Result<Self> {
        let mut methods: HashMap<&'static str, Arc<HostMethodTsfn>> = HashMap::new();

        for &(rust_name, js_name) in CAPABILITY_METHODS {
            // Skip if the host object has no property by this name.
            // JS semantics means `has_named_property` includes prototype-
            // inherited methods (which is what we want for class
            // instances).
            if !host_object.has_named_property(js_name).unwrap_or(false) {
                continue;
            }

            // Extract as a typed `Function` so we can `bind(this)`.
            // The signature is `(arg: Value) -> Promise<Option<Value>>`
            // to match the TSF type parameters we want downstream.
            //
            // If extraction fails (e.g. the property is a data field
            // instead of a function), we silently skip it: the user's
            // intent is clearly that this capability is unsupported.
            let js_function: Function<'_, serde_json::Value, Promise<Option<serde_json::Value>>> =
                match host_object.get_named_property(js_name) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

            // Bind `this` to the host object. The returned bound
            // function is a freshly-created JS function whose internal
            // `[[BoundThis]]` slot points at the host instance.
            let bound_function = js_function.bind(host_object).map_err(|e| {
                napi::Error::from_reason(format!(
                    "custom provider: failed to bind `this` for method `{js_name}`: {e}"
                ))
            })?;

            // Build a threadsafe function from the bound function.
            // `.weak::<true>()` makes the TSF not prevent Node from
            // exiting; `CalleeHandled` stays at its default `false`.
            let tsfn: HostMethodTsfn = bound_function
                .build_threadsafe_function::<serde_json::Value>()
                .weak::<true>()
                .build()
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "custom provider: failed to build threadsafe function for `{js_name}`: {e}"
                    ))
                })?;

            methods.insert(rust_name, Arc::new(tsfn));
        }

        Ok(Self { methods })
    }
}

#[async_trait]
impl HostDispatch for NodeHostDispatch {
    async fn call(
        &self,
        method: &str,
        request: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, BlazenError> {
        // Look up the pre-built TSF. If it is absent, the host object
        // did not declare this method at construction time.
        let tsfn = self.methods.get(method).ok_or_else(|| {
            BlazenError::unsupported(format!(
                "custom provider does not implement method `{method}`"
            ))
        })?;

        // Phase 1: schedule the JS callback on the Node main thread
        // and await its resolution. The TSF's inner future resolves
        // once napi has invoked the JS function and captured its
        // return value (a `Promise`).
        let promise = tsfn.call_async(request).await.map_err(|e| {
            BlazenError::provider(
                "custom",
                format!("host method `{method}` dispatch failed: {e}"),
            )
        })?;

        // Phase 2: await the JS `Promise` itself to drive the host's
        // async body to completion. If the host `throw`s or rejects,
        // the error is surfaced here.
        let value = promise.await.map_err(|e| {
            BlazenError::provider("custom", format!("host method `{method}` raised: {e}"))
        })?;

        // `undefined` / `null` returns are mapped to `Value::Null` so
        // that downstream `call_typed::<..., ()>` deserializes cleanly.
        Ok(value.unwrap_or(serde_json::Value::Null))
    }

    fn has_method(&self, method: &str) -> bool {
        self.methods.contains_key(method)
    }
}

// ---------------------------------------------------------------------------
// CustomProviderOptions
// ---------------------------------------------------------------------------

/// Optional configuration for a [`JsCustomProvider`].
#[napi(object)]
pub struct CustomProviderOptions {
    /// Short identifier used for logging and returned from
    /// [`ComputeProvider::provider_id`]. Defaults to `"custom"`.
    #[napi(js_name = "providerId")]
    pub provider_id: Option<String>,
}

// ---------------------------------------------------------------------------
// JsCustomProvider
// ---------------------------------------------------------------------------

/// A user-defined Blazen provider backed by a JavaScript class instance.
///
/// Wraps an arbitrary object whose async methods match Blazen's
/// capability trait names (`textToSpeech`, `cloneVoice`,
/// `generateImage`, etc.) and exposes them as a first-class provider.
/// The workflow engine treats the result as implementing every
/// capability trait whose methods the wrapped object provides; missing
/// methods return `UnsupportedError` when called.
///
/// Request/response shapes use Blazen's typed request/result types on
/// the JavaScript side and get serialized through napi's
/// `serde_json::Value` bridge to the wrapped object's methods, which
/// receive/return plain objects.
///
/// ```typescript
/// import { CustomProvider } from "blazen";
///
/// class MyElevenLabsProvider {
///     constructor(apiKey: string) {
///         this.client = new ElevenLabs({ apiKey });
///     }
///
///     async textToSpeech(request: { text: string; voice?: string }) {
///         const audio = await this.client.textToSpeech.convert({
///             voiceId: request.voice ?? "default",
///             text: request.text,
///             modelId: "eleven_multilingual_v2",
///         });
///         return {
///             audio: [{
///                 media: {
///                     base64: Buffer.from(audio).toString("base64"),
///                     mediaType: "mpeg",
///                 },
///             }],
///             timing: { totalMs: 0, queueMs: null, executionMs: null },
///             metadata: {},
///         };
///     }
/// }
///
/// const provider = new CustomProvider(
///     new MyElevenLabsProvider("..."),
///     { providerId: "elevenlabs" },
/// );
/// const audio = await provider.textToSpeech({
///     text: "hello",
///     voice: "rachel",
/// });
/// ```
#[napi(js_name = "CustomProvider")]
pub struct JsCustomProvider {
    inner: Arc<CustomProvider>,
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

    /// Wrap a JavaScript host object as a Blazen [`CustomProvider`].
    ///
    /// `hostObject` is a class instance (or plain object) whose async
    /// methods match Blazen capability trait method names
    /// (`textToSpeech`, `generateImage`, `cloneVoice`, ...). Host
    /// methods should be `async` and accept a single object argument
    /// shaped like the corresponding Blazen request type. Synchronous
    /// host methods are supported as long as they return a `Promise`
    /// explicitly -- the ordinary async dispatch path awaits the
    /// returned `Promise`.
    ///
    /// `options.providerId` is an optional short identifier used for
    /// logging and returned from [`JsCustomProvider::provider_id`].
    /// Defaults to `"custom"`.
    #[napi(constructor)]
    pub fn new(host_object: Object<'_>, options: Option<CustomProviderOptions>) -> Result<Self> {
        let provider_id = options
            .and_then(|o| o.provider_id)
            .unwrap_or_else(|| "custom".to_owned());

        let dispatch: Arc<dyn HostDispatch> =
            Arc::new(NodeHostDispatch::from_host_object(&host_object)?);

        Ok(Self {
            inner: Arc::new(CustomProvider::with_dispatch(provider_id, dispatch)),
        })
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// The provider identifier used for logging (e.g. `"elevenlabs"`).
    #[napi(js_name = "providerId", getter)]
    pub fn provider_id(&self) -> String {
        ComputeProvider::provider_id(self.inner.as_ref()).to_owned()
    }

    // -----------------------------------------------------------------
    // Audio generation
    // -----------------------------------------------------------------

    /// Synthesize speech by calling the host's `textToSpeech` async
    /// method.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsSpeechRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::SpeechRequest = request.into();
        let result = AudioGeneration::text_to_speech(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate music by calling the host's `generateMusic` async
    /// method.
    #[napi(js_name = "generateMusic")]
    pub async fn generate_music(&self, request: JsMusicRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::MusicRequest = request.into();
        let result = AudioGeneration::generate_music(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate sound effects by calling the host's `generateSfx` async
    /// method.
    #[napi(js_name = "generateSfx")]
    pub async fn generate_sfx(&self, request: JsMusicRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::MusicRequest = request.into();
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
        let rust_req: blazen_llm::compute::requests::VoiceCloneRequest = request.into();
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
        let rust_voice: blazen_llm::compute::results::VoiceHandle = voice.into();
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
        let rust_req: blazen_llm::compute::ImageRequest = request.into();
        let result = ImageGeneration::generate_image(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Upscale an image by calling the host's `upscaleImage` async
    /// method.
    #[napi(js_name = "upscaleImage")]
    pub async fn upscale_image(&self, request: JsUpscaleRequest) -> Result<JsImageResult> {
        let rust_req: blazen_llm::compute::UpscaleRequest = request.into();
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
        let rust_req: blazen_llm::compute::VideoRequest = request.into();
        let result = VideoGeneration::text_to_video(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }

    /// Generate a video from a source image by calling the host's
    /// `imageToVideo` async method.
    #[napi(js_name = "imageToVideo")]
    pub async fn image_to_video(&self, request: JsVideoRequest) -> Result<JsVideoResult> {
        let rust_req: blazen_llm::compute::VideoRequest = request.into();
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
        let rust_req: blazen_llm::compute::TranscriptionRequest = request.into();
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
        let rust_req: blazen_llm::compute::ThreeDRequest = request.into();
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
        let rust_req: blazen_llm::compute::BackgroundRemovalRequest = request.into();
        let result = BackgroundRemoval::remove_background(self.inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }
}
