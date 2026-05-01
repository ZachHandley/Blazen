//! Per-capability provider classes for WASM.
//!
//! Each class exposes one media-generation capability. Users instantiate
//! them by passing handler functions (or an object of named handlers) to
//! the constructor, then call the async methods to invoke them.
//!
//! Unlike the Node.js bindings which support class inheritance via
//! napi-rs, `wasm-bindgen` classes cannot be subclassed in JS. Instead,
//! users pass handler functions at construction time, and the async
//! methods delegate to those handlers.
//!
//! All capability methods accept a plain JS object matching the shape of
//! the corresponding Blazen request type and return a `Promise` that
//! resolves to a result object.

use std::pin::Pin;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// ---------------------------------------------------------------------------
// SendFuture wrapper (same pattern as agent.rs / js_completion.rs)
// ---------------------------------------------------------------------------

/// Wrapper that unsafely implements `Send` for a non-Send future.
/// SAFETY: WASM is single-threaded.
struct SendFuture<F>(F);

unsafe impl<F> Send for SendFuture<F> {}

impl<F: std::future::Future> std::future::Future for SendFuture<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We are not moving F, just projecting through the wrapper.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

// ---------------------------------------------------------------------------
// Shared helper: call a handler function from a JS handlers object
// ---------------------------------------------------------------------------

/// Extract a named handler function from a JS object and call it with
/// the given request argument, awaiting the result if it returns a
/// `Promise`.
async fn call_handler(
    handlers: &JsValue,
    method: &str,
    request: &JsValue,
) -> Result<JsValue, JsValue> {
    let handler = js_sys::Reflect::get(handlers, &JsValue::from_str(method))
        .map_err(|e| JsValue::from_str(&format!("failed to get handler '{method}': {e:?}")))?;

    if !handler.is_function() {
        return Err(JsValue::from_str(&format!(
            "handler '{method}' is not a function"
        )));
    }

    let func: &js_sys::Function = handler.unchecked_ref();
    let result = func
        .call1(&JsValue::NULL, request)
        .map_err(|e| JsValue::from_str(&format!("handler '{method}' threw: {e:?}")))?;

    if result.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = result.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("handler '{method}' rejected: {e:?}")))
    } else {
        Ok(result)
    }
}

/// Call a nullary handler (no request argument).
async fn call_handler_nullary(handlers: &JsValue, method: &str) -> Result<JsValue, JsValue> {
    let handler = js_sys::Reflect::get(handlers, &JsValue::from_str(method))
        .map_err(|e| JsValue::from_str(&format!("failed to get handler '{method}': {e:?}")))?;

    if !handler.is_function() {
        return Err(JsValue::from_str(&format!(
            "handler '{method}' is not a function"
        )));
    }

    let func: &js_sys::Function = handler.unchecked_ref();
    let result = func
        .call0(&JsValue::NULL)
        .map_err(|e| JsValue::from_str(&format!("handler '{method}' threw: {e:?}")))?;

    if result.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = result.unchecked_into();
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("handler '{method}' rejected: {e:?}")))
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// TypeScript type declarations
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_CAPABILITY_TYPES: &str = r#"
/** Handler for a single-method capability provider (e.g. TTSProvider). */
export type CapabilityHandler = (request: any) => Promise<any> | any;

/** Handlers for MusicProvider. */
export interface MusicHandlers {
    generateMusic: (request: any) => Promise<any> | any;
    generateSfx: (request: any) => Promise<any> | any;
}

/** Handlers for ImageProvider. */
export interface ImageHandlers {
    generateImage: (request: any) => Promise<any> | any;
    upscaleImage: (request: any) => Promise<any> | any;
}

/** Handlers for VideoProvider. */
export interface VideoHandlers {
    textToVideo: (request: any) => Promise<any> | any;
    imageToVideo: (request: any) => Promise<any> | any;
}

/** Handlers for VoiceProvider. */
export interface VoiceHandlers {
    cloneVoice: (request: any) => Promise<any> | any;
    listVoices: () => Promise<any> | any;
    deleteVoice: (voice: any) => Promise<any> | any;
}
"#;

// ===========================================================================
// 1. TTSProvider
// ===========================================================================

/// A text-to-speech provider backed by a JavaScript handler function.
///
/// ```js
/// const tts = new TTSProvider('elevenlabs', async (request) => {
///   const audio = await elevenlabs.textToSpeech(request);
///   return { audioData: audio, format: 'mp3' };
/// });
/// const result = await tts.textToSpeech({ text: 'Hello world', voice: 'alice' });
/// ```
#[wasm_bindgen(js_name = "TTSProvider")]
pub struct WasmTTSProvider {
    provider_id: String,
    handler: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmTTSProvider {}
unsafe impl Sync for WasmTTSProvider {}

#[wasm_bindgen(js_class = "TTSProvider")]
impl WasmTTSProvider {
    /// Create a new TTS provider.
    ///
    /// @param providerId - Short identifier (e.g. `"elevenlabs"`).
    /// @param handler    - Async function called with the speech request object.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handler: js_sys::Function) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handler,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Synthesize speech from text.
    #[wasm_bindgen(js_name = "textToSpeech")]
    pub fn text_to_speech(&self, request: JsValue) -> js_sys::Promise {
        let handler = self.handler.clone();
        future_to_promise(SendFuture(async move {
            let result = handler
                .call1(&JsValue::NULL, &request)
                .map_err(|e| JsValue::from_str(&format!("handler threw: {e:?}")))?;

            if result.has_type::<js_sys::Promise>() {
                let promise: js_sys::Promise = result.unchecked_into();
                wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("handler rejected: {e:?}")))
            } else {
                Ok(result)
            }
        }))
    }
}

// ===========================================================================
// 2. MusicProvider
// ===========================================================================

/// A music/SFX generation provider backed by JavaScript handler functions.
///
/// ```js
/// const music = new MusicProvider('suno', {
///   generateMusic: async (req) => ({ audioUrl: '...' }),
///   generateSfx: async (req) => ({ audioUrl: '...' }),
/// });
/// const result = await music.generateMusic({ prompt: 'upbeat jazz' });
/// ```
#[wasm_bindgen(js_name = "MusicProvider")]
pub struct WasmMusicProvider {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmMusicProvider {}
unsafe impl Sync for WasmMusicProvider {}

#[wasm_bindgen(js_class = "MusicProvider")]
impl WasmMusicProvider {
    /// Create a new music provider.
    ///
    /// @param providerId - Short identifier (e.g. `"suno"`).
    /// @param handlers   - Object with `generateMusic` and `generateSfx` async functions.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handlers: JsValue) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handlers,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Generate music from a prompt.
    #[wasm_bindgen(js_name = "generateMusic")]
    pub fn generate_music(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "generateMusic", &request).await
        }))
    }

    /// Generate a sound effect from a prompt.
    #[wasm_bindgen(js_name = "generateSfx")]
    pub fn generate_sfx(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "generateSfx", &request).await
        }))
    }
}

// ===========================================================================
// 3. ImageProvider
// ===========================================================================

/// An image generation provider backed by JavaScript handler functions.
///
/// ```js
/// const img = new ImageProvider('fal', {
///   generateImage: async (req) => ({ imageUrl: '...' }),
///   upscaleImage: async (req) => ({ imageUrl: '...' }),
/// });
/// const result = await img.generateImage({ prompt: 'a sunset over mountains' });
/// ```
#[wasm_bindgen(js_name = "ImageProvider")]
pub struct WasmImageProvider {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmImageProvider {}
unsafe impl Sync for WasmImageProvider {}

#[wasm_bindgen(js_class = "ImageProvider")]
impl WasmImageProvider {
    /// Create a new image provider.
    ///
    /// @param providerId - Short identifier (e.g. `"fal"`).
    /// @param handlers   - Object with `generateImage` and `upscaleImage` async functions.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handlers: JsValue) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handlers,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Generate an image from a prompt.
    #[wasm_bindgen(js_name = "generateImage")]
    pub fn generate_image(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "generateImage", &request).await
        }))
    }

    /// Upscale an existing image.
    #[wasm_bindgen(js_name = "upscaleImage")]
    pub fn upscale_image(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "upscaleImage", &request).await
        }))
    }
}

// ===========================================================================
// 4. VideoProvider
// ===========================================================================

/// A video generation provider backed by JavaScript handler functions.
///
/// ```js
/// const vid = new VideoProvider('runway', {
///   textToVideo: async (req) => ({ videoUrl: '...' }),
///   imageToVideo: async (req) => ({ videoUrl: '...' }),
/// });
/// const result = await vid.textToVideo({ prompt: 'a flying bird' });
/// ```
#[wasm_bindgen(js_name = "VideoProvider")]
pub struct WasmVideoProvider {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmVideoProvider {}
unsafe impl Sync for WasmVideoProvider {}

#[wasm_bindgen(js_class = "VideoProvider")]
impl WasmVideoProvider {
    /// Create a new video provider.
    ///
    /// @param providerId - Short identifier (e.g. `"runway"`).
    /// @param handlers   - Object with `textToVideo` and `imageToVideo` async functions.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handlers: JsValue) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handlers,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Generate a video from a text prompt.
    #[wasm_bindgen(js_name = "textToVideo")]
    pub fn text_to_video(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "textToVideo", &request).await
        }))
    }

    /// Generate a video from an image (image-to-video).
    #[wasm_bindgen(js_name = "imageToVideo")]
    pub fn image_to_video(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "imageToVideo", &request).await
        }))
    }
}

// ===========================================================================
// 5. ThreeDProvider
// ===========================================================================

/// A 3D model generation provider backed by a JavaScript handler function.
///
/// ```js
/// const three = new ThreeDProvider('meshy', async (request) => {
///   return { modelUrl: '...' };
/// });
/// const result = await three.generate3d({ prompt: 'a chair' });
/// ```
#[wasm_bindgen(js_name = "ThreeDProvider")]
pub struct WasmThreeDProvider {
    provider_id: String,
    handler: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmThreeDProvider {}
unsafe impl Sync for WasmThreeDProvider {}

#[wasm_bindgen(js_class = "ThreeDProvider")]
impl WasmThreeDProvider {
    /// Create a new 3D generation provider.
    ///
    /// @param providerId - Short identifier (e.g. `"meshy"`).
    /// @param handler    - Async function called with the 3D generation request.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handler: js_sys::Function) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handler,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Generate a 3D model from a prompt or image.
    #[wasm_bindgen(js_name = "generate3d")]
    pub fn generate_3d(&self, request: JsValue) -> js_sys::Promise {
        let handler = self.handler.clone();
        future_to_promise(SendFuture(async move {
            let result = handler
                .call1(&JsValue::NULL, &request)
                .map_err(|e| JsValue::from_str(&format!("handler threw: {e:?}")))?;

            if result.has_type::<js_sys::Promise>() {
                let promise: js_sys::Promise = result.unchecked_into();
                wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("handler rejected: {e:?}")))
            } else {
                Ok(result)
            }
        }))
    }
}

// ===========================================================================
// 6. BackgroundRemovalProvider
// ===========================================================================

/// A background removal provider backed by a JavaScript handler function.
///
/// ```js
/// const bgr = new BackgroundRemovalProvider('remove-bg', async (request) => {
///   return { imageUrl: '...' };
/// });
/// const result = await bgr.removeBackground({ imageUrl: 'https://...' });
/// ```
#[wasm_bindgen(js_name = "BackgroundRemovalProvider")]
pub struct WasmBackgroundRemovalProvider {
    provider_id: String,
    handler: js_sys::Function,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmBackgroundRemovalProvider {}
unsafe impl Sync for WasmBackgroundRemovalProvider {}

#[wasm_bindgen(js_class = "BackgroundRemovalProvider")]
impl WasmBackgroundRemovalProvider {
    /// Create a new background removal provider.
    ///
    /// @param providerId - Short identifier (e.g. `"remove-bg"`).
    /// @param handler    - Async function called with the removal request.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handler: js_sys::Function) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handler,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Remove the background from an image.
    #[wasm_bindgen(js_name = "removeBackground")]
    pub fn remove_background(&self, request: JsValue) -> js_sys::Promise {
        let handler = self.handler.clone();
        future_to_promise(SendFuture(async move {
            let result = handler
                .call1(&JsValue::NULL, &request)
                .map_err(|e| JsValue::from_str(&format!("handler threw: {e:?}")))?;

            if result.has_type::<js_sys::Promise>() {
                let promise: js_sys::Promise = result.unchecked_into();
                wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("handler rejected: {e:?}")))
            } else {
                Ok(result)
            }
        }))
    }
}

// ===========================================================================
// 7. VoiceProvider
// ===========================================================================

/// A voice cloning provider backed by JavaScript handler functions.
///
/// ```js
/// const voice = new VoiceProvider('elevenlabs', {
///   cloneVoice: async (req) => ({ voiceId: '...' }),
///   listVoices: async () => [{ voiceId: '...', name: '...' }],
///   deleteVoice: async (voice) => {},
/// });
/// const clone = await voice.cloneVoice({ audioUrls: ['...'] });
/// ```
#[wasm_bindgen(js_name = "VoiceProvider")]
pub struct WasmVoiceProvider {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmVoiceProvider {}
unsafe impl Sync for WasmVoiceProvider {}

#[wasm_bindgen(js_class = "VoiceProvider")]
impl WasmVoiceProvider {
    /// Create a new voice cloning provider.
    ///
    /// @param providerId - Short identifier (e.g. `"elevenlabs"`).
    /// @param handlers   - Object with `cloneVoice`, `listVoices`, and
    ///                     `deleteVoice` async functions.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: &str, handlers: JsValue) -> Self {
        Self {
            provider_id: provider_id.to_owned(),
            handlers,
        }
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Clone a voice from audio samples.
    #[wasm_bindgen(js_name = "cloneVoice")]
    pub fn clone_voice(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "cloneVoice", &request).await
        }))
    }

    /// List all available voices.
    #[wasm_bindgen(js_name = "listVoices")]
    pub fn list_voices(&self) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler_nullary(&handlers, "listVoices").await
        }))
    }

    /// Delete a previously-cloned voice.
    #[wasm_bindgen(js_name = "deleteVoice")]
    pub fn delete_voice(&self, voice: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "deleteVoice", &voice).await
        }))
    }
}
