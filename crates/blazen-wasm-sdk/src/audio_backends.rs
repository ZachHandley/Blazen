//! Proxy-request bindings for the native audio backends.
//!
//! The concrete audio inference engines re-exported by `blazen-llm`
//! ([`FasterWhisperBackend`](blazen_llm::FasterWhisperBackend),
//! [`SparkTtsBackend`](blazen_llm::SparkTtsBackend), and the
//! `*BackendHandle` provider wrappers) are native-only: they pull in
//! `CTranslate2` / candle / `hf-hub` weight loaders that do not compile to
//! `wasm32`. The WASM SDK therefore exposes these types as a **proxy-request
//! surface** — each class forwards its requests to a JS handler that calls a
//! remote inference endpoint (the same JS-handler pattern used by
//! [`crate::capability_providers`]). There is no local inference on `wasm32`.
//!
//! ```js
//! import { FasterWhisperConfig, SttBackendHandle } from '@blazen-dev/wasm';
//!
//! const stt = new SttBackendHandle('faster-whisper', {
//!   transcribe: async (req) => {
//!     const res = await fetch('/api/stt', { method: 'POST', body: JSON.stringify(req) });
//!     return await res.json(); // shape: TranscriptionResult
//!   },
//! });
//! const result = await stt.transcribe({ audioUrl: '/clip.wav', language: 'en' });
//! ```

use std::pin::Pin;

use js_sys::{Function, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};

// ---------------------------------------------------------------------------
// SendFuture — single-threaded wasm pattern (mirrors capability_providers).
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
// Shared proxy-handler dispatch.
// ---------------------------------------------------------------------------

/// Look up a function-valued method on the JS handler object.
fn lookup_handler(handlers: &JsValue, method: &str) -> Option<Function> {
    let key = JsValue::from_str(method);
    if !Reflect::has(handlers, &key).unwrap_or(false) {
        return None;
    }
    let candidate = Reflect::get(handlers, &key).ok()?;
    candidate.is_function().then(|| candidate.unchecked_into())
}

/// Invoke `method(arg)` on the JS handler object, awaiting any returned
/// promise and returning the resolved value.
async fn call_handler(handlers: &JsValue, method: &str, arg: &JsValue) -> Result<JsValue, JsValue> {
    let func = lookup_handler(handlers, method).ok_or_else(|| {
        JsValue::from_str(&format!(
            "audio backend proxy handler has no `{method}` method"
        ))
    })?;
    let raw = func.call1(handlers, arg)?;
    if raw.has_type::<js_sys::Promise>() {
        let promise: js_sys::Promise = raw.unchecked_into();
        JsFuture::from(promise).await
    } else {
        Ok(raw)
    }
}

/// Validate that `handlers` is a JS object and contains a callable `method`.
fn validate_handlers(provider_id: &str, handlers: &JsValue, method: &str) -> Result<(), JsValue> {
    if !handlers.is_object() {
        return Err(JsValue::from_str(&format!(
            "{provider_id}: handlers must be a JS object exposing a `{method}` method"
        )));
    }
    if lookup_handler(handlers, method).is_none() {
        return Err(JsValue::from_str(&format!(
            "{provider_id}: handlers object is missing the required `{method}` method"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FasterWhisperConfig
// ---------------------------------------------------------------------------

/// Configuration for the faster-whisper STT backend.
///
/// Mirrors [`blazen_llm::FasterWhisperConfig`]. On `wasm32` the config only
/// carries routing metadata (model id + optional revision) that the proxy
/// handler forwards to its remote endpoint — there is no local model dir or
/// decoder to configure because inference runs remotely.
#[wasm_bindgen(js_name = "FasterWhisperConfig")]
#[derive(Debug, Clone)]
pub struct WasmFasterWhisperConfig {
    model_id: String,
    revision: Option<String>,
}

#[wasm_bindgen(js_class = "FasterWhisperConfig")]
impl WasmFasterWhisperConfig {
    /// Create a new config. `model_id` defaults to
    /// `"Systran/faster-whisper-tiny"` when `undefined`.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(model_id: Option<String>, revision: Option<String>) -> Self {
        Self {
            model_id: model_id.unwrap_or_else(|| "Systran/faster-whisper-tiny".to_owned()),
            revision,
        }
    }

    /// The Hugging Face repo id for the `CTranslate2` Whisper bundle.
    #[wasm_bindgen(getter, js_name = "modelId")]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Optional Hugging Face Hub revision pin.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn revision(&self) -> Option<String> {
        self.revision.clone()
    }
}

// ---------------------------------------------------------------------------
// FasterWhisperBackend (proxy)
// ---------------------------------------------------------------------------

/// faster-whisper STT backend — proxy-request surface.
///
/// Native inference is unavailable on `wasm32`; construct with a JS handler
/// exposing an async `transcribe(request)` method that forwards to a remote
/// endpoint.
#[wasm_bindgen(js_name = "FasterWhisperBackend")]
pub struct WasmFasterWhisperBackend {
    config: WasmFasterWhisperConfig,
    handlers: JsValue,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmFasterWhisperBackend {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmFasterWhisperBackend {}

#[wasm_bindgen(js_class = "FasterWhisperBackend")]
impl WasmFasterWhisperBackend {
    /// Create a proxy-backed faster-whisper backend.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `handlers` is not a JS object exposing a
    /// `transcribe` method.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &WasmFasterWhisperConfig,
        handlers: JsValue,
    ) -> Result<WasmFasterWhisperBackend, JsValue> {
        validate_handlers("FasterWhisperBackend", &handlers, "transcribe")?;
        Ok(Self {
            config: config.clone(),
            handlers,
        })
    }

    /// Stable backend identifier.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn id(&self) -> String {
        self.config.model_id.clone()
    }

    /// Transcribe audio by forwarding `request` to the JS proxy handler.
    #[wasm_bindgen]
    pub fn transcribe(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "transcribe", &request).await
        }))
    }
}

// ---------------------------------------------------------------------------
// SttBackendHandle (proxy)
// ---------------------------------------------------------------------------

/// Provider wrapper around any STT backend — proxy-request surface.
///
/// Mirrors [`blazen_llm::SttBackendHandle`]. On `wasm32` it forwards
/// `transcribe` to a JS handler that calls a remote endpoint.
#[wasm_bindgen(js_name = "SttBackendHandle")]
pub struct WasmSttBackendHandle {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmSttBackendHandle {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmSttBackendHandle {}

#[wasm_bindgen(js_class = "SttBackendHandle")]
impl WasmSttBackendHandle {
    /// Wrap a JS proxy handler exposing an async `transcribe(request)` method.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `handlers` is missing `transcribe`.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: String, handlers: JsValue) -> Result<WasmSttBackendHandle, JsValue> {
        validate_handlers(&provider_id, &handlers, "transcribe")?;
        Ok(Self {
            provider_id,
            handlers,
        })
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Transcribe audio via the proxy handler.
    #[wasm_bindgen]
    pub fn transcribe(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "transcribe", &request).await
        }))
    }
}

// ---------------------------------------------------------------------------
// SparkTtsConfig
// ---------------------------------------------------------------------------

/// Configuration for the Spark-TTS backend.
///
/// Mirrors [`blazen_llm::SparkTtsConfig`]. On `wasm32` only the model-routing
/// metadata is carried; the proxy handler forwards it to a remote endpoint.
#[wasm_bindgen(js_name = "SparkTtsConfig")]
#[derive(Debug, Clone)]
pub struct WasmSparkTtsConfig {
    model_id: String,
    revision: Option<String>,
}

#[wasm_bindgen(js_class = "SparkTtsConfig")]
impl WasmSparkTtsConfig {
    /// Create a new config. `model_id` defaults to `"SparkAudio/Spark-TTS-0.5B"`.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(model_id: Option<String>, revision: Option<String>) -> Self {
        Self {
            model_id: model_id.unwrap_or_else(|| "SparkAudio/Spark-TTS-0.5B".to_owned()),
            revision,
        }
    }

    /// The Hugging Face repo id for the Spark-TTS weights.
    #[wasm_bindgen(getter, js_name = "modelId")]
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Optional Hugging Face Hub revision pin.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn revision(&self) -> Option<String> {
        self.revision.clone()
    }
}

// ---------------------------------------------------------------------------
// SparkTtsBackend (proxy)
// ---------------------------------------------------------------------------

/// Spark-TTS backend — proxy-request surface.
///
/// Native inference is unavailable on `wasm32`; construct with a JS handler
/// exposing an async `synthesize(request)` method that forwards to a remote
/// endpoint.
#[wasm_bindgen(js_name = "SparkTtsBackend")]
pub struct WasmSparkTtsBackend {
    config: WasmSparkTtsConfig,
    handlers: JsValue,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmSparkTtsBackend {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmSparkTtsBackend {}

#[wasm_bindgen(js_class = "SparkTtsBackend")]
impl WasmSparkTtsBackend {
    /// Create a proxy-backed Spark-TTS backend.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `handlers` is not a JS object exposing a
    /// `synthesize` method.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &WasmSparkTtsConfig,
        handlers: JsValue,
    ) -> Result<WasmSparkTtsBackend, JsValue> {
        validate_handlers("SparkTtsBackend", &handlers, "synthesize")?;
        Ok(Self {
            config: config.clone(),
            handlers,
        })
    }

    /// Stable backend identifier.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn id(&self) -> String {
        self.config.model_id.clone()
    }

    /// Synthesize speech by forwarding `request` to the JS proxy handler.
    #[wasm_bindgen]
    pub fn synthesize(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "synthesize", &request).await
        }))
    }
}

// ---------------------------------------------------------------------------
// TtsBackendHandle (proxy)
// ---------------------------------------------------------------------------

/// Provider wrapper around any TTS backend — proxy-request surface.
///
/// Mirrors [`blazen_llm::TtsBackendHandle`].
#[wasm_bindgen(js_name = "TtsBackendHandle")]
pub struct WasmTtsBackendHandle {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmTtsBackendHandle {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmTtsBackendHandle {}

#[wasm_bindgen(js_class = "TtsBackendHandle")]
impl WasmTtsBackendHandle {
    /// Wrap a JS proxy handler exposing an async `synthesize(request)` method.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `handlers` is missing `synthesize`.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: String, handlers: JsValue) -> Result<WasmTtsBackendHandle, JsValue> {
        validate_handlers(&provider_id, &handlers, "synthesize")?;
        Ok(Self {
            provider_id,
            handlers,
        })
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Synthesize speech via the proxy handler.
    #[wasm_bindgen]
    pub fn synthesize(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "synthesize", &request).await
        }))
    }
}

// ---------------------------------------------------------------------------
// MusicBackendHandle (proxy)
// ---------------------------------------------------------------------------

/// Provider wrapper around any music-generation backend — proxy-request
/// surface.
///
/// Mirrors [`blazen_llm::MusicBackendHandle`].
#[wasm_bindgen(js_name = "MusicBackendHandle")]
pub struct WasmMusicBackendHandle {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmMusicBackendHandle {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmMusicBackendHandle {}

#[wasm_bindgen(js_class = "MusicBackendHandle")]
impl WasmMusicBackendHandle {
    /// Wrap a JS proxy handler exposing an async `generateMusic(request)`
    /// method (and optionally `generateSfx(request)`).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `handlers` is missing `generateMusic`.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: String, handlers: JsValue) -> Result<WasmMusicBackendHandle, JsValue> {
        validate_handlers(&provider_id, &handlers, "generateMusic")?;
        Ok(Self {
            provider_id,
            handlers,
        })
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Generate music via the proxy handler.
    #[wasm_bindgen(js_name = "generateMusic")]
    pub fn generate_music(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "generateMusic", &request).await
        }))
    }

    /// Generate sound effects via the proxy handler.
    #[wasm_bindgen(js_name = "generateSfx")]
    pub fn generate_sfx(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "generateSfx", &request).await
        }))
    }
}

// ---------------------------------------------------------------------------
// CodecBackendHandle (proxy)
// ---------------------------------------------------------------------------

/// Provider wrapper around any neural audio codec backend — proxy-request
/// surface.
///
/// Mirrors [`blazen_llm::CodecBackendHandle`]. Forwards `encodeAudio` /
/// `decodeAudio` to a JS handler that calls a remote endpoint.
#[wasm_bindgen(js_name = "CodecBackendHandle")]
pub struct WasmCodecBackendHandle {
    provider_id: String,
    handlers: JsValue,
}

// SAFETY: wasm32 is single-threaded.
unsafe impl Send for WasmCodecBackendHandle {}
// SAFETY: wasm32 is single-threaded.
unsafe impl Sync for WasmCodecBackendHandle {}

#[wasm_bindgen(js_class = "CodecBackendHandle")]
impl WasmCodecBackendHandle {
    /// Wrap a JS proxy handler exposing async `encodeAudio(request)` /
    /// `decodeAudio(request)` methods.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `handlers` is missing `encodeAudio`.
    #[wasm_bindgen(constructor)]
    pub fn new(provider_id: String, handlers: JsValue) -> Result<WasmCodecBackendHandle, JsValue> {
        validate_handlers(&provider_id, &handlers, "encodeAudio")?;
        Ok(Self {
            provider_id,
            handlers,
        })
    }

    /// The provider identifier.
    #[wasm_bindgen(getter, js_name = "providerId")]
    #[must_use]
    pub fn provider_id(&self) -> String {
        self.provider_id.clone()
    }

    /// Encode PCM audio to codec tokens via the proxy handler.
    #[wasm_bindgen(js_name = "encodeAudio")]
    pub fn encode_audio(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "encodeAudio", &request).await
        }))
    }

    /// Decode codec tokens to PCM audio via the proxy handler.
    #[wasm_bindgen(js_name = "decodeAudio")]
    pub fn decode_audio(&self, request: JsValue) -> js_sys::Promise {
        let handlers = self.handlers.clone();
        future_to_promise(SendFuture(async move {
            call_handler(&handlers, "decodeAudio", &request).await
        }))
    }
}
