//! `wasm-bindgen` wrapper for [`blazen_llm::providers::fal::FalProvider`].
//!
//! Surfaces both LLM completion and the full fal.ai compute capability suite
//! (image generation, upscaling, video, audio, transcription, 3-D, background
//! removal).

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use blazen_llm::compute::{
    AudioGeneration, BackgroundRemoval, ImageGeneration, ThreeDGeneration, Transcription,
    VideoGeneration,
};
use blazen_llm::providers::fal::FalProvider;
use blazen_llm::traits::CompletionModel;

use super::{
    apply_request_options, as_dyn_completion, complete_promise, fetch_client, resolve_key,
    stream_promise,
};
use crate::completion_model::WasmCompletionModel;

/// A fal.ai provider with LLM, image, video, audio, transcription, 3-D and
/// background-removal support.
///
/// ```js
/// const fal = new FalProvider({ apiKey: 'fal-key-...' });
/// const text = await fal.complete([ChatMessage.user('Hello!')]);
/// const image = await fal.generateImage({ prompt: 'A cat in space' });
/// ```
#[wasm_bindgen(js_name = "FalProvider")]
pub struct WasmFalProvider {
    inner: Arc<FalProvider>,
}

// SAFETY: WASM is single-threaded.
unsafe impl Send for WasmFalProvider {}
unsafe impl Sync for WasmFalProvider {}

#[wasm_bindgen(js_class = "FalProvider")]
impl WasmFalProvider {
    /// Create a new fal.ai provider.
    ///
    /// `options` is an optional plain JS object with:
    /// - `apiKey` (string) -- defaults to `FAL_KEY` env var
    /// - `model` (string) -- override the default LLM model
    /// - `baseUrl` (string) -- override the API base URL
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<WasmFalProvider, JsValue> {
        let api_key_opt = read_string(&options, "apiKey");
        let model_opt = read_string(&options, "model");
        let base_url_opt = read_string(&options, "baseUrl");

        let api_key = resolve_key("fal", api_key_opt)?;
        let mut provider = FalProvider::new_with_client(api_key, fetch_client());
        if let Some(m) = model_opt {
            provider = provider.with_model(m);
        }
        if let Some(url) = base_url_opt {
            provider = provider.with_base_url(url);
        }
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// The default LLM model identifier for this provider instance.
    #[wasm_bindgen(getter, js_name = "modelId")]
    pub fn model_id(&self) -> String {
        self.inner.model_id().to_owned()
    }

    /// Convert into a generic [`CompletionModel`].
    #[wasm_bindgen(js_name = "toCompletionModel")]
    pub fn to_completion_model(&self) -> WasmCompletionModel {
        WasmCompletionModel::from_arc(as_dyn_completion(Arc::clone(&self.inner)))
    }

    // -----------------------------------------------------------------------
    // LLM completion
    // -----------------------------------------------------------------------

    /// Perform a non-streaming chat completion.
    #[wasm_bindgen]
    pub fn complete(&self, messages: JsValue) -> js_sys::Promise {
        complete_promise(
            as_dyn_completion(Arc::clone(&self.inner)),
            messages,
        )
    }

    /// Perform a non-streaming completion with additional options.
    #[wasm_bindgen(js_name = "completeWithOptions")]
    pub fn complete_with_options(
        &self,
        messages: JsValue,
        options: JsValue,
    ) -> js_sys::Promise {
        let model = as_dyn_completion(Arc::clone(&self.inner));
        future_to_promise(async move {
            let msgs = crate::chat_message::js_messages_to_vec(&messages)?;
            let request = blazen_llm::types::CompletionRequest::new(msgs);
            let request = apply_request_options(request, options)?;
            let response = model
                .complete(request)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&response)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Perform a streaming chat completion.
    #[wasm_bindgen]
    pub fn stream(&self, messages: JsValue, callback: js_sys::Function) -> js_sys::Promise {
        stream_promise(
            as_dyn_completion(Arc::clone(&self.inner)),
            messages,
            callback,
        )
    }

    // -----------------------------------------------------------------------
    // Image generation
    // -----------------------------------------------------------------------

    /// Generate an image from a text prompt.
    ///
    /// `request` is a plain JS object shaped like
    /// [`blazen_llm::compute::ImageRequest`].
    #[wasm_bindgen(js_name = "generateImage")]
    pub fn generate_image(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::ImageRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = ImageGeneration::generate_image(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Upscale an image.
    #[wasm_bindgen(js_name = "upscaleImage")]
    pub fn upscale_image(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::UpscaleRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = ImageGeneration::upscale_image(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    // -----------------------------------------------------------------------
    // Video generation
    // -----------------------------------------------------------------------

    /// Generate a video from a text prompt.
    #[wasm_bindgen(js_name = "textToVideo")]
    pub fn text_to_video(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::VideoRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = VideoGeneration::text_to_video(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Generate a video from a source image.
    #[wasm_bindgen(js_name = "imageToVideo")]
    pub fn image_to_video(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::VideoRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = VideoGeneration::image_to_video(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    // -----------------------------------------------------------------------
    // Audio generation
    // -----------------------------------------------------------------------

    /// Synthesize speech from text.
    #[wasm_bindgen(js_name = "textToSpeech")]
    pub fn text_to_speech(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::SpeechRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = AudioGeneration::text_to_speech(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Generate music from a text prompt.
    #[wasm_bindgen(js_name = "generateMusic")]
    pub fn generate_music(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::MusicRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = AudioGeneration::generate_music(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Generate a sound effect from a text prompt.
    #[wasm_bindgen(js_name = "generateSfx")]
    pub fn generate_sfx(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::MusicRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = AudioGeneration::generate_sfx(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    // -----------------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------------

    /// Transcribe audio.
    #[wasm_bindgen]
    pub fn transcribe(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::TranscriptionRequest =
                serde_wasm_bindgen::from_value(request)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = Transcription::transcribe(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    // -----------------------------------------------------------------------
    // 3-D generation
    // -----------------------------------------------------------------------

    /// Generate a 3-D model.
    #[wasm_bindgen(js_name = "generate3d")]
    pub fn generate_3d(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::ThreeDRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = ThreeDGeneration::generate_3d(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    // -----------------------------------------------------------------------
    // Background removal
    // -----------------------------------------------------------------------

    /// Remove the background from an image.
    #[wasm_bindgen(js_name = "removeBackground")]
    pub fn remove_background(&self, request: JsValue) -> js_sys::Promise {
        let provider = Arc::clone(&self.inner);
        future_to_promise(async move {
            let req: blazen_llm::compute::BackgroundRemovalRequest =
                serde_wasm_bindgen::from_value(request)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let result = BackgroundRemoval::remove_background(provider.as_ref(), req)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }
}

fn read_string(obj: &JsValue, key: &str) -> Option<String> {
    if !obj.is_object() {
        return None;
    }
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}
