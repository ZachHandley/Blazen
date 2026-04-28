//! JavaScript bindings for the local whisper.cpp transcription provider.
//!
//! Exposes [`JsWhisperCppProvider`] as a NAPI class with an async factory
//! constructor, async `transcribe` method, and `LocalModel` lifecycle
//! controls (`load`, `unload`, `isLoaded`).
//!
//! Runs transcription entirely on-device using whisper.cpp. The first call
//! downloads the GGML model (tens to hundreds of MB depending on the chosen
//! variant) and caches it for subsequent runs. No API key is required.
//!
//! whisper.cpp currently expects 16-bit PCM mono WAV at 16 kHz. Remote URLs
//! are not supported -- pass a local file path in `request.audioUrl`.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::compute::Transcription;
use blazen_llm::traits::LocalModel;
use blazen_llm::{WhisperCppProvider, WhisperOptions};

use crate::error::blazen_error_to_napi;
use crate::generated::{JsTranscriptionRequest, JsTranscriptionResult};
use crate::providers::transcription::JsWhisperOptions;

// ---------------------------------------------------------------------------
// JsWhisperCppProvider NAPI class
// ---------------------------------------------------------------------------

/// A local whisper.cpp transcription provider.
///
/// `JsWhisperOptions` and `JsWhisperModel` are defined in
/// [`crate::providers::transcription`] -- they were already in place for
/// the [`crate::providers::transcription::JsTranscription::whispercpp`]
/// factory and are reused here unchanged.
///
/// ```javascript
/// const provider = await WhisperCppProvider.create({ model: "base" });
/// const result = await provider.transcribe({
///   audioUrl: "/path/to/audio.wav",
/// });
/// console.log(result.text);
/// ```
#[napi(js_name = "WhisperCppProvider")]
pub struct JsWhisperCppProvider {
    inner: Arc<WhisperCppProvider>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsWhisperCppProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new whisper.cpp provider.
    ///
    /// This is async because whisper.cpp may download the GGML model file
    /// from `HuggingFace` on first use.
    #[napi(factory)]
    pub async fn create(options: Option<JsWhisperOptions>) -> Result<Self> {
        let opts: WhisperOptions = options.map(Into::into).unwrap_or_default();
        let provider = WhisperCppProvider::from_options(opts)
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    // -----------------------------------------------------------------
    // Provider info
    // -----------------------------------------------------------------

    /// Get the provider identifier (`"whispercpp"`).
    #[napi(js_name = "providerId", getter)]
    pub fn provider_id(&self) -> String {
        use blazen_llm::compute::ComputeProvider;
        self.inner.provider_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Transcription
    // -----------------------------------------------------------------

    /// Transcribe an audio clip to text.
    ///
    /// Pass `audioUrl` as a local file path -- whisper.cpp does not fetch
    /// remote URLs. The audio must be 16-bit PCM mono WAV at 16 kHz.
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
    // LocalModel lifecycle
    // -----------------------------------------------------------------

    /// Explicitly load the GGML model into memory.
    #[napi]
    pub async fn load(&self) -> Result<()> {
        LocalModel::load(self.inner.as_ref())
            .await
            .map_err(blazen_error_to_napi)
    }

    /// Drop the loaded model and free its memory.
    #[napi]
    pub async fn unload(&self) -> Result<()> {
        LocalModel::unload(self.inner.as_ref())
            .await
            .map_err(blazen_error_to_napi)
    }

    /// Whether the model is currently loaded.
    #[napi(js_name = "isLoaded")]
    pub async fn is_loaded(&self) -> bool {
        LocalModel::is_loaded(self.inner.as_ref()).await
    }
}
