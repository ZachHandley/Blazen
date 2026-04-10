//! Node.js wrapper for the audio transcription provider abstraction.
//!
//! Provides [`JsTranscription`] with factory methods for each supported
//! transcription backend (fal.ai, whisper.cpp, etc.). Mirrors the pattern
//! used for [`JsEmbeddingModel`](crate::types::embedding::JsEmbeddingModel)
//! and [`JsCompletionModel`](crate::providers::JsCompletionModel).

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::compute::Transcription;

use crate::error::blazen_error_to_napi;
use crate::generated::{JsFalOptions, JsTranscriptionRequest, JsTranscriptionResult};

// ---------------------------------------------------------------------------
// Feature-gated whisper.cpp option mirror types (manual, not auto-generated)
// ---------------------------------------------------------------------------

/// Whisper model size variant for the local whisper.cpp backend.
///
/// Larger models are more accurate but require more memory and are slower.
///
/// | Variant  | Params | RAM   |
/// |----------|--------|-------|
/// | tiny     | 39M    | ~1GB  |
/// | base     | 74M    | ~1GB  |
/// | small    | 244M   | ~2GB  |
/// | medium   | 769M   | ~5GB  |
/// | largeV3  | 1.5B   | ~10GB |
#[cfg(feature = "whispercpp")]
#[napi(string_enum)]
pub enum JsWhisperModel {
    #[napi(value = "tiny")]
    Tiny,
    #[napi(value = "base")]
    Base,
    #[napi(value = "small")]
    Small,
    #[napi(value = "medium")]
    Medium,
    #[napi(value = "largeV3")]
    LargeV3,
}

#[cfg(feature = "whispercpp")]
impl From<JsWhisperModel> for blazen_llm::WhisperModel {
    fn from(m: JsWhisperModel) -> Self {
        match m {
            JsWhisperModel::Tiny => Self::Tiny,
            JsWhisperModel::Base => Self::Base,
            JsWhisperModel::Small => Self::Small,
            JsWhisperModel::Medium => Self::Medium,
            JsWhisperModel::LargeV3 => Self::LargeV3,
        }
    }
}

/// Options for the local whisper.cpp transcription backend.
///
/// All fields are optional. When `model` is omitted, defaults to
/// `JsWhisperModel::Small`. When `language` is omitted, whisper.cpp will
/// auto-detect the spoken language.
///
/// ```javascript
/// const transcriber = Transcription.whispercpp({
///   model: "base",
///   language: "en",
/// });
/// ```
#[cfg(feature = "whispercpp")]
#[napi(object)]
pub struct JsWhisperOptions {
    /// Whisper model size (defaults to `"small"`).
    pub model: Option<JsWhisperModel>,
    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub device: Option<String>,
    /// ISO 639-1 language code (e.g. `"en"`, `"es"`). When absent,
    /// whisper auto-detects the language.
    pub language: Option<String>,
    /// Enable speaker diarization. Currently unsupported by the whisper.cpp
    /// backend; setting `true` will cause transcription calls to fail.
    pub diarize: Option<bool>,
    /// Directory to cache downloaded models. When absent, falls back to
    /// `$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

#[cfg(feature = "whispercpp")]
impl From<JsWhisperOptions> for blazen_llm::WhisperOptions {
    fn from(val: JsWhisperOptions) -> Self {
        Self {
            model: val.model.map(Into::into).unwrap_or_default(),
            device: val.device,
            language: val.language,
            diarize: val.diarize,
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
        }
    }
}

// ---------------------------------------------------------------------------
// JsTranscription wrapper
// ---------------------------------------------------------------------------

/// An audio transcription provider.
///
/// Use the static factory methods to create a transcriber for a specific
/// backend, then call `transcribe` to convert audio to text.
///
/// ```javascript
/// // Local, offline transcription via whisper.cpp
/// const transcriber = Transcription.whispercpp({ model: "base" });
/// const result = await transcriber.transcribe({ audioUrl: "audio.wav" });
/// console.log(result.text);
///
/// // Remote transcription via fal.ai (requires API key)
/// const transcriber = Transcription.fal();
/// const result = await transcriber.transcribe({
///   audioUrl: "https://example.com/audio.mp3",
/// });
/// ```
#[napi(js_name = "Transcription")]
pub struct JsTranscription {
    pub(crate) inner: Arc<dyn Transcription>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsTranscription {
    // -----------------------------------------------------------------
    // Factory: fal.ai
    // -----------------------------------------------------------------

    /// Create a fal.ai transcription provider.
    ///
    /// Requires a fal.ai API key via `options.apiKey` or the `FAL_KEY`
    /// environment variable. Supports remote audio URLs.
    #[napi(factory)]
    pub fn fal(options: Option<JsFalOptions>) -> Result<Self> {
        let opts: blazen_llm::types::provider_options::FalOptions =
            options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                blazen_llm::providers::fal::FalProvider::from_options(opts)
                    .map_err(blazen_error_to_napi)?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Provider info
    // -----------------------------------------------------------------

    /// Get the provider identifier (e.g. `"fal"`, `"whispercpp"`).
    #[napi(js_name = "providerId", getter)]
    pub fn provider_id(&self) -> String {
        self.inner.provider_id().to_owned()
    }

    // -----------------------------------------------------------------
    // Transcribe
    // -----------------------------------------------------------------

    /// Transcribe an audio clip to text.
    ///
    /// For local backends like whisper.cpp, pass `audioUrl` as a local file
    /// path (whisper.cpp does not fetch remote URLs).
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
}

// ---------------------------------------------------------------------------
// Feature-gated whisper.cpp factory (separate impl block required by napi-derive)
// ---------------------------------------------------------------------------

#[cfg(feature = "whispercpp")]
#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsTranscription {
    /// Create a local whisper.cpp transcription provider.
    ///
    /// Runs transcription entirely on-device using whisper.cpp. The first
    /// call downloads the GGML model (tens to hundreds of MB depending on
    /// the chosen variant) and caches it for subsequent runs. No API key
    /// is required.
    ///
    /// whisper.cpp currently expects **16-bit PCM mono WAV at 16 kHz**.
    /// Remote URLs are not supported -- pass a local file path in
    /// `request.audioUrl`.
    ///
    /// ```javascript
    /// const transcriber = await Transcription.whispercpp({ model: "base" });
    /// const result = await transcriber.transcribe({
    ///   audioUrl: "/path/to/audio.wav",
    /// });
    /// console.log(result.text);
    /// ```
    #[napi(factory)]
    pub async fn whispercpp(options: Option<JsWhisperOptions>) -> Result<Self> {
        let opts: blazen_llm::WhisperOptions = options.map(Into::into).unwrap_or_default();
        let provider = blazen_llm::WhisperCppProvider::from_options(opts)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }
}
