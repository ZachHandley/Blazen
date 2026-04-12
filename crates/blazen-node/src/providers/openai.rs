//! JavaScript bindings for the `OpenAI` provider.
//!
//! Exposes [`JsOpenAiProvider`] as a NAPI class with a factory constructor and
//! a `textToSpeech` async method backed by `OpenAI`'s `/v1/audio/speech`
//! endpoint. For LLM chat completions use the [`crate::providers::completion_model::JsCompletionModel`]
//! entry point instead — this wrapper intentionally surfaces only the
//! standalone compute capabilities that are specific to the `OpenAI` provider
//! (i.e. text-to-speech). The trait defaults for `generateMusic` /
//! `generateSfx` fall through to `Err(Unsupported)` and are therefore not
//! re-exported here.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::compute::AudioGeneration;
use blazen_llm::providers::openai::OpenAiProvider;
use blazen_llm::types::provider_options::ProviderOptions;

use crate::error::blazen_error_to_napi;
use crate::generated::{JsAudioResult, JsProviderOptions, JsSpeechRequest};

// ---------------------------------------------------------------------------
// OpenAiProvider NAPI class
// ---------------------------------------------------------------------------

/// An `OpenAI` compute provider exposing text-to-speech.
///
/// For chat completions and embeddings, use
/// [`CompletionModel.openai`](crate::providers::completion_model::JsCompletionModel::openai)
/// instead — this class is the standalone entry point for the compute
/// capabilities (currently text-to-speech) that the `OpenAI` provider
/// implements directly.
///
/// ```typescript
/// const openai = OpenAiProvider.create({ apiKey: "sk-..." });
/// const audio = await openai.textToSpeech({
///     text: "Hello, world!",
///     voice: "alloy",
/// });
/// ```
#[napi(js_name = "OpenAiProvider")]
pub struct JsOpenAiProvider {
    inner: Arc<OpenAiProvider>,
}

#[napi]
#[allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value
)]
impl JsOpenAiProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new `OpenAI` provider.
    ///
    /// `options` optionally overrides the API key, model, and base URL.
    /// When omitted, the API key is read from the `OPENAI_API_KEY`
    /// environment variable and the defaults from
    /// [`OpenAiProvider::from_options`] are applied.
    #[napi(factory)]
    pub fn create(options: Option<JsProviderOptions>) -> Result<Self> {
        let opts: ProviderOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(OpenAiProvider::from_options(opts).map_err(blazen_error_to_napi)?),
        })
    }

    // -----------------------------------------------------------------
    // Compute methods -- Audio
    // -----------------------------------------------------------------

    /// Synthesize speech from text via `OpenAI`'s `/v1/audio/speech`.
    #[napi(js_name = "textToSpeech")]
    pub async fn text_to_speech(&self, request: JsSpeechRequest) -> Result<JsAudioResult> {
        let rust_req: blazen_llm::compute::SpeechRequest = request.into();
        let inner = Arc::clone(&self.inner);
        let result = AudioGeneration::text_to_speech(inner.as_ref(), rust_req)
            .await
            .map_err(blazen_error_to_napi)?;
        Ok(result.into())
    }
}
