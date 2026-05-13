//! Universal `CustomProvider` -- the extension point for any provider.
//!
//! Supports two modes via the [`ApiProtocol`] selector:
//!
//! - **`OpenAi` protocol**: framework handles HTTP, SSE parsing, tool calls,
//!   retries. User supplies a base URL and model. Used for Ollama, LM Studio,
//!   llama.cpp's server, vLLM, TGI, any `OpenAI`-compatible endpoint.
//!
//! - **`Custom` protocol**: framework dispatches every method to a host-language
//!   object via [`HostDispatch`]. User implements `complete()`, `stream()`,
//!   media generation methods, etc. in their language (Python or JS).
//!
//! Both modes coexist on the same type, so a `CustomProvider` can speak `OpenAI`
//! for chat AND host-dispatch for media in one provider.
//!
//! ## Convenience constructors
//!
//! - [`CustomProvider::ollama`] / [`CustomProvider::lm_studio`] -- one-liner
//!   for the two most common local-model servers. Build an `OpenAi`-protocol
//!   `CustomProvider` with the right base URL.
//! - [`CustomProvider::openai_compat`] -- arbitrary `OpenAI`-compatible server.
//! - [`CustomProvider::with_dispatch`] -- subclass-style; host implements
//!   everything.
//!
//! Adding a new capability method to a trait like [`AudioGeneration`] in
//! `crate::compute::traits` automatically works for custom providers -- the
//! impl below just adds a one-line forward through `call_typed`. The language
//! bindings do not need to be updated as long as the host-side object has a
//! matching method.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use serde::{Serialize, de::DeserializeOwned};

use super::openai_compat::{AuthMethod, OpenAiCompatConfig, OpenAiCompatProvider};
use crate::compute::job::{ComputeRequest, ComputeResult, JobHandle, JobStatus};
use crate::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest, VoiceCloneRequest,
};
use crate::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, VideoResult, VoiceHandle,
};
use crate::compute::traits::{
    AudioGeneration, BackgroundRemoval, ComputeProvider, ImageGeneration, ThreeDGeneration,
    Transcription, VideoGeneration, VoiceCloning,
};
use crate::error::BlazenError;
use crate::http::HttpClient;
use crate::retry::RetryConfig;
use crate::traits::CompletionModel;
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

// ---------------------------------------------------------------------------
// ApiProtocol
// ---------------------------------------------------------------------------

/// Selects how a [`CustomProvider`] talks to its backend for completion calls.
///
/// Media-generation calls (audio, image, video, etc.) always go through the
/// optional [`HostDispatch`] regardless of which protocol is selected here.
#[derive(Debug, Clone)]
pub enum ApiProtocol {
    /// `OpenAI` Chat Completions wire format. Framework handles request body,
    /// SSE parsing, tool-call serialization, and retries. The wrapped config
    /// supplies the base URL, model, optional API key, and headers.
    OpenAi(OpenAiCompatConfig),

    /// User-defined protocol. The framework dispatches every completion
    /// method to a host-language object through [`HostDispatch`]. The host
    /// is expected to implement `complete` (and optionally `stream`).
    Custom,
    // Future: Anthropic(AnthropicConfig)
}

// ---------------------------------------------------------------------------
// HostDispatch trait
// ---------------------------------------------------------------------------

/// Invokes a named method on a host-language object with a JSON request and
/// returns a JSON response.
///
/// Implemented by language-specific shims:
/// - `blazen_py::providers::custom::PyHostDispatch` wraps `Py<PyAny>`.
/// - `blazen_node::providers::custom::NodeHostDispatch` wraps a JS object
///   reference behind a `ThreadsafeFunction`.
///
/// The dispatch layer is JSON-based so adding a new capability method doesn't
/// require language-specific plumbing -- as long as the host object has a
/// method by the corresponding name, everything works.
///
/// ## Method-name conventions
///
/// [`HostDispatch::call`] is invoked with the **Rust** method name
/// (`snake_case`). Language-specific shims are expected to translate to the
/// host language's naming convention (`text_to_speech` -> `textToSpeech` for
/// Node; `text_to_speech` verbatim for Python).
#[async_trait]
pub trait HostDispatch: Send + Sync + 'static {
    /// Invoke a method on the host-language object with a JSON-serializable
    /// request. Returns a JSON value that will be deserialized into the
    /// capability method's return type.
    ///
    /// Must return `Err(BlazenError::Unsupported { .. })` if the host
    /// object has no method with the given name. Must NOT panic if the
    /// host method exists but raises an exception -- wrap the exception
    /// in a [`BlazenError::Provider`] and return it as an error.
    ///
    /// # Errors
    ///
    /// Returns any error raised by the host-language implementation, or
    /// [`BlazenError::Unsupported`] if the method does not exist.
    async fn call(
        &self,
        method: &str,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, BlazenError>;

    /// Whether the host-language object has a method with the given name.
    /// Used to fast-path missing capabilities without paying the cost of a
    /// full call that would fail immediately.
    fn has_method(&self, method: &str) -> bool;
}

// ---------------------------------------------------------------------------
// CustomProvider
// ---------------------------------------------------------------------------

/// Universal user-extensible provider. See module-level docs.
pub struct CustomProvider {
    provider_id: String,
    protocol: ApiProtocol,
    /// Built-in chat backend (populated for `ApiProtocol::OpenAi`).
    completion_backend: Option<Arc<dyn CompletionModel>>,
    /// Host-language dispatch (populated for `ApiProtocol::Custom`, and
    /// optional alongside `OpenAi` for media capabilities).
    dispatch: Option<Arc<dyn HostDispatch>>,
    /// Provider-level default retry config. Pipeline / workflow / step / call
    /// scopes can override this; if all are `None`, this is the fallback.
    retry_config: Option<Arc<RetryConfig>>,
}

impl std::fmt::Debug for CustomProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomProvider")
            .field("provider_id", &self.provider_id)
            .field("protocol", &self.protocol)
            .field("has_dispatch", &self.dispatch.is_some())
            .field("has_completion_backend", &self.completion_backend.is_some())
            .finish_non_exhaustive()
    }
}

impl Clone for CustomProvider {
    fn clone(&self) -> Self {
        Self {
            provider_id: self.provider_id.clone(),
            protocol: self.protocol.clone(),
            completion_backend: self.completion_backend.clone(),
            dispatch: self.dispatch.clone(),
            retry_config: self.retry_config.clone(),
        }
    }
}

impl CustomProvider {
    /// Build a provider that dispatches every method to a host-language object
    /// via [`HostDispatch`]. Use for the "subclass and implement everything
    /// yourself" workflow.
    ///
    /// `provider_id` is used purely for logging and the
    /// [`ComputeProvider::provider_id`] return value -- it should be something
    /// meaningful to the user (e.g. `"elevenlabs"`, `"my-ollama"`).
    pub fn with_dispatch(provider_id: impl Into<String>, dispatch: Arc<dyn HostDispatch>) -> Self {
        Self {
            provider_id: provider_id.into(),
            protocol: ApiProtocol::Custom,
            completion_backend: None,
            dispatch: Some(dispatch),
            retry_config: None,
        }
    }

    /// Build a provider that speaks the `OpenAI` Chat Completions protocol.
    ///
    /// Use for Ollama, LM Studio, vLLM, or any other `OpenAI`-compatible
    /// server. The supplied [`OpenAiCompatConfig`] determines base URL, model,
    /// auth method, and headers.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn openai_compat(provider_id: impl Into<String>, config: OpenAiCompatConfig) -> Self {
        let backend: Arc<dyn CompletionModel> = Arc::new(OpenAiCompatProvider::new(config.clone()));
        Self {
            provider_id: provider_id.into(),
            protocol: ApiProtocol::OpenAi(config),
            completion_backend: Some(backend),
            dispatch: None,
            retry_config: None,
        }
    }

    /// Build a provider with the `OpenAI` chat backend AND a host dispatch
    /// for media capabilities. Lets a Python/JS user wire up Ollama for chat
    /// PLUS their own custom image/audio generation in a single provider.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn openai_compat_with_dispatch(
        provider_id: impl Into<String>,
        config: OpenAiCompatConfig,
        dispatch: Arc<dyn HostDispatch>,
    ) -> Self {
        let backend: Arc<dyn CompletionModel> = Arc::new(OpenAiCompatProvider::new(config.clone()));
        Self {
            provider_id: provider_id.into(),
            protocol: ApiProtocol::OpenAi(config),
            completion_backend: Some(backend),
            dispatch: Some(dispatch),
            retry_config: None,
        }
    }

    /// Convenience constructor for an Ollama server.
    ///
    /// Equivalent to [`Self::openai_compat`] with `base_url =
    /// format!("http://{host}:{port}/v1")` and no API key.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use blazen_llm::providers::custom::CustomProvider;
    ///
    /// // Local Ollama
    /// let p = CustomProvider::ollama("localhost", 11434, "llama3.1");
    ///
    /// // Ollama on the LAN
    /// let p = CustomProvider::ollama("192.168.1.50", 11434, "llama3.1");
    /// ```
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn ollama(host: impl AsRef<str>, port: u16, model: impl Into<String>) -> Self {
        let config = OpenAiCompatConfig {
            provider_name: "ollama".into(),
            base_url: format!("http://{}:{port}/v1", host.as_ref()),
            api_key: String::new(),
            default_model: model.into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        };
        Self::openai_compat("ollama", config)
    }

    /// Convenience constructor for an LM Studio server.
    ///
    /// Equivalent to [`Self::openai_compat`] with `base_url =
    /// format!("http://{host}:{port}/v1")` and no API key. LM Studio's
    /// default port is `1234`.
    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        feature = "reqwest",
        target_os = "wasi"
    ))]
    #[must_use]
    pub fn lm_studio(host: impl AsRef<str>, port: u16, model: impl Into<String>) -> Self {
        let config = OpenAiCompatConfig {
            provider_name: "lm_studio".into(),
            base_url: format!("http://{}:{port}/v1", host.as_ref()),
            api_key: String::new(),
            default_model: model.into(),
            auth_method: AuthMethod::Bearer,
            extra_headers: Vec::new(),
            query_params: Vec::new(),
            supports_model_listing: true,
        };
        Self::openai_compat("lm_studio", config)
    }

    /// Set the provider-level default retry configuration.
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(Arc::new(config));
        self
    }

    /// Inherent accessor: the configured `provider_id`.
    #[must_use]
    pub fn provider_id_str(&self) -> &str {
        &self.provider_id
    }

    /// The protocol this provider speaks.
    #[must_use]
    pub fn protocol(&self) -> &ApiProtocol {
        &self.protocol
    }

    /// Escape hatch for the underlying HTTP client.
    ///
    /// Returns the `OpenAI`-backed HTTP client when `protocol` is `OpenAi`;
    /// `None` for `Custom` protocol (the host owns its own client).
    #[must_use]
    pub fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        self.completion_backend
            .as_ref()
            .and_then(|b| b.http_client())
    }

    /// Internal: serialize a typed request, dispatch to host, deserialize the
    /// response.
    async fn call_typed<Req, Res>(
        &self,
        method: &'static str,
        request: Req,
    ) -> Result<Res, BlazenError>
    where
        Req: Serialize + Send,
        Res: DeserializeOwned,
    {
        let dispatch = self.dispatch.as_ref().ok_or_else(|| {
            BlazenError::unsupported(format!(
                "method `{method}` requires a host dispatch but none is configured"
            ))
        })?;
        let request_json = serde_json::to_value(&request)
            .map_err(|e| BlazenError::Serialization(e.to_string()))?;
        let response_json = dispatch.call(method, request_json).await?;
        serde_json::from_value(response_json).map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Fast-path: return Unsupported immediately if the host has no method by
    /// the given name, or if no dispatch is configured at all.
    fn ensure_method(&self, method: &str) -> Result<(), BlazenError> {
        let dispatch = self.dispatch.as_ref().ok_or_else(|| {
            BlazenError::unsupported(format!(
                "method `{method}` requires a host dispatch but none is configured"
            ))
        })?;
        if dispatch.has_method(method) {
            Ok(())
        } else {
            Err(BlazenError::unsupported(format!(
                "custom provider does not implement method `{method}`"
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// CompletionModel impl
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionModel for CustomProvider {
    fn model_id(&self) -> &str {
        match &self.protocol {
            ApiProtocol::OpenAi(cfg) => &cfg.default_model,
            ApiProtocol::Custom => &self.provider_id,
        }
    }

    fn retry_config(&self) -> Option<&Arc<RetryConfig>> {
        self.retry_config.as_ref()
    }

    fn http_client(&self) -> Option<Arc<dyn HttpClient>> {
        Self::http_client(self)
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, BlazenError> {
        if let Some(backend) = &self.completion_backend {
            return backend.complete(request).await;
        }
        // Custom protocol: dispatch to host
        self.ensure_method("complete")?;
        self.call_typed("complete", request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, BlazenError>> + Send>>, BlazenError>
    {
        if let Some(backend) = &self.completion_backend {
            return backend.stream(request).await;
        }
        Err(BlazenError::unsupported(
            "stream() not implemented for Custom protocol; host-dispatch \
             streaming requires async-iterable bridging not yet available",
        ))
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider + media trait impls (host-dispatch only)
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for CustomProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    async fn submit(&self, request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        self.ensure_method("submit")?;
        self.call_typed("submit", request).await
    }

    async fn status(&self, job: &JobHandle) -> Result<JobStatus, BlazenError> {
        self.ensure_method("status")?;
        self.call_typed("status", job.clone()).await
    }

    async fn result(&self, job: JobHandle) -> Result<ComputeResult, BlazenError> {
        self.ensure_method("result")?;
        self.call_typed("result", job).await
    }

    async fn cancel(&self, job: &JobHandle) -> Result<(), BlazenError> {
        self.ensure_method("cancel")?;
        let _: serde_json::Value = self.call_typed("cancel", job.clone()).await?;
        Ok(())
    }
}

#[async_trait]
impl AudioGeneration for CustomProvider {
    async fn text_to_speech(&self, request: SpeechRequest) -> Result<AudioResult, BlazenError> {
        self.ensure_method("text_to_speech")?;
        self.call_typed("text_to_speech", request).await
    }

    async fn generate_music(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        self.ensure_method("generate_music")?;
        self.call_typed("generate_music", request).await
    }

    async fn generate_sfx(&self, request: MusicRequest) -> Result<AudioResult, BlazenError> {
        self.ensure_method("generate_sfx")?;
        self.call_typed("generate_sfx", request).await
    }
}

#[async_trait]
impl VoiceCloning for CustomProvider {
    async fn clone_voice(&self, request: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        self.ensure_method("clone_voice")?;
        self.call_typed("clone_voice", request).await
    }

    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        self.ensure_method("list_voices")?;
        self.call_typed("list_voices", serde_json::Value::Null)
            .await
    }

    async fn delete_voice(&self, voice: &VoiceHandle) -> Result<(), BlazenError> {
        self.ensure_method("delete_voice")?;
        let _: serde_json::Value = self.call_typed("delete_voice", voice.clone()).await?;
        Ok(())
    }
}

#[async_trait]
impl ImageGeneration for CustomProvider {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        self.ensure_method("generate_image")?;
        self.call_typed("generate_image", request).await
    }

    async fn upscale_image(&self, request: UpscaleRequest) -> Result<ImageResult, BlazenError> {
        self.ensure_method("upscale_image")?;
        self.call_typed("upscale_image", request).await
    }
}

#[async_trait]
impl VideoGeneration for CustomProvider {
    async fn text_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        self.ensure_method("text_to_video")?;
        self.call_typed("text_to_video", request).await
    }

    async fn image_to_video(&self, request: VideoRequest) -> Result<VideoResult, BlazenError> {
        self.ensure_method("image_to_video")?;
        self.call_typed("image_to_video", request).await
    }
}

#[async_trait]
impl Transcription for CustomProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        self.ensure_method("transcribe")?;
        self.call_typed("transcribe", request).await
    }
}

#[async_trait]
impl ThreeDGeneration for CustomProvider {
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        self.ensure_method("generate_3d")?;
        self.call_typed("generate_3d", request).await
    }
}

#[async_trait]
impl BackgroundRemoval for CustomProvider {
    async fn remove_background(
        &self,
        request: BackgroundRemovalRequest,
    ) -> Result<ImageResult, BlazenError> {
        self.ensure_method("remove_background")?;
        self.call_typed("remove_background", request).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock dispatch that records every call and returns canned responses.
    struct MockDispatch {
        methods: Vec<&'static str>,
        calls: Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl MockDispatch {
        fn new(methods: Vec<&'static str>) -> Self {
            Self {
                methods,
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl HostDispatch for MockDispatch {
        async fn call(
            &self,
            method: &str,
            request: serde_json::Value,
        ) -> Result<serde_json::Value, BlazenError> {
            self.calls
                .lock()
                .unwrap()
                .push((method.to_owned(), request));
            match method {
                "text_to_speech" => Ok(serde_json::json!({
                    "audio": [],
                    "timing": { "total_ms": 0, "queue_ms": null, "execution_ms": null },
                    "metadata": {}
                })),
                _ => Ok(serde_json::json!({})),
            }
        }

        fn has_method(&self, method: &str) -> bool {
            self.methods.contains(&method)
        }
    }

    #[tokio::test]
    async fn text_to_speech_dispatches_to_host() {
        let dispatch: Arc<dyn HostDispatch> = Arc::new(MockDispatch::new(vec!["text_to_speech"]));
        let provider = CustomProvider::with_dispatch("mock", dispatch);

        let result = provider
            .text_to_speech(SpeechRequest::new("hello world"))
            .await
            .unwrap();

        assert_eq!(result.audio.len(), 0);
    }

    #[tokio::test]
    async fn unsupported_method_returns_error() {
        let dispatch: Arc<dyn HostDispatch> = Arc::new(MockDispatch::new(vec![]));
        let provider = CustomProvider::with_dispatch("mock", dispatch);

        let err = provider
            .text_to_speech(SpeechRequest::new("hi"))
            .await
            .expect_err("expected unsupported error");

        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn provider_id_reflects_constructor_argument() {
        let dispatch: Arc<dyn HostDispatch> = Arc::new(MockDispatch::new(vec![]));
        let provider = CustomProvider::with_dispatch("my-elevenlabs", dispatch);
        assert_eq!(ComputeProvider::provider_id(&provider), "my-elevenlabs");
    }

    #[test]
    fn ollama_builds_v1_url() {
        let p = CustomProvider::ollama("192.168.1.50", 11434, "llama3.1");
        assert_eq!(p.model_id(), "llama3.1");
        assert_eq!(p.provider_id_str(), "ollama");
        match p.protocol() {
            ApiProtocol::OpenAi(cfg) => {
                assert_eq!(cfg.base_url, "http://192.168.1.50:11434/v1");
                assert_eq!(cfg.provider_name, "ollama");
            }
            ApiProtocol::Custom => panic!("expected OpenAi protocol"),
        }
    }

    #[test]
    fn lm_studio_builds_v1_url() {
        let p = CustomProvider::lm_studio("localhost", 1234, "qwen2.5-coder");
        assert_eq!(p.model_id(), "qwen2.5-coder");
        match p.protocol() {
            ApiProtocol::OpenAi(cfg) => {
                assert_eq!(cfg.base_url, "http://localhost:1234/v1");
                assert_eq!(cfg.provider_name, "lm_studio");
            }
            ApiProtocol::Custom => panic!("expected OpenAi protocol"),
        }
    }

    #[tokio::test]
    async fn custom_protocol_complete_dispatches_to_host() {
        // Mock dispatch that returns a minimal CompletionResponse.
        struct CompleteDispatch;
        #[async_trait]
        impl HostDispatch for CompleteDispatch {
            async fn call(
                &self,
                _method: &str,
                _request: serde_json::Value,
            ) -> Result<serde_json::Value, BlazenError> {
                Ok(serde_json::json!({
                    "content": "hello from host",
                    "tool_calls": [],
                    "finish_reason": "stop",
                    "model": "host-model",
                    "usage": null,
                    "citations": [],
                    "reasoning": null
                }))
            }
            fn has_method(&self, method: &str) -> bool {
                method == "complete"
            }
        }
        let dispatch: Arc<dyn HostDispatch> = Arc::new(CompleteDispatch);
        let provider = CustomProvider::with_dispatch("test", dispatch);
        let req = CompletionRequest::new(vec![]);
        let resp = provider.complete(req).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("hello from host"));
    }

    #[tokio::test]
    async fn custom_protocol_stream_returns_unsupported() {
        let dispatch: Arc<dyn HostDispatch> = Arc::new(MockDispatch::new(vec!["stream"]));
        let provider = CustomProvider::with_dispatch("test", dispatch);
        let req = CompletionRequest::new(vec![]);
        let res = provider.stream(req).await;
        let Err(err) = res else {
            panic!("expected error");
        };
        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }
}
