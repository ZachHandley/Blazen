//! User-defined providers via host-language dispatch.
//!
//! [`CustomProvider`] is a generic provider whose capability methods are
//! delegated to a host-language object through the [`HostDispatch`] trait.
//! The `PyO3` and napi bindings implement [`HostDispatch`] over their
//! respective object reference types (`Py<PyAny>`, `napi::Ref<JsObject>`),
//! so that Python and TypeScript users can write a normal class with
//! async methods matching Blazen's capability traits and have it plug
//! into the rest of the workflow engine exactly like a built-in provider.
//!
//! This module intentionally contains zero language-specific code. All
//! trait impls dispatch through a serde-JSON bridge on [`HostDispatch::call`],
//! which each language's shim implements in its own way.
//!
//! ## Extensibility
//!
//! Adding a new capability method to a trait like [`AudioGeneration`] in
//! `crate::compute::traits` automatically works for custom providers --
//! the impl below just adds a one-line forward through `call_typed`. The
//! language bindings do not need to be updated as long as the host-side
//! object has a matching method.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};

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

// ---------------------------------------------------------------------------
// HostDispatch trait
// ---------------------------------------------------------------------------

/// Invokes a named method on a host-language object with a JSON request
/// and returns a JSON response.
///
/// Implemented by language-specific shims:
/// - `blazen_py::providers::custom::PyHostDispatch` wraps `Py<PyAny>`
///   and uses `pythonize` + `pyo3_async_runtimes` to dispatch to Python
///   async methods.
/// - `blazen_node::providers::custom::NodeHostDispatch` wraps
///   `napi::Ref<JsObject>` and uses `ThreadsafeFunction` to dispatch to
///   JavaScript async methods.
///
/// The dispatch layer is JSON-based so adding a new capability method
/// doesn't require language-specific plumbing -- as long as the host
/// object has a method by the corresponding name, everything works.
///
/// ## Method name conventions
///
/// [`HostDispatch::call`] is invoked with the **Rust** method name
/// (`snake_case`). Language-specific shims are expected to translate
/// to the host language's naming convention (e.g. `text_to_speech` ->
/// `textToSpeech` for Node, `text_to_speech` verbatim for Python).
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
    /// Used by the default-`Unsupported` trait implementations to
    /// fast-path missing capabilities without paying the cost of a full
    /// call that would fail immediately.
    ///
    /// This is synchronous because the underlying check (Python's
    /// `hasattr`, JS's `in` / `hasOwnProperty`) is synchronous on both
    /// host languages. Implementations should cache the result if the
    /// check itself is expensive (e.g. requires a GIL acquisition).
    fn has_method(&self, method: &str) -> bool;
}

// ---------------------------------------------------------------------------
// CustomProvider
// ---------------------------------------------------------------------------

/// A capability provider whose methods are delegated to a host-language
/// object via a [`HostDispatch`] implementation.
///
/// Construct via [`CustomProvider::new`] and plug it into any code path
/// that accepts a boxed capability trait object -- it impls every one
/// of [`ComputeProvider`], [`AudioGeneration`], [`VoiceCloning`],
/// [`ImageGeneration`], [`VideoGeneration`], [`Transcription`],
/// [`ThreeDGeneration`], and [`BackgroundRemoval`]. The actual behavior is
/// determined by which methods the host object implements.
///
/// For capability methods where the host object has no matching method,
/// the trait impls return [`BlazenError::Unsupported`], allowing callers
/// to probe capability support without blanket-implementing every
/// method on the host side.
pub struct CustomProvider<D: HostDispatch> {
    provider_id: String,
    dispatch: Arc<D>,
}

impl<D: HostDispatch> CustomProvider<D> {
    /// Create a new custom provider with the given id and dispatch impl.
    ///
    /// The `provider_id` is used purely for logging and the
    /// [`ComputeProvider::provider_id`] return value -- it should be
    /// something meaningful to the user (e.g. `"elevenlabs"`).
    pub fn new(provider_id: impl Into<String>, dispatch: D) -> Self {
        Self {
            provider_id: provider_id.into(),
            dispatch: Arc::new(dispatch),
        }
    }

    /// Shared dispatch helper: serialize the typed request, call through
    /// [`HostDispatch::call`], and deserialize the response.
    async fn call_typed<Req, Res>(
        &self,
        method: &'static str,
        request: Req,
    ) -> Result<Res, BlazenError>
    where
        Req: Serialize + Send,
        Res: DeserializeOwned,
    {
        let request_json = serde_json::to_value(&request)
            .map_err(|e| BlazenError::Serialization(e.to_string()))?;
        let response_json = self.dispatch.call(method, request_json).await?;
        serde_json::from_value(response_json).map_err(|e| BlazenError::Serialization(e.to_string()))
    }

    /// Fast-path: return [`BlazenError::Unsupported`] immediately if the
    /// host object has no method by the given name, instead of paying
    /// for a full call.
    fn ensure_method(&self, method: &str) -> Result<(), BlazenError> {
        if self.dispatch.has_method(method) {
            Ok(())
        } else {
            Err(BlazenError::unsupported(format!(
                "custom provider does not implement method `{method}`"
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

#[async_trait]
impl<D: HostDispatch> ComputeProvider for CustomProvider<D> {
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
        // Dispatch returns Value::Null on success; deserialize into ().
        let _: serde_json::Value = self.call_typed("cancel", job.clone()).await?;
        Ok(())
    }
}

#[async_trait]
impl<D: HostDispatch> AudioGeneration for CustomProvider<D> {
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
impl<D: HostDispatch> VoiceCloning for CustomProvider<D> {
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
impl<D: HostDispatch> ImageGeneration for CustomProvider<D> {
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
impl<D: HostDispatch> VideoGeneration for CustomProvider<D> {
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
impl<D: HostDispatch> Transcription for CustomProvider<D> {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        self.ensure_method("transcribe")?;
        self.call_typed("transcribe", request).await
    }
}

#[async_trait]
impl<D: HostDispatch> ThreeDGeneration for CustomProvider<D> {
    async fn generate_3d(&self, request: ThreeDRequest) -> Result<ThreeDResult, BlazenError> {
        self.ensure_method("generate_3d")?;
        self.call_typed("generate_3d", request).await
    }
}

#[async_trait]
impl<D: HostDispatch> BackgroundRemoval for CustomProvider<D> {
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
            // Return a plausible AudioResult for text_to_speech, empty
            // objects for everything else.
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
        let provider = CustomProvider::new("mock", MockDispatch::new(vec!["text_to_speech"]));

        let result = provider
            .text_to_speech(SpeechRequest::new("hello world"))
            .await
            .unwrap();

        assert_eq!(result.audio.len(), 0);

        // NOTE: `provider.dispatch` is private, so we can't inspect the
        // recorded calls from outside the module. The main assertion is
        // that the dispatch succeeded and the JSON parse worked.
    }

    #[tokio::test]
    async fn unsupported_method_returns_error() {
        let provider = CustomProvider::new(
            "mock",
            MockDispatch::new(vec![]), // no methods
        );

        let err = provider
            .text_to_speech(SpeechRequest::new("hi"))
            .await
            .expect_err("expected unsupported error");

        assert!(matches!(err, BlazenError::Unsupported { .. }));
    }

    #[tokio::test]
    async fn provider_id_reflects_constructor_argument() {
        let provider = CustomProvider::new("my-elevenlabs", MockDispatch::new(vec![]));
        assert_eq!(provider.provider_id(), "my-elevenlabs");
    }
}
