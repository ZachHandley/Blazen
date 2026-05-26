//! Dual-shape provider wrappers around an [`SttBackend`].
//!
//! See `Appendix B` of the PR-AUDIO plan for the rationale: a generic
//! [`SttBackendHandle<B>`] gives Rust callers monomorphized dispatch, while
//! [`DynSttProvider`] is the erased shape that the language bindings
//! (Python, Node, WASM, `UniFFI`, C-ABI) can pass across the ABI.

use std::path::Path;

use crate::error::SttError;
use crate::traits::{SttBackend, TranscriptionResult};

/// A typed, monomorphized wrapper around any [`SttBackend`].
///
/// Prefer this shape in Rust hot paths where the concrete backend type is
/// known at compile time. For ABI boundaries, use [`DynSttProvider`]
/// (obtainable from [`SttBackendHandle::into_dyn`]).
#[derive(Debug)]
pub struct SttBackendHandle<B: SttBackend> {
    backend: B,
}

impl<B: SttBackend> SttBackendHandle<B> {
    /// Wrap a concrete backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Borrow the inner backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Stable identifier of the wrapped backend (forwards to
    /// [`AudioBackend::id`](blazen_audio::AudioBackend::id)).
    #[must_use]
    pub fn id(&self) -> &str {
        self.backend.id()
    }

    /// Capability tag of the wrapped backend (forwards to
    /// [`AudioBackend::provider_kind`](blazen_audio::AudioBackend::provider_kind)).
    #[must_use]
    pub fn provider_kind(&self) -> &str {
        self.backend.provider_kind()
    }

    /// Load the wrapped backend.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] (mapped from
    /// [`blazen_audio::AudioError::Backend`]) on weight-load failure.
    pub async fn load(&self) -> Result<(), SttError> {
        self.backend
            .load()
            .await
            .map_err(|e| SttError::ModelLoad(e.to_string()))
    }

    /// Unload the wrapped backend.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] (mapped from
    /// [`blazen_audio::AudioError::Backend`]) on cleanup failure.
    pub async fn unload(&self) -> Result<(), SttError> {
        self.backend
            .unload()
            .await
            .map_err(|e| SttError::ModelLoad(e.to_string()))
    }

    /// Whether the wrapped backend is loaded and ready.
    pub async fn is_loaded(&self) -> bool {
        self.backend.is_loaded().await
    }

    /// Transcribe an audio file.
    ///
    /// # Errors
    ///
    /// Forwards any [`SttError`] returned by the underlying backend.
    pub async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        self.backend.transcribe(audio_path, language).await
    }
}

impl<B: SttBackend + 'static> SttBackendHandle<B> {
    /// Convert to an erased [`DynSttProvider`] suitable for binding /
    /// ABI boundaries that cannot carry generics.
    pub fn into_dyn(self) -> DynSttProvider {
        DynSttProvider {
            backend: Box::new(self.backend),
        }
    }
}

/// Erased version of [`SttBackendHandle`] for FFI / binding boundaries.
///
/// Holds a `Box<dyn SttBackend>` and forwards the same surface as the
/// typed [`SttBackendHandle`].
pub struct DynSttProvider {
    backend: Box<dyn SttBackend>,
}

impl std::fmt::Debug for DynSttProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynSttProvider")
            .field("id", &self.backend.id())
            .field("provider_kind", &self.backend.provider_kind())
            .finish()
    }
}

impl DynSttProvider {
    /// Wrap an already-boxed dynamic backend.
    #[must_use]
    pub fn from_boxed(backend: Box<dyn SttBackend>) -> Self {
        Self { backend }
    }

    /// Stable identifier of the wrapped backend.
    #[must_use]
    pub fn id(&self) -> &str {
        self.backend.id()
    }

    /// Capability tag of the wrapped backend.
    #[must_use]
    pub fn provider_kind(&self) -> &str {
        self.backend.provider_kind()
    }

    /// Load the wrapped backend.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] on weight-load failure.
    pub async fn load(&self) -> Result<(), SttError> {
        self.backend
            .load()
            .await
            .map_err(|e| SttError::ModelLoad(e.to_string()))
    }

    /// Unload the wrapped backend.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::ModelLoad`] on cleanup failure.
    pub async fn unload(&self) -> Result<(), SttError> {
        self.backend
            .unload()
            .await
            .map_err(|e| SttError::ModelLoad(e.to_string()))
    }

    /// Whether the wrapped backend is loaded and ready.
    pub async fn is_loaded(&self) -> bool {
        self.backend.is_loaded().await
    }

    /// Transcribe an audio file.
    ///
    /// # Errors
    ///
    /// Forwards any [`SttError`] returned by the underlying backend.
    pub async fn transcribe(
        &self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, SttError> {
        self.backend.transcribe(audio_path, language).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::TranscriptionSegment;
    use async_trait::async_trait;
    use blazen_audio::AudioBackend;

    /// A trivial in-memory `SttBackend` used only for typed/erased dispatch
    /// tests. Always returns the same canned [`TranscriptionResult`].
    struct StubBackend;

    #[async_trait]
    impl AudioBackend for StubBackend {
        fn id(&self) -> &'static str {
            "stub:stt"
        }
        fn provider_kind(&self) -> &'static str {
            "stt"
        }
    }

    #[async_trait]
    impl SttBackend for StubBackend {
        async fn transcribe(
            &self,
            _audio_path: &Path,
            _language: Option<&str>,
        ) -> Result<TranscriptionResult, SttError> {
            Ok(TranscriptionResult {
                text: "stub".into(),
                segments: vec![TranscriptionSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "stub".into(),
                }],
                language: Some("en".into()),
            })
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn typed_provider_forwards_id_and_kind() {
        let provider = SttBackendHandle::new(StubBackend);
        assert_eq!(provider.id(), "stub:stt");
        assert_eq!(provider.provider_kind(), "stt");
        assert!(provider.is_loaded().await);
        provider.load().await.expect("noop load");
        provider.unload().await.expect("noop unload");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn typed_provider_transcribes() {
        let provider = SttBackendHandle::new(StubBackend);
        let result = provider
            .transcribe(Path::new("/dev/null"), None)
            .await
            .expect("stub never fails");
        assert_eq!(result.text, "stub");
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.language.as_deref(), Some("en"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn into_dyn_preserves_behavior() {
        let dyn_provider = SttBackendHandle::new(StubBackend).into_dyn();
        assert_eq!(dyn_provider.id(), "stub:stt");
        assert_eq!(dyn_provider.provider_kind(), "stt");
        let result = dyn_provider
            .transcribe(Path::new("/dev/null"), Some("ja"))
            .await
            .expect("stub never fails");
        assert_eq!(result.text, "stub");
    }
}
