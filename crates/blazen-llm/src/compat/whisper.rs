//! Backwards-compatibility shim for the deleted
//! `blazen_audio_whispercpp::WhisperCppProvider` type.
//!
//! Wraps a [`blazen_audio_stt::DynSttProvider`] (constructed from
//! [`blazen_audio_stt::backends::whispercpp::WhisperCppBackend`]) so the
//! pre-restructure factory and trait-impl surface (`from_options`,
//! `ComputeProvider`, `Transcription`, `LocalModel`) keeps compiling
//! without changing any binding code.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio_stt::backends::whispercpp::{WhisperCppBackend, WhisperCppOptions};
use blazen_audio_stt::{DynSttProvider, SttBackendHandle, SttError};

use crate::compute::{
    ComputeProvider, ComputeRequest, ComputeResult, JobHandle, JobStatus, Transcription,
    TranscriptionRequest, TranscriptionResult,
};
use crate::error::BlazenError;
use crate::traits::LocalModel;

/// Deprecated alias for the pre-restructure `WhisperCppProvider`. Wraps a
/// [`DynSttProvider`] holding a [`WhisperCppBackend`].
///
/// New code should construct a [`WhisperCppBackend`] directly via
/// [`WhisperCppBackend::new`] and erase it into a [`DynSttProvider`] (the
/// erased provider already implements [`ComputeProvider`], [`Transcription`]
/// and [`LocalModel`] via the bridge in `backends/audio_stt.rs`).
#[derive(Clone)]
pub struct WhisperCppProvider {
    inner: Arc<DynSttProvider>,
}

impl WhisperCppProvider {
    /// Construct a provider from [`WhisperCppOptions`]. Despite being
    /// `async` (to preserve the pre-restructure call shape) this never
    /// awaits — weight loading still happens lazily on the first
    /// transcribe call (or eagerly via [`LocalModel::load`]).
    ///
    /// # Errors
    ///
    /// Forwards any [`SttError`] returned by
    /// [`WhisperCppBackend::new`] (option validation only).
    #[allow(clippy::unused_async)] // Async preserved for pre-restructure call-shape compat.
    pub async fn from_options(options: WhisperCppOptions) -> Result<Self, SttError> {
        let backend = WhisperCppBackend::new(options)?;
        Ok(Self {
            inner: Arc::new(SttBackendHandle::new(backend).into_dyn()),
        })
    }

    /// Borrow the wrapped erased provider.
    #[must_use]
    pub fn inner(&self) -> &DynSttProvider {
        self.inner.as_ref()
    }

    /// Eager load. See [`LocalModel::load`].
    ///
    /// # Errors
    ///
    /// Forwards [`SttError`] from the underlying backend.
    pub async fn load(&self) -> Result<(), SttError> {
        DynSttProvider::load(self.inner.as_ref()).await
    }

    /// Drop the loaded weights. See [`LocalModel::unload`].
    ///
    /// # Errors
    ///
    /// Forwards [`SttError`] from the underlying backend.
    pub async fn unload(&self) -> Result<(), SttError> {
        DynSttProvider::unload(self.inner.as_ref()).await
    }

    /// Whether the underlying weights are loaded.
    pub async fn is_loaded(&self) -> bool {
        DynSttProvider::is_loaded(self.inner.as_ref()).await
    }

    /// Hardware device string (always `"cpu"` for the legacy compat
    /// shape — the new backend doesn't track this directly).
    #[must_use]
    pub fn device_str(&self) -> Option<&str> {
        Some("cpu")
    }

    /// Direct path-based transcribe — the pre-restructure shape used by
    /// the binding-layer factory. Forwards to the underlying
    /// [`DynSttProvider::transcribe`].
    ///
    /// # Errors
    ///
    /// Forwards [`SttError`] from the underlying backend.
    pub async fn transcribe(
        &self,
        path: &Path,
        language: Option<&str>,
    ) -> Result<blazen_audio_stt::TranscriptionResult, SttError> {
        DynSttProvider::transcribe(self.inner.as_ref(), path, language).await
    }
}

impl std::fmt::Debug for WhisperCppProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperCppProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Trait forwarders — keep the existing `Arc<WhisperCppProvider>` call sites
// compiling without changing binding code.
// ---------------------------------------------------------------------------

#[async_trait]
impl ComputeProvider for WhisperCppProvider {
    fn provider_id(&self) -> &str {
        ComputeProvider::provider_id(self.inner.as_ref())
    }

    async fn submit(&self, request: ComputeRequest) -> Result<JobHandle, BlazenError> {
        ComputeProvider::submit(self.inner.as_ref(), request).await
    }

    async fn status(&self, job: &JobHandle) -> Result<JobStatus, BlazenError> {
        ComputeProvider::status(self.inner.as_ref(), job).await
    }

    async fn result(&self, job: JobHandle) -> Result<ComputeResult, BlazenError> {
        ComputeProvider::result(self.inner.as_ref(), job).await
    }

    async fn cancel(&self, job: &JobHandle) -> Result<(), BlazenError> {
        ComputeProvider::cancel(self.inner.as_ref(), job).await
    }
}

#[async_trait]
impl Transcription for WhisperCppProvider {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, BlazenError> {
        Transcription::transcribe(self.inner.as_ref(), request).await
    }
}

#[async_trait]
impl LocalModel for WhisperCppProvider {
    async fn load(&self) -> Result<(), BlazenError> {
        LocalModel::load(self.inner.as_ref()).await
    }

    async fn unload(&self) -> Result<(), BlazenError> {
        LocalModel::unload(self.inner.as_ref()).await
    }

    async fn is_loaded(&self) -> bool {
        LocalModel::is_loaded(self.inner.as_ref()).await
    }

    fn device(&self) -> crate::device::Device {
        LocalModel::device(self.inner.as_ref())
    }

    async fn load_adapter(
        &self,
        adapter_dir: &std::path::Path,
        options: crate::AdapterOptions,
    ) -> Result<crate::AdapterHandle, BlazenError> {
        LocalModel::load_adapter(self.inner.as_ref(), adapter_dir, options).await
    }
}

// Convenience for `Arc<WhisperCppProvider>` patterns in the bindings.
impl From<WhisperCppProvider> for Arc<dyn Transcription> {
    fn from(p: WhisperCppProvider) -> Self {
        Arc::new(p)
    }
}
