//! Typed and dynamically-erased TTS provider wrappers.
//!
//! [`TtsProvider<B>`] is the **monomorphized** wrapper: callers that
//! statically know which backend they want pay no virtual-call cost and
//! get the backend's full inherent surface back via `inner()`.
//!
//! [`DynTtsProvider`] is the **erased** wrapper around
//! `Box<dyn TtsBackend>` — used by the manager / pipeline layer when the
//! choice of backend is data-driven at runtime.

use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{
    AudioBackend, AudioError, CloneVoiceRequest, DesignVoiceRequest, GeneratedAudio,
    ListVoicesRequest, ListVoicesResponse, VoiceHandle,
};

use crate::{TtsBackend, TtsError, TtsOptions};

/// Typed, monomorphized TTS provider holding a concrete backend `B`.
///
/// Construct via [`TtsProvider::new`] when the backend type is known at
/// compile time (e.g. `TtsProvider::<AnyTtsBackend>::new(...)`). For
/// type-erased storage use [`DynTtsProvider`] instead.
#[derive(Debug, Clone)]
pub struct TtsProvider<B: TtsBackend> {
    backend: Arc<B>,
}

impl<B: TtsBackend> TtsProvider<B> {
    /// Wrap an already-constructed backend.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Wrap an already-`Arc`-wrapped backend (cheap clone path).
    #[must_use]
    pub fn from_arc(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Borrow the underlying backend.
    #[must_use]
    pub fn inner(&self) -> &B {
        &self.backend
    }

    /// Clone the inner `Arc<B>` for sharing across tasks.
    #[must_use]
    pub fn shared(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    /// Forward to [`TtsBackend::synthesize`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn synthesize(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        self.backend.synthesize(text, options).await
    }

    /// Forward to [`TtsBackend::list_voices`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn list_voices(
        &self,
        request: &ListVoicesRequest,
    ) -> Result<ListVoicesResponse, TtsError> {
        self.backend.list_voices(request).await
    }

    /// Forward to [`TtsBackend::clone_voice`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn clone_voice(&self, request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        self.backend.clone_voice(request).await
    }

    /// Forward to [`TtsBackend::design_voice`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn design_voice(&self, request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        self.backend.design_voice(request).await
    }

    /// Forward to [`TtsBackend::delete_voice`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn delete_voice(&self, voice_id: &str) -> Result<(), TtsError> {
        self.backend.delete_voice(voice_id).await
    }
}

/// Erased TTS provider — used by the manager / pipeline layer when the
/// backend is chosen at runtime.
pub struct DynTtsProvider {
    backend: Arc<dyn TtsBackend>,
}

impl std::fmt::Debug for DynTtsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTtsProvider")
            .field("id", &self.backend.id())
            .field("provider_kind", &self.backend.provider_kind())
            .finish()
    }
}

impl Clone for DynTtsProvider {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
        }
    }
}

impl DynTtsProvider {
    /// Wrap a boxed trait object.
    #[must_use]
    pub fn new(backend: Box<dyn TtsBackend>) -> Self {
        Self {
            backend: Arc::from(backend),
        }
    }

    /// Wrap an `Arc<dyn TtsBackend>` directly (zero-copy).
    #[must_use]
    pub fn from_arc(backend: Arc<dyn TtsBackend>) -> Self {
        Self { backend }
    }

    /// Build an erased provider from any concrete `TtsBackend`.
    #[must_use]
    pub fn erase<B: TtsBackend + 'static>(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// The wrapped backend's identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        self.backend.id()
    }

    /// The wrapped backend's capability kind.
    #[must_use]
    pub fn provider_kind(&self) -> &str {
        self.backend.provider_kind()
    }

    /// Forward to [`TtsBackend::synthesize`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn synthesize(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        self.backend.synthesize(text, options).await
    }

    /// Forward to [`TtsBackend::list_voices`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn list_voices(
        &self,
        request: &ListVoicesRequest,
    ) -> Result<ListVoicesResponse, TtsError> {
        self.backend.list_voices(request).await
    }

    /// Forward to [`TtsBackend::clone_voice`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn clone_voice(&self, request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        self.backend.clone_voice(request).await
    }

    /// Forward to [`TtsBackend::design_voice`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn design_voice(&self, request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        self.backend.design_voice(request).await
    }

    /// Forward to [`TtsBackend::delete_voice`].
    ///
    /// # Errors
    ///
    /// See [`TtsError`].
    pub async fn delete_voice(&self, voice_id: &str) -> Result<(), TtsError> {
        self.backend.delete_voice(voice_id).await
    }

    /// Forward to [`AudioBackend::load`].
    ///
    /// # Errors
    ///
    /// See [`AudioError`].
    pub async fn load(&self) -> Result<(), AudioError> {
        self.backend.load().await
    }

    /// Forward to [`AudioBackend::unload`].
    ///
    /// # Errors
    ///
    /// See [`AudioError`].
    pub async fn unload(&self) -> Result<(), AudioError> {
        self.backend.unload().await
    }

    /// Forward to [`AudioBackend::is_loaded`].
    pub async fn is_loaded(&self) -> bool {
        self.backend.is_loaded().await
    }
}

// Allow a `DynTtsProvider` to be used wherever an `AudioBackend` is expected.
#[async_trait]
impl AudioBackend for DynTtsProvider {
    fn id(&self) -> &str {
        self.backend.id()
    }

    fn provider_kind(&self) -> &str {
        self.backend.provider_kind()
    }

    async fn load(&self) -> Result<(), AudioError> {
        self.backend.load().await
    }

    async fn unload(&self) -> Result<(), AudioError> {
        self.backend.unload().await
    }

    async fn is_loaded(&self) -> bool {
        self.backend.is_loaded().await
    }
}

#[async_trait]
impl TtsBackend for DynTtsProvider {
    async fn synthesize(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        self.backend.synthesize(text, options).await
    }

    async fn list_voices(
        &self,
        request: &ListVoicesRequest,
    ) -> Result<ListVoicesResponse, TtsError> {
        self.backend.list_voices(request).await
    }

    async fn clone_voice(&self, request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        self.backend.clone_voice(request).await
    }

    async fn design_voice(&self, request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        self.backend.design_voice(request).await
    }

    async fn delete_voice(&self, voice_id: &str) -> Result<(), TtsError> {
        self.backend.delete_voice(voice_id).await
    }
}
