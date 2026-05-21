//! Generic and type-erased music-provider wrappers.
//!
//! [`MusicProvider`] is a thin generic newtype around a concrete
//! [`MusicBackend`] implementation. It exists so callers can keep static
//! dispatch when they know the engine type at compile time.
//!
//! [`DynMusicProvider`] is the `Arc<dyn MusicBackend>` shape used by the
//! manager / pipeline layer where engines are chosen at runtime.

use std::sync::Arc;

use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError, GeneratedAudio};

use crate::error::MusicError;
use crate::traits::MusicBackend;

/// Statically-dispatched wrapper around a concrete [`MusicBackend`].
///
/// Cheap to clone — the inner backend is held by value, so call sites that
/// need shared ownership should construct a [`DynMusicProvider`] (which
/// wraps the backend in an `Arc`) instead.
#[derive(Debug, Clone)]
pub struct MusicProvider<B: MusicBackend> {
    inner: B,
}

impl<B: MusicBackend> MusicProvider<B> {
    /// Wrap an existing backend.
    pub const fn new(backend: B) -> Self {
        Self { inner: backend }
    }

    /// Borrow the wrapped backend.
    pub const fn backend(&self) -> &B {
        &self.inner
    }

    /// Unwrap to the wrapped backend.
    pub fn into_inner(self) -> B {
        self.inner
    }

    /// Generate music. See [`MusicBackend::generate_music`].
    ///
    /// # Errors
    ///
    /// Forwards any [`MusicError`] returned by the wrapped backend.
    pub async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.inner.generate_music(prompt, duration_seconds).await
    }

    /// Generate a sound effect. See [`MusicBackend::generate_sfx`].
    ///
    /// # Errors
    ///
    /// Forwards any [`MusicError`] returned by the wrapped backend.
    pub async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.inner.generate_sfx(prompt, duration_seconds).await
    }
}

#[async_trait]
impl<B: MusicBackend> AudioBackend for MusicProvider<B> {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn provider_kind(&self) -> &str {
        self.inner.provider_kind()
    }

    async fn load(&self) -> Result<(), AudioError> {
        self.inner.load().await
    }

    async fn unload(&self) -> Result<(), AudioError> {
        self.inner.unload().await
    }

    async fn is_loaded(&self) -> bool {
        self.inner.is_loaded().await
    }
}

#[async_trait]
impl<B: MusicBackend> MusicBackend for MusicProvider<B> {
    async fn generate_music(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.inner.generate_music(prompt, duration_seconds).await
    }

    async fn generate_sfx(
        &self,
        prompt: &str,
        duration_seconds: f32,
    ) -> Result<GeneratedAudio, MusicError> {
        self.inner.generate_sfx(prompt, duration_seconds).await
    }
}

/// Type-erased music provider handle used by the manager / pipeline layer
/// when the concrete engine is selected at runtime.
pub type DynMusicProvider = Arc<dyn MusicBackend>;
