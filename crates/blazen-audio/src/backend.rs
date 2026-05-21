//! The base [`AudioBackend`] trait every audio engine implements.
//!
//! Capability-specific traits (TTS, STT, music, codec, voice management)
//! live in the per-engine crates; they extend `AudioBackend` and are
//! object-safe so they can be stored as `Arc<dyn ...>` for dynamic dispatch
//! from the manager / pipeline layer.

use async_trait::async_trait;

use crate::error::AudioError;

/// The base trait shared by every audio engine in the Blazen ecosystem.
///
/// Implementors are expected to be cheap to clone (typically `Arc`-wrapped
/// internally) and to be safe to share across tokio tasks.
///
/// ## Default methods
///
/// [`load`](Self::load), [`unload`](Self::unload), and
/// [`is_loaded`](Self::is_loaded) ship with sensible defaults for
/// stateless backends (e.g. HTTP providers that simply forward requests).
/// Backends that hold model weights in memory should override all three.
#[async_trait]
pub trait AudioBackend: Send + Sync {
    /// Stable, machine-readable identifier for this backend instance —
    /// typically the engine name plus an instance discriminator (e.g.
    /// `"whispercpp:base.en"`, `"openai:tts-1-hd"`).
    fn id(&self) -> &str;

    /// Coarse capability classification used by the manager / pipeline
    /// when routing requests. One of: `"tts"`, `"stt"`, `"music"`,
    /// `"codec"`. Multi-capability backends MAY return a hyphenated
    /// combination (e.g. `"tts-stt"`).
    fn provider_kind(&self) -> &str;

    /// Load any model weights or open any network sessions. Idempotent —
    /// calling `load` on an already-loaded backend returns `Ok(())`.
    ///
    /// Default implementation is a no-op suitable for stateless HTTP
    /// providers.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Backend`] if model loading fails.
    async fn load(&self) -> Result<(), AudioError> {
        Ok(())
    }

    /// Release any in-memory model weights or persistent sessions.
    /// Idempotent — calling `unload` on an already-unloaded backend
    /// returns `Ok(())`.
    ///
    /// Default implementation is a no-op suitable for stateless backends.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Backend`] if cleanup fails.
    async fn unload(&self) -> Result<(), AudioError> {
        Ok(())
    }

    /// Whether [`load`](Self::load) has completed and the backend is ready
    /// to serve requests.
    ///
    /// Default implementation returns `true` (stateless backends are
    /// always ready). Override for backends with explicit load state.
    async fn is_loaded(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StatelessHttpBackend;

    #[async_trait]
    impl AudioBackend for StatelessHttpBackend {
        fn id(&self) -> &'static str {
            "stateless-http"
        }
        fn provider_kind(&self) -> &'static str {
            "tts"
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn default_lifecycle_methods_are_noops() {
        let backend = StatelessHttpBackend;
        assert_eq!(backend.id(), "stateless-http");
        assert_eq!(backend.provider_kind(), "tts");
        assert!(backend.is_loaded().await);
        backend.load().await.expect("default load is a no-op");
        backend.unload().await.expect("default unload is a no-op");
    }
}
