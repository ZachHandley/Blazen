//! Provider wrappers exposing [`crate::CodecBackend`] to in-process
//! callers and to the binding / bridge layer.
//!
//! See `PR_AUDIO_PLAN.md` Appendix B for the full dual-shape rationale.
//! Short version: Rust callers benefit from monomorphization through
//! [`CodecProvider<B>`]; bindings (Python / Node / UniFFI / cabi) can't
//! cross generics through their C ABI and need an erased
//! [`DynCodecProvider`] (`Arc<dyn CodecBackend>`) instead.

use std::sync::Arc;

use crate::error::CodecError;
use crate::traits::CodecBackend;

/// Typed codec provider — wraps a concrete [`CodecBackend`] implementation
/// and is monomorphized by the compiler. Use this from Rust callers in
/// hot loops.
#[derive(Clone, Debug)]
pub struct CodecProvider<B: CodecBackend> {
    backend: Arc<B>,
}

impl<B: CodecBackend> CodecProvider<B> {
    /// Wrap an existing backend.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Wrap an already-`Arc`'d backend (lets two providers share one
    /// loaded model instance).
    #[must_use]
    pub fn from_arc(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Borrow the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    /// Erase to a [`DynCodecProvider`] for the bindings boundary.
    #[must_use]
    pub fn into_dyn(self) -> DynCodecProvider
    where
        B: 'static,
    {
        DynCodecProvider {
            backend: self.backend as Arc<dyn CodecBackend>,
        }
    }

    /// Forward to [`CodecBackend::encode_pcm`].
    ///
    /// # Errors
    ///
    /// Propagates [`CodecError`] from the underlying backend.
    pub async fn encode_pcm(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u32>, CodecError> {
        self.backend.encode_pcm(samples, sample_rate).await
    }

    /// Forward to [`CodecBackend::decode_tokens`].
    ///
    /// # Errors
    ///
    /// Propagates [`CodecError`] from the underlying backend.
    pub async fn decode_tokens(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> Result<Vec<f32>, CodecError> {
        self.backend.decode_tokens(tokens, num_codebooks).await
    }
}

// ---------------------------------------------------------------------------
// Erased provider for the bindings boundary
// ---------------------------------------------------------------------------

/// Type-erased codec provider — wraps `Arc<dyn CodecBackend>`. Use this
/// from the Python / Node / UniFFI / cabi bridge layers where generic
/// providers can't cross the FFI boundary.
#[derive(Clone)]
pub struct DynCodecProvider {
    backend: Arc<dyn CodecBackend>,
}

impl DynCodecProvider {
    /// Wrap a pre-erased backend.
    #[must_use]
    pub fn new(backend: Arc<dyn CodecBackend>) -> Self {
        Self { backend }
    }

    /// Borrow the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &Arc<dyn CodecBackend> {
        &self.backend
    }

    /// Forward to [`CodecBackend::encode_pcm`].
    ///
    /// # Errors
    ///
    /// Propagates [`CodecError`] from the underlying backend.
    pub async fn encode_pcm(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<u32>, CodecError> {
        self.backend.encode_pcm(samples, sample_rate).await
    }

    /// Forward to [`CodecBackend::decode_tokens`].
    ///
    /// # Errors
    ///
    /// Propagates [`CodecError`] from the underlying backend.
    pub async fn decode_tokens(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> Result<Vec<f32>, CodecError> {
        self.backend.decode_tokens(tokens, num_codebooks).await
    }
}

impl std::fmt::Debug for DynCodecProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynCodecProvider")
            .field("backend_id", &self.backend.id())
            .field("provider_kind", &self.backend.provider_kind())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use blazen_audio::AudioBackend;

    struct FakeCodec;

    #[async_trait]
    impl AudioBackend for FakeCodec {
        fn id(&self) -> &'static str {
            "fake-codec"
        }
        fn provider_kind(&self) -> &'static str {
            "codec"
        }
    }

    #[async_trait]
    impl CodecBackend for FakeCodec {
        async fn encode_pcm(
            &self,
            samples: &[f32],
            _sample_rate: u32,
        ) -> Result<Vec<u32>, CodecError> {
            // Identity-ish encode: one token per sample.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Ok(samples.iter().map(|s| (s.abs() * 1000.0) as u32).collect())
        }

        async fn decode_tokens(
            &self,
            tokens: &[u32],
            num_codebooks: usize,
        ) -> Result<Vec<f32>, CodecError> {
            if !tokens.len().is_multiple_of(num_codebooks) {
                return Err(CodecError::invalid_input("misaligned"));
            }
            #[allow(clippy::cast_precision_loss)]
            Ok(tokens.iter().map(|&t| (t as f32) / 1000.0).collect())
        }

        fn num_codebooks(&self) -> usize {
            1
        }
    }

    #[tokio::test]
    async fn typed_provider_forwards_to_backend() {
        let provider = CodecProvider::new(FakeCodec);
        let tokens = provider.encode_pcm(&[0.1, 0.2, 0.3], 24_000).await.unwrap();
        assert_eq!(tokens.len(), 3);
        let pcm = provider.decode_tokens(&tokens, 1).await.unwrap();
        assert_eq!(pcm.len(), 3);
    }

    #[tokio::test]
    async fn dyn_provider_forwards_to_backend() {
        let dyn_provider = CodecProvider::new(FakeCodec).into_dyn();
        let tokens = dyn_provider
            .encode_pcm(&[0.5], 24_000)
            .await
            .expect("encode succeeds");
        assert_eq!(tokens, vec![500]);
    }

    #[tokio::test]
    async fn dyn_provider_debug_includes_id() {
        let dyn_provider = CodecProvider::new(FakeCodec).into_dyn();
        let dbg = format!("{dyn_provider:?}");
        assert!(dbg.contains("fake-codec"));
        assert!(dbg.contains("codec"));
    }
}
