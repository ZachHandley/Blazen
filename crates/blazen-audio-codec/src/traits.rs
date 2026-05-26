//! The [`CodecBackend`] trait every neural audio codec implements.
//!
//! Codecs are PCM <-> discrete-token translators. Concrete backends
//! (EnCodec, DAC, SNAC, ...) implement this single trait and are then
//! consumable through [`crate::CodecBackendHandle`] (typed) or
//! [`crate::DynCodecProvider`] (erased, for binding layers).

use async_trait::async_trait;
use blazen_audio::AudioBackend;

use crate::error::CodecError;

/// A neural audio codec.
///
/// Implementors translate mono `f32` PCM samples to and from discrete
/// codebook tokens. Both methods are async so backends are free to do
/// blocking model loads / GPU dispatch on a background thread without
/// poisoning the caller's runtime.
///
/// ## Trait shape
///
/// - Extends [`AudioBackend`] so codec backends share the same lifecycle
///   surface (`load` / `unload` / `is_loaded`) as TTS / STT / music
///   backends.
/// - `provider_kind()` should return `"codec"` for plain codecs;
///   multi-capability backends MAY return a hyphenated combination.
///
/// ## Token layout
///
/// `encode_pcm` returns and `decode_tokens` consumes a flat row-major
/// `[codebook_0_t0, codebook_0_t1, ..., codebook_1_t0, ...]` vector of
/// `u32`. The caller is expected to know its codebook count (e.g.
/// 4 codebooks for EnCodec at 6 kbps) and reshape accordingly.
#[async_trait]
pub trait CodecBackend: AudioBackend {
    /// Encode mono PCM samples (`f32` in `[-1.0, 1.0]`) into discrete
    /// codebook tokens.
    ///
    /// `sample_rate` must match the codec's native rate (e.g. 24 kHz for
    /// `encodec_24khz`); backends MUST return
    /// [`CodecError::InvalidInput`] on a mismatch rather than silently
    /// resampling.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InvalidInput`] when `samples` is empty or the
    ///   sample rate disagrees with the codec's native rate.
    /// - [`CodecError::EngineNotAvailable`] when the backend's feature
    ///   flag was disabled at build time.
    /// - [`CodecError::HfHub`] / [`CodecError::Io`] on first call if a
    ///   weight download is required and fails.
    /// - [`CodecError::Candle`] for inference / tensor failures.
    async fn encode_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<u32>, CodecError>;

    /// Decode flat row-major codebook tokens back to mono PCM samples.
    ///
    /// `tokens.len()` MUST be a positive multiple of `num_codebooks`;
    /// otherwise backends return [`CodecError::InvalidInput`].
    ///
    /// # Errors
    ///
    /// Same surface as [`Self::encode_pcm`].
    async fn decode_tokens(
        &self,
        tokens: &[u32],
        num_codebooks: usize,
    ) -> Result<Vec<f32>, CodecError>;

    /// Native sample rate (Hz) the codec produces on decode and expects
    /// on encode. Default: 24 kHz (EnCodec). Override for codecs whose
    /// native rate differs (DAC: 44.1 kHz / 24 kHz variants;
    /// SNAC: 24 kHz / 32 kHz / 44.1 kHz variants).
    fn sample_rate(&self) -> u32 {
        24_000
    }

    /// Number of quantizer codebooks the codec ships at its default
    /// bandwidth. EnCodec 24 kHz at 6 kbps = 8 codebooks (4 at 3 kbps),
    /// DAC at 8 kbps = 9 codebooks, SNAC = 3 codebooks. Backends should
    /// override; the default is `1` so calling against an un-overridden
    /// implementation surfaces an obvious shape error rather than
    /// silently encoding into a 1-codebook stream.
    fn num_codebooks(&self) -> usize {
        1
    }
}
