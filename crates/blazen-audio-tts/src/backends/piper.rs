//! Piper ONNX TTS backend, powered by the vendored `piper-rs` fork in
//! `blazen-audio-piper-vendored`.
//!
//! Piper (<https://github.com/rhasspy/piper>) is a small, fast, fully
//! local TTS engine built on VITS. This backend wraps the patched
//! `piper-rs` (tract-onnx for inference, subprocess `espeak-ng` for
//! phonemization) behind the standard [`TtsBackend`] trait.
//!
//! # Runtime requirement
//!
//! Phonemization is delegated to the system `espeak-ng` binary at
//! synthesis time. The binary must be installed and on `PATH`:
//!
//! - Debian/Ubuntu: `apt install espeak-ng`
//! - macOS:         `brew install espeak-ng`
//! - Arch:          `pacman -S espeak-ng`
//!
//! # License posture
//!
//! The vendored `piper-rs` is MIT (upstream); Blazen ships it under the
//! workspace MPL-2.0. The runtime `espeak-ng` binary is GPL-3.0+, but
//! it is invoked over a process boundary (`tokio::process::Command`)
//! rather than linked, so no GPL inheritance flows into the Blazen
//! artifact. See `crates/blazen-audio-piper-vendored/VENDORED.md`.

#[cfg(feature = "piper")]
use std::path::PathBuf;

use async_trait::async_trait;
#[cfg(feature = "piper")]
use blazen_audio::AudioFormat;
use blazen_audio::{
    AudioBackend, CloneVoiceRequest, DesignVoiceRequest, GeneratedAudio, ListVoicesRequest,
    ListVoicesResponse, VoiceHandle,
};
#[cfg(feature = "piper")]
use blazen_audio::{VoiceDto, VoiceKind};

use crate::{TtsBackend, TtsError, TtsOptions};

/// Stable backend id, exposed at compile-time when `piper` is enabled
/// and at runtime via [`AudioBackend::id`].
pub const PIPER_BACKEND_ID: &str = "piper:vendored";

/// Piper ONNX TTS backend.
///
/// When the `piper` feature is **enabled**, this wraps a vendored
/// `piper-rs` voice and synthesizes WAV bytes. When **disabled**, every
/// method returns [`TtsError::EngineNotAvailable`] pointing at the
/// feature flag.
///
/// Construction goes through one of two paths:
///
/// - [`PiperBackend::new`] — zero-arg, returns an unloaded handle that
///   errors at synthesis time. Used by tests / catalog enumeration that
///   want a placeholder before a real voice file is available.
/// - [`PiperBackend::with_voice`] — loads a voice from `<voice>.onnx`
///   plus its sidecar `<voice>.onnx.json` config. This is the real
///   working backend.
#[cfg(feature = "piper")]
pub struct PiperBackend {
    /// Voice + tract plan, `None` when constructed via [`new`](Self::new).
    voice: Option<std::sync::Arc<blazen_audio_piper_vendored::Piper>>,
    model_path: Option<PathBuf>,
    config_path: Option<PathBuf>,
    /// Default Piper speaker id for multi-speaker voices. Used at
    /// `synthesize` time when [`TtsOptions::speaker_id`] is `None`.
    /// `None` here means "use Piper's default" (speaker 0 / single-speaker).
    default_speaker_id: Option<i64>,
}

#[cfg(feature = "piper")]
impl std::fmt::Debug for PiperBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PiperBackend")
            .field("model_path", &self.model_path)
            .field("config_path", &self.config_path)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "piper")]
impl Clone for PiperBackend {
    fn clone(&self) -> Self {
        Self {
            voice: self.voice.as_ref().map(std::sync::Arc::clone),
            model_path: self.model_path.clone(),
            config_path: self.config_path.clone(),
            default_speaker_id: self.default_speaker_id,
        }
    }
}

#[cfg(feature = "piper")]
impl Default for PiperBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "piper")]
impl PiperBackend {
    /// Zero-arg constructor returning an unloaded handle.
    ///
    /// Calls to [`synthesize`](TtsBackend::synthesize) on the unloaded
    /// handle return [`TtsError::ModelLoad`] with a hint to call
    /// [`with_voice`](Self::with_voice). Use this when you want a
    /// placeholder backend for tests, catalog enumeration, or
    /// just-in-time voice loading.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            voice: None,
            model_path: None,
            config_path: None,
            default_speaker_id: None,
        }
    }

    /// Load a Piper voice from a `<voice>.onnx` file and its sidecar
    /// `<voice>.onnx.json` config.
    ///
    /// `config_path` defaults to `<model_path>.json` when `None`.
    ///
    /// `default_speaker_id` is used when [`TtsOptions::speaker_id`] is
    /// `None` at synthesis time — typical for multi-speaker voices like
    /// `en_US-libritts_r-medium` (904 speakers). Pass `None` for
    /// single-speaker voices.
    ///
    /// # Errors
    ///
    /// - [`TtsError::ModelLoad`] if either file fails to open / parse,
    ///   or if the system `espeak-ng` binary is missing.
    pub fn with_voice(
        model_path: PathBuf,
        config_path: Option<PathBuf>,
        default_speaker_id: Option<i64>,
    ) -> Result<Self, TtsError> {
        let cfg = config_path.unwrap_or_else(|| {
            let mut p = model_path.clone().into_os_string();
            p.push(".json");
            PathBuf::from(p)
        });
        let voice =
            blazen_audio_piper_vendored::Piper::new(&model_path, &cfg).map_err(|e| match e {
                blazen_audio_piper_vendored::PiperError::EspeakNgMissing(msg) => {
                    TtsError::ModelLoad(format!("espeak-ng missing: {msg}"))
                }
                other => TtsError::ModelLoad(other.to_string()),
            })?;
        Ok(Self {
            voice: Some(std::sync::Arc::new(voice)),
            model_path: Some(model_path),
            config_path: Some(cfg),
            default_speaker_id,
        })
    }

    /// Stable backend id, exposed for compile-time match arms.
    #[must_use]
    pub const fn id_str() -> &'static str {
        PIPER_BACKEND_ID
    }

    /// Sample rate baked into the loaded voice. `None` if unloaded.
    #[must_use]
    pub fn sample_rate(&self) -> Option<u32> {
        self.voice.as_ref().map(|v| v.sample_rate())
    }
}

#[cfg(feature = "piper")]
#[async_trait]
impl AudioBackend for PiperBackend {
    fn id(&self) -> &'static str {
        PIPER_BACKEND_ID
    }

    fn provider_kind(&self) -> &'static str {
        "tts"
    }

    async fn is_loaded(&self) -> bool {
        self.voice.is_some()
    }
}

#[cfg(feature = "piper")]
#[async_trait]
impl TtsBackend for PiperBackend {
    async fn synthesize(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        let voice = self.voice.as_ref().ok_or_else(|| {
            TtsError::ModelLoad(
                "piper backend was constructed via `new()` without a voice file; \
                 call `PiperBackend::with_voice(path, config)` to load a `.onnx` voice"
                    .to_owned(),
            )
        })?;
        let voice = std::sync::Arc::clone(voice);
        let text_owned = text.to_owned();

        let sid = options
            .speaker_id
            .map(i64::from)
            .or(self.default_speaker_id);

        // Synthesis is CPU-bound (tract + subprocess wait); hop to the
        // blocking pool so we don't stall the async runtime.
        let (samples, sample_rate) = tokio::task::spawn_blocking(move || {
            voice.create(&text_owned, false, sid, None, None, None)
        })
        .await
        .map_err(|e| TtsError::Synthesis(format!("piper task panicked: {e}")))?
        .map_err(|e| TtsError::Synthesis(format!("piper synth failed: {e}")))?;

        let bytes = encode_wav_mono_f32(&samples, sample_rate);
        let duration_seconds = if sample_rate > 0 {
            #[allow(clippy::cast_precision_loss)]
            Some(samples.len() as f32 / sample_rate as f32)
        } else {
            None
        };

        Ok(GeneratedAudio {
            bytes,
            format: AudioFormat::Wav,
            sample_rate,
            channels: 1,
            duration_seconds,
        })
    }

    async fn list_voices(
        &self,
        _request: &ListVoicesRequest,
    ) -> Result<ListVoicesResponse, TtsError> {
        let Some(voice) = self.voice.as_ref() else {
            // Unloaded backend: nothing to enumerate.
            return Ok(ListVoicesResponse { voices: Vec::new() });
        };
        // Multi-speaker voices expose a `speaker_id_map` keyed by name;
        // single-speaker voices return `None` here, which collapses to
        // an empty Vec (the voice itself is identified by the .onnx
        // file path baked into `with_voice`, not by a sid name).
        let voices = voice
            .voices()
            .map(|m| {
                m.keys()
                    .map(|name| VoiceDto {
                        id: name.clone(),
                        name: name.clone(),
                        language: None,
                        kind: VoiceKind::Preset,
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(ListVoicesResponse { voices })
    }

    async fn clone_voice(&self, _request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(
            "piper voices are baked into the .onnx file; cloning is not supported".to_owned(),
        ))
    }

    async fn design_voice(&self, _request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::Unsupported(
            "piper voices are baked into the .onnx file; designing is not supported".to_owned(),
        ))
    }

    async fn delete_voice(&self, _voice_id: &str) -> Result<(), TtsError> {
        Err(TtsError::Unsupported(
            "piper voices are baked into the .onnx file; deletion is not supported".to_owned(),
        ))
    }
}

// ====================================================================
// Feature-disabled stub: matches the public surface so callers can
// reference `PiperBackend` without conditional compilation.
// ====================================================================

/// Reserved stub when the `piper` feature is disabled.
#[cfg(not(feature = "piper"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct PiperBackend;

#[cfg(not(feature = "piper"))]
impl PiperBackend {
    /// Construct the stub. Free of allocations.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Stable identifier for this backend.
    #[must_use]
    pub const fn id_str() -> &'static str {
        PIPER_BACKEND_ID
    }

    const DISABLED_MSG: &'static str =
        "blazen-audio-tts/piper feature is disabled; enable it to use the Piper backend";
}

#[cfg(not(feature = "piper"))]
#[async_trait]
impl AudioBackend for PiperBackend {
    fn id(&self) -> &'static str {
        PIPER_BACKEND_ID
    }

    fn provider_kind(&self) -> &'static str {
        "tts"
    }

    async fn is_loaded(&self) -> bool {
        false
    }
}

#[cfg(not(feature = "piper"))]
#[async_trait]
impl TtsBackend for PiperBackend {
    async fn synthesize(
        &self,
        _text: &str,
        _options: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        Err(TtsError::EngineNotAvailable(Self::DISABLED_MSG.to_owned()))
    }

    async fn list_voices(
        &self,
        _request: &ListVoicesRequest,
    ) -> Result<ListVoicesResponse, TtsError> {
        Err(TtsError::EngineNotAvailable(Self::DISABLED_MSG.to_owned()))
    }

    async fn clone_voice(&self, _request: CloneVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::EngineNotAvailable(Self::DISABLED_MSG.to_owned()))
    }

    async fn design_voice(&self, _request: DesignVoiceRequest) -> Result<VoiceHandle, TtsError> {
        Err(TtsError::EngineNotAvailable(Self::DISABLED_MSG.to_owned()))
    }

    async fn delete_voice(&self, _voice_id: &str) -> Result<(), TtsError> {
        Err(TtsError::EngineNotAvailable(Self::DISABLED_MSG.to_owned()))
    }
}

// ====================================================================
// WAV encoder — minimal RIFF/WAVE 16-bit PCM writer.
// ====================================================================

/// Encode mono f32 PCM samples (`[-1.0, 1.0]`) as a 16-bit RIFF/WAVE blob.
#[cfg(feature = "piper")]
fn encode_wav_mono_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let n_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let block_align = n_channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let data_size = u32::try_from(samples.len() * 2).unwrap_or(u32::MAX);
    let chunk_size = 36u32.saturating_add(data_size);

    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt subchunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&n_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        // Clamp then scale to i16 range.
        let clamped = s.clamp(-1.0, 1.0);
        #[allow(clippy::cast_possible_truncation)]
        let v = (clamped * f32::from(i16::MAX)) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "piper")]
    #[test]
    fn wav_header_round_trip() {
        let samples = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_mono_f32(&samples, 22_050);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // Each f32 sample → 2 bytes.
        assert_eq!(wav.len(), 44 + samples.len() * 2);
        // First sample is 0 → zero bytes.
        assert_eq!(&wav[44..46], &[0, 0]);
    }

    #[cfg(not(feature = "piper"))]
    #[tokio::test(flavor = "current_thread")]
    async fn disabled_backend_reports_feature_off() {
        let backend = PiperBackend::new();
        assert_eq!(backend.id(), PIPER_BACKEND_ID);
        assert_eq!(backend.provider_kind(), "tts");
        assert!(!backend.is_loaded().await);
        let err = backend
            .synthesize("hello", &TtsOptions::default())
            .await
            .expect_err("disabled stub should error");
        match err {
            TtsError::EngineNotAvailable(msg) => assert!(msg.contains("piper")),
            other => panic!("expected EngineNotAvailable, got {other:?}"),
        }
    }

    #[cfg(feature = "piper")]
    #[tokio::test(flavor = "current_thread")]
    async fn missing_model_file_is_model_load_error() {
        // Doesn't matter if espeak-ng is on PATH or not — both produce
        // ModelLoad (we squash EspeakNgMissing into ModelLoad).
        let result = PiperBackend::with_voice(
            std::path::PathBuf::from("/nonexistent/voice.onnx"),
            Some(std::path::PathBuf::from("/nonexistent/voice.onnx.json")),
            None,
        );
        match result {
            Err(TtsError::ModelLoad(_)) => {}
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[cfg(feature = "piper")]
    #[tokio::test(flavor = "current_thread")]
    async fn unloaded_backend_errors_on_synth() {
        let backend = PiperBackend::new();
        assert!(!backend.is_loaded().await);
        let err = backend
            .synthesize("hello", &TtsOptions::default())
            .await
            .expect_err("unloaded should fail");
        match err {
            TtsError::ModelLoad(msg) => assert!(msg.contains("with_voice")),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    // Live-models test: fetch the real Amy (en_US, low) voice from
    // `rhasspy/piper-voices` (~63 MB onnx + ~5 KB config) and synthesize
    // a short utterance. Gated because it requires network + ~63 MB of
    // model weights + the system `espeak-ng` binary on PATH at runtime.
    #[cfg(all(feature = "piper", feature = "live-models"))]
    #[tokio::test(flavor = "current_thread")]
    async fn live_synthesize_amy_low_voice() {
        // Use Blazen's ModelCache (already a workspace dep) instead of
        // raw hf-hub so the test doesn't drag in any new dev-deps and
        // honors $BLAZEN_CACHE_DIR like every other live-models test.
        let cache = blazen_model_cache::ModelCache::new().expect("model cache root");
        let onnx = cache
            .download(
                "rhasspy/piper-voices",
                "en/en_US/amy/low/en_US-amy-low.onnx",
                None,
            )
            .await
            .expect("fetch piper amy onnx weights");
        let cfg = cache
            .download(
                "rhasspy/piper-voices",
                "en/en_US/amy/low/en_US-amy-low.onnx.json",
                None,
            )
            .await
            .expect("fetch piper amy onnx sidecar config");

        // `with_voice` pre-flights espeak-ng. If the binary is missing,
        // skip the test cleanly rather than fail -- mirrors the snac
        // live test's "live deps absent = skip" philosophy.
        let backend = match PiperBackend::with_voice(onnx, Some(cfg), None) {
            Ok(b) => b,
            Err(TtsError::ModelLoad(msg)) if msg.contains("espeak-ng missing") => {
                eprintln!(
                    "live-models test skipped: espeak-ng binary not on PATH ({msg}). \
                     Install via `apt install espeak-ng` / `brew install espeak-ng`."
                );
                return;
            }
            Err(other) => panic!("unexpected load failure: {other:?}"),
        };
        assert!(backend.is_loaded().await, "backend should be loaded");

        let generated = backend
            .synthesize("Hello world from Blazen", &TtsOptions::default())
            .await
            .expect("synthesize Amy utterance");

        // WAV header sanity.
        assert!(
            generated.bytes.len() >= 44,
            "WAV output must be at least 44 bytes (header), got {}",
            generated.bytes.len()
        );
        assert_eq!(
            &generated.bytes[0..4],
            b"RIFF",
            "WAV must start with the RIFF magic"
        );
        assert_eq!(
            &generated.bytes[8..12],
            b"WAVE",
            "WAV must declare the WAVE format"
        );

        // Format metadata sanity.
        assert_eq!(generated.format, AudioFormat::Wav);
        assert_eq!(generated.channels, 1, "piper is mono");
        assert!(
            generated.sample_rate > 0,
            "sample rate must be set; got {}",
            generated.sample_rate
        );

        // PCM payload (everything after the 44-byte header) must contain
        // at least one nonzero byte -- guards against silent synthesis.
        let pcm = &generated.bytes[44..];
        assert!(
            pcm.iter().any(|&b| b != 0),
            "PCM payload must be nonzero (synthesized waveform shouldn't be pure silence)"
        );
    }
}
