//! Local TTS backend powered by [`any-tts`](https://crates.io/crates/any-tts).
//!
//! Wraps the `any-tts` crate (Kokoro-82M, `VibeVoice`, Qwen3-TTS) behind
//! the [`TtsBackend`] trait so it can be stored alongside other backends
//! (`OpenAiTtsBackend`, future `PiperBackend`, …) in a
//! [`DynTtsProvider`](crate::DynTtsProvider).
//!
//! Gated by the `anytts` feature flag — without it, downstream code
//! cannot construct an `AnyTtsBackend` at all.

use std::sync::Arc;

use any_tts::{ModelType, SynthesisRequest, TtsConfig};
use async_trait::async_trait;
use blazen_audio::{AudioBackend, AudioError, AudioFormat, GeneratedAudio};
use tokio::sync::Mutex;

use crate::TtsError;
use crate::options::{TtsModel, TtsOptions};
use crate::traits::TtsBackend;

/// Local `any-tts` backend.
///
/// Lazy-loads the underlying engine on first
/// [`synthesize`](TtsBackend::synthesize) call and caches the loaded
/// model behind a `tokio::sync::Mutex`. Subsequent calls reuse the
/// cached model.
pub struct AnyTtsBackend {
    /// Default options used when synth callers pass an "empty" override
    /// set. Per-call options always win, but unspecified fields fall back
    /// to these.
    defaults: TtsOptions,
    /// Stable backend identifier — derived from the configured model.
    id: String,
    /// Lazy-loaded engine handle.
    model: Arc<Mutex<Option<Box<dyn any_tts::TtsModel + Send + Sync>>>>,
}

impl std::fmt::Debug for AnyTtsBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyTtsBackend")
            .field("id", &self.id)
            .field("defaults", &self.defaults)
            .finish_non_exhaustive()
    }
}

impl AnyTtsBackend {
    /// Construct a backend with default options
    /// ([`TtsOptions::default`] → Kokoro-82M).
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::InvalidOptions`] if a string field is present
    /// but empty.
    pub fn new() -> Result<Self, TtsError> {
        Self::from_options(TtsOptions::default())
    }

    /// Construct a backend with caller-supplied default options.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::InvalidOptions`] if a string field is present
    /// but empty or `sample_rate` is set to zero.
    pub fn from_options(opts: TtsOptions) -> Result<Self, TtsError> {
        validate_options(&opts)?;
        let id = format!("anytts:{}", opts.model.unwrap_or_default().as_str());
        Ok(Self {
            defaults: opts,
            id,
            model: Arc::new(Mutex::new(None)),
        })
    }

    /// The default options this backend was constructed with.
    #[must_use]
    pub fn defaults(&self) -> &TtsOptions {
        &self.defaults
    }

    /// The configured model kind.
    #[must_use]
    pub fn model_kind(&self) -> TtsModel {
        self.defaults.model.unwrap_or_default()
    }
}

fn validate_options(opts: &TtsOptions) -> Result<(), TtsError> {
    if let Some(voice) = opts.voice.as_ref()
        && voice.is_empty()
    {
        return Err(TtsError::InvalidOptions(
            "voice must not be empty when specified".into(),
        ));
    }
    if let Some(language) = opts.language.as_ref()
        && language.is_empty()
    {
        return Err(TtsError::InvalidOptions(
            "language must not be empty when specified".into(),
        ));
    }
    if let Some(rate) = opts.sample_rate
        && rate == 0
    {
        return Err(TtsError::InvalidOptions(
            "sample_rate must be > 0 when specified".into(),
        ));
    }
    Ok(())
}

fn merged_options(base: &TtsOptions, override_opts: &TtsOptions) -> TtsOptions {
    TtsOptions {
        model: override_opts.model.or(base.model),
        model_id: override_opts
            .model_id
            .clone()
            .or_else(|| base.model_id.clone()),
        voice: override_opts.voice.clone().or_else(|| base.voice.clone()),
        language: override_opts
            .language
            .clone()
            .or_else(|| base.language.clone()),
        sample_rate: override_opts.sample_rate.or(base.sample_rate),
        speed: override_opts.speed.or(base.speed),
        response_format: override_opts
            .response_format
            .clone()
            .or_else(|| base.response_format.clone()),
        cache_dir: override_opts
            .cache_dir
            .clone()
            .or_else(|| base.cache_dir.clone()),
        speaker_id: override_opts.speaker_id.or(base.speaker_id),
    }
}

#[async_trait]
impl AudioBackend for AnyTtsBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_kind(&self) -> &'static str {
        "tts"
    }

    async fn load(&self) -> Result<(), AudioError> {
        // Eagerly populate the cached model so subsequent synthesize
        // calls don't pay first-call latency.
        let model = Arc::clone(&self.model);
        let model_kind = self.model_kind();
        let cache_dir = self.defaults.cache_dir.clone();
        let result: Result<(), TtsError> = tokio::task::spawn_blocking(move || {
            let mut guard = model.blocking_lock();
            if guard.is_some() {
                return Ok(());
            }
            let model_type = match model_kind {
                TtsModel::Kokoro82m => ModelType::Kokoro,
                TtsModel::VibeVoice => ModelType::VibeVoice,
                TtsModel::Qwen3Tts => ModelType::Qwen3Tts,
            };
            let mut config = TtsConfig::new(model_type);
            if let Some(dir) = cache_dir.as_ref() {
                config = config.with_model_path(dir.to_string_lossy().into_owned());
            }
            let loaded = any_tts::load_model(config).map_err(TtsError::from)?;
            *guard = Some(loaded);
            Ok(())
        })
        .await
        .map_err(|e| AudioError::Backend(format!("anytts load task panicked: {e}")))?;
        result.map_err(AudioError::from)
    }

    async fn unload(&self) -> Result<(), AudioError> {
        let mut guard = self.model.lock().await;
        *guard = None;
        Ok(())
    }

    async fn is_loaded(&self) -> bool {
        self.model.lock().await.is_some()
    }
}

impl From<any_tts::TtsError> for TtsError {
    fn from(value: any_tts::TtsError) -> Self {
        Self::ModelLoad(value.to_string())
    }
}

#[async_trait]
impl TtsBackend for AnyTtsBackend {
    #[allow(clippy::cast_precision_loss)]
    async fn synthesize(
        &self,
        text: &str,
        override_opts: &TtsOptions,
    ) -> Result<GeneratedAudio, TtsError> {
        if text.is_empty() {
            return Err(TtsError::Synthesis(
                "synthesize called with empty text".into(),
            ));
        }
        let merged = merged_options(&self.defaults, override_opts);
        validate_options(&merged)?;

        let model_kind = merged.model.unwrap_or_default();
        let voice = merged
            .voice
            .clone()
            .or_else(|| model_kind.default_voice().map(str::to_string));
        let language = merged.language.clone();
        let cache_dir = merged.cache_dir.clone();

        let model = Arc::clone(&self.model);
        let text = text.to_owned();
        let audio = tokio::task::spawn_blocking(move || -> Result<_, TtsError> {
            let mut guard = model.blocking_lock();
            if guard.is_none() {
                let model_type = match model_kind {
                    TtsModel::Kokoro82m => ModelType::Kokoro,
                    TtsModel::VibeVoice => ModelType::VibeVoice,
                    TtsModel::Qwen3Tts => ModelType::Qwen3Tts,
                };
                let mut config = TtsConfig::new(model_type);
                if let Some(dir) = cache_dir.as_ref() {
                    config = config.with_model_path(dir.to_string_lossy().into_owned());
                }
                let loaded = any_tts::load_model(config)?;
                *guard = Some(loaded);
            }
            let mut req = SynthesisRequest::new(text);
            if let Some(v) = voice {
                req = req.with_voice(v);
            }
            if let Some(l) = language {
                req = req.with_language(l);
            }
            let model_ref = guard.as_ref().expect("just-loaded model is Some");
            let audio = model_ref
                .synthesize(&req)
                .map_err(|e| TtsError::Synthesis(e.to_string()))?;
            Ok(audio)
        })
        .await
        .map_err(|e| TtsError::Synthesis(format!("synthesis task panicked: {e}")))??;

        let sample_rate = audio.sample_rate;
        let duration_seconds = if sample_rate == 0 {
            None
        } else {
            Some(audio.samples.len() as f32 / sample_rate as f32)
        };
        let bytes = audio.get_wav();
        Ok(GeneratedAudio {
            bytes,
            format: AudioFormat::Wav,
            sample_rate,
            channels: 1,
            duration_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn from_options_with_defaults() {
        let backend = AnyTtsBackend::new().expect("should succeed");
        assert_eq!(backend.model_kind(), TtsModel::Kokoro82m);
        assert_eq!(backend.id(), "anytts:kokoro");
        assert_eq!(backend.provider_kind(), "tts");
    }

    #[tokio::test]
    async fn from_options_rejects_empty_voice() {
        let opts = TtsOptions {
            voice: Some(String::new()),
            ..TtsOptions::default()
        };
        assert!(AnyTtsBackend::from_options(opts).is_err());
    }

    #[tokio::test]
    async fn from_options_rejects_zero_sample_rate() {
        let opts = TtsOptions {
            sample_rate: Some(0),
            ..TtsOptions::default()
        };
        assert!(AnyTtsBackend::from_options(opts).is_err());
    }

    #[tokio::test]
    async fn synthesize_empty_text_rejected() {
        let backend = AnyTtsBackend::new().unwrap();
        let err = backend
            .synthesize("", &TtsOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, TtsError::Synthesis(_)));
    }

    /// Live, network-dependent smoke test: loads the real Kokoro-82M
    /// weights via `any-tts` (downloading on first run) and synthesizes a
    /// short sample using the default `af_bella` voice. Skipped
    /// automatically if model load fails for environmental reasons (no
    /// network, HF rate limit, missing onnx runtime, etc.) — failures of
    /// the synthesis itself still fail the test.
    ///
    /// Gated by `BLAZEN_TEST_KOKORO=1` because `any-tts` treats Kokoro
    /// preset voice packs (`voices/*.pt`) as *optional* assets — its
    /// loader fetches the base weights but does not download `af_bella.pt`
    /// or any other voice file on its own, so this test additionally
    /// requires the user (or a previous run with the env opt-in) to have
    /// populated the voice cache. Marked `#[ignore]` so the default
    /// `cargo nextest run --all-features` invocation skips it.
    ///
    ///     BLAZEN_TEST_KOKORO=1 cargo nextest run \
    ///         --features anytts,live-models \
    ///         -E 'test(live_synthesize_kokoro_default_voice)' \
    ///         --run-ignored only
    #[cfg(all(feature = "anytts", feature = "live-models"))]
    #[tokio::test]
    #[ignore = "requires BLAZEN_TEST_KOKORO=1 and a cached Kokoro voice pack (e.g. af_bella.pt)"]
    async fn live_synthesize_kokoro_default_voice() {
        if std::env::var("BLAZEN_TEST_KOKORO").ok().as_deref() != Some("1") {
            eprintln!("skipping: BLAZEN_TEST_KOKORO != 1");
            return;
        }
        let backend = AnyTtsBackend::new().expect("construct default backend");
        if let Err(e) = backend.load().await {
            eprintln!("skipping live_synthesize_kokoro_default_voice: {e}");
            return;
        }
        let audio = backend
            .synthesize(
                "Hello, world. This is Blazen testing the TTS pipeline.",
                &TtsOptions::default(),
            )
            .await
            .expect("synthesize should succeed once model is loaded");

        assert!(audio.bytes.len() > 44, "WAV must be larger than header");
        assert_eq!(&audio.bytes[0..4], b"RIFF", "missing RIFF magic");
        assert_eq!(&audio.bytes[8..12], b"WAVE", "missing WAVE marker");
        assert!(
            audio.bytes[44..].iter().any(|&b| b != 0),
            "PCM payload is silent"
        );
        assert_eq!(audio.format, AudioFormat::Wav);
        assert_eq!(audio.channels, 1);
        assert!(audio.sample_rate > 0, "sample_rate must be positive");
        let duration = audio
            .duration_seconds
            .expect("duration must be reported when sample_rate > 0");
        assert!(
            (1.0..=15.0).contains(&duration),
            "duration {duration}s outside tolerant 1-15s window"
        );
    }
}
