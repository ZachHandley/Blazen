//! VC per-engine `#[uniffi::Object]` providers — P4.2.vc.
//!
//! One concrete `<Engine>Provider` per voice-conversion backend. Each wraps
//! the upstream `blazen_llm::providers::concrete::vc::<Engine>Provider`
//! handle in an `Arc` so UniFFI's foreign-language bindgens emit a real
//! per-engine class (Kotlin `class RvcProvider`, Swift `class RvcProvider`,
//! Go `*RvcProvider`, Ruby `Blazen::RvcProvider`, etc.).
//!
//! The whole module is implicitly gated by the parent
//! `#[cfg(feature = "audio-vc")]` declaration in [`super`], because the
//! upstream `blazen_llm::providers::concrete::vc` module itself lives under
//! that same gate. Individual engines that require an additional native
//! feature (`audio-vc-rvc`) carry their own `#[cfg]` on top.
//!
//! ## Method signature convention
//!
//! The upstream [`blazen_llm::VcProvider`] capability trait takes a
//! [`VoiceCloneRequest`](blazen_llm::compute::requests::VoiceCloneRequest)
//! DTO whose `name` field doubles as the target voice id and whose
//! `reference_urls[0]` doubles as the source utterance path / URL (see the
//! upstream `concrete::vc` docstring). The FFI surface unpacks that DTO
//! into flat `(input_path, target_voice_id)` arguments to keep the
//! foreign-language call sites ergonomic — they don't need to construct a
//! [`VoiceCloneRequest`] just to invoke a one-shot conversion.

#![allow(dead_code, unused_imports)]

use std::sync::Arc;

use crate::compute_vc::{TargetVoice, VcResult};
use crate::errors::BlazenError;
use crate::runtime::runtime;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a `BlazenError::Provider` with the canonical sentinel shape used
/// across the binding surface (mirrors the helpers in
/// [`super::stt`] / [`super::music`]).
fn provider_err(kind: &str, provider: &str, err: impl std::fmt::Display) -> BlazenError {
    BlazenError::Provider {
        kind: kind.to_string(),
        message: err.to_string(),
        provider: Some(provider.to_string()),
        status: None,
        endpoint: None,
        request_id: None,
        detail: None,
        retry_after_ms: None,
    }
}

/// Build a [`VoiceCloneRequest`](blazen_llm::compute::requests::VoiceCloneRequest)
/// for a `convert_voice` call from the flat FFI-style arguments.
///
/// `target_voice_id` lands in `name`; `input_path` becomes the sole entry
/// of `reference_urls`. This matches the contract documented on
/// `blazen_llm::providers::concrete::vc` (the upstream concrete provider
/// module).
fn build_convert_request(
    input_path: String,
    target_voice_id: String,
) -> blazen_llm::compute::requests::VoiceCloneRequest {
    blazen_llm::compute::requests::VoiceCloneRequest::new(target_voice_id, vec![input_path])
}

/// Build a [`VoiceCloneRequest`](blazen_llm::compute::requests::VoiceCloneRequest)
/// for a `clone_voice` call.
///
/// `voice_id` lands in `name`; `reference_path` becomes the sole entry of
/// `reference_urls`.
fn build_clone_request(
    voice_id: String,
    reference_path: String,
) -> blazen_llm::compute::requests::VoiceCloneRequest {
    blazen_llm::compute::requests::VoiceCloneRequest::new(voice_id, vec![reference_path])
}

/// Convert the upstream [`blazen_llm::compute::AudioResult`] into the
/// FFI-shaped [`VcResult`] DTO. Decodes the base64 payload (when present)
/// into raw `bytes` and pulls the MIME / sample-rate / duration from the
/// first clip, falling back to sensible defaults when fields are missing.
fn audio_result_to_vc_result(result: blazen_llm::compute::AudioResult) -> VcResult {
    let (bytes, mime_type, sample_rate, duration_seconds) = result
        .audio
        .first()
        .map(|clip| {
            let mime = clip.media.media_type.mime().to_owned();
            let bytes = clip
                .media
                .base64
                .as_deref()
                .map(|s| {
                    use base64::Engine as _;
                    base64::engine::general_purpose::STANDARD
                        .decode(s)
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            let sr = clip.sample_rate.unwrap_or(0);
            let dur = clip.duration_seconds.unwrap_or(0.0);
            (bytes, mime, sr, dur)
        })
        .unwrap_or_else(|| (Vec::new(), String::new(), 0, 0.0));
    VcResult {
        bytes,
        mime_type,
        sample_rate,
        duration_seconds,
    }
}

/// Convert an upstream [`blazen_llm::compute::results::VoiceHandle`] list
/// into the FFI-surface [`TargetVoice`] list. `VoiceHandle` doesn't carry
/// a sample rate, so `sample_rate_hz` lands as zero — callers that need
/// a real rate should consult the native [`crate::compute_vc::VcModel`]
/// path which talks to the lower-level
/// [`blazen_llm::VoiceConversionBackend`] trait.
fn voice_handles_to_target_voices(
    voices: Vec<blazen_llm::compute::results::VoiceHandle>,
) -> Vec<TargetVoice> {
    voices
        .into_iter()
        .map(|v| TargetVoice {
            id: v.id,
            label: Some(v.name),
            sample_rate_hz: 0,
        })
        .collect()
}

// ===========================================================================
// RvcProvider — native Retrieval-based Voice Conversion
// ===========================================================================

/// Native Retrieval-based Voice Conversion (RVC) provider.
///
/// Wraps [`blazen_llm::providers::concrete::vc::RvcProvider`] which itself
/// wraps `blazen_llm::RvcBackend`. Loads target-voice weights lazily from
/// the directory pointed to by the `BLAZEN_RVC_VOICE_DIR` environment
/// variable on the first `convert_voice` call. Defaults the inference
/// device to CPU; foreign callers needing CUDA / Metal should use the
/// central [`crate::compute_vc::new_rvc_model`] factory which exposes a
/// `device` parameter.
#[cfg(feature = "audio-vc-rvc")]
#[derive(uniffi::Object)]
pub struct RvcProvider {
    inner: Arc<blazen_llm::providers::concrete::vc::RvcProvider>,
}

#[cfg(feature = "audio-vc-rvc")]
#[uniffi::export(async_runtime = "tokio")]
impl RvcProvider {
    /// Build a new RVC-backed provider with the default CPU backend.
    ///
    /// Target voices are loaded lazily from `$BLAZEN_RVC_VOICE_DIR/<voice_id>/`
    /// on the first `convert_voice` call.
    #[uniffi::constructor]
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(blazen_llm::providers::concrete::vc::RvcProvider::new()),
        })
    }

    /// Convert the source utterance at `input_path` into the registered
    /// target voice `target_voice_id`.
    ///
    /// `input_path` is a local filesystem path to the source audio
    /// (16-bit PCM mono WAV at the backend's expected source rate,
    /// typically 16 kHz). `target_voice_id` selects which voice profile
    /// under `$BLAZEN_RVC_VOICE_DIR` to render into.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "VcConvert", ... }` when
    /// the source path is unreadable, the voice profile is missing, or
    /// the conversion pipeline fails.
    pub async fn convert_voice(
        self: Arc<Self>,
        input_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError> {
        use blazen_llm::providers::capabilities::VcProvider as _;
        let request = build_convert_request(input_path, target_voice_id);
        let result = self
            .inner
            .convert_voice(request)
            .await
            .map_err(|e| provider_err("VcConvert", "rvc", e))?;
        Ok(audio_result_to_vc_result(result))
    }

    /// Register a new target voice from the reference utterance at
    /// `reference_path` under the backend-scoped identifier `voice_id`.
    ///
    /// After this returns, subsequent `convert_voice` calls can pass
    /// `voice_id` as their `target_voice_id`.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "VcClone", ... }` when
    /// the reference path is unreadable or registration fails.
    pub async fn clone_voice(
        self: Arc<Self>,
        voice_id: String,
        reference_path: String,
    ) -> Result<(), BlazenError> {
        use blazen_llm::providers::capabilities::VcProvider as _;
        let request = build_clone_request(voice_id, reference_path);
        self.inner
            .clone_voice(request)
            .await
            .map_err(|e| provider_err("VcClone", "rvc", e))?;
        Ok(())
    }

    /// List the target voices currently known to the backend.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "VcListVoices", ... }`
    /// when the underlying backend's listing call fails.
    pub async fn list_target_voices(
        self: Arc<Self>,
    ) -> Result<Vec<TargetVoice>, BlazenError> {
        use blazen_llm::providers::capabilities::VcProvider as _;
        let voices = self
            .inner
            .list_voices()
            .await
            .map_err(|e| provider_err("VcListVoices", "rvc", e))?;
        Ok(voice_handles_to_target_voices(voices))
    }
}

#[cfg(feature = "audio-vc-rvc")]
#[uniffi::export]
impl RvcProvider {
    /// Synchronous variant of [`convert_voice`](Self::convert_voice).
    pub fn convert_voice_blocking(
        self: Arc<Self>,
        input_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.convert_voice(input_path, target_voice_id).await })
    }

    /// Synchronous variant of [`clone_voice`](Self::clone_voice).
    pub fn clone_voice_blocking(
        self: Arc<Self>,
        voice_id: String,
        reference_path: String,
    ) -> Result<(), BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.clone_voice(voice_id, reference_path).await })
    }

    /// Synchronous variant of
    /// [`list_target_voices`](Self::list_target_voices).
    pub fn list_target_voices_blocking(
        self: Arc<Self>,
    ) -> Result<Vec<TargetVoice>, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.list_target_voices().await })
    }
}

// ===========================================================================
// FalVcProvider — fal.ai cloud voice conversion
// ===========================================================================

/// fal.ai-backed voice-conversion provider.
///
/// Wraps [`blazen_llm::providers::concrete::vc::FalVcProvider`] which
/// routes requests through fal.ai's compute queue against the model id
/// supplied at construction time or per call (the upstream provider reads
/// `request.parameters.model` when present, else falls back to its built-in
/// default fal VC endpoint).
///
/// Voice cloning and listing are not part of fal's stable cloud VC surface
/// today, so this FFI class only exposes `convert_voice` — the upstream
/// trait's default `Unsupported` impls for `clone_voice` / `list_voices`
/// would surface a sentinel error to foreign callers, which is worse UX
/// than the method just not existing.
#[derive(uniffi::Object)]
pub struct FalVcProvider {
    inner: Arc<blazen_llm::providers::concrete::vc::FalVcProvider>,
}

#[uniffi::export(async_runtime = "tokio")]
impl FalVcProvider {
    /// Build a fal.ai-backed VC provider with the given API key.
    ///
    /// `api_key` may be empty when the fal client resolves it from the
    /// `FAL_KEY` environment variable.
    #[uniffi::constructor]
    #[must_use]
    pub fn new(api_key: String) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(blazen_llm::providers::concrete::vc::FalVcProvider::new(
                api_key,
            )),
        })
    }

    /// Convert the source utterance at `input_path` (an `http(s)://` /
    /// `data:` URL reachable by fal's workers) into the target voice
    /// `target_voice_id`.
    ///
    /// Unlike the native [`RvcProvider`], fal requires a URL — local file
    /// paths won't work since fal's workers can't reach the caller's
    /// disk. Pass a presigned / public URL or a `data:` URI.
    ///
    /// # Errors
    ///
    /// Returns `BlazenError::Provider { kind: "VcConvert", ... }` on HTTP
    /// / fal queue / endpoint-shape errors.
    pub async fn convert_voice(
        self: Arc<Self>,
        input_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError> {
        use blazen_llm::providers::capabilities::VcProvider as _;
        let request = build_convert_request(input_path, target_voice_id);
        let result = self
            .inner
            .convert_voice(request)
            .await
            .map_err(|e| provider_err("VcConvert", "fal", e))?;
        Ok(audio_result_to_vc_result(result))
    }
}

#[uniffi::export]
impl FalVcProvider {
    /// Synchronous variant of [`convert_voice`](Self::convert_voice).
    pub fn convert_voice_blocking(
        self: Arc<Self>,
        input_path: String,
        target_voice_id: String,
    ) -> Result<VcResult, BlazenError> {
        let this = Arc::clone(&self);
        runtime().block_on(async move { this.convert_voice(input_path, target_voice_id).await })
    }
}
