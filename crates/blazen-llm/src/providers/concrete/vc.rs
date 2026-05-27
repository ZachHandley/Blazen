//! VC (voice conversion) concrete provider classes — populated by P4.1.f-vc.
//!
//! Two concrete per-engine provider classes live here:
//!
//! - [`RvcProvider`] (feature `audio-vc-rvc`): wraps the native
//!   [`blazen_audio_vc::RvcBackend`] via the public
//!   [`blazen_llm::VoiceConversionBackend`] trait. Loads voice weights
//!   lazily from `$BLAZEN_RVC_VOICE_DIR/<voice_id>/`.
//! - [`FalVcProvider`]: wraps [`crate::providers::fal::FalProvider`]'s
//!   compute pipeline against any fal.ai voice-conversion endpoint
//!   (model id provided per request via `request.parameters.model`, or
//!   the default [`DEFAULT_FAL_VC_MODEL`]).
//!
//! Both classes implement the polymorphic
//! [`crate::providers::root::BaseProvider`] root plus the capability
//! sub-trait [`crate::providers::capabilities::VcProvider`]. They are
//! the consumer-facing surface that the napi-rs / `PyO3` / `UniFFI` /
//! cabi / Ruby / WASM bindings re-export verbatim.
//!
//! ## `VoiceCloneRequest` mapping
//!
//! [`VoiceCloneRequest`](crate::compute::requests::VoiceCloneRequest)
//! is the shared DTO for both conversion and cloning. For the
//! `convert_voice` operation:
//!
//! - `name` — interpreted as the target voice id to render into.
//! - `reference_urls[0]` — interpreted as the source utterance to
//!   convert (file path on disk for [`RvcProvider`], URL for
//!   [`FalVcProvider`]). The remaining entries are ignored.
//! - `parameters.model` — optional fal.ai endpoint override for
//!   [`FalVcProvider`].
//!
//! For the `clone_voice` operation:
//!
//! - `name` — desired backend-scoped id for the new voice.
//! - `reference_urls[0]` — clean reference utterance from the target
//!   speaker. Remaining entries are ignored by the current backends.

#![allow(dead_code, unused_imports)]

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};

use crate::compute::requests::VoiceCloneRequest;
use crate::compute::results::{AudioResult, VoiceHandle};
use crate::error::BlazenError;
use crate::media::{GeneratedAudio, MediaOutput, MediaType};
use crate::providers::capabilities::VcProvider;
use crate::providers::root::{BaseProvider, CapabilityKind, ProviderMetadata};
use crate::types::RequestTiming;

// ---------------------------------------------------------------------------
// RvcProvider — native Retrieval-based Voice Conversion
// ---------------------------------------------------------------------------

/// Default fal.ai voice-conversion endpoint used by [`FalVcProvider`]
/// when the caller does not override via `request.parameters.model`.
///
/// Picked to match the closest stable cloud VC offering on fal at the
/// time of writing; callers should pin their own model id in
/// production deployments via the request `parameters` map.
pub const DEFAULT_FAL_VC_MODEL: &str = "fal-ai/rvc";

/// Native Retrieval-based Voice Conversion provider.
///
/// Wraps any [`blazen_llm::VoiceConversionBackend`](crate::VoiceConversionBackend)
/// implementor (the only concrete today is
/// [`blazen_llm::RvcBackend`](crate::RvcBackend)) and exposes it
/// through the polymorphic [`VcProvider`] capability trait.
///
/// Construct via [`Self::new`] (defaults the inference device to CPU)
/// or [`Self::with_backend`] for a fully-configured backend (custom
/// device, custom voice directory, etc.).
#[cfg(feature = "audio-vc-rvc")]
pub struct RvcProvider {
    inner: Arc<dyn crate::VoiceConversionBackend>,
    metadata: ProviderMetadata,
}

#[cfg(feature = "audio-vc-rvc")]
impl RvcProvider {
    /// Build an RVC provider with a default CPU-backed
    /// [`crate::RvcBackend`].
    ///
    /// The backend reads target-voice weights lazily from the directory
    /// pointed to by the `BLAZEN_RVC_VOICE_DIR` environment variable on
    /// the first `convert_voice` call. To override that directory or
    /// pick a different device (CUDA / Metal), construct an
    /// [`crate::RvcBackend`] yourself and pass it to
    /// [`Self::with_backend`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_backend(Arc::new(crate::RvcBackend::default()))
    }

    /// Build an RVC provider from an already-configured backend.
    ///
    /// Use this when you need a non-CPU device or a non-default voice
    /// directory:
    ///
    /// ```ignore
    /// let backend = blazen_llm::RvcBackend::with_device(device);
    /// let provider = RvcProvider::with_backend(std::sync::Arc::new(backend));
    /// ```
    #[must_use]
    pub fn with_backend(inner: Arc<dyn crate::VoiceConversionBackend>) -> Self {
        let metadata = ProviderMetadata::new("rvc", CapabilityKind::Vc)
            .with_display_name("Retrieval-based Voice Conversion");
        Self { inner, metadata }
    }
}

#[cfg(feature = "audio-vc-rvc")]
impl Default for RvcProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "audio-vc-rvc")]
impl std::fmt::Debug for RvcProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvcProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "audio-vc-rvc")]
impl BaseProvider for RvcProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[cfg(feature = "audio-vc-rvc")]
#[async_trait]
impl VcProvider for RvcProvider {
    async fn convert_voice(&self, request: VoiceCloneRequest) -> Result<AudioResult, BlazenError> {
        use base64::Engine as _;

        // The shared `VoiceCloneRequest` DTO carries the target voice id
        // in `name` and the source utterance path in `reference_urls[0]`
        // (the trait contract documented in this module's docstring).
        let target_voice_id = request.name.clone();
        let source_path = request
            .reference_urls
            .first()
            .ok_or_else(|| {
                BlazenError::validation(
                    "RvcProvider.convert_voice requires at least one reference_urls entry \
                 (source utterance path)",
                )
            })?
            .clone();
        let path = std::path::PathBuf::from(&source_path);

        // Look up the voice up-front so we can stamp the correct sample
        // rate / duration metadata on the result. Backends that don't
        // implement listing fall back to zero (callers can re-parse the
        // WAV header themselves).
        let voices = self.inner.list_target_voices().await.unwrap_or_default();
        let sample_rate = voices
            .iter()
            .find(|v| v.id == target_voice_id)
            .map(|v| v.sample_rate_hz);

        let bytes = self
            .inner
            .convert_voice(&path, &target_voice_id)
            .await
            .map_err(|e| {
                BlazenError::provider(
                    self.metadata.provider_id.clone(),
                    format!("voice conversion failed: {e}"),
                )
            })?;

        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let media = MediaOutput::from_base64(encoded, MediaType::Wav);
        let clip = GeneratedAudio {
            media,
            duration_seconds: None,
            sample_rate,
            channels: Some(1),
        };

        Ok(AudioResult {
            audio: vec![clip],
            timing: RequestTiming {
                queue_ms: None,
                execution_ms: None,
                total_ms: None,
            },
            cost: None,
            usage: None,
            audio_seconds: 0.0,
            metadata: serde_json::json!({
                "provider": "rvc",
                "target_voice_id": target_voice_id,
            }),
        })
    }

    async fn clone_voice(&self, request: VoiceCloneRequest) -> Result<VoiceHandle, BlazenError> {
        let voice_id = request.name.clone();
        let reference = request
            .reference_urls
            .first()
            .ok_or_else(|| {
                BlazenError::validation(
                    "RvcProvider.clone_voice requires at least one reference_urls entry \
                 (reference utterance path)",
                )
            })?
            .clone();
        let path = std::path::PathBuf::from(&reference);

        self.inner
            .register_target_voice(&voice_id, &path)
            .await
            .map_err(|e| {
                BlazenError::provider(
                    self.metadata.provider_id.clone(),
                    format!("voice registration failed: {e}"),
                )
            })?;

        Ok(VoiceHandle {
            id: voice_id.clone(),
            name: voice_id,
            provider: self.metadata.provider_id.clone(),
            language: request.language,
            description: request.description,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        })
    }

    async fn list_voices(&self) -> Result<Vec<VoiceHandle>, BlazenError> {
        let voices = self.inner.list_target_voices().await.map_err(|e| {
            BlazenError::provider(
                self.metadata.provider_id.clone(),
                format!("listing target voices failed: {e}"),
            )
        })?;
        let provider_id = self.metadata.provider_id.clone();
        Ok(voices
            .into_iter()
            .map(|v| VoiceHandle {
                id: v.id.clone(),
                name: v.label.unwrap_or(v.id),
                provider: provider_id.clone(),
                language: None,
                description: None,
                metadata: serde_json::Value::Object(serde_json::Map::new()),
            })
            .collect())
    }

    async fn stream_convert_pcm(
        &self,
        input_pcm: Vec<f32>,
        target_voice_id: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<f32>, BlazenError>> + Send>>, BlazenError>
    {
        // The backend's `stream_convert` consumes a stream of source PCM
        // frames; wrap the single input buffer in a one-shot stream (the
        // capability surface ships one buffer per call, matching the
        // uniffi `stream_convert_pcm` adapter). Items yielded by the
        // backend stream carry `VcError`, which we map into `BlazenError`.
        let input_stream = futures_util::stream::once(async move { input_pcm });
        let out_stream = self
            .inner
            .stream_convert(Box::pin(input_stream), &target_voice_id)
            .await?;
        Ok(Box::pin(
            out_stream.map(|item| item.map_err(BlazenError::from)),
        ))
    }
}

// ---------------------------------------------------------------------------
// FalVcProvider — fal.ai cloud voice conversion
// ---------------------------------------------------------------------------

/// fal.ai-backed voice conversion provider.
///
/// Wraps a [`crate::providers::fal::FalProvider`] and routes
/// `convert_voice` calls through fal's generic compute queue against
/// the model id supplied via `request.parameters.model` (or
/// [`DEFAULT_FAL_VC_MODEL`] when unset).
///
/// Voice cloning and voice listing are not part of fal's stable cloud
/// surface today, so [`VcProvider::clone_voice`] / [`VcProvider::list_voices`]
/// inherit the trait's default `BlazenError::Unsupported` impls. Callers
/// can still register pre-built voices through fal directly by passing
/// the voice id via the request `parameters` map.
pub struct FalVcProvider {
    inner: Arc<crate::providers::fal::FalProvider>,
    metadata: ProviderMetadata,
}

impl FalVcProvider {
    /// Build a fal.ai VC provider from an API key.
    ///
    /// The default endpoint is [`DEFAULT_FAL_VC_MODEL`]. Per-call
    /// overrides go through `request.parameters.model`.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_inner(Arc::new(crate::providers::fal::FalProvider::new(api_key)))
    }

    /// Build a fal.ai VC provider from a pre-configured
    /// [`crate::providers::fal::FalProvider`] (custom HTTP client,
    /// custom retry/execution mode, etc.).
    #[must_use]
    pub fn with_inner(inner: Arc<crate::providers::fal::FalProvider>) -> Self {
        let metadata = ProviderMetadata::new("fal", CapabilityKind::Vc)
            .with_display_name("fal.ai Voice Conversion")
            .with_version(DEFAULT_FAL_VC_MODEL);
        Self { inner, metadata }
    }
}

impl std::fmt::Debug for FalVcProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FalVcProvider")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl BaseProvider for FalVcProvider {
    fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }
}

#[async_trait]
impl VcProvider for FalVcProvider {
    async fn convert_voice(&self, request: VoiceCloneRequest) -> Result<AudioResult, BlazenError> {
        // Pick the model: explicit per-call override via
        // `parameters.model`, else the default.
        let model = request
            .parameters
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_FAL_VC_MODEL)
            .to_owned();

        // Source audio: prefer the explicit `source_audio_url` parameter
        // (gives callers an escape hatch), else fall back to
        // `reference_urls[0]`.
        let source_audio = request
            .parameters
            .get("source_audio_url")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| request.reference_urls.first().cloned())
            .ok_or_else(|| {
                BlazenError::validation(
                    "FalVcProvider.convert_voice requires either parameters.source_audio_url \
                 or a reference_urls entry",
                )
            })?;

        let mut input = serde_json::json!({
            "audio_url": source_audio,
            "target_voice": request.name,
        });
        if let Some(ref language) = request.language {
            input["language"] = serde_json::json!(language);
        }
        // Merge caller-supplied parameters last so they can override the
        // canonical fields if a particular fal endpoint uses different
        // input key names. We exclude `model` since it's a routing
        // concern, not an endpoint input.
        if let serde_json::Value::Object(map) = &request.parameters
            && let serde_json::Value::Object(ref mut dst) = input
        {
            for (k, v) in map {
                if k == "model" || k == "source_audio_url" {
                    continue;
                }
                dst.insert(k.clone(), v.clone());
            }
        }

        let compute_req = crate::compute::job::ComputeRequest {
            model,
            input,
            webhook: None,
        };
        let result =
            <crate::providers::fal::FalProvider as crate::compute::traits::ComputeProvider>::run(
                self.inner.as_ref(),
                compute_req,
            )
            .await?;

        // fal's audio endpoints share a common output shape — re-use
        // the public [`parse_fal_audio`] helper isn't exported, so
        // we extract the audio URL ourselves with a minimal walker.
        // This mirrors what `parse_fal_audio` does, scoped to the
        // shapes fal emits for VC endpoints (`{audio:{url}}`,
        // `{audio_url}`, `{output:{audio:{url}}}`).
        let audio_url = extract_fal_audio_url(&result.output).ok_or_else(|| {
            BlazenError::provider(
                "fal",
                format!(
                    "fal VC response did not include an audio URL: {}",
                    result.output
                ),
            )
        })?;

        let media = MediaOutput::from_url(audio_url, MediaType::Wav);
        let clip = GeneratedAudio {
            media,
            duration_seconds: None,
            sample_rate: None,
            channels: None,
        };

        Ok(AudioResult {
            audio: vec![clip],
            timing: result.timing,
            cost: result.cost,
            usage: None,
            audio_seconds: 0.0,
            metadata: result.metadata,
        })
    }
}

/// Minimal walker that extracts an audio URL from fal's voice-conversion
/// JSON response.
///
/// Recognized shapes (in priority order):
/// - `{"audio": {"url": "..."}}`
/// - `{"audio_url": "..."}`
/// - `{"output": {"audio": {"url": "..."}}}`
/// - `{"output": {"audio_url": "..."}}`
fn extract_fal_audio_url(output: &serde_json::Value) -> Option<String> {
    fn url_at(value: &serde_json::Value) -> Option<String> {
        if let Some(s) = value.as_str() {
            return Some(s.to_owned());
        }
        value.get("url").and_then(|v| v.as_str()).map(str::to_owned)
    }

    if let Some(audio) = output.get("audio")
        && let Some(url) = url_at(audio)
    {
        return Some(url);
    }
    if let Some(audio_url) = output.get("audio_url").and_then(|v| v.as_str()) {
        return Some(audio_url.to_owned());
    }
    if let Some(nested) = output.get("output") {
        return extract_fal_audio_url(nested);
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_fal_audio_url_handles_nested_audio_object() {
        let output = serde_json::json!({"audio": {"url": "https://cdn.fal/x.wav"}});
        assert_eq!(
            extract_fal_audio_url(&output).as_deref(),
            Some("https://cdn.fal/x.wav"),
        );
    }

    #[test]
    fn extract_fal_audio_url_handles_flat_audio_url() {
        let output = serde_json::json!({"audio_url": "https://cdn.fal/y.wav"});
        assert_eq!(
            extract_fal_audio_url(&output).as_deref(),
            Some("https://cdn.fal/y.wav"),
        );
    }

    #[test]
    fn extract_fal_audio_url_handles_nested_output() {
        let output = serde_json::json!({"output": {"audio": {"url": "https://cdn.fal/z.wav"}}});
        assert_eq!(
            extract_fal_audio_url(&output).as_deref(),
            Some("https://cdn.fal/z.wav"),
        );
    }

    #[test]
    fn extract_fal_audio_url_returns_none_on_unknown_shape() {
        let output = serde_json::json!({"images": ["foo"]});
        assert_eq!(extract_fal_audio_url(&output), None);
    }

    #[test]
    fn fal_vc_provider_metadata_is_stamped() {
        let p = FalVcProvider::new("test-key");
        assert_eq!(p.provider_id(), "fal");
        assert_eq!(p.capability(), CapabilityKind::Vc);
        assert_eq!(p.metadata().version.as_deref(), Some(DEFAULT_FAL_VC_MODEL));
    }

    #[cfg(feature = "audio-vc-rvc")]
    #[test]
    fn rvc_provider_metadata_is_stamped() {
        let p = RvcProvider::new();
        assert_eq!(p.provider_id(), "rvc");
        assert_eq!(p.capability(), CapabilityKind::Vc);
    }
}
